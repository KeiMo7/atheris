# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for named custom mutators and the shared dictionary.

In-process tests cover the registry, the dictionary parser, contract
enforcement, and the direct invocation API.  End-to-end tests fork a child
fuzzer, following the style of ``fuzz_test_lib``, because a fatal strategy
failure stops the fuzzing run and must not take the test process with it.
"""

import hashlib
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest

import atheris
from atheris import custom_mutators


class DictionaryParsingTests(unittest.TestCase):

  def testPlainAndQuotedEntries(self):
    entries = atheris.parse_dictionary(
        'plain\n"quoted"\nkey="with_key"\n')
    self.assertEqual(entries, [b"plain", b"quoted", b"with_key"])

  def testBlankAndCommentLinesAreIgnored(self):
    entries = atheris.parse_dictionary(
        "# a comment\n\n   \n   # indented comment\nword\n")
    self.assertEqual(entries, [b"word"])

  def testEscapeCharactersAreRestored(self):
    entries = atheris.parse_dictionary(
        r'"quote\"inside"' "\n" r'"back\\slash"' "\n" r'"nul\x00byte"' "\n" +
        'key="value"\n')
    self.assertEqual(entries, [
        b'quote"inside',
        b"back\\slash",
        b"nul\x00byte",
        b"value",
    ])

  def testBadLineReportsLineNumber(self):
    text = '# comment\n"good"\n\n"unterminated\n"also good"\n'
    with self.assertRaises(atheris.DictionaryError) as raised:
      atheris.parse_dictionary(text, path="words.dict")
    self.assertEqual(raised.exception.line_number, 4)
    self.assertIn("line 4", str(raised.exception))
    self.assertIn("words.dict", str(raised.exception))

  def testBadEscapeReportsLineNumber(self):
    with self.assertRaises(atheris.DictionaryError) as raised:
      atheris.parse_dictionary('"fine"\n"bad \\q escape"\n')
    self.assertEqual(raised.exception.line_number, 2)

  def testOversizedEntryIsRejected(self):
    too_long = "a" * (atheris.MAX_DICTIONARY_ENTRY_SIZE + 1)
    with self.assertRaises(atheris.DictionaryError) as raised:
      atheris.parse_dictionary(f'"{too_long}"\n')
    self.assertEqual(raised.exception.line_number, 1)

  def testDictionaryFileOrderAndDigestAreStable(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      path = os.path.join(temp_dir, "words.dict")
      with open(path, "w", encoding="ascii") as handle:
        handle.write('"alpha"\n"beta"\n"gamma"\n')
      first = atheris.load_dictionary(path)
      second = atheris.load_dictionary(path)
      self.assertEqual(first.entries, (b"alpha", b"beta", b"gamma"))
      self.assertEqual(first.digest, second.digest)
      self.assertEqual(first, second)
      self.assertIn("[0002] b'gamma'", first.format_entries())

  def testMissingFileFailsAtStartup(self):
    with self.assertRaises(atheris.DictionaryError) as raised:
      atheris.load_dictionary("/definitely/not/a/dictionary.txt")
    self.assertIsNone(raised.exception.line_number)
    self.assertIn("does not exist", str(raised.exception))

  def testSerializedDictionaryRoundTrips(self):
    entries = [b"plain", b'quote"x', b"back\\slash", b"\x00\xffbinary"]
    dictionary = atheris.Dictionary(entries)
    with tempfile.TemporaryDirectory() as temp_dir:
      path = os.path.join(temp_dir, "roundtrip.dict")
      dictionary.write_libfuzzer_file(path)
      replayed = atheris.load_dictionary(path)
    self.assertEqual(replayed.entries, tuple(entries))
    self.assertEqual(replayed.digest, dictionary.digest)

  def testDictionaryIsImmutable(self):
    dictionary = atheris.Dictionary([b"one", b"two"])
    with self.assertRaises(AttributeError):
      dictionary.entries = (b"replaced",)
    with self.assertRaises(TypeError):
      dictionary.entries[0] = b"changed"
    self.assertEqual(dictionary.entries, (b"one", b"two"))


class RegistryTests(unittest.TestCase):
  """Registry behavior; each test starts from a pristine registration state."""

  def setUp(self):
    custom_mutators._reset_for_testing()

  def tearDown(self):
    custom_mutators._reset_for_testing()

  def testRegisterAndSelectTwoStrategies(self):
    calls = {"first": 0, "second": 0}

    def first(data, max_size, seed):
      calls["first"] += 1
      return data

    def second(data, max_size, seed):
      calls["second"] += 1
      return data + b"!"

    atheris.register_custom_mutator("first", first)
    atheris.register_custom_mutator("second", second)
    self.assertEqual(atheris.registered_names(), ["first", "second"])

    selected = atheris.custom_mutators.get_registered("second")
    self.assertIs(selected.mutator, second)
    self.assertIn("second", atheris.registered_custom_mutators())

    # Invoke the selected strategy the way the engine would; the other one
    # must not be called.
    result = atheris.invoke_custom_mutator("second", b"data", 32, 7)
    self.assertEqual(result, b"data!")
    self.assertEqual(calls, {"first": 0, "second": 1})

  def testDecoratorRegistration(self):
    @atheris.register("decorated")
    def decorated(data, max_size, seed):
      return data

    self.assertEqual(atheris.registered_names(), ["decorated"])
    self.assertIs(
        atheris.custom_mutators.get_registered("decorated").mutator,
        decorated)

  def testUnknownNameListsAvailableStrategies(self):
    atheris.register_custom_mutator("alpha", lambda data, max_size, seed: data)
    atheris.register_custom_mutator("beta", lambda data, max_size, seed: data)
    with self.assertRaises(atheris.UnknownCustomMutatorError) as raised:
      atheris.custom_mutators.get_registered("gamma")
    message = str(raised.exception)
    self.assertIn("'alpha'", message)
    self.assertIn("'beta'", message)
    self.assertIn("gamma", message)

  def testDuplicateNameRejectedWithoutSideEffects(self):
    def original(data, max_size, seed):
      return data

    atheris.register_custom_mutator("dup", original)
    with self.assertRaises(ValueError):
      atheris.register_custom_mutator("dup", lambda data, max_size, seed: b"x")
    self.assertIs(atheris.custom_mutators.get_registered("dup").mutator,
                  original)

  def testRegistrationAfterStartupIsRejected(self):
    atheris.register_custom_mutator("before", lambda data, max_size, seed: data)
    custom_mutators.freeze()  # Simulates a completed Setup().

    with self.assertRaises(atheris.RegistryFrozenError):
      atheris.register_custom_mutator("after", lambda data, max_size, seed: b"")
    with self.assertRaises(atheris.RegistryFrozenError):
      atheris.unregister_custom_mutator("before")

    # The effective configuration is untouched.
    self.assertEqual(atheris.registered_names(), ["before"])

  def testSetupAfterStartupIsRejected(self):
    custom_mutators.freeze()
    with self.assertRaises(atheris.RegistryFrozenError):
      atheris.Setup([sys.argv[0]], lambda data: None)

  def testRequiresDictionaryFlagRoundTrips(self):
    strategy = atheris.register_custom_mutator(
        "needs_words", lambda data, max_size, seed: data,
        requires_dictionary=True)
    self.assertTrue(strategy.requires_dictionary)
    self.assertTrue(
        atheris.custom_mutators.get_registered("needs_words")
        .requires_dictionary)

  def testBulkRegistrationIsAllOrNothing(self):
    def good(data, max_size, seed):
      return data

    atheris.register_custom_mutator("existing", good)
    with self.assertRaises(ValueError):
      atheris.register_custom_mutators({
          "new_one": good,
          "existing": good,  # Already registered -> the whole batch fails.
      })
    self.assertEqual(atheris.registered_names(), ["existing"])

    registered = atheris.register_custom_mutators(
        [("a", good), ("b", good)])
    self.assertEqual([s.name for s in registered], ["a", "b"])
    self.assertEqual(atheris.registered_names(), ["existing", "a", "b"])

  def testBulkRegistrationRejectedWhenFrozen(self):
    custom_mutators.freeze()
    with self.assertRaises(atheris.RegistryFrozenError):
      atheris.register_custom_mutators([("a", lambda d, m, s: d)])
    self.assertEqual(atheris.registered_names(), [])


class ContractTests(unittest.TestCase):

  def setUp(self):
    custom_mutators._reset_for_testing()

  def tearDown(self):
    custom_mutators._reset_for_testing()

  def testOversizedResultRaisesCatchableError(self):
    def greedy(data, max_size, seed):
      return b"X" * (max_size + 1)

    with self.assertRaises(atheris.CustomMutatorOutputTooLongError) as raised:
      atheris.invoke_custom_mutator(greedy, b"seed", 8, 1)
    self.assertIn("max_size=8", str(raised.exception))
    self.assertIn("greedy", str(raised.exception))
    # The process is still alive and the error is catchable.

  def testNonBytesResultNamesExpectedType(self):
    def wrong(data, max_size, seed):
      return "a string"

    with self.assertRaises(atheris.CustomMutatorTypeError) as raised:
      atheris.invoke_custom_mutator(wrong, b"seed", 8, 1)
    self.assertIn("must return bytes", str(raised.exception))
    self.assertIn("str", str(raised.exception))

  def testRaisingStrategyCarriesNameAndTraceback(self):
    def explodes(data, max_size, seed):
      raise ValueError("detonated")

    atheris.register_custom_mutator("explodes", explodes)
    with self.assertRaises(atheris.CustomMutatorRuntimeError) as raised:
      atheris.invoke_custom_mutator("explodes", b"seed", 8, 1)
    message = str(raised.exception)
    self.assertIn("'explodes'", message)
    self.assertIn("ValueError: detonated", message)
    self.assertIn("Traceback (most recent call last)", message)
    self.assertIn("in explodes", message)

  def testOtherStrategiesRemainUsableAfterOneFails(self):
    def good(data, max_size, seed):
      return data

    def bad(data, max_size, seed):
      raise RuntimeError("nope")

    atheris.register_custom_mutator("good", good)
    atheris.register_custom_mutator("bad", bad)
    with self.assertRaises(atheris.CustomMutatorRuntimeError):
      atheris.invoke_custom_mutator("bad", b"x", 8, 1)
    self.assertEqual(atheris.invoke_custom_mutator("good", b"x", 8, 1), b"x")

  def testContractErrorsAreDistinguishable(self):
    self.assertTrue(issubclass(atheris.CustomMutatorOutputTooLongError,
                               atheris.CustomMutatorContractError))
    self.assertTrue(issubclass(atheris.CustomMutatorTypeError,
                               atheris.CustomMutatorContractError))
    self.assertTrue(issubclass(atheris.CustomMutatorContractError,
                               atheris.CustomMutatorError))


class DictionaryScopeTests(unittest.TestCase):

  def setUp(self):
    custom_mutators._reset_for_testing()

  def tearDown(self):
    custom_mutators._reset_for_testing()

  def testEngineAndStrategiesShareOneDictionaryObject(self):
    dictionary = atheris.Dictionary([b"alpha", b"beta"])
    custom_mutators.set_dictionary(dictionary)
    seen = {}

    def capture(data, max_size, seed):
      seen["dictionary"] = atheris.get_dictionary()
      return data

    atheris.register_custom_mutator("captures", capture)
    invoker = atheris.StrategyInvoker(
        atheris.custom_mutators.get_registered("captures"), dictionary)
    invoker.invoke(b"", 8, 1)
    self.assertIs(seen["dictionary"], dictionary)
    self.assertIs(atheris.get_dictionary(), dictionary)
    # The strategy observed the very same ordered entries, not a copy.
    self.assertIs(atheris.get_dictionary_entries(), dictionary.entries)

  def testNoDictionaryMeansEmptyEntryView(self):
    self.assertIsNone(atheris.get_dictionary())
    self.assertEqual(atheris.get_dictionary_entries(), ())

  def testDictionaryCannotBeReplacedAfterStartup(self):
    dictionary = atheris.Dictionary([b"alpha"])
    custom_mutators.set_dictionary(dictionary)
    custom_mutators.freeze()
    with self.assertRaises(atheris.RegistryFrozenError):
      custom_mutators.set_dictionary(atheris.Dictionary([b"other"]))
    self.assertIs(atheris.get_dictionary(), dictionary)

  def testMutateDictionaryDecisionIsDocumented(self):
    custom_mutators.set_dictionary(atheris.Dictionary([b"word"]))
    atheris.register_custom_mutator(
        "uses_dict", lambda data, max_size, seed: data)
    invoker = atheris.StrategyInvoker(
        atheris.custom_mutators.get_registered("uses_dict"),
        atheris.get_dictionary())
    custom_mutators.set_active_invoker(invoker)
    description = atheris.DescribeMutationConfiguration()
    self.assertIn("use the dictionary", description)
    self.assertIn("fixed at startup", description)

    custom_mutators.set_dictionary(None)
    invoker = atheris.StrategyInvoker(
        atheris.custom_mutators.get_registered("uses_dict"), None)
    custom_mutators.set_active_invoker(invoker)
    description = atheris.DescribeMutationConfiguration()
    self.assertIn("do not use a dictionary", description)


class StartupConfigurationTests(unittest.TestCase):
  """Setup()-level validation; rejected configs are side-effect free."""

  def setUp(self):
    custom_mutators._reset_for_testing()

  def tearDown(self):
    custom_mutators._reset_for_testing()

  def _write(self, directory, name, text):
    path = os.path.join(directory, name)
    with open(path, "w", encoding="utf-8") as handle:
      handle.write(text)
    return path

  def testMalformedDictionaryAbortsStartup(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      path = self._write(temp_dir, "bad.dict", '"good"\n"broken\n')
      with self.assertRaises(atheris.DictionaryError) as raised:
        atheris.Setup([sys.argv[0]], lambda data: None, dictionary=path)
      self.assertEqual(raised.exception.line_number, 2)
      self.assertFalse(custom_mutators.is_frozen())

  def testMissingDictionaryAbortsStartup(self):
    with self.assertRaises(atheris.DictionaryError):
      atheris.Setup([sys.argv[0]], lambda data: None,
                    dictionary="/no/such/dictionary.txt")
    self.assertFalse(custom_mutators.is_frozen())

  def testUnknownStrategyNameAbortsStartup(self):
    atheris.register_custom_mutator("known", lambda data, max_size, seed: data)
    with self.assertRaises(atheris.UnknownCustomMutatorError) as raised:
      atheris.Setup([sys.argv[0]], lambda data: None,
                    custom_mutator_name="missing")
    self.assertIn("'known'", str(raised.exception))

  def testStrategyRequiringDictionaryRefusesStartup(self):
    atheris.register_custom_mutator(
        "needs_words", lambda data, max_size, seed: data,
        requires_dictionary=True)
    with self.assertRaises(atheris.AtherisSetupError) as raised:
      atheris.Setup([sys.argv[0]], lambda data: None,
                    custom_mutator_name="needs_words")
    self.assertIn("requires a dictionary", str(raised.exception))
    self.assertFalse(custom_mutators.is_frozen())

  def testDuplicateDictionaryArgumentsRejected(self):
    with self.assertRaises(atheris.AtherisSetupError):
      atheris.Setup([sys.argv[0], "-dict=a", "-dict=b"], lambda data: None)


def _run_child(test_case, source, args, timeout=90):
  """Writes ``source`` to a temp script and runs it with the current Python."""
  with tempfile.TemporaryDirectory() as temp_dir:
    script = os.path.join(temp_dir, "child_fuzzer.py")
    with open(script, "w", encoding="utf-8") as handle:
      handle.write(textwrap.dedent(source))
    return subprocess.run(
        [sys.executable, script] + list(args),
        cwd=temp_dir,
        capture_output=True,
        timeout=timeout,
        check=False)


SERVICE = """
    import sys

    import atheris

    CALLS = []


    def alpha(data, max_size, seed):
      if "alpha" not in CALLS:
        CALLS.append("alpha")
        sys.stderr.write("MUTATOR:alpha\\n")
        sys.stderr.flush()
      return data


    def beta(data, max_size, seed):
      if "beta" not in CALLS:
        CALLS.append("beta")
        sys.stderr.write("MUTATOR:beta\\n")
        sys.stderr.flush()
      return data


    @atheris.instrument_func
    def TestOneInput(data):
      if b"impossible-string-that-is-never-generated" in data:
        raise RuntimeError("unreachable")


    def main():
      atheris.register_custom_mutator("alpha", alpha)
      atheris.register_custom_mutator("beta", beta)
      name = sys.argv[1]
      atheris.Setup([sys.argv[0]] + sys.argv[2:], TestOneInput,
                    custom_mutator_name=name)
      atheris.Fuzz()


    if __name__ == "__main__":
      main()
"""


class EndToEndTests(unittest.TestCase):

  def testOnlySelectedStrategyIsCalled(self):
    alpha = _run_child(self, SERVICE, ["alpha", "-atheris_runs=100"])
    self.assertIn(b"MUTATOR:alpha", alpha.stderr)
    self.assertNotIn(b"MUTATOR:beta", alpha.stderr)

    beta = _run_child(self, SERVICE, ["beta", "-atheris_runs=100"])
    self.assertIn(b"MUTATOR:beta", beta.stderr)
    self.assertNotIn(b"MUTATOR:alpha", beta.stderr)

  def testRegistrationInsideFuzzingLoopIsRejected(self):
    source = """
        import sys

        import atheris

        RESULT = []


        def chosen(data, max_size, seed):
          return data


        @atheris.instrument_func
        def TestOneInput(data):
          if RESULT:
            return
          try:
            atheris.register_custom_mutator("late", chosen)
          except atheris.RegistryFrozenError:
            RESULT.append("rejected")
          RESULT.append(repr(atheris.registered_names()))
          sys.stderr.write("RESULT:%s\\n" % "|".join(RESULT))
          sys.stderr.flush()


        atheris.register_custom_mutator("chosen", chosen)
        atheris.Setup([sys.argv[0]] + sys.argv[1:], TestOneInput,
                      custom_mutator_name="chosen")
        atheris.Fuzz()
    """
    result = _run_child(self, source, ["-atheris_runs=5"])
    self.assertIn(b"RESULT:rejected|['chosen']", result.stderr)
    self.assertNotIn(b"late", result.stderr.split(b"RESULT:")[-1])

  def testOversizedReturnIsRejectedAndProcessSurvives(self):
    source = """
        import sys

        import atheris


        def greedy(data, max_size, seed):
          return b"X" * (max_size + 1)


        @atheris.instrument_func
        def TestOneInput(data):
          sys.stderr.write("ALIVE\\n")
          sys.stderr.flush()


        atheris.register_custom_mutator("greedy", greedy)
        atheris.Setup([sys.argv[0]] + sys.argv[1:], TestOneInput,
                      custom_mutator_name="greedy")
        atheris.Fuzz()
    """
    result = _run_child(self, source, ["-atheris_runs=20"])
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertIn(b"ALIVE", result.stderr)
    self.assertIn(b"exceeds max_size", result.stderr)
    self.assertIn(b"greedy", result.stderr)
    self.assertIn(b"mutation was rejected", result.stderr)
    self.assertIn(b"Done 20 in", result.stderr)

  def testRaisingStrategyStopsRunWithNameAndTraceback(self):
    source = """
        import sys

        import atheris


        def kaboom(data, max_size, seed):
          raise ValueError("detonated")


        @atheris.instrument_func
        def TestOneInput(data):
          sys.stderr.write("ALIVE\\n")
          sys.stderr.flush()


        atheris.register_custom_mutator("kaboom", kaboom)
        atheris.Setup([sys.argv[0]] + sys.argv[1:], TestOneInput,
                      custom_mutator_name="kaboom")
        atheris.Fuzz()
    """
    result = _run_child(self, source, ["-atheris_runs=20"])
    self.assertNotEqual(result.returncode, 0)
    self.assertIn(b"'kaboom'", result.stderr)
    self.assertIn(b"ValueError: detonated", result.stderr)
    self.assertIn(b"Traceback (most recent call last)", result.stderr)
    # The run stopped instead of silently spinning.
    self.assertNotIn(b"Done 20 in", result.stderr)

  def testSameSeedProducesSameMutationSequenceWithDictionary(self):
    source = """
        import hashlib
        import sys

        import atheris


        def echo(data, max_size, seed):
          return atheris.Mutate(data, max_size)


        @atheris.instrument_func
        def TestOneInput(data):
          sys.stdout.write("SEQ:%s\\n" % hashlib.sha256(data).hexdigest())
          sys.stdout.flush()


        atheris.register_custom_mutator("echo", echo)
        atheris.Setup([sys.argv[0]] + sys.argv[1:], TestOneInput,
                      custom_mutator_name="echo")
        atheris.Fuzz()
    """
    with tempfile.TemporaryDirectory() as temp_dir:
      dictionary = os.path.join(temp_dir, "words.dict")
      with open(dictionary, "w", encoding="ascii") as handle:
        handle.write('"ALPHA"\n# comment\n"BETA"\n"GAMMA\\x00"\n')

      def sequence():
        result = _run_child(self, source, [
            "-dict=" + dictionary, "-seed=4242", "-atheris_runs=400"
        ])
        self.assertEqual(result.returncode, 0, result.stderr)
        return [line for line in result.stdout.splitlines()
                if line.startswith(b"SEQ:")]

      first = sequence()
      second = sequence()
      self.assertGreater(len(first), 10)
      self.assertEqual(first, second)

  def testSetupDictionaryKwargReachesStrategyAndEngine(self):
    source = """
        import sys

        import atheris

        OBSERVED = []


        def observes(data, max_size, seed):
          if not OBSERVED:
            entries = atheris.get_dictionary_entries()
            OBSERVED.append(entries)
            sys.stderr.write("ENTRIES:%r\\n" % (entries,))
            sys.stderr.flush()
          return data


        @atheris.instrument_func
        def TestOneInput(data):
          pass


        atheris.register_custom_mutator("observes", observes)
        dictionary = sys.argv.pop()
        atheris.Setup([sys.argv[0]] + sys.argv[1:], TestOneInput,
                      custom_mutator_name="observes",
                      dictionary=dictionary)
        atheris.Fuzz()
    """
    with tempfile.TemporaryDirectory() as temp_dir:
      dictionary = os.path.join(temp_dir, "words.dict")
      with open(dictionary, "w", encoding="ascii") as handle:
        handle.write('"one"\n"two"\n')
      result = _run_child(
          self, source, ["-atheris_runs=20", dictionary])
    self.assertIn(b"ENTRIES:(b'one', b'two')", result.stderr)
    self.assertIn(b"Dictionary: 2 entries", result.stderr)

  def testLegacyCallableEntryPointStillWorks(self):
    source = """
        import sys

        import atheris


        def legacy(data, max_size, seed):
          sys.stderr.write("LEGACY\\n")
          sys.stderr.flush()
          return data


        @atheris.instrument_func
        def TestOneInput(data):
          pass


        atheris.Setup([sys.argv[0]] + sys.argv[1:], TestOneInput,
                      custom_mutator=legacy)
        atheris.Fuzz()
    """
    result = _run_child(self, source, ["-atheris_runs=20"])
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertIn(b"LEGACY", result.stderr)
    self.assertIn(b"Done 20 in", result.stderr)

  def testNoDictionaryDoesNotPrintConfiguration(self):
    source = """
        import sys

        import atheris


        def legacy(data, max_size, seed):
          return data


        @atheris.instrument_func
        def TestOneInput(data):
          pass


        atheris.Setup([sys.argv[0]] + sys.argv[1:], TestOneInput,
                      custom_mutator=legacy)
        atheris.Fuzz()
    """
    result = _run_child(self, source, ["-atheris_runs=5"])
    self.assertNotIn(b"INFO: Custom mutator:", result.stderr)
    self.assertNotIn(b"INFO: No dictionary provided.", result.stderr)

  def testStrategyCanBeSelectedOnCommandLine(self):
    source = """
        import sys

        import atheris


        def alpha(data, max_size, seed):
          sys.stderr.write("RAN:alpha\\n")
          sys.stderr.flush()
          return data


        def beta(data, max_size, seed):
          sys.stderr.write("RAN:beta\\n")
          sys.stderr.flush()
          return data


        @atheris.instrument_func
        def TestOneInput(data):
          pass


        atheris.register_custom_mutator("alpha", alpha)
        atheris.register_custom_mutator("beta", beta)
        atheris.Setup([sys.argv[0]] + sys.argv[1:], TestOneInput)
        atheris.Fuzz()
    """
    result = _run_child(
        self, source, ["-atheris_custom_mutator=beta", "-atheris_runs=20"])
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertIn(b"RAN:beta", result.stderr)
    self.assertNotIn(b"RAN:alpha", result.stderr)
    self.assertIn(b"INFO: Custom mutator: 'beta'", result.stderr)

  def testUnknownCommandLineStrategyAbortsWithOptions(self):
    source = """
        import sys

        import atheris


        def alpha(data, max_size, seed):
          return data


        def TestOneInput(data):
          pass


        atheris.register_custom_mutator("alpha", alpha)
        atheris.Setup([sys.argv[0]] + sys.argv[1:], TestOneInput)
        atheris.Fuzz()
    """
    result = _run_child(
        self, source, ["-atheris_custom_mutator=missing", "-atheris_runs=5"])
    self.assertNotEqual(result.returncode, 0)
    self.assertIn(b"missing", result.stderr)
    self.assertIn(b"'alpha'", result.stderr)

  def testShippedExampleRuns(self):
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    example = os.path.join(repo_root, "example_fuzzers",
                           "named_mutator_example.py")
    dictionary = os.path.join(repo_root, "example_fuzzers",
                              "named_mutator_example.dict")
    if not (os.path.exists(example) and os.path.exists(dictionary)):
      self.skipTest("example fuzzers are not available in this layout")
    result = subprocess.run(
        [
            sys.executable, example,
            "-atheris_custom_mutator=compressed",
            "-dict=" + dictionary,
            "-seed=42",
            "-atheris_runs=200000",
        ],
        capture_output=True,
        timeout=120,
        check=False)
    combined = result.stdout + result.stderr
    self.assertIn(b"INFO: Custom mutator: 'compressed'", combined)
    self.assertIn(b"Dictionary: 4 entries", combined)
    self.assertIn(b"Boom", combined)


if __name__ == "__main__":
  unittest.main()
