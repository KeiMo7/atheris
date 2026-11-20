# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for crash-sample replay and signature-preserving minimization.

These tests spawn real replay interpreters (via atheris.repro) against target
modules written into an isolated directory. Every target uses the ordinary
``TestOneInput(data: bytes)`` convention; nothing here runs the libFuzzer loop.
"""

import os
import sys
import tempfile
import threading
import time
import unittest

import atheris
from atheris import repro

# Target module source. It defines several distinct defects that all terminate
# the process the same way (exit code 1) but must produce distinct signatures.
_TARGET_SOURCE = '''
import os
import sys
import time


def branch_crash(data):
    # Requires a marker plus padding, so minimization has removable bytes.
    if len(data) >= 6 and data[:4] == b"CRSH" and data[4] == 0x01:
        raise RuntimeError("branch crash")


def value_crash(data):
    if len(data) >= 4 and data[:4] == b"VAL!":
        raise ValueError("value crash")


def index_crash(data):
    if len(data) >= 3 and data[:3] == b"IDX":
        raise IndexError("index crash")


def passthrough(data):
    # Never triggers.
    return len(data)


def signature_drift(data):
    # Raises ValueError only on one exact input; anything else that is short
    # enough raises a different RuntimeError. A minimizer that drops bytes will
    # observe a signature drift.
    if data == b"DRIFT" + b"\\x00\\x00\\x00":
        raise ValueError("exact match only")
    if data.startswith(b"DRIFT"):
        raise RuntimeError("near miss")


def slow_target(data):
    if data == b"WAIT":
        time.sleep(30.0)


def self_kill(data):
    # Fatal signal: the interpreter cannot catch this; replay must classify it
    # from the process return code / faulthandler trace.
    if data == b"KILL":
        faulthandler = __import__("faulthandler")
        faulthandler.enable()
        os.kill(os.getpid(), 11)  # SIGSEGV


def uses_state(data):
    # Module-level mutable state touched on every call. Fresh-process replay
    # must reset it, otherwise the observed behavior would depend on history.
    uses_state.calls = getattr(uses_state, "calls", 0) + 1
    if len(data) >= 2 and data[:2] == b"ST" and uses_state.calls != 1:
        raise AssertionError("state leaked across replays")
    if len(data) >= 2 and data[:2] == b"ST":
        raise RuntimeError("state clean")


def substitute_me(data):
    # Exact-length requirement: no byte can be removed, but any value after
    # the magic still triggers, so substitution is the only valid move.
    if len(data) == 8 and data[:4] == b"SUB!":
        raise RuntimeError("substitute target")
'''


class ReproTestBase(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self._tmp = tempfile.TemporaryDirectory(prefix="repro_test_")
    self.tmpdir = self._tmp.name
    target_path = os.path.join(self.tmpdir, "repro_target.py")
    with open(target_path, "w", encoding="utf-8") as f:
      f.write(_TARGET_SOURCE)
    self.target_dir = self.tmpdir
    self.timeout = 5.0

  def tearDown(self):
    super().tearDown()
    self._tmp.cleanup()

  def spec(self, qualname):
    return repro.TargetSpec(
        module="repro_target",
        qualname=qualname,
        target_search_path=(self.target_dir,))

  # Form 1: standalone replay yields a structured verdict, not just an exit.
  def test_form1_replay_verdict(self):
    result = repro.replay_sample(
        self.spec("branch_crash"), b"CRSH\x01" + b"junk-junk-junk",
        timeout=self.timeout)
    self.assertTrue(result.verdict.triggered)
    self.assertEqual(result.verdict.kind, "exception")
    self.assertEqual(result.verdict.exit_kind, "exception")
    self.assertEqual(result.verdict.exception_type, "RuntimeError")
    self.assertEqual(result.verdict.exit_code, 1)
    self.assertIsNotNone(result.verdict.signature)
    self.assertFalse(result.verdict.degraded)
    self.assertEqual(result.sample_length, 19)

  # Form 2: same defect replays to one stable signature; different defects with
  # identical exit codes get different signatures.
  def test_form2_stable_and_distinct_signatures_same_exit_code(self):
    samples = {
        "branch_crash": b"CRSH\x01zzzzzz",
        "value_crash": b"VAL!zzzz",
        "index_crash": b"IDXzz",
    }
    sigs = {}
    for qualname, sample in samples.items():
      r1 = repro.replay_sample(
          self.spec(qualname), sample, timeout=self.timeout,
          confirmations=2)
      r2 = repro.replay_sample(
          self.spec(qualname), sample, timeout=self.timeout)
      self.assertEqual(r1.verdict.exit_code, r2.verdict.exit_code)
      self.assertEqual(r1.verdict.signature, r2.verdict.signature)
      sigs[qualname] = r1.verdict.signature
    self.assertEqual(len(set(sigs.values())), 3, sigs)

  # Form 3 + acceptance: strict shrink with an identical signature.
  def test_form3_minimize_shrinks_with_signature_unchanged(self):
    original = b"CRSH\x01" + b"".join(bytes((i % 256,)) for i in range(64))
    report = repro.minimize_sample(
        self.spec("branch_crash"), original, timeout=self.timeout,
        max_attempts=300)
    self.assertEqual(report.status, "success", report.reason)
    self.assertLess(report.final_length, report.original_length)
    self.assertEqual(report.final_length, len(report.data))
    self.assertEqual(report.signature, report.baseline.verdict.signature)
    self.assertEqual(
        report.final_replay.verdict.signature,
        report.baseline.verdict.signature)
    self.assertGreaterEqual(report.attempts, 1)
    self.assertGreaterEqual(report.discarded_fragments, 1)
    # The minimized sample itself still reproduces, standalone.
    again = repro.replay_sample(
        self.spec("branch_crash"), report.data, timeout=self.timeout)
    self.assertEqual(again.verdict.signature, report.signature)

  # Form 4: monotonic convergence - every accepted candidate is a strict prefix
  # family member, never grows, and is independently reproducible.
  def test_form4_monotonic_and_each_step_reproducible(self):
    original = b"CRSH\x01" + b"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    report = repro.minimize_sample(
        self.spec("branch_crash"), original, timeout=self.timeout,
        max_attempts=300, confirmations=2)
    self.assertEqual(report.status, "success")
    self.assertLessEqual(report.final_length, report.original_length)
    # Independent replay in a brand new process reproduces the same signature.
    fresh = repro.replay_sample(
        self.spec("branch_crash"), report.data, timeout=self.timeout,
        confirmations=3)
    self.assertTrue(fresh.verdict.triggered)
    self.assertEqual(fresh.verdict.signature, report.signature)

  # Form 5: the report carries auditable evidence.
  def test_form5_report_evidence_fields(self):
    original = b"CRSH\x01padding-padding"
    report = repro.minimize_sample(
        self.spec("branch_crash"), original, timeout=self.timeout)
    self.assertEqual(report.original_length, len(original))
    self.assertEqual(report.final_length, len(report.data))
    self.assertIsNotNone(report.signature)
    self.assertIsInstance(report.attempts, int)
    self.assertIsInstance(report.discarded_fragments, int)
    self.assertIsInstance(report.to_dict(), dict)
    self.assertEqual(report.to_dict()["original_length"], len(original))

  # Form 6 + key path: minimization failure falls back to the original sample,
  # never returning a smaller unverified one.
  def test_form6_failure_returns_original(self):
    # The drift target only crashes with ValueError on the full exact input;
    # any truncation flips to RuntimeError, so no shrink can preserve the
    # signature.
    original = b"DRIFT\x00\x00\x00"
    report = repro.minimize_sample(
        self.spec("signature_drift"), original, timeout=self.timeout,
        max_attempts=100)
    self.assertIn(report.status, ("fallback", "unreproducible"))
    self.assertEqual(report.data, original)
    self.assertEqual(report.final_length, len(original))
    self.assertGreater(report.drift_count, 0)
    self.assertIn(
        report.reason, ("signature_drift", "no_reproducing_candidate"))

  # Form 7: repeated minimization is equivalent (length and signature stable).
  def test_form7_deterministic_repeat_minimize(self):
    original = b"CRSH\x01" + b"xyzzyxyzzyxyzzy"
    kwargs = dict(timeout=self.timeout, max_attempts=120)
    r1 = repro.minimize_sample(self.spec("branch_crash"), original, **kwargs)
    r2 = repro.minimize_sample(self.spec("branch_crash"), original, **kwargs)
    self.assertEqual(r1.status, "success")
    self.assertEqual(r2.status, "success")
    self.assertEqual(r1.final_length, r2.final_length)
    self.assertEqual(r1.signature, r2.signature)
    self.assertEqual(r1.data, r2.data)

  # Form 8: a non-triggering sample is "unreproducible", never a 0-byte success.
  def test_form8_non_triggering_sample(self):
    report = repro.minimize_sample(
        self.spec("branch_crash"), b"never-crashes", timeout=self.timeout)
    self.assertEqual(report.status, "unreproducible")
    self.assertEqual(report.reason, "sample does not trigger the target")
    self.assertEqual(report.data, b"never-crashes")
    self.assertGreater(report.final_length, 0)

    replay = repro.replay_sample(
        self.spec("passthrough"), b"whatever", timeout=self.timeout)
    self.assertFalse(replay.verdict.triggered)
    self.assertIsNone(replay.verdict.signature)
    self.assertEqual(replay.verdict.kind, "none")

  # Form 9: export/import round-trips byte-for-byte and replays to the same
  # signature.
  def test_form9_export_import_roundtrip(self):
    original = b"CRSH\x01" + b"export-me"
    report = repro.minimize_sample(
        self.spec("branch_crash"), original, timeout=self.timeout)
    self.assertEqual(report.status, "success")
    path = os.path.join(self.tmpdir, "min.bin")
    repro.export_sample(path, report.data)
    reloaded = repro.import_sample(path)
    self.assertEqual(reloaded, report.data)
    replayed = repro.replay_sample(
        self.spec("branch_crash"), reloaded, timeout=self.timeout)
    self.assertEqual(replayed.verdict.signature, report.signature)

    # A zero-byte file is a valid sample and must round-trip too.
    empty_path = os.path.join(self.tmpdir, "empty.bin")
    repro.export_sample(empty_path, b"")
    self.assertEqual(repro.import_sample(empty_path), b"")

  # Form 10: batching isolates failures; one bad item does not affect the rest.
  def test_form10_batch_isolation(self):
    samples = [
        b"CRSH\x01aaaa",       # triggers RuntimeError
        b"VAL!bbbb",           # triggers ValueError
        b"nothing-here",       # does not trigger
    ]
    reports = repro.batch_minimize(
        self.spec("branch_crash"), [samples[0], samples[2]],
        timeout=self.timeout)
    # Two independent items, both reported, second is unreproducible.
    self.assertEqual(len(reports), 2)
    self.assertEqual(reports[0].status, "success")
    self.assertEqual(reports[1].status, "unreproducible")

    replays = repro.batch_replay(
        self.spec("value_crash"),
        [b"VAL!bbbb", b"VAL!cccc", b"nothing-here"],
        timeout=self.timeout)
    self.assertEqual(len(replays), 3)
    self.assertTrue(replays[0].verdict.triggered)
    self.assertTrue(replays[1].verdict.triggered)
    self.assertFalse(replays[2].verdict.triggered)
    self.assertEqual(replays[0].verdict.exception_type, "ValueError")
    self.assertEqual(replays[1].verdict.exception_type, "ValueError")

  # Hard constraint 2: signatures contain no path or sample-derived hash.
  def test_signature_has_no_path_or_sample_content(self):
    result = repro.replay_sample(
        self.spec("branch_crash"), b"CRSH\x01qq", timeout=self.timeout)
    sig = result.verdict.signature
    self.assertNotIn(self.tmpdir, sig)
    self.assertNotIn("CRSH", sig)
    self.assertNotIn("repro_target.py", sig)

  # Hard constraint 3: mutable target state resets across replays.
  def test_state_is_reset_between_replays(self):
    r1 = repro.replay_sample(
        self.spec("uses_state"), b"ST11", timeout=self.timeout,
        confirmations=3)
    self.assertEqual(r1.verdict.exception_type, "RuntimeError")
    self.assertNotIn("state leaked", r1.stderr_tail)

  # Hard constraint 4: the original sample file is never modified.
  def test_original_sample_file_untouched(self):
    original = b"CRSH\x01" + b"do-not-touch"
    sample_path = os.path.join(self.tmpdir, "original.bin")
    repro.export_sample(sample_path, original)
    before = repro.import_sample(sample_path)
    data = repro.import_sample(sample_path)
    repro.minimize_sample(self.spec("branch_crash"), data, timeout=self.timeout)
    after = repro.import_sample(sample_path)
    self.assertEqual(before, after)
    self.assertEqual(after, original)

  # Hard constraint 6: a replay over the timeout is classified as a failure
  # verdict, and minimizing a hanging input falls back.
  def test_timeout_classified_as_failure(self):
    result = repro.replay_sample(
        self.spec("slow_target"), b"WAIT", timeout=0.7)
    self.assertTrue(result.verdict.triggered)
    self.assertEqual(result.verdict.kind, "timeout")
    self.assertTrue(result.verdict.degraded)
    self.assertEqual(result.verdict.signature, "degraded:timeout")

    report = repro.minimize_sample(
        self.spec("slow_target"), b"WAIT", timeout=0.7, max_attempts=5)
    # A timeout signature is degraded; minimization cannot preserve it while
    # shrinking and must return the original sample.
    self.assertEqual(report.data, b"WAIT")
    self.assertIn(report.status, ("fallback", "unreproducible"))

  # Degradation 1: target failing at import time is an environment error and
  # never enters minimization.
  def test_environment_error_on_broken_target(self):
    broken_dir = tempfile.mkdtemp(prefix="repro_broken_")
    try:
      with open(os.path.join(broken_dir, "broken_target.py"), "w",
                encoding="utf-8") as f:
        f.write("raise RuntimeError('init boom')\n")
      spec = repro.TargetSpec(
          module="broken_target",
          qualname="whatever",
          target_search_path=(broken_dir,))
      result = repro.replay_sample(spec, b"x", timeout=self.timeout)
      self.assertEqual(result.verdict.kind, "environment_error")
      self.assertFalse(result.verdict.triggered)
      self.assertIn("init boom", result.verdict.message)

      report = repro.minimize_sample(spec, b"x", timeout=self.timeout)
      self.assertEqual(report.status, "environment_error")
      self.assertEqual(report.data, b"x")
    finally:
      import shutil
      shutil.rmtree(broken_dir, ignore_errors=True)

  # Degradation 2: a fatal signal with no Python frames yields a degraded
  # signature that still discriminates signal from non-signal outcomes.
  def test_native_signal_degraded_signature(self):
    result = repro.replay_sample(
        self.spec("self_kill"), b"KILL", timeout=self.timeout)
    self.assertTrue(result.verdict.triggered)
    self.assertEqual(result.verdict.kind, "signal")
    self.assertEqual(result.verdict.signal_name, "SIGSEGV")
    self.assertTrue(result.verdict.signature.startswith(
        ("degraded:signal:SIGSEGV", "signal:SIGSEGV")))
    # Distinct from the timeout degraded signature.
    self.assertNotEqual(result.verdict.signature, "degraded:timeout")

  # Degradation 3: interruption retains the smallest confirmed candidate and is
  # marked incomplete.
  def test_interruption_retains_confirmed_candidate(self):
    original = b"CRSH\x01" + b"Q" * 200
    stop = threading.Event()

    def fire():
      time.sleep(0.4)
      stop.set()

    timer = threading.Thread(target=fire, daemon=True)
    timer.start()
    report = repro.minimize_sample(
        self.spec("branch_crash"), original, timeout=self.timeout,
        max_attempts=10_000, stop_event=stop)
    timer.join()
    self.assertEqual(report.status, "interrupted")
    self.assertTrue(report.incomplete)
    self.assertLessEqual(report.final_length, len(original))
    # The retained candidate is itself independently reproducible.
    verify = repro.replay_sample(
        self.spec("branch_crash"), report.data, timeout=self.timeout)
    self.assertEqual(verify.verdict.signature, report.signature)

  # Degradation 4: unreadable / corrupt sample paths are explicit errors and
  # never produce an empty sample.
  def test_unreadable_sample_is_explicit_error(self):
    missing = os.path.join(self.tmpdir, "does_not_exist.bin")
    with self.assertRaises(repro.SampleReadError):
      repro.import_sample(missing)
    dir_path = os.path.join(self.tmpdir, "a_directory")
    os.mkdir(dir_path)
    with self.assertRaises(repro.SampleReadError):
      repro.import_sample(dir_path)

  # Substitution phase: byte values can be replaced while length and signature
  # stay constant.
  def test_substitution_preserves_signature(self):
    original = b"SUB!" + b"\x01\x02\x03\x04"
    report = repro.minimize_sample(
        self.spec("substitute_me"), original, timeout=self.timeout,
        max_attempts=200)
    self.assertEqual(report.status, "success")
    self.assertEqual(report.final_length, len(original))
    self.assertEqual(
        report.final_replay.verdict.signature, report.signature)

  # Hard constraint 5: concurrent processing shares no mutable state and a
  # failing unit cannot pollute another unit's signature.
  def test_concurrent_units_are_isolated(self):
    samples = [b"CRSH\x01aa", b"VAL!bb", b"IDXcc", b"no-trigger"]
    # Two distinct targets deliberately share one target module process space
    # only through fresh subprocesses; run replay and minimize concurrently.
    concurrent = repro.batch_replay(
        self.spec("branch_crash"), samples, timeout=self.timeout,
        max_workers=3)
    serial = [
        repro.replay_sample(self.spec("branch_crash"), s, timeout=self.timeout)
        for s in samples
    ]
    self.assertEqual(len(concurrent), len(serial))
    for c, s in zip(concurrent, serial):
      self.assertEqual(c.verdict.signature, s.verdict.signature)
      self.assertEqual(c.verdict.kind, s.verdict.kind)

    mixed = repro.batch_minimize(
        self.spec("branch_crash"),
        [b"CRSH\x01" + b"x" * 30, b"unreproducible-input"],
        timeout=self.timeout, max_workers=2)
    self.assertEqual(mixed[0].status, "success")
    self.assertEqual(mixed[1].status, "unreproducible")
    # The failing unit must not have altered the successful unit's signature.
    self.assertEqual(
        mixed[0].final_replay.verdict.signature,
        mixed[0].signature)

  # Resource exhaustion is reported, not silently truncated into a "success".
  def test_resource_exhaustion_falls_back(self):
    original = b"CRSH\x01" + b"P" * 40
    report = repro.minimize_sample(
        self.spec("branch_crash"), original, timeout=self.timeout,
        max_attempts=2)
    self.assertEqual(report.status, "fallback")
    self.assertEqual(report.data, original)
    self.assertEqual(report.reason, "resource_exhausted")


if __name__ == "__main__":
  unittest.main()
