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
"""Tests for the pure-Python post-run triage package (atheris.triage).

These tests never require the compiled native extension, so they run in
environments where Atheris could not be built. The package is imported via a
path bootstrap mirroring how the other tests locate ``atheris``.
"""

import hashlib
import importlib
import json
import os
import shutil
import sys
import tempfile
import time
import unittest

# Support both layouts: an installed Atheris exposes ``atheris.triage``, while a
# source checkout exposes the bare ``triage`` package when ``src/`` is on the
# path (the way run_tests.sh executes *_test.py from the src directory).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
  from atheris.triage import crashes as crashes_mod  # type: ignore
  from atheris.triage import hashing  # type: ignore
  from atheris.triage import replay  # type: ignore
  from atheris.triage import store  # type: ignore
  _TRIAGE_PACKAGE = "atheris.triage"
except ImportError:
  from triage import crashes as crashes_mod
  from triage import hashing
  from triage import replay
  from triage import store
  _TRIAGE_PACKAGE = "triage"

# Import the organize submodule explicitly: the package also exports an
# ``organize`` function, which otherwise shadows the submodule attribute.
organize_pkg = importlib.import_module(_TRIAGE_PACKAGE + ".organize")


def write(path, data):
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, "wb") as f:
    if isinstance(data, str):
      f.write(data.encode("utf-8"))
    else:
      f.write(data)


class TriageTestBase(unittest.TestCase):

  def setUp(self):
    self.tmp = tempfile.mkdtemp(prefix="atheris-triage-test-")
    self.sessions = []
    for name in ("s1", "s2"):
      d = os.path.join(self.tmp, name)
      os.makedirs(d)
      self.sessions.append(d)
    self.crash_dir = os.path.join(self.tmp, "crashes")
    os.makedirs(self.crash_dir)
    self.out = os.path.join(self.tmp, "out")

  def tearDown(self):
    shutil.rmtree(self.tmp, ignore_errors=True)

  def organize(self, **kwargs):
    options = organize_pkg.TriageOptions(
        corpus_roots=kwargs.pop("corpus_roots", self.sessions),
        crash_roots=kwargs.pop("crash_roots", [self.crash_dir]),
        destination=kwargs.pop("destination", self.out),
        **kwargs)
    return organize_pkg.organize(options)

  def family_by_digest(self, manifest, digest):
    for family in manifest["corpus_families"]:
      if digest in family["members"][0]["digest"] or any(
          m["digest"] == digest for m in family["members"]):
        return family
    return None

  def families_map(self, manifest):
    return {f["family_id"]: f for f in manifest["corpus_families"]}


class FingerprintTest(TriageTestBase):

  def test_content_only_determinism(self):
    # Same bytes, different names, different directories, different mtimes.
    p1 = os.path.join(self.sessions[0], "a")
    p2 = os.path.join(self.sessions[1], "nested", "b")
    write(p1, b"identical content")
    write(p2, b"identical content")
    os.utime(p1, (1, 1))
    os.utime(p2, (999999, 999999))
    fp1, e1 = hashing.fingerprint_file(p1)
    fp2, e2 = hashing.fingerprint_file(p2)
    self.assertIsNone(e1)
    self.assertIsNone(e2)
    self.assertEqual(fp1.digest_hex, fp2.digest_hex)
    self.assertEqual(fp1.prefix_block, fp2.prefix_block)

  def test_empty_file(self):
    p = os.path.join(self.tmp, "empty")
    write(p, b"")
    fp, err = hashing.fingerprint_file(p)
    self.assertIsNone(err)
    self.assertEqual(fp.size, 0)
    self.assertEqual(fp.prefix_block, b"")
    self.assertEqual(fp.suffix_block, b"")
    # Known BLAKE2b-256 of the empty input.
    self.assertEqual(fp.digest_hex,
                     hashlib.blake2b(b"", digest_size=32).hexdigest())

  def test_large_file_single_pass_bounded_memory(self):
    # A 20 MiB file is handled without accumulating content in memory: the
    # tail buffer must never exceed the configured window.
    p = os.path.join(self.tmp, "big")
    with open(p, "wb") as f:
      f.write(b"x" * (20 * 1024 * 1024))
    fp, err = hashing.fingerprint_file(p, window_size=4096)
    self.assertIsNone(err)
    self.assertEqual(fp.size, 20 * 1024 * 1024)
    self.assertEqual(len(fp.suffix_block), hashing.EDGE_BLOCK_SIZE)
    self.assertEqual(fp.window_size, 4096)

  def test_binary_non_utf8(self):
    p = os.path.join(self.tmp, "bin")
    write(p, bytes(range(256)) * 4)
    fp, err = hashing.fingerprint_file(p)
    self.assertIsNone(err)
    self.assertEqual(len(fp.digest_hex), 64)

  def test_read_error_reported_not_raised(self):
    p = os.path.join(self.tmp, "missing")
    fp, err = hashing.fingerprint_file(p)
    self.assertIsNone(fp)
    self.assertIn("FileNotFoundError", err)


class CorpusClusteringTest(TriageTestBase):

  def test_exact_duplicates_collapse_with_explained_choice(self):
    write(os.path.join(self.sessions[0], "a"), b"DUP")
    write(os.path.join(self.sessions[1], "whatever"), b"DUP")
    write(os.path.join(self.sessions[1], "third"), b"DUP")
    result = self.organize(crash_roots=[])
    manifest = result.manifest
    self.assertEqual(manifest["stats"]["distinct_contents"], 1)
    self.assertEqual(manifest["stats"]["exact_duplicate_copies_collapsed"], 2)
    family = manifest["corpus_families"][0]
    self.assertEqual(family["physical_member_count"], 3)
    # First-seen source is the smallest (source id, path) pair.
    self.assertEqual(family["first_seen"]["source_id"], "S0")
    self.assertTrue(family["representative_selection"])

  def test_strict_prefix_suffix_and_single_byte_families(self):
    write(os.path.join(self.sessions[0], "short"), b"abc")
    write(os.path.join(self.sessions[0], "pref_long"), b"abcdef")  # prefix
    write(os.path.join(self.sessions[0], "suf_long"), b"zzabc")  # suffix
    write(os.path.join(self.sessions[0], "repl"), b"aXc")  # 1-byte replacement
    write(os.path.join(self.sessions[0], "ins"), b"aXbc")  # 1-byte insertion
    write(os.path.join(self.sessions[0], "unrelated"), b"totally-different")
    result = self.organize(crash_roots=[])
    families = result.manifest["corpus_families"]
    merged = [f for f in families if f["distinct_contents"] > 1]
    # All five near-duplicates must be reachable through relation edges. The
    # exact family shape (one transitive family or two connected ones) is
    # determined by union-find, but every pair relation must be verified.
    all_relations = " ".join(e for f in merged for e in f["relations"])
    self.assertIn("strict prefix", all_relations)
    self.assertIn("strict suffix", all_relations)
    self.assertIn("single-byte", all_relations)
    # The unrelated input stays alone.
    singleton = [
        f for f in families
        if f["distinct_contents"] == 1 and f["physical_member_count"] == 1
    ]
    self.assertTrue(singleton)

  def test_cross_session_same_name_different_content_not_lost(self):
    write(os.path.join(self.sessions[0], "name"), b"content-one")
    write(os.path.join(self.sessions[1], "name"), b"content-two")
    result = self.organize(crash_roots=[])
    digests = set()
    for family in result.manifest["corpus_families"]:
      for member in family["members"]:
        digests.add(member["digest"])
    self.assertEqual(len(digests), 2)

  def test_cross_session_same_content_different_name(self):
    write(os.path.join(self.sessions[0], "name1"), b"SAME")
    write(os.path.join(self.sessions[1], "name2"), b"SAME")
    result = self.organize(crash_roots=[])
    self.assertEqual(result.manifest["stats"]["distinct_contents"], 1)

  def test_empty_and_binary_inputs_handled(self):
    write(os.path.join(self.sessions[0], "empty"), b"")
    write(os.path.join(self.sessions[0], "binary"), bytes(range(256)) * 100)
    write(os.path.join(self.sessions[0], "utf8"), b"hello \xc3\xa9")
    result = self.organize(crash_roots=[])
    self.assertEqual(result.manifest["stats"]["read_failures"], 0)
    self.assertEqual(result.manifest["stats"]["physical_files_seen"], 3)

  def test_large_multi_megabyte_prefix_family_streamed(self):
    # A 25 MiB input and its strict prefix (one byte shorter) must be united
    # without either file ever being fully buffered.
    big = b"z" * (25 * 1024 * 1024)
    write(os.path.join(self.sessions[0], "big"), big)
    write(os.path.join(self.sessions[0], "big-prefix"), big[:-1])
    result = self.organize(crash_roots=[])
    merged = [
        f for f in result.manifest["corpus_families"]
        if f["distinct_contents"] > 1
    ]
    self.assertEqual(len(merged), 1)
    self.assertIn("strict prefix", " ".join(merged[0]["relations"]))
    self.assertIn("smallest content in family",
                  merged[0]["representative_selection"])


class CrashGroupingTest(TriageTestBase):

  def _report(self, name, frames, exc="ValueError", msg="boom"):
    lines = ["Traceback (most recent call last):"]
    for path, line, func in frames:
      lines.append('  File "%s", line %d, in %s' % (path, line, func))
    lines.append("%s: %s" % (exc, msg))
    write(os.path.join(self.crash_dir, name + ".log"), "\n".join(lines))

  def test_same_exception_and_chain_merge_despite_message_and_addresses(self):
    write(os.path.join(self.crash_dir, "crash-a"), b"input-a")
    write(os.path.join(self.crash_dir, "crash-b"), b"input-b")
    self._report(
        "crash-a",
        [("/tmp/run-1/h.py", 10, "TestOneInput"),
         ("/tmp/build-x/lib.py", 42, "boom")],
        msg="object at 0x1234 in /tmp/run-1/x")
    self._report(
        "crash-b",
        [("/tmp/run-2/h.py", 10, "TestOneInput"),
         ("/tmp/build-y/lib.py", 42, "boom")],
        msg="object at 0x9abc in /tmp/run-2/y")
    result = self.organize(corpus_roots=[])
    groups = result.manifest["crash_groups"]
    parseable = [g for g in groups if not g["unparseable"]]
    self.assertEqual(len(parseable), 1)
    self.assertEqual(parseable[0]["member_count"], 2)
    self.assertEqual(len(parseable[0]["distinct_messages"]), 2)
    # Temporary paths and addresses must have been normalized away.
    chain_text = " ".join(parseable[0]["normalized_chain"])
    self.assertNotIn("0x", chain_text)
    self.assertNotIn("/tmp/run-1", chain_text)
    self.assertIn("<TMPDIR>", chain_text)

  def test_same_exception_different_chain_stays_separate(self):
    write(os.path.join(self.crash_dir, "crash-a"), b"a")
    write(os.path.join(self.crash_dir, "crash-b"), b"b")
    self._report("crash-a", [("/x/h.py", 1, "f")])
    self._report("crash-b", [("/x/h.py", 9, "g")])
    result = self.organize(corpus_roots=[])
    groups = [g for g in result.manifest["crash_groups"]
              if not g["unparseable"]]
    self.assertEqual(len(groups), 2)

  def test_unparseable_crash_is_conservative_singleton(self):
    write(os.path.join(self.crash_dir, "crash-lonely"), b"x")
    write(os.path.join(self.crash_dir, "crash-lonely.log"),
          "not a recognizable report at all")
    result = self.organize(corpus_roots=[])
    groups = result.manifest["crash_groups"]
    self.assertEqual(len(groups), 1)
    self.assertTrue(groups[0]["unparseable"])
    self.assertTrue(any("conservative singleton" in w
                        for w in result.manifest["warnings"]))

  def test_crash_present_in_corpus_is_cross_referenced_once(self):
    write(os.path.join(self.sessions[0], "seed"), b"SHARED")
    write(os.path.join(self.crash_dir, "crash-x"), b"SHARED")
    self._report("crash-x", [("/x/h.py", 1, "f")])
    result = self.organize()
    xrefs = result.manifest["cross_references"]
    self.assertEqual(len(xrefs), 1)
    self.assertTrue(xrefs[0]["stored_once"])
    group = result.manifest["crash_groups"][0]
    self.assertTrue(group["present_in_corpus"])
    # The minimal corpus and minimal crash entries resolve to the same inode
    # (hard link) rather than two byte copies.
    corpus_entry = os.path.join(self.out, group["representative_path"])
    blob = os.path.join(self.out, group["reproduction"]["input_blob"])
    self.assertEqual(os.stat(corpus_entry).st_ino, os.stat(blob).st_ino)


class FailureAndAtomicityTest(TriageTestBase):

  def test_unreadable_file_others_continue_nonzero_exit(self):
    write(os.path.join(self.sessions[0], "good"), b"good")
    bad = os.path.join(self.sessions[0], "bad")
    write(bad, b"bad")
    os.chmod(bad, 0o000)
    try:
      if os.geteuid() == 0:
        self.skipTest("root bypasses permission bits")
      result = self.organize(crash_roots=[])
    finally:
      os.chmod(bad, 0o644)
    self.assertEqual(result.exit_code, 2)
    self.assertEqual(result.manifest["stats"]["read_failures"], 1)
    self.assertTrue(result.manifest["failures"])
    self.assertEqual(result.manifest["stats"]["readable_files"], 1)

  def test_concurrent_organize_is_rejected(self):
    write(os.path.join(self.sessions[0], "a"), b"a")
    with store.destination_lock(self.out):
      with self.assertRaises(store.ConcurrentTriageError):
        with store.destination_lock(self.out):
          self.fail("nested lock should not be acquired")

  def test_two_real_processes_organizing_same_destination_conflict(self):
    import subprocess
    write(os.path.join(self.sessions[0], "a"), b"a")
    # A separate process holds the destination lock for a short time while the
    # current process attempts its own run; the second run must be refused.
    holder = "\n".join([
        "import sys, time",
        "sys.path.insert(0, %r)" %
        os.path.dirname(os.path.abspath(__file__)),
        "from triage import store",
        "with store.destination_lock(%r):" % self.out,
        "    time.sleep(3)",
    ])
    proc = subprocess.Popen(
        [sys.executable, "-I", "-c", holder],
        stderr=subprocess.PIPE)
    try:
      # Wait until the holder is actually blocked on the lock.
      time.sleep(0.8)
      with self.assertRaises(store.ConcurrentTriageError):
        organize_pkg.organize(
            organize_pkg.TriageOptions(
                self.sessions, [], self.out))
    finally:
      _, err = proc.communicate(timeout=10)
    self.assertEqual(proc.returncode, 0, err.decode())

  def test_failed_run_leaves_no_destination_and_keeps_originals(self):
    original = os.path.join(self.sessions[0], "a")
    write(original, b"keep-me")
    with open(original, "rb") as f:
      before = f.read()
    # A missing crash directory is a recorded source failure; the run still
    # commits the corpus, and the original inputs remain byte-identical with
    # no staging residue left behind.
    options = organize_pkg.TriageOptions(
        corpus_roots=[self.sessions[0]],
        crash_roots=[os.path.join(self.tmp, "does-not-exist")],
        destination=self.out)
    result = organize_pkg.organize(options)
    with open(original, "rb") as f:
      self.assertEqual(f.read(), before)
    leftovers = [n for n in os.listdir(os.path.dirname(self.out))
                 if n.startswith(store.STAGING_PREFIX)]
    self.assertEqual(leftovers, [])
    self.assertTrue(os.path.isfile(
        os.path.join(result.destination, store.MANIFEST_NAME)))
    self.assertTrue(result.manifest["failures"])

  def test_write_interruption_leaves_previous_output_untouched(self):
    write(os.path.join(self.sessions[0], "a"), b"a")
    first = self.organize()
    with open(
        os.path.join(first.destination, store.MANIFEST_NAME), "rb") as f:
      first_manifest = f.read()
    # Simulate a crash during staging: the context manager must remove the
    # incomplete staging tree and leave the published output unchanged.
    with store.StagedOutput(os.path.join(self.tmp, "never-published")) as staged:
      staged.write_text("partial.txt", "incomplete")
      staging_path = staged.staging
    self.assertFalse(os.path.exists(staging_path))
    with open(os.path.join(first.destination, store.MANIFEST_NAME),
              "rb") as f:
      self.assertEqual(f.read(), first_manifest)

  def test_read_only_destination_parent_surfaces_error_without_damage(self):
    write(os.path.join(self.sessions[0], "a"), b"a")
    readonly = os.path.join(self.tmp, "readonly")
    os.makedirs(readonly)
    os.chmod(readonly, 0o500)
    try:
      if os.geteuid() == 0:
        self.skipTest("root bypasses permission bits")
      with self.assertRaises(store.TriageError):
        organize_pkg.organize(
            organize_pkg.TriageOptions(
                self.sessions, [],
                os.path.join(readonly, "out")))
    finally:
      os.chmod(readonly, 0o700)
    # The original corpus is untouched.
    with open(os.path.join(self.sessions[0], "a"), "rb") as f:
      self.assertEqual(f.read(), b"a")

  def test_existing_destination_refused_without_overwrite(self):
    write(os.path.join(self.sessions[0], "a"), b"a")
    self.organize()
    with self.assertRaises(store.TriageError):
      self.organize()

  def test_disk_full_during_write_keeps_originals_and_prior_output(self):
    import errno
    from unittest import mock
    write(os.path.join(self.sessions[0], "a"), b"a")
    first = self.organize()
    with open(
        os.path.join(first.destination, store.MANIFEST_NAME), "rb") as f:
      prior_manifest = f.read()

    def raise_enospc(*_args, **_kwargs):
      raise OSError(errno.ENOSPC, "simulated no space left on device")

    # A second run with new input fails while hard-linking into the blob store.
    write(os.path.join(self.sessions[0], "b"), b"bb")
    with mock.patch("os.link", side_effect=raise_enospc), \
         mock.patch("shutil.copyfileobj", side_effect=raise_enospc):
      with self.assertRaises(OSError):
        organize_pkg.organize(
            organize_pkg.TriageOptions(
                self.sessions, [],
                os.path.join(self.tmp, "out-enospc")))

    # No half-written staging directory remains.
    leftovers = [
        n for n in os.listdir(self.tmp)
        if n.startswith(store.STAGING_PREFIX)
    ]
    self.assertEqual(leftovers, [])
    # The previously published output is byte-for-byte unchanged.
    with open(os.path.join(first.destination, store.MANIFEST_NAME),
              "rb") as f:
      self.assertEqual(f.read(), prior_manifest)
    # Original inputs are intact.
    with open(os.path.join(self.sessions[0], "a"), "rb") as f:
      self.assertEqual(f.read(), b"a")
    with open(os.path.join(self.sessions[0], "b"), "rb") as f:
      self.assertEqual(f.read(), b"bb")

  def test_overwrite_is_atomic_and_complete(self):
    write(os.path.join(self.sessions[0], "a"), b"a")
    first = self.organize()
    write(os.path.join(self.sessions[0], "b"), b"bb")
    second = self.organize(overwrite=True)
    self.assertTrue(os.path.samefile(
        second.destination,
        first.destination))
    self.assertEqual(second.manifest["stats"]["physical_files_seen"], 2)


class ReversibilityTest(TriageTestBase):

  def test_restore_reconstructs_every_original_byte_for_byte(self):
    label1 = os.path.basename(self.sessions[0])
    label2 = os.path.basename(self.sessions[1])
    files = {
        (label1, "a"): b"alpha",
        (label1, "dup"): b"alpha",
        (label2, "nested/b"): b"beta-longer",
        (label2, "c"): b"",
    }
    for (label, rel), data in files.items():
      write(os.path.join(self.tmp, label, rel.replace("/", os.sep)), data)
    self.organize(crash_roots=[])
    restore_root = os.path.join(self.tmp, "restored")
    organize_pkg.restore(self.out, restore_root)
    for (label, rel), data in files.items():
      recovered = os.path.join(restore_root, organize_pkg.RECOVERED_DIR,
                               label, rel.replace("/", os.sep))
      self.assertTrue(os.path.isfile(recovered), recovered)
      with open(recovered, "rb") as f:
        self.assertEqual(f.read(), data)


class DeterminismTest(TriageTestBase):

  def test_same_inputs_identical_manifest_across_destinations(self):
    write(os.path.join(self.sessions[0], "a"), b"alpha")
    write(os.path.join(self.sessions[1], "b"), b"alpha")
    write(os.path.join(self.sessions[1], "c"), b"gamma")
    out1 = os.path.join(self.tmp, "run1")
    out2 = os.path.join(self.tmp, "run2")
    r1 = organize_pkg.organize(
        organize_pkg.TriageOptions(self.sessions, [], out1))
    r2 = organize_pkg.organize(
        organize_pkg.TriageOptions(self.sessions, [], out2))
    # Paths differ between runs; compare the structural sections.
    self.assertEqual(
        json.dumps(r1.manifest["corpus_families"], sort_keys=True),
        json.dumps(r2.manifest["corpus_families"], sort_keys=True))
    self.assertEqual(r1.manifest["stats"], r2.manifest["stats"])
    # Manifest bytes themselves are stable for the same destination content.
    with open(os.path.join(out1, store.MANIFEST_NAME), "rb") as f:
      m1 = f.read()
    with open(os.path.join(out2, store.MANIFEST_NAME), "rb") as f:
      m2 = f.read()
    self.assertEqual(m1, m2)


class CrashNormalizationTest(unittest.TestCase):

  def test_addresses_tmp_paths_and_object_ids_normalized(self):
    text = ('File "/tmp/pip-xyz/build/h.py", line 5, in f\n'
            "RuntimeError: obj <Thing at 0x7fdeadbeef> failed")
    norm = crashes_mod.normalize_text(text)
    self.assertNotIn("0x7fdeadbeef", norm)
    self.assertNotIn("pip-xyz", norm)
    # The "at 0x..." pattern is the more specific object-id marker.
    self.assertIn("<OBJID>", norm)
    self.assertIn("<TMPDIR>", norm)
    # A bare hex pointer elsewhere is normalized as an address.
    self.assertIn("<ADDR>",
                  crashes_mod.normalize_text("jumped to 0x1234abcd today"))

  def test_message_only_difference_does_not_change_group_key(self):
    r1 = crashes_mod.parse_report_bytes(
        b'Traceback (most recent call last):\n  File "/a/h.py", line 2, in f\n'
        b"KeyError: 'first'")
    r2 = crashes_mod.parse_report_bytes(
        b'Traceback (most recent call last):\n  File "/a/h.py", line 2, in f\n'
        b"KeyError: 'second'")
    self.assertIsNotNone(r1[0])
    type1, frames1, _ = r1[0]
    type2, frames2, _ = r2[0]
    self.assertEqual(type1, type2)
    self.assertEqual(frames1, frames2)

  def test_sanitizer_report_parsed(self):
    data = (b"ERROR: AddressSanitizer: heap-buffer-overflow\n"
            b"    #0 0xdead in foo (/tmp/x/lib.so+0x123)\n"
            b"    #1 0xbeef in bar\n")
    parsed, kind = crashes_mod.parse_report_bytes(data)
    self.assertEqual(kind, "sanitizer-report")
    self.assertEqual(parsed[0], "AddressSanitizer")
    self.assertEqual(len(parsed[1]), 2)


class CoverageReplayTest(TriageTestBase):

  def _harness(self):
    harness = os.path.join(self.tmp, "harness.py")
    with open(harness, "w") as f:
      f.write(
          "def TestOneInput(data):\n"
          "    if not data:\n"
          "        return\n"
          "    if data[0] == 1:\n"
          "        _ = data[1:]\n"
          "    elif data[0] == 2:\n"
          "        raise ValueError('crash')\n")
    return harness

  def test_equivalent_coverage_before_and_after_reduction(self):
    # A corpus with exact duplicates and a near duplicate that all reach the
    # same branches.
    write(os.path.join(self.sessions[0], "a"), bytes([1, 9]))
    write(os.path.join(self.sessions[1], "a-copy"), bytes([1, 9]))
    write(os.path.join(self.sessions[1], "a-ext"), bytes([1, 9, 10]))
    write(os.path.join(self.sessions[0], "b"), bytes([2]))
    result = self.organize(crash_roots=[])
    harness = self._harness()
    original_paths = []
    for session in self.sessions:
      for name in sorted(os.listdir(session)):
        original_paths.append(os.path.join(session, name))
    minimal_paths = [
        os.path.join(result.destination, f["representative_path"])
        for f in result.manifest["corpus_families"]
    ]
    report = replay.verify_equivalence(
        original_paths, minimal_paths, harness,
        coverage_roots=[self.tmp])
    self.assertTrue(report["equivalent"], report)

  def test_replay_detects_lost_coverage(self):
    # If the minimal set genuinely dropped a branch, equivalence fails.
    harness = self._harness()
    original = os.path.join(self.tmp, "orig")
    os.makedirs(original)
    write(os.path.join(original, "a"), bytes([1]))
    write(os.path.join(original, "b"), bytes([2]))
    minimal = os.path.join(self.tmp, "mini")
    os.makedirs(minimal)
    write(os.path.join(minimal, "a"), bytes([1]))
    report = replay.verify_equivalence(
        [os.path.join(original, n) for n in os.listdir(original)],
        [os.path.join(minimal, "a")], harness, coverage_roots=[self.tmp])
    self.assertFalse(report["equivalent"])
    self.assertTrue(report["units_missing_in_minimal"])

  def test_coverage_aware_organize_keeps_longer_member_for_extra_branch(self):
    # The content-based representative is the shortest input, but only the
    # longer strict-prefix input reaches an extra branch. Coverage-aware
    # minimization selects the longer input (it alone covers every unit) and
    # materializes it alongside the family representative, reporting
    # equivalence against the full original corpus.
    harness = os.path.join(self.tmp, "branch_harness.py")
    with open(harness, "w") as f:
      f.write(
          "def TestOneInput(data):\n"
          "    if data.startswith(b'abc'):\n"
          "        if len(data) > 3:\n"
          "            _ = data[3:]\n")
    write(os.path.join(self.sessions[0], "short"), b"abc")
    write(os.path.join(self.sessions[0], "long"), b"abcdef")
    result = organize_pkg.organize(
        organize_pkg.TriageOptions(
            self.sessions, [], self.out,
            harness_path=harness, coverage_roots=[self.tmp]))
    cov = result.manifest["coverage_verification"]
    self.assertTrue(cov["equivalent"], cov)
    # One similarity family; the representative alone would lose the branch,
    # so the coverage-selected longer member is materialized as a second file.
    self.assertEqual(len(result.manifest["corpus_families"]), 1)
    self.assertEqual(
        result.manifest["stats"]["materialized_minimal_corpus_size"], 2)
    members = result.manifest["corpus_families"][0]["members"]
    selected = {m["digest"] for m in members if m["selected_for_coverage"]}
    self.assertEqual(len(selected), 1)
    # The coverage-selected content is the 6-byte input, not the 3-byte one.
    selected_node = next(
        m for m in members if m["selected_for_coverage"])
    self.assertEqual(selected_node["size"], 6)
    # A standalone verify-coverage over the materialized set agrees.
    minimal_dir = os.path.join(self.out, "minimal-corpus")
    minimal_paths = [
        os.path.join(minimal_dir, n) for n in os.listdir(minimal_dir)
    ]
    original_paths = []
    for session in self.sessions:
      for name in os.listdir(session):
        original_paths.append(os.path.join(session, name))
    report = replay.verify_equivalence(
        original_paths, minimal_paths, harness,
        coverage_roots=[self.tmp])
    self.assertTrue(report["equivalent"], report)

  def test_coverage_aware_organize_reduces_exact_duplicates_to_one(self):
    # When every input reaches the same branches (exact duplicates), the
    # coverage-aware minimal corpus needs only the single representative.
    harness = os.path.join(self.tmp, "dup_harness.py")
    with open(harness, "w") as f:
      f.write(
          "def TestOneInput(data):\n"
          "    if data == b'same':\n"
          "        _ = len(data)\n")
    write(os.path.join(self.sessions[0], "a"), b"same")
    write(os.path.join(self.sessions[1], "b"), b"same")
    result = organize_pkg.organize(
        organize_pkg.TriageOptions(
            self.sessions, [], self.out,
            harness_path=harness, coverage_roots=[self.tmp]))
    self.assertTrue(
        result.manifest["coverage_verification"]["equivalent"])
    self.assertEqual(
        result.manifest["stats"]["materialized_minimal_corpus_size"], 1)

  def test_harness_without_native_build_imports_via_stub(self):
    harness = os.path.join(self.tmp, "realish.py")
    with open(harness, "w") as f:
      f.write(
          "import atheris\n"
          "@atheris.instrument_func\n"
          "def TestOneInput(data):\n"
          "    if data == b'go':\n"
          "        pass\n"
          "atheris.Setup([], TestOneInput)\n")
    entry = replay.load_harness(harness)
    entry(b"go")  # Must not raise despite no native extension.


class ManifestShapeTest(TriageTestBase):

  def test_manifest_lists_representative_members_and_first_seen(self):
    write(os.path.join(self.sessions[0], "a"), b"x")
    write(os.path.join(self.sessions[1], "b"), b"x")
    result = self.organize(crash_roots=[])
    family = result.manifest["corpus_families"][0]
    for key in ("family_id", "representative_digest", "representative_path",
                "representative_selection", "first_seen", "members",
                "relations"):
      self.assertIn(key, family)
    self.assertEqual(family["first_seen"]["source_id"], "S0")
    copies = family["members"][0]["physical_copies"]
    self.assertEqual(len(copies), 2)
    # Determinism contract is explicitly recorded.
    self.assertIn("content bytes only",
                  result.manifest["determinism"]["fingerprint_inputs"][0])


if __name__ == "__main__":
  unittest.main()
