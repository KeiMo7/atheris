# Copyright 2025 Google LLC
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
"""Tests for deterministic corpus replay and coverage consistency.

Run directly with the isolated interpreter so untrusted working-directory
modules cannot shadow stdlib imports::

    python3 -I src/replay_replay_test.py

The replay package is pure Python and does not require the compiled atheris
native extensions, which lets these tests run in environments without a C++
toolchain.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

# Allow running straight from the source tree (src/) as well as an installed
# atheris package.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
  sys.path.insert(0, _THIS_DIR)

from replay import corpus as corpus_lib  # noqa: E402
from replay import engine as engine_lib  # noqa: E402
from replay import snapshot as snapshot_lib  # noqa: E402
from replay.corpus import CorpusLoadError, load_corpus  # noqa: E402
from replay.coverage import (  # noqa: E402
    COVERAGE_SEMANTICS_VERSION,
    CoverageMap,
    CoverageUnavailableError,
    canonical_location,
    get_collector,
)
from replay.engine import (  # noqa: E402
    MAX_WORKERS_LIMIT,
    ReplayConfig,
    WorkerLimitError,
    replay_corpus,
)
from replay.snapshot import (  # noqa: E402
    Snapshot,
    SnapshotFormatVersionError,
    compare_snapshots,
    load_snapshot,
    save_snapshot,
)

# A target module written into a temp directory at test time.  Keeping the
# target in its own file makes its edge ids path-based and independently
# inspectable.
_TARGET_SOURCE = b'''
def replay_target(data):
    if not data:
        return "empty"
    kind = data[0]
    if kind == 1:
        if len(data) > 1 and data[1] == 2:
            return "one-two"
        return "one"
    if kind == 2:
        total = 0
        for byte in data[1:]:
            total += byte
        return total
    if kind == 3:
        raise ValueError("boom")
    return "other"


def broken_target(data):
    # Always raises: used to prove per-entry failure isolation.
    raise RuntimeError("broken target")


# Re-used by the interruption test: the interruptible function *is* the target
# (no wrapper frame), driven by a module-global switch so its line coverage is
# identical to the plain target.
interrupt_after = 0
interrupt_calls = 0


def interruptible_target(data):
    global interrupt_calls
    if not data:
        result = "empty"
    else:
        kind = data[0]
        if kind == 1:
            if len(data) > 1 and data[1] == 2:
                result = "one-two"
            else:
                result = "one"
        elif kind == 2:
            total = 0
            for byte in data[1:]:
                total += byte
            result = total
        elif kind == 3:
            raise ValueError("boom")
        else:
            result = "other"
    interrupt_calls += 1
    if interrupt_after and interrupt_calls >= interrupt_after:
        raise KeyboardInterrupt()
    return result
'''


class ReplayTestBase(unittest.TestCase):

  def setUp(self):
    self._tmp = tempfile.mkdtemp(prefix="atheris-replay-test-")
    self.corpus_dir = os.path.join(self._tmp, "corpus")
    os.makedirs(self.corpus_dir)
    target_path = os.path.join(self._tmp, "replay_target_mod.py")
    with open(target_path, "wb") as f:
      f.write(_TARGET_SOURCE)
    sys.path.insert(0, self._tmp)
    import replay_target_mod  # noqa: E402

    self.target_module = replay_target_mod
    self.target = replay_target_mod.replay_target
    self.broken_target = replay_target_mod.broken_target
    self.interruptible_target = replay_target_mod.interruptible_target
    replay_target_mod.interrupt_calls = 0
    replay_target_mod.interrupt_after = 0

  def tearDown(self):
    sys.path.remove(self._tmp)
    sys.modules.pop("replay_target_mod", None)
    shutil.rmtree(self._tmp, ignore_errors=True)

  def write_entry(self, name, content):
    path = os.path.join(self.corpus_dir, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
      f.write(content)
    return path


class CorpusLoadingTest(ReplayTestBase):

  def test_deterministic_order_and_dedup(self):
    self.write_entry("c.bin", b"\x01\x02")
    self.write_entry("a.bin", b"\x01\x02")  # duplicate content
    self.write_entry("b.bin", b"\x02\x03\x04")
    loaded = load_corpus(self.corpus_dir)
    self.assertEqual(
        [e.relative_path for e in loaded.entries],
        ["a.bin", "b.bin", "c.bin"],
    )
    by_name = {e.relative_path: e for e in loaded.entries}
    self.assertIsNone(by_name["a.bin"].duplicate_of)
    self.assertEqual(by_name["c.bin"].duplicate_of, "a.bin")
    # The canonical (de-duplicated) view keeps coverage accounting intact:
    # duplicate removal only drops redundant work, never an edge definition.
    self.assertEqual(
        [e.relative_path for e in loaded.canonical_entries],
        ["a.bin", "b.bin"],
    )
    # Scope hash is stable across re-loads.
    self.assertEqual(
        corpus_lib.corpus_scope_hash(loaded.entries),
        corpus_lib.corpus_scope_hash(load_corpus(self.corpus_dir).entries),
    )

  def test_unreadable_entry_is_named_not_fatal(self):
    self.write_entry("good.bin", b"\x01")
    bad = self.write_entry("sub/bad.bin", b"\x02")
    os.chmod(bad, 0)
    try:
      if os.access(bad, os.R_OK):
        self.skipTest("running as root; permission denial cannot be tested")
      loaded = load_corpus(self.corpus_dir)
    finally:
      os.chmod(bad, 0o600)
    self.assertEqual([e.relative_path for e in loaded.entries], ["good.bin"])
    self.assertEqual(len(loaded.failures), 1)
    self.assertEqual(loaded.failures[0].relative_path,
                     os.path.join("sub", "bad.bin"))
    self.assertEqual(loaded.failures[0].error_type, "PermissionError")
    # No content leakage in the failure record.
    self.assertNotIn(repr(b"\x02"), loaded.failures[0].error_message)

  def test_missing_corpus_raises_explicitly(self):
    with self.assertRaises(CorpusLoadError):
      load_corpus(os.path.join(self._tmp, "does-not-exist"))

  def test_original_corpus_never_modified(self):
    self.write_entry("a.bin", b"\x01\x02")
    self.write_entry("b.bin", b"\x02\x03")

    def snapshot_tree():
      snapshot = {}
      for root, _, files in os.walk(self.corpus_dir):
        for name in files:
          p = os.path.join(root, name)
          with open(p, "rb") as f:
            snapshot[p] = (os.stat(p).st_mtime_ns, f.read())
      return snapshot

    before = snapshot_tree()
    config = ReplayConfig(target=self.target, max_workers=2)
    replay_corpus(self.corpus_dir, config)
    self.assertEqual(before, snapshot_tree())


class CoverageSemanticsTest(unittest.TestCase):

  def test_saturating_counter_semantics(self):
    first = CoverageMap({"e": 200})
    second = CoverageMap({"e": 100, "f": 1})
    merged = CoverageMap.merge(first, second)
    self.assertEqual(merged.counts["e"], 255)  # saturated, never wrapped
    self.assertEqual(merged.counts["f"], 1)

  def test_merge_is_order_independent(self):
    maps = [
        CoverageMap({"a": 1, "b": 5}),
        CoverageMap({"b": 251, "c": 3}),
        CoverageMap({"a": 10, "c": 253}),
    ]
    forward = CoverageMap()
    for m in maps:
      forward = CoverageMap.merge(forward, m)
    reverse = CoverageMap()
    for m in reversed(maps):
      reverse = CoverageMap.merge(reverse, m)
    self.assertEqual(forward, reverse)

  def test_invalid_counts_rejected(self):
    with self.assertRaises(ValueError):
      CoverageMap({"e": 256})
    with self.assertRaises(ValueError):
      CoverageMap({"e": -1})

  def test_canonical_location_is_path_based(self):
    # Same relative module location under two different sys.path roots must
    # produce the same edge id: machine/path independence for imported code.
    root_a = os.path.join(self._tmp_root(), "a")
    root_b = os.path.join(self._tmp_root(), "b")
    old_path = list(sys.path)
    try:
      sys.path[:] = [root_a] + [p for p in old_path if p]
      loc_a = canonical_location(os.path.join(root_a, "pkg", "m.py"), 7,
                                 "pkg.m.f")
      sys.path[:] = [root_b] + [p for p in old_path if p]
      loc_b = canonical_location(os.path.join(root_b, "pkg", "m.py"), 7,
                                 "pkg.m.f")
      self.assertEqual(loc_a, loc_b)
      self.assertTrue(loc_a.startswith(os.path.join("pkg", "m.py")))
    finally:
      sys.path[:] = old_path

  @staticmethod
  def _tmp_root():
    return tempfile.mkdtemp(prefix="atheris-canon-")


class ReplayConsistencyTest(ReplayTestBase):

  def _fixed_corpus(self):
    # Fixed, diverse corpus that exercises every branch of the target.
    self.write_entry("01-empty.bin", b"")
    self.write_entry("02-one.bin", b"\x01")
    self.write_entry("03-one-two.bin", b"\x01\x02")
    self.write_entry("04-loop.bin", b"\x02\x05\x06\x07")
    self.write_entry("05-other.bin", b"\x09")
    self.write_entry("06-raises.bin", b"\x03")
    # Exact duplicate of an existing input.
    self.write_entry("07-dup.bin", b"\x01\x02")

  def test_repeated_replay_has_identical_coverage(self):
    """Acceptance: same corpus twice -> coverage exactly equal."""
    self._fixed_corpus()
    first = replay_corpus(
        self.corpus_dir, ReplayConfig(target=self.target, run_id="run-A")
    )
    second = replay_corpus(
        self.corpus_dir, ReplayConfig(target=self.target, run_id="run-B")
    )
    self.assertEqual(first.coverage, second.coverage)
    self.assertEqual(
        first.coverage.ordered_items(), second.coverage.ordered_items()
    )
    self.assertEqual(first.corpus_scope_hash, second.corpus_scope_hash)
    # Different run ids must not perturb the statistics.
    self.assertNotEqual(first.run_id, second.run_id)
    self.assertIn("07-dup.bin", first.skipped_duplicates)

  def test_parallel_and_serial_counts_are_equal(self):
    """Acceptance: parallel replay matches serial replay exactly."""
    self._fixed_corpus()
    serial = replay_corpus(
        self.corpus_dir, ReplayConfig(target=self.target, max_workers=1)
    )
    parallel = replay_corpus(
        self.corpus_dir, ReplayConfig(target=self.target, max_workers=4)
    )
    wide = replay_corpus(
        self.corpus_dir, ReplayConfig(target=self.target, max_workers=8)
    )
    self.assertEqual(serial.coverage, parallel.coverage)
    self.assertEqual(serial.coverage, wide.coverage)
    # Edge-level identity, not just headline counts.
    self.assertEqual(
        serial.coverage.ordered_items(), parallel.coverage.ordered_items()
    )
    # Every entry appears exactly once, in ordinal order, regardless of
    # scheduling.
    names_serial = [r.relative_path for r in serial.results]
    names_parallel = [r.relative_path for r in parallel.results]
    self.assertEqual(names_serial, names_parallel)
    self.assertEqual(
        names_serial, sorted(names_serial)
    )

  def test_result_independent_of_machine_core_count_simulation(self):
    """Varying worker count (proxy for core count) cannot change counts."""
    self._fixed_corpus()
    reference = replay_corpus(
        self.corpus_dir, ReplayConfig(target=self.target, max_workers=1)
    )
    for workers in (2, 3, 5, 7, 13):
      result = replay_corpus(
          self.corpus_dir,
          ReplayConfig(target=self.target, max_workers=workers),
      )
      self.assertEqual(
          result.coverage,
          reference.coverage,
          msg=f"coverage diverged with {workers} workers",
      )

  def test_failing_entry_is_isolated_and_attributed(self):
    # One entry makes even the good target raise; the batch must finish and
    # attribute the failure to that exact corpus item.
    self.write_entry("good-1.bin", b"\x01")
    self.write_entry("good-2.bin", b"\x02\x01")
    self.write_entry("bad.bin", b"\x03")  # raises ValueError
    result = replay_corpus(
        self.corpus_dir, ReplayConfig(target=self.target)
    )
    self.assertEqual(len(result.entry_failures), 1)
    failure = result.entry_failures[0]
    self.assertEqual(failure.relative_path, "bad.bin")
    self.assertEqual(failure.error_type, "ValueError")
    # Sensitive/raw input content must not appear in the record; the target's
    # message is generic here, but assert no byte payload leaked.
    self.assertNotIn(repr(b"\x03"), failure.error_message)
    replayed = {r.relative_path for r in result.results}
    self.assertEqual(replayed, {"good-1.bin", "good-2.bin"})
    # Coverage from the good entries is still collected.
    self.assertGreater(result.coverage.total_edges, 0)

  def test_always_broken_target_does_not_kill_batch(self):
    self.write_entry("a.bin", b"\x01")
    self.write_entry("b.bin", b"\x02")
    result = replay_corpus(
        self.corpus_dir, ReplayConfig(target=self.broken_target)
    )
    self.assertEqual(len(result.entry_failures), 2)
    self.assertEqual(
        sorted(f.relative_path for f in result.entry_failures),
        ["a.bin", "b.bin"],
    )

  def test_resume_after_interruption_matches_full_run(self):
    self._fixed_corpus()
    checkpoint = os.path.join(self._tmp, "state", "checkpoint.json")

    # The interruptible target is a real top-level function (no wrapper frame),
    # so its line coverage is identical to the uninterrupted target.  Flip the
    # module-global switch to interrupt after three entries.
    self.target_module.interrupt_after = 3
    self.target_module.interrupt_calls = 0

    interrupted = False
    try:
      replay_corpus(
          self.corpus_dir,
          ReplayConfig(
              target=self.interruptible_target,
              checkpoint_path=checkpoint,
              run_id="partial",
          ),
      )
    except KeyboardInterrupt:
      interrupted = True
    self.assertTrue(interrupted)
    self.assertTrue(os.path.exists(checkpoint))

    # No half-open temp files left beside the checkpoint.
    state_dir = os.path.dirname(checkpoint)
    self.assertFalse(
        any(name.endswith(".tmp") for name in os.listdir(state_dir))
    )

    # Resume with the interruption disabled; the same target continues from
    # the checkpoint and must reach exactly the full-run coverage.
    self.target_module.interrupt_after = 0
    resumed = replay_corpus(
        self.corpus_dir,
        ReplayConfig(
            target=self.interruptible_target,
            checkpoint_path=checkpoint,
            run_id="resumed",
        ),
    )
    self.assertTrue(resumed.resumed)
    full = replay_corpus(
        self.corpus_dir,
        ReplayConfig(
            target=self.interruptible_target, run_id="full-reference"
        ),
    )
    self.assertEqual(resumed.coverage, full.coverage)
    self.assertEqual(
        sorted(r.relative_path for r in resumed.results),
        sorted(r.relative_path for r in full.results),
    )

  def test_resume_rejects_different_corpus_scope(self):
    self.write_entry("a.bin", b"\x01")
    checkpoint = os.path.join(self._tmp, "cp.json")
    replay_corpus(
        self.corpus_dir,
        ReplayConfig(target=self.target, checkpoint_path=checkpoint),
    )
    # Mutate content under the same path -> scope hash changes.
    self.write_entry("a.bin", b"\x01\x02")
    with self.assertRaises(ValueError):
      replay_corpus(
          self.corpus_dir,
          ReplayConfig(target=self.target, checkpoint_path=checkpoint),
      )

  def test_worker_limit_is_rejected_explicitly(self):
    self.write_entry("a.bin", b"\x01")
    with self.assertRaises(WorkerLimitError):
      ReplayConfig(target=self.target, max_workers=0)
    with self.assertRaises(WorkerLimitError):
      ReplayConfig(
          target=self.target, max_workers=MAX_WORKERS_LIMIT + 1
      )

  def test_staging_temp_directory_is_always_cleaned(self):
    self.write_entry("a.bin", b"\x01")
    self.write_entry("b.bin", b"\x02")
    tmp_root = tempfile.gettempdir()
    before = set(os.listdir(tmp_root))
    replay_corpus(self.corpus_dir, ReplayConfig(target=self.target,
                                                max_workers=3))
    leftovers = [
        name
        for name in os.listdir(tmp_root)
        if name.startswith("atheris-replay-") and name not in before
    ]
    self.assertEqual(leftovers, [])
    # Also true when the target blows up on every entry.
    before = set(os.listdir(tmp_root))
    replay_corpus(
        self.corpus_dir, ReplayConfig(target=self.broken_target)
    )
    leftovers = [
        name
        for name in os.listdir(tmp_root)
        if name.startswith("atheris-replay-") and name not in before
    ]
    self.assertEqual(leftovers, [])


class SnapshotTest(ReplayTestBase):

  def _snapshot(self, run_id="run-1"):
    self.write_entry("a.bin", b"\x01\x02")
    self.write_entry("b.bin", b"\x02\x03\x04")
    result = replay_corpus(
        self.corpus_dir, ReplayConfig(target=self.target, run_id=run_id)
    )
    return result.to_snapshot(load_corpus(self.corpus_dir)), result

  def test_snapshot_records_scope_and_run_identity(self):
    snapshot, result = self._snapshot()
    self.assertEqual(snapshot.run_id, "run-1")
    self.assertEqual(snapshot.semantics_version, COVERAGE_SEMANTICS_VERSION)
    self.assertEqual(snapshot.corpus_scope_hash, result.corpus_scope_hash)
    self.assertEqual(snapshot.corpus_entry_count, 2)
    self.assertEqual(snapshot.corpus_replayed_count, 2)
    self.assertEqual(snapshot.coverage.total_edges,
                     snapshot.to_dict()["total_edges"])

  def test_snapshot_roundtrip_and_diff_identical(self):
    snapshot, _ = self._snapshot()
    path = os.path.join(self._tmp, "out", "snapshot.json")
    save_snapshot(snapshot, path)
    reloaded = load_snapshot(path)
    self.assertEqual(reloaded.coverage, snapshot.coverage)
    self.assertEqual(reloaded.run_id, snapshot.run_id)
    diff = compare_snapshots(snapshot, reloaded)
    self.assertTrue(diff.comparable)
    self.assertTrue(diff.identical)
    # No temp residue.
    self.assertFalse(
        any(n.endswith(".tmp") for n in os.listdir(os.path.dirname(path)))
    )

  def test_two_runs_snapshots_are_identical(self):
    self.write_entry("a.bin", b"\x01\x02")
    first = replay_corpus(
        self.corpus_dir, ReplayConfig(target=self.target, run_id="r1")
    ).to_snapshot(load_corpus(self.corpus_dir))
    second = replay_corpus(
        self.corpus_dir, ReplayConfig(target=self.target, run_id="r2")
    ).to_snapshot(load_corpus(self.corpus_dir))
    diff = compare_snapshots(first, second)
    self.assertTrue(diff.comparable, msg=diff.reason)
    self.assertTrue(diff.identical)

  def test_diff_refuses_different_scope(self):
    snapshot, _ = self._snapshot()
    other = Snapshot(
        run_id="other",
        corpus_scope_hash="deadbeef" * 8,
        corpus_entry_count=1,
        corpus_canonical_count=1,
        corpus_replayed_count=1,
        coverage=snapshot.coverage,
    )
    diff = compare_snapshots(snapshot, other)
    self.assertFalse(diff.comparable)
    self.assertIn("Corpus scope", diff.reason)

  def test_unknown_new_snapshot_version_refused(self):
    snapshot, _ = self._snapshot()
    path = os.path.join(self._tmp, "future.json")
    payload = snapshot.to_dict()
    payload["format_version"] = "99.0"
    with open(path, "w", encoding="utf-8") as f:
      json.dump(payload, f)
    with self.assertRaises(SnapshotFormatVersionError):
      load_snapshot(path)

  def test_snapshot_is_durable_json_with_sorted_edges(self):
    snapshot, _ = self._snapshot()
    path = os.path.join(self._tmp, "snap.json")
    save_snapshot(snapshot, path)
    with open(path, "r", encoding="utf-8") as f:
      payload = json.load(f)
    edges = [item["edge"] for item in payload["edges"]]
    self.assertEqual(edges, sorted(edges))
    # Indices are dense and follow the sorted order, so an edge's index is
    # stable regardless of the order it was first hit.
    self.assertEqual(
        [item["index"] for item in payload["edges"]],
        list(range(len(edges))),
    )


class DegradationTest(ReplayTestBase):

  def test_no_collector_raises_explicitly_not_empty_result(self):
    self.write_entry("a.bin", b"\x01")
    original_get = engine_lib.get_collector

    def force_unavailable():
      raise CoverageUnavailableError("simulated missing backend")

    engine_lib.get_collector = force_unavailable
    try:
      with self.assertRaises(CoverageUnavailableError):
        replay_corpus(
            self.corpus_dir, ReplayConfig(target=self.target)
        )
    finally:
      engine_lib.get_collector = original_get

  def test_real_collector_is_available_in_this_runtime(self):
    collector = get_collector()
    try:
      self.assertIn(
          collector.backend_name, ("sys.monitoring", "sys.settrace")
      )
      with collector.record_entry():
        _ = 1 + 1  # execute a real line
      cov = collector.get_entry_map()
      self.assertIsInstance(cov, CoverageMap)
    finally:
      collector.shutdown()


if __name__ == "__main__":
  unittest.main(verbosity=2)
