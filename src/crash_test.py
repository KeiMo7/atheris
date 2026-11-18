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

import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import textwrap
import unittest

from atheris.crash import CrashMinimizer
from atheris.crash import CrashStore
from atheris.crash import TargetRunner
from atheris.crash import extract_crash_signature
from atheris.crash import main


MULTI_ROOT_TARGET = """\
import os
import sys


def root_a(data):
  if b"CRASH_A" in data:
    raise RuntimeError("root A")


def root_b(data):
  if b"CRASH_B" in data:
    raise ValueError("root B")


data = open(sys.argv[1], "rb").read()
root_a(data)
root_b(data)
if b"NO_STACK" in data:
  os._exit(7)
"""


SLOW_ON_SMALL_INPUT_TARGET = """\
import sys
import time


data = open(sys.argv[1], "rb").read()
if b"CRASH_A" in data:
  if len(data) < 32:
    time.sleep(10)
  raise RuntimeError("root A")
"""


ATHERIS_TARGET = """\
import sys
import atheris


@atheris.instrument_func
def TestOneInput(data):
  if b"CRASH_A" in data:
    raise RuntimeError("root A")


atheris.Setup(sys.argv, TestOneInput)
atheris.Fuzz()
"""


class CrashManagerTests(unittest.TestCase):

  def setUp(self):
    self.temp_dir = tempfile.TemporaryDirectory()
    self.root = Path(self.temp_dir.name)
    self.target = self.root / "multi_root_target.py"
    self.target.write_text(MULTI_ROOT_TARGET)
    self.runner = TargetRunner(
        [sys.executable, "-S", str(self.target), "{input}"],
        timeout=5)

  def tearDown(self):
    self.runner.close()
    self.temp_dir.cleanup()

  def _artifact(self, name, data):
    path = self.root / name
    path.write_bytes(data)
    return path

  def test_extracts_python_and_sanitizer_stacks(self):
    python = extract_crash_signature(
        textwrap.dedent("""\
            Traceback (most recent call last):
              File "/tmp/fuzz.py", line 10, in TestOneInput
              File "/tmp/fuzz.py", line 20, in crash_a
            RuntimeError: boom
            """),
        1,
        False,
        "/tmp/input",
        "sample")
    self.assertEqual(python.kind, "python")
    self.assertEqual(python.crash_type, "RuntimeError")
    self.assertIn("crash_a", python.frames[-1])

    native = extract_crash_signature(
        textwrap.dedent("""\
            ==1==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x1234
                #0 0x1234 in crash_a /src/a.c:10:2
                #1 0x5678 in main /src/b.c:20:3
            SUMMARY: AddressSanitizer: heap-buffer-overflow /src/a.c:10:2
            """),
        -6,
        False,
        "/tmp/input",
        "sample")
    self.assertEqual(native.kind, "AddressSanitizer")
    self.assertEqual(native.crash_type, "heap-buffer-overflow")
    self.assertIn("/src/a.c:10:2", native.frames[0])

  def test_groups_persists_and_windows(self):
    store_dir = self.root / "store"
    clock = [1000.0]
    store = CrashStore(store_dir, clock=lambda: clock[0])

    artifacts = [
        self._artifact("crash-a-1", b"a" * 100 + b"CRASH_A"),
        self._artifact("crash-a-2", b"b" * 200 + b"CRASH_A"),
        self._artifact("crash-a-3", b"c" * 300 + b"CRASH_A"),
        self._artifact("crash-b-1", b"d" * 100 + b"CRASH_B"),
        self._artifact("crash-b-2", b"e" * 200 + b"CRASH_B"),
        self._artifact("crash-no-stack-1", b"NO_STACK-1"),
        self._artifact("crash-no-stack-2", b"NO_STACK-2"),
    ]
    for index, artifact in enumerate(artifacts):
      clock[0] = 1000.0 + index * 100
      store.ingest_file(artifact, self.runner)

    groups = store.groups()
    self.assertEqual(len(groups), 4)
    by_type = {
        group["signature"]["crash_type"]: group
        for group in groups
        if not group["signature"]["extraction_failed"]
    }
    self.assertEqual(by_type["RuntimeError"]["total_count"], 3)
    self.assertEqual(by_type["ValueError"]["total_count"], 2)
    no_stack = [
        group for group in groups
        if group["signature"]["extraction_failed"]
    ]
    self.assertEqual(len(no_stack), 2)
    self.assertTrue(all(group["total_count"] == 1 for group in no_stack))
    for group in groups:
      self.assertIsNotNone(group["representative"])
      self.assertIn("path", group["representative"])

    reopened = CrashStore(store_dir, clock=lambda: 1700.0)
    duplicate = reopened.ingest_file(artifacts[0], self.runner)
    self.assertTrue(duplicate["duplicate"])
    self.assertEqual(
        {group["id"]: group["total_count"] for group in reopened.groups()},
        {group["id"]: group["total_count"] for group in groups})

    clock[0] = 1600.0
    store.ingest_file(
        self._artifact("crash-a-4", b"new" + b"CRASH_A"), self.runner)
    window = reopened.groups(since=1200, until=1400)
    self.assertEqual(len(window), 2)
    window_by_type = {
        group["signature"]["crash_type"]: group
        for group in window
        if not group["signature"]["extraction_failed"]
    }
    self.assertEqual(window_by_type["RuntimeError"]["window_count"], 1)
    self.assertEqual(window_by_type["ValueError"]["window_count"], 1)

  def test_minimizes_with_reproduction_and_step_history(self):
    store = CrashStore(self.root / "min-store")
    store.ingest_file(
        self._artifact("small-a", b"w" * 10 + b"CRASH_A"), self.runner)
    artifact = self._artifact(
        "large-a", b"P" * 2000 + b"CRASH_A" + b"S" * 2000)
    ingested = store.ingest_file(artifact, self.runner)
    sample_id = ingested["sample_id"]

    minimizer = CrashMinimizer(
        store,
        self.runner,
        max_steps=1000,
        time_budget=30,
        run_timeout=5,
        verify_repetitions=2)
    result = minimizer.minimize(sample_id)

    self.assertLess(result.minimized_size, 32)
    self.assertLess(result.minimized_size, result.original_size)
    self.assertTrue(result.verified)
    self.assertTrue(result.record["steps"])
    for step in result.record["steps"]:
      self.assertIn("input_size", step)
      self.assertIn("status", step)
      self.assertIn("returncode", step)

    minimized = (store.root / result.minimized_path).read_bytes()
    replay = self.runner.run(minimized)
    self.assertEqual(replay.status, "crash")
    original = store.get_sample(sample_id)["outcome"]["signature"]["key"]
    self.assertEqual(replay.signature.key, original)

    persisted = store.get_sample(sample_id)["minimization"]
    self.assertEqual(persisted["original_size"], 4007)
    self.assertEqual(persisted["minimized_size"], result.minimized_size)
    self.assertGreater(len(persisted["steps"]), 0)
    group = store.groups(include_samples=True)
    representative = next(
        item["representative"] for item in group
        if item["id"] == result.group_id)
    self.assertTrue(representative["path"].startswith("minimized/"))

  def test_step_budget_keeps_best_result(self):
    store = CrashStore(self.root / "budget-store")
    artifact = self._artifact("budget-b", b"CRASH_B" + b"x" * 1000)
    sample_id = store.ingest_file(artifact, self.runner)["sample_id"]
    minimizer = CrashMinimizer(
        store,
        self.runner,
        max_steps=2,
        time_budget=10,
        run_timeout=5,
        verify_repetitions=2)
    result = minimizer.minimize(sample_id)

    self.assertEqual(result.stop_reason, "step_budget_exhausted")
    # The original reproduction is recorded as step 0 and is not charged to the
    # candidate execution budget.
    self.assertLessEqual(result.step_count, 3)
    self.assertIsNotNone(result.minimized_path)
    best = (store.root / result.minimized_path).read_bytes()
    replay = self.runner.run(best)
    self.assertEqual(replay.status, "crash")

  def test_target_timeout_rolls_back_and_records_reason(self):
    slow_target = self.root / "slow_target.py"
    slow_target.write_text(SLOW_ON_SMALL_INPUT_TARGET)
    slow_runner = TargetRunner(
        [sys.executable, "-S", str(slow_target), "{input}"],
        timeout=1)
    try:
      store = CrashStore(self.root / "timeout-store")
      artifact = self._artifact("slow-a", b"CRASH_A" + b"x" * 100)
      sample_id = store.ingest_file(artifact, slow_runner)["sample_id"]
      minimizer = CrashMinimizer(
          store,
          slow_runner,
          max_steps=100,
          time_budget=20,
          run_timeout=0.2,
          verify_repetitions=1)
      result = minimizer.minimize(sample_id)
      self.assertEqual(result.stop_reason, "target_timeout")
      self.assertIsNotNone(result.minimized_path)
      best = (store.root / result.minimized_path).read_bytes()
      replay = slow_runner.run(best)
      self.assertEqual(replay.status, "crash")
      timeout_steps = [
          step for step in result.record["steps"]
          if step["reason"] == "target_timeout"
      ]
      self.assertTrue(timeout_steps)
    finally:
      slow_runner.close()

  def test_store_state_is_valid_json_and_samples_are_complete(self):
    store = CrashStore(self.root / "atomic-store")
    artifact = self._artifact("atomic-a", b"atomic" + b"CRASH_A")
    result = store.ingest_file(artifact, self.runner)
    state = json.loads((store.root / "state.json").read_text())
    sample_path = store.root / state["groups"][result["group_id"]][
        "samples"][result["sample_id"]]["path"]
    self.assertEqual(sample_path.read_bytes(), artifact.read_bytes())
    self.assertFalse(list(store.root.rglob("*.tmp")))

  def test_cli_accepts_target_separator(self):
    store_dir = self.root / "cli-store"
    artifact = self._artifact("cli-a", b"CRASH_A")
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
      status = main([
          "triage",
          "--store", str(store_dir),
          "--artifacts", str(artifact),
          "--",
      ] + self.runner.command)
    self.assertEqual(status, 0)
    self.assertIn("processed=1", output.getvalue())

  def test_replays_real_atheris_artifact(self):
    fuzzer = self.root / "atheris_target.py"
    fuzzer.write_text(ATHERIS_TARGET)
    artifact = self._artifact(
        "atheris-crash", b"prefix" + b"x" * 100 + b"CRASH_A")
    runner = TargetRunner(
        [sys.executable, str(fuzzer), "{input}", "-atheris_runs=1"],
        timeout=10)
    try:
      store = CrashStore(self.root / "atheris-store")
      result = store.ingest_file(artifact, runner)
    finally:
      runner.close()
    signature = result["group"]["signature"]
    self.assertEqual(signature["crash_type"], "RuntimeError")
    self.assertFalse(signature["extraction_failed"])
    self.assertIn("TestOneInput", signature["frames"][-1])


if __name__ == "__main__":
  unittest.main()
