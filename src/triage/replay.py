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
"""Full replay and coverage-equivalence verification.

Hard constraint of triage: replaying the *minimal* set must reach exactly the
same coverage units as replaying every original input. This module replays a
set of byte inputs against a user-provided harness and collects the set of
coverage units hit.

Two backends:
  * ``trace`` (default, always available): CPython line coverage via
    sys.settrace. Units are (source filename, line number) pairs. Pure Python,
    needs no native build, and is deterministic for a fixed input set.
  * ``native``: Atheris' instrumented 8-bit counters, when a real build is
    available. Units are counter indices observed during the replay.

The harness is a Python file exposing a ``TestOneInput(data: bytes)`` callable
(or another name given explicitly). To make ordinary fuzz harnesses importable
outside a fuzzing run, ``atheris.Setup``/``Fuzz`` calls are neutralized with an
in-memory stub when the native extension is not importable.
"""

import importlib.util
import os
import sys
import types
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple

from .hashing import hash_bytes

HARNESS_ENTRY_DEFAULT = "TestOneInput"


class ReplayError(Exception):
  pass


def _install_atheris_stub() -> bool:
  """Neutralize atheris.Setup/Fuzz for harness import without a native build.

  Returns True when a stub was installed. A real atheris package is preferred
  whenever it imports successfully.
  """
  try:
    import atheris
    if getattr(atheris, "native_extension_available", lambda: True)():
      return False
  except ImportError:
    pass

  stub = types.ModuleType("atheris")
  stub.Setup = lambda *args, **kwargs: None
  stub.Fuzz = lambda *args, **kwargs: None
  stub.enabled_hooks = set()
  stub.ALL_REMAINING = None

  def _identity(func):
    return func

  stub.instrument_func = _identity
  stub.instrument_all = _identity

  class _InstrumentImports:

    def __enter__(self):
      return self

    def __exit__(self, *args):
      return False

  stub.instrument_imports = _InstrumentImports
  sys.modules["atheris"] = stub
  return True


def load_harness(harness_path: str,
                 entry_name: str = HARNESS_ENTRY_DEFAULT) -> Callable[[bytes],
                                                                      None]:
  """Import a harness file and return its entry-point callable."""
  if not os.path.isfile(harness_path):
    raise ReplayError("harness not found: %s" % harness_path)
  _install_atheris_stub()

  module_name = "atheris_triage_harness"
  spec = importlib.util.spec_from_file_location(module_name, harness_path)
  if spec is None or spec.loader is None:
    raise ReplayError("cannot load harness: %s" % harness_path)
  module = importlib.util.module_from_spec(spec)
  sys.modules[module_name] = module
  try:
    spec.loader.exec_module(module)  # type: ignore[union-attr]
  except SystemExit:
    # Some harnesses call sys.exit after Fuzz(); import time must survive it.
    pass
  entry = getattr(module, entry_name, None)
  if not callable(entry):
    raise ReplayError(
        "harness %s does not expose a callable %s()" %
        (harness_path, entry_name))
  return entry


def _read_input(path: str) -> bytes:
  with open(path, "rb") as f:
    return f.read()


class LineCoverageCollector:
  """sys.settrace-based line coverage unit collector."""

  def __init__(self):
    self.units: Set[Tuple[str, int]] = set()
    self._root_dirs: Set[Optional[str]] = None

  def scope_to(self, root_dirs: Iterable[Optional[str]]) -> None:
    if root_dirs:
      self._root_dirs = {os.path.abspath(p) for p in root_dirs}

  def _trace(self, frame, event, arg):
    if event == "call":
      return self._trace
    if event == "line":
      filename = os.path.abspath(frame.f_code.co_filename)
      if self._root_dirs is None or any(
          filename.startswith(root + os.sep) for root in self._root_dirs):
        self.units.add((filename, frame.f_lineno))
    return None

  def __enter__(self):
    sys.settrace(self._trace)
    return self

  def __exit__(self, *args):
    sys.settrace(None)


class NativeCoverageCollector:
  """Coverage via Atheris' native 8-bit counters when a build exists."""

  def __init__(self):
    self.units: Set[int] = set()
    self._mock = None

  def __enter__(self):
    from atheris.mock_libfuzzer import mockutils  # type: ignore
    self._mock = mockutils
    return self

  def __exit__(self, *args):
    self._mock = None

  def snapshot(self) -> None:
    if self._mock is not None:
      counters = self._mock.get_8bit_counters()
      self.units.update(i for i, v in enumerate(counters) if v != 0)


def replay_inputs(
    input_paths: Iterable[str],
    harness_path: str,
    entry_name: str = HARNESS_ENTRY_DEFAULT,
    backend: str = "trace",
    coverage_roots: Optional[List[str]] = None,
) -> Tuple[set, dict]:
  """Replay every input once; return (coverage_unit_set, stats)."""
  entry = load_harness(harness_path, entry_name)
  paths = list(input_paths)
  stats = {
      "inputs_replayed": 0,
      "input_errors": 0,
      "digests": [],
  }

  if backend == "native":
    collector = NativeCoverageCollector()
  elif backend == "trace":
    collector = LineCoverageCollector()
    collector.scope_to(coverage_roots)
  else:
    raise ReplayError("unknown coverage backend: %s" % backend)

  with collector:
    for path in paths:
      try:
        data = _read_input(path)
      except OSError:
        stats["input_errors"] += 1
        continue
      try:
        entry(data)
      except Exception:  # pylint: disable=broad-except  # noqa: S110
        # A crash on replay is an expected outcome for crash inputs; coverage
        # accumulated up to the exception is what matters. Swallowing it here
        # is intentional, not a silently ignored error.
        pass
      stats["inputs_replayed"] += 1
      stats["digests"].append(hash_bytes(data))
      if backend == "native":
        collector.snapshot()

  return set(collector.units), stats


def verify_equivalence(original_paths: Iterable[str],
                       minimal_paths: Iterable[str],
                       harness_path: str,
                       entry_name: str = HARNESS_ENTRY_DEFAULT,
                       backend: str = "trace",
                       coverage_roots: Optional[List[str]] = None) -> dict:
  """Replay both sets and assert the coverage-unit sets are exactly equal."""
  before, stats_before = replay_inputs(
      original_paths, harness_path, entry_name, backend, coverage_roots)
  after, stats_after = replay_inputs(
      minimal_paths, harness_path, entry_name, backend, coverage_roots)
  missing = sorted(before - after)
  extra = sorted(after - before)
  return {
      "equivalent": before == after,
      "backend": backend,
      "units_before": len(before),
      "units_after": len(after),
      "units_missing_in_minimal": missing,
      "units_new_in_minimal": extra,
      "replayed_before": stats_before["inputs_replayed"],
      "replayed_after": stats_after["inputs_replayed"],
      "read_errors_before": stats_before["input_errors"],
      "read_errors_after": stats_after["input_errors"],
  }


def collect_per_input_coverage(
    labeled_paths: List[Tuple[str, str]],
    harness_path: str,
    entry_name: str = HARNESS_ENTRY_DEFAULT,
    backend: str = "trace",
    coverage_roots: Optional[List[str]] = None
) -> Tuple[Dict[str, frozenset], frozenset, dict]:
  """Replay each labeled input and record the coverage units it reaches.

  Args:
    labeled_paths: list of (label, path) pairs. The label (e.g. a content
      digest) is the key of the returned coverage map.

  Returns:
    (coverage_by_label, all_units, stats).
  """
  entry = load_harness(harness_path, entry_name)
  coverage: Dict[str, frozenset] = {}
  all_units: set = set()
  errors = 0

  for label, path in labeled_paths:
    if backend == "native":
      collector = NativeCoverageCollector()
    else:
      collector = LineCoverageCollector()
      collector.scope_to(coverage_roots)
    try:
      data = _read_input(path)
    except OSError:
      errors += 1
      coverage[label] = frozenset()
      continue
    with collector:
      try:
        entry(data)
      except Exception:  # pylint: disable=broad-except  # noqa: S110
        # Coverage accumulated up to an expected crash still counts.
        pass
      if backend == "native":
        collector.snapshot()
    units = frozenset(collector.units)
    coverage[label] = units
    all_units.update(units)

  stats = {"inputs_replayed": len(labeled_paths), "input_errors": errors}
  return coverage, frozenset(all_units), stats


def coverage_preserving_selection(
    coverage_by_label: Dict[str, frozenset],
    target_units: frozenset,
    preferred_order: Optional[List[str]] = None,
) -> List[str]:
  """Greedily select the fewest labels whose coverage union equals the target.

  This is the classic greedy set-cover approximation. Labels are considered in
  ``preferred_order`` when several add the same number of new units, so family
  representatives are kept whenever an equivalent choice exists. The result is
  deterministic for a fixed input mapping and ordering.

  Returns the selected labels; when target coverage cannot be reached (some
  unit only appears on an unreadable input), every non-empty label is returned
  and the gap is left for the caller's equivalence check to report.
  """
  preferred_rank = {
      label: rank for rank, label in enumerate(preferred_order or [])
  }
  remaining = set(target_units)
  chosen: List[str] = []
  available = set(coverage_by_label)

  def rank(label: str) -> int:
    return preferred_rank.get(label, len(preferred_rank) + 1)

  while remaining:
    best = None
    best_gain = 0
    for label in available:
      gain = len(remaining & coverage_by_label[label])
      if gain > best_gain or (
          gain == best_gain and gain > 0 and
          (best is None or (rank(label), label) < (rank(best), best))):
        best = label
        best_gain = gain
    if best is None or best_gain == 0:
      # No remaining label covers any outstanding unit; include every label
      # with coverage so the caller can still run a full equivalence check.
      for label in sorted(available):
        if coverage_by_label[label]:
          chosen.append(label)
      break
    chosen.append(best)
    available.discard(best)
    remaining.difference_update(coverage_by_label[best])

  return chosen
