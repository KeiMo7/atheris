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
"""Command line entry point: ``python -m atheris.replay``.

This is an additive interface; the existing atheris fuzzing entry points and
their arguments are untouched.  It lets a team replay a fixed corpus, write a
comparable coverage snapshot, and diff two snapshots without re-importing the
corpus by hand.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from typing import Callable, Optional, Sequence

from .corpus import CorpusLoadError
from .engine import MAX_WORKERS_LIMIT
from .engine import ReplayConfig
from .engine import WorkerLimitError
from .engine import replay_corpus
from .coverage import CoverageUnavailableError
from .snapshot import SnapshotDiff
from .snapshot import compare_snapshots
from .snapshot import load_snapshot
from .snapshot import save_snapshot


def _load_target(spec: str) -> Callable[[bytes], None]:
  """Resolves ``module.path:callable`` into a callable."""
  if ":" not in spec:
    raise argparse.ArgumentTypeError(
        f"Target {spec!r} must be 'module.path:function_name'."
    )
  module_name, attr = spec.split(":", 1)
  module = importlib.import_module(module_name)
  target = getattr(module, attr, None)
  if not callable(target):
    raise argparse.ArgumentTypeError(
        f"{attr!r} in module {module_name!r} is not callable."
    )
  return target


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
      prog="python -m atheris.replay",
      description="Deterministic corpus replay with coverage snapshots.",
  )
  sub = parser.add_subparsers(dest="command", required=True)

  run = sub.add_parser("run", help="Replay a corpus and write a snapshot.")
  run.add_argument("--corpus", required=True, help="Corpus directory.")
  run.add_argument(
      "--target",
      required=True,
      type=_load_target,
      help="Target callable as module.path:function (takes bytes).",
  )
  run.add_argument(
      "--snapshot", required=True, help="Where to write the coverage snapshot."
  )
  run.add_argument(
      "--workers",
      type=int,
      default=1,
      help=f"Parallel workers (1..{MAX_WORKERS_LIMIT}).",
  )
  run.add_argument("--run-id", default=None, help="Explicit run identifier.")
  run.add_argument(
      "--checkpoint", default=None, help="Checkpoint file for resumable runs."
  )

  diff = sub.add_parser("diff", help="Compare two coverage snapshots.")
  diff.add_argument("first", help="First snapshot path.")
  diff.add_argument("second", help="Second snapshot path.")
  return parser


def _cmd_run(args: argparse.Namespace) -> int:
  config = ReplayConfig(
      target=args.target,
      max_workers=args.workers,
      run_id=args.run_id,
      checkpoint_path=args.checkpoint,
  )
  result = replay_corpus(args.corpus, config)
  from .corpus import corpus_scope_hash, load_corpus

  snapshot = result.to_snapshot(load_corpus(args.corpus))
  save_snapshot(snapshot, args.snapshot)

  print(f"run_id: {result.run_id}")
  print(f"collector: {result.collector_backend}")
  print(f"resumed: {result.resumed}")
  print(f"corpus_scope_hash: {result.corpus_scope_hash}")
  print(f"entries replayed: {len(result.results)}")
  print(f"duplicates skipped: {len(result.skipped_duplicates)}")
  print(f"entry failures: {len(result.entry_failures)}")
  for failure in result.entry_failures:
    print(
        f"  FAIL {failure.relative_path}: {failure.error_type}:"
        f" {failure.error_message}"
    )
  for failure in result.corpus_failures:
    print(
        f"  UNREADABLE {failure.relative_path}: {failure.error_type}:"
        f" {failure.error_message}"
    )
  print(f"total edges: {result.coverage.total_edges}")
  print(f"total hits: {result.coverage.total_hits}")
  print(f"snapshot: {args.snapshot}")
  return 0


def _cmd_diff(args: argparse.Namespace) -> int:
  first = load_snapshot(args.first)
  second = load_snapshot(args.second)
  diff: SnapshotDiff = compare_snapshots(first, second)
  if not diff.comparable:
    print(f"NOT COMPARABLE: {diff.reason}")
    return 2
  for edge in diff.only_in_first:
    print(f"only in first:  {edge}")
  for edge in diff.only_in_second:
    print(f"only in second: {edge}")
  for edge, left, right in diff.count_changed:
    print(f"count changed:  {edge}: {left} -> {right}")
  if diff.identical:
    print("IDENTICAL: coverage snapshots match exactly.")
    return 0
  print("DIFFERENT: snapshots cover the same corpus scope but disagree.")
  return 1


def main(argv: Optional[Sequence[str]] = None) -> int:
  parser = _build_parser()
  args = parser.parse_args(argv)
  try:
    if args.command == "run":
      return _cmd_run(args)
    return _cmd_diff(args)
  except (
      CorpusLoadError,
      WorkerLimitError,
      CoverageUnavailableError,
  ) as exc:
    # Provenance/error text only; never input content.
    print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
    return 3


if __name__ == "__main__":
  sys.exit(main())
