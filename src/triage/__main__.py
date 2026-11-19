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
"""Command line entry point: ``python -m atheris.triage``.

Nothing here is invoked by a normal fuzzing run. The triage feature is opt-in
and lives entirely behind this module, preserving every existing entry-point
option and output format.
"""

import argparse
import importlib
import json
import os
import sys
from typing import List, Optional

from . import replay, store

# `triage.organize` is both a submodule and a function re-exported by the
# package __init__ (the function rebinding shadows the submodule attribute).
# Load the submodule explicitly so its TriageOptions/TriageResult are
# reachable regardless of whether the package is imported as `atheris.triage`
# or run directly as `python -m triage`.
organize_mod = importlib.import_module(__package__ + ".organize")


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
      prog="python -m atheris.triage",
      description=("Organize fuzzer corpus and crash artifacts into a "
                   "reviewable, reproducible minimal set."))
  sub = parser.add_subparsers(dest="command", required=True)

  p_org = sub.add_parser("organize", help="triage corpus/crash directories")
  p_org.add_argument(
      "--corpus", action="append", default=[], metavar="DIR",
      help="corpus directory; repeatable for cross-session overlays")
  p_org.add_argument(
      "--crashes", action="append", default=[], metavar="DIR",
      help="crash directory; repeatable")
  p_org.add_argument("--output", required=True, metavar="DIR",
                     help="destination for the minimal set (never an input)")
  p_org.add_argument("--overwrite", action="store_true",
                     help="replace an existing destination atomically")
  p_org.add_argument(
      "--harness", default=None, metavar="FILE",
      help=("fuzz harness exposing the entry point; when given, the minimal "
            "corpus is made coverage-equivalent by replaying all inputs"))
  p_org.add_argument("--entry", default="TestOneInput",
                     help="harness entry-point callable name")
  p_org.add_argument("--backend", choices=("trace", "native"),
                     default="trace", help="coverage backend for the harness")
  p_org.add_argument("--coverage-root", action="append", default=[],
                     help="restrict trace units to files beneath these dirs")
  p_org.add_argument("--json", action="store_true",
                     help="emit the manifest summary as JSON")

  p_cov = sub.add_parser(
      "verify-coverage",
      help="replay original vs minimal inputs and compare coverage units")
  p_org_group = p_cov.add_mutually_exclusive_group(required=True)
  p_org_group.add_argument("--original-dir", action="append", default=[],
                           metavar="DIR")
  p_org_group.add_argument("--from-output", metavar="DIR",
                           help="an existing triage output directory")
  p_cov.add_argument("--minimal-dir", action="append", default=[],
                     metavar="DIR")
  p_cov.add_argument("--harness", required=True,
                     help="Python file exposing the fuzz entry point")
  p_cov.add_argument("--entry", default=replay.HARNESS_ENTRY_DEFAULT)
  p_cov.add_argument("--backend", choices=("trace", "native"),
                     default="trace")
  p_cov.add_argument("--coverage-root", action="append", default=[],
                     help="restrict trace units to files beneath these dirs")

  p_res = sub.add_parser("restore", help="reconstruct original inputs")
  p_res.add_argument("--output", required=True,
                     help="triage output directory containing the manifest")
  p_res.add_argument("--restore-to", required=True,
                     help="directory to reconstruct the original tree into")

  return parser


def _run_organize(args) -> int:
  if not args.corpus and not args.crashes:
    print("error: provide at least one --corpus or --crashes directory",
          file=sys.stderr)
    return 2
  options = organize_mod.TriageOptions(
      corpus_roots=args.corpus,
      crash_roots=args.crashes,
      destination=args.output,
      overwrite=args.overwrite,
      harness_path=args.harness,
      harness_entry=args.entry,
      coverage_backend=args.backend,
      coverage_roots=args.coverage_root or None,
  )
  try:
    result = organize_mod.organize(options)
  except store.ConcurrentTriageError as e:
    print("error: %s" % e, file=sys.stderr)
    return 3
  except store.TriageError as e:
    print("error: %s" % e, file=sys.stderr)
    return 2

  summary = {
      "destination": result.destination,
      "stats": result.manifest["stats"],
      "coverage_verification": result.manifest.get("coverage_verification"),
      "warnings": result.warnings,
      "failures": result.manifest["failures"],
  }
  if args.json:
    print(json.dumps(summary, indent=2, sort_keys=True))
  else:
    _print_human_summary(result.manifest)
    cov = result.manifest.get("coverage_verification")
    if cov is not None:
      print("  coverage equivalent:       %s (%d -> %d units, %d inputs)" %
            (cov["equivalent"], cov["units_original"], cov["units_minimal"],
             cov["contents_selected_for_coverage"]))
  if result.exit_code != 0:
    print("warning: %d file(s) could not be read; see manifest failures" %
          result.manifest["stats"]["read_failures"], file=sys.stderr)
  return result.exit_code


def _print_human_summary(manifest: dict) -> None:
  stats = manifest["stats"]
  print("Triage complete.")
  print("  physical files scanned:     %d" % stats["physical_files_seen"])
  print("  distinct contents:          %d" % stats["distinct_contents"])
  print("  exact duplicates collapsed: %d" %
        stats["exact_duplicate_copies_collapsed"])
  print("  similarity families:        %d (%d with merges)" %
        (stats["similarity_families"], stats["families_with_merges"]))
  print("  minimal corpus size:        %d" % stats["minimal_corpus_size"])
  print("  crash inputs / groups:      %d / %d" %
        (stats["crash_inputs"], stats["crash_groups"]))
  if stats["read_failures"]:
    print("  read failures:              %d (listed in manifest)" %
          stats["read_failures"])


def _run_verify_coverage(args) -> int:
  if args.from_output:
    manifest = store.load_manifest(args.from_output)
    if manifest is None:
      print("error: no manifest in %s" % args.from_output, file=sys.stderr)
      return 2
    original_paths: List[str] = []
    for source in manifest["sources"]:
      if source["kind"] != "corpus":
        continue
      for family in manifest["corpus_families"]:
        for member in family["members"]:
          for copy in member["physical_copies"]:
            if copy["source_id"] == source["source_id"]:
              original_paths.append(
                  os.path.join(source["root"],
                               copy["path"].replace("/", os.sep)))
    minimal_paths = [
        os.path.join(args.from_output, f["representative_path"])
        for f in manifest["corpus_families"]
    ]
  else:
    original_paths = _expand_dirs(args.original_dir)
    minimal_paths = _expand_dirs(args.minimal_dir)

  report = replay.verify_equivalence(
      original_paths=original_paths,
      minimal_paths=minimal_paths,
      harness_path=args.harness,
      entry_name=args.entry,
      backend=args.backend,
      coverage_roots=args.coverage_root or None,
  )
  print(json.dumps(report, indent=2, sort_keys=True))
  return 0 if report["equivalent"] else 1


def _expand_dirs(dirs: List[str]) -> List[str]:
  paths = []
  for root in dirs:
    for dirpath, _, filenames in os.walk(root):
      for name in sorted(filenames):
        paths.append(os.path.join(dirpath, name))
  return sorted(paths)


def _run_restore(args) -> int:
  try:
    path = organize_mod.restore(args.output, args.restore_to)
  except store.TriageError as e:
    print("error: %s" % e, file=sys.stderr)
    return 2
  print("Restored original inputs under %s" % path)
  return 0


def main(argv: Optional[List[str]] = None) -> int:
  parser = _build_parser()
  args = parser.parse_args(argv)
  if args.command == "organize":
    return _run_organize(args)
  if args.command == "verify-coverage":
    return _run_verify_coverage(args)
  if args.command == "restore":
    return _run_restore(args)
  parser.error("unknown command: %s" % args.command)
  return 2


if __name__ == "__main__":
  sys.exit(main())
