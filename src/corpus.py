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
"""Portable, corruption-tolerant corpus archives.

The archive is UTF-8 JSON Lines.  The first line is a format header and every
following line contains one independent corpus entry.  A damaged line can
therefore be skipped without preventing the rest of the corpus from loading.
"""

from __future__ import annotations

import argparse
import base64
import binascii
from dataclasses import dataclass, field
import json
import os
from pathlib import Path, PurePosixPath
import sys
from typing import Iterable, Sequence

from .artifact_io import atomic_write_bytes
from .artifact_io import atomic_write_text
from .artifact_io import sha256_bytes

FORMAT_NAME = "atheris-corpus-jsonl"
FORMAT_VERSION = 1


@dataclass
class ImportReport:
  imported: int = 0
  skipped: int = 0
  errors: list[str] = field(default_factory=list)


@dataclass
class ExportReport:
  exported: int = 0
  skipped: int = 0
  errors: list[str] = field(default_factory=list)


def _iter_corpus_files(paths: Iterable[os.PathLike[str] | str]):
  for raw_path in paths:
    path = Path(raw_path)
    if path.is_file():
      yield path, path.name
      continue
    if not path.is_dir():
      raise FileNotFoundError(path)
    for child in sorted(path.rglob("*")):
      if child.is_file() and not child.name.startswith("."):
        yield child, child.relative_to(path).as_posix()


def export_corpus(paths: Iterable[os.PathLike[str] | str],
                  output: os.PathLike[str] | str) -> ExportReport:
  """Exports one or more corpus files/directories to a portable archive."""
  report = ExportReport()
  lines = [json.dumps({
      "format": FORMAT_NAME,
      "version": FORMAT_VERSION,
  }, sort_keys=True)]

  for source, relative_name in _iter_corpus_files(paths):
    try:
      data = source.read_bytes()
    except OSError as e:
      report.skipped += 1
      report.errors.append(f"{source}: {e}")
      continue

    entry_name = PurePosixPath(relative_name)
    if entry_name.is_absolute() or ".." in entry_name.parts:
      report.skipped += 1
      report.errors.append(f"{source}: unsafe corpus path")
      continue

    lines.append(json.dumps({
        "name": entry_name.as_posix(),
        "size": len(data),
        "sha256": sha256_bytes(data),
        "data": base64.b64encode(data).decode("ascii"),
    }, sort_keys=True))
    report.exported += 1

  atomic_write_text(output, "\n".join(lines) + "\n")
  return report


def _safe_destination(root: Path, name: str) -> Path | None:
  entry_name = PurePosixPath(name)
  if not name or entry_name.is_absolute() or ".." in entry_name.parts:
    return None
  destination = root.joinpath(*entry_name.parts)
  try:
    destination.relative_to(root)
  except ValueError:
    return None
  return destination


def import_corpus(archive: os.PathLike[str] | str,
                  output: os.PathLike[str] | str,
                  *,
                  overwrite: bool = True) -> ImportReport:
  """Imports a corpus archive, skipping only damaged entries."""
  report = ImportReport()
  root = Path(output)
  root.mkdir(parents=True, exist_ok=True)

  with open(archive, "r", encoding="utf-8") as f:
    header_line = f.readline()
  try:
    header = json.loads(header_line)
  except json.JSONDecodeError:
    header = None
  if not isinstance(header, dict) or header.get("format") != FORMAT_NAME:
    raise ValueError("not an Atheris corpus archive")
  if header.get("version") != FORMAT_VERSION:
    raise ValueError(
        f"unsupported corpus archive version: {header.get('version')!r}")

  with open(archive, "r", encoding="utf-8", errors="replace") as f:
    next(f, None)
    for line_number, line in enumerate(f, start=2):
      if not line.strip():
        continue
      try:
        entry = json.loads(line)
        if not isinstance(entry, dict):
          raise ValueError("entry is not an object")
        name = entry["name"]
        expected_size = entry["size"]
        expected_sha256 = entry["sha256"]
        encoded = entry["data"]
        if not isinstance(name, str):
          raise ValueError("entry name is not a string")
        if not isinstance(expected_size, int) or expected_size < 0:
          raise ValueError("invalid entry size")
        if not isinstance(expected_sha256, str):
          raise ValueError("invalid entry checksum")
        if not isinstance(encoded, str):
          raise ValueError("invalid entry data")

        data = base64.b64decode(encoded, validate=True)
        if len(data) != expected_size:
          raise ValueError(
              f"size mismatch: expected {expected_size}, got {len(data)}")
        actual_sha256 = sha256_bytes(data)
        if actual_sha256 != expected_sha256:
          raise ValueError("sha256 mismatch")

        destination = _safe_destination(root, name)
        if destination is None:
          raise ValueError("unsafe corpus path")
        if destination.exists() and not overwrite:
          raise ValueError("destination already exists")
        atomic_write_bytes(destination, data)
        report.imported += 1
      except (binascii.Error, KeyError, OSError, TypeError, ValueError) as e:
        report.skipped += 1
        report.errors.append(f"line {line_number}: {e}")

  return report


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
      prog="python -m atheris.corpus",
      description="Export and import corruption-tolerant Atheris corpora.")
  subparsers = parser.add_subparsers(dest="command", required=True)

  export_parser = subparsers.add_parser(
      "export", help="export corpus files or directories")
  export_parser.add_argument("--output", required=True)
  export_parser.add_argument("inputs", nargs="+")

  import_parser = subparsers.add_parser(
      "import", help="import a corpus archive")
  import_parser.add_argument("--archive", required=True)
  import_parser.add_argument("--output", required=True)
  import_parser.add_argument(
      "--no-overwrite", action="store_true",
      help="skip entries whose destination already exists")
  return parser


def main(argv: Sequence[str] | None = None) -> int:
  args = _build_parser().parse_args(argv)
  if args.command == "export":
    report = export_corpus(args.inputs, args.output)
    print(f"exported={report.exported} skipped={report.skipped}")
  else:
    report = import_corpus(
        args.archive, args.output, overwrite=not args.no_overwrite)
    print(f"imported={report.imported} skipped={report.skipped}")
    for error in report.errors:
      print(f"warning: {error}", file=sys.stderr)
  return 0


if __name__ == "__main__":
  sys.exit(main())
