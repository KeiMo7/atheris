# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Versioned coverage snapshots for long-term storage and direct diffing.

Snapshot format
---------------

A snapshot is a UTF-8 JSON document with a top-level ``format_version``.
Reading a snapshot whose *major* version is newer than what this code
understands raises :class:`SnapshotFormatVersionVersionError` -- semantics are
never silently reinterpreted.  Within a major version, unknown extra fields
are tolerated (forward-compatible minor bumps).

Every snapshot carries provenance answering "are these two snapshots
comparable?":

* ``semantics_version`` -- the counting rule version from
  :mod:`atheris.replay.coverage`;
* ``corpus_scope_hash`` -- identity of the corpus range the snapshot covers;
* ``corpus_entry_count`` / ``corpus_canonical_count`` -- the human-readable
  range description;
* ``run_id`` -- the specific replay run that produced this snapshot.

Writes are atomic (temp file in the same directory, then ``os.replace``) so a
crash can never leave a half-written snapshot for the next run to read.
"""

from __future__ import annotations

import dataclasses
import json
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from .coverage import COVERAGE_SEMANTICS_VERSION
from .coverage import CoverageMap

SNAPSHOT_FORMAT_VERSION = "1.0"
_SNAPSHOT_FORMAT_MAJOR = 1


class SnapshotFormatVersionError(ValueError):
  """Raised when a snapshot uses a newer, unsupported format version."""


@dataclasses.dataclass(frozen=True)
class Snapshot:
  """An exported coverage snapshot with full provenance."""

  run_id: str
  corpus_scope_hash: str
  corpus_entry_count: int
  corpus_canonical_count: int
  corpus_replayed_count: int
  coverage: CoverageMap
  semantics_version: str = COVERAGE_SEMANTICS_VERSION
  format_version: str = SNAPSHOT_FORMAT_VERSION
  extra: Dict[str, Any] = dataclasses.field(default_factory=dict)

  def to_dict(self) -> Dict[str, Any]:
    # Edge indices are assigned from the sorted edge ids, so a dense index
    # means the same edge on every machine.  The canonical edge string is kept
    # alongside, making the file self-describing and independently diffable.
    edges = []
    for index, (edge, count) in enumerate(self.coverage.ordered_items()):
      edges.append({"index": index, "edge": edge, "count": count})
    payload = {
        "format_version": self.format_version,
        "semantics_version": self.semantics_version,
        "run_id": self.run_id,
        "corpus_scope_hash": self.corpus_scope_hash,
        "corpus_entry_count": self.corpus_entry_count,
        "corpus_canonical_count": self.corpus_canonical_count,
        "corpus_replayed_count": self.corpus_replayed_count,
        "total_edges": self.coverage.total_edges,
        "total_hits": self.coverage.total_hits,
        "edges": edges,
    }
    if self.extra:
      payload["extra"] = self.extra
    return payload

  def to_json(self) -> str:
    return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


@dataclasses.dataclass(frozen=True)
class SnapshotDiff:
  """Result of comparing two snapshots.

  ``comparable`` is False when provenance shows the snapshots describe
  different corpus scopes or different counting semantics; in that case
  ``reason`` explains why and edge-level diffing is refused.
  """

  comparable: bool
  reason: Optional[str]
  only_in_first: List[str]
  only_in_second: List[str]
  count_changed: List[Tuple[str, int, int]]

  @property
  def identical(self) -> bool:
    return (
        self.comparable
        and not self.only_in_first
        and not self.only_in_second
        and not self.count_changed
    )


def _parse_format_version(version: str) -> int:
  try:
    return int(str(version).split(".", 1)[0])
  except (ValueError, IndexError) as exc:
    raise SnapshotFormatVersionError(
        f"Malformed snapshot format_version: {version!r}"
    ) from exc


def load_snapshot(path: str) -> Snapshot:
  """Loads a snapshot, refusing (not reinterpreting) unknown new versions."""
  with open(path, "r", encoding="utf-8") as f:
    payload = json.load(f)

  format_version = payload.get("format_version")
  if format_version is None:
    raise SnapshotFormatVersionError(
        "Snapshot has no format_version; refusing to guess its semantics."
    )
  major = _parse_format_version(format_version)
  if major > _SNAPSHOT_FORMAT_MAJOR:
    raise SnapshotFormatVersionError(
        f"Snapshot format_version {format_version} is newer than this code"
        f" supports ({SNAPSHOT_FORMAT_VERSION}); refusing to read it under"
        " potentially different semantics."
    )

  counts: Dict[str, int] = {}
  for item in payload.get("edges", []):
    counts[item["edge"]] = int(item["count"])

  return Snapshot(
      run_id=payload["run_id"],
      corpus_scope_hash=payload["corpus_scope_hash"],
      corpus_entry_count=int(payload["corpus_entry_count"]),
      corpus_canonical_count=int(payload["corpus_canonical_count"]),
      corpus_replayed_count=int(payload["corpus_replayed_count"]),
      coverage=CoverageMap(counts),
      semantics_version=payload.get(
          "semantics_version", "unknown"
      ),
      format_version=str(format_version),
      extra=payload.get("extra", {}),
  )


def save_snapshot(snapshot: Snapshot, path: str) -> None:
  """Atomically writes ``snapshot`` to ``path`` (no half-open files)."""
  directory = os.path.dirname(os.path.abspath(path)) or "."
  os.makedirs(directory, exist_ok=True)
  # NamedTemporaryFile in the same directory so os.replace is atomic on the
  # same filesystem; cleaned up explicitly on every failure path.
  tmp = tempfile.NamedTemporaryFile(
      mode="w",
      encoding="utf-8",
      dir=directory,
      prefix=".atheris-snapshot-",
      suffix=".tmp",
      delete=False,
  )
  tmp_path = tmp.name
  try:
    tmp.write(snapshot.to_json())
    tmp.flush()
    os.fsync(tmp.fileno())
    tmp.close()
    os.replace(tmp_path, path)
  except BaseException:
    tmp.close()
    try:
      os.unlink(tmp_path)
    except OSError:
      pass
    raise


def compare_snapshots(first: Snapshot, second: Snapshot) -> SnapshotDiff:
  """Diffs two snapshots, first checking that they are comparable."""
  if first.semantics_version != second.semantics_version:
    return SnapshotDiff(
        comparable=False,
        reason=(
            "Counting semantics differ: "
            f"{first.semantics_version!r} vs {second.semantics_version!r}."
        ),
        only_in_first=[],
        only_in_second=[],
        count_changed=[],
    )
  if first.corpus_scope_hash != second.corpus_scope_hash:
    return SnapshotDiff(
        comparable=False,
        reason=(
            "Corpus scope differs: "
            f"{first.corpus_scope_hash[:12]!r} vs "
            f"{second.corpus_scope_hash[:12]!r}; the snapshots do not cover"
            " the same corpus range."
        ),
        only_in_first=[],
        only_in_second=[],
        count_changed=[],
    )

  first_counts = first.coverage.counts
  second_counts = second.coverage.counts
  first_edges = set(first_counts)
  second_edges = set(second_counts)
  only_first = sorted(first_edges - second_edges)
  only_second = sorted(second_edges - first_edges)
  changed = []
  for edge in sorted(first_edges & second_edges):
    if first_counts[edge] != second_counts[edge]:
      changed.append((edge, first_counts[edge], second_counts[edge]))
  return SnapshotDiff(
      comparable=True,
      reason=None,
      only_in_first=only_first,
      only_in_second=only_second,
      count_changed=changed,
  )
