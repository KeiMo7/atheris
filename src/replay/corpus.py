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
"""Deterministic corpus loading and content-addressable de-duplication.

The original corpus directory is opened read-only.  Nothing here writes into
it.  De-duplication and any staging work happen on :class:`CorpusEntry`
metadata (plus, if requested, hardlinks/copies inside a caller-owned temp
directory), so the on-disk corpus layout never needs to be migrated.

Order semantics
---------------

Entries are ordered by their corpus-relative path using plain lexicographic
byte ordering.  That single rule fixes the replay order regardless of
filesystem enumeration order, worker count or machine.  Parallel replay
shards that fixed sequence by ordinal, so scheduling can never drift the
ordering and the final (order-independent) counts are reproducible anyway.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
from typing import List, Optional

# Corpus entries above this size are still loaded, but the loader records a
# warning so that accidentally pointed-at log files are visible.  Not a hard
# cap: hard constraints reject oversubscription of *workers*, not input size.
_LARGE_ENTRY_BYTES = 64 * 1024 * 1024


class CorpusLoadError(Exception):
  """Raised when the corpus directory itself cannot be read.

  A single bad entry never raises this; bad entries are returned on
  :class:`LoadedCorpus` as :attr:`LoadedCorpus.failures` so the batch can
  continue with the offending corpus item named explicitly.
  """


@dataclasses.dataclass(frozen=True)
class CorpusEntry:
  """One replayable corpus item.

  Attributes:
    ordinal: Zero-based position in the deterministic replay order.
    relative_path: Corpus-relative path using ``/`` separators; stable across
      machines and the human-readable identity used in error reports.
    absolute_path: Read-only location of the original file.
    content_sha256: Hex digest of the file content; the de-duplication key and
      the identity recorded in snapshots.
    size: File size in bytes.
    duplicate_of: Relative path of the first entry with identical content, or
      ``None`` for canonical entries.  Duplicates are skipped at replay time but
      remain visible here; de-duplication therefore cannot change coverage
      accounting, only redundant work.
  """

  ordinal: int
  relative_path: str
  absolute_path: str
  content_sha256: str
  size: int
  duplicate_of: Optional[str] = None

  def read(self) -> bytes:
    """Reads the entry content.  Never writes to the corpus directory."""
    with open(self.absolute_path, "rb") as f:
      return f.read()


@dataclasses.dataclass(frozen=True)
class CorpusFailure:
  """A corpus item that could not be loaded, identified by path."""

  relative_path: str
  absolute_path: str
  error_type: str
  error_message: str


@dataclasses.dataclass(frozen=True)
class LoadedCorpus:
  entries: List[CorpusEntry]
  failures: List[CorpusFailure]
  corpus_path: str

  @property
  def canonical_entries(self) -> List[CorpusEntry]:
    return [e for e in self.entries if e.duplicate_of is None]


def _iter_corpus_files(corpus_path: str):
  """Yields absolute file paths under corpus_path in a defined order."""
  collected = []
  for dirpath, dirnames, filenames in os.walk(corpus_path):
    # Sort in-place so os.walk descends deterministically too.
    dirnames.sort()
    for name in sorted(filenames):
      collected.append(os.path.join(dirpath, name))
  collected.sort()
  return collected


def load_corpus(corpus_path: str) -> LoadedCorpus:
  """Loads and de-duplicates a corpus directory without modifying it.

  Returns every readable entry (duplicates flagged, not deleted) plus a list
  of unreadable items.  Raises :class:`CorpusLoadError` only when the corpus
  directory itself is missing or is not a directory.
  """
  if not os.path.exists(corpus_path):
    raise CorpusLoadError(f"Corpus path does not exist: {corpus_path}")
  if not os.path.isdir(corpus_path):
    raise CorpusLoadError(f"Corpus path is not a directory: {corpus_path}")

  abs_corpus = os.path.abspath(corpus_path)
  entries: List[CorpusEntry] = []
  failures: List[CorpusFailure] = []
  # First content hash wins; this is what makes de-duplication deterministic.
  seen_hashes = {}

  for absolute in _iter_corpus_files(abs_corpus):
    relative = os.path.relpath(absolute, abs_corpus).replace(os.sep, "/")
    try:
      # Read-only open; the original corpus is never opened for writing.
      with open(absolute, "rb") as f:
        content = f.read()
    except OSError as exc:
      # Name the exact corpus item; include exception type and message only,
      # never file content (which may be sensitive).
      failures.append(
          CorpusFailure(
              relative_path=relative,
              absolute_path=absolute,
              error_type=type(exc).__name__,
              error_message=str(exc),
          )
      )
      continue

    digest = hashlib.sha256(content).hexdigest()
    duplicate_of = seen_hashes.get(digest)
    if duplicate_of is None:
      seen_hashes[digest] = relative
    entries.append(
        CorpusEntry(
            ordinal=len(entries),
            relative_path=relative,
            absolute_path=absolute,
            content_sha256=digest,
            size=len(content),
            duplicate_of=duplicate_of,
        )
    )

  return LoadedCorpus(
      entries=entries, failures=failures, corpus_path=abs_corpus
  )


def corpus_scope_hash(entries: List[CorpusEntry]) -> str:
  """Computes the canonical identity of a corpus *scope*.

  The scope is the ordered list of (relative path, content hash) pairs of all
  entries, including duplicates.  Two replays are only comparable when their
  scope hashes match -- this value is written into every snapshot.
  """
  hasher = hashlib.sha256()
  for entry in entries:
    hasher.update(entry.relative_path.encode("utf-8"))
    hasher.update(b"\x00")
    hasher.update(entry.content_sha256.encode("ascii"))
    hasher.update(b"\x00")
  return hasher.hexdigest()
