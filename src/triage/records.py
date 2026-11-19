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
"""In-memory record types shared across the triage pipeline.

Only metadata is retained in memory; file contents are streamed from disk on
demand. With 100k files the per-record footprint is a few hundred bytes, so
the whole metadata set comfortably fits in RAM while content memory stays
bounded by the hashing window and read chunk size.
"""

import dataclasses
from typing import List, Optional


@dataclasses.dataclass(frozen=True)
class Source:
  """One scanned directory (typically one fuzzing session's output)."""

  source_id: str  # Stable, order-derived identifier: "S0", "S1", ...
  root: str  # Absolute path as given by the user.
  kind: str  # "corpus" or "crashes".
  label: str  # Human-readable session label; defaults to directory basename.


@dataclasses.dataclass
class FileRecord:
  """One physical file discovered beneath a source root."""

  record_id: str  # "F0", "F1", ... assigned in deterministic scan order.
  source_id: str
  relative_path: str  # Forward-slash path relative to the source root.
  absolute_path: str
  size: int
  digest: str
  prefix_digest: str
  suffix_digest: str
  prefix_block: bytes
  suffix_block: bytes
  window_size: int
  read_error: Optional[str] = None

  @property
  def readable(self) -> bool:
    return self.read_error is None


@dataclasses.dataclass
class ContentNode:
  """One distinct byte sequence, grouping every physical copy of it."""

  digest: str
  size: int
  prefix_digest: str
  suffix_digest: str
  prefix_block: bytes
  suffix_block: bytes
  window_size: int
  # Physical files holding exactly these bytes. Sorted deterministically.
  records: List[FileRecord] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Family:
  """A similarity family of one or more content nodes.

  Exact duplicates are the same node. Nodes joined by prefix/suffix/single-byte
  expansion edges are united into one family. A family containing a single node
  whose content has a single physical copy is an unreduced singleton.
  """

  family_id: str
  node_digests: List[str]
  representative_digest: str
  # Human-understandable explanation of why this representative was chosen.
  representative_reason: str
  relation_edges: List[str]  # e.g. "A is a strict prefix of B"
