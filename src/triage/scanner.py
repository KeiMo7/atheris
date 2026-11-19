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
"""Deterministic one-pass scanning of corpus and crash directories.

Scanning order is fully sorted (relative path within source, sources in the
order given), which makes record ids, family ids and every downstream output
reproducible across machines. Each file is opened once: its content is hashed
in a single streaming pass and the handle is released immediately, so total
memory is independent of corpus size and individual file size.
"""

import os
from typing import List, Optional, Tuple

from .hashing import DEFAULT_CHUNK_SIZE, DEFAULT_WINDOW_SIZE, fingerprint_file
from .records import ContentNode, FileRecord, Source


def list_files(root: str) -> List[str]:
  """List regular files beneath root as deterministic forward-slash relpaths."""
  found = []
  for dirpath, dirnames, filenames in os.walk(root):
    # Sort in place so os.walk's traversal order is deterministic too.
    dirnames.sort()
    for name in sorted(filenames):
      absolute = os.path.join(dirpath, name)
      # Symlinks to regular files are followed; broken symlinks surface as
      # read errors rather than silently disappearing.
      if os.path.isdir(absolute):
        continue
      rel = os.path.relpath(absolute, root)
      found.append(rel.replace(os.sep, "/"))
  return found


def make_sources(corpus_roots: List[str],
                 crash_roots: Optional[List[str]] = None) -> List[Source]:
  """Build ordered Source objects from user-provided directories."""
  sources = []
  idx = 0
  for root in corpus_roots:
    sources.append(
        Source(
            source_id="S%d" % idx,
            root=os.path.abspath(root),
            kind="corpus",
            label=os.path.basename(os.path.abspath(root.rstrip("/"))) or root,
        ))
    idx += 1
  for root in (crash_roots or []):
    sources.append(
        Source(
            source_id="S%d" % idx,
            root=os.path.abspath(root),
            kind="crashes",
            label=os.path.basename(os.path.abspath(root.rstrip("/"))) or root,
        ))
    idx += 1
  return sources


def scan(sources: List[Source],
         chunk_size: int = DEFAULT_CHUNK_SIZE,
         window_size: int = DEFAULT_WINDOW_SIZE
         ) -> Tuple[List[FileRecord], List[ContentNode], List[dict]]:
  """Scan every source once.

  Returns:
    records: every physical file, including ones that failed to read (their
      read_error field is populated and their digest is empty).
    nodes: distinct readable contents, each holding all physical copies.
    failures: structured failure entries for the final report.
  """
  records: List[FileRecord] = []
  failures: List[dict] = []
  nodes_by_digest = {}
  record_idx = 0

  for source in sources:
    if not os.path.isdir(source.root):
      failures.append({
          "source_id": source.source_id,
          "path": source.root,
          "error": "source directory does not exist or is not a directory",
      })
      continue
    for rel in list_files(source.root):
      absolute = os.path.join(source.root, rel.replace("/", os.sep))
      record_id = "F%d" % record_idx
      record_idx += 1
      fp, error = fingerprint_file(
          absolute, chunk_size=chunk_size, window_size=window_size)
      if error is not None:
        rec = FileRecord(
            record_id=record_id,
            source_id=source.source_id,
            relative_path=rel,
            absolute_path=absolute,
            size=-1,
            digest="",
            prefix_digest="",
            suffix_digest="",
            prefix_block=b"",
            suffix_block=b"",
            window_size=window_size,
            read_error=error,
        )
        records.append(rec)
        failures.append({
            "record_id": record_id,
            "source_id": source.source_id,
            "path": rel,
            "error": error,
        })
        continue
      rec = FileRecord(
          record_id=record_id,
          source_id=source.source_id,
          relative_path=rel,
          absolute_path=absolute,
          size=fp.size,
          digest=fp.digest_hex,
          prefix_digest=fp.prefix_digest_hex,
          suffix_digest=fp.suffix_digest_hex,
          prefix_block=fp.prefix_block,
          suffix_block=fp.suffix_block,
          window_size=fp.window_size,
      )
      records.append(rec)
      node = nodes_by_digest.get(fp.digest_hex)
      if node is None:
        node = ContentNode(
            digest=fp.digest_hex,
            size=fp.size,
            prefix_digest=fp.prefix_digest_hex,
            suffix_digest=fp.suffix_digest_hex,
            prefix_block=fp.prefix_block,
            suffix_block=fp.suffix_block,
            window_size=fp.window_size,
        )
        nodes_by_digest[fp.digest_hex] = node
      node.records.append(rec)

  # Deterministic record ordering inside each node.
  for node in nodes_by_digest.values():
    node.records.sort(key=lambda r: (r.source_id, r.relative_path))
  return records, list(nodes_by_digest.values()), failures
