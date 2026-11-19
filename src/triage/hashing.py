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
"""Streaming content fingerprints.

A fingerprint depends only on file *content*: never on path, timestamps or how
many times a file was seen. Identical byte sequences therefore produce
identical fingerprints on every machine, which is the basis for the
determinism guarantee of the triage package.

Each file is read in a single streaming pass. Full contents are never held in
memory: the full hash is updated incrementally, and only the last
``window_size`` bytes are buffered so that a suffix fingerprint is available
for files of any size, including multi-tens-of-megabyte inputs.
"""

import dataclasses
import hashlib
from typing import Optional
from typing import Tuple

# 64 KiB streaming chunks and a 64 KiB prefix/suffix window. Memory used per
# in-flight file is therefore bounded regardless of file size.
DEFAULT_CHUNK_SIZE = 1 << 16
DEFAULT_WINDOW_SIZE = 1 << 16
# Fixed-size edge blocks used for near-duplicate candidate bucketing. These are
# deliberately independent of the file size: the first/last N bytes of a tiny
# file and of a huge file are described on the same footing. 8 bytes is long
# enough that accidental bucket collisions are rare in fuzzer corpora, and
# short enough that prefix/suffix extensions land in a shared bucket.
EDGE_BLOCK_SIZE = 8

# BLAKE2b-256: collision resistance comfortably beyond SHA-256's practical
# security margin for this use case, available in the standard library, and
# fast. The algorithm name is recorded alongside every digest so a future
# migration never creates ambiguous fingerprints.
HASH_ALGORITHM = "blake2b-256"
_DIGEST_SIZE = 32


def new_hasher():
  return hashlib.blake2b(digest_size=_DIGEST_SIZE)


@dataclasses.dataclass(frozen=True)
class Fingerprint:
  """Content-derived identity of one file.

  Attributes:
    size: file size in bytes.
    digest_hex: hex digest of the full content.
    prefix_digest_hex: digest of the first EDGE_BLOCK_SIZE bytes (or of the
      whole content when the file is shorter). Used together with
      prefix_block for length-independent candidate bucketing.
    suffix_digest_hex: digest of the last EDGE_BLOCK_SIZE bytes (same
      short-file rule).
    prefix_block: the raw first EDGE_BLOCK_SIZE bytes (whole content if
      shorter). Candidate pairing compares these blocks directly, which makes
      a strict-prefix pair share a bucket regardless of either file's length.
    suffix_block: the raw last EDGE_BLOCK_SIZE bytes (same rule).
    window_size: tail-buffer window that was used.
  """

  size: int
  digest_hex: str
  prefix_digest_hex: str
  suffix_digest_hex: str
  prefix_block: bytes
  suffix_block: bytes
  window_size: int

  def as_dict(self) -> dict:
    d = dataclasses.asdict(self)
    # Raw blocks are not part of the serialized manifest identity.
    return d


def fingerprint_file(
    path: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    window_size: int = DEFAULT_WINDOW_SIZE,
) -> Tuple[Optional[Fingerprint], Optional[str]]:
  """Fingerprint a file in one streaming pass.

  Returns:
    (fingerprint, None) on success, or (None, error_message) when the file
    cannot be opened or read. The caller is responsible for recording the
    failure and continuing with the remaining files.
  """
  full = new_hasher()
  prefix = new_hasher()
  suffix = new_hasher()
  prefix_bytes = bytearray()
  tail = bytearray()
  size = 0
  try:
    with open(path, "rb") as f:
      while True:
        chunk = f.read(chunk_size)
        if not chunk:
          break
        full.update(chunk)
        if len(prefix_bytes) < EDGE_BLOCK_SIZE:
          prefix_bytes.extend(chunk[:EDGE_BLOCK_SIZE - len(prefix_bytes)])
        size += len(chunk)
        # Keep at most the last window_size bytes in memory.
        tail.extend(chunk)
        if len(tail) > window_size:
          del tail[:len(tail) - window_size]
  except OSError as e:
    return None, "%s: %s" % (type(e).__name__, e)

  prefix_block = bytes(prefix_bytes)
  suffix_block = bytes(tail[-min(EDGE_BLOCK_SIZE, len(tail)):]) if tail else b""
  prefix.update(prefix_block)
  suffix.update(suffix_block)
  return (
      Fingerprint(
          size=size,
          digest_hex=full.hexdigest(),
          prefix_digest_hex=prefix.hexdigest(),
          suffix_digest_hex=suffix.hexdigest(),
          prefix_block=prefix_block,
          suffix_block=suffix_block,
          window_size=window_size,
      ),
      None,
  )
  return (
      Fingerprint(
          size=size,
          digest_hex=full.hexdigest(),
          prefix_digest_hex=prefix.hexdigest(),
          suffix_digest_hex=suffix.hexdigest(),
          window_size=window_size,
      ),
      None,
  )


def hash_bytes(data: bytes) -> str:
  """Content hash of an in-memory byte string (used by tests and replay)."""
  h = new_hasher()
  h.update(data)
  return h.hexdigest()
