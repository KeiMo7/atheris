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
"""Small, dependency-free helpers for durable artifact storage.

The helpers in this module deliberately avoid external libraries.  Artifacts
are written to a temporary file in the destination directory, flushed, and
then published with ``os.replace``.  This gives readers either the old file or
the complete new file, even if the writer is killed part-way through.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable


def utc_now(clock: Callable[[], float] | None = None) -> str:
  """Returns an ISO-8601 UTC timestamp."""
  import datetime

  timestamp = clock() if clock is not None else None
  if timestamp is None:
    value = datetime.datetime.now(datetime.timezone.utc)
  else:
    value = datetime.datetime.fromtimestamp(timestamp, datetime.timezone.utc)
  return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
  return hashlib.sha256(data).hexdigest()


def sha256_file(path: os.PathLike[str] | str) -> str:
  digest = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _sync_directory(path: Path) -> None:
  """Best-effort fsync of a directory after publishing a file."""
  flags = getattr(os, "O_DIRECTORY", 0) | os.O_RDONLY
  try:
    fd = os.open(path, flags)
  except OSError:
    return
  try:
    os.fsync(fd)
  except OSError:
    pass
  finally:
    os.close(fd)


def atomic_write_bytes(path: os.PathLike[str] | str, data: bytes) -> None:
  """Atomically writes ``data`` to ``path``."""
  destination = Path(path)
  destination.parent.mkdir(parents=True, exist_ok=True)
  fd, temporary_name = tempfile.mkstemp(
      prefix=f".{destination.name}.", suffix=".tmp",
      dir=str(destination.parent))
  temporary = Path(temporary_name)
  try:
    with os.fdopen(fd, "wb") as f:
      f.write(data)
      f.flush()
      os.fsync(f.fileno())
    os.replace(temporary, destination)
    _sync_directory(destination.parent)
  except BaseException:
    try:
      temporary.unlink()
    except FileNotFoundError:
      pass
    raise


def atomic_write_text(path: os.PathLike[str] | str,
                      text: str,
                      encoding: str = "utf-8") -> None:
  atomic_write_bytes(path, text.encode(encoding))


def atomic_write_json(path: os.PathLike[str] | str, value: Any) -> None:
  serialized = json.dumps(
      value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
  atomic_write_text(path, serialized)

