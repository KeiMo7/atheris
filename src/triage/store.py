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
"""Transactionally materialized triage output.

Guarantees:
  * Original artifact directories are only ever opened for reading; no input
    file is moved, renamed or deleted.
  * Output is assembled under a staging directory inside the destination's
    parent and published with a single atomic rename, so an interrupted or
    failed run leaves either the previous complete output or nothing at all.
  * Distinct contents are stored once in a content-addressed blob store.
    Hard links avoid duplicating bytes; a regular copy fallback covers
    cross-device / filesystem-without-hardlink cases.
  * A flock-based lock makes two concurrent runs against the same destination
    fail fast instead of overwriting each other.
  * Every represented file and every merged member is recorded in the manifest,
  * which is sufficient to reconstruct the original layout (reversibility).
"""

import contextlib
import errno
import fcntl
import json
import os
import shutil
from typing import Iterator, Optional

MANIFEST_NAME = "triage-manifest.json"
LOCK_NAME = ".triage.lock"
STAGING_PREFIX = ".triage-staging-"
BLOBS_DIRNAME = "blobs"


class TriageError(Exception):
  """Raised for user-visible triage failures (locking, IO, conflicts)."""


class ConcurrentTriageError(TriageError):
  """Raised when another process holds the destination lock."""


@contextlib.contextmanager
def destination_lock(destination: str) -> Iterator[None]:
  """Exclusive lock on the destination, released automatically on exit."""
  parent = os.path.abspath(os.path.dirname(destination.rstrip("/")) or ".")
  try:
    os.makedirs(parent, exist_ok=True)
  except OSError as e:
    raise TriageError(
        "cannot create destination parent %s: %s" % (parent, e))
  lock_path = os.path.join(parent, LOCK_NAME)
  try:
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
  except OSError as e:
    raise TriageError(
        "cannot acquire destination lock %s: %s" % (lock_path, e))
  try:
    try:
      fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as e:
      if e.errno in (errno.EWOULDBLOCK, errno.EAGAIN):
        raise ConcurrentTriageError(
            "another triage process is already working on %s (lock: %s)" %
            (destination, lock_path))
      raise
    yield
  finally:
    try:
      fcntl.flock(lock_fd, fcntl.LOCK_UN)
    finally:
      os.close(lock_fd)


class StagedOutput:
  """Context manager that builds output in staging and atomically publishes."""

  def __init__(self, destination: str, overwrite: bool = False):
    self.destination = os.path.abspath(destination)
    self.overwrite = overwrite
    self.parent = os.path.abspath(
        os.path.dirname(self.destination.rstrip("/")) or ".")
    self.staging = os.path.join(
        self.parent,
        STAGING_PREFIX + os.path.basename(self.destination.rstrip("/")))
    self.blobs_dir = os.path.join(self.staging, BLOBS_DIRNAME)
    self._committed = False

  def __enter__(self) -> "StagedOutput":
    if os.path.exists(self.destination):
      if not self.overwrite:
        raise TriageError(
            "destination %s already exists; pass overwrite=True or choose a "
            "new directory. Existing output is never modified in place." %
            self.destination)
    # Remove a stale staging directory from a previous crashed run.
    if os.path.exists(self.staging):
      shutil.rmtree(self.staging)
    os.makedirs(self.blobs_dir)
    return self

  def blob_path(self, digest: str) -> str:
    # Shard by the first two digest characters to keep directories small.
    return os.path.join(self.blobs_dir, digest[:2], digest)

  def add_blob(self, source_file: str, digest: str) -> str:
    """Place one distinct content into the blob store exactly once."""
    target = self.blob_path(digest)
    if os.path.exists(target):
      return target
    os.makedirs(os.path.dirname(target), exist_ok=True)
    try:
      os.link(source_file, target)
    except OSError as e:
      if e.errno not in (errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP,
                         errno.EMLINK):
        raise
      # Filesystem does not support hard links here: fall back to a streamed
      # copy so memory use stays bounded for large inputs.
      with open(source_file, "rb") as src, open(target, "wb") as dst:
        shutil.copyfileobj(src, dst, length=1 << 20)
    return target

  def write_text(self, relative: str, text: str) -> None:
    path = os.path.join(self.staging, relative)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Atomic per-file write inside staging as well.
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
      f.write(text)
    os.replace(tmp, path)

  def write_manifest(self, manifest: dict) -> None:
    # sort_keys + fixed indentation gives byte-stable manifests for the same
    # logical result, which supports cross-machine result comparison.
    self.write_text(
        MANIFEST_NAME,
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False))

  def commit(self) -> None:
    if self._committed:
      return
    # fsync the staging tree before the rename so a crash cannot publish a
    # directory whose contents never reached disk.
    _fsync_tree(self.staging)
    if os.path.exists(self.destination) and self.overwrite:
      backup = self.destination + ".old"
      if os.path.exists(backup):
        shutil.rmtree(backup)
      os.rename(self.destination, backup)
      try:
        os.rename(self.staging, self.destination)
      except OSError:
        os.rename(backup, self.destination)
        raise
      shutil.rmtree(backup)
    else:
      os.rename(self.staging, self.destination)
    self._committed = True

  def __exit__(self, exc_type, exc, tb) -> None:
    if not self._committed and os.path.exists(self.staging):
      shutil.rmtree(self.staging, ignore_errors=True)


def _fsync_tree(root: str) -> None:
  for dirpath, dirnames, filenames in os.walk(root):
    for name in filenames:
      path = os.path.join(dirpath, name)
      try:
        fd = os.open(path, os.O_RDONLY)
      except OSError:
        continue
      try:
        os.fsync(fd)
      finally:
        os.close(fd)
  fd = os.open(root, os.O_RDONLY)
  try:
    os.fsync(fd)
  finally:
    os.close(fd)


def manifest_path(destination: str) -> str:
  return os.path.join(destination, MANIFEST_NAME)


def load_manifest(destination: str) -> Optional[ dict]:
  path = manifest_path(destination)
  if not os.path.isfile(path):
    return None
  with open(path, "r", encoding="utf-8") as f:
    return json.load(f)
