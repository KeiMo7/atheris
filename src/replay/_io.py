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
"""Shared filesystem helpers: atomic JSON writes with no leftover temp files."""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any


def atomic_write_json(path: str, payload: Any) -> None:
  """Writes JSON to ``path`` atomically.

  The temp file is created in the destination directory (so ``os.replace`` is
  atomic on one filesystem) and removed on every failure path.  A crashed
  replay therefore can never leave a half-open file that a later run would
  mistake for valid state.
  """
  directory = os.path.dirname(os.path.abspath(path)) or "."
  os.makedirs(directory, exist_ok=True)
  tmp = tempfile.NamedTemporaryFile(
      mode="w",
      encoding="utf-8",
      dir=directory,
      prefix=".atheris-replay-",
      suffix=".tmp",
      delete=False,
  )
  tmp_path = tmp.name
  try:
    tmp.write(json.dumps(payload, indent=2, sort_keys=True))
    tmp.write("\n")
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
