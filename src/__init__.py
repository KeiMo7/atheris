# Copyright 2021 Google Inc.
# Copyright 2021 Fraunhofer FKIE
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
"""Atheris is a coverage-guided Python fuzzing engine."""

import sys
from typing import List

from .function_hooks import _hook_str
from .function_hooks import enabled_hooks
from .function_hooks import gen_match
from .import_hook import instrument_imports

from .instrument_bytecode import instrument_all
from .instrument_bytecode import instrument_func
from .instrument_bytecode import patch_code

# The native extension requires a compiled build (libFuzzer/clang). Some
# functionality, notably the post-run triage package (atheris.triage), is pure
# Python and must remain importable in environments where the extension could
# not be built. Import failures are therefore deferred until a native symbol is
# actually used, rather than raised at import time.
_NATIVE_IMPORT_ERROR = None
try:
  # MyPy cannot find native code.
  from .native import _reserve_counter  # type: ignore[import]
  from .native import _trace_branch  # type: ignore[import]
  from .native import _trace_cmp  # type: ignore[import]
  from .native import _trace_regex_match  # type: ignore[import]
  from .native import ALL_REMAINING  # type: ignore[import]
  from .native import build_mode  # type: ignore[import]
  from .native import Fuzz  # type: ignore[import]
  from .native import FuzzedDataProvider  # type: ignore[import]
  from .native import Mutate  # type: ignore[import]
  from .native import Setup  # type: ignore[import]
  from .native import UpdateCounterArrays  # type: ignore[import]
except ImportError as native_error:  # pragma: no cover - exercised without build
  _NATIVE_IMPORT_ERROR = native_error

  def _native_unavailable(*_args, **_kwargs):
    raise ImportError(
        "The Atheris native extension is not available in this environment "
        "(the underlying error was: %s). Build Atheris with a libFuzzer/"
        "clang toolchain to use fuzzing functionality. The pure-Python "
        "atheris.triage package remains available without the native build."
        % _NATIVE_IMPORT_ERROR)

  _reserve_counter = _native_unavailable
  _trace_branch = _native_unavailable
  _trace_cmp = _native_unavailable
  _trace_regex_match = _native_unavailable
  UpdateCounterArrays = _native_unavailable
  Fuzz = _native_unavailable
  Setup = _native_unavailable
  Mutate = _native_unavailable
  FuzzedDataProvider = _native_unavailable
  ALL_REMAINING = None

  def build_mode():
    return "unavailable"

from .utils import path


def native_extension_available() -> bool:
  """Returns whether the compiled native extension was imported successfully."""
  return _NATIVE_IMPORT_ERROR is None


# PyInstaller Support
# PyInstaller doesn't automatically support lazy imports, which happens because
# we dynamically decide whether to import the with/without_libfuzzer versions of
# the core module. This function tells it where to look for a hook-atheris.py
# file.


def get_hook_dirs() -> List[str]:
  import os  # pylint: disable=g-import-not-at-top
  return [os.path.dirname(__file__)]
