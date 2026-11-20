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

"""Subprocess worker for crash-sample replay and minimization.

This module is an implementation detail of :mod:`atheris.repro`. It is invoked
as ``python -m atheris.repro_worker <spec_file>`` in a fresh interpreter. The
worker imports the user's target the same way libFuzzer does and calls it with
exactly one ``bytes`` argument, matching the ``TestOneInput(data)`` convention.

Communication with the parent process is through files named in the spec:

* ``sample_path``: raw bytes that will be passed to the target.
* ``result_path``: a JSON verdict envelope written by the worker.

The worker never writes to the sample file. A fatal native signal (which cannot
be caught by Python code) simply leaves no envelope behind; the parent infers
the signal verdict from the child's negative return code.
"""

import faulthandler
import importlib
import json
import os
import sys
import traceback

# Exit codes used by the worker itself.
_EXIT_TARGET_EXCEPTION = 1
_EXIT_ENVIRONMENT_ERROR = 2


def _load_target(spec):
  """Imports the target callable described by ``spec``.

  Raises:
      Exception: any import or attribute-resolution failure, which the caller
          turns into an environment error.
  """
  for path_entry in spec.get("sys_path", []):
    if path_entry and path_entry not in sys.path:
      sys.path.insert(0, path_entry)

  module = importlib.import_module(spec["module"])
  target = module
  for attr in spec["qualname"].split("."):
    target = getattr(target, attr)
  if not callable(target):
    raise TypeError(
        f"Target {spec['module']}.{spec['qualname']} is not callable.")
  return target


def _frame_keys(tb, limit=16):
  """Runtime-observable frame identity: qualified name + line number only.

  File paths and sample-derived data are deliberately excluded so the signature
  neither depends on where the code is checked out nor on the input itself.
  """
  frames = []
  while tb is not None and len(frames) < limit:
    code = tb.tb_frame.f_code
    name = getattr(code, "co_qualname", code.co_name)
    frames.append(f"{name}:{tb.tb_lineno}")
    tb = tb.tb_next
  return frames


def _exception_type_name(exc):
  cls = type(exc)
  module = cls.__module__
  if module == "builtins":
    return cls.__qualname__
  return f"{module}.{cls.__qualname__}"


def _write_envelope(result_path, envelope):
  # Write to a sibling temp file and rename so the parent never observes a
  # partially written envelope.
  tmp_path = result_path + ".tmp"
  with open(tmp_path, "w", encoding="utf-8") as f:
    json.dump(envelope, f)
  os.replace(tmp_path, result_path)


def main(argv):
  # Emit Python traces for fatal signals on stderr. This is the only
  # runtime-observable information available for native crashes.
  faulthandler.enable()

  if len(argv) != 2:
    sys.stderr.write("repro_worker: expected exactly one spec path\n")
    return _EXIT_ENVIRONMENT_ERROR

  with open(argv[1], "r", encoding="utf-8") as f:
    spec = json.load(f)
  result_path = spec["result_path"]

  try:
    with open(spec["sample_path"], "rb") as f:
      data = f.read()
  except OSError as exc:
    _write_envelope(result_path, {
        "stage": "environment_error",
        "exit_kind": "environment_error",
        "error": f"failed to read sample: {exc!r}",
    })
    return _EXIT_ENVIRONMENT_ERROR

  try:
    target = _load_target(spec)
  except BaseException as exc:  # Import-time failure is an environment error.
    _write_envelope(result_path, {
        "stage": "environment_error",
        "exit_kind": "environment_error",
        "error": f"{_exception_type_name(exc)}: {exc}",
        "frames": _frame_keys(exc.__traceback__),
    })
    return _EXIT_ENVIRONMENT_ERROR

  envelope = {
      "stage": "ready",
      "exit_kind": "none",
      "exception_type": None,
      "exception_message": None,
      "frames": [],
      "exit_code": 0,
  }

  try:
    target(data)
  except SystemExit as exc:
    # sys.exit() is intercepted before the interpreter's default handling.
    code = exc.code
    if code is None:
      code = 0
    elif isinstance(code, str):
      envelope["exception_message"] = code
      code = 1
    elif not isinstance(code, int):
      code = 1
    envelope.update({
        "exit_kind": "exit",
        "exception_type": "SystemExit",
        "exit_code": code,
        "frames": _frame_keys(exc.__traceback__),
    })
    _write_envelope(result_path, envelope)
    return code if 0 <= code <= 255 else 1
  except BaseException as exc:
    envelope.update({
        "exit_kind": "exception",
        "exception_type": _exception_type_name(exc),
        "exception_message": str(exc),
        "frames": _frame_keys(exc.__traceback__),
        "exit_code": _EXIT_TARGET_EXCEPTION,
    })
    # Keep stderr informative for humans while the structured verdict goes to
    # the envelope.
    traceback.print_exc()
    _write_envelope(result_path, envelope)
    return _EXIT_TARGET_EXCEPTION

  _write_envelope(result_path, envelope)
  return 0


if __name__ == "__main__":
  sys.exit(main(sys.argv))
