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
"""Crash deduplication, durable storage, and budgeted minimization.

The crash manager intentionally runs the target in a subprocess.  Consequently
all features are derived from the observable crash report (exit status,
signal, stdout/stderr, and normalized stack frames) plus the input bytes.  It
does not inspect or depend on the crashed process's internal state.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import dataclasses
from dataclasses import dataclass, field
import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from typing import Callable, Iterable, Iterator, Sequence

from .artifact_io import atomic_write_bytes
from .artifact_io import atomic_write_json
from .artifact_io import sha256_bytes
from .artifact_io import utc_now

STORE_FORMAT = "atheris-crash-store"
STORE_VERSION = 1
MAX_STORED_OUTPUT = 64 * 1024
DEFAULT_STACK_DEPTH = 8

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")
_PYTHON_FRAME_RE = re.compile(
    r'^\s*File "(?P<file>.*)", line (?P<line>\d+), '
    r"in (?P<function>.+?)\s*$")
_EXCEPTION_RE = re.compile(
    r"^(?P<type>[A-Za-z_][A-Za-z0-9_.]*)\s*:")
_SANITIZER_RE = re.compile(
    r"ERROR: (?P<tool>AddressSanitizer|UndefinedBehaviorSanitizer|"
    r"MemorySanitizer|ThreadSanitizer|LeakSanitizer)"
    r"(?::\s*(?P<kind>[A-Za-z0-9_-]+))?")
_SUMMARY_RE = re.compile(
    r"SUMMARY: (?P<tool>AddressSanitizer|UndefinedBehaviorSanitizer|"
    r"MemorySanitizer|ThreadSanitizer|LeakSanitizer)"
    r":\s*(?P<kind>\S+)")
_NATIVE_LOCATION_RE = re.compile(
    r"(?P<location>(?:[A-Za-z]:)?(?:/|\./|\.\./|[A-Za-z0-9_.-]+/)"
    r"[^ \t:]+:\d+(?::\d+)?)\s*$")
_HEX_ADDRESS_RE = re.compile(r"\b0x[0-9a-fA-F]+\b")
_LINUX_SIGNAL_RE = re.compile(r"ERROR: libFuzzer: deadly signal")


def _bounded_output(text: str) -> str:
  if len(text) <= MAX_STORED_OUTPUT:
    return text
  return text[-MAX_STORED_OUTPUT:]


def _signal_name(number: int | None) -> str | None:
  if number is None:
    return None
  try:
    return signal.Signals(number).name
  except ValueError:
    return f"SIG{number}"


def _normalize_path(path: str, input_path: str | None) -> str:
  normalized = path.replace("\\", "/")
  if input_path:
    for candidate in (input_path, os.path.abspath(input_path)):
      normalized = normalized.replace(candidate.replace("\\", "/"), "<input>")
  normalized = _HEX_ADDRESS_RE.sub("0xADDR", normalized)
  return normalized


def _normalize_text(text: str, input_path: str | None) -> str:
  text = _ANSI_ESCAPE_RE.sub("", text)
  text = text.replace("\r\n", "\n")
  if input_path:
    for candidate in (input_path, os.path.abspath(input_path)):
      text = text.replace(candidate, "<input>")
  return _HEX_ADDRESS_RE.sub("0xADDR", text)


@dataclass(frozen=True)
class CrashSignature:
  """A stable description of an observable crash."""

  kind: str
  crash_type: str
  frames: tuple[str, ...]
  extraction_failed: bool = False
  extraction_error: str | None = None
  key: str = ""

  def to_dict(self) -> dict:
    return {
        "key": self.key,
        "kind": self.kind,
        "crash_type": self.crash_type,
        "frames": list(self.frames),
        "extraction_failed": self.extraction_failed,
        "extraction_error": self.extraction_error,
    }


@dataclass
class CrashOutcome:
  """Result of replaying one input against the target."""

  returncode: int
  signal: int | None
  timed_out: bool
  duration_ms: int
  status: str
  stdout: str
  stderr: str
  signature: CrashSignature
  input_path: str | None = None

  @property
  def crashed(self) -> bool:
    return self.status in ("crash", "timeout")

  def to_dict(self, include_output: bool = True) -> dict:
    value = {
        "returncode": self.returncode,
        "signal": self.signal,
        "signal_name": _signal_name(self.signal),
        "timed_out": self.timed_out,
        "duration_ms": self.duration_ms,
        "status": self.status,
        "signature": self.signature.to_dict(),
    }
    if include_output:
      value["stdout"] = self.stdout
      value["stderr"] = self.stderr
    return value


def _extract_python(
    text: str, input_path: str | None) -> tuple[str | None, tuple[str, ...]]:
  frames = []
  frame_lines = []
  lines = text.splitlines()
  for index, line in enumerate(lines):
    match = _PYTHON_FRAME_RE.match(line)
    if match:
      frame_lines.append(index)
      frames.append(
          f"{_normalize_path(match.group('file'), input_path)}:"
          f"{match.group('line')}:{match.group('function')}")

  exception_type = None
  ignored_prefixes = (
      "File ", "Traceback ", "Thread ", "Current thread ", "WARNING:",
      "INFO:", "ERROR:", "SUMMARY:", "Running:")
  for frame_index in reversed(frame_lines):
    for line in lines[frame_index + 1:]:
      stripped = line.strip()
      if not stripped or line[:1].isspace():
        continue
      if stripped.startswith(ignored_prefixes):
        break
      match = _EXCEPTION_RE.match(stripped)
      if match:
        exception_type = match.group("type")
        break
      break
    if exception_type:
      break
  return exception_type, tuple(frames)


def _extract_native_frames(
    text: str, input_path: str | None) -> tuple[str, ...]:
  frames = []
  for line in text.splitlines():
    stripped = line.strip()
    if not stripped.startswith("#"):
      continue
    match = re.match(r"#(?P<index>\d+)\s+(?P<body>.*)", stripped)
    if not match:
      continue
    body = _HEX_ADDRESS_RE.sub("0xADDR", match.group("body")).strip()
    body = re.sub(r"^0xADDR\s+", "", body)
    body = re.sub(r"^in\s+", "", body)
    location_match = _NATIVE_LOCATION_RE.search(body)
    if location_match:
      location = _normalize_path(location_match.group("location"), input_path)
      function = body[:location_match.start()].strip() or "<unknown>"
      frames.append(f"{location}:{function}")
    elif body:
      frames.append(body)

  if not frames:
    summary_match = _SUMMARY_RE.search(text)
    if summary_match:
      tail = text[summary_match.end():].strip().splitlines()
      if tail:
        frames.append(_normalize_path(tail[0].strip(), input_path))
  return tuple(frames)


def extract_crash_signature(text: str,
                            returncode: int,
                            timed_out: bool,
                            input_path: str | None = None,
                            sample_sha256: str | None = None,
                            stack_depth: int = DEFAULT_STACK_DEPTH
                            ) -> CrashSignature:
  """Derives a grouping key from process-observable crash evidence."""
  normalized_text = _normalize_text(text, input_path)
  exception_type, python_frames = _extract_python(
      normalized_text, input_path)
  sanitizer = _SANITIZER_RE.search(normalized_text)
  summary = _SUMMARY_RE.search(normalized_text)
  native_frames = _extract_native_frames(normalized_text, input_path)

  frames: tuple[str, ...] = ()
  kind = "unknown"
  crash_type = "unknown"

  match = sanitizer or summary
  if match:
    tool = match.group("tool")
    bug_kind = match.group("kind")
    frames = (native_frames or python_frames)[-stack_depth:]
    kind = tool
    crash_type = bug_kind or "crash"
  elif python_frames:
    frames = python_frames[-stack_depth:]
    if exception_type:
      kind = "python"
      crash_type = exception_type
    elif timed_out:
      kind = "timeout"
      crash_type = "timeout"
    else:
      kind = "python"
      crash_type = "exception"
  elif native_frames:
    frames = native_frames[-stack_depth:]
    kind = "native"
    crash_type = "crash"
  elif timed_out or "=== Timeout:" in normalized_text:
    kind = "timeout"
    crash_type = "timeout"
  elif returncode < 0:
    kind = "signal"
    crash_type = _signal_name(-returncode) or str(-returncode)
  elif _LINUX_SIGNAL_RE.search(normalized_text):
    kind = "libfuzzer"
    crash_type = "deadly signal"
  else:
    kind = "exit"
    crash_type = f"exit_{returncode}"

  extraction_failed = not frames
  extraction_error = None
  if extraction_failed:
    extraction_error = "no stable stack frames found"

  if extraction_failed:
    key = f"ungrouped:{sample_sha256 or 'unknown'}"
  else:
    key = (
        f"{kind}:{crash_type}|" + "->".join(frames))

  return CrashSignature(
      kind=kind,
      crash_type=crash_type,
      frames=frames,
      extraction_failed=extraction_failed,
      extraction_error=extraction_error,
      key=key)


class TargetRunner:
  """Runs a target in a subprocess and parses observable crash evidence."""

  def __init__(self,
               command: Sequence[str] | str,
               *,
               timeout: float = 10.0,
               cwd: str | os.PathLike[str] | None = None,
               env: dict[str, str] | None = None,
               input_placeholder: str = "{input}"):
    if isinstance(command, str):
      command = shlex.split(command)
    if not command:
      raise ValueError("target command must not be empty")
    self.command = list(command)
    self.timeout = timeout
    self.cwd = cwd
    self.env = dict(os.environ if env is None else env)
    self.input_placeholder = input_placeholder
    self._work_dir = Path(tempfile.mkdtemp(prefix="atheris-replay-"))
    self._input_path = self._work_dir / "input"

  def _expanded_command(self, input_path: Path) -> list[str]:
    has_placeholder = any(
        self.input_placeholder in argument for argument in self.command)
    if has_placeholder:
      return [
          argument.replace(self.input_placeholder, str(input_path))
          for argument in self.command
      ]
    return self.command + [str(input_path)]

  def run(self, data: bytes) -> CrashOutcome:
    atomic_write_bytes(self._input_path, data)
    command = self._expanded_command(self._input_path)
    env = dict(self.env)
    env["ATHERIS_CRASH_INPUT"] = str(self._input_path)

    start = time.monotonic()
    timed_out = False
    try:
      process = subprocess.Popen(
          command,
          cwd=self.cwd,
          env=env,
          stdin=subprocess.DEVNULL,
          stdout=subprocess.PIPE,
          stderr=subprocess.PIPE,
          start_new_session=True,
      )
    except OSError as e:
      duration_ms = int((time.monotonic() - start) * 1000)
      text = f"failed to start target: {e}"
      signature = extract_crash_signature(
          text, 127, False, str(self._input_path), sha256_bytes(data))
      return CrashOutcome(
          returncode=127, signal=None, timed_out=False, duration_ms=duration_ms,
          status="crash", stdout="", stderr=text, signature=signature,
          input_path=str(self._input_path))

    try:
      stdout_bytes, stderr_bytes = process.communicate(timeout=self.timeout)
    except subprocess.TimeoutExpired:
      timed_out = True
      try:
        os.killpg(process.pid, signal.SIGKILL)
      except ProcessLookupError:
        pass
      stdout_bytes, stderr_bytes = process.communicate()

    duration_ms = int((time.monotonic() - start) * 1000)
    stdout = _bounded_output(
        stdout_bytes.decode("utf-8", errors="backslashreplace"))
    stderr = _bounded_output(
        stderr_bytes.decode("utf-8", errors="backslashreplace"))
    if timed_out:
      status = "timeout"
    elif process.returncode != 0:
      status = "crash"
    else:
      status = "clean"
    signal_number = -process.returncode if process.returncode < 0 else None
    signature = extract_crash_signature(
        stdout + stderr,
        process.returncode,
        timed_out,
        str(self._input_path),
        sha256_bytes(data))
    return CrashOutcome(
        returncode=process.returncode,
        signal=signal_number,
        timed_out=timed_out,
        duration_ms=duration_ms,
        status=status,
        stdout=stdout,
        stderr=stderr,
        signature=signature,
        input_path=str(self._input_path))

  def close(self) -> None:
    import shutil
    shutil.rmtree(self._work_dir, ignore_errors=True)

  def __enter__(self) -> "TargetRunner":
    return self

  def __exit__(self, exc_type, exc_value, traceback) -> None:
    self.close()

  def __del__(self):
    try:
      self.close()
    except Exception:  # pylint: disable=broad-except
      pass


def _parse_timestamp(value: str | float | int | None) -> float | None:
  if value is None:
    return None
  if isinstance(value, (int, float)):
    return float(value)
  parsed = value.strip()
  if parsed.endswith("Z"):
    parsed = parsed[:-1] + "+00:00"
  return datetime.datetime.fromisoformat(parsed).timestamp()


class CrashStore:
  """Persistent, concurrency-safe crash group store."""

  def __init__(self,
               root: str | os.PathLike[str],
               *,
               clock: Callable[[], float] | None = None,
               monotonic_clock: Callable[[], float] | None = None):
    self.root = Path(root)
    self._crashes_dir = self.root / "crashes"
    self._minimized_dir = self.root / "minimized"
    self._state_path = self.root / "state.json"
    self._lock_path = self.root / ".lock"
    self._clock = clock
    self._monotonic_clock = monotonic_clock or time.monotonic
    self._crashes_dir.mkdir(parents=True, exist_ok=True)
    self._minimized_dir.mkdir(parents=True, exist_ok=True)

    with self._lock():
      if not self._state_path.exists():
        self._write_state(self._new_state())
      else:
        self._read_state()

  def _now(self) -> str:
    return utc_now(self._clock)

  def _monotonic(self) -> float:
    return self._monotonic_clock()

  def _new_state(self) -> dict:
    now = self._now()
    return {
        "format": STORE_FORMAT,
        "version": STORE_VERSION,
        "created_at": now,
        "updated_at": now,
        "groups": {},
        "sample_to_group": {},
    }

  @contextmanager
  def _lock(self) -> Iterator[None]:
    self.root.mkdir(parents=True, exist_ok=True)
    with open(self._lock_path, "a+b") as lock_file:
      fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
      try:
        yield
      finally:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

  def _read_state(self) -> dict:
    try:
      state = json.loads(self._state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
      raise RuntimeError(f"cannot read crash store {self._state_path}: {e}") from e
    if not isinstance(state, dict) or state.get("format") != STORE_FORMAT:
      raise RuntimeError(f"invalid crash store format in {self._state_path}")
    if state.get("version") != STORE_VERSION:
      raise RuntimeError(
          f"unsupported crash store version: {state.get('version')!r}")
    state.setdefault("groups", {})
    state.setdefault("sample_to_group", {})
    return state

  def _write_state(self, state: dict) -> None:
    state["updated_at"] = self._now()
    atomic_write_json(self._state_path, state)

  def _relative_path(self, path: Path) -> str:
    return path.relative_to(self.root).as_posix()

  def path_for_sample(self, sample_id: str) -> Path | None:
    path = self._crashes_dir / sample_id
    return path if path.is_file() else None

  def _group_id(self, signature_key: str) -> str:
    return sha256_bytes(signature_key.encode("utf-8"))[:20]

  def _new_group(self, outcome: CrashOutcome, sample_id: str,
                 now: str) -> dict:
    return {
        "id": self._group_id(outcome.signature.key),
        "signature_key": outcome.signature.key,
        "signature": outcome.signature.to_dict(),
        "count": 0,
        "unique_samples": 0,
        "first_seen": now,
        "last_seen": now,
        "representative": None,
        "samples": {},
    }

  def ingest(self,
             data: bytes,
             outcome: CrashOutcome,
             *,
             source: str | None = None) -> dict:
    """Persists one replayed sample and folds it into a crash group."""
    if not outcome.crashed:
      raise ValueError("refusing to ingest an input that did not crash")

    sample_id = sha256_bytes(data)
    sample_path = self._crashes_dir / sample_id
    atomic_write_bytes(sample_path, data)
    now = self._now()

    with self._lock():
      state = self._read_state()
      existing_group_id = state["sample_to_group"].get(sample_id)
      if existing_group_id and existing_group_id in state["groups"]:
        group = state["groups"][existing_group_id]
        is_new = False
      else:
        group_id = self._group_id(outcome.signature.key)
        group = state["groups"].get(group_id)
        is_new = group is None
        if group is None:
          group = self._new_group(outcome, sample_id, now)
          state["groups"][group_id] = group
        state["sample_to_group"][sample_id] = group_id

      if sample_id not in group["samples"]:
        group["samples"][sample_id] = {
            "id": sample_id,
            "path": self._relative_path(sample_path),
            "size": len(data),
            "sha256": sample_id,
            "source": source,
            "added_at": now,
            "outcome": outcome.to_dict(),
            "minimization": None,
        }
        group["unique_samples"] = len(group["samples"])
        group["count"] += 1
      else:
        is_new = False
      group["last_seen"] = now

      representative = self._choose_representative(group)
      group["representative"] = representative
      self._write_state(state)

    return {
        "sample_id": sample_id,
        "group_id": group["id"],
        "new_group": is_new,
        "duplicate": not is_new and existing_group_id is not None,
        "group": deepcopy(group),
    }

  def _choose_representative(self, group: dict) -> dict:
    candidates = []
    for sample_id, sample in group["samples"].items():
      minimization = sample.get("minimization") or {}
      best_size = minimization.get("minimized_size") or sample["size"]
      best_path = (
          minimization.get("minimized_path")
          if minimization.get("minimized_size") is not None else None)
      candidates.append((
          best_size,
          sample.get("added_at", ""),
          sample_id,
          sample,
          best_path,
      ))
    if not candidates:
      return None
    best_size, _, sample_id, sample, best_path = min(candidates)
    return {
        "sample_id": sample_id,
        "path": best_path or sample["path"],
        "size": best_size,
        "sha256": sample["sha256"],
        "signature": group["signature"],
    }

  def ingest_file(self, path: str | os.PathLike[str],
                  runner: TargetRunner) -> dict:
    data = Path(path).read_bytes()
    outcome = runner.run(data)
    return self.ingest(data, outcome, source=str(path))

  def ingest_paths(self,
                   paths: Iterable[str | os.PathLike[str]],
                   runner: TargetRunner) -> list[dict]:
    files = []
    for raw_path in paths:
      path = Path(raw_path)
      if path.is_file():
        files.append(path)
      elif path.is_dir():
        files.extend(path.rglob("*"))
      else:
        raise FileNotFoundError(path)

    results = []
    for path in sorted(files):
      if not path.is_file() or path.name.startswith("."):
        continue
      results.append(self.ingest_file(path, runner))
    return results

  def get_sample(self, sample_id: str) -> dict:
    with self._lock():
      state = self._read_state()
      group_id = state["sample_to_group"].get(sample_id)
      if group_id is None:
        raise KeyError(sample_id)
      return deepcopy(state["groups"][group_id]["samples"][sample_id])

  def sample_bytes(self, sample_id: str) -> bytes:
    sample = self.get_sample(sample_id)
    return (self.root / sample["path"]).read_bytes()

  def groups(self,
             since: str | float | None = None,
             until: str | float | None = None,
             *,
             include_samples: bool = False) -> list[dict]:
    """Returns groups with events in the requested half-open time window."""
    since_value = _parse_timestamp(since)
    until_value = _parse_timestamp(until)
    with self._lock():
      state = self._read_state()

    result = []
    for group in state["groups"].values():
      selected_samples = {}
      for sample_id, sample in group["samples"].items():
        added_at = _parse_timestamp(sample["added_at"])
        if since_value is not None and added_at < since_value:
          continue
        if until_value is not None and added_at >= until_value:
          continue
        selected_samples[sample_id] = deepcopy(sample)
      if not selected_samples:
        continue
      item = {
          "id": group["id"],
          "signature_key": group["signature_key"],
          "signature": deepcopy(group["signature"]),
          "total_count": group["count"],
          "window_count": len(selected_samples),
          "first_seen": min(
              sample["added_at"] for sample in selected_samples.values()),
          "last_seen": max(
              sample["added_at"] for sample in selected_samples.values()),
          "representative": deepcopy(group["representative"]),
      }
      if include_samples:
        item["samples"] = selected_samples
      result.append(item)
    result.sort(key=lambda item: (item["first_seen"], item["id"]))
    return result

  def _record_minimization(self, sample_id: str, record: dict) -> None:
    with self._lock():
      state = self._read_state()
      group_id = state["sample_to_group"].get(sample_id)
      if group_id is None:
        raise KeyError(sample_id)
      group = state["groups"][group_id]
      sample = group["samples"][sample_id]
      sample["minimization"] = record
      group["representative"] = self._choose_representative(group)
      self._write_state(state)


def _same_crash(original: CrashOutcome, candidate: CrashOutcome) -> bool:
  if not candidate.crashed:
    return False
  if original.timed_out != candidate.timed_out:
    return False
  if original.signature.extraction_failed:
    if not candidate.signature.extraction_failed:
      return False
    return (original.returncode == candidate.returncode and
            original.signal == candidate.signal and
            original.signature.crash_type == candidate.signature.crash_type)
  if candidate.signature.extraction_failed:
    return False
  return original.signature.key == candidate.signature.key


@dataclass
class MinimizeResult:
  sample_id: str
  group_id: str
  original_size: int
  minimized_size: int
  minimized_path: str | None
  stop_reason: str
  verified: bool
  step_count: int
  elapsed_seconds: float
  record: dict = field(default_factory=dict)

  def to_dict(self) -> dict:
    return dataclasses.asdict(self)


class CrashMinimizer:
  """Delta-debugging minimizer with persistent, bounded execution history."""

  def __init__(self,
               store: CrashStore,
               runner: TargetRunner,
               *,
               max_steps: int = 500,
               time_budget: float = 60.0,
               run_timeout: float | None = None,
               verify_repetitions: int = 2):
    if max_steps <= 0:
      raise ValueError("max_steps must be positive")
    if time_budget <= 0:
      raise ValueError("time_budget must be positive")
    if verify_repetitions < 0:
      raise ValueError("verify_repetitions must not be negative")
    self.store = store
    self.runner = runner
    if run_timeout is not None:
      self.runner.timeout = run_timeout
    self.max_steps = max_steps
    self.time_budget = time_budget
    self.verify_repetitions = verify_repetitions

  def minimize(self, sample_id: str) -> MinimizeResult:
    sample = self.store.get_sample(sample_id)
    original_data = self.store.sample_bytes(sample_id)
    started_monotonic = self.store._monotonic()  # pylint: disable=protected-access
    deadline = started_monotonic + self.time_budget
    original_outcome = self.runner.run(original_data)
    stored_outcome = CrashOutcome(
        returncode=sample["outcome"]["returncode"],
        signal=sample["outcome"]["signal"],
        timed_out=sample["outcome"]["timed_out"],
        duration_ms=sample["outcome"]["duration_ms"],
        status=sample["outcome"]["status"],
        stdout=sample["outcome"].get("stdout", ""),
        stderr=sample["outcome"].get("stderr", ""),
        signature=CrashSignature(
            kind=sample["outcome"]["signature"]["kind"],
            crash_type=sample["outcome"]["signature"]["crash_type"],
            frames=tuple(sample["outcome"]["signature"]["frames"]),
            extraction_failed=sample["outcome"]["signature"][
                "extraction_failed"],
            extraction_error=sample["outcome"]["signature"].get(
                "extraction_error"),
            key=sample["outcome"]["signature"]["key"],
        ),
    )

    record = {
        "started_at": self.store._now(),  # pylint: disable=protected-access
        "finished_at": None,
        "budget": {
            "max_steps": self.max_steps,
            "time_budget_seconds": self.time_budget,
            "run_timeout_seconds": self.runner.timeout,
            "verify_repetitions": self.verify_repetitions,
        },
        "original_size": len(original_data),
        "original_sha256": sample_id,
        "original_outcome": original_outcome.to_dict(),
        "best_size": len(original_data),
        "best_sha256": sample_id,
        "best_path": sample["path"],
        "minimized_size": None,
        "minimized_path": None,
        "verified": False,
        "stop_reason": "running",
        "steps": [],
    }

    original_matches = (
        original_outcome.crashed and
        _same_crash(stored_outcome, original_outcome))
    record["steps"].append({
        "step": 0,
        "phase": "original",
        "input_size": len(original_data),
        "candidate_sha256": sample_id,
        "status": original_outcome.status,
        "returncode": original_outcome.returncode,
        "signal": original_outcome.signal,
        "timed_out": original_outcome.timed_out,
        "signature_key": original_outcome.signature.key,
        "accepted": original_matches,
        "reason": "accepted" if original_matches else "original_mismatch",
        "duration_ms": original_outcome.duration_ms,
    })
    self.store._record_minimization(sample_id, record)  # pylint: disable=protected-access

    if not original_matches:
      record["stop_reason"] = "original_crash_not_reproduced"
      record["finished_at"] = self.store._now()  # pylint: disable=protected-access
      self.store._record_minimization(sample_id, record)  # pylint: disable=protected-access
      return self._result(sample_id, sample, record, started_monotonic)

    best_data = original_data
    best_outcome = original_outcome
    accepted = [(best_data, best_outcome)]
    step_number = 0
    stop_reason = "unchanged"

    def test_candidate(candidate: bytes, phase: str) -> tuple[bool, str]:
      nonlocal step_number, best_data, best_outcome, stop_reason
      if step_number >= self.max_steps:
        stop_reason = "step_budget_exhausted"
        return False, stop_reason
      if self.store._monotonic() >= deadline:  # pylint: disable=protected-access
        stop_reason = "time_budget_exhausted"
        return False, stop_reason

      step_number += 1
      outcome = self.runner.run(candidate)
      matches = _same_crash(original_outcome, outcome)
      reason = "accepted" if matches else "crash_type_changed"
      if outcome.timed_out:
        reason = "target_timeout"
      elif not outcome.crashed:
        reason = "did_not_crash"

      step = {
          "step": step_number,
          "phase": phase,
          "input_size": len(candidate),
          "candidate_sha256": sha256_bytes(candidate),
          "status": outcome.status,
          "returncode": outcome.returncode,
          "signal": outcome.signal,
          "timed_out": outcome.timed_out,
          "signature_key": outcome.signature.key,
          "accepted": matches,
          "reason": reason,
          "duration_ms": outcome.duration_ms,
      }
      record["steps"].append(step)

      if matches and len(candidate) < len(best_data):
        best_data = candidate
        best_outcome = outcome
        accepted.append((best_data, best_outcome))
        minimized_path = (
            self.store._minimized_dir /  # pylint: disable=protected-access
            sha256_bytes(best_data))
        atomic_write_bytes(minimized_path, best_data)
        record["best_size"] = len(best_data)
        record["best_sha256"] = sha256_bytes(best_data)
        record["best_path"] = (
            self.store._relative_path(minimized_path))  # pylint: disable=protected-access
        record["minimized_size"] = len(best_data)
        record["minimized_path"] = record["best_path"]

      record["stop_reason"] = reason if outcome.timed_out else stop_reason
      self.store._record_minimization(  # pylint: disable=protected-access
          sample_id, record)
      return matches, reason

    # Empty input is always worth checking: it is the true minimum if the
    # target crashes without consuming any bytes.
    if original_data and self.verify_repetitions >= 0:
      matches, reason = test_candidate(b"", "empty")
      if matches:
        stop_reason = "minimal"
      elif reason in (
          "target_timeout", "step_budget_exhausted",
          "time_budget_exhausted"):
        stop_reason = reason

    if not stop_reason.startswith("target_") and stop_reason != "minimal":
      data = best_data
      granularity = 2
      while data and step_number < self.max_steps:
        if self.store._monotonic() >= deadline:  # pylint: disable=protected-access
          stop_reason = "time_budget_exhausted"
          break
        if granularity > len(data):
          break
        chunk_size = int(math.ceil(len(data) / granularity))
        reduced = False
        for start in range(0, len(data), chunk_size):
          candidate = data[:start] + data[start + chunk_size:]
          matches, reason = test_candidate(candidate, "ddmin")
          if reason in ("step_budget_exhausted", "time_budget_exhausted"):
            stop_reason = reason
            break
          if reason == "target_timeout":
            stop_reason = reason
            break
          if matches:
            data = best_data
            granularity = max(2, granularity - 1)
            reduced = True
            break
        if stop_reason in (
            "target_timeout", "step_budget_exhausted", "time_budget_exhausted"):
          break
        if not reduced:
          if granularity >= len(data):
            stop_reason = "no_further_reduction"
            break
          granularity = min(len(data), granularity * 2)
      if data and stop_reason == "unchanged":
        stop_reason = "no_further_reduction"

    verified = False
    verification_reason = None
    for _ in range(self.verify_repetitions):
      if step_number >= self.max_steps:
        verification_reason = "step_budget_exhausted"
        break
      if self.store._monotonic() >= deadline:  # pylint: disable=protected-access
        verification_reason = "time_budget_exhausted"
        break
      step_number += 1
      outcome = self.runner.run(best_data)
      matches = _same_crash(original_outcome, outcome)
      record["steps"].append({
          "step": step_number,
          "phase": "verify",
          "input_size": len(best_data),
          "candidate_sha256": sha256_bytes(best_data),
          "status": outcome.status,
          "returncode": outcome.returncode,
          "signal": outcome.signal,
          "timed_out": outcome.timed_out,
          "signature_key": outcome.signature.key,
          "accepted": matches,
          "reason": "accepted" if matches else "verification_failed",
          "duration_ms": outcome.duration_ms,
      })
      if not matches:
        verification_reason = "verification_failed"
        break
    else:
      verified = bool(self.verify_repetitions)

    if verification_reason == "verification_failed" and len(accepted) > 1:
      # Roll back to the last candidate that matched during reduction.
      for fallback_data, fallback_outcome in reversed(accepted[:-1]):
        if step_number >= self.max_steps:
          verification_reason = "step_budget_exhausted"
          break
        if self.store._monotonic() >= deadline:  # pylint: disable=protected-access
          verification_reason = "time_budget_exhausted"
          break
        step_number += 1
        fallback = self.runner.run(fallback_data)
        fallback_matches = _same_crash(original_outcome, fallback)
        record["steps"].append({
            "step": step_number,
            "phase": "fallback_verify",
            "input_size": len(fallback_data),
            "candidate_sha256": sha256_bytes(fallback_data),
            "status": fallback.status,
            "returncode": fallback.returncode,
            "signal": fallback.signal,
            "timed_out": fallback.timed_out,
            "signature_key": fallback.signature.key,
            "accepted": fallback_matches,
            "reason": "accepted" if fallback_matches else "verification_failed",
            "duration_ms": fallback.duration_ms,
        })
        if fallback_matches:
          best_data = fallback_data
          best_outcome = fallback_outcome
          record["best_size"] = len(best_data)
          record["best_sha256"] = sha256_bytes(best_data)
          record["minimized_size"] = len(best_data)
          fallback_path = (
              self.store._minimized_dir /  # pylint: disable=protected-access
              sha256_bytes(best_data))
          atomic_write_bytes(fallback_path, best_data)
          record["best_path"] = (
              self.store._relative_path(fallback_path))  # pylint: disable=protected-access
          record["minimized_path"] = record["best_path"]
          verified = True
          break
      stop_reason = verification_reason
    elif verification_reason:
      stop_reason = verification_reason
      verified = False

    record["finished_at"] = self.store._now()  # pylint: disable=protected-access
    record["stop_reason"] = stop_reason
    record["verified"] = verified
    record["minimized_size"] = len(best_data)
    record["best_size"] = len(best_data)
    record["best_sha256"] = sha256_bytes(best_data)
    if len(best_data) < len(original_data) or record["minimized_path"] is None:
      minimized_path = (
          self.store._minimized_dir /  # pylint: disable=protected-access
          sha256_bytes(best_data))
      atomic_write_bytes(minimized_path, best_data)
      record["minimized_path"] = (
          self.store._relative_path(minimized_path))  # pylint: disable=protected-access
      record["best_path"] = record["minimized_path"]
    self.store._record_minimization(sample_id, record)  # pylint: disable=protected-access
    return self._result(sample_id, sample, record, started_monotonic)

  def _result(self, sample_id: str, sample: dict, record: dict,
              started_monotonic: float) -> MinimizeResult:
    return MinimizeResult(
        sample_id=sample_id,
        group_id=self.store._read_state()["sample_to_group"][sample_id],  # pylint: disable=protected-access
        original_size=record["original_size"],
        minimized_size=record["minimized_size"] or sample["size"],
        minimized_path=record.get("minimized_path"),
        stop_reason=record["stop_reason"],
        verified=bool(record.get("verified")),
        step_count=len(record["steps"]),
        elapsed_seconds=round(
            self.store._monotonic() - started_monotonic, 3),  # pylint: disable=protected-access
        record=deepcopy(record),
    )


def _print_group_table(groups: list[dict]) -> None:
  if not groups:
    print("no crash groups in the requested window")
    return
  print("group_id  count  window  first_seen                 representative")
  for group in groups:
    representative = group.get("representative") or {}
    print(
        f"{group['id']}  {group['total_count']:5d}  "
        f"{group['window_count']:6d}  {group['first_seen']:<25} "
        f"{representative.get('path', '-')} ({representative.get('size', 0)} bytes)")


def _target_runner(args) -> TargetRunner:
  target = list(args.target)
  if target and target[0] == "--":
    target = target[1:]
  if not target:
    raise ValueError("a target command is required after `--`")
  return TargetRunner(
      target,
      timeout=args.run_timeout,
      cwd=args.cwd)


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
      prog="python -m atheris.crash",
      description="Deduplicate, persist, and minimize Atheris crash artifacts.")
  subparsers = parser.add_subparsers(dest="command", required=True)

  triage_parser = subparsers.add_parser(
      "triage", help="replay crash artifacts and add them to a store")
  triage_parser.add_argument("--store", required=True)
  triage_parser.add_argument("--artifacts", nargs="+", required=True)
  triage_parser.add_argument("--run-timeout", type=float, default=10.0)
  triage_parser.add_argument("--cwd")
  triage_parser.add_argument(
      "--minimize", action="store_true",
      help="minimize newly ingested samples after replaying them")
  triage_parser.add_argument("--max-steps", type=int, default=500)
  triage_parser.add_argument(
      "--minimize-time-budget", type=float, default=60.0)
  triage_parser.add_argument("--verify-repetitions", type=int, default=2)
  triage_parser.add_argument(
      "target", nargs=argparse.REMAINDER,
      help="target command; use {input} for the input path")

  groups_parser = subparsers.add_parser(
      "groups", help="show groups in a time window")
  groups_parser.add_argument("--store", required=True)
  groups_parser.add_argument("--since")
  groups_parser.add_argument("--until")
  groups_parser.add_argument("--json", action="store_true")
  groups_parser.add_argument("--samples", action="store_true")

  minimize_parser = subparsers.add_parser(
      "minimize", help="minimize a stored crash sample")
  minimize_parser.add_argument("--store", required=True)
  minimize_parser.add_argument("--sample", required=True)
  minimize_parser.add_argument("--max-steps", type=int, default=500)
  minimize_parser.add_argument("--time-budget", type=float, default=60.0)
  minimize_parser.add_argument("--run-timeout", type=float, default=10.0)
  minimize_parser.add_argument("--verify-repetitions", type=int, default=2)
  minimize_parser.add_argument("--cwd")
  minimize_parser.add_argument("--json", action="store_true")
  minimize_parser.add_argument(
      "target", nargs=argparse.REMAINDER,
      help="target command; use {input} for the input path")
  return parser


def main(argv: Sequence[str] | None = None) -> int:
  args = _build_parser().parse_args(argv)
  store = CrashStore(args.store)

  if args.command == "groups":
    groups = store.groups(
        args.since, args.until, include_samples=args.samples)
    if args.json:
      print(json.dumps(groups, ensure_ascii=False, indent=2, sort_keys=True))
    else:
      _print_group_table(groups)
    return 0

  runner = _target_runner(args)
  try:
    if args.command == "triage":
      results = store.ingest_paths(args.artifacts, runner)
      group_ids = {result["group_id"] for result in results}
      unique_samples = {result["sample_id"] for result in results}
      print(
          f"processed={len(results)} unique_samples={len(unique_samples)} "
          f"groups={len(group_ids)}")
      for result in results:
        print(
            f"{result['sample_id'][:16]} -> group {result['group_id']} "
            f"(count={result['group']['count']}, "
            f"representative={result['group']['representative']['path']})")
        if args.minimize and not result["duplicate"]:
          minimizer = CrashMinimizer(
              store,
              runner,
              max_steps=args.max_steps,
              time_budget=args.minimize_time_budget,
              run_timeout=args.run_timeout,
              verify_repetitions=args.verify_repetitions)
          minimized = minimizer.minimize(result["sample_id"])
          print(
              f"  minimized {minimized.original_size} -> "
              f"{minimized.minimized_size} bytes "
              f"(verified={minimized.verified}, "
              f"stop_reason={minimized.stop_reason})")
    else:
      minimizer = CrashMinimizer(
          store,
          runner,
          max_steps=args.max_steps,
          time_budget=args.time_budget,
          run_timeout=args.run_timeout,
          verify_repetitions=args.verify_repetitions)
      result = minimizer.minimize(args.sample)
      if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
      else:
        print(
            f"sample={result.sample_id} group={result.group_id} "
            f"{result.original_size} -> {result.minimized_size} bytes, "
            f"steps={result.step_count}, verified={result.verified}, "
            f"stop_reason={result.stop_reason}")
        print(f"minimized={result.minimized_path}")
  finally:
    runner.close()
  return 0


if __name__ == "__main__":
  sys.exit(main())
