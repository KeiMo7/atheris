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

"""Crash-sample replay and deterministic minimization for Atheris.

This module turns an archived crash sample into verifiable evidence instead of
a process exit code. It provides:

* :func:`replay_sample` - run one archived sample through the *same*
  ``TestOneInput(data: bytes)`` convention used by fuzzing, in an isolated
  interpreter, and return a structured verdict plus a stable crash signature.
* :func:`minimize_sample` - shrink a reproducing sample while keeping the
  signature invariant. Every accepted candidate is independently replayed in a
  fresh process; failure modes fall back to the original sample, never to a
  smaller unverified one.
* :func:`export_sample` / :func:`import_sample` - lossless round-tripping.
* :func:`batch_replay` / :func:`batch_minimize` - batched invocation where one
  failing item never affects the others.

Design constraints honored here:

* The target is always invoked as ``target(bytes)`` - there is no separate
  "minimization-only" code path or input decoder.
* Signatures are built solely from runtime-observable information (exit kind,
  exception type, traceback frames, fatal signal). Sample hashes and file paths
  are never part of a signature.
* Every replay happens in a fresh subprocess: native counters, statistics and
  any module-level mutable state start at a deterministic zero state, and a
  native crash cannot take down the caller.
* The original sample bytes are never mutated in place; all candidates live in
  per-call temporary directories.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
from concurrent import futures
from typing import Callable
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

# How much child stderr to retain on the verdict (diagnostics only).
_STDERR_LIMIT = 4096

# Exit code the worker uses when the target raised a Python exception.
_WORKER_EXCEPTION_EXIT = 1
# Exit code the worker uses when the target could not even be loaded.
_WORKER_ENVIRONMENT_EXIT = 2

_FAULTHANDLER_FRAME_RE = re.compile(r'File ".*", line (\d+) in (\S+)')


class SampleReadError(Exception):
  """Raised when an archived sample cannot be read or is not a regular file."""


@dataclasses.dataclass(frozen=True)
class TargetSpec:
  """Identifies the fuzz target to replay.

  Attributes:
      module: importable module name containing the target.
      qualname: dotted attribute path from the module to the target callable.
      target_search_path: extra directories prepended to ``sys.path`` in the
          replay interpreter so the target module can be imported.
  """

  module: str
  qualname: str
  target_search_path: Tuple[str, ...] = ()

  @staticmethod
  def coerce(
      target: Union["TargetSpec", str, Callable],
      target_search_path: Optional[Sequence[str]] = None) -> "TargetSpec":
    """Builds a TargetSpec from a spec, a ``"module:qualname"`` string, or a
    direct callable."""
    extra = tuple(target_search_path or ())
    if isinstance(target, TargetSpec):
      if target_search_path and not target.target_search_path:
        return dataclasses.replace(target, target_search_path=extra)
      return target
    if isinstance(target, str):
      if ":" not in target:
        raise ValueError(
            "String targets must use the 'module:qualname' convention, got: "
            + target)
      module, qualname = target.split(":", 1)
      return TargetSpec(module, qualname, extra)
    if callable(target):
      module = getattr(target, "__module__", None)
      qualname = getattr(target, "__qualname__", None)
      if not module or not qualname:
        raise ValueError(
            "Cannot derive an importable TargetSpec from this callable; pass "
            "TargetSpec(module, qualname) or a 'module:qualname' string "
            "instead.")
      return TargetSpec(module, qualname, extra)
    raise TypeError(f"Unsupported target type: {type(target)!r}")


@dataclasses.dataclass(frozen=True)
class CrashVerdict:
  """Structured outcome of one replay.

  Attributes:
      triggered: whether the replay exhibited a crash-like event (uncaught
          exception, fatal signal, explicit exit, or timeout).
      kind: coarse category - one of ``none``, ``exception``, ``signal``,
          ``exit``, ``timeout``, ``environment_error``.
      exit_kind: how the child actually terminated - ``none``, ``exception``,
          ``signal``, ``exit``, ``timeout``, ``environment_error``.
      exit_code: the raw process return code (negative means killed by signal
          with that number).
      signal_name: signal name when ``kind == 'signal'``, else None.
      exception_type: fully qualified exception type name, else None.
      frames: runtime-observed traceback frames as ``qualname:lineno`` strings.
      signature: stable crash signature (None when nothing triggered).
      degraded: True when the signature is coarse-grained (fatal signal without
          Python frames, or timeout) and only trigger-vs-not discrimination is
          available.
      message: short diagnostic message (environment errors, timeout, ...).
  """

  triggered: bool
  kind: str
  exit_kind: str
  exit_code: int
  signal_name: Optional[str]
  exception_type: Optional[str]
  frames: Tuple[str, ...]
  signature: Optional[str]
  degraded: bool
  message: Optional[str] = None

  def to_dict(self) -> dict:
    return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ReplayResult:
  """The verdict of a replay plus bookkeeping for callers and reports."""

  verdict: CrashVerdict
  sample_length: int
  duration_seconds: float
  stderr_tail: str

  def to_dict(self) -> dict:
    out = dataclasses.asdict(self)
    return out


@dataclasses.dataclass(frozen=True)
class MinimizeReport:
  """Verifiable outcome of a minimization run.

  The success contract is purely signature preservation: ``status ==
  'success'`` implies ``data`` replays with ``signature`` and is no longer than
  the original. Any other status returns the original bytes (except
  ``interrupted``, which returns the smallest *confirmed* candidate and marks
  ``incomplete``).
  """

  status: str
  reason: Optional[str]
  original_length: int
  final_length: int
  signature: Optional[str]
  degraded: bool
  attempts: int
  discarded_fragments: int
  drift_count: int
  timeout_count: int
  incomplete: bool
  data: bytes
  baseline: ReplayResult
  final_replay: Optional[ReplayResult]

  def to_dict(self) -> dict:
    out = dataclasses.asdict(self)
    # dataclasses.asdict already recurses; bytes survive as bytes, which JSON
    # callers can base64 themselves. Keep the structure explicit.
    return out


def _signal_name(signum: int) -> str:
  try:
    return signal.Signals(signum).name
  except ValueError:
    return f"SIG{signum}"


def _parse_faulthandler_frames(stderr: str) -> Tuple[str, ...]:
  frames = []
  for match in _FAULTHANDLER_FRAME_RE.finditer(stderr):
    frames.append(f"{match.group(2)}:{match.group(1)}")
  return tuple(frames)


def _build_signature(kind: str,
                     exception_type: Optional[str],
                     frames: Sequence[str],
                     signal_name: Optional[str],
                     exit_code: Optional[int] = None) -> Tuple[str, bool]:
  """Returns (signature, degraded)."""
  frame_part = ";".join(frames)
  if kind == "exception":
    return f"exc:{exception_type}|{frame_part}", False
  if kind == "exit":
    return f"exit:SystemExit({exit_code})|{frame_part}", False
  if kind == "signal":
    if frames:
      return f"signal:{signal_name}|{frame_part}", False
    # No Python frames were observable: nothing distinguishes two different
    # native faults beyond the signal number.
    return f"degraded:signal:{signal_name}", True
  if kind == "timeout":
    return "degraded:timeout", True
  # environment_error and none do not yield a crash signature.
  return "", True


def _run_one_replay(target: TargetSpec,
                    sample_path: str,
                    timeout: float,
                    workdir: str) -> ReplayResult:
  """Runs a single replay in a fresh interpreter and parses its verdict."""
  spec_path = os.path.join(workdir, "worker_spec.json")
  result_path = os.path.join(workdir, "verdict.json")

  # The child resolves imports the same way this interpreter does, plus any
  # explicit target search paths. Deduplicated, order-preserving.
  child_sys_path = tuple(
      dict.fromkeys(tuple(target.target_search_path) + tuple(sys.path)))
  spec = {
      "module": target.module,
      "qualname": target.qualname,
      "sys_path": [p for p in child_sys_path if p],
      "sample_path": sample_path,
      "result_path": result_path,
  }
  with open(spec_path, "w", encoding="utf-8") as f:
    json.dump(spec, f)

  # `python -m atheris.repro_worker` must be able to import atheris before it
  # can read the spec, so the package location travels through PYTHONPATH rather
  # than through the spec. The private tempdir cwd prevents stdlib shadowing.
  import atheris  # local import: keeps module import light for non-users
  atheris_parent = os.path.dirname(os.path.dirname(os.path.abspath(
      atheris.__file__)))
  path_parts = [
      *(target.target_search_path),
      atheris_parent,
      os.environ.get("PYTHONPATH", ""),
  ]
  env = os.environ.copy()
  env["PYTHONPATH"] = os.pathsep.join(p for p in path_parts if p)
  # Do not pick up user site customization or bytecode caches.
  env["PYTHONNOUSERSITE"] = "1"
  env["PYTHONDONTWRITEBYTECODE"] = "1"

  argv = [sys.executable, "-m", "atheris.repro_worker", spec_path]
  start = __import__("time").monotonic()
  try:
    proc = subprocess.Popen(
        argv,
        cwd=workdir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
  except OSError as exc:
    duration = __import__("time").monotonic() - start
    verdict = CrashVerdict(
        triggered=False,
        kind="environment_error",
        exit_kind="environment_error",
        exit_code=-1,
        signal_name=None,
        exception_type=None,
        frames=(),
        signature=None,
        degraded=True,
        message=f"failed to start replay interpreter: {exc!r}",
    )
    return ReplayResult(verdict, -1, duration, "")

  try:
    stdout, stderr = proc.communicate(timeout=timeout)
    timed_out = False
  except subprocess.TimeoutExpired:
    timed_out = True
    try:
      os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
      pass
    stdout, stderr = proc.communicate()

  duration = __import__("time").monotonic() - start
  stderr_text = stderr.decode("utf-8", "backslashreplace")
  stderr_tail = stderr_text[-_STDERR_LIMIT:]
  returncode = proc.returncode

  envelope = None
  if not timed_out and os.path.exists(result_path):
    try:
      with open(result_path, "r", encoding="utf-8") as f:
        envelope = json.load(f)
    except (OSError, ValueError):
      envelope = None

  sample_length = -1
  try:
    sample_length = os.path.getsize(sample_path)
  except OSError:
    pass

  if timed_out:
    sig, degraded = _build_signature("timeout", None, (), None)
    verdict = CrashVerdict(
        triggered=True,
        kind="timeout",
        exit_kind="timeout",
        exit_code=returncode if returncode is not None else -signal.SIGKILL,
        signal_name=None,
        exception_type=None,
        frames=(),
        signature=sig,
        degraded=degraded,
        message=f"replay exceeded {timeout}s and was killed",
    )
    return ReplayResult(verdict, sample_length, duration, stderr_tail)

  if envelope is not None and envelope.get("stage") == "environment_error":
    verdict = CrashVerdict(
        triggered=False,
        kind="environment_error",
        exit_kind="environment_error",
        exit_code=returncode,
        signal_name=None,
        exception_type=None,
        frames=tuple(envelope.get("frames", ())),
        signature=None,
        degraded=True,
        message=envelope.get("error", "target initialization failed"),
    )
    return ReplayResult(verdict, sample_length, duration, stderr_tail)

  if returncode is not None and returncode < 0:
    # Killed by a fatal signal. The interpreter is gone, so faulthandler's
    # stderr trace is the only runtime-observable frame information.
    signum = -returncode
    sname = _signal_name(signum)
    frames = _parse_faulthandler_frames(stderr_text)
    sig, degraded = _build_signature("signal", None, frames, sname)
    verdict = CrashVerdict(
        triggered=True,
        kind="signal",
        exit_kind="signal",
        exit_code=returncode,
        signal_name=sname,
        exception_type=None,
        frames=frames,
        signature=sig,
        degraded=degraded,
        message=f"child killed by {sname}",
    )
    return ReplayResult(verdict, sample_length, duration, stderr_tail)

  if envelope is None:
    # No signal, no envelope: treat as an environment-level failure rather
    # than inventing a crash signature.
    verdict = CrashVerdict(
        triggered=False,
        kind="environment_error",
        exit_kind="environment_error",
        exit_code=returncode if returncode is not None else -1,
        signal_name=None,
        exception_type=None,
        frames=(),
        signature=None,
        degraded=True,
        message="replay produced no verdict envelope",
    )
    return ReplayResult(verdict, sample_length, duration, stderr_tail)

  kind = envelope["exit_kind"]
  if kind == "none":
    verdict = CrashVerdict(
        triggered=False,
        kind="none",
        exit_kind="none",
        exit_code=returncode,
        signal_name=None,
        exception_type=None,
        frames=(),
        signature=None,
        degraded=False,
    )
    return ReplayResult(verdict, sample_length, duration, stderr_tail)

  frames = tuple(envelope.get("frames", ()))
  if kind == "exception":
    sig, degraded = _build_signature(
        "exception", envelope.get("exception_type"), frames, None)
  elif kind == "exit":
    sig, degraded = _build_signature(
        "exit",
        envelope.get("exception_type"),
        frames,
        None,
        exit_code=envelope.get("exit_code"),
    )
  else:  # pragma: no cover - defensive, worker kinds are enumerated above.
    sig, degraded = "", True

  verdict = CrashVerdict(
      triggered=True,
      kind=kind,
      exit_kind=kind,
      exit_code=returncode,
      signal_name=None,
      exception_type=envelope.get("exception_type"),
      frames=frames,
      signature=sig,
      degraded=degraded,
      message=envelope.get("exception_message"),
  )
  return ReplayResult(verdict, sample_length, duration, stderr_tail)


def replay_sample(
    target: Union[TargetSpec, str, Callable],
    data: bytes,
    *,
    timeout: float = 5.0,
    target_search_path: Optional[Sequence[str]] = None,
    confirmations: int = 1) -> ReplayResult:
  """Replays one archived sample without starting a fuzzing loop.

  Args:
      target: the fuzz target, specified the same way as in fuzzing (a
          callable taking one ``bytes`` argument), a ``"module:qualname"``
          string, or a :class:`TargetSpec`.
      data: raw sample bytes; passed to the target unchanged.
      timeout: per-replay wall-clock limit in seconds. Timeouts count as a
          failure-class verdict, never as a hang.
      target_search_path: directories added to the replay interpreter's
          ``sys.path`` so the target module can be imported.
      confirmations: number of independent replays (each in a fresh process).
          All must agree; the last result is returned.

  Returns:
      A :class:`ReplayResult` containing the structured verdict and signature.
  """
  if not isinstance(data, (bytes, bytearray)):
    raise TypeError("sample data must be bytes")
  if confirmations < 1:
    raise ValueError("confirmations must be >= 1")
  spec = TargetSpec.coerce(target, target_search_path)
  data = bytes(data)

  last = None
  with tempfile.TemporaryDirectory(prefix="atheris_replay_") as workdir:
    sample_path = os.path.join(workdir, "sample.bin")
    with open(sample_path, "wb") as f:
      f.write(data)
    for _ in range(confirmations):
      last = _run_one_replay(spec, sample_path, timeout, workdir)
      if last.verdict.kind == "environment_error":
        # Re-running cannot fix an initialization failure; stop early.
        break
  return last


def export_sample(path: str, data: bytes) -> None:
  """Writes sample bytes verbatim. The parent directory must already exist."""
  if not isinstance(data, (bytes, bytearray)):
    raise TypeError("sample data must be bytes")
  with open(path, "wb") as f:
    f.write(bytes(data))


def import_sample(path: str) -> bytes:
  """Reads an archived sample verbatim.

  Raises:
      SampleReadError: if the path is missing, unreadable, or not a regular
          file. An empty regular file is a valid (zero-byte) sample.
  """
  if not os.path.exists(path):
    raise SampleReadError(f"sample does not exist: {path}")
  if not os.path.isfile(path):
    raise SampleReadError(f"sample is not a regular file: {path}")
  try:
    with open(path, "rb") as f:
      return f.read()
  except OSError as exc:
    raise SampleReadError(f"failed to read sample {path}: {exc!r}") from exc


class _Minimizer:
  """Stateful, deterministic, signature-preserving delta debugger."""

  def __init__(self,
               target: TargetSpec,
               data: bytes,
               signature: str,
               timeout: float,
               max_attempts: int,
               confirmations: int,
               stop_event: Optional[threading.Event]):
    self._target = target
    self._signature = signature
    self._timeout = timeout
    self._max_attempts = max_attempts
    self._confirmations = confirmations
    self._stop_event = stop_event

    self._workdir = tempfile.TemporaryDirectory(prefix="atheris_min_")
    self._sample_path = os.path.join(self._workdir.name, "candidate.bin")
    self.attempts = 0
    self.discarded_fragments = 0
    self.drift_count = 0
    self.timeout_count = 0
    self.exhausted = False
    self.interrupted = False
    self.best = data

  def close(self) -> None:
    self._workdir.cleanup()

  def _candidate_keeps_signature(self, candidate: bytes) -> bool:
    """Replays a candidate in fresh processes; True iff every replay matches
    the baseline signature."""
    if self._stop_event is not None and self._stop_event.is_set():
      self.interrupted = True
      return False
    if self.attempts >= self._max_attempts:
      self.exhausted = True
      return False

    self.attempts += 1
    with open(self._sample_path, "wb") as f:
      f.write(candidate)

    for _ in range(self._confirmations):
      result = _run_one_replay(
          self._target, self._sample_path, self._timeout, self._workdir.name)
      verdict = result.verdict
      if verdict.kind == "timeout":
        self.timeout_count += 1
        return False
      if not verdict.triggered:
        return False
      if verdict.signature != self._signature:
        self.drift_count += 1
        return False
    return True

  def _accept(self, candidate: bytes) -> bool:
    # Hard monotonicity guard: never accept a longer candidate.
    if len(candidate) > len(self.best):
      raise AssertionError("minimizer generated a growing candidate")
    if self._candidate_keeps_signature(candidate):
      if len(candidate) < len(self.best):
        self.discarded_fragments += 1
      self.best = candidate
      return True
    return False

  def run(self) -> None:
    self._truncate_tail()
    if self.interrupted or self.exhausted:
      return
    self._ddmin()
    if self.interrupted or self.exhausted:
      return
    self._substitute_bytes()

  def _truncate_tail(self) -> None:
    """Binary-search the shortest reproducing prefix."""
    lo, hi = 0, len(self.best)
    while lo < hi:
      if self.interrupted or self.exhausted:
        return
      mid = (lo + hi) // 2
      candidate = self.best[:mid]
      if self._accept(candidate):
        hi = mid
      else:
        if self.interrupted or self.exhausted:
          return
        lo = mid + 1
    self.best = self.best[:lo]

  def _ddmin(self) -> None:
    """Classic ddmin: remove one of n chunks at a time, refining
    granularity."""
    n = 2
    while len(self.best) >= 2 and n <= len(self.best):
      if self.interrupted or self.exhausted:
        return
      chunk_size = len(self.best) // n
      if chunk_size < 1:
        break
      chunks = [
          self.best[i * chunk_size:(i + 1) * chunk_size] for i in range(n)
      ]
      # Keep any remainder bytes attached to the final chunk.
      tail = self.best[n * chunk_size:]
      if tail:
        chunks[-1] += tail

      progress = False
      for i in range(n):
        if self.interrupted or self.exhausted:
          return
        candidate = b"".join(c for j, c in enumerate(chunks) if j != i)
        if self._accept(candidate):
          progress = True
          n = max(2, n - 1)
          break
      if not progress:
        if n >= len(self.best):
          break
        n = min(len(self.best), n * 2)

  def _substitute_bytes(self) -> None:
    """Replace each byte with fixed sentinels; only signature-keeping
    replacements are accepted. Length is unchanged throughout."""
    for i in range(len(self.best)):
      if self.interrupted or self.exhausted:
        return
      for value in (0x00, 0xFF):
        if self.best[i] == value:
          continue
        candidate = self.best[:i] + bytes((value,)) + self.best[i + 1:]
        if self._accept(candidate):
          break  # position simplified, move on with the new best


def minimize_sample(
    target: Union[TargetSpec, str, Callable],
    data: bytes,
    *,
    timeout: float = 5.0,
    max_attempts: int = 200,
    confirmations: int = 1,
    target_search_path: Optional[Sequence[str]] = None,
    stop_event: Optional[threading.Event] = None,
    output_path: Optional[str] = None) -> MinimizeReport:
  """Minimizes a crash sample with signature preservation as the sole success
  condition.

  Args:
      target: fuzz target (callable, ``"module:qualname"`` or TargetSpec).
      data: the archived crash sample bytes. Never mutated in place.
      timeout: per-replay wall-clock limit.
      max_attempts: hard cap on replay attempts (resource bound).
      confirmations: independent fresh-process replays each candidate must
          survive with the identical signature.
      target_search_path: extra import directories for the replay interpreter.
      stop_event: when set between replays, minimization stops and the smallest
          confirmed candidate is retained with ``status='interrupted'``.
      output_path: if given, the returned sample bytes are also written there
          (including on fallback and interruption).

  Returns:
      A :class:`MinimizeReport` with original/final lengths, the signature,
      attempt and discard counters, and explicit fallback reasons.
  """
  if not isinstance(data, (bytes, bytearray)):
    raise TypeError("sample data must be bytes")
  if max_attempts < 1:
    raise ValueError("max_attempts must be >= 1")
  spec = TargetSpec.coerce(target, target_search_path)
  original = bytes(data)

  baseline = replay_sample(
      spec, original, timeout=timeout, confirmations=confirmations)
  baseline_verdict = baseline.verdict

  def finish(status,
             reason,
             result_data,
             incomplete,
             minimizer=None,
             final_replay=None):
    attempts = minimizer.attempts if minimizer else 0
    discarded = minimizer.discarded_fragments if minimizer else 0
    drift = minimizer.drift_count if minimizer else 0
    timeouts = minimizer.timeout_count if minimizer else 0
    if output_path is not None:
      export_sample(output_path, result_data)
    return MinimizeReport(
        status=status,
        reason=reason,
        original_length=len(original),
        final_length=len(result_data),
        signature=baseline_verdict.signature,
        degraded=baseline_verdict.degraded,
        attempts=attempts,
        discarded_fragments=discarded,
        drift_count=drift,
        timeout_count=timeouts,
        incomplete=incomplete,
        data=result_data,
        baseline=baseline,
        final_replay=final_replay)

  # Failure/degradation mode 1: target fails before it can process input.
  if baseline_verdict.kind == "environment_error":
    return finish("environment_error",
                  baseline_verdict.message or "environment_error", original,
                  False)

  # Form 8: a sample that does not trigger is reported, never "success, len 0".
  if not baseline_verdict.triggered or baseline_verdict.signature is None:
    return finish("unreproducible", "sample does not trigger the target",
                  original, False)

  minimizer = _Minimizer(
      spec,
      original,
      baseline_verdict.signature,
      timeout=timeout,
      max_attempts=max_attempts,
      confirmations=confirmations,
      stop_event=stop_event)
  try:
    minimizer.run()
    best = minimizer.best

    if minimizer.interrupted:
      # Degradation mode 3: keep the smallest confirmed candidate and mark the
      # run incomplete. `best` is always itself a confirmed candidate.
      final_replay = replay_sample(spec, best, timeout=timeout)
      return finish("interrupted", "stopped before minimization completed",
                    best, True, minimizer, final_replay)

    final_replay = replay_sample(
        spec, best, timeout=timeout, confirmations=confirmations)

    # Independent final verification: never return a smaller unverified sample.
    verified = (final_replay.verdict.triggered and
                final_replay.verdict.signature == baseline_verdict.signature)
    if not verified:
      return finish("fallback", "final verification failed", original, False,
                    minimizer, final_replay)

    if best == original:
      # No reduction survived. Attribute the failure deterministically.
      if minimizer.drift_count:
        reason = "signature_drift"
      elif minimizer.timeout_count:
        reason = "timeout"
      elif minimizer.exhausted:
        reason = "resource_exhausted"
      else:
        reason = "no_reproducing_candidate"
      return finish("fallback", reason, original, False, minimizer,
                    final_replay)

    if minimizer.exhausted:
      # Attempt budget spent mid-search: fall back to the original rather than
      # presenting an unfinished search as a result.
      return finish("fallback", "resource_exhausted", original, False,
                    minimizer, final_replay)

    return finish("success", None, best, False, minimizer, final_replay)
  finally:
    minimizer.close()


def _check_pool_size(max_workers: int) -> int:
  if max_workers < 1:
    raise ValueError("max_workers must be >= 1")
  return max_workers


def batch_replay(
    target: Union[TargetSpec, str, Callable],
    samples: Sequence[bytes],
    *,
    timeout: float = 5.0,
    max_workers: int = 1,
    target_search_path: Optional[Sequence[str]] = None
) -> Tuple[Optional[ReplayResult], ...]:
  """Replays multiple samples independently.

  Each sample gets its own temporary directory and interpreter state. A failure
  in one item is captured as an environment-error verdict and never affects the
  others. Results are returned in input order.
  """
  _check_pool_size(max_workers)
  spec = TargetSpec.coerce(target, target_search_path)
  results = [None] * len(samples)

  def work(index, sample):
    try:
      return index, replay_sample(spec, sample, timeout=timeout)
    except BaseException as exc:  # isolate per-item failures
      verdict = CrashVerdict(
          triggered=False,
          kind="environment_error",
          exit_kind="environment_error",
          exit_code=-1,
          signal_name=None,
          exception_type=type(exc).__qualname__,
          frames=(),
          signature=None,
          degraded=True,
          message=repr(exc),
      )
      return index, ReplayResult(verdict, len(sample) if hasattr(
          sample, "__len__") else -1, 0.0, "")

  with futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
    for index, result in pool.map(lambda i: work(i, samples[i]),
                                  range(len(samples))):
      results[index] = result
  return tuple(results)


def batch_minimize(
    target: Union[TargetSpec, str, Callable],
    samples: Sequence[bytes],
    *,
    timeout: float = 5.0,
    max_attempts: int = 200,
    max_workers: int = 1,
    target_search_path: Optional[Sequence[str]] = None
) -> Tuple[Optional[MinimizeReport], ...]:
  """Minimizes multiple samples independently; one failure never pollutes the
  others. Results are returned in input order."""
  _check_pool_size(max_workers)
  spec = TargetSpec.coerce(target, target_search_path)
  reports = [None] * len(samples)

  def work(index, sample):
    try:
      return index, minimize_sample(
          spec, sample, timeout=timeout, max_attempts=max_attempts)
    except BaseException as exc:
      verdict = CrashVerdict(
          triggered=False,
          kind="environment_error",
          exit_kind="environment_error",
          exit_code=-1,
          signal_name=None,
          exception_type=type(exc).__qualname__,
          frames=(),
          signature=None,
          degraded=True,
          message=repr(exc),
      )
      placeholder = ReplayResult(verdict, len(sample), 0.0, "")
      return index, MinimizeReport(
          status="environment_error",
          reason=repr(exc),
          original_length=len(sample),
          final_length=len(sample),
          signature=None,
          degraded=True,
          attempts=0,
          discarded_fragments=0,
          drift_count=0,
          timeout_count=0,
          incomplete=False,
          data=sample,
          baseline=placeholder,
          final_replay=None)

  with futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
    for index, report in pool.map(lambda i: work(i, samples[i]),
                                  range(len(samples))):
      reports[index] = report
  return tuple(reports)
