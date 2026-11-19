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
"""The deterministic replay engine.

Guarantees enforced here:

* The original corpus directory is never modified.  A private temporary
  staging directory holds hardlinks (falling back to copies) of the canonical
  entries, and is always removed on exit -- including by exceptions.
* Replay order is the fixed ordinal order produced by
  :mod:`atheris.replay.corpus`.  Parallel workers receive static ordinal
  shards, so scheduling order can neither reorder inputs nor change the final
  counts (counts merge commutatively anyway).
* A single failing entry is recorded and skipped; the batch continues.
* Coverage collection failure is reported explicitly
  (:class:`CoverageUnavailableError`), never as an empty result.
* Progress checkpoints make an interrupted replay resumable from the exact
  entry it stopped at; the resumed run ends with identical coverage to a
  single uninterrupted run.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import shutil
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional

from ._io import atomic_write_json
from .corpus import CorpusEntry
from .corpus import CorpusFailure
from .corpus import LoadedCorpus
from .corpus import corpus_scope_hash
from .corpus import load_corpus
from .coverage import COVERAGE_SEMANTICS_VERSION
from .coverage import CoverageMap
from .coverage import get_collector
from .snapshot import Snapshot

# Hard upper bound on parallelism.  Oversubscription is rejected explicitly
# instead of spawning an unbounded pile of workers.
MAX_WORKERS_LIMIT = 64
# Checkpoint at least every N entries in addition to every flush request.
_CHECKPOINT_INTERVAL = 16
# Exception messages longer than this are truncated in records; messages can
# contain fragments of input data, so records keep them short and generic.
_MAX_ERROR_MESSAGE_CHARS = 200


class WorkerLimitError(ValueError):
  """Raised when a requested worker count is invalid or exceeds the cap."""


@dataclasses.dataclass(frozen=True)
class EntryFailure:
  """A replay failure attributed to one specific corpus entry."""

  relative_path: str
  content_sha256: str
  ordinal: int
  error_type: str
  error_message: str


@dataclasses.dataclass(frozen=True)
class EntryResult:
  relative_path: str
  content_sha256: str
  ordinal: int
  coverage: CoverageMap


@dataclasses.dataclass
class ReplayConfig:
  """Configuration for a corpus replay.

  Attributes:
    target: callable taking one ``bytes`` argument -- the same shape as a
      libFuzzer ``TestOneInput``.
    max_workers: number of replay workers.  Must satisfy
      ``1 <= max_workers <= MAX_WORKERS_LIMIT``; the default of 1 is purely
      serial.  Any value above the cap is rejected.
    run_id: explicit run identifier stored in snapshots and checkpoints.  A
      random hex id is generated when omitted.
    checkpoint_path: when set, replay progress is checkpointed here and a
      restart with the same path resumes from the first unfinished entry.
    checkpoint_interval: flush the checkpoint after at most this many new
      entries.
    stage_corpus: when True (default), canonical entries are hardlinked (or
      copied as a fallback) into a private temp directory before replay.  The
      original corpus is still never opened for writing.
  """

  target: Callable[[bytes], None]
  max_workers: int = 1
  run_id: Optional[str] = None
  checkpoint_path: Optional[str] = None
  checkpoint_interval: int = _CHECKPOINT_INTERVAL
  stage_corpus: bool = True

  def __post_init__(self):
    if not callable(self.target):
      raise WorkerLimitError("ReplayConfig.target must be callable.")
    if not isinstance(self.max_workers, int):
      raise WorkerLimitError("max_workers must be an int.")
    if self.max_workers < 1:
      raise WorkerLimitError(
          f"max_workers must be >= 1, got {self.max_workers}."
      )
    if self.max_workers > MAX_WORKERS_LIMIT:
      raise WorkerLimitError(
          f"max_workers={self.max_workers} exceeds the hard limit of"
          f" {MAX_WORKERS_LIMIT}; refusing to schedule an unbounded batch."
      )
    if self.checkpoint_interval < 1:
      raise WorkerLimitError("checkpoint_interval must be >= 1.")
    if self.run_id is None:
      self.run_id = uuid.uuid4().hex


@dataclasses.dataclass
class ReplayResult:
  coverage: CoverageMap
  results: List[EntryResult]
  entry_failures: List[EntryFailure]
  corpus_failures: List[CorpusFailure]
  skipped_duplicates: List[str]
  run_id: str
  resumed: bool
  collector_backend: str
  corpus_scope_hash: str
  degraded: bool = False
  degradation_reason: Optional[str] = None

  def to_snapshot(self, loaded: LoadedCorpus) -> Snapshot:
    replayed = len(self.results)
    return Snapshot(
        run_id=self.run_id,
        corpus_scope_hash=self.corpus_scope_hash,
        corpus_entry_count=len(loaded.entries),
        corpus_canonical_count=len(loaded.canonical_entries),
        corpus_replayed_count=replayed,
        coverage=self.coverage,
    )


def _target_fingerprint(target: Callable) -> str:
  """Stable identity of the target code for checkpoint validation.

  Identity is the target's source location plus qualified name -- the same
  information a human uses to confirm "the same target".  Bytecode is
  deliberately excluded: minor interpreter patch-level changes to codegen
  must not invalidate a checkpoint, while moving or renaming the target (or
  pointing the run at a different function) still rejects a stale checkpoint.
  """
  code = getattr(target, "__code__", None)
  hasher = hashlib.sha256()
  if code is not None:
    hasher.update(code.co_filename.encode("utf-8", "replace"))
    hasher.update(getattr(code, "co_qualname", code.co_name).encode(
        "utf-8", "replace"))
    hasher.update(str(code.co_firstlineno).encode("ascii"))
  else:
    hasher.update(repr(target).encode("utf-8", "replace"))
  return hasher.hexdigest()


def _sanitize_message(exc: BaseException) -> str:
  """Returns a bounded, content-free description of an exception.

  Exception text can echo input bytes (which may be sensitive), so only the
  type name and a truncated message are retained; raw input is never logged or
  written to a checkpoint/snapshot.
  """
  message = str(exc).replace("\n", " ")
  if len(message) > _MAX_ERROR_MESSAGE_CHARS:
    message = message[:_MAX_ERROR_MESSAGE_CHARS] + "...<truncated>"
  return message


class _StagingArea:
  """Private temp directory holding the de-duplicated, staged corpus."""

  def __init__(self):
    self._dir = tempfile.mkdtemp(prefix="atheris-replay-")
    self.staged_paths: Dict[str, str] = {}

  @property
  def directory(self) -> str:
    return self._dir

  def stage(self, entry: CorpusEntry) -> str:
    # Flatten into content-named files; names come from the already-validated
    # sha256 hex digest, so no traversal risk and duplicates map to one file.
    dest = os.path.join(self._dir, entry.content_sha256)
    if not os.path.exists(dest):
      try:
        os.link(entry.absolute_path, dest)
      except OSError:
        # Cross-device or unsupported hardlinks: copy instead.
        shutil.copyfile(entry.absolute_path, dest)
    self.staged_paths[entry.relative_path] = dest
    return dest

  def cleanup(self) -> None:
    shutil.rmtree(self._dir, ignore_errors=True)


class _Checkpoint:
  """On-disk progress marker, written atomically."""

  FORMAT_VERSION = "1.0"

  def __init__(self, path: Optional[str]):
    self.path = path
    self.completed: Dict[str, str] = {}  # relative_path -> sha256
    # Coverage accumulated by the completed entries; carried across restarts.
    self.coverage = CoverageMap()
    # Result metadata for completed entries (ordinal, path, hash), so a
    # resumed run can present the whole corpus, not just its remaining shard.
    self.completed_results: List[EntryResult] = []

  def load(self, scope_hash: str, target_fp: str):
    if not self.path or not os.path.exists(self.path):
      return False
    import json  # local import keeps module import light

    with open(self.path, "r", encoding="utf-8") as f:
      data = json.load(f)
    if data.get("format_version") != self.FORMAT_VERSION:
      raise ValueError(
          "Checkpoint format version mismatch; refusing to resume under"
          " unknown semantics."
      )
    if data.get("corpus_scope_hash") != scope_hash:
      raise ValueError(
          "Checkpoint was written for a different corpus scope; refusing to"
          " mix coverage from two corpora."
      )
    if data.get("target_fingerprint") != target_fp:
      raise ValueError(
          "Checkpoint was written for a different target function; refusing"
          " to resume."
      )
    self.completed = dict(data.get("completed", {}))
    self.coverage = CoverageMap(dict(data.get("coverage_counts", {})))
    self.completed_results = [
        EntryResult(
            relative_path=item["relative_path"],
            content_sha256=item["content_sha256"],
            ordinal=int(item["ordinal"]),
            coverage=CoverageMap(),  # per-entry detail is not retained
        )
        for item in data.get("completed_results", [])
    ]
    return True

  def flush(
      self,
      *,
      run_id: str,
      scope_hash: str,
      target_fp: str,
      coverage: Optional[CoverageMap] = None,
  ) -> None:
    if not self.path:
      return
    if coverage is not None:
      self.coverage = coverage
    payload = {
        "format_version": self.FORMAT_VERSION,
        "semantics_version": COVERAGE_SEMANTICS_VERSION,
        "run_id": run_id,
        "corpus_scope_hash": scope_hash,
        "target_fingerprint": target_fp,
        "completed": self.completed,
        # Canonical edge ids -> counts; order-independent and self-describing.
        "coverage_counts": self.coverage.counts,
        "completed_results": [
            {
                "relative_path": r.relative_path,
                "content_sha256": r.content_sha256,
                "ordinal": r.ordinal,
            }
            for r in self.completed_results
        ],
    }
    atomic_write_json(self.path, payload)

  def mark(self, relative_path: str, sha256: str) -> None:
    self.completed[relative_path] = sha256

  def record_result(self, result: "EntryResult") -> None:
    self.completed_results.append(result)


def _run_one(
    target: Callable[[bytes], None],
    entry: CorpusEntry,
    read_from: str,
    collector,
) -> EntryResult:
  """Replays one entry; raises are converted by the caller."""
  with open(read_from, "rb") as f:
    data = f.read()
  with collector.record_entry():
    target(data)
  return EntryResult(
      relative_path=entry.relative_path,
      content_sha256=entry.content_sha256,
      ordinal=entry.ordinal,
      coverage=collector.get_entry_map(),
  )


def replay_corpus(
    corpus_path: str,
    config: ReplayConfig,
) -> ReplayResult:
  """Replays a corpus and returns reproducible coverage statistics.

  Raises:
    CorpusLoadError: if the corpus directory itself is unusable.
    WorkerLimitError: on invalid parallelism configuration.
    CoverageUnavailableError: if coverage collection cannot be set up (the
      caller must treat this as explicit degradation, not empty coverage).
  """
  loaded = load_corpus(corpus_path)
  scope_hash = corpus_scope_hash(loaded.entries)
  target_fp = _target_fingerprint(config.target)

  checkpoint = _Checkpoint(config.checkpoint_path)
  resumed = checkpoint.load(scope_hash, target_fp)

  canonical = loaded.canonical_entries
  skipped_duplicates = [
      e.relative_path for e in loaded.entries if e.duplicate_of is not None
  ]

  # Entries already finished by an earlier run.  Verify content identity so a
  # changed file under the same path is replayed rather than silently skipped.
  pending: List[CorpusEntry] = []
  for entry in canonical:
    recorded_sha = checkpoint.completed.get(entry.relative_path)
    if recorded_sha == entry.content_sha256:
      continue
    pending.append(entry)

  # Results from any previous run are restored so the returned list spans the
  # whole corpus scope, even when this process only replayed the remainder.
  results: List[EntryResult] = list(checkpoint.completed_results)
  entry_failures: List[EntryFailure] = []
  # Start from whatever the checkpoint already accumulated, so a resumed run
  # ends with the same coverage as one uninterrupted run.
  merged = checkpoint.coverage
  merge_lock = threading.Lock()
  checkpoint_lock = threading.Lock()
  since_flush = 0

  staging: Optional[_StagingArea] = None
  if config.stage_corpus:
    staging = _StagingArea()

  def staged_path(entry: CorpusEntry) -> str:
    if staging is not None:
      return staging.stage(entry)
    return entry.absolute_path

  def handle_failure(entry: CorpusEntry, exc: BaseException) -> None:
    # SystemExit from a target must not kill the batch; KeyboardInterrupt is
    # allowed to propagate after the finally-block cleans the staging dir.
    failure = EntryFailure(
        relative_path=entry.relative_path,
        content_sha256=entry.content_sha256,
        ordinal=entry.ordinal,
        error_type=type(exc).__name__,
        error_message=_sanitize_message(exc),
    )
    with merge_lock:
      entry_failures.append(failure)

  def process(entry: CorpusEntry) -> Optional[EntryResult]:
    nonlocal since_flush, merged
    try:
      result = _run_one(
          config.target, entry, staged_path(entry), collector
      )
    except KeyboardInterrupt:
      raise
    except BaseException as exc:  # include SystemExit from the target
      handle_failure(entry, exc)
      return None
    with merge_lock:
      results.append(result)
      checkpoint.mark(entry.relative_path, entry.content_sha256)
      checkpoint.record_result(result)
      # Fold this entry in immediately so the checkpoint always captures the
      # coverage of every entry marked completed.
      merged = CoverageMap.merge(merged, result.coverage)
      since_flush += 1
      should_flush = since_flush >= config.checkpoint_interval
      if should_flush:
        since_flush = 0
    if should_flush:
      with checkpoint_lock:
        checkpoint.flush(
            run_id=config.run_id,
            scope_hash=scope_hash,
            target_fp=target_fp,
            coverage=merged,
        )
    return result

  try:
    # Explicit failure if coverage cannot be collected -- never a silent empty
    # map masquerading as real coverage.
    collector = get_collector()
    with collector:
      try:
        if config.max_workers == 1:
          # Deterministic serial path; process() folds each entry into the
          # shared merged map under merge_lock.
          for entry in pending:
            process(entry)
        else:
          # Static, ordinal-based sharding: worker w owns ordinals w, w+N, ...
          # The shard assignment depends only on ordinal and worker count,
          # never on completion timing, so scheduling cannot drift the replay
          # order.  process() still folds into the shared merged map, which
          # makes the checkpoint complete; the shard maps are returned too for
          # the order-independent final merge.
          shards: List[List[CorpusEntry]] = [
              [] for _ in range(config.max_workers)
          ]
          for position, entry in enumerate(pending):
            shards[position % config.max_workers].append(entry)

          with ThreadPoolExecutor(
              max_workers=config.max_workers,
              thread_name_prefix="atheris-replay",
          ) as pool:
            futures = [
                pool.submit(_drain_shard, shard, process)
                for shard in shards
                if shard
            ]
            # Drain to propagate exceptions; every entry's coverage was
            # already folded into `merged` by process() under merge_lock, so
            # the shard maps must not be merged a second time.
            for future in futures:
              future.result()
      except KeyboardInterrupt:
        # Preserve everything finished before the interruption so a restart
        # continues from the exact entry that was in flight.
        checkpoint.flush(
            run_id=config.run_id,
            scope_hash=scope_hash,
            target_fp=target_fp,
            coverage=merged,
        )
        raise

      checkpoint.flush(
          run_id=config.run_id,
          scope_hash=scope_hash,
          target_fp=target_fp,
          coverage=merged,
      )
  finally:
    if staging is not None:
      # Guarantees no half-open temp files remain for the next run.
      staging.cleanup()

  # Results are presented in canonical ordinal order regardless of which
  # worker produced them, keeping the observable output deterministic.
  results.sort(key=lambda r: r.ordinal)
  entry_failures.sort(key=lambda f: f.ordinal)

  return ReplayResult(
      coverage=merged,
      results=results,
      entry_failures=entry_failures,
      corpus_failures=loaded.failures,
      skipped_duplicates=sorted(skipped_duplicates),
      run_id=config.run_id,
      resumed=resumed,
      collector_backend=collector.backend_name,
      corpus_scope_hash=scope_hash,
  )


def _drain_shard(
    shard: List[CorpusEntry],
    process: Callable[[CorpusEntry], Optional[EntryResult]],
) -> CoverageMap:
  """Runs one static shard to completion and merges within the shard.

  Merging inside the shard means cross-thread sharing of coverage state is
  unnecessary: each thread only touches its own map, and the commutative
  final merge yields identical counts for every scheduling interleaving.
  """
  shard_map = CoverageMap()
  for entry in shard:
    result = process(entry)
    if result is not None:
      shard_map = CoverageMap.merge(shard_map, result.coverage)
  return shard_map
