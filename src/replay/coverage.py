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
"""The single definition of replay coverage semantics.

Hard constraint (see the replay design): the coverage counting rule must be
defined exactly once and shared by collection, aggregation and export.  This
module is that definition.

Counting rule
-------------

Each *edge* is identified by a canonical location string,
``"<canonical-path>:<lineno>:<qualname>"``.  A replay of one corpus entry
produces a per-entry hit set: every edge executed at least once by that entry
is recorded with an 8-bit saturating count, i.e. each execution increments the
counter up to :data:`SATURATING_COUNTER_MAX` and never wraps.  This mirrors the
8-bit ``unsigned char`` counters used by the native atheris/libFuzzer path
(``counters.cc``).

Aggregation of per-entry maps is commutative and associative: for each edge
the saturating counts are added with saturation.  Therefore the merged result
does not depend on replay order, on the number of workers, or on which worker
handled which entry -- which is what makes serial and parallel replay produce
identical counts.

Edge identity is derived solely from the executed code object (file, line,
qualified function name) -- never from process addresses, counter allocation
order, thread ids or machine core count -- so the same target and corpus yield
the same edge ids on every machine.
"""

from __future__ import annotations

import os
import sys
import threading
from typing import Dict, Optional

# Bumped only when the meaning of an edge or of a count changes.  Recorded in
# every snapshot so old snapshots can be detected instead of silently re-read
# under new semantics.
COVERAGE_SEMANTICS_VERSION = "1.0"

# 8-bit saturating counters, matching the native instrumentation counters.
SATURATING_COUNTER_MAX = 255

# A free tool id for sys.monitoring (valid range is 0..5).
_MONITORING_TOOL_ID = 5
_MONITORING_TOOL_NAME = "atheris.replay"


class CoverageUnavailableError(RuntimeError):
  """Raised when no coverage collection backend is available.

  Callers must not treat this as an empty coverage result: degradation is
  reported explicitly, never as a silently-empty map.
  """


def _saturating_add(left: int, right: int) -> int:
  total = left + right
  return total if total <= SATURATING_COUNTER_MAX else SATURATING_COUNTER_MAX


def canonical_location(filename: str, lineno: int, qualname: str) -> str:
  """Returns the canonical, machine-independent id of an executed edge.

  Absolute paths that live under a directory on ``sys.path`` are rewritten to
  a module-style relative path (e.g. ``/x/replay/coverage.py`` imported from
  ``/x`` becomes ``replay/coverage.py``), so moving a source tree to another
  machine does not change edge ids.  Paths outside every ``sys.path`` entry
  (e.g. the standard library) are stored as-is, still keyed by content
  location rather than by any runtime address.
  """
  normalized = os.path.normpath(filename)
  candidates = []
  for base in sys.path:
    if not base:
      continue
    base_norm = os.path.normpath(base)
    prefix = base_norm + os.sep
    # Written via str.startswith(mapped tuple) rather than an "or" of two
    # startswith calls: some 3.14 builds mis-cache the adaptive LOAD_ATTR of
    # the second call and resolve `startswith` as `start`, raising
    # AttributeError at runtime.
    if normalized == base_norm or normalized.startswith((prefix,)):
      candidates.append(os.path.relpath(normalized, base_norm))
  path_part = min(candidates, key=len) if candidates else normalized
  return f"{path_part}:{lineno}:{qualname}"


class CoverageMap:
  """An immutable-ish map of edge id -> saturating count.

  This is the one data structure used by collection (a fresh map per corpus
  entry), aggregation (:meth:`merge`) and export (snapshot serialization).
  """

  __slots__ = ("_counts",)

  def __init__(self, counts: Optional[Dict[str, int]] = None):
    if counts is None:
      self._counts = {}
    else:
      # Defensive copy plus validation; counts always stay in 0..255.
      self._counts = {}
      for edge, count in counts.items():
        if not isinstance(edge, str) or not edge:
          raise ValueError(f"Invalid edge id: {edge!r}")
        if not isinstance(count, int) or not 0 <= count <= SATURATING_COUNTER_MAX:
          raise ValueError(f"Invalid counter value for {edge!r}: {count!r}")
        self._counts[edge] = count

  @staticmethod
  def merge(left: "CoverageMap", right: "CoverageMap") -> "CoverageMap":
    """Order-independent union with saturating per-edge addition."""
    merged = dict(left._counts)
    for edge, count in right._counts.items():
      if edge in merged:
        merged[edge] = _saturating_add(merged[edge], count)
      else:
        merged[edge] = count
    return CoverageMap(merged)

  def add_hit(self, edge: str) -> None:
    """Records one execution of ``edge`` (saturating)."""
    current = self._counts.get(edge, 0)
    if current < SATURATING_COUNTER_MAX:
      self._counts[edge] = current + 1

  @property
  def counts(self) -> Dict[str, int]:
    # Return a copy so callers cannot mutate the internal accounting.
    return dict(self._counts)

  @property
  def total_edges(self) -> int:
    """Number of distinct edges hit -- the headline coverage number."""
    return len(self._counts)

  @property
  def total_hits(self) -> int:
    """Sum of all saturating counters."""
    return sum(self._counts.values())

  def ordered_items(self):
    """All (edge, count) pairs sorted by edge id.

    Sorting by the canonical string makes exported indices independent of the
    order in which edges happened to be observed.
    """
    return sorted(self._counts.items())

  def __eq__(self, other) -> bool:
    return isinstance(other, CoverageMap) and self._counts == other._counts

  def __bool__(self) -> bool:
    return bool(self._counts)

  def __repr__(self) -> str:
    return f"CoverageMap(edges={self.total_edges}, hits={self.total_hits})"


class CoverageCollector:
  """Abstract coverage collection backend.

  A collector is used as a context manager::

      with get_collector() as collector:
          with collector.record_entry():
              target(data)
          entry_map = collector.get_entry_map()

  Implementations must guarantee that :meth:`record_entry` produces a map
  containing only edges executed inside the ``with`` block, and that nested or
  concurrent ``record_entry`` blocks (different threads) stay isolated.
  """

  backend_name = "abstract"

  def __enter__(self) -> "CoverageCollector":
    return self

  def __exit__(self, exc_type, exc, tb) -> None:
    self.shutdown()

  def record_entry(self):  # pragma: no cover - interface only
    raise NotImplementedError

  def get_entry_map(self) -> CoverageMap:  # pragma: no cover
    raise NotImplementedError

  def shutdown(self) -> None:  # pragma: no cover
    raise NotImplementedError


class _MonitoringCollector(CoverageCollector):
  """Collector backed by :pep:`669` ``sys.monitoring`` (Python 3.12+)."""

  backend_name = "sys.monitoring"

  def __init__(self) -> None:
    monitoring = sys.monitoring  # type: ignore[attr-defined]
    self._monitoring = monitoring
    self._tool_id = _MONITORING_TOOL_ID
    # One in-progress map per thread -> parallel replay entries stay isolated.
    self._thread_local = threading.local()
    self._depth = threading.local()
    try:
      monitoring.use_tool_id(self._tool_id, _MONITORING_TOOL_NAME)
    except ValueError:
      # Tool id already registered by us (e.g. previous collector in-process).
      if monitoring.get_tool_info(self._tool_id) != _MONITORING_TOOL_NAME:
        raise
    monitoring.register_callback(
        self._tool_id, monitoring.events.LINE, self._on_line
    )

  def _current_map(self) -> Optional[CoverageMap]:
    return getattr(self._thread_local, "map", None)

  def _on_line(self, code, line_number):
    cov_map = self._current_map()
    if cov_map is not None:
      qualname = getattr(code, "co_qualname", code.co_name)
      cov_map.add_hit(canonical_location(code.co_filename, line_number,
                                         qualname))
    # Returning None means "do not disable the event for this location".
    return None

  def record_entry(self):
    collector = self

    class _Record:

      def __enter__(self):
        depth = getattr(collector._depth, "value", 0)
        if depth == 0:
          collector._thread_local.map = CoverageMap()
        collector._depth.value = depth + 1
        # LINE events fire only while the tool is enabled for this thread.
        collector._monitoring.set_events(
            collector._tool_id, collector._monitoring.events.LINE
        )
        return self

      def __exit__(self, exc_type, exc, tb):
        depth = collector._depth.value - 1
        collector._depth.value = depth
        if depth == 0:
          collector._monitoring.set_events(collector._tool_id, 0)
        return False

    return _Record()

  def get_entry_map(self) -> CoverageMap:
    cov_map = self._current_map()
    if cov_map is None:
      return CoverageMap()
    return cov_map

  def shutdown(self) -> None:
    try:
      self._monitoring.set_events(self._tool_id, 0)
      self._monitoring.register_callback(
          self._tool_id, self._monitoring.events.LINE, None
      )
      self._monitoring.free_tool_id(self._tool_id)
    except (ValueError, AttributeError):
      # Best-effort cleanup; never mask a real failure during shutdown.
      pass


class _TraceCollector(CoverageCollector):
  """Fallback collector backed by :func:`sys.settrace`.

  Used when ``sys.monitoring`` is unavailable (Python < 3.12).  Thread-local
  trace functions give the same per-thread isolation as the monitoring backend.
  """

  backend_name = "sys.settrace"

  def __init__(self) -> None:
    self._thread_local = threading.local()
    self._depth = threading.local()

  def _tracer(self, frame, event, arg):
    if event == "line":
      cov_map = getattr(self._thread_local, "map", None)
      if cov_map is not None:
        code = frame.f_code
        qualname = getattr(code, "co_qualname", code.co_name)
        cov_map.add_hit(canonical_location(code.co_filename, frame.f_lineno,
                                           qualname))
    return self._tracer

  def record_entry(self):
    collector = self

    class _Record:

      def __enter__(self):
        depth = getattr(collector._depth, "value", 0)
        if depth == 0:
          collector._thread_local.map = CoverageMap()
          sys.settrace(collector._tracer)
        collector._depth.value = depth + 1
        return self

      def __exit__(self, exc_type, exc, tb):
        depth = collector._depth.value - 1
        collector._depth.value = depth
        if depth == 0:
          sys.settrace(None)
        return False

    return _Record()

  def get_entry_map(self) -> CoverageMap:
    cov_map = getattr(self._thread_local, "map", None)
    return cov_map if cov_map is not None else CoverageMap()

  def shutdown(self) -> None:
    sys.settrace(None)


def get_collector() -> CoverageCollector:
  """Returns a working coverage collector or raises explicitly.

  Degradation is never silent: if no tracing backend exists this raises
  :class:`CoverageUnavailableError` rather than returning a collector that
  would produce empty results.
  """
  if hasattr(sys, "monitoring"):
    try:
      return _MonitoringCollector()
    except (ValueError, TypeError) as exc:
      raise CoverageUnavailableError(
          "sys.monitoring is present but could not be configured for coverage"
          " collection; refusing to replay without coverage."
      ) from exc
  if hasattr(sys, "settrace"):
    return _TraceCollector()
  raise CoverageUnavailableError(
      "Neither sys.monitoring nor sys.settrace is available on this Python"
      " runtime; coverage collection is unavailable and replay cannot produce"
      " trustworthy statistics."
  )
