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
"""Deterministic corpus replay with reproducible coverage snapshots.

This subpackage provides a *replay* path that is deliberately separate from the
coverage-guided fuzzing loop.  It replays a fixed corpus against a fixed target
and produces coverage statistics that are:

* identical across repeated runs with the same corpus and target;
* identical between serial and parallel replay (scheduling order cannot change
  the final counts);
* independent of the number of CPU cores on the machine;
* exportable as versioned, long-term-stable snapshots that can be diffed.

The coverage counting semantics are defined in exactly one place
(:mod:`atheris.replay.coverage`) and shared by collection, aggregation and
export.  Nothing in this package modifies the original corpus: all staging work
happens inside a temporary directory that is removed on exit.
"""

from .coverage import COVERAGE_SEMANTICS_VERSION
from .coverage import CoverageMap
from .coverage import CoverageUnavailableError
from .coverage import SATURATING_COUNTER_MAX
from .coverage import get_collector

from .corpus import CorpusEntry
from .corpus import CorpusLoadError
from .corpus import load_corpus

from .engine import ReplayConfig
from .engine import ReplayResult
from .engine import WorkerLimitError
from .engine import replay_corpus

from .snapshot import Snapshot
from .snapshot import SnapshotDiff
from .snapshot import SnapshotFormatVersionError
from .snapshot import compare_snapshots
from .snapshot import load_snapshot
from .snapshot import save_snapshot

__all__ = [
    "COVERAGE_SEMANTICS_VERSION",
    "SATURATING_COUNTER_MAX",
    "CoverageMap",
    "CoverageUnavailableError",
    "CorpusEntry",
    "CorpusLoadError",
    "ReplayConfig",
    "ReplayResult",
    "WorkerLimitError",
    "Snapshot",
    "SnapshotDiff",
    "SnapshotFormatVersionError",
    "compare_snapshots",
    "get_collector",
    "load_corpus",
    "load_snapshot",
    "replay_corpus",
    "save_snapshot",
]
