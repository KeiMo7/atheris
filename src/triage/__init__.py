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
"""Post-run artifact triage: turn fuzzer output into a reviewable minimal set.

This package is pure Python and does not import Atheris' native extension, so
it remains usable in environments where Atheris could not be compiled. It is
invoked explicitly (``python -m atheris.triage ...``); nothing in the normal
fuzzing entry point changes behaviour unless this package is used.
"""

from .organize import TriageOptions, TriageResult, organize

__all__ = ["TriageOptions", "TriageResult", "organize"]
