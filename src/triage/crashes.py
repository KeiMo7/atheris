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
"""Crash artifact parsing, normalization and grouping.

Grouping uses only *structured* information: the exception type plus a
normalized call chain. File sizes and hash prefixes are never grouping keys.
Two crashes merge exactly when both fields agree; a difference in message text
alone does not split a group.

Normalization removes the volatile parts of a call chain:
  * hexadecimal addresses and pointer-like values,
  * temporary directory prefixes (``/tmp/...``, TMPDIR, build paths),
  * CPython object identifiers (``0x...`` suffixes on repr strings),
while keeping file names, function/qual names and source line numbers, which
are the stable identity of a frame.

Artifacts that cannot be parsed never join an existing group: they form their
own conservative singleton group and are explicitly marked ``unparseable``.
"""

import os
import re
from typing import Dict, List, Optional, Tuple

from .records import FileRecord, Source

# Sidecar report suffixes accepted next to a crash input file.
REPORT_SUFFIXES = (".log", ".txt", ".json", ".report")

# A libFuzzer-style crash file name begins with one of these prefixes. This is
# only used to *discover* crash inputs; the original on-disk names and
# locations are never modified.
CRASH_NAME_PREFIXES = ("crash-", "leak-", "timeout-", "oom-", "slow-unit-")

_HEX_RE = re.compile(r"0x[0-9a-fA-F]+")
_TMP_RE = re.compile(
    r"(?P<prefix>(?:[A-Za-z]:)?(?:/[^\s'\"]*)?(?:tmp|Temp|TEMP|var/folders|"
    r"run/user/\d+|private/var/folders)/[^\s:'\"]*?)/(?P<tail>[^/\s:'\"]+)")
_OBJ_ID_RE = re.compile(r"at 0x[0-9a-fA-F]+")
_FRAME_RE = re.compile(
    r"""^\s*File\s+"(?P<file>[^"]+)",\s*
        line\s+(?P<line>\d+),\s*
        in\s+(?P<func>.+?)\s*$""",
    re.VERBOSE,
)
_EXC_LINE_RE = re.compile(
    r"^(?P<type>[A-Za-z_][\w.]*(?:Error|Exception|Warning|Interrupt|Exit|"
    r"Fault|Failure|Killed|Aborted))\b(?:\s*:\s*(?P<msg>.*))?$")
_SANITIZER_RE = re.compile(
    r"(?P<type>AddressSanitizer|UndefinedBehaviorSanitizer|"
    r"LeakSanitizer|MemorySanitizer|ThreadSanitizer|libFuzzer):\s*(?P<msg>.*)")


def normalize_text(text: str) -> str:
  """Strip volatile substrings from free-form text (paths, addresses, ids)."""
  text = _OBJ_ID_RE.sub("at <OBJID>", text)
  text = _HEX_RE.sub("<ADDR>", text)
  # Collapse temporary paths repeatedly so nested temp prefixes are covered.
  prev = None
  while prev != text:
    prev = text
    text = _TMP_RE.sub(lambda m: "<TMPDIR>/" + m.group("tail"), text)
  return text


def _normalize_frame(file_path: str, line: str, func: str) -> str:
  return '{"file": "%s", "line": %s, "func": "%s"}' % (
      normalize_text(file_path).replace('"', '\\"'),
      line,
      normalize_text(func).strip().replace('"', '\\"'),
  )


def parse_python_traceback(
    text: str,
) -> Tuple[str, List[str], Optional[str]]:
  """Parse a standard Python traceback.

  Returns (exception_type, normalized_frames, message) or None when the text
  does not contain a recognizable Python exception.
  """
  lines = text.splitlines()
  frames: List[str] = []
  exc_type = None
  exc_msg = ""
  for line in lines:
    m = _FRAME_RE.match(line)
    if m:
      frames.append(
          _normalize_frame(m.group("file"), m.group("line"),
                           m.group("func")))
      continue
    em = _EXC_LINE_RE.match(line.strip())
    if em:
      # A terminal exception line is valid with frames (full traceback) or
      # without them (one-line report captured by an external wrapper).
      exc_type = em.group("type").split(".")[-1]
      exc_msg = em.group("msg") or ""
      break
  if exc_type is None:
    return None
  return exc_type, frames, normalize_text(exc_msg)


def parse_sanitizer_report(
    text: str,
) -> Tuple[str, List[str], Optional[str]]:
  """Parse an ASan/UBSan/libFuzzer-style report into the same triple."""
  m = _SANITIZER_RE.search(text)
  if not m:
    return None
  exc_type = m.group("type")
  frames: List[str] = []
  frame_re = re.compile(
      r"^\s*#\d+\s+(?:0x[0-9a-fA-F]+\s+)?(?P<body>.+?)\s*$")
  for line in text.splitlines():
    fm = frame_re.match(line)
    if fm:
      body = normalize_text(fm.group("body"))
      # Keep the symbol portion; drop argument values that may contain data.
      body = re.sub(r"\(.*\)$", "(...)", body)
      frames.append('{"symbol": "%s"}' % body.replace('"', '\\"'))
      if len(frames) >= 64:
        break
  return exc_type, frames, normalize_text(m.group("msg"))


def parse_report_bytes(data: bytes):
  """Parse raw report bytes using a lossless surrogate-safe decoding."""
  text = data.decode("utf-8", errors="replace")
  parsed = parse_python_traceback(text)
  if parsed is not None:
    return parsed, "python-traceback"
  parsed = parse_sanitizer_report(text)
  if parsed is not None:
    return parsed, "sanitizer-report"
  return None, "unparseable"


def looks_like_crash_input(name: str) -> bool:
  base = os.path.basename(name)
  return base.startswith(CRASH_NAME_PREFIXES)


def is_sidecar_report(name: str) -> bool:
  """True for files that carry crash reports rather than crash inputs."""
  base = os.path.basename(name)
  return base.endswith(REPORT_SUFFIXES)


class CrashRecord:
  """One crash trigger plus (optionally) its structured report."""

  def __init__(self, input_record: FileRecord):
    self.input_record = input_record
    self.report_path: Optional[str] = None
    self.exception_type: Optional[str] = None
    self.frames: List[str] = []
    self.message = ""
    self.report_kind = "unparseable"

  @property
  def group_key(self) -> Tuple[str, Tuple[str, ...]]:
    if self.exception_type is None:
      return None
    return self.exception_type, tuple(self.frames)

  @property
  def unparseable(self) -> bool:
    return self.exception_type is None

  def as_member_dict(self) -> dict:
    return {
        "record_id": self.input_record.record_id,
        "source_id": self.input_record.source_id,
        "path": self.input_record.relative_path,
        "digest": self.input_record.digest,
        "report": (os.path.basename(self.report_path)
                   if self.report_path else None),
        "report_kind": self.report_kind,
    }


def _find_sidecar_report(rec: FileRecord) -> Optional[str]:
  for suffix in REPORT_SUFFIXES:
    candidate = rec.absolute_path + suffix
    if os.path.isfile(candidate):
      return candidate
  # Same-stem report: crash-abcd + crash-abcd.log is covered above; also accept
  # a sibling "<name>.<suffix>" regardless of extension ordering.
  directory = os.path.dirname(rec.absolute_path)
  if os.path.isdir(directory):
    base = os.path.basename(rec.absolute_path)
    for entry in sorted(os.listdir(directory)):
      if entry.startswith(base + ".") and entry.endswith(REPORT_SUFFIXES):
        return os.path.join(directory, entry)
  return None


def build_crash_records(crash_files: List[FileRecord],
                        sources: List[Source]) -> List[CrashRecord]:
  """Attach parseable structured information to every crash input."""
  del sources  # roots are already absolute on the records.
  result = []
  for rec in crash_files:
    crash = CrashRecord(rec)
    report_path = _find_sidecar_report(rec)
    if report_path is None:
      result.append(crash)
      continue
    crash.report_path = report_path
    try:
      with open(report_path, "rb") as f:
        data = f.read()
    except OSError:
      # Treat an unreadable report the same as a missing one: the crash stays
      # a conservative singleton rather than being forced into a group.
      result.append(crash)
      continue
    parsed, kind = parse_report_bytes(data)
    crash.report_kind = kind
    if parsed is not None:
      crash.exception_type, crash.frames, crash.message = parsed
    result.append(crash)
  return result


def group_crashes(
    crash_records: List[CrashRecord],
) -> Tuple[List[dict], List[str]]:
  """Group crashes by (exception type, normalized call chain).

  Unparseable crashes and crashes without reports become singletons. Returns
  deterministic group dictionaries and warnings.
  """
  groups: Dict[Tuple[str, Tuple[str, ...]], List[CrashRecord]] = {}
  singletons: List[CrashRecord] = []
  warnings: List[str] = []
  for crash in crash_records:
    key = crash.group_key
    if key is None:
      singletons.append(crash)
      continue
    groups.setdefault(key, []).append(crash)

  result = []
  group_idx = 0
  for key in sorted(groups):
    members = sorted(
        groups[key],
        key=lambda c: (c.input_record.source_id,
                       c.input_record.relative_path))
    exc_type, frames = key
    representative = members[0]
    result.append({
        "crash_group_id": "C%d" % group_idx,
        "exception_type": exc_type,
        "normalized_chain": list(frames),
        "unparseable": False,
        "member_count": len(members),
        "representative": representative.as_member_dict(),
        "members": [c.as_member_dict() for c in members],
        "distinct_messages": sorted({c.message for c in members if c.message}),
        "first_seen": _first_seen(members),
    })
    group_idx += 1

  for crash in sorted(
      singletons,
      key=lambda c: (c.input_record.source_id, c.input_record.relative_path)):
    warnings.append(
        "crash input %s/%s has no parseable structured report; kept as a "
        "conservative singleton" %
        (crash.input_record.source_id, crash.input_record.relative_path))
    result.append({
        "crash_group_id": "C%d" % group_idx,
        "exception_type": crash.exception_type,
        "normalized_chain": list(crash.frames),
        "unparseable": True,
        "member_count": 1,
        "representative": crash.as_member_dict(),
        "members": [crash.as_member_dict()],
        "distinct_messages": [],
        "first_seen": _first_seen([crash]),
    })
    group_idx += 1
  return result, warnings


def _first_seen(members: List[CrashRecord]) -> dict:
  first = min(members,
              key=lambda c: (c.input_record.source_id,
                             c.input_record.relative_path))
  rec = first.input_record
  return {
      "source_id": rec.source_id,
      "path": rec.relative_path,
      "record_id": rec.record_id,
  }
