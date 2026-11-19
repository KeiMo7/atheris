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
"""End-to-end triage orchestration and reversible restoration."""

import dataclasses
import os
from typing import List, Optional

from . import crashes as crashes_mod
from . import scanner, similarity, store
from .hashing import HASH_ALGORITHM
from .records import ContentNode, FileRecord

MANIFEST_VERSION = 1
MINIMAL_CORPUS_DIR = "minimal-corpus"
MINIMAL_CRASHES_DIR = "minimal-crashes"
RECOVERED_DIR = "recovered"


@dataclasses.dataclass
class TriageOptions:
  """User-facing knobs for :func:`organize`."""

  corpus_roots: List[str]
  crash_roots: List[Optional[str]] = None
  destination: str = "./triage-output"
  overwrite: bool = False
  # When a harness is supplied, the minimal corpus is made coverage-equivalent
  # by replaying every distinct corpus content and selecting the smallest set
  # of family members that reaches the full original coverage. Without it the
  # reduction is purely content-based and equivalence is verified separately
  # with the verify-coverage command.
  harness_path: Optional[str] = None
  harness_entry: str = "TestOneInput"
  coverage_backend: str = "trace"
  coverage_roots: Optional[List[str]] = None


@dataclasses.dataclass
class TriageResult:
  destination: str
  manifest: dict
  exit_code: int
  warnings: List[str]

  @property
  def equivalent_coverage(self) -> Optional[bool]:
    return self.manifest.get("coverage_verification", {}).get("equivalent")


def _first_source_from(records: List[FileRecord]) -> dict:
  rec = min(records, key=lambda r: (r.source_id, r.relative_path))
  return {
      "source_id": rec.source_id,
      "path": rec.relative_path,
      "record_id": rec.record_id,
  }


def _node_first_source(node: ContentNode) -> dict:
  return _first_source_from(node.records)


def _node_member_entries_from(records: List[FileRecord]) -> List[dict]:
  return [{
      "record_id": r.record_id,
      "source_id": r.source_id,
      "path": r.relative_path,
      "size": r.size,
  } for r in records]


def _node_member_entries(node: ContentNode) -> List[dict]:
  return _node_member_entries_from(node.records)


def organize(options: TriageOptions) -> TriageResult:
  """Run the full triage pipeline and publish a minimal reviewable set."""
  warnings: List[str] = []
  sources = scanner.make_sources(options.corpus_roots, options.crash_roots)

  with store.destination_lock(options.destination):
    records, nodes, failures = scanner.scan(sources)
    by_digest = {n.digest: n for n in nodes}

    crash_source_ids = {
        s.source_id for s in sources if s.kind == "crashes"
    }
    corpus_source_ids = {
        s.source_id for s in sources if s.kind == "corpus"
    }

    crash_records = [
        r for r in records
        if r.readable and r.source_id in crash_source_ids
        and crashes_mod.looks_like_crash_input(r.relative_path)
        and not crashes_mod.is_sidecar_report(r.relative_path)
    ]
    crash_records = sorted(
        crash_records, key=lambda r: (r.source_id, r.relative_path))
    crash_records_obj = crashes_mod.build_crash_records(crash_records, sources)
    crash_groups, crash_warnings = crashes_mod.group_crashes(crash_records_obj)
    warnings.extend(crash_warnings)

    # Nodes that appear under a corpus source, and crash-input digest set for
    # cross referencing. A digest can exist in both worlds.
    corpus_digests = {
        r.digest
        for r in records
        if r.readable and r.source_id in corpus_source_ids
    }
    crash_digests = {r.digest for r in crash_records}

    # Similarity families describe the corpus only. Crash inputs are grouped
    # separately by structured exception identity; shared digests are linked
    # through cross references rather than merged into the minimal corpus.
    corpus_nodes = [
        n for n in nodes
        if any(r.source_id in corpus_source_ids for r in n.records)
    ]
    families, fam_warnings, fam_stats = similarity.build_families(
        corpus_nodes)
    warnings.extend(fam_warnings)

    # Optional coverage-aware minimization. Each distinct corpus content is
    # replayed once; the smallest set of contents that reaches the full
    # original coverage is selected, preferring family representatives.
    coverage_verification = None
    selected_digests = None
    if options.harness_path is not None:
      from . import replay as replay_mod
      labeled_paths = []
      for node in corpus_nodes:
        source_file = min(
            node.records, key=lambda r: (r.source_id, r.relative_path)
        ).absolute_path
        labeled_paths.append((node.digest, source_file))
      coverage_by_digest, target_units, cov_stats = (
          replay_mod.collect_per_input_coverage(
              labeled_paths,
              options.harness_path,
              entry_name=options.harness_entry,
              backend=options.coverage_backend,
              coverage_roots=options.coverage_roots))
      # Prefer each family's representative first, then members in digest order.
      preferred = []
      for family in families:
        preferred.append(family.representative_digest)
        preferred.extend(
            d for d in family.node_digests
            if d != family.representative_digest)
      selected_digests = set(
          replay_mod.coverage_preserving_selection(
              coverage_by_digest, target_units, preferred))
      reached = frozenset().union(
          *(coverage_by_digest[d] for d in selected_digests)) if selected_digests else frozenset()
      coverage_verification = {
          "backend": options.coverage_backend,
          "equivalent": reached == target_units,
          "units_original": len(target_units),
          "units_minimal": len(reached),
          "units_missing_in_minimal": sorted(target_units - reached),
          "distinct_corpus_contents_replayed": cov_stats["inputs_replayed"],
          "replay_read_errors": cov_stats["input_errors"],
          "contents_selected_for_coverage": len(selected_digests),
      }
      if not coverage_verification["equivalent"]:
        warnings.append(
            "coverage-aware minimization could not preserve every unit; "
            "see coverage_verification.units_missing_in_minimal")

    with store.StagedOutput(options.destination,
                            options.overwrite) as out:
      # 1. Blob store: one physical copy of each distinct readable content.
      for node in nodes:
        source_file = min(
            node.records, key=lambda r: (r.source_id, r.relative_path)
        ).absolute_path
        out.add_blob(source_file, node.digest)

      # 2. Minimal corpus. Without a harness this is one entry per family
      # representative. With a harness, additional family members needed to
      # preserve coverage are materialized alongside the representative.
      family_entries = []
      minimal_corpus_files = []
      for family in families:
        rep_digest = family.representative_digest
        rep_node = by_digest[rep_digest]
        rep_first = _node_first_source(rep_node)
        rel_out = "%s/%s" % (MINIMAL_CORPUS_DIR, rep_digest)
        minimal_corpus_files.append((rep_node, rel_out))
        all_members = []
        for digest in family.node_digests:
          member_node = by_digest[digest]
          # Corpus-family membership lists corpus physical copies only; the
          # same digest in a crash directory is reported via cross refs.
          corpus_copies = [
              r for r in member_node.records
              if r.source_id in corpus_source_ids
          ]
          coverage_selected = (
              selected_digests is not None and digest in selected_digests)
          all_members.append({
              "digest": digest,
              "size": member_node.size,
              "physical_copies": _node_member_entries_from(corpus_copies),
              "first_seen": _first_source_from(corpus_copies),
              "is_representative": digest == rep_digest,
              "selected_for_coverage": coverage_selected,
          })
          # A non-representative member required to keep coverage equivalent.
          if coverage_selected and digest != rep_digest:
            extra_rel = "%s/%s" % (MINIMAL_CORPUS_DIR, digest)
            minimal_corpus_files.append((member_node, extra_rel))
        family_entries.append({
            "family_id": family.family_id,
            "representative_digest": rep_digest,
            "representative_path": rel_out,
            "representative_selection": family.representative_reason,
            "first_seen": rep_first,
            "distinct_contents": len(family.node_digests),
            "physical_member_count": sum(
                len(m["physical_copies"]) for m in all_members),
            "also_triggers_crash": rep_digest in crash_digests,
            "relations": family.relation_edges,
            "members": all_members,
        })

      # 3. Minimal crash set: one entry per crash group representative.
      crash_entries = []
      for group in crash_groups:
        rep_digest = group["representative"]["digest"]
        rel_out = "%s/%s" % (MINIMAL_CRASHES_DIR, rep_digest)
        in_corpus = rep_digest in corpus_digests
        crash_entries.append({
            "crash_group_id": group["crash_group_id"],
            "exception_type": group["exception_type"],
            "normalized_chain": group["normalized_chain"],
            "unparseable": group["unparseable"],
            "member_count": group["member_count"],
            "representative_path": rel_out,
            "representative": group["representative"],
            "members": group["members"],
            "distinct_messages": group["distinct_messages"],
            "first_seen": group["first_seen"],
            "present_in_corpus": in_corpus,
            "reproduction": {
                "input_blob": "blobs/%s/%s" % (rep_digest[:2], rep_digest),
                "minimal_crash_path": rel_out,
                "original_crash_path":
                    group["representative"]["path"],
                "source_id": group["representative"]["source_id"],
                "note": ("replay with the same harness entry point used for "
                         "the original run; the blob content is byte-identical "
                         "to the original crash input"),
            },
        })

      # 4. Cross references (digest present in both corpus and crashes).
      shared = sorted(corpus_digests & crash_digests)
      cross_references = [{
          "digest": digest,
          "corpus_blob": "blobs/%s/%s" % (digest[:2], digest),
          "crash_blob": "blobs/%s/%s" % (digest[:2], digest),
          "stored_once": True,
          "corpus_sources": sorted({
              (r.source_id, r.relative_path)
              for r in by_digest[digest].records
              if r.source_id in corpus_source_ids
          }),
          "crash_sources": sorted({
              (r.source_id, r.relative_path)
              for r in by_digest[digest].records
              if r.source_id in crash_source_ids
          }),
      } for digest in shared]

      # 5. Materialize the minimal trees as hard links into the blob store.
      for node, rel_out in minimal_corpus_files:
        out.add_blob(
            min(node.records,
                key=lambda r: (r.source_id, r.relative_path)).absolute_path,
            node.digest)
        _link_blob_into_staging(out, node.digest, rel_out)
      for entry in crash_entries:
        digest = entry["representative"]["digest"]
        _link_blob_into_staging(out, digest,
                                entry["representative_path"])

      manifest = {
          "manifest_version": MANIFEST_VERSION,
          "hash_algorithm": HASH_ALGORITHM,
          "determinism": {
              "fingerprint_inputs": ["content bytes only"],
              "excluded_from_fingerprint":
                  ["path", "timestamps", "run count", "file name"],
              "scan_order": "sorted source order, sorted relative paths",
          },
          "sources": [dataclasses.asdict(s) for s in sources],
          "stats": _build_stats(records, corpus_nodes, nodes, family_entries,
                                crash_groups, failures, fam_stats,
                                minimal_corpus_files),
          "coverage_verification": coverage_verification,
          "corpus_families": family_entries,
          "crash_groups": crash_entries,
          "cross_references": cross_references,
          "failures": failures,
          "warnings": warnings,
          "reversibility": {
              "principle": ("original directories are never modified; every "
                            "merged or relocated member is listed with its "
                            "source and digest and can be reconstructed via "
                            "restore()"),
              "blob_store": "blobs/<first2>/<full-digest>",
          },
      }
      out.write_manifest(manifest)
      out.commit()

  exit_code = 0 if not failures else 2
  return TriageResult(
      destination=os.path.abspath(options.destination),
      manifest=manifest,
      exit_code=exit_code,
      warnings=warnings,
  )


def _link_blob_into_staging(out: store.StagedOutput, digest: str,
                            relative: str) -> None:
  target = os.path.join(out.staging, relative)
  os.makedirs(os.path.dirname(target), exist_ok=True)
  if os.path.exists(target):
    return
  blob = out.blob_path(digest)
  try:
    os.link(blob, target)
  except OSError:
    import shutil
    shutil.copyfile(blob, target)


def _build_stats(records, corpus_nodes, all_nodes, family_entries,
                 crash_groups, failures, fam_stats,
                 minimal_corpus_files) -> dict:
  readable = [r for r in records if r.readable]
  reduced_corpus = sum(1 for f in family_entries
                       if f["physical_member_count"] > 1
                       or f["distinct_contents"] > 1)
  corpus_physical_copies = sum(
      len([r for r in node.records]) for node in corpus_nodes)
  return {
      "physical_files_seen": len(records),
      "readable_files": len(readable),
      "read_failures": len(failures),
      "distinct_contents": len(all_nodes),
      "exact_duplicate_copies_collapsed":
          corpus_physical_copies - len(corpus_nodes),
      "similarity_families": len(family_entries),
      "families_with_merges": reduced_corpus,
      # One representative per family; extra entries appear only when a harness
      # is provided and coverage preservation requires them.
      "minimal_corpus_size": len(family_entries),
      "materialized_minimal_corpus_size": len(minimal_corpus_files),
      "crash_inputs": sum(g["member_count"] for g in crash_groups),
      "crash_groups": len(crash_groups),
      "similarity": fam_stats,
  }


def restore(destination: str, restore_root: str) -> str:
  """Reconstruct every original input from representatives + manifest.

  For each recorded physical file, its content is copied from the blob store
  back to ``<restore_root>/<source_label>/<original_relative_path>``. The
  output is a faithful byte-for-byte reconstruction; the original directories
  remain untouched.
  """
  manifest = store.load_manifest(destination)
  if manifest is None:
    raise store.TriageError("no manifest found in %s" % destination)
  os.makedirs(restore_root, exist_ok=True)
  restored = 0
  sources = {s["source_id"]: s for s in manifest["sources"]}
  for family in manifest["corpus_families"]:
    for member in family["members"]:
      for copy in member["physical_copies"]:
        _restore_one(destination, restore_root, sources, member["digest"],
                     copy)
        restored += 1
  for group in manifest["crash_groups"]:
    for member in group["members"]:
      _restore_one(destination, restore_root, sources, member["digest"], {
          "source_id": member["source_id"],
          "path": member["path"],
      })
      restored += 1
  return restore_root


def _restore_one(destination: str, restore_root: str, sources: dict,
                 digest: str, copy: dict) -> None:
  source = sources[copy["source_id"]]
  target = os.path.join(restore_root, RECOVERED_DIR, source["label"],
                        copy["path"].replace("/", os.sep))
  os.makedirs(os.path.dirname(target), exist_ok=True)
  blob = os.path.join(destination, "blobs", digest[:2], digest)
  with open(blob, "rb") as src, open(target, "wb") as dst:
    import shutil
    shutil.copyfileobj(src, dst, length=1 << 20)
