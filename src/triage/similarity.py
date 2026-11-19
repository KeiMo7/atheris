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
"""Similarity families: exact duplicates and near-duplicate clustering.

Relations supported (the shorter content is always named first):
  * exact equality            -> a single ContentNode (handled at scan time)
  * strict prefix:  short == long[:len(short)]
  * strict suffix:  short == long[-len(short):]
  * single-byte expansion: long is short with exactly one byte inserted,
    replaced or appended (size difference in {0, 1}; for equal size the two
    contents must differ at exactly one position)

The relation graph is built from *candidate* pairs only: contents can only be
prefix/suffix related when their prefix/suffix fingerprints match, and can
only be single-byte expansions when their sizes differ by at most one.
Candidates are then verified by bounded streaming reads, so no pair's full
contents are ever held in memory at once.
"""

import collections
import os
from typing import Dict, Iterable, List, Optional, Tuple

from .hashing import DEFAULT_CHUNK_SIZE
from .records import ContentNode, Family

# Prefix/suffix buckets larger than this are pathological (e.g. huge numbers of
# files sharing an empty prefix). Pairwise verification inside them is skipped
# and the condition is reported; nothing is merged on a fingerprint alone.
DEFAULT_MAX_BUCKET_PAIRS = 20000
# Overall cap on retained candidate pairs. Beyond this, remaining buckets are
# skipped and the condition is reported; candidate generation never holds more
# than this many tuples, bounding memory on pathological corpora.
DEFAULT_MAX_CANDIDATE_PAIRS = 2_000_000
# Overall cap on bytes read while verifying candidate relations. Hitting it is
# conservative: remaining pairs are left unmerged rather than guessed at.
DEFAULT_MAX_VERIFY_BYTES = 1 << 34  # 16 GiB


class _UnionFind:
  """Union-find keyed by content digest, deterministic iteration."""

  def __init__(self, keys: Iterable[str]):
    self._parent = {k: k for k in keys}

  def find(self, key: str) -> str:
    root = key
    while self._parent[root] != root:
      root = self._parent[root]
    while self._parent[key] != root:
      self._parent[key], key = root, self._parent[key]
    return root

  def union(self, a: str, b: str) -> None:
    ra, rb = self.find(a), self.find(b)
    if ra != rb:
      # Always attach the lexicographically larger root under the smaller one
      # so component structure does not depend on union order.
      if ra < rb:
        self._parent[rb] = ra
      else:
        self._parent[ra] = rb


def _record_path(node: ContentNode) -> str:
  """Choose the physical file used for streaming reads of this node.

  All copies share identical bytes, so any would do; the lexicographically
  smallest (source_id, relative_path) pair is chosen to make the choice
  explainable and reproducible.
  """
  cached = getattr(node, "_record_path", None)
  if cached is not None:
    return cached
  rec = min(node.records, key=lambda r: (r.source_id, r.relative_path))
  node._record_path = rec.absolute_path  # type: ignore[attr-defined]
  return rec.absolute_path


def _stream_compare(path_short: str, path_long: str, mode: str,
                    chunk_size: int, size_short: int) -> bool:
  """Verify a strict-prefix or strict-suffix relation by streaming.

  mode is "prefix" or "suffix". Caller guarantees size(long) > size(short).
  """
  offset_long = 0 if mode == "prefix" else None
  try:
    with open(path_short, "rb") as fs, open(path_long, "rb") as fl:
      if offset_long is None:
        fl.seek(0, os.SEEK_END)
        offset_long = fl.tell() - size_short
        fl.seek(offset_long)
      while True:
        a = fs.read(chunk_size)
        if not a:
          return True
        b = fl.read(len(a))
        if a != b:
          return False
  except OSError:
    return False


def _edge_blocks_rule_out_single_byte(na: "ContentNode",
                                      nb: "ContentNode") -> bool:
  """Cheap in-memory pre-check before any streaming verification.

  Two contents in a single-byte relation can differ at only one position.
  Therefore their prefix blocks and suffix blocks may differ by at most the
  block that contains that position. If BOTH edge blocks already differ at
  more than one byte, a single-byte relation is impossible and the pair is
  rejected without opening either file.
  """
  def block_distance(a: bytes, b: bytes) -> int:
    n = min(len(a), len(b))
    return sum(1 for i in range(n) if a[i] != b[i]) + abs(len(a) - len(b))

  pd = block_distance(na.prefix_block, nb.prefix_block)
  if pd > 1:
    sd = block_distance(na.suffix_block, nb.suffix_block)
    if sd > 1:
      return True
  return False


def _edge_blocks_rule_out_extension(short: "ContentNode",
                                    long_: "ContentNode") -> bool:
  """Reject a delta-one extension candidate using both edge blocks.

  When the longer content is the shorter content plus exactly one byte, the
  two prefix blocks and the two suffix blocks can each differ at only the
  single position of the edit. If either edge-block pair has an edit distance
  greater than one, no single-byte extension can relate these contents.
  """
  def edit_distance_at_most_one(a: bytes, b: bytes) -> bool:
    if abs(len(a) - len(b)) > 1:
      return False
    i = j = diffs = 0
    while i < len(a) and j < len(b):
      if a[i] == b[j]:
        i += 1
        j += 1
      else:
        diffs += 1
        if diffs > 1:
          return False
        if len(a) == len(b):
          i += 1
          j += 1
        elif len(a) < len(b):
          j += 1
        else:
          i += 1
    diffs += abs((len(a) - i) - (len(b) - j))
    return diffs <= 1

  # For a one-byte extension, BOTH edge-block pairs must each be at edit
  # distance at most one. Either pair exceeding it proves the contents cannot
  # be related by a single inserted/appended byte.
  return not (
      edit_distance_at_most_one(short.prefix_block, long_.prefix_block) and
      edit_distance_at_most_one(short.suffix_block, long_.suffix_block))


def _stream_single_byte_relation(path_a: str, path_b: str, size_a: int,
                                 size_b: int, chunk_size: int) -> Optional[str]:
  """Check whether two same-or-adjacent-size contents differ by one byte.

  Returns a human-readable relation description when the relation holds, else
  None. Size difference must be 0 (single-byte replacement) or 1 (single-byte
  insertion/append on the larger file). Sizes are passed in to avoid re-stat.
  """
  if abs(size_a - size_b) > 1:
    return None
  # Normalize so that b is the larger-or-equal content.
  if size_a > size_b:
    path_a, path_b = path_b, path_a
    size_a, size_b = size_b, size_a
  try:
    with open(path_a, "rb") as fa, open(path_b, "rb") as fb:
      diffs = 0
      while True:
        ca = fa.read(chunk_size)
        cb = fb.read(chunk_size)
        if not ca and not cb:
          break
        if ca == cb:
          continue
        diffs += 1
        if diffs > 1:
          return None
        # Locate the first differing byte inside the mismatching chunks.
        idx = 0
        while idx < min(len(ca), len(cb)) and ca[idx] == cb[idx]:
          idx += 1
        if size_a == size_b:
          # Replacement: tails after the differing position must be equal.
          if ca[idx + 1:] != cb[idx + 1:]:
            return None
          return ("single-byte replacement at byte offset within file "
                  "(equal-length contents)")
        # Insertion/append case: skip one byte in the larger file and compare
        # the remainder chunk-by-chunk.
        fb.seek(fb.tell() - len(cb) + idx + 1)
        fa.seek(fa.tell() - len(ca) + idx)
        while True:
          ta = fa.read(chunk_size)
          tb = fb.read(chunk_size)
          if ta != tb:
            return None
          if not ta:
            return ("single-byte insertion/append: larger content has one "
                    "extra byte")
  except OSError:
    return None


def _candidate_pairs(
    nodes: List[ContentNode],
    max_bucket_pairs: int,
    max_candidate_pairs: int
) -> Tuple[List[Tuple[str, str, str]], List[str]]:
  """Generate candidate (digest_a, digest_b, bucket_kind) pairs.

  Two independent, near-linear indexes are used instead of a length-prefix
  Cartesian product:

  * Exact edge-block buckets (prefix and suffix): members sharing an identical
    8-byte edge are prefix/suffix or single-byte candidates. A strict prefix
    pair always shares a prefix block (the shorter content's full prefix), so
    this index never misses a genuine relation.
  * Single-mutation edge buckets: each node is additionally indexed under
    every one-byte deletion of its edge blocks. Two blocks at edit distance 1
    therefore collide, covering single-byte replacement/insertion candidates
    whose edge blocks are not identical.
  """
  # A single dict keyed by the canonical (smaller, larger) digest pair. This
  # avoids keeping both a dedup set and a parallel list of the same tuples.
  pair_map: Dict[Tuple[str, str], str] = {}
  warnings = []
  size_of = {n.digest: n.size for n in nodes}
  budget_reached = False

  def add(a: str, b: str, kind: str) -> None:
    nonlocal budget_reached
    if budget_reached or a == b:
      return
    key = (a, b) if a < b else (b, a)
    if key not in pair_map:
      pair_map[key] = kind
      if len(pair_map) >= max_candidate_pairs:
        budget_reached = True
        warnings.append(
            "global candidate budget (%d) reached; remaining relation "
            "candidates left unmerged" % max_candidate_pairs)

  def emit_indexed(buckets: Dict[bytes, List[str]], edge_kind: str) -> None:
    """Emit pairs from an edge-block bucket with size-aware kind labels.

    Equal-size members can only be related by a single-byte replacement
    (kind "size"); different-size members are prefix/suffix (or insertion)
    candidates (kind edge_kind).
    """
    for members in buckets.values():
      members = sorted(set(members))
      if len(members) < 2:
        continue
      n_pairs = len(members) * (len(members) - 1) // 2
      if n_pairs > max_bucket_pairs:
        warnings.append(
            "%s edge bucket with %d members (%d candidate pairs); skipping "
            "pairwise verification to avoid pathological work" %
            (edge_kind, len(members), n_pairs))
        continue
      for i in range(len(members)):
        for j in range(i + 1, len(members)):
          a, b = members[i], members[j]
          if size_of[a] == size_of[b]:
            add(a, b, "size")
          else:
            add(a, b, edge_kind)

  # Exact edge-block matches.
  prefix_buckets: Dict[bytes, List[str]] = collections.defaultdict(list)
  suffix_buckets: Dict[bytes, List[str]] = collections.defaultdict(list)
  for node in nodes:
    prefix_buckets[node.prefix_block].append(node.digest)
    suffix_buckets[node.suffix_block].append(node.digest)
  emit_indexed(prefix_buckets, "prefix")
  emit_indexed(suffix_buckets, "suffix")

  # A byte trie over edge blocks finds every block that is equal to or a
  # strict extension of another block, in O(block length) per lookup. This is
  # what strict-prefix pairs (and strict-suffix pairs via reversed blocks)
  # require without a length-prefix Cartesian product.
  def block_containment_pairs(block_attr: str, reverse: bool,
                              kind: str) -> None:
    # Strict prefix/suffix relations only exist across *different* content
    # sizes: the shorter content's edge block is a prefix of the longer's.
    # Equal-size nodes whose blocks merely share leading bytes are unrelated
    # and are handled (if at all) by the single-mutation index.
    trie: dict = {}
    holders: Dict[bytes, List[Tuple[str, int]]] = collections.defaultdict(list)
    for node in nodes:
      block = getattr(node, block_attr)
      key = block[::-1] if reverse else block
      holders[key].append((node.digest, node.size))
      cursor = trie
      for byte in key:
        cursor = cursor.setdefault(byte, {})
      cursor.setdefault("$", []).append((node.digest, node.size))
    # holders keys are already trie keys (reversed for suffix), so they are
    # used directly here; reversing again would miss every match.
    for key, shorter_entries in holders.items():
      cursor = trie
      for byte in key:
        child = cursor.get(byte)
        if child is None:
          cursor = None
          break
        cursor = child
      if cursor is None:
        continue
      # Collect descendant (longer-block) nodes with their sizes.
      longer_entries: List[Tuple[str, int]] = []
      stack = [cursor]
      while stack:
        cur = stack.pop()
        for edge, child in cur.items():
          if edge == "$":
            longer_entries.extend(child)
          else:
            stack.append(child)
      candidate_count = sum(
          1 for _, ss in shorter_entries for _, ls in longer_entries
          if ls > ss)
      if candidate_count > max_bucket_pairs:
        # Same conservative rule as the flat edge buckets: a bucket this broad
        # is almost always a shared constant prefix across unrelated inputs,
        # so pairwise verification is skipped rather than guessed at.
        warnings.append(
            "%s containment bucket with %d candidate pairs; skipping "
            "pairwise verification to avoid pathological work" %
            (kind, candidate_count))
        continue
      for short_digest, short_size in shorter_entries:
        for long_digest, long_size in longer_entries:
          if long_size > short_size:
            add(short_digest, long_digest, kind)

  block_containment_pairs("prefix_block", False, "prefix")
  block_containment_pairs("suffix_block", True, "suffix")

  # Single-byte replacement candidates via one-byte deletion products of both
  # edge blocks. A shared deletion product is a necessary (not sufficient)
  # condition for an edit-distance-1 pair; streaming verification is always
  # the source of truth, so false collisions are rejected rather than merged.
  # Bucket size is capped via emit_indexed/max_bucket_pairs. This also covers
  # contents shorter than EDGE_BLOCK_SIZE, whose edge blocks are the whole
  # content and therefore never share an exact block after an edit.
  one_mut: Dict[bytes, List[str]] = collections.defaultdict(list)
  for node in nodes:
    emitted_empty = False
    for block in (node.prefix_block, node.suffix_block):
      if not block:
        if not emitted_empty:
          one_mut[b""].append(node.digest)
          emitted_empty = True
        continue
      for pos in range(len(block)):
        one_mut[block[:pos] + block[pos + 1:]].append(node.digest)
  emit_indexed(one_mut, "insertion")

  # Exact edge-block matches with a size difference of one cover edge
  # insertions/appends; interior insertions across sizes differing by one are
  # restricted to nodes already sharing a prefix block.
  block_index: Dict[bytes, List[ContentNode]] = collections.defaultdict(list)
  for node in nodes:
    block_index[node.prefix_block].append(node)
  for group in block_index.values():
    group.sort(key=lambda n: (n.size, n.digest))
    for i, node in enumerate(group):
      for other in group[i + 1:]:
        if other.size - node.size > 1:
          break
        if other.size - node.size == 1:
          add(node.digest, other.digest, "insertion")

  # Deterministic order for downstream processing and reproducible output.
  pairs = [(a, b, pair_map[(a, b)]) for (a, b) in sorted(pair_map)]
  return pairs, warnings


def build_families(
    nodes: List[ContentNode],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_bucket_pairs: int = DEFAULT_MAX_BUCKET_PAIRS,
    max_verify_bytes: int = DEFAULT_MAX_VERIFY_BYTES,
    max_candidate_pairs: int = DEFAULT_MAX_CANDIDATE_PAIRS,
) -> Tuple[List[Family], List[str], dict]:
  """Cluster distinct contents into similarity families.

  Returns (families, warnings, stats). Families are returned in a deterministic
  order and each carries the edges that caused its members to be united.
  """
  nodes = sorted(nodes, key=lambda n: n.digest)
  by_digest = {n.digest: n for n in nodes}
  uf = _UnionFind(n.digest for n in nodes)
  pairs, warnings = _candidate_pairs(
      nodes, max_bucket_pairs, max_candidate_pairs)
  edges: Dict[Tuple[str, str], str] = {}
  bytes_read = 0
  capped = False

  # Path-compressing find is effectively O(1) amortized; a separate root cache
  # that rebuilds on every union is strictly worse when unions are frequent.
  def root_of(digest: str) -> str:
    return uf.find(digest)

  for digest_a, digest_b, kind in pairs:
    if bytes_read > max_verify_bytes:
      capped = True
      break
    # Already in the same family via a previously verified edge; no need to
    # re-open files for another relation between the same two components.
    if root_of(digest_a) == root_of(digest_b):
      continue
    na, nb = by_digest[digest_a], by_digest[digest_b]
    size_a, size_b = na.size, nb.size
    # In-memory rejection before touching the filesystem: a genuine single-
    # byte relation cannot differ at two or more positions in both edges.
    if kind in ("size", "insertion") and _edge_blocks_rule_out_single_byte(
        na, nb):
      continue
    relation = None
    if size_a == size_b:
      # Distinct digests with equal size: only a single-byte replacement can
      # keep the pair in the same candidate bucket legitimately.
      if kind == "size":
        relation = _stream_single_byte_relation(
            _record_path(na), _record_path(nb), size_a, size_b, chunk_size)
    else:
      short, long_ = (na, nb) if size_a < size_b else (nb, na)
      delta = long_.size - short.size
      if kind == "prefix":
        if delta == 1 and _edge_blocks_rule_out_extension(short, long_):
          pass
        elif _stream_compare(
            _record_path(short), _record_path(long_), "prefix", chunk_size,
            short.size):
          relation = ("strict prefix" if delta > 1
                      else "strict prefix (one-byte append)")
      elif kind == "suffix":
        if delta == 1 and _edge_blocks_rule_out_extension(short, long_):
          pass
        elif _stream_compare(
            _record_path(short), _record_path(long_), "suffix", chunk_size,
            short.size):
          relation = ("strict suffix" if delta > 1
                      else "strict suffix (one-byte prefix insertion)")
      elif delta == 1:
        # Interior single-byte insertion not visible as an edge relation.
        relation = _stream_single_byte_relation(
            _record_path(short), _record_path(long_), short.size,
            long_.size, chunk_size)
    if relation:
      # Approximate bytes consumed for the global cap.
      bytes_read += min(size_a, size_b) + 1
      uf.union(digest_a, digest_b)
      edges[(digest_a, digest_b) if digest_a < digest_b
            else (digest_b, digest_a)] = relation

  if capped:
    warnings.append(
        "verification byte budget (%d) exhausted; remaining candidate pairs "
        "left unmerged" % max_verify_bytes)

  # Collect components deterministically.
  components: Dict[str, List[str]] = collections.defaultdict(list)
  for node in nodes:
    components[uf.find(node.digest)].append(node.digest)
  families: List[Family] = []
  for idx, root in enumerate(sorted(components)):
    members = sorted(components[root])
    representative = _choose_representative(members, by_digest)
    family_edges = [
        "%s ~ %s (%s)" % (a, b, edges[(a, b)])
        for (a, b) in sorted(edges)
        if uf.find(a) == root
    ]
    families.append(
        Family(
            family_id="G%d" % idx,
            node_digests=members,
            representative_digest=representative,
            representative_reason=_representative_reason(
                representative, members, by_digest),
            relation_edges=family_edges,
        ))
  families.sort(key=lambda f: f.family_id)
  stats = {
      "candidate_pairs": len(pairs),
      "verified_edges": len(edges),
      "verification_bytes": bytes_read,
      "families": len(families),
  }
  return families, warnings, stats


def _choose_representative(members: List[str],
                           by_digest: Dict[str, ContentNode]) -> str:
  """Smallest content wins; ties (same size, different bytes) broken by digest.

  The shortest member can always regenerate every expansion-based edge inside
  the family and is the cheapest to replay and store.
  """
  return min(members, key=lambda d: (by_digest[d].size, d))


def _representative_reason(representative: str, members: List[str],
                           by_digest: Dict[str, ContentNode]) -> str:
  rep = by_digest[representative]
  shortest = min(members, key=lambda d: (by_digest[d].size, d))
  if len(members) == 1:
    return "only member of the family"
  if representative == shortest:
    return ("smallest content in family (%d bytes); digest tie-break among "
            "equally small members" % rep.size)
  return "fallback selection"
