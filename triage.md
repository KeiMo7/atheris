# Post-run artifact triage (`atheris.triage`)

After a long fuzzing campaign the output directories are dominated by noise:
most corpus inputs are byte-identical or near-identical copies, and the same
crash is recorded over and over under different inputs. `atheris.triage` turns
those directories into a small, reviewable, reproducible set.

It is an **opt-in, post-run** feature. Nothing in a normal fuzzing run changes
unless you invoke it explicitly. Original corpus and crash directories are only
ever read — never moved, renamed or deleted.

## When to use it

Run it after one or more fuzzing sessions finish:

```bash
python -m atheris.triage organize \
    --corpus ./corpus-session-1 \
    --corpus ./corpus-session-2 \
    --crashes ./crashes \
    --output ./triage-output
```

`--corpus` and `--crashes` are repeatable so cross-session overlays are handled
in one run. The package is pure Python and imports without Atheris' compiled
native extension, so it also works in environments where Atheris itself could
not be built.

## What it produces

Everything lands under the single `--output` directory, assembled
transactionally (a failed or interrupted run leaves either the previous
complete output or nothing at all):

```
triage-output/
  triage-manifest.json     # full machine- and human-readable record
  blobs/<aa>/<full-hash>   # content-addressed store, one copy per distinct input
  minimal-corpus/<hash>    # one entry per similarity family representative
  minimal-crashes/<hash>   # one entry per crash group representative
```

`minimal-corpus` and `minimal-crashes` entries are hard links into `blobs/`
(with a copy fallback on filesystems without hard links), so an input that is
both a corpus member and a crash trigger is stored once and cross-referenced,
not duplicated.

### Guarantees

- **Content-only fingerprints.** Identity is a BLAKE2b-256 hash of the bytes.
  Paths, file names, timestamps and run counts never participate, so the same
  corpus produces the same result on every machine.
- **Exact duplicates collapse.** Identical bytes from any session or file name
  become one `ContentNode`; the chosen representative and every physical copy
  are recorded with a human-readable reason.
- **Near-duplicate families.** A strict prefix, a strict suffix and a
  single-byte expansion (replacement, insertion or append) unite contents into
  one similarity family. Candidate pairs are found with near-linear edge-block
  indexes and a byte trie, then confirmed by bounded streaming reads — no pair
  is merged on a fingerprint alone.
- **Structured crash grouping.** Crashes merge only on exception type plus a
  normalized call chain. A difference in message text alone does not split a
  group; a different call chain always does. Hex addresses, temporary paths and
  `at 0x...` object ids are normalized before comparison. Unparseable reports
  become conservative singletons and never join an existing group.
- **Reversible.** Every merged or relocated member is listed in the manifest
  with its source and content digest, and can be reconstructed:

  ```bash
  python -m atheris.triage restore \
      --output ./triage-output \
      --restore-to ./recovered
  ```

- **Bounded memory.** Each file is read once in a streaming pass; only an
  8-byte edge block and a fixed-size tail window are retained per file, so
  individual inputs of tens of megabytes and corpora of hundreds of thousands
  of files never pull full contents into memory. Candidate generation and
  verification have explicit budgets; when a bucket is pathological it is
  skipped (and warned about) rather than guessed at.

## Verifying coverage equivalence

The hard correctness requirement is that reducing the corpus must not lose
coverage. Two complementary mechanisms are provided:

### Coverage-aware minimization (preferred)

Pass the fuzz harness at organize time. Every distinct corpus content is
replayed once and a greedy set-cover selection keeps the smallest set of
contents that reaches the *exact* same coverage units as the full corpus,
preferring each family's representative when several inputs are equivalent.
The representative is always materialized; any additional member needed for
coverage is added alongside it, so the minimal corpus is guaranteed equivalent
(or the manifest records precisely which units could not be preserved):

```bash
python -m atheris.triage organize \
    --corpus ./corpus \
    --output ./triage-output \
    --harness ./my_fuzz_target.py \
    --coverage-root .
```

The result's `coverage_verification` section and the `selected_for_coverage`
member flags document the decision. Without `--harness`, reduction stays
purely content-based and no harness is imported.

### Standalone verification

Replay an original set and an arbitrary minimal set against the same harness
and compare the coverage-unit sets:

```bash
python -m atheris.triage verify-coverage \
    --from-output ./triage-output \
    --harness ./my_fuzz_target.py \
    --coverage-root .
```

The harness is a normal Python file exposing `TestOneInput(data: bytes)` (use
`--entry` for another name). The default `trace` backend uses CPython line
coverage and needs no native build; `--backend native` uses Atheris'
instrumented counters when a build is available. The command exits non-zero if
the two coverage-unit sets differ and reports exactly which units were lost.

A purely content-based representative (e.g. the shortest input) is not
guaranteed to reach every branch, so prefer coverage-aware minimization when
you need the minimal corpus to be replay-equivalent.

## Failures and concurrency

- A file that cannot be read does not stop the run; it is recorded under
  `failures` in the manifest and the process exits with status `2` after
  processing everything else.
- Two processes organizing the same destination detect each other through an
  `flock` lock; the second fails fast instead of overwriting work.
- Output is written to a staging directory and published with a single atomic
  rename, so running out of space or being interrupted mid-write leaves the
  previously published output (and all original directories) intact.
- An existing destination is refused unless `--overwrite` is given, in which
  case replacement is itself atomic.

## Programmatic use

```python
from atheris.triage import TriageOptions, organize

result = organize(TriageOptions(
    corpus_roots=["./corpus"],
    crash_roots=["./crashes"],
    destination="./triage-output",
))
print(result.exit_code, result.manifest["stats"])
```

The manifest's `stats` section only *adds* fields; it never reuses the progress
or statistics output of a live fuzzing run, whose format is unchanged.
