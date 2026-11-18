# Crash Deduplication and Corpus Management

Atheris keeps its existing `Setup()`/`Fuzz()` behavior and command-line
semantics unless the optional crash-management commands below are used. The
manager replays crash artifacts in subprocesses, so features never depend on
the crashed process's Python state.

## Crash triage

Replay one or more libFuzzer crash artifacts into a durable store:

```bash
python -m atheris.crash triage \
  --store ./crash-store \
  --artifacts ./crashes \
  -- python ./my_fuzzer.py '{input}' -atheris_runs=1
```

`{input}` is replaced with the replay input path. If it is omitted, the path is
appended to the command. The exit status of the target determines whether the
sample crashed; stdout and stderr are retained as crash-scene evidence.

Add `--minimize` to minimize each newly ingested sample immediately:

```bash
python -m atheris.crash triage \
  --store ./crash-store \
  --artifacts ./crashes \
  --minimize \
  --max-steps 500 \
  --minimize-time-budget 60 \
  --run-timeout 10 \
  --verify-repetitions 2 \
  -- python ./my_fuzzer.py '{input}' -atheris_runs=1
```

### Grouping features

The grouping key is computed only from the input bytes and observable crash
evidence:

- Python crash: exception type plus the last eight normalized traceback frames
  (`file:line:function`). Exception messages and addresses are excluded so
  variable payloads do not split one root cause.
- Sanitizer/native crash: sanitizer or signal type, bug class (for example
  `heap-buffer-overflow`), and up to eight normalized native frames. Addresses
  are replaced with `0xADDR`.
- Timeout: timeout status plus the available normalized stack.
- Stack extraction failure: the sample receives a unique
  `ungrouped:<sha256>` key and can never be merged with another sample.

The SHA-256 of the input is also the sample identity. Re-scanning the same
artifact after a restart is idempotent and does not increment its group again.
Different inputs with the same key increment the group count. The representative
sample is always the smallest current sample or its verified minimized output.

## Viewing groups

```bash
python -m atheris.crash groups --store ./crash-store
python -m atheris.crash groups \
  --store ./crash-store \
  --since 2026-11-18T00:00:00Z \
  --until 2026-11-19T00:00:00Z \
  --json
```

The time window is half-open (`since <= added_at < until`). Output includes
the total group count, count inside the window, first/last occurrence, and the
representative sample. This makes roots introduced after a change easy to
identify.

## Minimization

Minimize an already stored sample:

```bash
python -m atheris.crash minimize \
  --store ./crash-store \
  --sample <sample-sha256> \
  --max-steps 500 \
  --time-budget 60 \
  --run-timeout 10 \
  --verify-repetitions 2 \
  -- python ./my_fuzzer.py '{input}' -atheris_runs=1
```

The minimizer uses byte-level delta debugging and accepts a candidate only if
its crash type and normalized stack key match the original crash. When stack
extraction failed, the original exit code/signal and crash type must match.
Before and after minimization, the target is replayed to verify the result.

Budgets:

- `--max-steps`: maximum candidate target executions. The initial reproduction
  is recorded as step 0 and is not charged to this budget.
- `--time-budget`: wall-clock seconds for candidate executions and verification.
- `--run-timeout`: per-replay process timeout.
- `--verify-repetitions`: number of successful replays required for the final
  result.

Every execution is persisted with its phase, input size, exit code, signal,
timeout flag, status, resulting signature, acceptance decision, reason, and
duration. If the step or time budget is exhausted, the last accepted input is
kept. A candidate timeout or crash-type change is rejected and recorded; on
verification failure the manager rolls back to the previous accepted result.

## Corpus export and import

```bash
python -m atheris.corpus export \
  --output ./corpus.jsonl \
  ./corpus ./extra-corpus

python -m atheris.corpus import \
  --archive ./corpus.jsonl \
  --output ./restored-corpus
```

The archive is UTF-8 JSON Lines:

```json
{"format":"atheris-corpus-jsonl","version":1}
{"name":"nested/input-1","size":4,"sha256":"...","data":"base64..."}
```

Each entry is independent and contains its relative path, uncompressed size,
SHA-256, and base64 data. During import, malformed JSON, invalid base64,
truncated entries, size/checksum mismatches, and unsafe paths are skipped while
the remaining entries are loaded. Imported files are written atomically.

## Store layout and recovery

```text
crash-store/
  .lock                 advisory inter-process lock
  state.json            versioned group/event metadata
  crashes/<sha256>      original sample bytes
  minimized/<sha256>    accepted minimized bytes
```

Sample bytes and state snapshots are written to a temporary file in the
destination directory, fsynced, and published with `os.replace`. A killed
writer therefore leaves either the previous complete file or the new complete
file. Group updates are serialized with `fcntl.flock`, so restarted or parallel
manager processes continue accumulating the same store.

## Python API

```python
from atheris.crash import CrashMinimizer, CrashStore, TargetRunner

runner = TargetRunner(
    ["python", "my_fuzzer.py", "{input}", "-atheris_runs=1"],
    timeout=10)
store = CrashStore("./crash-store")
result = store.ingest_file("./crashes/crash-abc", runner)

minimizer = CrashMinimizer(store, runner, max_steps=500, time_budget=60)
minimized = minimizer.minimize(result["sample_id"])
```

`atheris.corpus.export_corpus()` and `atheris.corpus.import_corpus()` provide
the equivalent API for corpus migration.

## Degraded behavior

- No extractable call stack: a unique group is created for that sample.
- Candidate timeout: the candidate is not accepted; the previous best remains.
- Crash type/stack changed: the candidate is rejected and the reason is stored.
- Verification failure: rollback to the previous accepted input.
- Budget exhausted: partial best result and complete step history remain stored.
- Damaged corpus entry: only that entry is skipped.

## Acceptance example

The multi-root test target in `src/crash_test.py` intentionally raises
`RuntimeError`, raises `ValueError`, and exits without a stack. Replaying seven
artifacts produced the following grouping result:

| crash type | samples before | groups | representative | rep. size |
| --- | ---: | ---: | --- | ---: |
| `RuntimeError` | 3 | 1 | `crashes/96a7b57b...c4647b` | 107 bytes |
| `ValueError` | 2 | 1 | `crashes/22ec5232...ca3c1ec` | 107 bytes |
| `exit_7`, no stack | 1 | 1 | `crashes/9e4d5f38...c2e624` | 10 bytes |
| `exit_7`, no stack | 1 | 1 | `crashes/f69106ff...17ce3a` | 10 bytes |

Thus 7 raw artifacts became 4 groups. The two no-stack samples remained
separate as required.

For a 4007-byte `RuntimeError` input containing large constant prefixes and
suffixes, minimization produced the 7-byte input `CRASH_A` (`43524153485f41`)
in 71 recorded executions. The result was replayed twice with the same
normalized signature and exited with the same Python crash. Minimization
reported `no_further_reduction`.

For corpus migration, the test exported two entries, damaged one entry's
base64 payload, and imported the archive. One valid entry was restored and one
damaged entry was skipped without aborting the import.
