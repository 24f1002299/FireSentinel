# Evaluation-only data

This directory is reserved for frozen scoring references and labels. It is not
represented in `firesentinel.config.Settings`, and agent runtime inputs reject
paths beneath it.

The FIRMS ingester writes two deterministic files under `firms/`:

- `firms-event-labels.json` contains only normalized acquisition time, WGS84
  coordinates, confidence, brightness, instrument, and derived event windows.
- `firms-event-labels.audit.json` records source SHA-256 values, row and event
  counts, date range, clustering parameters, normalization statistics, and the
  SHA-256 of the label file.

Generate them from permitted local FIRMS CSV exports with:

```powershell
.\.venv\Scripts\python -m scripts.tasks firms-ingest --source path\to\firms.csv
```

The command ignores all non-permitted CSV columns, rejects malformed non-blank
rows, and will not replace changed output without `--overwrite`.

## Matched benchmark inventory

`benchmark-build` reads `firms/firms-event-labels.json` together with an
evaluation-only `observation-inventory.json`. It needs at least 60 eligible
FIRMS-event windows and 60 FIRMS-excluded candidate controls; otherwise it
fails rather than publishing an undersized benchmark.

Each inventory source is a pinned GOES-18 C07 or C14 object:

```json
{
  "source_id": "c07-001",
  "bucket": "noaa-goes18",
  "object_key": "ABI-L2-CMIPF/.../OR_ABI-L2-CMIPF-M6C07_G18_s...nc",
  "size_bytes": 123456,
  "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
}
```

Each window specifies an anchor time/location, view-zenith angle, usable data
fraction, and exactly four observations: C07 `initial`, C07 `later`, C14
`alternate`, and C07 `baseline`. Observation times must exactly equal their
source object scan starts. The generator requires a 30-minute-to-seven-day
baseline, a 10-to-90-minute later view, and an alternate view within 20
minutes.

```powershell
.\.venv\Scripts\python -m scripts.tasks benchmark-build
```

It writes `benchmark/benchmark-cases.json` and `benchmark/benchmark.audit.json`.
Controls match season, a two-degree region cell, local solar hour, view zenith,
and usable fraction. They are excluded within 50 km and 24 hours of every
FIRMS detection. The audit records the random seed, input hashes, source hashes,
case counts, and the benchmark hash.

## Frozen development, test, and stress splits

After manually inspecting a small deterministic sample, freeze the benchmark
with a reviewer name and a checked-in/local UTF-8 note file:

```powershell
.\.venv\Scripts\python -m scripts.tasks benchmark-freeze `
  --reviewer "reviewer-name" `
  --review-notes path\to\day12-review-notes.txt
```

The command reads the hash-audited benchmark and refuses unsafe splits. It
treats every shared FIRMS event, 2-degree WGS84 cell, and UTC ISO week as one
connected split component, so a neighboring frame cannot move independently
into development, test, or stress. It writes `frozen/` artifacts containing
three manifests, scoring-only test/stress label files, an internal assignment
map, and `frozen-splits.audit.json`. Every artifact has a canonical SHA-256;
the audit includes the manual-review notes/hash, leakage result, and per-split
distributions for season, local hour, view angle, missingness, FIRMS
confidence, and C07/C14 availability.

`test.manifest.json` and `stress.manifest.json` have opaque case IDs and omit
labels, event links, and matched-control links. Their label files are
scoring-only. Future model-selection commands must use the same guarded
development-only boundary; the current guard can be checked with:

```powershell
.\.venv\Scripts\python -m scripts.tasks tune
```

Passing a test/stress manifest or either label file to `tune --manifest` fails
before it reads labels. Re-run a frozen set without `--overwrite`; changed
artifacts require that explicit flag and a new review decision.

## Sealed frozen evaluation

Run the documented three-mode frozen comparison with:

```powershell
.\.venv\Scripts\python -m scripts.tasks frozen-evaluation `
  --evidence-template path\to\evidence-job.json
```

It reads the scoring-only test/stress labels only inside the evaluation module
after verifying every frozen artifact. The sealed output is
`frozen-results/frozen-evaluation.json`; it has input/code/configuration hashes,
per-case results, and test/stress/combined aggregate tables. Repeating the
command verifies and reuses the matching report. `--rerun --overwrite` is the
explicit replacement path.
