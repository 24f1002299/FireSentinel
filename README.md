# FireSentinel

FireSentinel is a local, reproducible thermal-evidence review workflow. It
includes explicit configuration, JSON logs, a deterministic OpenCV runtime
smoke test, a bounded reviewer outcome, and a Streamlit evidence reviewer.

## Setup

This lock targets Windows x86-64 and CPython 3.13.7 (see `.python-version`). A
fresh checkout requires no `.env` file, services, data download, or global
tools.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python -m pip install --require-virtualenv -r requirements.lock
```

## Commands

Run every project action through the portable task runner:

```powershell
.\.venv\Scripts\python -m scripts.tasks test
.\.venv\Scripts\python -m scripts.tasks format
.\.venv\Scripts\python -m scripts.tasks format-check
.\.venv\Scripts\python -m scripts.tasks lint
.\.venv\Scripts\python -m scripts.tasks typecheck
```

Selected source downloads are controlled only by the checked-in JSON manifest.
Each verified source is stored once under `data/source-cache/` by its SHA-256;
later requests validate and reuse the cached bytes. The initial empty manifest
makes this safe to run before a source is chosen:

```powershell
.\.venv\Scripts\python -m scripts.tasks download
.\.venv\Scripts\python -m scripts.tasks cache-inspect
.\.venv\Scripts\python -m scripts.tasks cache-clean-case --case-id pine-creek
.\.venv\Scripts\python -m scripts.tasks replay
.\.venv\Scripts\python -m scripts.tasks evaluate
```

Launch the evidence reviewer at `http://localhost:8501`:

```powershell
.\.venv\Scripts\python -m scripts.tasks ui
```

Use `Ctrl+C` to stop Streamlit. Remove only generated caches with:

```powershell
.\.venv\Scripts\python -m scripts.tasks clean
```

To also delete locally generated artifacts, run:

```powershell
.\.venv\Scripts\python -m scripts.tasks clean --artifacts
```

## Evidence reviewer

The reviewer discovers completed `evidence.json` packets under `artifacts/`.
It presents case and location context, a chronological evidence strip,
measurements, source masks and contours, considered/selected bounded actions,
reason codes, resource use, outcome, warnings, and provenance. It summarizes
these fields rather than showing packet JSON.

Use the three deterministic buttons to explain the intended development-only
stories without any source files:

- **Emerging event** shows persistent aligned thermal evidence and sends it to
  reviewer escalation, not a wildfire determination.
- **Matched control** shows a first candidate that does not persist and ends
  with no persistent thermal evidence.
- **Abstention** shows poor coverage and an exhausted observation budget ending
  in insufficient evidence.

The page is a local review aid only. Its thermal evidence and outcomes must
not be used for emergency response, dispatch, or a wildfire conclusion.

## Predictable failure handling

The bounded loop records missing or hash-mismatched cached sources, elapsed
time limits, insufficient disk space, artifact-write failures, unusable
frames, and failed geographic alignment as closed reason/error codes. Cache,
timeout, and artifact failures abstain; they never use a substitute source or
partial output. An unusable or unaligned observation may consume one recorded
allowlisted replacement retry, then the loop abstains if the replacement is
still inadequate.

Every tool error stored in the loop journal includes fixed recovery guidance.
The reviewer also discovers a terminal journal that has no completed evidence
packet, so a reviewer can see its error and recovery action rather than
mistaking an incomplete artifact for evidence. Local evidence packets are
shown only when their completion marker, packet hash, and all declared asset
hashes verify successfully.

## Layout

```text
src/firesentinel/
  core/          typed evidence, trace, and artifact record contracts
  data/          dataset manifest and download code
  vision/        OpenCV pipeline components
  agent/         decision and replay logic
  evaluation/    metrics and evaluation runs
  ui/            Streamlit application
manifests/       checked-in data declarations
scripts/         portable task runner and runtime verification
tests/           smoke and scaffold tests
artifacts/       ignored generated outputs
docs/            project and development documentation
evaluation-data/ frozen evaluation-only scoring references and label audits
```

See [the development workflow](docs/development.md) for configuration details
and direct module commands. The locked OpenCV runtime evidence remains in
`docs/runtime-baseline.json` and `docs/opencv-build-info.txt`.

Evidence packets use strict canonical JSON records; see
[the evidence-record contract](docs/evidence-records.md).

The checked-in synthetic thermal fixture manifest in
`src/firesentinel/vision/` provides seven small, deterministic offline cases
for persistent/transient heat, image shifts, and frame-quality failures. It
uses seed `20260819`, fixed `1e-6` absolute tolerance, and SHA-256 integrity
checks; the test suite verifies both repeated generation and deliberate
corruption handling without downloading data.

GOES-18 discovery is available as the typed `firesentinel.data.goes18` API. It
uses anonymous public S3 catalog requests for only `ABI-L2-CMIPF` Channels 7
and 14, with immutable local hourly catalog snapshots stored under
`data/catalog/`. No AWS credentials are read or required.

The download command accepts pinned `cases`/`sources` manifest entries; see
[the manifest format](manifests/README.md). It records JSON receipts containing
declared source size, transfer bytes, verified checksum, elapsed time, retry
count, and whether the result was a cache hit. Case cleanup removes exactly one
case's references and retains blobs that another case still uses.

## Real-event OpenCV slice

The checked-in Park Fire manifest pins two historical GOES-18 Channel 7 objects
and the reviewed crop and image-processing configuration. Download its sources
once, then regenerate and verify the complete evidence packet without network
access:

```powershell
.\.venv\Scripts\python -m firesentinel.data.download --manifest manifests\park-fire-20240725.json
.\.venv\Scripts\python -m scripts.tasks slice
```

The second command is cached-only. It recreates `evidence.json` and the
annotated `before-after.png` under the case artifact directory, checks the
pinned evidence and image hashes, and prints both contour hashes. See
[the slice notes](docs/real-event-slice.md) for the audit scope and limitations.

## Calibrated GOES crops

`firesentinel.data.goes_crop` converts a verified cached GOES `CMI` object into
a compact calibrated regional artifact. Pass its `DownloadReceipt.cache_path`,
non-wrapping WGS84 bounds, and padding policy; the stage decodes the source
scale/offset itself, masks fill/range/DQF/off-Earth pixels, and clips padding at
the source edge. The resulting deterministic `.npz` contains calibrated data,
invalid mask, DQF, per-pixel latitude/longitude, scan coordinates, timing,
projection/calibration metadata, source hash, and a canonical content checksum.

## Mask-aware OpenCV tiles

`firesentinel.vision.tiles` converts a calibrated crop into physically clipped,
mask-aware-resized analysis values plus a separate robust `uint8` display image.
The original calibration is retained unchanged, invalid output pixels remain
`NaN`/masked and black in displays, and optional CLAHE is emitted only as a
secondary review image. Tile metadata records parameters, masks, ranges,
timings, and the OpenCV build hash. See [tile preparation](docs/tile-preparation.md).

## Observation quality gate

`firesentinel.vision.quality` measures calibrated, mask-aware missingness,
usable coverage, clipping/saturation, contrast span, standard-deviation
texture, and mean adjacent-pixel difference before any anomaly interpretation.
Its checked-in thresholds are restricted to development cases and the offline
synthetic fixtures; runtime quality assessment reads no evaluation labels.
Poor coverage, blank, clipped/saturated, or low-contrast observations receive
explicit reason codes, a zero fire-evidence confidence cap, and have apparent
candidate masks cleared by `apply_quality_gate`.

## Contextual thermal anomalies

`firesentinel.vision.anomalies.extract_contextual_anomalies` consumes aligned,
calibrated Channel 7 and Channel 14 arrays with their invalid masks. It derives
a mask-aware local Channel 7 contrast map and a calibrated C07-minus-C14 map,
requires both thresholds, then uses OpenCV 5 morphology, connected components,
contours, area filtering, and optional edge-distance filtering. Each retained
component records its area, centroid, source-array contrast/difference
measurements, edge proximity, and an annotated overlay; poor-quality channels
clear all candidate regions before they can become fire evidence.

## Temporal persistence

`firesentinel.vision.persistence.measure_temporal_persistence` receives each
candidate mask with calibrated C07 values and its latitude/longitude grids. It
nearest-resamples every available observation onto one geospatial common grid,
then matches only adjacent components that satisfy both centroid-distance and
intersection-over-union limits. The result reports track count, overlap, area
and temperature trend, disappearance, and a bounded confidence. A `None`
observation is an explicit continuity break, never an interpolated look.

## Deterministic evidence jobs

`firesentinel.vision.engine` combines catalog provenance, calibrated C07/C14
crops, mask-aware preparation, quality gating, contextual anomalies, and
persistence into one local content-addressed packet. It accepts only local
source paths selected by a prior catalog lookup; it performs no downloads.

```powershell
.\.venv\Scripts\python -m scripts.tasks evidence `
  --job path\to\evidence-job.json `
  --timeout-seconds 120
```

The job manifest contains `case_id`, crop and tile parameters, and at least two
observations with C07/C14 `catalog_key` plus `source_path` entries. Each run
writes `evidence.json`, NPY measurement/mask arrays, annotated overlays, and a
completion marker beneath `artifacts/{case_id}/{content_hash}/`. Runtime
timings do not affect the content hash; an identical job reuses the completed
packet. Timeout, cancellation, corrupt inputs, and failed writes are
classified, and staged files are never published as completed artifacts.

## Development baseline comparison

`firesentinel.evaluation.runner` provides the fixed comparisons before an
adaptive policy is introduced. It accepts only the frozen development manifest
and resolves each manifest source through the verified local cache; it never
downloads a missing source. Both modes use the same crop policy, tile and
threshold configuration, evidence engine, and cautious outcome function.

```powershell
.\.venv\Scripts\python -m scripts.tasks baselines `
  --evidence-template path\to\evidence-job.json `
  --crop-half-height-degrees 0.25 `
  --crop-half-width-degrees 0.25
```

The one-shot baseline always selects the manifest's `initial` C07 object and
its prescribed `alternate` C14 contextual companion. The fixed bundle always
selects `baseline`, `initial`, `later` (C07) and `alternate` (C14) for every
case. The report records the selected source objects, Channel 7 count, paired
evidence time steps, bytes, zero network-download bytes, latency, outcomes,
errors, and content-addressed evidence IDs in the same schema for both modes.
The template supplies the existing Day 17 tile/quality/anomaly/persistence
settings; its own source paths are not replayed for the benchmark cases.

## Bounded observation tools

`firesentinel.agent.tools.BoundedObservationTools` exposes only
`next_timestamp`, `alternate_band`, `pre_event_baseline`, `finalize`,
`abstain`, and `request_human_review`. An action accepts an allowlisted
observation ID only: source paths, bands, and thresholds are fixed in a local
tool manifest. Cache files are confined to the declared source-cache root and
verified against their declared byte size and SHA-256 before processing.

Each accepted observation replays the cumulative C07/C14 evidence job, so the
returned evidence ID identifies updated masks, overlays, and persistence—not a
black-box score. Tool responses include a `Budget`, cumulative evidence IDs,
and a structured error on rejection. Repeating a successful request is
idempotent; no request may exceed three observations, byte or elapsed-time
limits, and terminal actions close the session. Tool-manifest loading rejects
the entire evaluation-label subtree.

## Transparent agent policy

`firesentinel.agent.policy.TransparentAgentPolicy` is a stateless, ordered rule
table. It accepts explicit evidence facts, the current `Budget`, currently
allowlisted actions, and an optional prior tool reply; it never reads files,
uses an LLM, keeps hidden state, or estimates speculative future value.

The policy prioritizes tool failures and exhausted budgets, then poor quality
and band conflict, then persistent evidence (human review), absent persistence
(finalize), and weak contextual contrast (an allowlisted comparison). Each
`PolicyDecision` records satisfied conditions, the selected rule and reason,
all considered/rejected actions, and measured evidence changes from the prior
packet. `apply_policy_decision` is the separate adapter that invokes the
already bounded Day 19 tool.

## Calibrated reviewer outcomes

`firesentinel.agent.outcomes` applies one shared, development-only outcome
table to accumulated thermal evidence. Its pinned configuration requires two
usable observations before a no-persistent-evidence result, and two aligned
persistence measurements with at least 0.50 confidence before review
escalation. Confidence below that boundary does not become a stronger outcome.

Poor coverage or contrast, alignment failure, and exhausted budgets end in
`insufficient_evidence`; conflicting bands end in `human_review`. A completed
comparison without threshold persistence ends in `no_persistent_evidence`.
Every outcome includes reason codes and fixed plain-language explanation
templates for reviewers. These are thermal-evidence workflow outcomes only;
they do not establish incident type, origin, extent, or operational status.

## Checkpointed local agent loop

`firesentinel.agent.loop` combines observation, evidence analysis, transparent
policy selection, bounded tool use, and terminal calibration in one explicit
state machine. It writes a complete JSONL checkpoint after every transition:
`observe`, `analyze`, `decide`, `act`, `finalize`, `abstain`, `review`, or
`failure`. Re-running the same command resumes from the last complete record;
an unfinished final write is discarded before the next checkpoint is appended.

```powershell
.\.venv\Scripts\python -m scripts.tasks agent-loop `
  --tool-manifest path\to\tool-manifest.json `
  --maximum-bytes 500000000
```

The resumed bounded-tool session restores its selected observations, evidence
IDs, elapsed time, bytes, and retry counts. Already completed observation IDs
are unavailable to the resumed policy, and a restart while an action was in
progress reruns only that content-addressed evidence job, which reuses its
existing artifact rather than publishing a duplicate.

## Evaluation-only FIRMS references

Local FIRMS CSV exports can be normalized into isolated event references for
offline scoring. The ingester retains only acquisition time, WGS84 coordinates,
confidence, brightness, and instrument; it removes exact normalized duplicates
and clusters detections by an audited time and geodesic-distance window.

```powershell
.\.venv\Scripts\python -m scripts.tasks firms-ingest --source path\to\firms.csv
```

It writes labels and a separate audit record to `evaluation-data/firms/`.
That directory is intentionally absent from runtime settings, and agent replay
inputs reject it. See [evaluation-data/README.md](evaluation-data/README.md)
for the artifact schema and overwrite behavior.

## Matched evaluation benchmark

With Day 10 FIRMS labels and a pinned observation-window inventory in
`evaluation-data/`, build the 60-positive/60-control minimum benchmark with:

```powershell
.\.venv\Scripts\python -m scripts.tasks benchmark-build
```

The build validates each C07 initial/later/baseline and C14 alternate source
reference, excludes controls near FIRMS detections, records matching variables
and the random seed, and writes independently hash-audited files under
`evaluation-data/benchmark/`.

## Frozen benchmark splits

Once the benchmark exists, inspect the deterministic Day 12 sample and save
your notes before freezing the three evaluation views:

```powershell
.\.venv\Scripts\python -m scripts.tasks benchmark-freeze `
  --reviewer "reviewer-name" `
  --review-notes path\to\day12-review-notes.txt
```

The freezer assigns connected groups by FIRMS event, 2-degree geographic cell,
and UTC week, never individual frames. It fails when those groupings cannot
form independent development, test, and stress splits. The frozen audit stores
hashes, the leakage result, manual-inspection notes, and distribution reviews.
Test and stress manifests are blind; their labels remain scoring-only and the
`tune` task accepts only the frozen development manifest.
