# FireSentinel

FireSentinel is a local, reproducible fire-detection workflow. Day 3 supplies
the project skeleton: explicit configuration, JSON logs, a deterministic
OpenCV runtime smoke test, and an empty Streamlit shell.

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

Launch the UI shell at `http://localhost:8501`:

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
