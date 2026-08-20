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

## Calibrated GOES crops

`firesentinel.data.goes_crop` converts a verified cached GOES `CMI` object into
a compact calibrated regional artifact. Pass its `DownloadReceipt.cache_path`,
non-wrapping WGS84 bounds, and padding policy; the stage decodes the source
scale/offset itself, masks fill/range/DQF/off-Earth pixels, and clips padding at
the source edge. The resulting deterministic `.npz` contains calibrated data,
invalid mask, DQF, per-pixel latitude/longitude, scan coordinates, timing,
projection/calibration metadata, source hash, and a canonical content checksum.
