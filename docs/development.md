# Development workflow

All repository commands are explicit and use the standard-library task runner:

```powershell
.\.venv\Scripts\python -m scripts.tasks <task>
```

`format`, `format-check`, `lint`, and `typecheck` run Ruff or mypy from the
locked environment. `test` runs pytest. No global formatter, linter, task
runner, or environment-variable file is required.

## Configuration

The defaults work from a clean checkout. Set these optional environment
variables only to override local paths or log verbosity:

| Variable | Default | Purpose |
| --- | --- | --- |
| `FIRE_SENTINEL_LOG_LEVEL` | `INFO` | One of `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL`. |
| `FIRE_SENTINEL_DATA_DIR` | `data/` | Download destination. |
| `FIRE_SENTINEL_ARTIFACTS_DIR` | `artifacts/` | Generated outputs. |
| `FIRE_SENTINEL_MANIFESTS_DIR` | `manifests/` | Dataset manifest location. |
| `FIRE_SENTINEL_CATALOG_CACHE_DIR` | `data/catalog/` | Immutable cached NOAA GOES-18 catalog listings. |
| `FIRE_SENTINEL_SOURCE_CACHE_DIR` | `data/source-cache/` | Verified content-addressed source-object cache. |

Application logs are JSON emitted to standard error.

## GOES-18 object discovery

`firesentinel.data.goes18` resolves only public GOES-18 `ABI-L2-CMIPF`
full-disk objects for Channels `C07` and `C14`. It uses anonymous HTTPS S3
list requests, so it does not require AWS credentials or an AWS SDK. Each
hourly listing is cached atomically under `data/catalog/`; later lookups reuse
the exact catalog snapshot and its discovery time.

`Goes18ObjectDiscovery.resolve()` returns either a `Goes18ObjectReference`
with bucket, key, size, scan start/end, and discovery timestamp, or a typed
`MissingFrame`. The nearest scan is measured from the object scan start, and
an exact midpoint tie chooses the earlier scan. Catalog access failures remain
exceptions so an unavailable catalog cannot be mistaken for absent imagery.

## Data and workflow placeholders

`manifests/datasets.json` starts empty, therefore the `download` command is a
successful no-op in a clean checkout. Add only explicit `http`/`https` entries
with a `name`, `source_url`, and ideally a `sha256`; see the manifest README.

## Verified source cache

For pinned case sources, `download` streams into a private temporary file,
checks the declared byte size and SHA-256, then atomically publishes the blob
under its digest. Failed, interrupted, and checksum-invalid transfers have no
object or source index entry. A later request rechecks the local blob and is
reported as a cache hit with zero transfer bytes.

Use these safe cache commands:

```powershell
.\.venv\Scripts\python -m scripts.tasks cache-inspect
.\.venv\Scripts\python -m scripts.tasks cache-clean-case --case-id pine-creek
```

The latter removes only `pine-creek`'s source references. Content still linked
by another case remains in the cache.

The `replay` and `evaluate` tasks deliberately validate JSONL input before the
later model and policy milestones add their behavior:

```powershell
.\.venv\Scripts\python -m scripts.tasks replay
.\.venv\Scripts\python -m scripts.tasks evaluate
```

They accept input through their underlying modules when needed:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python -m firesentinel.agent.replay --input artifacts\events.jsonl
.\.venv\Scripts\python -m firesentinel.evaluation.run --input artifacts\evaluation.jsonl
```

## Frozen evaluation boundary

`benchmark-freeze` is the required gate between a candidate benchmark and
model selection. It verifies the existing benchmark hashes, groups cases by
the positive FIRMS event, recomputed two-degree WGS84 cell, and UTC ISO week,
then assigns whole connected components to development, test, or stress.
The output audit records a passing leakage check and the distribution review:
season, local hour, view angle, missingness (`1 - usable_fraction`), FIRMS
confidence, and required C07/C14 bundle availability.

The command requires a reviewer and non-empty manual-inspection notes. This
makes the saved audit a review record rather than an assertion that automation
performed a visual inspection. Frozen test and stress manifests expose only
opaque IDs and model inputs; their labels and the split assignment map are
scoring-only. Tuning code must call `tuning_manifest_path` (or the `tune`
task) and will be rejected unless it uses `development.manifest.json` under
`evaluation-data/frozen/`.
