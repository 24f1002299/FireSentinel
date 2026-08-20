# Day 27 local performance profile

Use the bounded local profiler with any existing Day 17 evidence-job manifest:

```powershell
.\.venv\Scripts\python -m scripts.tasks profile `
  --job path\to\evidence-job.json `
  --output artifacts\performance-profile.json
```

It measures a local catalog-cache read, verified-source metadata access, crop
loading, alignment/OpenCV stages, persistence, residual artifact and metadata
work, and the reviewer view-model load. The latter is the deterministic local
part of UI rendering; browser/Streamlit transport is intentionally not treated
as evidence-pipeline performance.

## Measured bottleneck and change

The initial local profile used the cached Park Fire C07 initial/later files,
with the initial file reused as the fixed contextual companion. This makes the
otherwise duplicated companion crop visible without making any scientific
claim about that stand-in channel.

| Component | Before (ms) | After (ms) | Result |
| --- | ---: | ---: | --- |
| Crop loading | 3,099.9 | 1,516.9 | 51.1% lower |
| Local catalog-cache hit | -- | 0.7 | Not a bottleneck |
| Source-cache metadata access | -- | 0.2 | Not a bottleneck |
| Alignment | -- | 25.4 | Not a bottleneck |
| OpenCV prepare + anomaly | -- | 10.4 | Not a bottleneck |
| Persistence | -- | 25.2 | Not a bottleneck |
| Artifact/metadata residual | -- | 20.2 | Not a bottleneck |
| Reviewer model load | -- | 158.3 | Below crop loading |

The optimization is a job-local cache in the evidence engine, keyed by the
resolved source path. `CalibratedCrop` objects are immutable after validation,
so repeated use of the same source in one bounded job skips repeated NetCDF
decode/projection work. The cache never survives a job and therefore cannot
hide changed files across replays.

The numerical-parity contract compares the cache path with four byte-identical
source copies that deliberately defeat the cache; both produce the same
content-addressed evidence hash. The frozen-evaluation implementation hash
includes the engine, so an existing Day 25 report must be rerun and resealed
before claims are reused. This workspace has no frozen test/stress inputs, so
an actual frozen rerun is intentionally not fabricated here.
