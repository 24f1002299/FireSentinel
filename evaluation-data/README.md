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
