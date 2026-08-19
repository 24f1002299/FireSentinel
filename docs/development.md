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

Application logs are JSON emitted to standard error.

## Data and workflow placeholders

`manifests/datasets.json` starts empty, therefore the `download` command is a
successful no-op in a clean checkout. Add only explicit `http`/`https` entries
with a `name`, `source_url`, and ideally a `sha256`; see the manifest README.

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
