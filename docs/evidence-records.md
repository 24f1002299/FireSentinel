# Evidence and trace records

FireSentinel stores its Day 4 evidence packet as canonical JSON emitted by
`firesentinel.core.records`. The contracts use Python dataclasses and the
standard library only; they are intentionally not a database or a general
schema framework.

Every top-level JSON value has a `record_type` and `schema_version` (currently
`1`). The typed trace record types are `manifest_case`, `observation_request`,
`source_object`, `vision_evidence`, `action`, `budget`, `trace`, and `outcome`.
`Trace` is the complete packet: it embeds the linked case, requests, sources,
evidence, actions, budget, and terminal outcome, then validates those links.

The Day 9 cached-only vertical slice emits a separate
`real_event_evidence` record (also schema version `1`) before the agent and
trace stages exist. It records two source hashes, calibrated-crop hashes,
display/threshold/morphology mask hashes, exact external contour points and
hashes, connected-component measurements, and the annotated reviewer-panel
PNG hash. Its content hash covers every field except itself. `scripts.tasks
slice` verifies that record and image against the hashes pinned in the audited
event manifest.

## Shared conventions

- Timestamps are timezone-aware UTC values and serialize as RFC 3339 strings
  ending in `Z`.
- Locations are WGS 84 decimal degrees, serialized as `{ "lat": ..., "lon": ... }`.
- Measurements always carry one of `K`, `km2`, `px`, `s`, `B`, `deg`, or `1`.
- A missing measurement is JSON `null` and must name a `missing_reason`; a
  present measurement must have a null `missing_reason`.
- Confidence is a finite number in the inclusive range `0.0` through `1.0`.
- Reasons are closed, lowercase `ReasonCode` values. This keeps decisions
  comparable while any reviewer prose can live separately from factual records.
- Content hashes are lowercase, 64-character SHA-256 hex digests.
- `source_object` records retain the exact source bucket, object key, byte size,
  scan start, scan end, and source discovery time. The SHA-256 becomes known
  when the selected object has been downloaded and verified.

`Outcome` requires at least one evidence ID and a
`ConfigurationReference`. `Trace` rejects an outcome unless every outcome
evidence ID exists in its evidence list and both the outcome and every evidence
record use the trace configuration. It also checks all request, source, action,
and case links.

## Artifact layout

The deterministic packet directory is:

```text
artifacts/{case_id}/{content_hash}/
  trace.json
  ... future masks, overlays, and measurements ...
```

Use `canonical_content_hash(trace)` for the packet hash and
`artifact_directory(settings.artifacts_dir, trace.case.case_id, trace_hash)`
to get the safe path. The helper only derives the directory; later writers can
create it atomically after their artifacts are complete.

## Local evidence-engine packets

Day 17 also emits a separate `local_evidence_job` packet from
`firesentinel.vision.engine`. Its content hash covers the path-free job
configuration, selected catalog keys, source/crop provenance, stable tile
metadata, quality/anomaly/persistence measurements, warnings, and hashes of
all saved NPY/PNG assets. Wall-clock timings are included for review but are
excluded from the content-addressed ID.

The engine first writes all assets to a private staging directory. It writes
`completion.json` only after `evidence.json` and every declared asset is ready,
then atomically renames the staging directory to the content-addressed final
path. Existing packets are hash-verified before reuse. Thus timeout,
cancellation, source failure, or a write failure cannot be treated as a
completed packet.

## Development baseline reports

`firesentinel.evaluation.runner` writes a `development_evidence_baselines`
report containing parallel `one_shot` and `fixed_bundle` sections. Each
per-case entry has the identical outcome, observation, evidence, resource, and
error fields. `observation_count` counts selected source objects while
`channel7_observation_count` makes the one predetermined C07 observation
explicit. The fixed bundle selects the pinned C07 `baseline`, `initial`, and
`later` sources plus the pinned C14 `alternate` source for every case.

The report references completed Day 17 evidence packets by content hash and
artifact directory. `selected_source_bytes` is the immutable manifest sum and
`downloaded_bytes` is always zero: baseline replay only calls
`VerifiedSourceCache.require_cached`, which refuses a cache miss instead of
performing a network request. Per-case failures retain a classified error and
do not prevent the remaining development cases or the other mode from being
reported.

## Sealed frozen-evaluation report

`firesentinel.evaluation.frozen_run` emits a
`firesentinel_frozen_evaluation` record only after validating the complete
frozen benchmark set. It contains sorted `per_case_results` for `one_shot`,
`fixed_bundle`, and `adaptive` modes, plus `aggregate_tables` for test, stress,
and combined rows. `analysis_status` is fixed to
`frozen_before_error_analysis`.

The evaluator stores input hashes for both blind manifests, scoring-only label
files, the path-free evidence template, and the active evaluator/policy/outcome
and vision implementation. Aggregates include macro F1, precision, recall,
seeded percentile bootstrap intervals, abstention coverage, initial-one-shot
ambiguity resolution, observation/byte totals, latency, and errors. An
existing changed report is refused unless an explicit rerun and overwrite are
requested. A matching report is verified and reused by the documented task.

For scoring only, `review_escalation` maps to positive thermal evidence and
`no_persistent_evidence` maps to control. `human_review`,
`insufficient_evidence`, and `failed` remain abstentions; abstentions count as
false negatives for their true class in macro metrics. This evaluation mapping
does not claim confirmed wildfire occurrence.

## Bounded observation-tool replies

Day 19's `ToolResult` is an in-process agent reply rather than a new evidence
packet. It returns the action, allowlisted observation ID, cumulative evidence
content hashes, a `Budget` resource record, terminal state, idempotence flag,
and an optional structured error. Every accepted observation is replayed as a
new cumulative Day 17 packet, so the returned latest content hash identifies
the updated C07/C14 anomaly and persistence measurements.

Tool manifests include exactly the case's allowlisted observation IDs and
pair each action with predeclared C07/C14 cached source objects. The action API
cannot receive paths or URLs. Before reading, source paths must stay under the
configured source-cache root and outside `evaluation-data`; size and SHA-256
are verified. Tool-manifest file loading uses the same label boundary, so an
agent cannot use a tool request to read evaluation labels or arbitrary files.

## Transparent policy decisions

Day 20's `PolicyDecision` is a pure, reviewer-facing rule evaluation. Its
inputs are an `EvidenceSnapshot`, a `Budget`, allowlisted observation actions,
and optionally the previous evidence and last `ToolResult`. It stores the
selected action, rule, satisfied conditions, human-readable selection reason,
every considered and rejected action, and field-by-field evidence changes.

The ordered table does not contain model output or a future-value score. A
prior tool failure or exhausted budget selects abstention; poor quality and
band conflict select a permitted comparison before persistent evidence can
escalate to human review. Absent persistence selects finalization, and weak
contextual evidence selects a deterministic follow-up. The selected action is
only executed by `apply_policy_decision`, which delegates to the bounded tool
surface described above.

## Checkpointed agent-loop journal

Day 22 adds `firesentinel.agent.loop`, whose JSONL journal is distinct from the
terminal core `Trace` record. Every journal line is a complete state snapshot:
it includes the state transition, selected observations, cumulative evidence
IDs, `Budget`, analysis facts, selected policy action, most recent tool reply,
and any terminal outcome. A resume reads the last complete line and restores
the bounded tool session from those facts.

The journal is flushed and synced after each line. A truncated final line is
ignored while loading and removed before the next append; a malformed earlier
line is rejected. This lets an interrupted replay continue without selecting a
completed observation again. If interruption occurred after an action was
checkpointed but before its result was recorded, the content-addressed evidence
job may be replayed, but it reuses the same completed artifact.

## Calibrated outcomes

`firesentinel.agent.outcomes` is the shared terminal calibration layer for
development reports and the local agent. `OutcomeThresholds` records its
development-only selection scope and its explicit review, insufficient, and
no-persistent-evidence boundaries. `OutcomeEvidence` contains only measured
observation counts, candidate and persistence facts, closed reason codes, and
the budget-exhausted flag.

The calibrator terminates rather than strengthening weak cases: poor quality,
alignment failure, or a depleted budget produce `insufficient_evidence`; band
conflict produces `human_review`; and a complete usable comparison without
threshold persistence produces `no_persistent_evidence`. Only threshold
persistence produces `review_escalation`. `CalibratedOutcome.explanation`
combines fixed outcome and reason-code templates, keeping reviewer language
deterministic and limited to thermal evidence.

## Streamlit evidence reviewer

Day 23's local reviewer reads completed `real_event_evidence` and
`local_evidence_job` packets from the artifacts directory. It normalizes them
into a display model, so packet JSON and terminal logs are not exposed in the
review flow. A nearby complete Day 22 `agent-loop.jsonl` checkpoint supplies
the selected and considered actions, resource budget, reason codes, and
terminal outcome when available.

The interface displays location context, initial ambiguity, time-ordered
observations, calibrated measurements, masks, contours, evidence changes,
warnings, and provenance. Its three built-in deterministic stories cover
review escalation, no persistent evidence, and safe abstention. Every story
uses reviewer-facing thermal-evidence language and does not establish a
wildfire determination.

## Failure and recovery records

Day 24 extends the `last_tool_result.error` object stored in every bounded-loop
checkpoint with a fixed `recovery_action`. It covers unavailable or corrupt
cache bytes, elapsed-time limits, insufficient disk space, and artifact write
failures without allowing alternative sources or partial outputs. The next
policy decision abstains after those tool failures.

For `coverage_insufficient`, blank/saturated frames, and `alignment_failed`,
the loop can consume exactly one persisted recovery retry for an allowlisted
replacement observation. Its transition event records that retry. A second
unusable/unaligned result selects explicit abstention. Persistence results
write `alignment_failed` as a closed reason code when a real observation has
no valid mapping on the common geographic grid.

The reviewer verifies a `local_evidence_job` packet's completion marker,
packet SHA-256, and every declared asset hash and size before displaying it.
If no valid packet exists, it can show the loop journal as a trace-only case
with its terminal outcome, error, and fixed recovery action. It never presents
a staged, corrupt, or partially written artifact as valid evidence.
