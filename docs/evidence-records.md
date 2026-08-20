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
