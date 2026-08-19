# FireSentinel local MVP product contract

**Status:** Frozen on Day 1. **Normative scope:** one local Python process, historical replay, deterministic artifacts.  
**Labels:** **M** = mandatory for release; **O** = optional and removable; **X** = excluded. Optional work may start only after all mandatory acceptance evidence exists. Adding an excluded item requires an explicit contract revision.

## One-page acceptance matrix

| ID | Area | Feature and acceptance condition | Class | Release evidence |
|---|---|---|:---:|---|
| PC-01 | Product | Replay historical GOES-18 ABI data over the western United States, using Channels 7 and 14, from a pinned case manifest. No live claim is made. | M | `src/firesentinel/data/` + deterministic manifest replay test |
| PC-02 | Runtime | Install a pinned OpenCV 5 environment on one computer and run the product as one Python application process with one local artifact directory. | M | lockfile + runtime smoke-test hash and recorded build info |
| PC-03 | Vision | Preserve calibrated values and validity masks; perform mask-aware scaling, thresholding, morphology, connected components, and contours deterministically. | M | `src/firesentinel/vision/engine.py` + golden fixture test |
| PC-04 | Vision | Measure frame quality and contextual Channel 7 / Channel 14 thermal evidence; invalid, blank, saturated, or low-contrast pixels cannot produce confident evidence. | M | quality/anomaly fixture tests with expected reason codes |
| PC-05 | Vision | Align observations from geospatial metadata and measure region persistence, overlap, area trend, temperature trend, and disappearance. | M | persistent-versus-transient golden test |
| PC-06 | Evidence | Write an inspectable, content-hashed packet containing source provenance, configuration, measurements, masks, contours, overlays, warnings, timing, and outcome links. | M | `src/firesentinel/core/records.py` + golden JSON/overlay replay |
| PC-07 | Agent | Offer only: next timestamp, alternate band, pre-event baseline, finalize, abstain, and request human review; enforce manifest allowlist plus observation, time, byte, and retry budgets. | M | `src/firesentinel/agent/tools.py` contract tests and budget trace |
| PC-08 | Agent | Use a deterministic rule policy with no hidden state: log considered/rejected actions, satisfied conditions, selection reason, evidence changes, resource use, and terminal action. | M | `src/firesentinel/agent/policy.py` table tests + reviewer trace |
| PC-09 | Agent | Make no more than three visual observations and always terminate safely as review escalation, no persistent evidence, insufficient evidence/abstention, human review, or explicit failure. | M | loop termination/budget tests over all frozen cases |
| PC-10 | Replay | Provide one deterministic CLI command that recreates a case and its artifact IDs; interrupted or failed work never appears complete. | M | `src/firesentinel/cli.py` end-to-end replay test |
| PC-11 | Evaluation | Freeze at least 60 positive and 60 season/region/local-time/view-geometry/data-quality matched-control windows, split without event, place, or time leakage; runtime code cannot read labels. | M | manifest integrity/leakage tests and dataset audit metric |
| PC-12 | Evaluation | Compare one-shot, fixed-bundle, and adaptive modes on identical frozen cases and shared evidence/outcome logic. | M | `src/firesentinel/evaluation/runner.py` parity test |
| PC-13 | Evaluation | Reproduce macro F1, precision, recall, bootstrap intervals, ambiguous-case resolution, abstention coverage, observations, bytes, latency, and errors from one documented command. | M | frozen aggregate table plus per-case results |
| PC-14 | UI | A local Streamlit reviewer shows case context, chronological evidence, calibrated measurements, masks/contours, initial ambiguity, considered and selected actions, reasons, changed evidence, budgets, outcome, provenance, warnings, and limitations. | M | `src/firesentinel/ui/app.py` + reviewer-page screenshot |
| PC-15 | Demo | Provide deterministic one-click emerging-fire, matched-control, and forced-abstention stories satisfying RS-1 through RS-3 below. | M | three saved traces and a five-minute demo moment |
| PC-16 | Reliability | Missing source, corrupt cache, timeout, insufficient disk, unusable frame, and poor alignment end in a documented safe state; partial/corrupt artifacts are never presented as valid. | M | injected-failure system test and abstention trace |
| PC-17 | Documentation | Document setup, replay, evaluation, UI, cleanup, architecture, agent workflow, licenses, provenance, limitations, and offline demo; disclose that local-only architecture is competition-ineligible without meaningful AWS. | M | `README.md`, `docs/architecture.md`, and clean-install demo |
| PC-18 | Responsible use | Describe outputs only as thermal evidence, never confirmed wildfire; state historical/non-operational status, uncertainty, data limits, human-review role, and prohibition on emergency or dispatch decisions. | M | `docs/responsible-use.md` terminology test + visible UI notice |
| PC-19 | Vision | Add Channel 2 imagery and/or CLAHE when it improves reviewed evidence without changing calibrated measurements. | O | Optional ablation result; otherwise absent from release |
| PC-20 | Architecture | Live monitoring or operational alerts; other satellites/regions; AWS/other cloud; queues, distributed workers, microservices, Kubernetes, or infrastructure as code. | X | Scope audit finds no implementation/configuration |
| PC-21 | Vision/agent | Optical flow, perspective registration, LLM controller, arbitrary file/network tools, speculative future-value planning, full MTBS geometry, or manual image-labeling pipeline. | X | Dependency/API audit finds no implementation |

The mandatory release gate is binary: every **M** row must have its named artifact and passing evidence; **O** rows do not compensate for missing mandatory evidence.

## Reviewer stories

### RS-1 — Emerging fire (review escalation)

**Given** a pinned historical case whose first Channel 7 observation contains a valid but ambiguous contextual hot region, **when** the adaptive policy sees that persistence is not yet established, **then** it selects an allowed later timestamp and records why. The later aligned observation overlaps and strengthens the region, so the trace ends in **request human review / review escalation**, never “confirmed wildfire.” The reviewer can point to the first ambiguity, selected action, persistence change, final outcome, and limitation in the UI. Budget: at most three observations.

### RS-2 — Matched control (no persistent evidence)

**Given** a control matched on season, region, local time, view geometry, and data quality whose first observation contains an ambiguous or transient response, **when** the policy requests the allowed comparison observation, **then** the response disappears or fails the persistence rule. The trace ends in **no persistent evidence**, displays the matching/provenance context, and does not escalate. The reviewer can identify the evidence change and why extra observation stopped. Budget: at most three observations.

### RS-3 — Forced abstention (insufficient evidence)

**Given** a pinned case with unusable coverage, failed alignment, conflicting bands, or an unavailable allowed follow-up, **when** the policy exhausts the useful allowlisted action or a budget, **then** confident evidence is blocked and the trace ends in **abstain / insufficient evidence** (or explicit human review where safety policy requires it). The UI exposes the reason code, failed/rejected actions, consumed budget, and recovery guidance; it never substitutes invalid or partial data. Budget: at most three observations.

## Day 1 verification

- Matrix IDs PC-01 through PC-21 cover product/data, runtime, vision, evidence, agent, replay, evaluation, UI/demo, reliability, documentation, responsible use, optional scope, and exclusions.
- Every mandatory row names at least one code artifact, test, metric/table, screenshot, trace, or demo moment in **Release evidence**.
- RS-1, RS-2, and RS-3 each define setup, evidence-driven action, terminal outcome, reviewer-visible proof, safety language, and observation bound.

