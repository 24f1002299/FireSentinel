# FireSentinel: Lean 30-Day Local-Only Implementation Plan

> **Competition warning:** This plan does not satisfy the OpenCV AI Competition 2026 requirement for a meaningful AWS component. It is suitable as a local proof of concept, portfolio project, research prototype, or fallback development plan, but it is not an eligible final competition architecture unless an AWS component is added later.

## Fixed MVP scope

- Historical replay only; no live wildfire monitoring or operational alerts.
- One satellite and region: GOES-18 over the western United States by default.
- Two required ABI bands: Channel 7 at 3.9 micrometers and Channel 14 at 11.2 micrometers. Channel 2 is stretch scope only.
- OpenCV 5 pipeline: mask-aware scaling, thresholding, morphology, connected components, contours, metadata-based alignment, and temporal persistence.
- Bounded agent actions: next timestamp, alternate band, pre-event baseline, finalize, abstain, and request human review.
- Maximum three visual observations per investigation.
- Local filesystem cache and evidence store; no cloud services, queues, distributed workers, or infrastructure as code.
- One Streamlit reviewer application and one deterministic command-line replay path.
- Evaluation: one-shot, fixed-bundle, and adaptive-agent modes on at least 60 positive and 60 matched-control windows.

## Definition of done

The project is complete when a reviewer can install it on one computer, select a historical manifest, see OpenCV evidence cause a later agent action, inspect the evidence packet, replay a failure case, and reproduce the comparison between one-shot, fixed-bundle, and adaptive modes.

## Scope rules

- Do not add AWS, other clouds, queues, microservices, Kubernetes, live feeds, optical flow, perspective registration, or an LLM controller.
- Keep one Python application process and one local artifact directory.
- A failed stretch feature is removed rather than delaying the working vertical slice.
- Stop feature development after Day 24; Days 25-30 are for evidence, presentation, and reliability.

## Day 1 — Freeze the local product contract

**Goal:** Define the smallest complete local system and prevent the original architecture from returning through scope creep.

**Tasks:**

- Convert the vision, agent, evaluation, UI, documentation, and responsible-use requirements into a one-page acceptance matrix.
- Mark every feature as mandatory, optional, or excluded.
- Define three reviewer stories: emerging fire, matched control, and forced abstention.

**Verify:** Every mandatory item maps to one code artifact, test, metric, screenshot, trace, or demo moment.

**Commit:** docs(scope): freeze lean local mvp and acceptance criteria

## Day 2 — Prove the local OpenCV 5 runtime

**Goal:** Establish a reproducible development environment before touching satellite data.

**Tasks:**

- Create a pinned Python environment with OpenCV 5 and the minimum geospatial, data, testing, and UI dependencies.
- Record OpenCV version, build information, CPU architecture, Python version, and dependency lock.
- Run resize, threshold, morphology, connected-component, and contour smoke tests.

**Verify:** A clean environment installs from the lockfile and produces the expected smoke-test hashes using OpenCV 5.

**Commit:** spike(runtime): verify pinned local opencv5 environment

## Day 3 — Scaffold the repository and commands

**Goal:** Create a small maintainable project with obvious paths for data, vision, agent logic, evaluation, UI, and documentation.

**Tasks:**

- Create packages for data, vision, agent, evaluation, UI, tests, manifests, scripts, artifacts, and documentation.
- Add configuration, structured logging, formatting, linting, typing, and a task runner.
- Document setup, test, download, replay, evaluate, UI, and cleanup commands.

**Verify:** A clean checkout runs the smoke tests and launches an empty Streamlit application without hidden configuration.

**Commit:** chore(repo): scaffold lean local firesentinel project

## Day 4 — Define evidence and trace records

**Goal:** Make observations, measurements, decisions, budgets, and outcomes inspectable without overbuilding schemas.

**Tasks:**

- Define typed JSON records for a manifest case, observation request, source object, vision evidence, action, budget, trace, and outcome.
- Standardize UTC timestamps, coordinates, units, missing values, confidence ranges, and reason codes.
- Define a simple artifact-directory convention based on case ID and content hash.

**Verify:** Golden records round-trip through JSON, invalid values fail validation, and every outcome links to its evidence and configuration.

**Commit:** feat(core): add local evidence trace and artifact contracts

## Day 5 — Build deterministic synthetic fixtures

**Goal:** Enable fast vision and policy development without repeated downloads.

**Tasks:**

- Generate small arrays for persistent heat, transient heat, image shift, missing pixels, saturated pixels, low contrast, and empty frames.
- Add expected masks, components, persistence values, and reason codes.
- Pin seeds and numeric tolerances and retain a small offline fixture bundle.

**Verify:** Repeated fixture generation is deterministic and deliberate corruption fails for the expected reason.

**Commit:** test(fixtures): add deterministic thermal and quality cases

## Day 6 — Implement GOES-18 object discovery

**Goal:** Resolve a requested time and band into stable NOAA object references without AWS credentials.

**Tasks:**

- Support only the selected GOES-18 ABI product and Channels 7 and 14.
- Implement anonymous catalog access, scan-time parsing, nearest-scan selection, local catalog caching, and typed missing-frame results.
- Store source bucket, key, size, scan timestamps, and discovery time in each manifest record.

**Verify:** Known-event queries return the same ordered references and boundary tests select the expected observation.

**Commit:** feat(data): add anonymous goes18 object discovery

## Day 7 — Implement download and immutable local caching

**Goal:** Make external source access repeatable while avoiding unnecessary repeated downloads.

**Tasks:**

- Download selected source objects to a content-addressed local cache with retry and checksum validation.
- Record source size, downloaded bytes, checksum, elapsed time, and cache hits.
- Add cache inspection and safe case-specific cleanup commands.

**Verify:** Interrupted downloads do not appear complete, checksum failures are rejected, and a second request uses the verified cache.

**Commit:** feat(data): add verified local source cache

## Day 8 — Implement calibrated regional crop extraction

**Goal:** Convert a cached GOES object into a small calibrated regional array with correct validity and location metadata.

**Tasks:**

- Read projection, scan coordinates, fill values, calibration coefficients, and data-quality metadata.
- Convert latitude-longitude bounds to source indices and extract a padded crop without edge wrapping.
- Save calibrated arrays, invalid masks, geographic bounds, crop parameters, timing, and checksum.

**Verify:** Reference coordinates land within tolerance, calibration matches an independent sample, and repeated crops are identical.

**Commit:** feat(data): extract calibrated local goes crops

## Day 9 — Complete one real-event vertical slice

**Goal:** Produce visible OpenCV evidence from one real historical event before building a benchmark or agent.

**Tasks:**

- Create one manually audited event manifest with an initial Channel 7 frame and a later observation.
- Apply display scaling, thresholding, morphology, connected components, and contours using OpenCV 5.
- Write one JSON evidence record and annotated before-after panel.

**Verify:** One command recreates the same measurement, contour, hashes, and reviewer image from the cached source.

**Commit:** feat(slice): replay one historical event locally

## Day 10 — Ingest isolated FIRMS references

**Goal:** Build event-level scoring references without exposing them to runtime decision code.

**Tasks:**

- Ingest permitted FIRMS fields for acquisition time, coordinates, confidence, brightness, and instrument.
- Normalize timestamps and coordinates, remove duplicates, and cluster nearby detections into event windows.
- Place labels under an evaluation-only path excluded from runtime configuration.

**Verify:** Counts, date ranges, source hashes, and normalization statistics are recorded, and agent code cannot load the labels.

**Commit:** feat(eval-data): ingest isolated local firms references

## Day 11 — Generate positives and matched controls

**Goal:** Build a useful benchmark without manual image labeling or a full MTBS geometry pipeline.

**Tasks:**

- Generate positive windows from FIRMS clusters with eligible initial, later, alternate-band, and baseline observations.
- Sample controls by season, region, local time, view geometry, and data quality while excluding nearby FIRMS detections.
- Save matching variables, source references, random seed, and integrity hashes.

**Verify:** At least 60 positives and 60 controls are generated, required references resolve, and rebuilding is deterministic.

**Commit:** feat(dataset): generate event windows and matched controls

## Day 12 — Audit, split, and freeze the benchmark

**Goal:** Establish a stable evaluation basis and prevent neighboring frames from leaking across splits.

**Tasks:**

- Split by event, geographic cell, and time period rather than by frame.
- Review distributions for season, hour, view angle, missingness, confidence, and band availability.
- Manually inspect a small sample and freeze development, test, and stress manifests with hashes.

**Verify:** Leakage checks pass, audit notes are saved, and the frozen test labels are inaccessible to tuning commands.

**Commit:** data(release): freeze audited local benchmark

## Day 13 — Build mask-aware tile preparation

**Goal:** Convert calibrated data into stable OpenCV inputs without losing scientific meaning.

**Tasks:**

- Implement physical clipping, robust display scaling, mask-aware resize, and optional CLAHE.
- Preserve calibrated arrays separately from display images.
- Emit processing parameters, input masks, ranges, timings, and OpenCV build metadata.

**Verify:** Invalid pixels never become evidence, hot-cold ordering is preserved, and golden outputs remain within tolerance.

**Commit:** feat(vision): add deterministic mask-aware preparation

## Day 14 — Measure observation quality

**Goal:** Detect unusable or incomplete frames before interpreting apparent thermal anomalies.

**Tasks:**

- Measure missing-pixel fraction, saturation, usable coverage, contrast, and simple texture statistics.
- Convert them into bounded fields and explicit poor-quality reason codes.
- Select thresholds using only development cases and synthetic fixtures.

**Verify:** Blank, clipped, missing, and low-contrast fixtures trigger the correct reason and cannot yield confident fire evidence.

**Commit:** feat(vision): add quality coverage and reason codes

## Day 15 — Extract contextual thermal anomalies

**Goal:** Produce interpretable candidate regions from Channels 7 and 14 instead of a black-box classification.

**Tasks:**

- Calculate local Channel 7 contrast and Channel 7-minus-Channel 14 contextual differences.
- Apply thresholding, morphology, connected components, contours, and region filtering in OpenCV 5.
- Return component area, centroid, contrast, edge proximity, mask, and annotated overlay.

**Verify:** Injected hot regions are localized, isolated noise and invalid pixels are rejected, and measurements match source arrays.

**Commit:** feat(vision): extract contextual thermal regions

## Day 16 — Measure temporal persistence

**Goal:** Separate sustained evidence from transient or displaced responses using a simple, defensible method.

**Tasks:**

- Place observations on a common grid using geospatial metadata.
- Match regions using centroid distance and mask intersection-over-union.
- Measure persistence count, overlap, area trend, temperature trend, and disappearance.

**Verify:** Persistent fixtures outscore transient ones, missing observations do not create continuity, and poor overlap causes low confidence.

**Commit:** feat(vision): measure metadata-aligned persistence

## Day 17 — Package a deterministic evidence engine

**Goal:** Expose catalog, crop, preparation, quality, anomaly, and persistence stages through one idempotent local job.

**Tasks:**

- Define a single evidence-job function and command-line interface.
- Write JSON evidence, masks, overlays, timings, warnings, provenance, and content hashes to the case artifact directory.
- Add timeout, cancellation, error classification, and golden end-to-end replay tests.

**Verify:** Running an identical job twice returns the same artifact IDs and failures never leave partial outputs marked complete.

**Commit:** feat(vision): expose deterministic local evidence engine

## Day 18 — Implement one-shot and fixed-bundle baselines

**Goal:** Create fair comparisons before implementing the adaptive policy.

**Tasks:**

- Define one-shot as one predetermined Channel 7 observation.
- Define fixed-bundle as the same predetermined multi-time, two-band observations for every case.
- Reuse identical evidence, cache, threshold, and outcome logic where the modes overlap.

**Verify:** Both modes complete the development manifest and report outcomes, observations, bytes, latency, errors, and evidence comparably.

**Commit:** feat(eval): add local one-shot and fixed-bundle modes

## Day 19 — Implement bounded observation tools

**Goal:** Give the agent a small action space that visibly changes later OpenCV processing.

**Tasks:**

- Implement next-timestamp, alternate-band, pre-event-baseline, finalize, abstain, and human-review tools.
- Enforce the manifest allowlist, maximum three observations, elapsed-time limit, and byte budget.
- Make repeated requests idempotent and return evidence IDs, resource use, and structured errors.

**Verify:** Contract tests cover allowed and forbidden transitions and no tool can access evaluation labels or arbitrary files.

**Commit:** feat(agent): add bounded local observation tools

## Day 20 — Implement the transparent agent policy

**Goal:** Select the next observation from evidence reasons with no LLM, hidden state, or speculative future-value equation.

**Tasks:**

- Define rules for poor quality, weak contextual contrast, absent persistence, evidence conflict, success, and exhausted budget.
- Log considered actions, satisfied conditions, selection reason, rejected actions, and evidence changes.
- Add table-driven tests for emerging, control, abstention, and tool-failure scenarios.

**Verify:** The same trace always selects the same action and controlled evidence changes cause the expected different action.

**Commit:** feat(agent): select second looks with deterministic policy

## Day 21 — Calibrate outcomes and abstention

**Goal:** Convert accumulated evidence into cautious reviewer-facing outcomes without claiming wildfire confirmation.

**Tasks:**

- Define development-only thresholds for review escalation, insufficient evidence, and no persistent evidence.
- Force abstention or review for poor coverage, alignment failure, conflicting evidence, and exhausted budgets.
- Generate plain-language explanations from fixed reason-code templates.

**Verify:** Every low-confidence case terminates safely and terminology tests prevent thermal anomalies from being called confirmed wildfires.

**Commit:** feat(agent): calibrate outcomes abstention and explanations

## Day 22 — Complete the bounded local agent loop

**Goal:** Join perception, decision, action, and termination into one reliable deterministic replay.

**Tasks:**

- Implement explicit observe, analyze, decide, act, finalize, abstain, review, and failure states.
- Persist the trace after every transition and enforce observation, time, byte, and retry budgets.
- Add resume support from the last complete trace record.

**Verify:** Every development case reaches one terminal state, no loop exceeds its budget, and interrupted cases resume without duplicate artifacts.

**Commit:** feat(agent): complete bounded local investigation loop

## Day 23 — Build the Streamlit evidence reviewer

**Goal:** Let a reviewer understand the initial ambiguity, action, changed evidence, result, and limitations quickly.

**Tasks:**

- Add case selection, location context, chronological evidence strip, measurements, masks, contours, and outcome.
- Display considered actions, selected action, reason codes, observation count, bytes, warnings, and provenance.
- Add deterministic emerging-event, matched-control, and abstention demo buttons.

**Verify:** A non-developer can explain all three demo stories without opening raw JSON or terminal logs.

**Commit:** feat(ui): add local evidence review dashboard

## Day 24 — Add essential failure handling

**Goal:** Make predictable failures visible and safe without building enterprise resilience infrastructure.

**Tasks:**

- Handle missing source, corrupt cache entry, timeout, insufficient disk, unusable frame, and poor alignment.
- Add one retry where appropriate, followed by explicit abstention or human review.
- Show errors and recovery actions in both the trace and UI.

**Verify:** Every injected failure reaches the documented terminal behavior and no corrupt or partial artifact is presented as valid evidence.

**Commit:** test(system): add local failure and recovery paths

## Day 25 — Run the frozen evaluation

**Goal:** Measure whether adaptive observation improves evidence efficiency without sacrificing too much task performance.

**Tasks:**

- Run one-shot, fixed-bundle, and adaptive modes on frozen test and stress manifests.
- Compute macro F1, precision, recall, bootstrap intervals, ambiguous-case resolution, abstention coverage, observations, bytes, and latency.
- Freeze per-case results and aggregate tables before beginning error analysis.

**Verify:** A single documented command reproduces every reported number from pinned manifests, configuration, and code.

**Commit:** eval(release): freeze local baseline and agent metrics

## Day 26 — Analyze errors and agent value

**Goal:** Identify where the second-look policy helps, wastes work, or fails.

**Tasks:**

- Review false positives, false negatives, abstentions, and cases where extra observations did not help.
- Compare fixed-bundle and adaptive recall against their mean observations and bytes.
- Select representative success, control, abstention, and genuine limitation cases.

**Verify:** Every headline claim is supported by a table and trace, and representative failures appear beside successes.

**Commit:** docs(results): analyze local agent value and limitations

## Day 27 — Optimize only demonstrated bottlenecks

**Goal:** Improve demo responsiveness and batch reproducibility without adding architectural complexity.

**Tasks:**

- Profile catalog access, crop loading, OpenCV stages, artifact writing, and UI rendering.
- Optimize only the top measured bottleneck using local caching, vectorization, or reduced duplicate work.
- Re-run numerical parity and frozen evaluation checks after optimization.

**Verify:** The target bottleneck improves measurably while evidence outputs and evaluation results remain within tolerance.

**Commit:** perf(local): optimize measured replay bottleneck

## Day 28 — Test and polish reviewer experience

**Goal:** Make the project understandable and credible within a five-minute demonstration.

**Tasks:**

- Run five structured reviewer sessions with the three deterministic demo cases.
- Fix only comprehension, accessibility, navigation, loading, and visual-hierarchy issues.
- Add color-safe overlays, text alternatives, prominent limitations, and one-click reset.

**Verify:** At least four of five reviewers identify the ambiguity, selected action, changed evidence, outcome, and limitation without coaching.

**Commit:** feat(ui): polish local judge experience

## Day 29 — Produce local release and presentation artifacts

**Goal:** Make the project reproducible, understandable, and demonstrable offline.

**Tasks:**

- Finish the report, architecture diagram, agent workflow, dependency lock, licenses, setup guide, and responsible-use section.
- Package a small permitted fixture/demo bundle plus scripts to fetch larger public inputs.
- Record a draft five-minute video showing the problem, live trace, OpenCV evidence, changed action, metrics, and failure case.

**Verify:** A fresh evaluator installs the project, replays one bundled case, reproduces a sample metric, and completes the video in under five minutes.

**Commit:** docs(release): assemble local report replay and demo

## Day 30 — Freeze and publish the local prototype

**Goal:** Release a stable local-only project with no last-day feature changes.

**Tasks:**

- Tag the final code, manifests, configuration, result tables, and documentation.
- Run the clean-install test, complete test suite, three demo stories, frozen sample evaluation, and two timed rehearsals.
- Record the final video, publish the repository or judge archive, and clearly disclose the absence of AWS deployment.

**Verify:** All artifacts match the final tag, the deterministic replay works offline, the video is under five minutes, and limitations are explicit.

**Commit:** release: publish firesentinel local prototype
