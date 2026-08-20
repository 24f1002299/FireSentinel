"""Normalize local evidence packets into a small reviewer-facing case model.

The Streamlit page deliberately consumes this model instead of displaying JSON
directly.  It supports the historical real-event packet, local evidence jobs,
and the bounded-agent transition journal.  None of its terms diagnose or
confirm a wildfire; it only reports the bounded thermal-evidence outcome.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import numpy.typing as npt

from firesentinel.agent.loop import load_last_complete_transition
from firesentinel.agent.outcomes import explain_reason_codes
from firesentinel.agent.tools import recovery_action_for_tool_error
from firesentinel.core.records import ReasonCode

type Scalar = str | int | float | bool | None
type TableRow = dict[str, Scalar]
type Contour = tuple[tuple[int, int], ...]
type ImageArray = npt.NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class ReviewerMeasurement:
    """A concise measurement, including an explicit absent-value explanation."""

    name: str
    value: str
    unit: str
    missing_reason: str | None = None

    def to_row(self) -> TableRow:
        """Return a table row without exposing packet structure."""

        return {
            "Measurement": self.name.replace("_", " "),
            "Value": self.value,
            "Unit": self.unit,
            "If unavailable": self.missing_reason or "—",
        }


@dataclass(frozen=True, slots=True)
class ReviewerComponent:
    """A retained thermal-candidate region measured from calibrated arrays."""

    label: int
    area_pixels: int
    bounding_box: str
    centroid: str

    def to_row(self) -> TableRow:
        """Return a reviewer-readable component table row."""

        return {
            "Region": self.label,
            "Area (px)": self.area_pixels,
            "Bounding box (x, y, w, h)": self.bounding_box,
            "Centroid (x, y)": self.centroid,
        }


@dataclass(frozen=True, slots=True)
class ReviewerObservation:
    """One chronological visual observation and its extracted evidence."""

    observation_id: str
    observed_at: str
    channel: str
    candidate_pixels: int
    components: tuple[ReviewerComponent, ...]
    contours: tuple[Contour, ...]
    reason_codes: tuple[str, ...]
    overlay_path: Path | None = None
    candidate_mask_path: Path | None = None
    maximum_kelvin: float | None = None

    @property
    def contour_vertex_count(self) -> int:
        """Return the total displayed contour geometry size."""

        return sum(len(contour) for contour in self.contours)


@dataclass(frozen=True, slots=True)
class ReviewerOutcome:
    """One bounded reviewer outcome, never a wildfire determination."""

    label: str
    state: str | None
    confidence: float | None
    explanation: str
    terminal: bool


@dataclass(frozen=True, slots=True)
class ReviewerCase:
    """Everything needed to explain one evidence story without raw JSON."""

    case_id: str
    title: str
    source_kind: str
    location: str
    initial_ambiguity: str
    observations: tuple[ReviewerObservation, ...]
    measurements: tuple[ReviewerMeasurement, ...]
    outcome: ReviewerOutcome
    considered_actions: tuple[TableRow, ...]
    selected_action: str | None
    evidence_changes: tuple[TableRow, ...]
    reason_codes: tuple[str, ...]
    warnings: tuple[str, ...]
    provenance: tuple[TableRow, ...]
    budget: tuple[TableRow, ...]
    reviewer_panel_path: Path | None = None
    errors: tuple[str, ...] = ()
    recovery_actions: tuple[str, ...] = ()

    @property
    def is_demo(self) -> bool:
        """Whether this case is a deterministic explanatory demonstration."""

        return self.source_kind == "deterministic demo"


@dataclass(frozen=True, slots=True)
class ReviewerCatalog:
    """The available parsed cases plus non-fatal artifact-read warnings."""

    cases: tuple[ReviewerCase, ...]
    warnings: tuple[str, ...]


def discover_reviewer_cases(artifacts_root: Path) -> ReviewerCatalog:
    """Discover completed packets without treating malformed files as cases."""

    root = Path(artifacts_root)
    if not root.exists():
        return ReviewerCatalog((), (f"Artifact directory is unavailable: {root}",))

    cases: list[ReviewerCase] = []
    warnings: list[str] = []
    for evidence_path in sorted(root.glob("**/evidence.json")):
        try:
            payload = json.loads(evidence_path.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping):
                raise ValueError("packet root is not an object")
            if payload.get("record_type") == "local_evidence_job":
                _verify_completed_local_packet(evidence_path, payload)
            case = reviewer_case_from_packet(payload, evidence_path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            warnings.append(
                f"Skipped unreadable evidence packet {evidence_path}: {error}"
            )
            continue
        if case is None:
            warnings.append(f"Skipped unsupported evidence packet: {evidence_path}")
        else:
            cases.append(case)
    known_case_ids = {case.case_id for case in cases}
    for trace_path in sorted(root.glob("**/agent-loop.jsonl")):
        try:
            trace_case = _trace_only_case(trace_path)
        except ValueError as error:
            warnings.append(f"Skipped unreadable loop trace {trace_path}: {error}")
            continue
        if trace_case is not None and trace_case.case_id not in known_case_ids:
            cases.append(trace_case)
            known_case_ids.add(trace_case.case_id)
    return ReviewerCatalog(
        tuple(sorted(cases, key=lambda case: (case.title.lower(), case.case_id))),
        tuple(warnings),
    )


def _verify_completed_local_packet(
    evidence_path: Path, payload: Mapping[str, object]
) -> None:
    """Reject incomplete or changed local artifacts before the UI can show them."""

    completion_path = evidence_path.parent / "completion.json"
    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("local evidence has no readable completion marker") from error
    if not isinstance(completion, Mapping) or completion.get("record_type") != (
        "evidence_job_completion"
    ):
        raise ValueError("local evidence completion marker is invalid")
    if completion.get("schema_version") != payload.get("schema_version"):
        raise ValueError("local evidence completion schema does not match")
    content_hash = _text(payload.get("content_hash"))
    expected_evidence_hash = _text(completion.get("evidence_sha256"))
    if not content_hash or completion.get("content_hash") != content_hash:
        raise ValueError("local evidence completion content hash does not match")
    if sha256(evidence_path.read_bytes()).hexdigest() != expected_evidence_hash:
        raise ValueError("local evidence bytes do not match the completion marker")
    artifacts = _sequence(payload.get("artifacts"))
    for item in artifacts:
        if not isinstance(item, Mapping):
            raise ValueError("local evidence artifact entry is invalid")
        filename = _text(item.get("filename"))
        expected_hash = _text(item.get("sha256"))
        size_bytes = _integer(item.get("size_bytes"))
        path = _safe_artifact_path(evidence_path.parent, filename)
        if path is None or not path.is_file() or not expected_hash:
            raise ValueError("local evidence artifact is incomplete")
        if size_bytes is None or path.stat().st_size != size_bytes:
            raise ValueError("local evidence artifact size does not match")
        if sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError("local evidence artifact hash does not match")


def reviewer_case_from_packet(
    payload: Mapping[str, object], evidence_path: Path
) -> ReviewerCase | None:
    """Turn one known local packet shape into its reviewer view model."""

    record_type = payload.get("record_type")
    if record_type == "real_event_evidence":
        return _real_event_case(payload, evidence_path)
    if record_type == "local_evidence_job":
        return _local_job_case(payload, evidence_path)
    return None


def demo_cases() -> tuple[ReviewerCase, ...]:
    """Return the fixed stories used to teach the reviewer interface."""

    common_provenance: tuple[TableRow, ...] = (
        {"Item": "Source", "Value": "Deterministic development demonstration"},
        {"Item": "Scope", "Value": "Reviewer training only; not an incident feed"},
    )
    emerging = ReviewerCase(
        case_id="demo-emerging-event",
        title="Demo: emerging thermal evidence",
        source_kind="deterministic demo",
        location="Illustrative 8 by 8 km local crop (not a live location)",
        initial_ambiguity=(
            "The first observation contains one small warm region. It could be a "
            "short-lived artifact, so the loop asks for an aligned later observation."
        ),
        observations=(
            _demo_observation("first", "10:00 UTC", 9, ((2, 2), (4, 2), (4, 4))),
            _demo_observation("later", "10:10 UTC", 15, ((3, 2), (5, 2), (5, 5))),
        ),
        measurements=(
            ReviewerMeasurement("persistent candidate regions", "2", "observations"),
            ReviewerMeasurement("aligned overlap", "0.63", "IoU"),
            ReviewerMeasurement("candidate area change", "+6", "px"),
        ),
        outcome=ReviewerOutcome(
            "Review escalation",
            "review_escalation",
            0.72,
            "Persistent aligned thermal evidence reached the development review "
            "threshold. A qualified reviewer must interpret it; this is not a "
            "wildfire determination.",
            True,
        ),
        considered_actions=(
            _action_row(
                "next timestamp",
                "selected",
                "A later aligned observation can test persistence.",
            ),
            _action_row(
                "finalize",
                "not selected",
                "One observation was not enough to assess persistence.",
            ),
        ),
        selected_action="Next timestamp, then request human review",
        evidence_changes=(
            {
                "Evidence change": "Candidate region",
                "Before": "one small region",
                "After": "aligned region remains",
            },
            {
                "Evidence change": "Persistence",
                "Before": "unknown",
                "After": "2 aligned observations",
            },
        ),
        reason_codes=("thermal_evidence_persistent", "human_review_required"),
        warnings=("Development demonstration; no incident or wildfire claim is made.",),
        provenance=common_provenance,
        budget=_demo_budget(2, 2_048, 0),
    )
    matched_control = ReviewerCase(
        case_id="demo-matched-control",
        title="Demo: matched control",
        source_kind="deterministic demo",
        location="Illustrative matched local crop (not a live location)",
        initial_ambiguity=(
            "A small warm region appears at first. A matched later control is needed "
            "to distinguish persistence from a transient response."
        ),
        observations=(
            _demo_observation("first", "10:00 UTC", 10, ((2, 3), (4, 3), (4, 5))),
            _demo_observation("control", "10:10 UTC", 0, ()),
        ),
        measurements=(
            ReviewerMeasurement("persistent candidate regions", "0", "observations"),
            ReviewerMeasurement("aligned overlap", "0.00", "IoU"),
            ReviewerMeasurement("control candidate pixels", "0", "px"),
        ),
        outcome=ReviewerOutcome(
            "No persistent evidence",
            "no_persistent_evidence",
            0.0,
            "The completed usable comparison did not show a persistent thermal "
            "region at the development threshold. This does not determine whether "
            "a wildfire is present or absent.",
            True,
        ),
        considered_actions=(
            _action_row(
                "next timestamp",
                "selected",
                "The matched control tests the first observation.",
            ),
            _action_row(
                "request human review",
                "not selected",
                "No conflict or persistence remained after the control.",
            ),
        ),
        selected_action="Finalize after matched control",
        evidence_changes=(
            {
                "Evidence change": "Candidate region",
                "Before": "one small region",
                "After": "no region in control",
            },
            {
                "Evidence change": "Persistence",
                "Before": "unknown",
                "After": "not measured",
            },
        ),
        reason_codes=("thermal_anomaly_weak", "no_persistent_evidence"),
        warnings=(
            "Development demonstration; a negative thermal result is not a "
            "safety conclusion.",
        ),
        provenance=common_provenance,
        budget=_demo_budget(2, 2_048, 0),
    )
    abstention = ReviewerCase(
        case_id="demo-abstention",
        title="Demo: safe abstention",
        source_kind="deterministic demo",
        location="Illustrative partial-coverage crop (not a live location)",
        initial_ambiguity=(
            "Part of the local crop is unavailable. The visible warm pixels cannot "
            "support a reliable temporal comparison."
        ),
        observations=(
            _demo_observation("partial", "10:00 UTC", 7, ((1, 1), (3, 1), (3, 3))),
        ),
        measurements=(
            ReviewerMeasurement("usable coverage", "0.41", "fraction"),
            ReviewerMeasurement("candidate pixels", "7", "px"),
            ReviewerMeasurement("usable comparisons", "0", "observations"),
        ),
        outcome=ReviewerOutcome(
            "Insufficient evidence - abstained",
            "insufficient_evidence",
            0.0,
            "The loop safely abstained because coverage was below the development "
            "limit. It does not strengthen this thermal observation into a "
            "wildfire claim.",
            True,
        ),
        considered_actions=(
            _action_row(
                "next timestamp", "not selected", "The observation budget is exhausted."
            ),
            _action_row(
                "abstain", "selected", "Poor coverage prevents a reliable comparison."
            ),
        ),
        selected_action="Abstain",
        evidence_changes=(
            {
                "Evidence change": "Coverage",
                "Before": "required comparison",
                "After": "41% usable",
            },
            {
                "Evidence change": "Decision",
                "Before": "ambiguous",
                "After": "safely abstained",
            },
        ),
        reason_codes=(
            "coverage_insufficient",
            "budget_exhausted",
            "insufficient_evidence",
        ),
        warnings=(
            "Observation budget exhausted; no further observation was requested.",
        ),
        provenance=common_provenance,
        budget=_demo_budget(1, 1_024, 0, maximum_observations=1),
    )
    return emerging, matched_control, abstention


def reason_explanations(reason_codes: Sequence[str]) -> tuple[str, ...]:
    """Translate known closed codes through the fixed Day 21 templates."""

    known: list[ReasonCode] = []
    for reason in reason_codes:
        try:
            known.append(ReasonCode(reason))
        except ValueError:
            continue
    if not known:
        return ()
    return explain_reason_codes(tuple(dict.fromkeys(known)))


def contour_preview(observation: ReviewerObservation) -> ImageArray:
    """Render packet contour coordinates as a deterministic small mask preview."""

    height = 112
    width = 160
    canvas = np.full((height, width, 3), 18, dtype=np.uint8)
    points = [point for contour in observation.contours for point in contour]
    if not points:
        return canvas
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    span_x = max(max(xs) - min(xs), 1)
    span_y = max(max(ys) - min(ys), 1)

    def scale(point: tuple[int, int]) -> tuple[int, int]:
        x = 12 + round((point[0] - min(xs)) * (width - 24) / span_x)
        y = 12 + round((point[1] - min(ys)) * (height - 24) / span_y)
        return x, y

    for contour in observation.contours:
        if len(contour) == 1:
            x, y = scale(contour[0])
            canvas[y, x] = (0, 210, 255)
            continue
        for first, second in zip(contour, (*contour[1:], contour[0]), strict=True):
            _draw_line(canvas, scale(first), scale(second))
    return canvas


def load_candidate_mask(path: Path | None) -> ImageArray | None:
    """Load a local candidate mask only when it is a safe two-dimensional array."""

    if path is None or not path.is_file():
        return None
    try:
        array = np.load(path, allow_pickle=False)
    except (OSError, ValueError):
        return None
    if array.ndim != 2:
        return None
    mask = np.asarray(array > 0, dtype=np.uint8) * np.uint8(255)
    return np.repeat(mask[:, :, np.newaxis], 3, axis=2)


def _real_event_case(
    payload: Mapping[str, object], evidence_path: Path
) -> ReviewerCase:
    frames = _sequence(payload.get("frames"))
    observations = tuple(
        _real_observation(frame, evidence_path.parent, index)
        for index, frame in enumerate(frames)
        if isinstance(frame, Mapping)
    )
    coordinates = _mapping(payload.get("coordinates"))
    location = (
        _coordinate_text(coordinates) or "Location is not included in this packet"
    )
    panel = _mapping(payload.get("reviewer_panel"))
    panel_path = _safe_artifact_path(evidence_path.parent, _text(panel.get("filename")))
    configuration = _mapping(payload.get("configuration"))
    provenance = _base_provenance(
        evidence_path,
        configuration,
        _text(payload.get("content_hash")),
        _text(payload.get("opencv_version")),
    )
    source_hashes = sorted(
        {
            _source_hash_from_frame(frame)
            for frame in frames
            if isinstance(frame, Mapping) and _source_hash_from_frame(frame)
        }
    )
    provenance += tuple(
        {"Item": f"Source hash {index + 1}", "Value": source}
        for index, source in enumerate(source_hashes)
    )
    return ReviewerCase(
        case_id=_text(payload.get("case_id")) or evidence_path.parent.parent.name,
        title=_text(payload.get("title")) or "Historical thermal evidence packet",
        source_kind="historical evidence packet",
        location=location,
        initial_ambiguity=(
            "This historical Channel 7 slice shows changing warm-region "
            "measurements. It is retained for manual review context, not an "
            "operational determination."
        ),
        observations=observations,
        measurements=_packet_measurements(payload.get("measurements")),
        outcome=ReviewerOutcome(
            "Historical evidence context only",
            None,
            None,
            "No bounded agent outcome is attached to this packet. The thermal slice "
            "is not confirmation of a wildfire.",
            False,
        ),
        considered_actions=(),
        selected_action=None,
        evidence_changes=_real_evidence_changes(observations),
        reason_codes=(),
        warnings=(
            "Historical, manually audited display packet; it is not a live alert.",
            "Channel 7 thermal measurements alone do not confirm a wildfire.",
        ),
        provenance=provenance,
        budget=(),
        reviewer_panel_path=panel_path,
    )


def _local_job_case(payload: Mapping[str, object], evidence_path: Path) -> ReviewerCase:
    raw_observations = _sequence(payload.get("observations"))
    observations = tuple(
        _local_observation(item, evidence_path.parent, index)
        for index, item in enumerate(raw_observations)
        if isinstance(item, Mapping)
    )
    persistence = _mapping(payload.get("persistence"))
    loop = _loop_summary(evidence_path, _text(payload.get("case_id")))
    persistence_reasons = _text_items(persistence.get("reason_codes"))
    all_reasons = tuple(
        dict.fromkeys(
            reason
            for observation in observations
            for reason in (*observation.reason_codes, *persistence_reasons)
        )
    )
    outcome = loop.outcome
    if outcome is None:
        outcome = ReviewerOutcome(
            "Evidence packet ready for bounded decision",
            None,
            None,
            "No complete bounded-loop outcome was found beside this packet. The "
            "displayed thermal evidence remains unconfirmed and needs the next "
            "bounded decision or reviewer interpretation.",
            False,
        )
    reason_codes = tuple(dict.fromkeys((*all_reasons, *loop.reason_codes)))
    warnings = tuple(
        dict.fromkeys((*_text_items(payload.get("warnings")), *loop.warnings))
    )
    configuration = _mapping(payload.get("configuration"))
    case_id = _text(payload.get("case_id")) or evidence_path.parent.parent.name
    return ReviewerCase(
        case_id=case_id,
        title=f"Local evidence: {case_id}",
        source_kind="local evidence job",
        location=_local_location(raw_observations),
        initial_ambiguity=_local_ambiguity(observations, persistence),
        observations=observations,
        measurements=_local_measurements(observations, persistence),
        outcome=outcome,
        considered_actions=loop.considered_actions,
        selected_action=loop.selected_action,
        evidence_changes=loop.evidence_changes or _local_evidence_changes(persistence),
        reason_codes=reason_codes,
        warnings=warnings,
        provenance=_base_provenance(
            evidence_path,
            configuration,
            _text(payload.get("content_hash")),
            None,
        ),
        budget=loop.budget,
        errors=loop.errors,
        recovery_actions=loop.recovery_actions,
    )


@dataclass(frozen=True, slots=True)
class _LoopSummary:
    outcome: ReviewerOutcome | None
    considered_actions: tuple[TableRow, ...]
    selected_action: str | None
    evidence_changes: tuple[TableRow, ...]
    reason_codes: tuple[str, ...]
    warnings: tuple[str, ...]
    budget: tuple[TableRow, ...]
    errors: tuple[str, ...] = ()
    recovery_actions: tuple[str, ...] = ()


def _trace_only_case(trace_path: Path) -> ReviewerCase | None:
    """Expose a failed or interrupted loop even when no evidence packet exists."""

    checkpoint = load_last_complete_transition(trace_path)
    if checkpoint is None:
        return None
    case_id = _text(checkpoint.get("case_id"))
    if not case_id:
        return None
    loop = _loop_summary_from_trace(trace_path, case_id)
    outcome = loop.outcome or ReviewerOutcome(
        "Bounded loop has no completed outcome",
        None,
        None,
        "The loop stopped before producing a completed local evidence packet. "
        "Do not treat any staged or partial files as evidence.",
        False,
    )
    return ReviewerCase(
        case_id=case_id,
        title=f"Bounded loop: {case_id}",
        source_kind="bounded agent trace",
        location=(
            "No completed evidence packet recorded location context for this trace."
        ),
        initial_ambiguity=(
            "The bounded loop encountered a predictable failure before it could "
            "publish a complete evidence packet."
        ),
        observations=(),
        measurements=(),
        outcome=outcome,
        considered_actions=loop.considered_actions,
        selected_action=loop.selected_action,
        evidence_changes=loop.evidence_changes,
        reason_codes=loop.reason_codes,
        warnings=loop.warnings,
        provenance=({"Item": "Loop trace", "Value": str(trace_path)},),
        budget=loop.budget,
        errors=loop.errors,
        recovery_actions=loop.recovery_actions,
    )


def _loop_summary(evidence_path: Path, case_id: str) -> _LoopSummary:
    candidates = (
        evidence_path.parent / "agent-loop.jsonl",
        evidence_path.parent.parent / "agent-loop.jsonl",
    )
    trace_path = next((path for path in candidates if path.is_file()), None)
    if trace_path is None:
        return _LoopSummary(None, (), None, (), (), (), ())
    return _loop_summary_from_trace(trace_path, case_id)


def _loop_summary_from_trace(trace_path: Path, case_id: str) -> _LoopSummary:
    """Summarize the last valid checkpoint and its most recent tool failure."""

    try:
        checkpoint = load_last_complete_transition(trace_path)
    except ValueError as error:
        return _LoopSummary(
            None, (), None, (), (), (f"Loop trace unreadable: {error}",), ()
        )
    if checkpoint is None:
        return _LoopSummary(None, (), None, (), (), ("Loop trace is empty.",), ())
    if checkpoint.get("case_id") != case_id:
        return _LoopSummary(
            None,
            (),
            None,
            (),
            (),
            ("Loop trace case does not match this evidence packet.",),
            (),
        )
    outcome_data = _mapping(checkpoint.get("outcome"))
    raw_reasons = _text_items(outcome_data.get("reason_codes"))
    outcome = None
    if outcome_data:
        state = _text(outcome_data.get("state"))
        confidence = _number(outcome_data.get("confidence"))
        outcome = ReviewerOutcome(
            _outcome_label(state),
            state or None,
            confidence,
            _text(outcome_data.get("explanation"))
            or "A bounded reviewer outcome was recorded.",
            _text(checkpoint.get("to_state"))
            in {"finalize", "abstain", "review", "failure"},
        )
    decision = _mapping(checkpoint.get("decision"))
    selected = _mapping(decision.get("selected_action"))
    selected_action = _action_name(selected) or None
    considered: list[TableRow] = []
    for item in _sequence(decision.get("considered_actions")):
        if isinstance(item, Mapping):
            considered.append(
                _action_row(
                    _action_name(item) or "bounded action",
                    _text(item.get("status")) or "considered",
                    _text(item.get("reason")) or "No reason was recorded.",
                )
            )
    evidence_changes: tuple[TableRow, ...] = tuple(
        {
            "Evidence change": _text(change.get("field")) or "evidence",
            "Before": _display_value(change.get("before")),
            "After": _display_value(change.get("after")),
        }
        for change in _sequence(decision.get("evidence_changes"))
        if isinstance(change, Mapping)
    )
    budget = _budget_rows(_mapping(checkpoint.get("budget")))
    error_data = _latest_trace_error(trace_path)
    errors: tuple[str, ...] = ()
    recovery_actions: tuple[str, ...] = ()
    if error_data:
        error_code = _text(error_data.get("code"))
        detail = _text(error_data.get("detail"))
        errors = (
            f"{error_code.replace('_', ' ') or 'bounded tool error'}: "
            f"{detail or 'No detail was recorded.'}",
        )
        recovery = _text(error_data.get("recovery_action"))
        if not recovery and error_code:
            try:
                recovery = recovery_action_for_tool_error(error_code)
            except ValueError:
                recovery = "Do not use partial evidence; inspect the recorded error."
        recovery_actions = (recovery,) if recovery else ()
    return _LoopSummary(
        outcome,
        tuple(considered),
        selected_action,
        evidence_changes,
        raw_reasons,
        (),
        budget,
        errors,
        recovery_actions,
    )


def _latest_trace_error(trace_path: Path) -> Mapping[str, object]:
    """Return the last persisted tool error, even after a terminal abstention."""

    latest: Mapping[str, object] = {}
    try:
        lines = trace_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return latest
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            break
        if not isinstance(record, Mapping):
            continue
        error = _mapping(_mapping(record.get("last_tool_result")).get("error"))
        if error:
            latest = error
    return latest


def _real_observation(
    frame: Mapping[str, object], root: Path, index: int
) -> ReviewerObservation:
    del root
    components = _components(frame.get("components"))
    contours = _contours(frame.get("contours_xy"))
    pixel_count = _integer(frame.get("morphology_pixel_count"))
    return ReviewerObservation(
        observation_id=_text(frame.get("observation_id")) or f"frame-{index + 1}",
        observed_at=_text(frame.get("scan_start")) or "time unavailable",
        channel=_text(frame.get("channel")) or "thermal channel",
        candidate_pixels=pixel_count
        if pixel_count is not None
        else sum(item.area_pixels for item in components),
        components=components,
        contours=contours,
        reason_codes=(),
        maximum_kelvin=_number(frame.get("maximum_kelvin")),
    )


def _local_observation(
    item: Mapping[str, object], root: Path, index: int
) -> ReviewerObservation:
    anomaly = _mapping(item.get("anomaly"))
    crop = _mapping(item.get("channel7_crop"))
    timing = _mapping(crop.get("timing"))
    observation_id = _text(item.get("observation_id")) or f"observation-{index + 1}"
    directory = _safe_artifact_path(root, f"observations/{observation_id}")
    overlay = None if directory is None else directory / "overlay.png"
    candidate_mask = None if directory is None else directory / "candidate-mask.npy"
    return ReviewerObservation(
        observation_id=observation_id,
        observed_at=_text(timing.get("start")) or "time unavailable",
        channel="C07 and C14 contextual comparison",
        candidate_pixels=_integer(anomaly.get("candidate_pixel_count")) or 0,
        components=_components(anomaly.get("components")),
        contours=_contours(anomaly.get("contours_xy")),
        reason_codes=_text_items(anomaly.get("reason_codes")),
        overlay_path=overlay if overlay is not None and overlay.is_file() else None,
        candidate_mask_path=(
            candidate_mask
            if candidate_mask is not None and candidate_mask.is_file()
            else None
        ),
    )


def _components(value: object) -> tuple[ReviewerComponent, ...]:
    components: list[ReviewerComponent] = []
    for index, item in enumerate(_sequence(value)):
        if not isinstance(item, Mapping):
            continue
        components.append(
            ReviewerComponent(
                _integer(item.get("label")) or index + 1,
                _integer(item.get("area_pixels")) or 0,
                _coordinate_list(item.get("bounding_box_xywh")),
                _coordinate_list(item.get("centroid_xy")),
            )
        )
    return tuple(components)


def _contours(value: object) -> tuple[Contour, ...]:
    result: list[Contour] = []
    for raw_contour in _sequence(value):
        points: list[tuple[int, int]] = []
        for raw_point in _sequence(raw_contour):
            coordinates = _sequence(raw_point)
            if len(coordinates) != 2:
                continue
            x = _integer(coordinates[0])
            y = _integer(coordinates[1])
            if x is not None and y is not None:
                points.append((x, y))
        if points:
            result.append(tuple(points))
    return tuple(result)


def _packet_measurements(value: object) -> tuple[ReviewerMeasurement, ...]:
    measurements: list[ReviewerMeasurement] = []
    for item in _sequence(value):
        if not isinstance(item, Mapping):
            continue
        raw_value = item.get("value")
        measurements.append(
            ReviewerMeasurement(
                _text(item.get("name")) or "measurement",
                _display_value(raw_value),
                _text(item.get("unit")) or "—",
                _text(item.get("missing_reason")) or None,
            )
        )
    return tuple(measurements)


def _local_measurements(
    observations: Sequence[ReviewerObservation], persistence: Mapping[str, object]
) -> tuple[ReviewerMeasurement, ...]:
    measurements: list[ReviewerMeasurement] = []
    for observation in observations:
        measurements.append(
            ReviewerMeasurement(
                f"{observation.observation_id} candidate pixels",
                str(observation.candidate_pixels),
                "px",
            )
        )
    for key, unit in (
        ("persistence_count", "observations"),
        ("mean_intersection_over_union", "IoU"),
        ("confidence", "confidence"),
    ):
        value = persistence.get(key)
        if value is not None:
            measurements.append(ReviewerMeasurement(key, _display_value(value), unit))
    return tuple(measurements)


def _local_location(observations: Sequence[object]) -> str:
    for observation in observations:
        crop = _mapping(_mapping(observation).get("channel7_crop"))
        bounds = _mapping(crop.get("geographic_bounds"))
        if not bounds:
            bounds = _mapping(crop.get("requested_bounds"))
        south = _number(bounds.get("south"))
        west = _number(bounds.get("west"))
        north = _number(bounds.get("north"))
        east = _number(bounds.get("east"))
        if None not in (south, west, north, east):
            assert south is not None and west is not None
            assert north is not None and east is not None
            return (
                "Local crop: "
                f"{south:.4f}° to {north:.4f}° latitude; "
                f"{west:.4f}° to {east:.4f}° longitude"
            )
    return "Local geographic crop; coordinates were not recorded in this packet"


def _local_ambiguity(
    observations: Sequence[ReviewerObservation], persistence: Mapping[str, object]
) -> str:
    candidate_count = sum(observation.candidate_pixels for observation in observations)
    persistence_count = _integer(persistence.get("persistence_count")) or 0
    return (
        f"{candidate_count} candidate pixels were measured across "
        f"{len(observations)} observation(s). Persistence count is "
        f"{persistence_count}; bounded policy and quality checks determine whether "
        "that is adequate for review."
    )


def _real_evidence_changes(
    observations: Sequence[ReviewerObservation],
) -> tuple[TableRow, ...]:
    if len(observations) < 2:
        return ()
    first = observations[0]
    last = observations[-1]
    return (
        {
            "Evidence change": "Candidate-mask pixels",
            "Before": str(first.candidate_pixels),
            "After": str(last.candidate_pixels),
        },
        {
            "Evidence change": "Retained components",
            "Before": str(len(first.components)),
            "After": str(len(last.components)),
        },
    )


def _local_evidence_changes(persistence: Mapping[str, object]) -> tuple[TableRow, ...]:
    changes: list[TableRow] = []
    for key in ("area_trend", "temperature_trend", "mean_intersection_over_union"):
        if key in persistence:
            changes.append(
                {
                    "Evidence change": key.replace("_", " "),
                    "Before": "across observations",
                    "After": _display_value(persistence.get(key)),
                }
            )
    return tuple(changes)


def _base_provenance(
    evidence_path: Path,
    configuration: Mapping[str, object],
    content_hash: str,
    opencv_version: str | None,
) -> tuple[TableRow, ...]:
    values: list[TableRow] = [
        {"Item": "Evidence packet", "Value": str(evidence_path)},
    ]
    configuration_id = _text(configuration.get("configuration_id"))
    if configuration_id:
        values.append({"Item": "Configuration", "Value": configuration_id})
    content = _text(configuration.get("content_hash")) or content_hash
    if content:
        values.append({"Item": "Content hash", "Value": content})
    if opencv_version:
        values.append({"Item": "OpenCV version", "Value": opencv_version})
    return tuple(values)


def _source_hash_from_frame(frame: object) -> str:
    return _text(_mapping(frame).get("source_sha256"))


def _coordinate_text(coordinates: Mapping[str, object]) -> str:
    latitude = _number(coordinates.get("lat"))
    longitude = _number(coordinates.get("lon"))
    if latitude is None or longitude is None:
        return ""
    return f"{latitude:.4f}°, {longitude:.4f}°"


def _safe_artifact_path(root: Path, relative: str) -> Path | None:
    if not relative:
        return None
    candidate = (root / relative).resolve()
    resolved_root = root.resolve()
    return candidate if candidate.is_relative_to(resolved_root) else None


def _draw_line(
    canvas: ImageArray, start: tuple[int, int], end: tuple[int, int]
) -> None:
    steps = max(abs(end[0] - start[0]), abs(end[1] - start[1]), 1) + 1
    xs = np.rint(np.linspace(start[0], end[0], steps)).astype(np.intp)
    ys = np.rint(np.linspace(start[1], end[1], steps)).astype(np.intp)
    canvas[ys, xs] = (0, 210, 255)


def _demo_observation(
    observation_id: str, observed_at: str, pixels: int, contour: Contour
) -> ReviewerObservation:
    components = () if not contour else (ReviewerComponent(1, pixels, "demo", "demo"),)
    contours = () if not contour else (contour,)
    return ReviewerObservation(
        observation_id,
        observed_at,
        "C07/C14 demo comparison",
        pixels,
        components,
        contours,
        (),
    )


def _demo_budget(
    observations: int,
    bytes_used: int,
    retries: int,
    *,
    maximum_observations: int = 3,
) -> tuple[TableRow, ...]:
    return (
        {"Budget": "Observations", "Used": observations, "Limit": maximum_observations},
        {"Budget": "Bytes", "Used": bytes_used, "Limit": 4_096},
        {"Budget": "Retries", "Used": retries, "Limit": 0},
    )


def _action_row(action: str, status: str, reason: str) -> TableRow:
    return {"Action": action, "Status": status, "Why": reason}


def _action_name(value: Mapping[str, object]) -> str:
    action = _text(value.get("action_type"))
    observation_id = _text(value.get("observation_id"))
    return " ".join(part for part in (action.replace("_", " "), observation_id) if part)


def _budget_rows(budget: Mapping[str, object]) -> tuple[TableRow, ...]:
    rows: list[TableRow] = []
    fields = (
        ("Observations", "used_observations", "max_observations"),
        ("Bytes", "used_bytes", "max_bytes"),
        ("Elapsed seconds", "used_elapsed_seconds", "max_elapsed_seconds"),
        ("Retries", "used_retries", "max_retries"),
    )
    for label, used, maximum in fields:
        if used in budget and maximum in budget:
            rows.append(
                {
                    "Budget": label,
                    "Used": _display_value(budget.get(used)),
                    "Limit": _display_value(budget.get(maximum)),
                }
            )
    return tuple(rows)


def _outcome_label(state: str) -> str:
    return {
        "review_escalation": "Review escalation",
        "human_review": "Human review",
        "no_persistent_evidence": "No persistent evidence",
        "insufficient_evidence": "Insufficient evidence - abstained",
        "failed": "Processing failed",
    }.get(state, "Bounded outcome")


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> Sequence[object]:
    return value if isinstance(value, (list, tuple)) else ()


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _text_items(value: object) -> tuple[str, ...]:
    return tuple(item for item in _sequence(value) if isinstance(item, str) and item)


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _integer(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _coordinate_list(value: object) -> str:
    values = _sequence(value)
    return "(" + ", ".join(_display_value(item) for item in values) + ")"


def _display_value(value: object) -> str:
    if value is None:
        return "not recorded"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (str, int, bool)):
        return str(value)
    return "not recorded"


__all__ = [
    "ReviewerCase",
    "ReviewerCatalog",
    "ReviewerComponent",
    "ReviewerMeasurement",
    "ReviewerObservation",
    "ReviewerOutcome",
    "contour_preview",
    "demo_cases",
    "discover_reviewer_cases",
    "load_candidate_mask",
    "reason_explanations",
    "reviewer_case_from_packet",
]
