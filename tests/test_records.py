"""Golden contracts for the Day 4 evidence and trace records."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from firesentinel.core.records import (
    Action,
    ActionType,
    Budget,
    Channel,
    ConfigurationReference,
    Coordinates,
    ManifestCase,
    Measurement,
    ObservationRequest,
    Outcome,
    OutcomeState,
    ReasonCode,
    RecordValidationError,
    SourceObject,
    Trace,
    Unit,
    VisionEvidence,
    artifact_directory,
    canonical_content_hash,
    record_from_json,
)

HASH_A = "a" * 64
HASH_B = "b" * 64
TIME = datetime(2026, 8, 19, 12, 0, tzinfo=UTC)
GoldenRecords = tuple[
    ManifestCase,
    ObservationRequest,
    SourceObject,
    VisionEvidence,
    Action,
    Budget,
    Outcome,
    Trace,
]


@pytest.fixture
def golden_records() -> GoldenRecords:
    coordinates = Coordinates(latitude=38.5, longitude=-120.2)
    configuration = ConfigurationReference("vision-v1", HASH_A)
    case = ManifestCase(
        case_id="sierra-20260819",
        title="Sierra historical replay",
        location=coordinates,
        created_at=TIME,
        content_hash=HASH_B,
        allowed_observation_ids=("obs-001",),
    )
    request = ObservationRequest(
        observation_id="obs-001",
        case_id=case.case_id,
        requested_at=TIME,
        observation_time=TIME + timedelta(minutes=5),
        channel=Channel.C07,
        coordinates=coordinates,
    )
    source = SourceObject(
        source_id="source-001",
        observation_id=request.observation_id,
        provider="noaa-goes18",
        bucket="noaa-goes18",
        object_key="ABI-L2-CMIPF/2026/example.nc",
        content_hash=HASH_A,
        size_bytes=1_024,
        scan_time=request.observation_time,
        discovered_at=request.observation_time + timedelta(minutes=1),
    )
    evidence = VisionEvidence(
        evidence_id="evidence-001",
        case_id=case.case_id,
        observation_id=request.observation_id,
        source_id=source.source_id,
        configuration=configuration,
        created_at=source.discovered_at,
        coordinates=coordinates,
        measurements=(
            Measurement("hot_region_area", 0.42, Unit.SQUARE_KILOMETRES),
            Measurement("thermal_delta", None, Unit.KELVIN, ReasonCode.CONTRAST_LOW),
        ),
        confidence=0.65,
        reason_codes=(ReasonCode.THERMAL_ANOMALY_WEAK, ReasonCode.CONTRAST_LOW),
        content_hash=HASH_B,
    )
    action = Action(
        action_id="action-001",
        case_id=case.case_id,
        action_type=ActionType.NEXT_TIMESTAMP,
        created_at=evidence.created_at,
        reason_codes=(ReasonCode.THERMAL_ANOMALY_WEAK,),
        evidence_ids=(evidence.evidence_id,),
        selected=True,
    )
    budget = Budget(
        max_observations=3,
        used_observations=1,
        max_bytes=10_000,
        used_bytes=1_024,
        max_elapsed_seconds=60.0,
        used_elapsed_seconds=3.5,
        max_retries=1,
        used_retries=0,
    )
    outcome = Outcome(
        outcome_id="outcome-001",
        trace_id="trace-001",
        case_id=case.case_id,
        state=OutcomeState.INSUFFICIENT_EVIDENCE,
        created_at=evidence.created_at + timedelta(seconds=1),
        evidence_ids=(evidence.evidence_id,),
        configuration=configuration,
        confidence=0.0,
        reason_codes=(ReasonCode.INSUFFICIENT_EVIDENCE,),
    )
    trace = Trace(
        trace_id=outcome.trace_id,
        case=case,
        configuration=configuration,
        started_at=TIME,
        completed_at=outcome.created_at,
        observation_requests=(request,),
        sources=(source,),
        evidence=(evidence,),
        actions=(action,),
        budget=budget,
        outcome=outcome,
    )
    return case, request, source, evidence, action, budget, outcome, trace


def test_golden_records_round_trip_through_canonical_json(
    golden_records: GoldenRecords,
) -> None:
    for record in golden_records:
        serialized = record.to_json()

        assert type(record).from_json(serialized) == record
        assert record_from_json(serialized) == record
        assert json.loads(serialized)["schema_version"] == 1


def test_timestamps_coordinates_units_nulls_and_confidence_are_standardized(
    golden_records: GoldenRecords,
) -> None:
    evidence = golden_records[3]
    encoded = evidence.to_dict()

    assert encoded["created_at"].endswith("Z")
    assert encoded["coordinates"] == {"lat": 38.5, "lon": -120.2}
    assert encoded["measurements"][0]["unit"] == "km2"
    assert encoded["measurements"][1]["value"] is None
    assert 0.0 <= encoded["confidence"] <= 1.0


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: Coordinates(91, 0), "latitude"),
        (
            lambda: Measurement("thermal_delta", None, Unit.KELVIN),
            "missing_reason",
        ),
        (
            lambda: Measurement("thermal_delta", 1.0, Unit.KELVIN, ReasonCode.VALID),
            "missing_reason",
        ),
        (
            lambda: ObservationRequest(
                "obs-002",
                "sierra-20260819",
                TIME.replace(tzinfo=None),
                TIME,
                Channel.C07,
                Coordinates(0, 0),
            ),
            "UTC",
        ),
        (
            lambda: Budget(1, 2, 1, 0, 1.0, 0.0, 0, 0),
            "used_observations",
        ),
        (
            lambda: VisionEvidence(
                "evidence-002",
                "sierra-20260819",
                "obs-001",
                "source-001",
                ConfigurationReference("vision-v1", HASH_A),
                TIME,
                Coordinates(0, 0),
                (Measurement("coverage", 1.0, Unit.DIMENSIONLESS),),
                1.01,
                (ReasonCode.VALID,),
                HASH_B,
            ),
            "confidence",
        ),
    ],
)
def test_invalid_values_fail_validation(
    factory: Callable[[], object], match: str
) -> None:
    with pytest.raises(RecordValidationError, match=match):
        factory()


def test_trace_rejects_outcome_without_matching_evidence_or_configuration(
    golden_records: GoldenRecords,
) -> None:
    trace = golden_records[7]
    bad_outcome = Outcome(
        outcome_id="outcome-002",
        trace_id=trace.trace_id,
        case_id=trace.case.case_id,
        state=OutcomeState.INSUFFICIENT_EVIDENCE,
        created_at=trace.completed_at,
        evidence_ids=("evidence-unknown",),
        configuration=trace.configuration,
        confidence=0.0,
        reason_codes=(ReasonCode.INSUFFICIENT_EVIDENCE,),
    )
    with pytest.raises(RecordValidationError, match="outcome.evidence_ids"):
        Trace(
            trace_id=trace.trace_id,
            case=trace.case,
            configuration=trace.configuration,
            started_at=trace.started_at,
            completed_at=trace.completed_at,
            observation_requests=trace.observation_requests,
            sources=trace.sources,
            evidence=trace.evidence,
            actions=trace.actions,
            budget=trace.budget,
            outcome=bad_outcome,
        )

    mismatched_configuration = ConfigurationReference("vision-v2", HASH_B)
    bad_configuration_outcome = Outcome(
        outcome_id="outcome-003",
        trace_id=trace.trace_id,
        case_id=trace.case.case_id,
        state=OutcomeState.INSUFFICIENT_EVIDENCE,
        created_at=trace.completed_at,
        evidence_ids=(trace.evidence[0].evidence_id,),
        configuration=mismatched_configuration,
        confidence=0.0,
        reason_codes=(ReasonCode.INSUFFICIENT_EVIDENCE,),
    )
    with pytest.raises(RecordValidationError, match="outcome.configuration"):
        Trace(
            trace_id=trace.trace_id,
            case=trace.case,
            configuration=trace.configuration,
            started_at=trace.started_at,
            completed_at=trace.completed_at,
            observation_requests=trace.observation_requests,
            sources=trace.sources,
            evidence=trace.evidence,
            actions=trace.actions,
            budget=trace.budget,
            outcome=bad_configuration_outcome,
        )


def test_artifact_directory_is_case_and_hash_scoped(
    tmp_path: Path, golden_records: GoldenRecords
) -> None:
    trace = golden_records[7]
    trace_hash = canonical_content_hash(trace)

    assert artifact_directory(tmp_path, trace.case.case_id, trace_hash) == (
        tmp_path / trace.case.case_id / trace_hash
    )
    with pytest.raises(RecordValidationError, match="case_id"):
        artifact_directory(tmp_path, "../escape", trace_hash)
