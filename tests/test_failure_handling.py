"""Injected predictable-failure contracts for bounded local evidence review."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

import firesentinel.agent.tools as agent_tools
from firesentinel.agent.loop import AgentLoopState, BoundedAgentLoop
from firesentinel.agent.tools import ToolErrorCode, ToolManifest
from firesentinel.core.records import ReasonCode
from firesentinel.ui.reviewer import discover_reviewer_cases
from firesentinel.vision.engine import (
    EvidenceJob,
    EvidenceJobFailure,
    EvidenceJobResult,
    EvidenceJobTimeout,
)
from firesentinel.vision.persistence import (
    GeospatialGrid,
    PersistenceParameters,
    TemporalObservation,
    measure_temporal_persistence,
)
from tests.test_agent_tools import _manifest


def _open_loop(
    manifest: ToolManifest,
    *,
    project_root: Path,
    cache_root: Path,
    maximum_bytes: int,
) -> BoundedAgentLoop:
    return BoundedAgentLoop.open(
        manifest,
        source_cache_root=cache_root,
        artifacts_root=project_root / "artifacts",
        project_root=project_root,
        trace_path=project_root
        / "artifacts"
        / manifest.case.case_id
        / "agent-loop.jsonl",
        maximum_bytes=maximum_bytes,
        maximum_elapsed_seconds=60.0,
        maximum_observations=3,
    )


def _trace_records(trace_path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line
    ]


@pytest.mark.parametrize(
    ("name", "prepare", "expected_error"),
    [
        (
            "missing source",
            lambda _: None,
            ToolErrorCode.SOURCE_UNAVAILABLE,
        ),
        (
            "corrupt cache entry",
            lambda source: source.write_bytes(b"changed cached bytes"),
            ToolErrorCode.SOURCE_CORRUPT,
        ),
    ],
)
def test_cache_failures_abstain_and_remain_visible_to_the_reviewer(
    tmp_path: Path,
    name: str,
    prepare: Callable[[Path], None],
    expected_error: ToolErrorCode,
) -> None:
    del name
    manifest, project_root, cache_root, source_path = _manifest(tmp_path)
    maximum_bytes = source_path.stat().st_size * 8
    if expected_error is ToolErrorCode.SOURCE_UNAVAILABLE:
        source_path.unlink()
    else:
        prepare(source_path)

    result = _open_loop(
        manifest,
        project_root=project_root,
        cache_root=cache_root,
        maximum_bytes=maximum_bytes,
    ).run()

    assert result.state is AgentLoopState.ABSTAIN
    assert result.outcome is not None
    assert result.outcome.reason_codes[-1] is ReasonCode.INSUFFICIENT_EVIDENCE
    records = _trace_records(result.trace_path)
    errors = [
        record["last_tool_result"]["error"]
        for record in records
        if isinstance(record["last_tool_result"], dict)
        and isinstance(record["last_tool_result"].get("error"), dict)
    ]
    assert errors
    assert errors[-1]["code"] == expected_error.value
    assert errors[-1]["recovery_action"]
    assert not list(result.trace_path.parent.glob("*/completion.json"))

    catalog = discover_reviewer_cases(project_root / "artifacts")
    case = next(item for item in catalog.cases if item.case_id == manifest.case.case_id)
    assert case.outcome.state == "insufficient_evidence"
    assert case.errors
    assert case.recovery_actions


@pytest.mark.parametrize(
    ("failure", "expected_error"),
    [
        (
            EvidenceJobTimeout(0.001),
            ToolErrorCode.ELAPSED_TIME_EXHAUSTED,
        ),
        (
            EvidenceJobFailure(
                ReasonCode.ARTIFACT_WRITE_FAILED,
                "insufficient disk space while writing local evidence",
            ),
            ToolErrorCode.INSUFFICIENT_DISK,
        ),
    ],
)
def test_timeout_and_disk_failures_abstain_without_publishing_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: EvidenceJobFailure,
    expected_error: ToolErrorCode,
) -> None:
    manifest, project_root, cache_root, source_path = _manifest(tmp_path)

    def fail_evidence_job(*_: object, **__: object) -> EvidenceJobResult:
        raise failure

    monkeypatch.setattr(agent_tools, "run_evidence_job", fail_evidence_job)
    result = _open_loop(
        manifest,
        project_root=project_root,
        cache_root=cache_root,
        maximum_bytes=source_path.stat().st_size * 8,
    ).run()

    assert result.state is AgentLoopState.ABSTAIN
    assert result.outcome is not None
    assert not result.evidence_ids
    records = _trace_records(result.trace_path)
    error = next(
        record["last_tool_result"]["error"]
        for record in records
        if isinstance(record["last_tool_result"], dict)
        and isinstance(record["last_tool_result"].get("error"), dict)
    )
    assert error["code"] == expected_error.value
    assert error["recovery_action"]
    assert not list(result.trace_path.parent.glob("*/completion.json"))


@pytest.mark.parametrize(
    "reason",
    (ReasonCode.COVERAGE_INSUFFICIENT, ReasonCode.ALIGNMENT_FAILED),
)
def test_unusable_or_unaligned_evidence_uses_one_retry_then_abstains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reason: ReasonCode
) -> None:
    manifest, project_root, cache_root, source_path = _manifest(tmp_path)
    calls: list[tuple[str, ...]] = []

    def quality_limited_evidence(
        job: EvidenceJob, artifacts_root: Path, **_: object
    ) -> EvidenceJobResult:
        observation_ids = tuple(item.observation_id for item in job.observations)
        calls.append(observation_ids)
        evidence_id = f"{len(calls):064x}"
        destination = artifacts_root / job.case_id / evidence_id
        destination.mkdir(parents=True)
        payload = {
            "observations": [
                {
                    "anomaly": {
                        "candidate_pixel_count": 0,
                        "components": [],
                        "reason_codes": [reason.value]
                        if reason is not ReasonCode.ALIGNMENT_FAILED
                        else [],
                    }
                }
                for _ in observation_ids
            ],
            "persistence": {
                "persistence_count": 0,
                "confidence": 0.0,
                "reason_codes": [reason.value]
                if reason is ReasonCode.ALIGNMENT_FAILED
                else [],
            },
        }
        (destination / "evidence.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        return EvidenceJobResult(job.case_id, evidence_id, destination, False, ())

    monkeypatch.setattr(agent_tools, "run_evidence_job", quality_limited_evidence)
    result = _open_loop(
        manifest,
        project_root=project_root,
        cache_root=cache_root,
        maximum_bytes=source_path.stat().st_size * 8,
    ).run()

    assert result.state is AgentLoopState.ABSTAIN
    assert result.outcome is not None
    assert reason in result.outcome.reason_codes
    assert calls == [("later",), ("later", "next2")]
    records = _trace_records(result.trace_path)
    assert "recovery_retry_completed" in {record["event"] for record in records}
    final_budget = records[-1]["budget"]
    final_decision = records[-1]["decision"]
    assert isinstance(final_budget, dict)
    assert isinstance(final_decision, dict)
    assert final_budget["used_retries"] == 1
    assert final_decision["rule"] == "recovery_retry_exhausted_abstain"


def test_geospatial_failure_is_closed_reason_code_not_low_confidence_evidence() -> None:
    first_grid = GeospatialGrid(
        np.zeros((2, 2), dtype=np.float64), np.zeros((2, 2), dtype=np.float64)
    )
    distant_grid = GeospatialGrid(
        np.full((2, 2), 20.0, dtype=np.float64), np.zeros((2, 2), dtype=np.float64)
    )
    values = np.full((2, 2), 320.0, dtype=np.float32)
    mask = np.full((2, 2), 255, dtype=np.uint8)
    valid = np.zeros((2, 2), dtype=bool)
    first = TemporalObservation("first", mask, values, valid, first_grid)
    distant = TemporalObservation("distant", mask, values, valid, distant_grid)

    result = measure_temporal_persistence(
        (first, distant),
        PersistenceParameters(maximum_resample_distance_kilometres=1.0),
    )

    assert result.reason_codes == (ReasonCode.ALIGNMENT_FAILED,)
    assert result.persistence_count == 1
    assert result.confidence == 0.0
