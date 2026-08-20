"""End-to-end contracts for the checkpointed bounded local agent loop."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

import firesentinel.agent.tools as agent_tools
from firesentinel.agent.loop import (
    AgentLoopState,
    BoundedAgentLoop,
    load_last_complete_transition,
)
from firesentinel.agent.tools import ToolManifest
from firesentinel.vision.engine import EvidenceJob, EvidenceJobResult
from tests.test_agent_tools import _manifest


def _open_loop(
    manifest: ToolManifest,
    *,
    project_root: Path,
    cache_root: Path,
    trace_path: Path,
    maximum_bytes: int,
    maximum_observations: int = 3,
) -> BoundedAgentLoop:
    return BoundedAgentLoop.open(
        manifest,
        source_cache_root=cache_root,
        artifacts_root=project_root / "artifacts",
        project_root=project_root,
        trace_path=trace_path,
        maximum_bytes=maximum_bytes,
        maximum_elapsed_seconds=60.0,
        maximum_observations=maximum_observations,
    )


def _evidence_runner(
    calls: list[tuple[str, ...]],
) -> Callable[..., EvidenceJobResult]:
    def run(job: EvidenceJob, artifacts_root: Path, **_: object) -> EvidenceJobResult:
        observation_ids = tuple(item.observation_id for item in job.observations)
        calls.append(observation_ids)
        evidence_id = f"{len(observation_ids):064x}"
        destination = artifacts_root / job.case_id / evidence_id
        destination.mkdir(parents=True, exist_ok=True)
        if len(observation_ids) == 1:
            reasons = ["thermal_anomaly_weak"]
            candidate_count = 3
        else:
            reasons = ["no_persistent_evidence"]
            candidate_count = 0
        payload = {
            "observations": [
                {
                    "anomaly": {
                        "candidate_pixel_count": candidate_count,
                        "components": [{"label": index + 1}] if candidate_count else [],
                        "reason_codes": reasons,
                    }
                }
                for index, _ in enumerate(observation_ids)
            ],
            "persistence": {
                "persistence_count": 0,
                "confidence": 0.0,
            },
        }
        (destination / "evidence.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        return EvidenceJobResult(job.case_id, evidence_id, destination, False, ())

    return run


def _events(trace_path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def test_loop_reaches_terminal_finalize_and_persists_every_state_transition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, project_root, cache_root, source_path = _manifest(tmp_path)
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(agent_tools, "run_evidence_job", _evidence_runner(calls))
    trace_path = project_root / "artifacts" / "tool-case" / "agent-loop.jsonl"

    result = _open_loop(
        manifest,
        project_root=project_root,
        cache_root=cache_root,
        trace_path=trace_path,
        maximum_bytes=source_path.stat().st_size * 8,
    ).run()

    assert result.is_terminal
    assert result.state is AgentLoopState.FINALIZE
    assert result.outcome is not None
    assert result.outcome.state.value == "no_persistent_evidence"
    assert calls == [("later",), ("later", "next2")]
    events = _events(trace_path)
    assert [event["sequence"] for event in events] == list(range(1, len(events) + 1))
    assert {event["to_state"] for event in events} >= {
        "observe",
        "analyze",
        "decide",
        "act",
        "finalize",
    }
    for event in events:
        budget = event["budget"]
        assert isinstance(budget, dict)
        assert budget["used_observations"] <= budget["max_observations"]
        assert budget["used_bytes"] <= budget["max_bytes"]
        assert budget["used_elapsed_seconds"] <= budget["max_elapsed_seconds"]
        assert budget["used_retries"] <= budget["max_retries"]


def test_interrupted_loop_resumes_from_last_complete_record_without_repeating_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, project_root, cache_root, source_path = _manifest(tmp_path)
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(agent_tools, "run_evidence_job", _evidence_runner(calls))
    trace_path = project_root / "artifacts" / "tool-case" / "agent-loop.jsonl"
    maximum_bytes = source_path.stat().st_size * 8

    paused = _open_loop(
        manifest,
        project_root=project_root,
        cache_root=cache_root,
        trace_path=trace_path,
        maximum_bytes=maximum_bytes,
    ).run(transition_limit=4)
    assert paused.state is AgentLoopState.ANALYZE
    assert not paused.is_terminal
    assert calls == [("later",)]

    with trace_path.open("a", encoding="utf-8") as trace:
        trace.write('{"partial":')
    checkpoint = load_last_complete_transition(trace_path)
    assert checkpoint is not None
    assert checkpoint["to_state"] == "analyze"

    resumed = _open_loop(
        manifest,
        project_root=project_root,
        cache_root=cache_root,
        trace_path=trace_path,
        maximum_bytes=maximum_bytes,
    ).run()

    assert resumed.state is AgentLoopState.FINALIZE
    assert calls == [("later",), ("later", "next2")]
    assert (
        len(list((project_root / "artifacts" / "tool-case").glob("*/evidence.json")))
        == 2
    )
    final_checkpoint = load_last_complete_transition(trace_path)
    assert final_checkpoint is not None
    assert final_checkpoint["to_state"] == "finalize"


def test_resume_refuses_changed_resource_limits(tmp_path: Path) -> None:
    manifest, project_root, cache_root, source_path = _manifest(tmp_path)
    trace_path = project_root / "artifacts" / "tool-case" / "agent-loop.jsonl"
    maximum_bytes = source_path.stat().st_size * 8
    _open_loop(
        manifest,
        project_root=project_root,
        cache_root=cache_root,
        trace_path=trace_path,
        maximum_bytes=maximum_bytes,
    )

    with pytest.raises(ValueError, match="persisted budget"):
        _open_loop(
            manifest,
            project_root=project_root,
            cache_root=cache_root,
            trace_path=trace_path,
            maximum_bytes=maximum_bytes + 1,
        )


def test_unreadable_completed_evidence_ends_in_explicit_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, project_root, cache_root, source_path = _manifest(tmp_path)

    def missing_packet(
        job: EvidenceJob, artifacts_root: Path, **_: object
    ) -> EvidenceJobResult:
        evidence_id = "f" * 64
        destination = artifacts_root / job.case_id / evidence_id
        destination.mkdir(parents=True, exist_ok=True)
        return EvidenceJobResult(job.case_id, evidence_id, destination, False, ())

    monkeypatch.setattr(agent_tools, "run_evidence_job", missing_packet)
    trace_path = project_root / "artifacts" / "tool-case" / "agent-loop.jsonl"

    result = _open_loop(
        manifest,
        project_root=project_root,
        cache_root=cache_root,
        trace_path=trace_path,
        maximum_bytes=source_path.stat().st_size * 8,
    ).run()

    assert result.state is AgentLoopState.FAILURE
    assert result.outcome is not None
    assert result.outcome.state.value == "failed"


def test_loop_uses_review_and_abstain_as_explicit_safe_terminal_states(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, project_root, cache_root, source_path = _manifest(tmp_path)

    def persistent_after_two(
        job: EvidenceJob, artifacts_root: Path, **_: object
    ) -> EvidenceJobResult:
        evidence_id = f"{len(job.observations):064x}"
        destination = artifacts_root / job.case_id / evidence_id
        destination.mkdir(parents=True, exist_ok=True)
        persistent = len(job.observations) == 2
        payload = {
            "observations": [
                {
                    "anomaly": {
                        "candidate_pixel_count": 3,
                        "components": [{"label": index + 1}],
                        "reason_codes": [] if persistent else ["thermal_anomaly_weak"],
                    }
                }
                for index, _ in enumerate(job.observations)
            ],
            "persistence": {
                "persistence_count": 2 if persistent else 0,
                "confidence": 0.6 if persistent else 0.0,
            },
        }
        (destination / "evidence.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        return EvidenceJobResult(job.case_id, evidence_id, destination, False, ())

    monkeypatch.setattr(agent_tools, "run_evidence_job", persistent_after_two)
    review = _open_loop(
        manifest,
        project_root=project_root,
        cache_root=cache_root,
        trace_path=project_root / "artifacts" / "tool-case" / "review.jsonl",
        maximum_bytes=source_path.stat().st_size * 8,
    ).run()

    assert review.state is AgentLoopState.REVIEW
    assert review.outcome is not None
    assert review.outcome.state.value == "review_escalation"

    monkeypatch.setattr(agent_tools, "run_evidence_job", _evidence_runner([]))
    abstention = _open_loop(
        manifest,
        project_root=project_root,
        cache_root=cache_root,
        trace_path=project_root / "artifacts" / "tool-case" / "abstain.jsonl",
        maximum_bytes=source_path.stat().st_size * 8,
        maximum_observations=1,
    ).run()

    assert abstention.state is AgentLoopState.ABSTAIN
    assert abstention.outcome is not None
    assert abstention.outcome.state.value == "insufficient_evidence"
