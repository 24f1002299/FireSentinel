"""Contract tests for the bounded, manifest-only observation tools."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import firesentinel.agent.tools as agent_tools
from firesentinel.agent.tools import (
    AllowedObservation,
    BoundedObservationTools,
    ToolErrorCode,
    ToolManifest,
    ToolSource,
    load_tool_manifest,
)
from firesentinel.core.records import ActionType, Channel, Coordinates, ManifestCase
from firesentinel.vision.engine import (
    EvidenceJob,
    EvidenceJobObservation,
    EvidenceJobResult,
    EvidenceJobSource,
)
from firesentinel.vision.tiles import TilePreparationParameters
from tests.test_goes_crop import _latitude_longitude, _parameters_at, _source


def _manifest(tmp_path: Path) -> tuple[ToolManifest, Path, Path, Path]:
    project_root = tmp_path / "project"
    cache_root = project_root / "data" / "source-cache"
    cache_root.mkdir(parents=True)
    source_path = cache_root / "cached-source.nc"
    _source(source_path)
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    source_size = source_path.stat().st_size

    def source(source_id: str) -> ToolSource:
        return ToolSource(
            source_id=source_id,
            catalog_key=f"s3://noaa-goes18/{source_id}.nc#{digest}",
            source_path=source_path,
            size_bytes=source_size,
            sha256=digest,
        )

    latitude, longitude = _latitude_longitude(2, 2)
    now = datetime(2025, 1, 1, tzinfo=UTC)
    case = ManifestCase(
        case_id="tool-case",
        title="Bounded local replay",
        location=Coordinates(latitude, longitude),
        created_at=now,
        content_hash="a" * 64,
        allowed_observation_ids=("baseline", "alternate", "later", "next2"),
    )
    template_source = EvidenceJobSource("template", source_path)
    template = EvidenceJob(
        case_id="template-case",
        crop_parameters=_parameters_at(2, 2),
        tile_parameters=TilePreparationParameters(200.0, 240.0, 0.0, 1.0),
        observations=(
            EvidenceJobObservation("initial", template_source, template_source),
            EvidenceJobObservation("later", template_source, template_source),
        ),
    )
    common_c14 = source("c14-common")
    observations = (
        AllowedObservation(
            "baseline",
            ActionType.PRE_EVENT_BASELINE,
            now - timedelta(hours=1),
            Channel.C07,
            source("c07-baseline"),
            common_c14,
        ),
        AllowedObservation(
            "alternate",
            ActionType.ALTERNATE_BAND,
            now + timedelta(minutes=10),
            Channel.C14,
            source("c07-alternate-context"),
            common_c14,
        ),
        AllowedObservation(
            "later",
            ActionType.NEXT_TIMESTAMP,
            now + timedelta(minutes=20),
            Channel.C07,
            source("c07-later"),
            common_c14,
        ),
        AllowedObservation(
            "next2",
            ActionType.NEXT_TIMESTAMP,
            now + timedelta(minutes=30),
            Channel.C07,
            source("c07-next2"),
            common_c14,
        ),
    )
    return (
        ToolManifest(case, template, observations),
        project_root,
        cache_root,
        source_path,
    )


def _tools(
    tmp_path: Path, *, maximum_bytes: int | None = None
) -> tuple[BoundedObservationTools, Path, Path]:
    manifest, project_root, cache_root, source_path = _manifest(tmp_path)
    return (
        BoundedObservationTools(
            manifest,
            source_cache_root=cache_root,
            artifacts_root=project_root / "artifacts",
            project_root=project_root,
            maximum_bytes=(
                source_path.stat().st_size * 8
                if maximum_bytes is None
                else maximum_bytes
            ),
            maximum_elapsed_seconds=120.0,
        ),
        project_root,
        source_path,
    )


def test_selected_observations_rerun_cumulative_opencv_evidence_and_are_idempotent(
    tmp_path: Path,
) -> None:
    tools, project_root, _ = _tools(tmp_path)

    baseline = tools.pre_event_baseline("baseline")
    later = tools.next_timestamp("later")
    repeated = tools.next_timestamp("later")

    assert baseline.accepted and later.accepted and repeated.accepted
    assert baseline.budget.used_observations == 1
    assert later.budget.used_observations == 2
    assert later.budget.used_bytes > baseline.budget.used_bytes
    assert later.evidence_ids[-1] != baseline.evidence_ids[-1]
    assert repeated.idempotent
    assert repeated.evidence_ids == later.evidence_ids
    assert repeated.budget.used_observations == 2
    latest_packet = (
        project_root
        / "artifacts"
        / "tool-case"
        / later.evidence_ids[-1]
        / "evidence.json"
    )
    evidence = json.loads(latest_packet.read_text(encoding="utf-8"))
    assert [item["observation_id"] for item in evidence["observations"]] == [
        "baseline",
        "later",
    ]
    assert evidence["persistence"]["missing_observation_count"] == 0


def test_allowed_forbidden_and_terminal_transitions_are_structured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tools, _, _ = _tools(tmp_path)
    calls: list[tuple[str, ...]] = []

    def fake_evidence_job(
        job: EvidenceJob, artifacts_root: Path, **_: object
    ) -> EvidenceJobResult:
        del artifacts_root
        calls.append(tuple(item.observation_id for item in job.observations))
        return EvidenceJobResult(
            job.case_id,
            f"{len(calls):064x}",
            tmp_path / "fake-artifact",
            False,
            (),
        )

    monkeypatch.setattr(agent_tools, "run_evidence_job", fake_evidence_job)
    wrong_tool = tools.next_timestamp("alternate")
    forbidden = tools.next_timestamp("outside-allowlist")
    first = tools.pre_event_baseline("baseline")
    terminal = tools.finalize()
    terminal_repeat = tools.finalize()
    after_terminal = tools.next_timestamp("later")

    assert not wrong_tool.accepted
    assert wrong_tool.error is not None
    assert wrong_tool.error.code is ToolErrorCode.ACTION_NOT_ALLOWED
    assert not forbidden.accepted
    assert forbidden.error is not None
    assert forbidden.error.code is ToolErrorCode.OBSERVATION_NOT_ALLOWED
    assert first.accepted and calls == [("baseline",)]
    assert terminal.accepted and terminal.terminal_action is ActionType.FINALIZE
    assert terminal_repeat.accepted and terminal_repeat.idempotent
    assert not after_terminal.accepted
    assert after_terminal.error is not None
    assert after_terminal.error.code is ToolErrorCode.TERMINAL


def test_observation_byte_and_elapsed_limits_reject_without_running_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tools, _, _ = _tools(tmp_path, maximum_bytes=1)
    calls: list[EvidenceJob] = []

    def fake_evidence_job(
        job: EvidenceJob, artifacts_root: Path, **_: object
    ) -> EvidenceJobResult:
        del artifacts_root
        calls.append(job)
        return EvidenceJobResult(job.case_id, "1" * 64, tmp_path, False, ())

    monkeypatch.setattr(agent_tools, "run_evidence_job", fake_evidence_job)
    byte_limited = tools.pre_event_baseline("baseline")
    assert not byte_limited.accepted
    assert byte_limited.error is not None
    assert byte_limited.error.code is ToolErrorCode.BYTE_BUDGET_EXHAUSTED
    assert not calls

    manifest, project_root, cache_root, _ = _manifest(tmp_path / "elapsed")
    now = [0.0]
    elapsed_tools = BoundedObservationTools(
        manifest,
        source_cache_root=cache_root,
        artifacts_root=project_root / "artifacts",
        project_root=project_root,
        maximum_bytes=10**9,
        maximum_elapsed_seconds=1.0,
        clock=lambda: now[0],
    )
    now[0] = 1.0
    timed_out = elapsed_tools.pre_event_baseline("baseline")
    assert not timed_out.accepted
    assert timed_out.error is not None
    assert timed_out.error.code is ToolErrorCode.ELAPSED_TIME_EXHAUSTED


def test_maximum_three_observations_and_all_terminal_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tools, _, _ = _tools(tmp_path)

    def fake_evidence_job(
        job: EvidenceJob, artifacts_root: Path, **_: object
    ) -> EvidenceJobResult:
        del artifacts_root
        return EvidenceJobResult(
            job.case_id,
            f"{len(job.observations):064x}",
            tmp_path,
            False,
            (),
        )

    monkeypatch.setattr(agent_tools, "run_evidence_job", fake_evidence_job)
    assert tools.pre_event_baseline("baseline").accepted
    assert tools.alternate_band("alternate").accepted
    assert tools.next_timestamp("later").accepted
    exhausted = tools.next_timestamp("next2")
    assert not exhausted.accepted
    assert exhausted.error is not None
    assert exhausted.error.code is ToolErrorCode.OBSERVATION_BUDGET_EXHAUSTED

    abstention_tools, _, _ = _tools(tmp_path / "abstain")
    review_tools, _, _ = _tools(tmp_path / "review")
    assert abstention_tools.abstain().terminal_action is ActionType.ABSTAIN
    assert (
        review_tools.request_human_review().terminal_action
        is ActionType.REQUEST_HUMAN_REVIEW
    )


def test_manifest_and_cache_scope_block_labels_and_arbitrary_files(
    tmp_path: Path,
) -> None:
    manifest, project_root, cache_root, source_path = _manifest(tmp_path)
    labels_path = project_root / "evaluation-data" / "secret-labels.json"
    labels_path.parent.mkdir(parents=True)
    labels_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="evaluation-only labels"):
        load_tool_manifest(labels_path, project_root=project_root)

    outside_path = project_root / "outside.nc"
    _source(outside_path)
    outside_hash = hashlib.sha256(outside_path.read_bytes()).hexdigest()
    altered = AllowedObservation(
        "baseline",
        ActionType.PRE_EVENT_BASELINE,
        manifest.observations_by_id["baseline"].observation_time,
        Channel.C07,
        ToolSource(
            "outside-source",
            "catalog-outside",
            outside_path,
            outside_path.stat().st_size,
            outside_hash,
        ),
        manifest.observations_by_id["baseline"].channel14,
    )
    unsafe_manifest = ToolManifest(
        manifest.case,
        manifest.evidence_template,
        (altered, *manifest.observations[1:]),
    )
    with pytest.raises(ValueError, match="source_cache_root"):
        BoundedObservationTools(
            unsafe_manifest,
            source_cache_root=cache_root,
            artifacts_root=project_root / "artifacts",
            project_root=project_root,
            maximum_bytes=source_path.stat().st_size * 8,
            maximum_elapsed_seconds=60.0,
        )
