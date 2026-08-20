"""Golden end-to-end contracts for the deterministic local evidence engine."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import cast

import pytest

from firesentinel.core.records import ReasonCode
from firesentinel.data.goes_crop import (
    CalibratedCrop,
    CropParameters,
    extract_calibrated_crop,
)
from firesentinel.performance import profile_local_replay
from firesentinel.vision.engine import (
    EvidenceJob,
    EvidenceJobCancelled,
    EvidenceJobFailure,
    EvidenceJobObservation,
    EvidenceJobSource,
    EvidenceJobTimeout,
    load_evidence_job,
    main,
    run_evidence_job,
)
from firesentinel.vision.tiles import TilePreparationParameters
from tests.test_goes_crop import _parameters_at, _source


def _job(tmp_path: Path, *, missing_source: bool = False) -> EvidenceJob:
    source_path = tmp_path / "source.nc"
    _source(source_path)
    selected_path = tmp_path / "missing.nc" if missing_source else source_path
    source = EvidenceJobSource(
        "ABI-L2-CMIPF/2025/001/00/OR_ABI-L2-CMIPF-M6C07_G18_s20250010000000.nc",
        selected_path,
    )
    channel14 = EvidenceJobSource(
        "ABI-L2-CMIPF/2025/001/00/OR_ABI-L2-CMIPF-M6C14_G18_s20250000000.nc",
        selected_path,
    )
    return EvidenceJob(
        case_id="engine-case",
        crop_parameters=_parameters_at(2, 2),
        tile_parameters=TilePreparationParameters(200.0, 240.0, 0.0, 1.0),
        observations=(
            EvidenceJobObservation("initial", source, channel14),
            EvidenceJobObservation("later", source, channel14),
        ),
    )


def test_identical_local_evidence_jobs_reuse_one_complete_content_addressed_packet(
    tmp_path: Path,
) -> None:
    job = _job(tmp_path)
    artifacts_root = tmp_path / "artifacts"

    first = run_evidence_job(job, artifacts_root)
    second = run_evidence_job(job, artifacts_root)

    assert not first.reused_existing_artifact
    assert second.reused_existing_artifact
    assert first.content_hash == second.content_hash
    assert first.artifact_directory == second.artifact_directory
    evidence_path = first.artifact_directory / "evidence.json"
    completion_path = first.artifact_directory / "completion.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    assert evidence["content_hash"] == first.content_hash
    assert completion["content_hash"] == first.content_hash
    assert set(evidence) >= {
        "configuration",
        "observations",
        "persistence",
        "warnings",
        "artifacts",
        "timings_milliseconds",
    }
    assert len(evidence["artifacts"]) == 14
    for item in evidence["artifacts"]:
        assert (first.artifact_directory / item["filename"]).is_file()


def test_repeated_source_crop_is_cached_without_changing_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "cached").mkdir()
    cached_job = _job(tmp_path / "cached")
    original_crop = extract_calibrated_crop
    crop_calls = 0

    def counted_crop(source_path: Path, parameters: CropParameters) -> CalibratedCrop:
        nonlocal crop_calls
        crop_calls += 1
        return original_crop(source_path, parameters)

    monkeypatch.setattr(
        "firesentinel.vision.engine.extract_calibrated_crop", counted_crop
    )
    cached = run_evidence_job(cached_job, tmp_path / "cached-artifacts")
    assert crop_calls == 1

    source = cached_job.observations[0].channel7.source_path
    copies: list[Path] = []
    for index in range(4):
        copy = tmp_path / "uncached" / f"source-{index}.nc"
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, copy)
        copies.append(copy)
    initial, later = cached_job.observations
    uncached_job = EvidenceJob(
        cached_job.case_id,
        cached_job.crop_parameters,
        cached_job.tile_parameters,
        (
            EvidenceJobObservation(
                initial.observation_id,
                EvidenceJobSource(initial.channel7.catalog_key, copies[0]),
                EvidenceJobSource(initial.channel14.catalog_key, copies[1]),
            ),
            EvidenceJobObservation(
                later.observation_id,
                EvidenceJobSource(later.channel7.catalog_key, copies[2]),
                EvidenceJobSource(later.channel14.catalog_key, copies[3]),
            ),
        ),
    )
    uncached = run_evidence_job(uncached_job, tmp_path / "uncached-artifacts")

    assert cached.content_hash == uncached.content_hash
    assert (
        json.loads(
            (cached.artifact_directory / "evidence.json").read_text(encoding="utf-8")
        )["content_hash"]
        == uncached.content_hash
    )


def test_local_profile_covers_cache_crop_opencv_artifact_and_reviewer_stages(
    tmp_path: Path,
) -> None:
    (tmp_path / "profile").mkdir()
    profile = profile_local_replay(_job(tmp_path / "profile"))

    assert profile["record_type"] == "firesentinel_local_performance_profile"
    assert cast(float, profile["source_cache_access_milliseconds"]) >= 0.0
    assert cast(float, profile["crop_loading_milliseconds"]) > 0.0
    assert cast(float, profile["artifact_and_metadata_milliseconds"]) >= 0.0
    assert cast(float, profile["ui_reviewer_model_loading_milliseconds"]) >= 0.0
    stages = cast(dict[str, float], profile["opencv_stages_milliseconds"])
    assert set(stages) == {"prepare", "anomaly"}


def test_cli_loads_a_portable_job_manifest_and_reports_the_artifact(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    job = _job(tmp_path)
    manifest_path = tmp_path / "job.json"
    manifest_path.write_text(
        json.dumps(job.to_dict(include_paths=True), sort_keys=True), encoding="utf-8"
    )

    loaded = load_evidence_job(manifest_path)
    exit_code = main(
        [
            "--job",
            str(manifest_path),
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--timeout-seconds",
            "60",
        ]
    )

    assert loaded.case_id == job.case_id
    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "completed"
    assert Path(output["artifact_directory"]).is_dir()


def test_timeout_and_cancellation_never_publish_a_completion_marker(
    tmp_path: Path,
) -> None:
    job = _job(tmp_path)
    artifacts_root = tmp_path / "artifacts"

    clock_values = iter((0.0, 1.0))

    def expired_clock() -> float:
        return next(clock_values)

    with pytest.raises(EvidenceJobTimeout) as timeout:
        run_evidence_job(job, artifacts_root, timeout_seconds=0.1, clock=expired_clock)
    assert timeout.value.reason_code is ReasonCode.TIMEOUT

    checks = 0

    def cancel_during_write() -> bool:
        nonlocal checks
        checks += 1
        return checks >= 29

    with pytest.raises(EvidenceJobCancelled) as cancelled:
        run_evidence_job(
            job,
            artifacts_root,
            cancellation_requested=cancel_during_write,
        )
    assert cancelled.value.reason_code is ReasonCode.CANCELLED
    case_directory = artifacts_root / job.case_id
    assert not list(case_directory.rglob("completion.json"))
    assert not list(case_directory.glob(".evidence-staging-*"))


def test_missing_selected_source_is_classified_without_publishing_an_artifact(
    tmp_path: Path,
) -> None:
    job = _job(tmp_path, missing_source=True)
    artifacts_root = tmp_path / "artifacts"

    with pytest.raises(EvidenceJobFailure) as failure:
        run_evidence_job(job, artifacts_root)

    assert failure.value.reason_code is ReasonCode.SOURCE_MISSING
    assert not (artifacts_root / job.case_id).exists()


def test_invalid_engine_configuration_fails_before_running_sources(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.nc"
    _source(source_path)
    source = EvidenceJobSource("catalog-key", source_path)
    with pytest.raises(ValueError, match="at least two observations"):
        EvidenceJob(
            case_id="invalid-case",
            crop_parameters=_parameters_at(2, 2),
            tile_parameters=TilePreparationParameters(200.0, 240.0),
            observations=(EvidenceJobObservation("only", source, source),),
        )
