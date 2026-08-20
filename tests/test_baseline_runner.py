"""Golden contracts for fair one-shot and fixed-bundle development baselines."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pytest

from firesentinel.evaluation.runner import (
    FIXED_BUNDLE_ROLES,
    ONE_SHOT_ROLES,
    BaselineMode,
    BaselineParameters,
    BaselineSource,
    load_development_manifest,
    run_development_baselines,
    write_baseline_report,
)
from firesentinel.vision.engine import (
    EvidenceJob,
    EvidenceJobObservation,
    EvidenceJobSource,
)
from firesentinel.vision.tiles import TilePreparationParameters
from tests.test_goes_crop import _latitude_longitude, _parameters_at, _source


def _timestamp(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _development_manifest(
    path: Path, *, case_ids: tuple[str, ...] = ("case-a",)
) -> Path:
    latitude, longitude = _latitude_longitude(2, 2)
    moment = datetime(2025, 1, 1, tzinfo=UTC)
    cases = []
    for case_id in case_ids:
        observations = []
        for role, channel, offset, size in (
            ("baseline", "C07", timedelta(hours=-1), 101),
            ("initial", "C07", timedelta(), 102),
            ("later", "C07", timedelta(minutes=20), 103),
            ("alternate", "C14", timedelta(minutes=10), 104),
        ):
            source_id = f"{case_id}-{role}"
            observations.append(
                {
                    "role": role,
                    "channel": channel,
                    "observation_time_utc": _timestamp(moment + offset),
                    "source": {
                        "source_id": source_id,
                        "bucket": "noaa-goes18",
                        "object_key": f"ABI-L2-CMIPF/{source_id}.nc",
                        "size_bytes": size,
                        "sha256": hashlib.sha256(source_id.encode()).hexdigest(),
                    },
                }
            )
        cases.append(
            {
                "case_id": case_id,
                "label": "positive",
                "anchor": {"latitude": latitude, "longitude": longitude},
                "observations": observations,
            }
        )
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "firesentinel_frozen_split_manifest",
                "frozen": True,
                "split": "development",
                "labels_visible_to_tuning": True,
                "cases": cases,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _parameters(tmp_path: Path) -> BaselineParameters:
    source_path = tmp_path / "template-source.nc"
    _source(source_path)
    source = EvidenceJobSource("template-c07", source_path)
    template = EvidenceJob(
        case_id="template-case",
        crop_parameters=_parameters_at(2, 2),
        tile_parameters=TilePreparationParameters(200.0, 240.0, 0.0, 1.0),
        observations=(
            EvidenceJobObservation("initial", source, source),
            EvidenceJobObservation("later", source, source),
        ),
    )
    return BaselineParameters(
        template,
        crop_half_height_degrees=0.08,
        crop_half_width_degrees=0.08,
    )


def test_modes_complete_each_development_case_with_comparable_reports(
    tmp_path: Path,
) -> None:
    manifest_path = _development_manifest(tmp_path / "development.manifest.json")
    source_path = tmp_path / "cached-source.nc"
    _source(source_path)
    calls: list[tuple[str, str]] = []

    def resolve(case_id: str, source: BaselineSource) -> Path:
        calls.append((case_id, source.role))
        return source_path

    result = run_development_baselines(
        manifest_path,
        tmp_path / "artifacts",
        _parameters(tmp_path),
        source_resolver=resolve,
    )
    report = result.to_dict()

    modes = cast(dict[str, dict[str, object]], report["modes"])
    assert set(modes) == {"one_shot", "fixed_bundle"}
    one_shot = modes["one_shot"]
    fixed_bundle = modes["fixed_bundle"]
    assert one_shot["selection_roles"] == list(ONE_SHOT_ROLES)
    assert fixed_bundle["selection_roles"] == list(FIXED_BUNDLE_ROLES)
    assert calls == [
        ("case-a", "initial"),
        ("case-a", "alternate"),
        ("case-a", "baseline"),
        ("case-a", "initial"),
        ("case-a", "later"),
        ("case-a", "alternate"),
    ]

    one_case = cast(list[dict[str, object]], one_shot["cases"])[0]
    fixed_case = cast(list[dict[str, object]], fixed_bundle["cases"])[0]
    for case in (one_case, fixed_case):
        assert set(case) == {
            "case_id",
            "mode",
            "outcome",
            "observations",
            "observation_count",
            "channel7_observation_count",
            "evidence_time_step_count",
            "evidence",
            "resources",
            "errors",
        }
        resources = cast(dict[str, object], case["resources"])
        outcome = cast(dict[str, object], case["outcome"])
        evidence = cast(list[dict[str, object]], case["evidence"])
        assert set(outcome) == {"state", "reason_codes", "confidence", "explanation"}
        assert isinstance(outcome["explanation"], str)
        assert "confirmed wildfire" not in outcome["explanation"].lower()
        assert set(resources) == {
            "selected_source_bytes",
            "downloaded_bytes",
            "latency_milliseconds",
        }
        assert case["errors"] == []
        assert len(evidence) == 1
        assert Path(cast(str, evidence[0]["artifact_directory"])).is_dir()
    assert one_case["observation_count"] == 2
    assert one_case["channel7_observation_count"] == 1
    assert one_case["evidence_time_step_count"] == 1
    assert (
        cast(dict[str, object], one_case["resources"])["selected_source_bytes"] == 206
    )
    assert fixed_case["observation_count"] == 4
    assert fixed_case["channel7_observation_count"] == 3
    assert fixed_case["evidence_time_step_count"] == 3
    assert (
        cast(dict[str, object], fixed_case["resources"])["selected_source_bytes"] == 410
    )
    assert (
        cast(dict[str, object], one_shot["summary"])["case_count"]
        == cast(dict[str, object], fixed_bundle["summary"])["case_count"]
        == 1
    )

    report_path = write_baseline_report(result, tmp_path / "baseline-report.json")
    assert json.loads(report_path.read_text(encoding="utf-8")) == report


def test_replay_reuses_the_same_mode_specific_evidence_artifact_ids(
    tmp_path: Path,
) -> None:
    manifest_path = _development_manifest(tmp_path / "development.manifest.json")
    source_path = tmp_path / "cached-source.nc"
    _source(source_path)

    def resolve(_: str, __: object) -> Path:
        return source_path

    first = run_development_baselines(
        manifest_path,
        tmp_path / "artifacts",
        _parameters(tmp_path),
        source_resolver=resolve,
    )
    second = run_development_baselines(
        manifest_path,
        tmp_path / "artifacts",
        _parameters(tmp_path),
        source_resolver=resolve,
    )

    for mode in BaselineMode:
        first_result = next(result for result in first.results if result.mode == mode)
        second_result = next(result for result in second.results if result.mode == mode)
        assert (
            first_result.evidence[0].content_hash
            == second_result.evidence[0].content_hash
        )
        assert (
            first_result.evidence[0].artifact_directory
            == second_result.evidence[0].artifact_directory
        )
        assert second_result.evidence[0].reused_existing_artifact


def test_source_failure_is_reported_per_mode_without_stopping_other_cases(
    tmp_path: Path,
) -> None:
    manifest_path = _development_manifest(
        tmp_path / "development.manifest.json", case_ids=("case-a", "case-b")
    )
    source_path = tmp_path / "cached-source.nc"
    _source(source_path)

    def resolve(case_id: str, _: object) -> Path:
        if case_id == "case-a":
            raise FileNotFoundError("selected object is absent")
        return source_path

    result = run_development_baselines(
        manifest_path,
        tmp_path / "artifacts",
        _parameters(tmp_path),
        source_resolver=resolve,
    )

    failed = [item for item in result.results if item.case_id == "case-a"]
    completed = [item for item in result.results if item.case_id == "case-b"]
    assert all(item.outcome_state.value == "failed" for item in failed)
    assert all(item.errors[0].reason_code.value == "source_missing" for item in failed)
    assert all(not item.errors and item.evidence for item in completed)


def test_runner_rejects_nondevelopment_manifests(tmp_path: Path) -> None:
    manifest_path = _development_manifest(tmp_path / "test.manifest.json")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["split"] = "test"
    payload["labels_visible_to_tuning"] = False
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="frozen development manifest"):
        load_development_manifest(manifest_path)
