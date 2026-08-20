"""Tests for deterministic FIRMS-positive and matched-control benchmark builds."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from firesentinel.evaluation.benchmark import (
    DEFAULT_CASES_PER_CLASS,
    build_benchmark,
    verify_benchmark,
    write_benchmark,
)


def _timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _goes_object_key(channel: str, start: datetime) -> str:
    end = start + timedelta(minutes=9, seconds=50)
    start_token = f"{start:%Y%j%H%M%S}0"
    end_token = f"{end:%Y%j%H%M%S}0"
    return (
        f"ABI-L2-CMIPF/{start:%Y}/{start:%j}/{start:%H}/"
        f"OR_ABI-L2-CMIPF-M6{channel}_G18_s{start_token}_e{end_token}_c{end_token}.nc"
    )


def _benchmark_inputs(tmp_path: Path) -> tuple[Path, Path]:
    sources: dict[tuple[datetime, str], dict[str, object]] = {}

    def source_id(moment: datetime, channel: str) -> str:
        key = (moment, channel)
        if key not in sources:
            identifier = f"source-{channel.lower()}-{moment:%Y%j%H%M}"
            sources[key] = {
                "source_id": identifier,
                "bucket": "noaa-goes18",
                "object_key": _goes_object_key(channel, moment),
                "size_bytes": 1000,
                "sha256": hashlib.sha256(identifier.encode()).hexdigest(),
            }
        return str(sources[key]["source_id"])

    def window(
        identifier: str,
        moment: datetime,
        latitude: float,
        *,
        view_zenith_degrees: float,
        usable_fraction: float,
    ) -> dict[str, object]:
        role_times = {
            "baseline": moment - timedelta(hours=1),
            "initial": moment,
            "later": moment + timedelta(minutes=20),
            "alternate": moment + timedelta(minutes=10),
        }
        role_channels = {
            "baseline": "C07",
            "initial": "C07",
            "later": "C07",
            "alternate": "C14",
        }
        return {
            "window_id": identifier,
            "anchor": {
                "acquisition_time_utc": _timestamp(moment),
                "latitude": latitude,
                "longitude": -120.0,
            },
            "view_zenith_degrees": view_zenith_degrees,
            "usable_fraction": usable_fraction,
            "observations": [
                {
                    "role": role,
                    "channel": role_channels[role],
                    "observation_time_utc": _timestamp(role_times[role]),
                    "source_id": source_id(role_times[role], role_channels[role]),
                }
                for role in ("baseline", "initial", "later", "alternate")
            ],
        }

    events: list[dict[str, object]] = []
    windows: list[dict[str, object]] = []
    start = datetime(2024, 7, 1, 20, 0, tzinfo=UTC)
    for index in range(DEFAULT_CASES_PER_CLASS):
        moment = start + timedelta(days=index)
        event_id = f"firms-{index:03d}"
        events.append(
            {
                "event_id": event_id,
                "start_time_utc": _timestamp(moment),
                "end_time_utc": _timestamp(moment),
                "centroid_latitude": 30.0,
                "centroid_longitude": -120.0,
                "detections": [
                    {
                        "acquisition_time_utc": _timestamp(moment),
                        "latitude": 30.0,
                        "longitude": -120.0,
                    }
                ],
            }
        )
        windows.append(
            window(
                f"positive-{index:03d}",
                moment,
                30.1,
                view_zenith_degrees=30.0 + index % 3,
                usable_fraction=0.92,
            )
        )
        windows.append(
            window(
                f"control-{index:03d}",
                moment,
                31.5,
                view_zenith_degrees=30.5 + index % 3,
                usable_fraction=0.93,
            )
        )
    labels_path = tmp_path / "firms-event-labels.json"
    labels_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "firms_event_reference_labels",
                "evaluation_only": True,
                "source_hashes": [],
                "clustering": {},
                "events": events,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    inventory_path = tmp_path / "observation-inventory.json"
    inventory_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "goes18_observation_window_inventory",
                "sources": sorted(
                    sources.values(), key=lambda source: str(source["source_id"])
                ),
                "windows": windows,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return labels_path, inventory_path


def test_builds_60_firms_positives_and_matched_firms_excluded_controls(
    tmp_path: Path,
) -> None:
    labels_path, inventory_path = _benchmark_inputs(tmp_path)

    build = build_benchmark(labels_path, inventory_path, random_seed=1234)
    benchmark_path, audit_path = write_benchmark(build, tmp_path / "first")
    verify_benchmark(benchmark_path, audit_path)

    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    positives = [case for case in benchmark["cases"] if case["label"] == "positive"]
    controls = [case for case in benchmark["cases"] if case["label"] == "control"]
    assert len(positives) == DEFAULT_CASES_PER_CLASS
    assert len(controls) == DEFAULT_CASES_PER_CLASS
    assert audit["counts"] == {
        "control_candidates_after_firms_exclusion": DEFAULT_CASES_PER_CLASS,
        "controls": DEFAULT_CASES_PER_CLASS,
        "positive_candidates": DEFAULT_CASES_PER_CLASS,
        "positives": DEFAULT_CASES_PER_CLASS,
        "requested_cases_per_class": DEFAULT_CASES_PER_CLASS,
    }
    assert audit["random_seed"] == 1234
    assert benchmark["inputs"] == {
        "firms_labels_sha256": hashlib.sha256(labels_path.read_bytes()).hexdigest(),
        "observation_inventory_sha256": hashlib.sha256(
            inventory_path.read_bytes()
        ).hexdigest(),
    }
    assert audit["source_reference_hashes"]
    assert all(
        len(source_hash) == 64 for source_hash in audit["source_reference_hashes"]
    )
    positive_ids = {case["case_id"] for case in positives}
    assert {case["matched_positive_case_id"] for case in controls} == positive_ids
    assert all(
        case["matching_deltas"]["local_time_hour_difference"] <= 1
        and case["matching_deltas"]["view_zenith_degrees_difference"] <= 10
        and case["matching_deltas"]["usable_fraction_difference"] <= 0.1
        for case in controls
    )
    assert all(
        {observation["role"] for observation in case["observations"]}
        == {"baseline", "initial", "later", "alternate"}
        for case in benchmark["cases"]
    )

    second_build = build_benchmark(labels_path, inventory_path, random_seed=1234)
    second_benchmark_path, second_audit_path = write_benchmark(
        second_build, tmp_path / "second"
    )
    assert benchmark_path.read_bytes() == second_benchmark_path.read_bytes()
    assert audit_path.read_bytes() == second_audit_path.read_bytes()


def test_rejects_unresolved_observation_sources(tmp_path: Path) -> None:
    labels_path, inventory_path = _benchmark_inputs(tmp_path)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory["windows"][0]["observations"][0]["source_id"] = "missing-source"
    inventory_path.write_text(json.dumps(inventory, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="does not resolve"):
        build_benchmark(labels_path, inventory_path)


def test_refuses_to_claim_a_benchmark_below_60_pairs(tmp_path: Path) -> None:
    labels_path, inventory_path = _benchmark_inputs(tmp_path)
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    labels["events"] = labels["events"][:59]
    labels_path.write_text(json.dumps(labels, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="insufficient matched benchmark cases"):
        build_benchmark(labels_path, inventory_path)
