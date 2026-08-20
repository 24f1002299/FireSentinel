"""Contracts for leakage-safe split auditing and frozen manifests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from firesentinel.evaluation.benchmark import build_benchmark, write_benchmark
from firesentinel.evaluation.freeze import (
    FROZEN_AUDIT_FILENAME,
    freeze_benchmark,
    verify_frozen_benchmark,
    write_frozen_benchmark,
)
from firesentinel.evaluation.tuning import tuning_manifest_path
from tests.test_benchmark import _benchmark_inputs


def _groupable_benchmark_inputs(tmp_path: Path) -> tuple[Path, Path]:
    labels_path, inventory_path = _benchmark_inputs(tmp_path)
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    events = labels["events"]
    windows = inventory["windows"]
    for index, event in enumerate(events):
        latitude = -59.9 + index * 2
        event["centroid_latitude"] = latitude
        event["detections"][0]["latitude"] = latitude
        windows[index * 2]["anchor"]["latitude"] = latitude
        windows[index * 2 + 1]["anchor"]["latitude"] = latitude + 1.4
    labels_path.write_text(json.dumps(labels, sort_keys=True), encoding="utf-8")
    inventory_path.write_text(json.dumps(inventory, sort_keys=True), encoding="utf-8")
    return labels_path, inventory_path


def _frozen_directory(tmp_path: Path) -> tuple[Path, Path]:
    labels_path, inventory_path = _groupable_benchmark_inputs(tmp_path)
    build = build_benchmark(labels_path, inventory_path, random_seed=1234)
    benchmark_path, benchmark_audit_path = write_benchmark(
        build, tmp_path / "benchmark"
    )
    frozen = freeze_benchmark(
        benchmark_path,
        benchmark_audit_path,
        reviewer="qa-reviewer",
        review_notes="Checked the selected anchor locations and four-band bundles.",
    )
    output_directory = tmp_path / "frozen"
    write_frozen_benchmark(frozen, output_directory)
    verify_frozen_benchmark(output_directory, benchmark_path=benchmark_path)
    return output_directory, benchmark_path


def test_freeze_groups_events_cells_and_weeks_and_records_distributions(
    tmp_path: Path,
) -> None:
    output_directory, benchmark_path = _frozen_directory(tmp_path)

    audit = json.loads(
        (output_directory / FROZEN_AUDIT_FILENAME).read_text(encoding="utf-8")
    )
    assert audit["leakage_check"]["status"] == "passed"
    assert set(audit["leakage_check"]["axes"]) == {
        "event_id",
        "geographic_cell",
        "time_period",
    }
    assert set(audit["distributions"]["all"]) >= {
        "season",
        "local_hour",
        "view_angle_degrees",
        "missingness_fraction",
        "confidence",
        "band_availability",
    }
    assert audit["manual_inspection"]["selected_case_count"] == 6
    assert audit["manual_inspection"]["notes_sha256"]
    assert len(audit["frozen_file_sha256"]) == 6

    test_manifest = json.loads(
        (output_directory / "test.manifest.json").read_text(encoding="utf-8")
    )
    assert test_manifest["labels_visible_to_tuning"] is False
    assert all(
        "label" not in case
        and "event_id" not in case
        and case["case_id"].startswith("test-")
        for case in test_manifest["cases"]
    )
    verify_frozen_benchmark(output_directory, benchmark_path=benchmark_path)


def test_tuning_boundary_allows_only_development_manifest(tmp_path: Path) -> None:
    output_directory, _ = _frozen_directory(tmp_path)
    project_root = tmp_path
    evaluation_root = project_root / "evaluation-data" / "frozen"
    evaluation_root.parent.mkdir(parents=True)
    output_directory.replace(evaluation_root)

    development = tuning_manifest_path(
        evaluation_root / "development.manifest.json", project_root=project_root
    )
    assert development.name == "development.manifest.json"
    with pytest.raises(ValueError, match="test and stress labels are scoring-only"):
        tuning_manifest_path(
            evaluation_root / "test.manifest.json", project_root=project_root
        )
    with pytest.raises(ValueError, match="test and stress labels are scoring-only"):
        tuning_manifest_path(
            evaluation_root / "test-labels.json", project_root=project_root
        )


def test_freeze_refuses_a_benchmark_with_one_shared_geographic_component(
    tmp_path: Path,
) -> None:
    labels_path, inventory_path = _benchmark_inputs(tmp_path)
    benchmark = build_benchmark(labels_path, inventory_path, random_seed=1234)
    benchmark_path, benchmark_audit_path = write_benchmark(
        benchmark, tmp_path / "benchmark"
    )

    with pytest.raises(ValueError, match="fewer than three independent components"):
        freeze_benchmark(
            benchmark_path,
            benchmark_audit_path,
            reviewer="qa-reviewer",
            review_notes="Review cannot proceed because the split is unsafe.",
        )
