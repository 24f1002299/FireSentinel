"""Tests for isolated FIRMS event-reference ingestion."""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import fields
from pathlib import Path

import pytest

import firesentinel.agent
from firesentinel.agent import replay
from firesentinel.agent.label_boundary import runtime_input_path
from firesentinel.config import Settings, load_settings
from firesentinel.evaluation.firms import (
    AUDIT_FILENAME,
    LABELS_FILENAME,
    ingest_firms_csvs,
    write_evaluation_references,
)


def _write_source(path: Path) -> None:
    path.write_text(
        "latitude,longitude,brightness,bright_ti4,acq_date,acq_time,confidence,instrument,frp\n"
        "34.0000004,-118.0000004,325.1234,,2024-07-25,930,H,viirs,99\n"
        "34.0000000,-118.0000000,325.1230,,2024-07-25,0930,high,VIIRS,1\n"
        "34.0300000,-118.0000000,,325.123,2024-07-25,0955,85,modis,2\n"
        "35.0000000,-119.0000000,330.0,,2024-07-25,1200,n,MODIS,3\n",
        encoding="utf-8",
    )


def test_ingestion_retains_only_permitted_fields_and_audits_normalization(
    tmp_path: Path,
) -> None:
    source = tmp_path / "firms.csv"
    _write_source(source)

    ingestion = ingest_firms_csvs(
        [source], maximum_distance_km=10, maximum_time_gap_minutes=60
    )

    assert ingestion.source_row_count == 4
    assert ingestion.duplicate_count == 1
    assert len(ingestion.normalized_detections) == 3
    assert len(ingestion.events) == 2
    assert ingestion.events[0].start_time.isoformat() == "2024-07-25T09:30:00+00:00"
    assert ingestion.events[0].end_time.isoformat() == "2024-07-25T09:55:00+00:00"
    assert ingestion.events[0].detections[1].brightness_kelvin == 325.123
    assert ingestion.events[0].detections[1].confidence == "85"
    assert ingestion.events[0].detections[1].instrument == "MODIS"

    labels = ingestion.labels_payload()
    detected_fields = set(labels["events"][0]["detections"][0])  # type: ignore[index]
    assert detected_fields == {
        "acquisition_time_utc",
        "latitude",
        "longitude",
        "confidence",
        "brightness_kelvin",
        "instrument",
    }
    assert labels["source_hashes"] == [hashlib.sha256(source.read_bytes()).hexdigest()]

    output_directory = tmp_path / "evaluation-data" / "firms"
    labels_path, audit_path = write_evaluation_references(ingestion, output_directory)
    assert labels_path.name == LABELS_FILENAME
    assert audit_path.name == AUDIT_FILENAME
    assert write_evaluation_references(ingestion, output_directory) == (
        labels_path,
        audit_path,
    )

    labels_bytes = labels_path.read_bytes()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["labels_sha256"] == hashlib.sha256(labels_bytes).hexdigest()
    assert audit["counts"] == {
        "blank_rows_ignored": 0,
        "duplicate_detections_removed": 1,
        "event_windows": 2,
        "normalized_detections": 4,
        "source_files": 1,
        "source_rows": 4,
        "unique_detections": 3,
    }
    assert audit["date_range"] == {
        "first_acquisition_time_utc": "2024-07-25T09:30:00Z",
        "last_acquisition_time_utc": "2024-07-25T12:00:00Z",
    }
    assert audit["normalization_statistics"]["coordinate_precision_decimal_places"] == 6
    assert audit["normalization_statistics"]["timestamp_timezone"] == "UTC"


def test_ingestion_rejects_bad_rows_instead_of_silently_dropping_them(
    tmp_path: Path,
) -> None:
    source = tmp_path / "bad-firms.csv"
    source.write_text(
        "latitude,longitude,brightness,acq_date,acq_time,confidence,instrument\n"
        "91,0,300,2024-07-25,1200,h,VIIRS\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"bad-firms.csv:2: latitude"):
        ingest_firms_csvs([source])


def test_agent_boundary_rejects_evaluation_data_and_runtime_settings_do_not_expose_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    labels_path = tmp_path / "evaluation-data" / "firms" / LABELS_FILENAME
    labels_path.parent.mkdir(parents=True)
    labels_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="cannot read evaluation-only labels"):
        runtime_input_path(labels_path, project_root=tmp_path)
    monkeypatch.setattr(replay, "load_settings", lambda: load_settings(root=tmp_path))
    with pytest.raises(ValueError, match="cannot read evaluation-only labels"):
        replay.main(["--input", str(labels_path)])
    assert "evaluation_data" not in {field.name for field in fields(Settings)}
    assert "labels" not in {field.name for field in fields(Settings)}


def test_agent_package_has_no_evaluation_imports() -> None:
    agent_directory = Path(firesentinel.agent.__file__).parent
    for source in agent_directory.rglob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("firesentinel.evaluation")
            if isinstance(node, ast.Import):
                assert all(
                    not alias.name.startswith("firesentinel.evaluation")
                    for alias in node.names
                )
