"""Contracts for the reviewer-safe evidence presentation model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from streamlit.testing.v1 import AppTest

from firesentinel.ui.reviewer import (
    contour_preview,
    demo_cases,
    discover_reviewer_cases,
    load_candidate_mask,
    reason_explanations,
)


def _local_packet() -> dict[str, object]:
    return {
        "record_type": "local_evidence_job",
        "schema_version": 1,
        "case_id": "review-case",
        "content_hash": "a" * 64,
        "configuration": {"configuration_id": "review-config"},
        "observations": [
            {
                "observation_id": "initial",
                "channel7_crop": {
                    "timing": {"start": "2025-01-01T10:00:00Z"},
                    "geographic_bounds": {
                        "south": 39.8,
                        "west": -121.8,
                        "north": 39.9,
                        "east": -121.7,
                    },
                },
                "anomaly": {
                    "candidate_pixel_count": 6,
                    "reason_codes": ["thermal_anomaly_weak"],
                    "components": [
                        {
                            "label": 1,
                            "area_pixels": 6,
                            "bounding_box_xywh": [2, 3, 2, 3],
                            "centroid_xy": [2.5, 4.0],
                        }
                    ],
                    "contours_xy": [[[2, 3], [3, 3], [3, 5], [2, 5]]],
                },
            }
        ],
        "persistence": {
            "persistence_count": 0,
            "mean_intersection_over_union": 0.0,
            "confidence": 0.0,
        },
        "warnings": ["initial:C07:contrast_low"],
        "artifacts": [],
    }


def _real_packet() -> dict[str, object]:
    return {
        "record_type": "real_event_evidence",
        "schema_version": 1,
        "case_id": "historical-case",
        "title": "Historical thermal slice",
        "coordinates": {"lat": 39.8457, "lon": -121.7233},
        "content_hash": "b" * 64,
        "configuration": {"configuration_id": "historical-config"},
        "frames": [
            {
                "observation_id": "before",
                "scan_start": "2024-07-25T22:30:20Z",
                "channel": "C07",
                "morphology_pixel_count": 12,
                "maximum_kelvin": 375.0,
                "source_sha256": "c" * 64,
                "components": [
                    {
                        "label": 1,
                        "area_pixels": 12,
                        "bounding_box_xywh": [1, 2, 3, 4],
                        "centroid_xy": [2.5, 4.0],
                    }
                ],
                "contours_xy": [[[1, 2], [3, 2], [3, 5]]],
            }
        ],
        "measurements": [
            {
                "name": "largest_hot_region",
                "value": 12.0,
                "unit": "px",
                "missing_reason": None,
            }
        ],
        "reviewer_panel": {"filename": "before-after.png"},
    }


def test_reviewer_discovers_packets_and_exposes_masks_contours_and_location(
    tmp_path: Path,
) -> None:
    local_directory = tmp_path / "review-case" / ("a" * 64)
    observation_directory = local_directory / "observations" / "initial"
    observation_directory.mkdir(parents=True)
    np.save(
        observation_directory / "candidate-mask.npy",
        np.asarray([[0, 255], [255, 0]], dtype=np.uint8),
    )
    (local_directory / "evidence.json").write_text(
        json.dumps(_local_packet()), encoding="utf-8"
    )
    real_directory = tmp_path / "historical-case" / ("b" * 64)
    real_directory.mkdir(parents=True)
    (real_directory / "evidence.json").write_text(
        json.dumps(_real_packet()), encoding="utf-8"
    )
    broken_directory = tmp_path / "broken" / "packet"
    broken_directory.mkdir(parents=True)
    (broken_directory / "evidence.json").write_text("{", encoding="utf-8")

    catalog = discover_reviewer_cases(tmp_path)

    assert [case.case_id for case in catalog.cases] == [
        "historical-case",
        "review-case",
    ]
    assert len(catalog.warnings) == 1
    local_case = next(case for case in catalog.cases if case.case_id == "review-case")
    observation = local_case.observations[0]
    assert "39.8000" in local_case.location
    assert observation.candidate_pixels == 6
    assert observation.contour_vertex_count == 4
    assert load_candidate_mask(observation.candidate_mask_path) is not None
    assert contour_preview(observation).shape == (112, 160, 3)
    assert reason_explanations(local_case.reason_codes) == (
        "A thermal anomaly was present but did not meet the persistence threshold.",
    )


def test_each_deterministic_demo_explains_a_complete_safe_story() -> None:
    demos = {case.case_id: case for case in demo_cases()}

    assert set(demos) == {
        "demo-emerging-event",
        "demo-matched-control",
        "demo-abstention",
    }
    for case in demos.values():
        assert case.outcome.terminal
        assert case.selected_action
        assert case.considered_actions
        assert case.observations
        assert case.warnings
        assert case.provenance
        assert {row["Budget"] for row in case.budget} >= {
            "Observations",
            "Bytes",
        }
        reviewer_text = " ".join(
            (
                case.title,
                case.initial_ambiguity,
                case.outcome.label,
                case.outcome.explanation,
                *case.reason_codes,
            )
        ).lower()
        assert "confirmed wildfire" not in reviewer_text

    assert demos["demo-emerging-event"].outcome.state == "review_escalation"
    assert demos["demo-matched-control"].outcome.state == "no_persistent_evidence"
    assert demos["demo-abstention"].outcome.state == "insufficient_evidence"


def test_streamlit_page_uses_summaries_instead_of_raw_json() -> None:
    app_source = (
        Path(__file__).parents[1] / "src" / "firesentinel" / "ui" / "app.py"
    ).read_text(encoding="utf-8")

    assert "Chronological evidence strip" in app_source
    assert "Location context" in app_source
    assert "Reason codes" in app_source
    assert "Deterministic demos" in app_source
    assert "st.json(" not in app_source
    assert "st.code(" not in app_source


def test_streamlit_reviewer_loads_each_demo_from_its_button() -> None:
    app_path = Path(__file__).parents[1] / "src" / "firesentinel" / "ui" / "app.py"
    page = AppTest.from_file(str(app_path)).run(timeout=15)

    expected_headers = (
        "Demo: emerging thermal evidence",
        "Demo: matched control",
        "Demo: safe abstention",
    )
    assert [button.label for button in page.button] == [
        "Emerging event",
        "Matched control",
        "Abstention",
    ]
    for index, expected_header in enumerate(expected_headers):
        page.button[index].click().run(timeout=15)
        assert not page.exception
        assert page.header[0].value == expected_header
