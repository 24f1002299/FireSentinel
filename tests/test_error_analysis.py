"""Contracts for sealed Day 26 error and agent-value analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from firesentinel.evaluation import error_analysis
from firesentinel.evaluation.error_analysis import (
    ERROR_ANALYSIS_RECORD_TYPE,
    analyze_frozen_evaluation,
    verify_error_analysis,
    write_error_analysis,
)
from firesentinel.evaluation.frozen_run import FROZEN_EVALUATION_RECORD_TYPE


def _case_row(
    split: str,
    case_id: str,
    label: str,
    mode: str,
    prediction: str | None,
    observations: int,
) -> dict[str, object]:
    state = {
        "positive": "review_escalation",
        "control": "no_persistent_evidence",
        None: "insufficient_evidence",
    }[prediction]
    return {
        "split": split,
        "case_id": case_id,
        "label": label,
        "mode": mode,
        "outcome": {
            "state": state,
            "reason_codes": ["insufficient_evidence"],
            "confidence": 0.5,
        },
        "prediction": prediction,
        "observation_count": observations,
        "channel7_observation_count": observations,
        "evidence_time_step_count": observations,
        "resources": {
            "selected_source_bytes": observations * 100,
            "downloaded_bytes": 0,
            "latency_milliseconds": float(observations * 10),
        },
        "errors": [],
        "evidence_ids": [f"{case_id}-{mode}"],
        "trace_path": (
            None if mode != "adaptive" else f"C:/traces/{split}-{case_id}.jsonl"
        ),
    }


def _report(path: Path) -> Path:
    definitions = (
        ("test", "test-success", "positive", None, "positive", "positive", 1, 4, 2),
        ("test", "test-control", "control", "control", "control", "control", 1, 4, 1),
        ("stress", "stress-abstain", "positive", None, None, None, 1, 4, 3),
        (
            "stress",
            "stress-limit",
            "control",
            "control",
            "control",
            "positive",
            1,
            4,
            3,
        ),
    )
    rows: list[dict[str, object]] = []
    for (
        split,
        case_id,
        label,
        one,
        fixed,
        adaptive,
        one_obs,
        fixed_obs,
        adaptive_obs,
    ) in definitions:
        rows.extend(
            (
                _case_row(split, case_id, label, "one_shot", one, one_obs),
                _case_row(split, case_id, label, "fixed_bundle", fixed, fixed_obs),
                _case_row(split, case_id, label, "adaptive", adaptive, adaptive_obs),
            )
        )
    payload = {
        "schema_version": 1,
        "record_type": FROZEN_EVALUATION_RECORD_TYPE,
        "analysis_status": "frozen_before_error_analysis",
        "input_hashes": {
            "test.manifest.json": "1" * 64,
            "test-labels.json": "2" * 64,
            "stress.manifest.json": "3" * 64,
            "stress-labels.json": "4" * 64,
        },
        "configuration": {},
        "per_case_results": rows,
        "aggregate_tables": [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _complete_checkpoint(case_id: str) -> dict[str, object]:
    return {
        "case_id": case_id,
        "to_state": "review",
        "sequence": 12,
        "selected_observation_ids": ["initial", "later"],
        "evidence_ids": ["evidence-a"],
        "outcome": {"state": "review_escalation"},
    }


def test_analysis_links_error_tables_representatives_and_traces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _report(tmp_path / "frozen-evaluation.json")

    # Use the path itself to retain the opaque case id in the mocked checkpoint.
    def checkpoint(path: Path) -> dict[str, object]:
        case_id = path.name.removesuffix(".jsonl").split("-", maxsplit=1)[1]
        return _complete_checkpoint(case_id)

    monkeypatch.setattr(error_analysis, "load_last_complete_transition", checkpoint)
    analysis = analyze_frozen_evaluation(report)
    payload = analysis.to_dict()

    assert payload["record_type"] == ERROR_ANALYSIS_RECORD_TYPE
    tables = cast(dict[str, object], payload["tables"])
    efficiency = cast(list[dict[str, object]], tables["mode_efficiency"])
    adaptive = next(row for row in efficiency if row["mode"] == "adaptive")
    fixed = next(row for row in efficiency if row["mode"] == "fixed_bundle")
    assert adaptive["recall"] == fixed["recall"] == 0.5
    assert adaptive["mean_observations"] == 2.25
    assert fixed["mean_observations"] == 4.0

    errors = cast(list[dict[str, object]], tables["error_outcomes"])
    adaptive_errors = next(row for row in errors if row["mode"] == "adaptive")
    assert adaptive_errors["false_positive_case_keys"] == ["stress/stress-limit"]
    assert adaptive_errors["abstention_case_keys"] == ["stress/stress-abstain"]

    no_help = cast(list[dict[str, object]], tables["extra_observations_without_help"])
    assert [row["case_id"] for row in no_help] == ["stress-abstain", "stress-limit"]
    representatives = cast(
        dict[str, dict[str, object]], payload["representative_cases"]
    )
    assert representatives["success"]["case_key"] == "test/test-success"
    assert representatives["control"]["case_key"] == "test/test-control"
    assert representatives["abstention"]["case_key"] == "stress/stress-abstain"
    assert representatives["genuine_limitation"]["case_key"] == "stress/stress-limit"
    assert all(
        claim["table"] and claim["trace_case_keys"]
        for claim in cast(list[dict[str, object]], payload["headline_claims"])
    )

    output = write_error_analysis(analysis, tmp_path / "error-analysis.json")
    verify_error_analysis(output, report)
    assert write_error_analysis(analysis, output) == output


def test_analysis_rejects_unsealed_or_incomplete_mode_rows(tmp_path: Path) -> None:
    report = _report(tmp_path / "frozen-evaluation.json")
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["analysis_status"] = "not_sealed"
    report.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="sealed before error analysis"):
        analyze_frozen_evaluation(report)

    payload["analysis_status"] = "frozen_before_error_analysis"
    rows = payload["per_case_results"]
    assert isinstance(rows, list)
    rows.pop()
    report.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="cover every case exactly once"):
        analyze_frozen_evaluation(report)


def test_verify_requires_each_headline_to_link_table_and_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _report(tmp_path / "frozen-evaluation.json")
    monkeypatch.setattr(
        error_analysis,
        "load_last_complete_transition",
        lambda path: _complete_checkpoint("test-success"),
    )
    analysis = analyze_frozen_evaluation(report)
    output = write_error_analysis(analysis, tmp_path / "error-analysis.json")
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["headline_claims"][0]["trace_case_keys"] = []
    output.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="supporting trace"):
        verify_error_analysis(output, report)
