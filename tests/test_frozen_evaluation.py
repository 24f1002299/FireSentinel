"""Contracts for sealed test/stress evaluation scoring and reporting."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from firesentinel.core.records import Channel, OutcomeState, ReasonCode
from firesentinel.evaluation import frozen_run
from firesentinel.evaluation.frozen_run import (
    EvaluationMode,
    FrozenCase,
    FrozenCaseResult,
    FrozenEvaluationParameters,
    load_frozen_scoring_cases,
    run_frozen_evaluation,
    verify_frozen_evaluation_report,
    write_frozen_evaluation_report,
)
from firesentinel.evaluation.runner import BaselineCase, BaselineSource
from tests.test_baseline_runner import _parameters
from tests.test_freeze import _frozen_directory


def _case(case_id: str) -> BaselineCase:
    moment = datetime(2025, 1, 1, tzinfo=UTC).isoformat().replace("+00:00", "Z")
    return BaselineCase(
        case_id,
        34.0,
        -118.0,
        tuple(
            BaselineSource(
                role,
                channel,
                moment,
                f"{case_id}-{role}",
                "bucket",
                f"key/{role}",
                100,
                "0" * 64,
            )
            for role, channel in (
                ("baseline", Channel.C07),
                ("initial", Channel.C07),
                ("later", Channel.C07),
                ("alternate", Channel.C14),
            )
        ),
    )


def _row(
    split: str,
    case_id: str,
    label: str,
    mode: EvaluationMode,
    outcome: OutcomeState,
) -> FrozenCaseResult:
    prediction = {
        OutcomeState.REVIEW_ESCALATION: "positive",
        OutcomeState.NO_PERSISTENT_EVIDENCE: "control",
    }.get(outcome)
    return FrozenCaseResult(
        split,
        case_id,
        label,
        mode,
        outcome,
        (ReasonCode.INSUFFICIENT_EVIDENCE,),
        0.5,
        prediction,
        1,
        1,
        1,
        100,
        0,
        5.0,
        (),
        (),
    )


def test_metrics_keep_abstentions_visible_and_measure_resolution(
    tmp_path: Path,
) -> None:
    rows = (
        _row(
            "test",
            "test-000000000000000000000001",
            "positive",
            EvaluationMode.ONE_SHOT,
            OutcomeState.INSUFFICIENT_EVIDENCE,
        ),
        _row(
            "test",
            "test-000000000000000000000002",
            "control",
            EvaluationMode.ONE_SHOT,
            OutcomeState.NO_PERSISTENT_EVIDENCE,
        ),
        _row(
            "test",
            "test-000000000000000000000001",
            "positive",
            EvaluationMode.ADAPTIVE,
            OutcomeState.REVIEW_ESCALATION,
        ),
        _row(
            "test",
            "test-000000000000000000000002",
            "control",
            EvaluationMode.ADAPTIVE,
            OutcomeState.NO_PERSISTENT_EVIDENCE,
        ),
    )
    parameters = FrozenEvaluationParameters(_parameters(tmp_path), bootstrap_samples=25)

    tables = frozen_run._aggregate_tables(rows, parameters)
    one_shot = next(
        table
        for table in tables
        if table["split"] == "test" and table["mode"] == "one_shot"
    )
    adaptive = next(
        table
        for table in tables
        if table["split"] == "test" and table["mode"] == "adaptive"
    )

    one_metrics = one_shot["classification"]
    assert isinstance(one_metrics, dict)
    assert one_metrics["macro_f1"] == 0.5
    assert one_metrics["abstention_coverage"] == 0.5
    assert adaptive["ambiguous_case_resolution"] == {
        "one_shot_ambiguous_case_count": 1,
        "resolved_case_count": 1,
        "resolution_rate": 1.0,
    }
    assert adaptive["classification"] == {
        "macro_f1": 1.0,
        "macro_precision": 1.0,
        "macro_recall": 1.0,
        "abstention_coverage": 0.0,
        "abstention_count": 0,
        "per_class": {
            "positive": {
                "true_positive": 1,
                "false_positive": 0,
                "false_negative": 0,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
            },
            "control": {
                "true_positive": 1,
                "false_positive": 0,
                "false_negative": 0,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
            },
        },
    }


def test_runner_seals_all_modes_and_refuses_changed_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cases = (
        FrozenCase(_case("test-000000000000000000000001"), "test", "positive"),
        FrozenCase(_case("stress-000000000000000000000001"), "stress", "control"),
    )
    input_hashes = {
        "test.manifest.json": "1" * 64,
        "test-labels.json": "2" * 64,
        "stress.manifest.json": "3" * 64,
        "stress-labels.json": "4" * 64,
    }
    monkeypatch.setattr(frozen_run, "verify_frozen_benchmark", lambda _: None)
    monkeypatch.setattr(
        frozen_run, "load_frozen_scoring_cases", lambda _: (cases, dict(input_hashes))
    )

    def fixed(
        frozen: FrozenCase,
        mode: EvaluationMode,
        *_: object,
        **__: object,
    ) -> FrozenCaseResult:
        outcome = (
            OutcomeState.REVIEW_ESCALATION
            if frozen.label == "positive"
            else OutcomeState.NO_PERSISTENT_EVIDENCE
        )
        return _row(frozen.split, frozen.case.case_id, frozen.label, mode, outcome)

    def adaptive(frozen: FrozenCase, *_: object, **__: object) -> FrozenCaseResult:
        return fixed(frozen, EvaluationMode.ADAPTIVE)

    monkeypatch.setattr(frozen_run, "_run_fixed_case", fixed)
    monkeypatch.setattr(frozen_run, "_run_adaptive_case", adaptive)
    parameters = FrozenEvaluationParameters(_parameters(tmp_path), bootstrap_samples=10)

    result = run_frozen_evaluation(
        tmp_path / "frozen",
        tmp_path / "artifacts",
        parameters,
        source_resolver=lambda _case_id, _source: tmp_path / "cache" / "source.nc",
        project_root=tmp_path,
    )
    assert len(result.cases) == 6
    assert len(result.aggregates) == 9
    assert result.to_dict()["analysis_status"] == "frozen_before_error_analysis"
    assert set(result.input_hashes) == {
        *input_hashes,
        "evidence_template_configuration_sha256",
        "implementation_sha256",
    }

    report = write_frozen_evaluation_report(result, tmp_path / "report.json")
    assert json.loads(report.read_text(encoding="utf-8"))["per_case_results"]
    assert write_frozen_evaluation_report(result, report) == report
    changed = FrozenEvaluationParameters(_parameters(tmp_path), bootstrap_samples=11)
    changed_result = frozen_run.FrozenEvaluationResult(
        result.input_hashes,
        changed.configuration_dict(),
        result.cases,
        result.aggregates,
    )
    with pytest.raises(FileExistsError, match="refusing to replace"):
        write_frozen_evaluation_report(changed_result, report)


def test_loader_reads_only_opaque_cases_after_freeze(tmp_path: Path) -> None:
    frozen_directory, _ = _frozen_directory(tmp_path)
    cases, hashes = load_frozen_scoring_cases(frozen_directory)

    assert cases
    assert {case.split for case in cases} == {"test", "stress"}
    assert all(case.case.case_id.startswith(("test-", "stress-")) for case in cases)
    assert all(case.label in {"positive", "control"} for case in cases)
    assert set(hashes) == {
        "test.manifest.json",
        "test-labels.json",
        "stress.manifest.json",
        "stress-labels.json",
    }


def test_verify_report_detects_changed_evidence_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parameters = FrozenEvaluationParameters(_parameters(tmp_path), bootstrap_samples=5)
    input_hashes = {
        "test.manifest.json": "1" * 64,
        "test-labels.json": "2" * 64,
        "stress.manifest.json": "3" * 64,
        "stress-labels.json": "4" * 64,
    }
    monkeypatch.setattr(
        frozen_run,
        "load_frozen_scoring_cases",
        lambda _: ((), dict(input_hashes)),
    )
    result = frozen_run.FrozenEvaluationResult(
        {
            **input_hashes,
            "evidence_template_configuration_sha256": frozen_run._sha256(
                frozen_run._canonical_json(
                    parameters.baseline.evidence_template.to_dict(include_paths=False)
                )
            ),
            "implementation_sha256": frozen_run._implementation_sha256(),
        },
        parameters.configuration_dict(),
        (),
        (),
    )
    path = write_frozen_evaluation_report(result, tmp_path / "report.json")
    verify_frozen_evaluation_report(
        path, tmp_path / "frozen", parameters.baseline.evidence_template
    )
