"""Freeze trace-supported error analysis from a sealed Day 25 report.

This module never reruns evidence, changes a threshold, or reads a benchmark
label file.  Its sole input is the already sealed frozen-evaluation report.
It derives reviewer-facing error tables and deterministic representatives from
that report, attaching the persisted adaptive-loop trace wherever available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from firesentinel.agent.loop import load_last_complete_transition
from firesentinel.config import load_settings
from firesentinel.evaluation.frozen_run import (
    FROZEN_EVALUATION_RECORD_TYPE,
    EvaluationMode,
)

ERROR_ANALYSIS_SCHEMA_VERSION = 1
ERROR_ANALYSIS_RECORD_TYPE = "firesentinel_frozen_error_analysis"
_MODES = tuple(mode.value for mode in EvaluationMode)
_LABELS = ("positive", "control")
_SPLITS = ("test", "stress")


@dataclass(frozen=True, slots=True)
class _CaseResult:
    """Validated Day 25 row, deliberately limited to the analysis facts."""

    split: str
    case_id: str
    label: str
    mode: str
    outcome_state: str
    reason_codes: tuple[str, ...]
    confidence: float
    prediction: str | None
    observation_count: int
    selected_source_bytes: int
    latency_milliseconds: float
    errors: tuple[dict[str, str], ...]
    evidence_ids: tuple[str, ...]
    trace_path: str | None

    @property
    def is_correct(self) -> bool:
        return self.prediction == self.label

    @property
    def key(self) -> tuple[str, str]:
        return self.split, self.case_id

    def to_dict(self) -> dict[str, object]:
        return {
            "split": self.split,
            "case_id": self.case_id,
            "label": self.label,
            "mode": self.mode,
            "outcome_state": self.outcome_state,
            "reason_codes": list(self.reason_codes),
            "confidence": self.confidence,
            "prediction": self.prediction,
            "correct": self.is_correct,
            "observation_count": self.observation_count,
            "selected_source_bytes": self.selected_source_bytes,
            "latency_milliseconds": self.latency_milliseconds,
            "errors": list(self.errors),
            "evidence_ids": list(self.evidence_ids),
            "trace_path": self.trace_path,
        }


@dataclass(frozen=True, slots=True)
class FrozenErrorAnalysis:
    """An immutable, post-evaluation analysis record with no tuning output."""

    source_evaluation_sha256: str
    source_input_hashes: dict[str, str]
    tables: dict[str, object]
    representative_cases: dict[str, object]
    headline_claims: tuple[dict[str, object], ...]
    trace_index: dict[str, dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": ERROR_ANALYSIS_SCHEMA_VERSION,
            "record_type": ERROR_ANALYSIS_RECORD_TYPE,
            "analysis_status": "frozen_after_evaluation_before_tuning",
            "source_evaluation_sha256": self.source_evaluation_sha256,
            "source_input_hashes": dict(sorted(self.source_input_hashes.items())),
            "methodology": {
                "no_evidence_replay": True,
                "no_threshold_or_policy_tuning": True,
                "false_positive": "control labelled case predicted positive",
                "decisive_false_negative": ("positive labelled case predicted control"),
                "abstention": (
                    "prediction is absent (human review, insufficient evidence, "
                    "or failure)"
                ),
                "extra_observation_help": (
                    "adaptive uses more observations than one-shot and changes an "
                    "incorrect or abstaining one-shot result into the true class"
                ),
                "trace_support": (
                    "representative adaptive cases require a readable persisted "
                    "terminal trace"
                ),
            },
            "tables": self.tables,
            "representative_cases": self.representative_cases,
            "headline_claims": list(self.headline_claims),
            "trace_index": dict(sorted(self.trace_index.items())),
        }


def analyze_frozen_evaluation(report_path: Path) -> FrozenErrorAnalysis:
    """Derive tables and representatives from one sealed evaluation report."""

    source = Path(report_path)
    raw = _read_bytes(source, "frozen evaluation report")
    payload = _object(raw, "frozen evaluation report")
    _validate_source_report(payload)
    source_hashes = _string_mapping(payload["input_hashes"], "input_hashes")
    rows = _rows(payload["per_case_results"])
    by_mode = _rows_by_mode(rows)
    trace_index = _trace_index(by_mode[EvaluationMode.ADAPTIVE.value])
    tables: dict[str, object] = {
        "mode_efficiency": _mode_efficiency_table(by_mode),
        "error_outcomes": _error_outcome_table(by_mode),
        "agent_value": _agent_value_table(by_mode),
        "extra_observations_without_help": _extra_observation_table(by_mode),
    }
    representatives = _representatives(by_mode, trace_index)
    claims = _headline_claims(tables, representatives, trace_index)
    return FrozenErrorAnalysis(
        _sha256(raw), source_hashes, tables, representatives, claims, trace_index
    )


def write_error_analysis(
    analysis: FrozenErrorAnalysis, path: Path, *, overwrite: bool = False
) -> Path:
    """Atomically write a frozen analysis, refusing a changed replacement."""

    if not isinstance(analysis, FrozenErrorAnalysis):
        raise TypeError("analysis must be FrozenErrorAnalysis")
    destination = Path(path)
    content = _canonical_json(analysis.to_dict()) + b"\n"
    if destination.exists() and destination.read_bytes() != content and not overwrite:
        raise FileExistsError(
            "refusing to replace frozen error analysis "
            f"'{destination}'; use --overwrite"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", delete=False, dir=destination.parent, prefix=f".{destination.name}."
    ) as temporary:
        temporary.write(content)
        temporary_path = Path(temporary.name)
    temporary_path.replace(destination)
    return destination


def verify_error_analysis(path: Path, evaluation_report_path: Path) -> None:
    """Confirm a stored analysis is tied to the exact sealed evaluation report."""

    payload = _object(
        _read_bytes(path, "frozen error analysis"), "frozen error analysis"
    )
    if payload.get("record_type") != ERROR_ANALYSIS_RECORD_TYPE:
        raise ValueError("frozen error analysis has an unexpected record_type")
    if payload.get("analysis_status") != "frozen_after_evaluation_before_tuning":
        raise ValueError("frozen error analysis has an unexpected status")
    expected = _sha256(_read_bytes(evaluation_report_path, "frozen evaluation report"))
    if payload.get("source_evaluation_sha256") != expected:
        raise ValueError("frozen error analysis does not match the sealed evaluation")
    claims = payload.get("headline_claims")
    tables = payload.get("tables")
    trace_index = payload.get("trace_index")
    if not isinstance(claims, list) or not claims:
        raise ValueError("frozen error analysis must contain headline claims")
    if not isinstance(tables, Mapping) or not isinstance(trace_index, Mapping):
        raise ValueError("frozen error analysis lacks support tables or traces")
    for claim in claims:
        if not isinstance(claim, Mapping):
            raise ValueError("headline claim is invalid")
        table = claim.get("table")
        traces = claim.get("trace_case_keys")
        if not isinstance(table, str) or not table:
            raise ValueError("headline claim lacks a supporting table")
        if table not in tables:
            raise ValueError("headline claim refers to an unknown supporting table")
        if not isinstance(traces, list) or not traces:
            raise ValueError("headline claim lacks a supporting trace")
        for trace_key in traces:
            trace = trace_index.get(trace_key)
            if not isinstance(trace_key, str) or not isinstance(trace, Mapping):
                raise ValueError("headline claim refers to an unknown supporting trace")
            if trace.get("status") != "complete":
                raise ValueError(
                    "headline claim refers to an unreadable supporting trace"
                )


def _validate_source_report(payload: Mapping[str, object]) -> None:
    if payload.get("record_type") != FROZEN_EVALUATION_RECORD_TYPE:
        raise ValueError("error analysis requires a frozen evaluation report")
    if payload.get("analysis_status") != "frozen_before_error_analysis":
        raise ValueError("evaluation report was not sealed before error analysis")
    if not isinstance(payload.get("input_hashes"), Mapping):
        raise ValueError("frozen evaluation report lacks input hashes")
    if not isinstance(payload.get("per_case_results"), list):
        raise ValueError("frozen evaluation report lacks per-case results")
    if not isinstance(payload.get("aggregate_tables"), list):
        raise ValueError("frozen evaluation report lacks aggregate tables")


def _rows(value: object) -> tuple[_CaseResult, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("frozen evaluation report must have per-case rows")
    rows = tuple(_row(item) for item in value)
    expected_keys = {
        (split, case_id, mode)
        for split in _SPLITS
        for case_id in {row.case_id for row in rows if row.split == split}
        for mode in _MODES
    }
    actual_keys = {(row.split, row.case_id, row.mode) for row in rows}
    if actual_keys != expected_keys or len(actual_keys) != len(rows):
        raise ValueError("evaluation rows must cover every case exactly once per mode")
    return tuple(sorted(rows, key=lambda row: (row.split, row.case_id, row.mode)))


def _row(value: object) -> _CaseResult:
    if not isinstance(value, Mapping):
        raise ValueError("per-case result must be an object")
    split = _required_choice(value, "split", _SPLITS)
    case_id = _required_text(value, "case_id")
    label = _required_choice(value, "label", _LABELS)
    mode = _required_choice(value, "mode", _MODES)
    outcome = value.get("outcome")
    resources = value.get("resources")
    if not isinstance(outcome, Mapping) or not isinstance(resources, Mapping):
        raise ValueError("per-case result lacks outcome or resources")
    prediction = value.get("prediction")
    if prediction is not None and prediction not in _LABELS:
        raise ValueError("per-case prediction is invalid")
    reason_codes = _string_tuple(outcome.get("reason_codes"), "outcome.reason_codes")
    errors = _error_rows(value.get("errors"))
    trace_path = value.get("trace_path")
    if trace_path is not None and not isinstance(trace_path, str):
        raise ValueError("per-case trace_path is invalid")
    return _CaseResult(
        split=split,
        case_id=case_id,
        label=label,
        mode=mode,
        outcome_state=_required_text(outcome, "state"),
        reason_codes=reason_codes,
        confidence=_number(outcome.get("confidence"), "outcome.confidence"),
        prediction=cast(str | None, prediction),
        observation_count=_non_negative_integer(
            value.get("observation_count"), "observation_count"
        ),
        selected_source_bytes=_non_negative_integer(
            resources.get("selected_source_bytes"), "selected_source_bytes"
        ),
        latency_milliseconds=_non_negative_number(
            resources.get("latency_milliseconds"), "latency_milliseconds"
        ),
        errors=errors,
        evidence_ids=_string_tuple(value.get("evidence_ids"), "evidence_ids"),
        trace_path=trace_path,
    )


def _rows_by_mode(rows: tuple[_CaseResult, ...]) -> dict[str, tuple[_CaseResult, ...]]:
    grouped = {mode: tuple(row for row in rows if row.mode == mode) for mode in _MODES}
    case_keys = {row.key for row in grouped[EvaluationMode.ONE_SHOT.value]}
    if any({row.key for row in grouped[mode]} != case_keys for mode in _MODES):
        raise ValueError("modes do not cover the same frozen cases")
    return grouped


def _mode_efficiency_table(
    by_mode: Mapping[str, tuple[_CaseResult, ...]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for mode in _MODES:
        values = by_mode[mode]
        positive = tuple(row for row in values if row.label == "positive")
        recall = _ratio(
            sum(row.prediction == "positive" for row in positive), len(positive)
        )
        rows.append(
            {
                "mode": mode,
                "case_count": len(values),
                "recall": recall,
                "mean_observations": _mean([row.observation_count for row in values]),
                "mean_selected_source_bytes": _mean(
                    [row.selected_source_bytes for row in values]
                ),
                "mean_latency_milliseconds": _mean(
                    [row.latency_milliseconds for row in values]
                ),
            }
        )
    return rows


def _error_outcome_table(
    by_mode: Mapping[str, tuple[_CaseResult, ...]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for mode in _MODES:
        values = by_mode[mode]
        false_positive = tuple(
            row
            for row in values
            if row.label == "control" and row.prediction == "positive"
        )
        decisive_false_negative = tuple(
            row
            for row in values
            if row.label == "positive" and row.prediction == "control"
        )
        abstentions = tuple(row for row in values if row.prediction is None)
        scoring_false_negatives = tuple(
            row
            for row in values
            if row.label == "positive" and row.prediction != "positive"
        )
        rows.append(
            {
                "mode": mode,
                "false_positive_count": len(false_positive),
                "false_positive_case_keys": _case_keys(false_positive),
                "decisive_false_negative_count": len(decisive_false_negative),
                "decisive_false_negative_case_keys": _case_keys(
                    decisive_false_negative
                ),
                "abstention_count": len(abstentions),
                "abstention_case_keys": _case_keys(abstentions),
                "scoring_false_negative_count": len(scoring_false_negatives),
            }
        )
    return rows


def _agent_value_table(
    by_mode: Mapping[str, tuple[_CaseResult, ...]],
) -> list[dict[str, object]]:
    one_shot = _by_key(by_mode[EvaluationMode.ONE_SHOT.value])
    rows: list[dict[str, object]] = []
    for mode in (EvaluationMode.FIXED_BUNDLE.value, EvaluationMode.ADAPTIVE.value):
        values = _by_key(by_mode[mode])
        improved = tuple(
            current
            for key, current in values.items()
            if current.is_correct and not one_shot[key].is_correct
        )
        regressed = tuple(
            current
            for key, current in values.items()
            if not current.is_correct and one_shot[key].is_correct
        )
        unchanged_correct = sum(
            current.is_correct and one_shot[key].is_correct
            for key, current in values.items()
        )
        rows.append(
            {
                "comparison": f"{mode}_versus_one_shot",
                "improved_case_count": len(improved),
                "improved_case_keys": _case_keys(improved),
                "regressed_case_count": len(regressed),
                "regressed_case_keys": _case_keys(regressed),
                "unchanged_correct_case_count": unchanged_correct,
            }
        )
    return rows


def _extra_observation_table(
    by_mode: Mapping[str, tuple[_CaseResult, ...]],
) -> list[dict[str, object]]:
    one_shot = _by_key(by_mode[EvaluationMode.ONE_SHOT.value])
    adaptive = _by_key(by_mode[EvaluationMode.ADAPTIVE.value])
    rows: list[dict[str, object]] = []
    for key, current in adaptive.items():
        initial = one_shot[key]
        extra = current.observation_count > initial.observation_count
        helped = extra and current.is_correct and not initial.is_correct
        if extra and not helped:
            rows.append(
                {
                    "split": current.split,
                    "case_id": current.case_id,
                    "label": current.label,
                    "one_shot": initial.to_dict(),
                    "adaptive": current.to_dict(),
                    "reason": _no_help_reason(initial, current),
                }
            )
    return sorted(rows, key=lambda row: (str(row["split"]), str(row["case_id"])))


def _no_help_reason(initial: _CaseResult, adaptive: _CaseResult) -> str:
    if not adaptive.is_correct and initial.is_correct:
        return "adaptive regressed after requesting more observations"
    if adaptive.prediction is None:
        return "adaptive remained safely abstaining after more observations"
    if adaptive.prediction == initial.prediction:
        return "adaptive kept the same decision after more observations"
    return "adaptive changed the decision without correcting the one-shot result"


def _trace_index(rows: tuple[_CaseResult, ...]) -> dict[str, dict[str, object]]:
    return {_trace_key(row): _trace_summary(row) for row in rows}


def _trace_summary(row: _CaseResult) -> dict[str, object]:
    if row.trace_path is None:
        return {
            "status": "unavailable",
            "reason": "adaptive result has no persisted trace path",
        }
    path = Path(row.trace_path)
    try:
        checkpoint = load_last_complete_transition(path)
    except ValueError as error:
        return {"status": "unavailable", "path": str(path), "reason": str(error)}
    if checkpoint is None:
        return {
            "status": "unavailable",
            "path": str(path),
            "reason": "trace has no complete transition",
        }
    if checkpoint.get("case_id") != row.case_id:
        return {
            "status": "unavailable",
            "path": str(path),
            "reason": "trace case_id does not match evaluation row",
        }
    state = checkpoint.get("to_state")
    outcome = checkpoint.get("outcome")
    if state not in {"finalize", "abstain", "review", "failure"}:
        return {
            "status": "unavailable",
            "path": str(path),
            "reason": "trace has no terminal transition",
        }
    return {
        "status": "complete",
        "path": str(path),
        "terminal_state": state,
        "sequence": checkpoint.get("sequence"),
        "selected_observation_ids": checkpoint.get("selected_observation_ids"),
        "evidence_ids": checkpoint.get("evidence_ids"),
        "outcome": outcome,
    }


def _representatives(
    by_mode: Mapping[str, tuple[_CaseResult, ...]],
    trace_index: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    one_shot = _by_key(by_mode[EvaluationMode.ONE_SHOT.value])
    adaptive = _by_key(by_mode[EvaluationMode.ADAPTIVE.value])
    fixed = _by_key(by_mode[EvaluationMode.FIXED_BUNDLE.value])

    success = _first_trace_supported(
        trace_index,
        (
            row
            for row in adaptive.values()
            if row.label == "positive"
            and row.is_correct
            and not one_shot[row.key].is_correct
        ),
    )
    control = _first_trace_supported(
        trace_index,
        (row for row in adaptive.values() if row.label == "control" and row.is_correct),
    )
    abstention = _first_trace_supported(
        trace_index, (row for row in adaptive.values() if row.prediction is None)
    )
    limitation = _first_trace_supported(
        trace_index,
        (
            row
            for row in adaptive.values()
            if row.observation_count > one_shot[row.key].observation_count
            and not (row.is_correct and not one_shot[row.key].is_correct)
            and not row.is_correct
            and row.prediction is not None
        ),
    )
    if limitation is None:
        limitation = _first_trace_supported(
            trace_index,
            (
                row
                for row in adaptive.values()
                if row.observation_count > one_shot[row.key].observation_count
                and not (row.is_correct and not one_shot[row.key].is_correct)
                and not row.is_correct
            ),
        )

    return {
        "success": _representative("success", success, one_shot, fixed, trace_index),
        "control": _representative("control", control, one_shot, fixed, trace_index),
        "abstention": _representative(
            "abstention", abstention, one_shot, fixed, trace_index
        ),
        "genuine_limitation": _representative(
            "genuine_limitation", limitation, one_shot, fixed, trace_index
        ),
    }


def _first_trace_supported(
    trace_index: Mapping[str, Mapping[str, object]], rows: Iterable[_CaseResult]
) -> _CaseResult | None:
    """Select only a readable trace-backed case in deterministic key order."""

    return next(
        (
            row
            for row in sorted(rows, key=lambda row: row.key)
            if trace_index[_trace_key(row)].get("status") == "complete"
        ),
        None,
    )


def _representative(
    category: str,
    adaptive: _CaseResult | None,
    one_shot: Mapping[tuple[str, str], _CaseResult],
    fixed: Mapping[tuple[str, str], _CaseResult],
    trace_index: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    if adaptive is None:
        return {
            "category": category,
            "status": "unavailable",
            "reason": "no matching adaptive case exists in the sealed evaluation",
        }
    trace = trace_index[_trace_key(adaptive)]
    if trace.get("status") != "complete":
        return {
            "category": category,
            "status": "unavailable",
            "reason": "matching case has no readable persisted adaptive trace",
            "adaptive": adaptive.to_dict(),
            "trace": dict(trace),
        }
    return {
        "category": category,
        "status": "selected",
        "case_key": _trace_key(adaptive),
        "one_shot": one_shot[adaptive.key].to_dict(),
        "fixed_bundle": fixed[adaptive.key].to_dict(),
        "adaptive": adaptive.to_dict(),
        "trace": dict(trace),
    }


def _headline_claims(
    tables: Mapping[str, object],
    representatives: Mapping[str, object],
    trace_index: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    efficiency = tables["mode_efficiency"]
    assert isinstance(efficiency, list)
    by_mode = {row["mode"]: row for row in efficiency if isinstance(row, Mapping)}
    fixed = by_mode[EvaluationMode.FIXED_BUNDLE.value]
    adaptive = by_mode[EvaluationMode.ADAPTIVE.value]
    assert isinstance(fixed, Mapping) and isinstance(adaptive, Mapping)
    traces = _selected_trace_keys(representatives)
    if not traces:
        traces = _complete_trace_keys(trace_index)
    if not traces:
        raise ValueError("error analysis needs at least one readable adaptive trace")
    no_help = tables["extra_observations_without_help"]
    assert isinstance(no_help, list)
    errors = tables["error_outcomes"]
    assert isinstance(errors, list)
    adaptive_errors = next(
        row
        for row in errors
        if isinstance(row, Mapping) and row.get("mode") == EvaluationMode.ADAPTIVE.value
    )
    assert isinstance(adaptive_errors, Mapping)
    return (
        {
            "claim": (
                "Adaptive recall is "
                f"{adaptive['recall']}, compared with fixed-bundle recall "
                f"{fixed['recall']}; mean observations are "
                f"{adaptive['mean_observations']} versus {fixed['mean_observations']}, "
                "respectively."
            ),
            "table": "mode_efficiency",
            "trace_case_keys": traces,
        },
        {
            "claim": (
                f"{len(no_help)} adaptive cases used more observations than one-shot "
                "without correcting the one-shot result."
            ),
            "table": "extra_observations_without_help",
            "trace_case_keys": traces,
        },
        {
            "claim": (
                f"Adaptive produced {adaptive_errors['false_positive_count']} false "
                "positives, "
                f"{adaptive_errors['decisive_false_negative_count']} decisive false "
                "negatives, and "
                f"{adaptive_errors['abstention_count']} abstentions."
            ),
            "table": "error_outcomes",
            "trace_case_keys": traces,
        },
    )


def _selected_trace_keys(representatives: Mapping[str, object]) -> list[str]:
    keys: list[str] = []
    for value in representatives.values():
        if isinstance(value, Mapping) and value.get("status") == "selected":
            key = value.get("case_key")
            if isinstance(key, str):
                keys.append(key)
    return list(dict.fromkeys(keys))


def _complete_trace_keys(index: Mapping[str, Mapping[str, object]]) -> list[str]:
    return [
        key for key, item in sorted(index.items()) if item.get("status") == "complete"
    ]


def _by_key(rows: tuple[_CaseResult, ...]) -> dict[tuple[str, str], _CaseResult]:
    return {row.key: row for row in rows}


def _case_keys(rows: tuple[_CaseResult, ...]) -> list[str]:
    return [_trace_key(row) for row in sorted(rows, key=lambda row: row.key)]


def _trace_key(row: _CaseResult) -> str:
    return f"{row.split}/{row.case_id}"


def _error_rows(value: object) -> tuple[dict[str, str], ...]:
    if not isinstance(value, list):
        raise ValueError("errors must be a list")
    rows: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("error row must be an object")
        rows.append(_string_mapping(item, "error row"))
    return tuple(rows)


def _string_tuple(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be a string list")
    return tuple(cast(list[str], value))


def _string_mapping(value: object, field: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    ):
        raise ValueError(f"{field} must be a string mapping")
    return {
        key: item
        for key, item in value.items()
        if isinstance(key, str) and isinstance(item, str)
    }


def _required_choice(
    value: Mapping[str, object], field: str, choices: tuple[str, ...]
) -> str:
    item = value.get(field)
    if item not in choices:
        raise ValueError(f"{field} is invalid")
    return item


def _required_text(value: Mapping[str, object], field: str) -> str:
    item = value.get(field)
    if not isinstance(item, str) or not item:
        raise ValueError(f"{field} must be non-empty text")
    return item


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    return float(value)


def _non_negative_number(value: object, field: str) -> float:
    number = _number(value, field)
    if number < 0.0:
        raise ValueError(f"{field} must be non-negative")
    return number


def _non_negative_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _read_bytes(path: Path, description: str) -> bytes:
    try:
        return Path(path).read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read {description}: {path}") from error


def _object(raw: bytes, description: str) -> dict[str, object]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid {description} JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be an object")
    return value


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _mean(values: list[int | float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def main(argv: list[str] | None = None) -> int:
    """Freeze error analysis from one already sealed Day 25 report."""

    settings = load_settings()
    default_root = settings.root_dir / "evaluation-data" / "frozen-results"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-report",
        type=Path,
        default=default_root / "frozen-evaluation.json",
    )
    parser.add_argument(
        "--output", type=Path, default=default_root / "error-analysis.json"
    )
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args(argv)
    evaluation_root = (settings.root_dir / "evaluation-data").resolve()
    for option, path in {
        "--evaluation-report": arguments.evaluation_report,
        "--output": arguments.output,
    }.items():
        if not Path(path).resolve().is_relative_to(evaluation_root):
            parser.error(f"{option} must remain inside {evaluation_root}")
    analysis = analyze_frozen_evaluation(arguments.evaluation_report)
    output = write_error_analysis(
        analysis, arguments.output, overwrite=arguments.overwrite
    )
    verify_error_analysis(output, arguments.evaluation_report)
    print(
        json.dumps(
            {
                "output": str(output),
                "status": "sealed",
                "headline_claim_count": len(analysis.headline_claims),
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "ERROR_ANALYSIS_RECORD_TYPE",
    "FrozenErrorAnalysis",
    "analyze_frozen_evaluation",
    "main",
    "verify_error_analysis",
    "write_error_analysis",
]


if __name__ == "__main__":
    raise SystemExit(main())
