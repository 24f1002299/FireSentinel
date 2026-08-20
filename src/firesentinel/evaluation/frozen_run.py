"""Run and freeze the sealed test/stress evaluation comparison.

This is deliberately separate from the development baseline runner.  It first
validates the complete frozen benchmark set, then reads scoring-only labels in
this evaluation module only.  The perception and outcome code remains exactly
the same as the development comparison; the adaptive arm uses the bounded
agent loop and its persisted traces.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import cast
from urllib.parse import quote

from firesentinel.agent.loop import BoundedAgentLoop
from firesentinel.agent.tools import AllowedObservation, ToolManifest, ToolSource
from firesentinel.config import load_settings
from firesentinel.core.records import (
    ActionType,
    Channel,
    Coordinates,
    ManifestCase,
    OutcomeState,
    ReasonCode,
)
from firesentinel.data.source_cache import SourceRequest, VerifiedSourceCache
from firesentinel.evaluation.freeze import (
    LABEL_FILENAMES,
    MANIFEST_FILENAMES,
    default_frozen_directory,
    verify_frozen_benchmark,
)
from firesentinel.evaluation.runner import (
    BaselineCase,
    BaselineMode,
    BaselineParameters,
    BaselineSource,
    _calibrated_outcome,
    _classify_error,
    _evidence_job,
    _selection_roles,
)
from firesentinel.vision.engine import (
    EvidenceJob,
    EvidenceJobFailure,
    load_evidence_job,
    run_evidence_job,
)

FROZEN_EVALUATION_SCHEMA_VERSION = 1
FROZEN_EVALUATION_RECORD_TYPE = "firesentinel_frozen_evaluation"
DEFAULT_BOOTSTRAP_SAMPLES = 1_000
DEFAULT_BOOTSTRAP_SEED = 20260825
_SCORING_SPLITS = ("test", "stress")
_LABELS = ("positive", "control")


class EvaluationMode(StrEnum):
    """The three frozen, comparable observation modes."""

    ONE_SHOT = "one_shot"
    FIXED_BUNDLE = "fixed_bundle"
    ADAPTIVE = "adaptive"


@dataclass(frozen=True, slots=True)
class FrozenCase:
    """One opaque manifest case plus its scoring-only class label."""

    case: BaselineCase
    split: str
    label: str

    def __post_init__(self) -> None:
        if self.split not in _SCORING_SPLITS:
            raise ValueError("frozen evaluation split must be test or stress")
        if self.label not in _LABELS:
            raise ValueError("frozen evaluation label must be positive or control")


@dataclass(frozen=True, slots=True)
class FrozenEvaluationParameters:
    """Pinned configuration for all three modes in a frozen run."""

    baseline: BaselineParameters
    maximum_observations: int = 3
    maximum_elapsed_seconds: float = 120.0
    maximum_retries: int = 1
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED

    def __post_init__(self) -> None:
        if not isinstance(self.baseline, BaselineParameters):
            raise TypeError("baseline must be BaselineParameters")
        if not 1 <= self.maximum_observations <= 3:
            raise ValueError("maximum_observations must be within [1, 3]")
        if not _positive_finite(self.maximum_elapsed_seconds):
            raise ValueError("maximum_elapsed_seconds must be finite and positive")
        if not isinstance(self.maximum_retries, int) or self.maximum_retries < 0:
            raise ValueError("maximum_retries must be a non-negative integer")
        if not isinstance(self.bootstrap_samples, int) or self.bootstrap_samples < 1:
            raise ValueError("bootstrap_samples must be a positive integer")
        if isinstance(self.bootstrap_seed, bool) or not isinstance(
            self.bootstrap_seed, int
        ):
            raise ValueError("bootstrap_seed must be an integer")

    def configuration_dict(self) -> dict[str, object]:
        return {
            "shared_evidence": self.baseline.configuration_dict(),
            "adaptive_limits": {
                "maximum_observations": self.maximum_observations,
                "maximum_elapsed_seconds": self.maximum_elapsed_seconds,
                "maximum_retries": self.maximum_retries,
                "maximum_bytes": "sum of each case's distinct allowlisted sources",
            },
            "bootstrap": {
                "samples": self.bootstrap_samples,
                "seed": self.bootstrap_seed,
                "confidence_level": 0.95,
            },
            "scoring": {
                "classes": list(_LABELS),
                "review_escalation_prediction": "positive",
                "no_persistent_evidence_prediction": "control",
                "human_review_insufficient_and_failed": "abstain",
                "abstentions_count_as_false_negatives_for_the_true_class": True,
                "ambiguous_case_definition": "one_shot abstention",
                "resolved_ambiguity_definition": (
                    "the evaluated mode returns positive or control for that same case"
                ),
            },
        }


@dataclass(frozen=True, slots=True)
class FrozenCaseResult:
    """Frozen scored result for one mode/case pair, including resources."""

    split: str
    case_id: str
    label: str
    mode: EvaluationMode
    outcome_state: OutcomeState
    reason_codes: tuple[ReasonCode, ...]
    confidence: float
    prediction: str | None
    observation_count: int
    channel7_observation_count: int
    evidence_time_step_count: int
    selected_source_bytes: int
    downloaded_bytes: int
    latency_milliseconds: float
    errors: tuple[dict[str, str], ...]
    evidence_ids: tuple[str, ...]
    trace_path: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "split": self.split,
            "case_id": self.case_id,
            "label": self.label,
            "mode": self.mode.value,
            "outcome": {
                "state": self.outcome_state.value,
                "reason_codes": [reason.value for reason in self.reason_codes],
                "confidence": self.confidence,
            },
            "prediction": self.prediction,
            "observation_count": self.observation_count,
            "channel7_observation_count": self.channel7_observation_count,
            "evidence_time_step_count": self.evidence_time_step_count,
            "resources": {
                "selected_source_bytes": self.selected_source_bytes,
                "downloaded_bytes": self.downloaded_bytes,
                "latency_milliseconds": self.latency_milliseconds,
            },
            "errors": list(self.errors),
            "evidence_ids": list(self.evidence_ids),
            "trace_path": self.trace_path,
        }


@dataclass(frozen=True, slots=True)
class FrozenEvaluationResult:
    """Complete sealed report, written before any error-analysis workflow."""

    input_hashes: dict[str, str]
    configuration: dict[str, object]
    cases: tuple[FrozenCaseResult, ...]
    aggregates: tuple[dict[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": FROZEN_EVALUATION_SCHEMA_VERSION,
            "record_type": FROZEN_EVALUATION_RECORD_TYPE,
            "analysis_status": "frozen_before_error_analysis",
            "input_hashes": dict(sorted(self.input_hashes.items())),
            "configuration": self.configuration,
            "per_case_results": [item.to_dict() for item in self.cases],
            "aggregate_tables": list(self.aggregates),
        }


SourceResolver = Callable[[str, BaselineSource], Path]


def run_frozen_evaluation(
    frozen_directory: Path,
    artifacts_root: Path,
    parameters: FrozenEvaluationParameters,
    *,
    source_cache_directory: Path | None = None,
    source_resolver: SourceResolver | None = None,
    project_root: Path | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> FrozenEvaluationResult:
    """Run all modes over verified frozen test and stress manifests.

    No fetch is attempted: default source resolution is a verified-cache lookup.
    The optional resolver is intentionally limited to local integration fixtures.
    """

    if not isinstance(parameters, FrozenEvaluationParameters):
        raise TypeError("parameters must be FrozenEvaluationParameters")
    root = Path(frozen_directory)
    verify_frozen_benchmark(root)
    cases, input_hashes = load_frozen_scoring_cases(root)
    input_hashes["evidence_template_configuration_sha256"] = _sha256(
        _canonical_json(
            parameters.baseline.evidence_template.to_dict(include_paths=False)
        )
    )
    input_hashes["implementation_sha256"] = _implementation_sha256()
    resolver = source_resolver
    if resolver is None:
        if source_cache_directory is None:
            raise ValueError(
                "source_cache_directory is required without source_resolver"
            )
        resolver = _cache_resolver(VerifiedSourceCache(source_cache_directory))
    cache_root = (
        Path(source_cache_directory)
        if source_cache_directory is not None
        else _resolver_root(cases, resolver)
    )
    if project_root is None:
        project_root = load_settings().root_dir

    results: list[FrozenCaseResult] = []
    for mode in EvaluationMode:
        for frozen_case in cases:
            if mode is EvaluationMode.ADAPTIVE:
                result = _run_adaptive_case(
                    frozen_case,
                    Path(artifacts_root),
                    parameters,
                    resolver,
                    cache_root=cache_root,
                    project_root=Path(project_root),
                    clock=clock,
                )
            else:
                result = _run_fixed_case(
                    frozen_case,
                    mode,
                    Path(artifacts_root),
                    parameters,
                    resolver,
                    clock=clock,
                )
            results.append(result)
    ordered = tuple(
        sorted(results, key=lambda item: (item.split, item.mode, item.case_id))
    )
    aggregates = _aggregate_tables(ordered, parameters)
    return FrozenEvaluationResult(
        input_hashes=input_hashes,
        configuration=parameters.configuration_dict(),
        cases=ordered,
        aggregates=aggregates,
    )


def load_frozen_scoring_cases(
    frozen_directory: Path,
) -> tuple[tuple[FrozenCase, ...], dict[str, str]]:
    """Load opaque cases and scoring labels after the benchmark verifier succeeds."""

    root = Path(frozen_directory)
    loaded: list[FrozenCase] = []
    hashes: dict[str, str] = {}
    for split in _SCORING_SPLITS:
        manifest_path = root / MANIFEST_FILENAMES[split]
        labels_path = root / LABEL_FILENAMES[split]
        manifest_bytes = manifest_path.read_bytes()
        label_bytes = labels_path.read_bytes()
        hashes[manifest_path.name] = _sha256(manifest_bytes)
        hashes[labels_path.name] = _sha256(label_bytes)
        manifest = _json_object(manifest_bytes, manifest_path.name)
        labels = _json_object(label_bytes, labels_path.name)
        raw_cases = manifest.get("cases")
        raw_labels = labels.get("labels")
        if not isinstance(raw_cases, list) or not isinstance(raw_labels, list):
            raise ValueError("verified frozen scoring inputs have invalid case rows")
        labels_by_id: dict[str, str] = {}
        for row in raw_labels:
            if not isinstance(row, Mapping):
                raise ValueError("frozen scoring label must be an object")
            case_id, label = row.get("case_id"), row.get("label")
            if not isinstance(case_id, str) or label not in _LABELS:
                raise ValueError("frozen scoring label is invalid")
            labels_by_id[case_id] = str(label)
        for raw_case in raw_cases:
            case = _baseline_case_from_frozen(raw_case)
            label = labels_by_id.get(case.case_id)
            if label is None:
                raise ValueError("frozen scoring labels do not cover a manifest case")
            loaded.append(FrozenCase(case, split, label))
    return tuple(
        sorted(loaded, key=lambda item: (item.split, item.case.case_id))
    ), hashes


def write_frozen_evaluation_report(
    result: FrozenEvaluationResult, path: Path, *, overwrite: bool = False
) -> Path:
    """Atomically write the immutable report, refusing changed replacements."""

    if not isinstance(result, FrozenEvaluationResult):
        raise TypeError("result must be FrozenEvaluationResult")
    destination = Path(path)
    contents = _canonical_json(result.to_dict()) + b"\n"
    if destination.exists() and destination.read_bytes() != contents and not overwrite:
        raise FileExistsError(
            "refusing to replace frozen evaluation report "
            f"'{destination}'; use --overwrite"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", delete=False, dir=destination.parent, prefix=f".{destination.name}."
    ) as temporary:
        temporary.write(contents)
        temporary_path = Path(temporary.name)
    temporary_path.replace(destination)
    return destination


def verify_frozen_evaluation_report(
    path: Path, frozen_directory: Path, evidence_template: EvidenceJob
) -> None:
    """Confirm a stored report still refers to the exact pinned inputs."""

    payload = _json_object(Path(path).read_bytes(), "frozen evaluation report")
    if payload.get("record_type") != FROZEN_EVALUATION_RECORD_TYPE:
        raise ValueError("frozen evaluation report has an unexpected record_type")
    if payload.get("analysis_status") != "frozen_before_error_analysis":
        raise ValueError("frozen evaluation report was not sealed before analysis")
    input_hashes = payload.get("input_hashes")
    if not isinstance(input_hashes, Mapping):
        raise ValueError("frozen evaluation report lacks input hashes")
    _, expected = load_frozen_scoring_cases(frozen_directory)
    expected["evidence_template_configuration_sha256"] = _sha256(
        _canonical_json(evidence_template.to_dict(include_paths=False))
    )
    expected["implementation_sha256"] = _implementation_sha256()
    if dict(input_hashes) != expected:
        raise ValueError("frozen evaluation report no longer matches pinned inputs")


def _run_fixed_case(
    frozen: FrozenCase,
    mode: EvaluationMode,
    artifacts_root: Path,
    parameters: FrozenEvaluationParameters,
    resolver: SourceResolver,
    *,
    clock: Callable[[], float],
) -> FrozenCaseResult:
    baseline_mode = BaselineMode(mode.value)
    case = frozen.case
    started = clock()
    roles = _selection_roles(baseline_mode)
    observations = tuple(case.sources_by_role[role] for role in roles)
    selected_bytes = sum(item.size_bytes for item in observations)
    try:
        paths = {
            item.source_id: Path(resolver(case.case_id, item)) for item in observations
        }
        evidence = run_evidence_job(
            _evidence_job(case, baseline_mode, paths, parameters.baseline),
            artifacts_root / mode.value,
            timeout_seconds=parameters.maximum_elapsed_seconds,
            clock=clock,
        )
        outcome = _calibrated_outcome(evidence, parameters.baseline.outcome_thresholds)
        errors: tuple[dict[str, str], ...] = ()
        evidence_ids = (evidence.content_hash,)
    except Exception as error:
        classified = _classify_error(error)
        outcome_state = OutcomeState.FAILED
        reasons = (classified.reason_code,)
        confidence = 0.0
        errors = (classified.to_dict(),)
        return FrozenCaseResult(
            frozen.split,
            case.case_id,
            frozen.label,
            mode,
            outcome_state,
            reasons,
            confidence,
            None,
            len(observations),
            sum(item.channel == Channel.C07 for item in observations),
            1 if mode is EvaluationMode.ONE_SHOT else 3,
            selected_bytes,
            0,
            _elapsed_milliseconds(started, clock),
            errors,
            (),
            None,
        )
    return FrozenCaseResult(
        frozen.split,
        case.case_id,
        frozen.label,
        mode,
        outcome.state,
        outcome.reason_codes,
        outcome.confidence,
        _prediction(outcome.state),
        len(observations),
        sum(item.channel == Channel.C07 for item in observations),
        1 if mode is EvaluationMode.ONE_SHOT else 3,
        selected_bytes,
        0,
        _elapsed_milliseconds(started, clock),
        errors,
        evidence_ids,
        None,
    )


def _run_adaptive_case(
    frozen: FrozenCase,
    artifacts_root: Path,
    parameters: FrozenEvaluationParameters,
    resolver: SourceResolver,
    *,
    cache_root: Path,
    project_root: Path,
    clock: Callable[[], float],
) -> FrozenCaseResult:
    case = frozen.case
    started = clock()
    try:
        paths = {
            source.source_id: Path(resolver(case.case_id, source))
            for source in case.sources
        }
        manifest = _tool_manifest(case, paths, parameters.baseline)
        maximum_bytes = sum(
            source.size_bytes
            for source in {source.source_id: source for source in case.sources}.values()
        )
        trace_path = (
            artifacts_root
            / "adaptive-traces"
            / frozen.split
            / case.case_id
            / "agent-loop.jsonl"
        )
        loop = BoundedAgentLoop.open(
            manifest,
            source_cache_root=cache_root,
            artifacts_root=artifacts_root / "adaptive",
            project_root=project_root,
            trace_path=trace_path,
            maximum_bytes=maximum_bytes,
            maximum_elapsed_seconds=parameters.maximum_elapsed_seconds,
            maximum_observations=parameters.maximum_observations,
            maximum_retries=parameters.maximum_retries,
            clock=clock,
        )
        loop_result = loop.run()
        outcome = loop_result.outcome
        if outcome is None:
            raise EvidenceJobFailure(
                ReasonCode.CONFIGURATION_INVALID,
                "adaptive loop ended without an outcome",
            )
        selected = tuple(
            manifest.observations_by_id[item] for item in _selected_ids(trace_path)
        )
        source_by_id = {
            source.source_id: source
            for item in selected
            for source in (item.channel7, item.channel14)
        }
        return FrozenCaseResult(
            frozen.split,
            case.case_id,
            frozen.label,
            EvaluationMode.ADAPTIVE,
            outcome.state,
            outcome.reason_codes,
            outcome.confidence,
            _prediction(outcome.state),
            loop_result.budget.used_observations,
            sum(item.action_type is not ActionType.ALTERNATE_BAND for item in selected),
            loop_result.budget.used_observations,
            sum(item.size_bytes for item in source_by_id.values()),
            0,
            _elapsed_milliseconds(started, clock),
            (),
            loop_result.evidence_ids,
            str(trace_path),
        )
    except Exception as error:
        classified = _classify_error(error)
        return FrozenCaseResult(
            frozen.split,
            case.case_id,
            frozen.label,
            EvaluationMode.ADAPTIVE,
            OutcomeState.FAILED,
            (classified.reason_code,),
            0.0,
            None,
            0,
            0,
            0,
            0,
            0,
            _elapsed_milliseconds(started, clock),
            (classified.to_dict(),),
            (),
            None,
        )


def _tool_manifest(
    case: BaselineCase, paths: Mapping[str, Path], parameters: BaselineParameters
) -> ToolManifest:
    sources = case.sources_by_role
    alternate = sources["alternate"]

    def tool_source(source: BaselineSource) -> ToolSource:
        return ToolSource(
            source.source_id,
            source.catalog_key,
            paths[source.source_id],
            source.size_bytes,
            source.sha256,
        )

    actions = {
        "initial": ActionType.NEXT_TIMESTAMP,
        "later": ActionType.NEXT_TIMESTAMP,
        "baseline": ActionType.PRE_EVENT_BASELINE,
        "alternate": ActionType.ALTERNATE_BAND,
    }
    definitions = tuple(
        AllowedObservation(
            role,
            actions[role],
            _utc_timestamp(sources[role].observation_time_utc),
            Channel.C14 if actions[role] is ActionType.ALTERNATE_BAND else Channel.C07,
            tool_source(sources["initial"] if role == "alternate" else sources[role]),
            tool_source(alternate),
        )
        for role in ("initial", "later", "baseline", "alternate")
    )
    return ToolManifest(
        ManifestCase(
            case.case_id,
            "Frozen evaluation case",
            Coordinates(case.latitude, case.longitude),
            min(item.observation_time for item in definitions),
            _sha256(case.case_id.encode("utf-8")),
            tuple(item.observation_id for item in definitions),
        ),
        _template_for_case(case, parameters),
        definitions,
    )


def _template_for_case(
    case: BaselineCase, parameters: BaselineParameters
) -> EvidenceJob:
    """Give the loop a crop centred on the opaque frozen-case anchor."""

    template = parameters.evidence_template
    # The supplied template is used for every numerical parameter; only its crop
    # centre changes to the frozen case, as it does for the baseline arms.
    from firesentinel.evaluation.runner import _crop_parameters

    return EvidenceJob(
        case_id=case.case_id,
        crop_parameters=_crop_parameters(case, parameters),
        tile_parameters=template.tile_parameters,
        observations=template.observations,
        allow_single_observation=template.allow_single_observation,
        anomaly_parameters=template.anomaly_parameters,
        quality_thresholds=template.quality_thresholds,
        persistence_parameters=template.persistence_parameters,
    )


def _selected_ids(trace_path: Path) -> tuple[str, ...]:
    """Read the last complete journal snapshot, never a partial trace line."""

    from firesentinel.agent.loop import load_last_complete_transition

    checkpoint = load_last_complete_transition(trace_path)
    if checkpoint is None:
        return ()
    raw = checkpoint.get("selected_observation_ids")
    return (
        tuple(raw)
        if isinstance(raw, list) and all(isinstance(item, str) for item in raw)
        else ()
    )


def _aggregate_tables(
    results: tuple[FrozenCaseResult, ...], parameters: FrozenEvaluationParameters
) -> tuple[dict[str, object], ...]:
    tables: list[dict[str, object]] = []
    for split in (*_SCORING_SPLITS, "combined"):
        split_rows = tuple(
            item for item in results if split == "combined" or item.split == split
        )
        for mode in EvaluationMode:
            rows = tuple(item for item in split_rows if item.mode is mode)
            tables.append(_aggregate(rows, split, mode, results, parameters))
    return tuple(tables)


def _aggregate(
    rows: tuple[FrozenCaseResult, ...],
    split: str,
    mode: EvaluationMode,
    all_results: tuple[FrozenCaseResult, ...],
    parameters: FrozenEvaluationParameters,
) -> dict[str, object]:
    metrics = _classification_metrics(rows)
    one_shot = {
        (row.split, row.case_id): row
        for row in all_results
        if row.mode is EvaluationMode.ONE_SHOT
    }
    ambiguous = tuple(
        row for row in rows if one_shot[(row.split, row.case_id)].prediction is None
    )
    resolved = sum(row.prediction is not None for row in ambiguous)
    samples = _bootstrap(
        rows, parameters.bootstrap_samples, parameters.bootstrap_seed, split, mode
    )
    return {
        "split": split,
        "mode": mode.value,
        "case_count": len(rows),
        "classification": metrics,
        "bootstrap_95_percent_intervals": {
            name: _interval([item[name] for item in samples])
            for name in ("macro_f1", "macro_precision", "macro_recall")
        },
        "ambiguous_case_resolution": {
            "one_shot_ambiguous_case_count": len(ambiguous),
            "resolved_case_count": resolved,
            "resolution_rate": _ratio(resolved, len(ambiguous)),
        },
        "resources": {
            "observations": sum(row.observation_count for row in rows),
            "mean_observations": _mean([float(row.observation_count) for row in rows]),
            "selected_source_bytes": sum(row.selected_source_bytes for row in rows),
            "mean_selected_source_bytes": _mean(
                [float(row.selected_source_bytes) for row in rows]
            ),
            "downloaded_bytes": sum(row.downloaded_bytes for row in rows),
            "mean_downloaded_bytes": _mean(
                [float(row.downloaded_bytes) for row in rows]
            ),
            "latency_milliseconds": round(
                sum(row.latency_milliseconds for row in rows), 6
            ),
            "mean_latency_milliseconds": _mean(
                [row.latency_milliseconds for row in rows]
            ),
        },
        "error_case_count": sum(bool(row.errors) for row in rows),
    }


def _classification_metrics(rows: tuple[FrozenCaseResult, ...]) -> dict[str, object]:
    per_class: dict[str, dict[str, float | int]] = {}
    for label in _LABELS:
        true_positive = sum(
            row.label == label and row.prediction == label for row in rows
        )
        false_positive = sum(
            row.label != label and row.prediction == label for row in rows
        )
        false_negative = sum(
            row.label == label and row.prediction != label for row in rows
        )
        precision = _ratio(true_positive, true_positive + false_positive)
        recall = _ratio(true_positive, true_positive + false_negative)
        per_class[label] = {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": precision,
            "recall": recall,
            "f1": _ratio(2.0 * precision * recall, precision + recall),
        }
    return {
        "macro_f1": _mean([float(per_class[label]["f1"]) for label in _LABELS]),
        "macro_precision": _mean(
            [float(per_class[label]["precision"]) for label in _LABELS]
        ),
        "macro_recall": _mean([float(per_class[label]["recall"]) for label in _LABELS]),
        "abstention_coverage": _ratio(
            sum(row.prediction is None for row in rows), len(rows)
        ),
        "abstention_count": sum(row.prediction is None for row in rows),
        "per_class": per_class,
    }


def _bootstrap(
    rows: tuple[FrozenCaseResult, ...],
    samples: int,
    seed: int,
    split: str,
    mode: EvaluationMode,
) -> tuple[dict[str, float], ...]:
    if not rows:
        return ()
    seeded = random.Random(f"{seed}:{split}:{mode.value}")
    values: list[dict[str, float]] = []
    for _ in range(samples):
        drawn = tuple(rows[seeded.randrange(len(rows))] for _ in rows)
        metrics = _classification_metrics(drawn)
        values.append(
            {
                "macro_f1": cast(float, metrics["macro_f1"]),
                "macro_precision": cast(float, metrics["macro_precision"]),
                "macro_recall": cast(float, metrics["macro_recall"]),
            }
        )
    return tuple(values)


def _interval(values: list[float]) -> dict[str, float]:
    if not values:
        return {"lower": 0.0, "upper": 0.0}
    ordered = sorted(values)
    return {
        "lower": round(ordered[math.floor(0.025 * (len(ordered) - 1))], 6),
        "upper": round(ordered[math.ceil(0.975 * (len(ordered) - 1))], 6),
    }


def _baseline_case_from_frozen(value: object) -> BaselineCase:
    if not isinstance(value, Mapping):
        raise ValueError("frozen manifest case must be an object")
    case_id, anchor, observations = (
        value.get("case_id"),
        value.get("anchor"),
        value.get("observations"),
    )
    if (
        not isinstance(case_id, str)
        or not isinstance(anchor, Mapping)
        or not isinstance(observations, list)
    ):
        raise ValueError("frozen manifest case is invalid")
    latitude, longitude = anchor.get("latitude"), anchor.get("longitude")
    if (
        isinstance(latitude, bool)
        or not isinstance(latitude, (int, float))
        or isinstance(longitude, bool)
        or not isinstance(longitude, (int, float))
    ):
        raise ValueError("frozen manifest anchor is invalid")
    sources = tuple(_baseline_source_from_frozen(item) for item in observations)
    return BaselineCase(case_id, float(latitude), float(longitude), sources)


def _baseline_source_from_frozen(value: object) -> BaselineSource:
    if not isinstance(value, Mapping) or not isinstance(value.get("source"), Mapping):
        raise ValueError("frozen manifest observation is invalid")
    source = value["source"]
    assert isinstance(source, Mapping)
    try:
        return BaselineSource(
            role=str(value["role"]),
            channel=Channel(str(value["channel"])),
            observation_time_utc=str(value["observation_time_utc"]),
            source_id=str(source["source_id"]),
            bucket=str(source["bucket"]),
            object_key=str(source["object_key"]),
            size_bytes=int(source["size_bytes"]),
            sha256=str(source["sha256"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("frozen manifest source is invalid") from error


def _cache_resolver(cache: VerifiedSourceCache) -> SourceResolver:
    def resolve(case_id: str, source: BaselineSource) -> Path:
        return cache.require_cached(
            SourceRequest(
                case_id=case_id,
                source_id=source.source_id,
                source_url=(
                    f"https://{source.bucket}.s3.amazonaws.com/"
                    f"{quote(source.object_key, safe='/')}"
                ),
                source_size_bytes=source.size_bytes,
                expected_sha256=source.sha256,
            )
        )

    return resolve


def _resolver_root(cases: tuple[FrozenCase, ...], resolver: SourceResolver) -> Path:
    """Infer the fixture cache root only for the explicitly injected resolver."""

    first = cases[0].case.sources[0]
    return Path(resolver(cases[0].case.case_id, first)).resolve().parent


def _prediction(outcome: OutcomeState) -> str | None:
    if outcome is OutcomeState.REVIEW_ESCALATION:
        return "positive"
    if outcome is OutcomeState.NO_PERSISTENT_EVIDENCE:
        return "control"
    return None


def _utc_timestamp(value: str) -> datetime:
    if not value.endswith("Z"):
        raise ValueError("frozen observation time must be UTC")
    return datetime.fromisoformat(f"{value[:-1]}+00:00").astimezone(UTC)


def _json_object(raw: bytes, description: str) -> dict[str, object]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid {description} JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be an object")
    return value


def _positive_finite(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _ratio(numerator: float | int, denominator: float | int) -> float:
    return round(float(numerator) / float(denominator), 6) if denominator else 0.0


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def _elapsed_milliseconds(started: float, clock: Callable[[], float]) -> float:
    return round(max(0.0, (clock() - started) * 1000.0), 6)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _implementation_sha256() -> str:
    """Pin the local evaluator, policy, outcome, and evidence implementation."""

    root = Path(__file__).resolve().parents[1]
    parts = (
        root / "evaluation" / "frozen_run.py",
        root / "evaluation" / "runner.py",
        root / "agent" / "loop.py",
        root / "agent" / "policy.py",
        root / "agent" / "outcomes.py",
        root / "agent" / "tools.py",
        root / "vision" / "engine.py",
        root / "vision" / "persistence.py",
    )
    digest = hashlib.sha256()
    for path in parts:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def main(argv: list[str] | None = None) -> int:
    """Run once, then reuse the sealed report on repeat documented invocations."""

    settings = load_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-template", type=Path, required=True)
    parser.add_argument("--frozen-dir", type=Path, default=default_frozen_directory())
    parser.add_argument("--source-cache", type=Path, default=settings.source_cache_dir)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=settings.artifacts_dir / "frozen-evaluation",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.root_dir
        / "evaluation-data"
        / "frozen-results"
        / "frozen-evaluation.json",
    )
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--maximum-observations", type=int, default=3)
    parser.add_argument("--maximum-retries", type=int, default=1)
    parser.add_argument(
        "--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES
    )
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="run again instead of reusing the sealed matching report",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace a changed report; requires --rerun",
    )
    arguments = parser.parse_args(argv)
    evaluation_root = (settings.root_dir / "evaluation-data").resolve()
    for option, path in {
        "--frozen-dir": arguments.frozen_dir,
        "--output": arguments.output,
    }.items():
        if not Path(path).resolve().is_relative_to(evaluation_root):
            parser.error(f"{option} must remain inside {evaluation_root}")
    if arguments.overwrite and not arguments.rerun:
        parser.error("--overwrite requires --rerun")
    template = load_evidence_job(arguments.evidence_template)
    if arguments.output.exists() and not arguments.rerun:
        verify_frozen_benchmark(arguments.frozen_dir)
        verify_frozen_evaluation_report(
            arguments.output, arguments.frozen_dir, template
        )
        print(
            json.dumps(
                {"output": str(arguments.output), "status": "reused_frozen_report"},
                sort_keys=True,
            )
        )
        return 0
    parameters = FrozenEvaluationParameters(
        BaselineParameters(template),
        maximum_observations=arguments.maximum_observations,
        maximum_elapsed_seconds=arguments.timeout_seconds,
        maximum_retries=arguments.maximum_retries,
        bootstrap_samples=arguments.bootstrap_samples,
        bootstrap_seed=arguments.bootstrap_seed,
    )
    result = run_frozen_evaluation(
        arguments.frozen_dir,
        arguments.artifacts_dir,
        parameters,
        source_cache_directory=arguments.source_cache,
        project_root=settings.root_dir,
    )
    output = write_frozen_evaluation_report(
        result, arguments.output, overwrite=arguments.overwrite
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "status": "sealed",
                "aggregate_table_count": len(result.aggregates),
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "DEFAULT_BOOTSTRAP_SAMPLES",
    "DEFAULT_BOOTSTRAP_SEED",
    "EvaluationMode",
    "FROZEN_EVALUATION_RECORD_TYPE",
    "FrozenCase",
    "FrozenCaseResult",
    "FrozenEvaluationParameters",
    "FrozenEvaluationResult",
    "load_frozen_scoring_cases",
    "main",
    "run_frozen_evaluation",
    "verify_frozen_evaluation_report",
    "write_frozen_evaluation_report",
]


if __name__ == "__main__":
    raise SystemExit(main())
