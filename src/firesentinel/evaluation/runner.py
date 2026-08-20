"""Run fair one-shot and fixed-bundle evidence baselines on development cases.

The modes differ only in their declared observation selection.  They share the
same verified local cache, crop policy, Day 14--17 evidence implementation,
thresholds, and cautious outcome function.  This module is deliberately
development-manifest-only: it never reads scoring-only labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from urllib.parse import quote

from firesentinel.agent.outcomes import (
    DEVELOPMENT_OUTCOME_THRESHOLDS,
    CalibratedOutcome,
    OutcomeEvidence,
    OutcomeThresholds,
    calibrate_outcome,
)
from firesentinel.config import load_settings
from firesentinel.core.records import Channel, OutcomeState, ReasonCode
from firesentinel.data.goes_crop import CropParameters, GeographicBounds
from firesentinel.data.source_cache import (
    SourceCacheCorruptionError,
    SourceCacheError,
    SourceRequest,
    VerifiedSourceCache,
)
from firesentinel.evaluation.tuning import tuning_manifest_path
from firesentinel.vision.engine import (
    EvidenceJob,
    EvidenceJobFailure,
    EvidenceJobObservation,
    EvidenceJobResult,
    EvidenceJobSource,
    run_evidence_job,
)

BASELINE_SCHEMA_VERSION = 2
ONE_SHOT_ROLES = ("initial", "alternate")
FIXED_BUNDLE_ROLES = ("baseline", "initial", "later", "alternate")
_CHANNEL7_ROLES = ("baseline", "initial", "later")
_REQUIRED_ROLES = frozenset(FIXED_BUNDLE_ROLES)
_ROLE_CHANNELS = {
    "baseline": Channel.C07,
    "initial": Channel.C07,
    "later": Channel.C07,
    "alternate": Channel.C14,
}
_IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9_-]{0,79}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class BaselineMode(StrEnum):
    """The fixed selections that make the baseline comparison fair."""

    ONE_SHOT = "one_shot"
    FIXED_BUNDLE = "fixed_bundle"


@dataclass(frozen=True, slots=True)
class BaselineSource:
    """One hash-pinned benchmark source and its prescribed role."""

    role: str
    channel: Channel
    observation_time_utc: str
    source_id: str
    bucket: str
    object_key: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if self.role not in _REQUIRED_ROLES:
            raise ValueError("baseline source role is unsupported")
        if self.channel != _ROLE_CHANNELS[self.role]:
            raise ValueError("baseline source channel does not match its role")
        if not isinstance(
            self.observation_time_utc, str
        ) or not self.observation_time_utc.endswith("Z"):
            raise ValueError("baseline source observation_time_utc must be UTC")
        if not isinstance(self.source_id, str) or not _IDENTIFIER.fullmatch(
            self.source_id
        ):
            raise ValueError("baseline source_id must be a lowercase identifier")
        if not isinstance(self.bucket, str) or not self.bucket:
            raise ValueError("baseline source bucket must be non-empty")
        if not isinstance(self.object_key, str) or not self.object_key:
            raise ValueError("baseline source object_key must be non-empty")
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int):
            raise ValueError("baseline source size_bytes must be an integer")
        if self.size_bytes <= 0:
            raise ValueError("baseline source size_bytes must be positive")
        if not isinstance(self.sha256, str) or not _SHA256.fullmatch(self.sha256):
            raise ValueError("baseline source sha256 must be a lowercase digest")

    @property
    def catalog_key(self) -> str:
        """Return the immutable catalog identity saved in evidence provenance."""

        return f"s3://{self.bucket}/{self.object_key}#{self.sha256}"

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "channel": self.channel.value,
            "observation_time_utc": self.observation_time_utc,
            "source_id": self.source_id,
            "bucket": self.bucket,
            "object_key": self.object_key,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class BaselineCase:
    """The label-free inputs from one frozen development-manifest case."""

    case_id: str
    latitude: float
    longitude: float
    sources: tuple[BaselineSource, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not _IDENTIFIER.fullmatch(self.case_id):
            raise ValueError("baseline case_id must be a lowercase identifier")
        for field, value, minimum, maximum in (
            ("latitude", self.latitude, -90.0, 90.0),
            ("longitude", self.longitude, -180.0, 180.0),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"baseline {field} must be numeric")
            if (
                not math.isfinite(float(value))
                or not minimum <= float(value) <= maximum
            ):
                raise ValueError(f"baseline {field} is outside its WGS84 range")
        roles = tuple(source.role for source in self.sources)
        if len(self.sources) != len(_REQUIRED_ROLES) or set(roles) != _REQUIRED_ROLES:
            raise ValueError("baseline case must include the complete fixed bundle")
        if len(roles) != len(set(roles)):
            raise ValueError("baseline case must not repeat source roles")

    @property
    def sources_by_role(self) -> dict[str, BaselineSource]:
        return {source.role: source for source in self.sources}


@dataclass(frozen=True, slots=True)
class BaselineParameters:
    """Shared evidence configuration and deterministic crop extent policy."""

    evidence_template: EvidenceJob
    crop_half_height_degrees: float = 0.25
    crop_half_width_degrees: float = 0.25
    outcome_thresholds: OutcomeThresholds = DEVELOPMENT_OUTCOME_THRESHOLDS

    def __post_init__(self) -> None:
        if not isinstance(self.evidence_template, EvidenceJob):
            raise TypeError("evidence_template must be an EvidenceJob")
        if not isinstance(self.outcome_thresholds, OutcomeThresholds):
            raise TypeError("outcome_thresholds must be OutcomeThresholds")
        for field, value in (
            ("crop_half_height_degrees", self.crop_half_height_degrees),
            ("crop_half_width_degrees", self.crop_half_width_degrees),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{field} must be a finite positive number")
            number = float(value)
            if not math.isfinite(number) or not 0.0 < number < 90.0:
                raise ValueError(f"{field} must be within (0, 90)")
            object.__setattr__(self, field, number)

    def configuration_dict(self) -> dict[str, object]:
        """Return the path-free configuration shared by both baseline modes."""

        template = self.evidence_template
        crop = template.crop_parameters
        return {
            "crop_policy": {
                "center": "frozen case anchor latitude/longitude",
                "half_height_degrees": self.crop_half_height_degrees,
                "half_width_degrees": self.crop_half_width_degrees,
                "padding_pixels": crop.padding_pixels,
                "maximum_usable_dqf": crop.maximum_usable_dqf,
                "edge_samples": crop.edge_samples,
            },
            "tile_parameters": template.tile_parameters.to_dict(),
            "anomaly_parameters": template.anomaly_parameters.to_dict(),
            "quality_thresholds": template.quality_thresholds.to_dict(),
            "persistence_parameters": template.persistence_parameters.to_dict(),
            "outcome_thresholds": self.outcome_thresholds.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class BaselineError:
    """A case-local classified error; remaining development cases still run."""

    reason_code: ReasonCode
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {"reason_code": self.reason_code.value, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class BaselineCaseResult:
    """Comparable outcome, resources, evidence, and errors for one mode/case."""

    case_id: str
    mode: BaselineMode
    outcome_state: OutcomeState
    outcome_reason_codes: tuple[ReasonCode, ...]
    outcome_confidence: float
    outcome_explanation: str
    observations: tuple[BaselineSource, ...]
    evidence_time_step_count: int
    evidence: tuple[EvidenceJobResult, ...]
    selected_source_bytes: int
    downloaded_bytes: int
    latency_milliseconds: float
    errors: tuple[BaselineError, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not _IDENTIFIER.fullmatch(self.case_id):
            raise ValueError("baseline result case_id must be a lowercase identifier")
        if not isinstance(self.mode, BaselineMode):
            raise TypeError("mode must be BaselineMode")
        if not isinstance(self.outcome_state, OutcomeState):
            raise TypeError("outcome_state must be OutcomeState")
        if not 0.0 <= self.outcome_confidence <= 1.0:
            raise ValueError("outcome_confidence must be within [0, 1]")
        if (
            not isinstance(self.outcome_explanation, str)
            or not self.outcome_explanation
        ):
            raise ValueError("outcome_explanation must be non-empty")
        if self.selected_source_bytes < 0 or self.downloaded_bytes < 0:
            raise ValueError("baseline byte measurements must be non-negative")
        if self.evidence_time_step_count < 0:
            raise ValueError("baseline evidence_time_step_count must be non-negative")
        if self.latency_milliseconds < 0.0:
            raise ValueError("baseline latency_milliseconds must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "mode": self.mode.value,
            "outcome": {
                "state": self.outcome_state.value,
                "reason_codes": [reason.value for reason in self.outcome_reason_codes],
                "confidence": self.outcome_confidence,
                "explanation": self.outcome_explanation,
            },
            "observations": [source.to_dict() for source in self.observations],
            "observation_count": len(self.observations),
            "channel7_observation_count": sum(
                source.channel == Channel.C07 for source in self.observations
            ),
            "evidence_time_step_count": self.evidence_time_step_count,
            "evidence": [item.to_dict() for item in self.evidence],
            "resources": {
                "selected_source_bytes": self.selected_source_bytes,
                "downloaded_bytes": self.downloaded_bytes,
                "latency_milliseconds": self.latency_milliseconds,
            },
            "errors": [error.to_dict() for error in self.errors],
        }


@dataclass(frozen=True, slots=True)
class BaselineRunResult:
    """Both complete baseline runs over one immutable development manifest."""

    manifest_sha256: str
    configuration: dict[str, object]
    results: tuple[BaselineCaseResult, ...]

    def to_dict(self) -> dict[str, object]:
        by_mode = {
            mode: tuple(result for result in self.results if result.mode == mode)
            for mode in BaselineMode
        }
        return {
            "schema_version": BASELINE_SCHEMA_VERSION,
            "record_type": "development_evidence_baselines",
            "manifest_sha256": self.manifest_sha256,
            "configuration": self.configuration,
            "modes": {
                mode.value: {
                    "selection_roles": list(_selection_roles(mode)),
                    "summary": _summary(by_mode[mode]),
                    "cases": [result.to_dict() for result in by_mode[mode]],
                }
                for mode in BaselineMode
            },
        }


SourceResolver = Callable[[str, BaselineSource], Path]


def load_development_manifest(path: Path) -> tuple[str, tuple[BaselineCase, ...]]:
    """Load only the non-label fields from a frozen development manifest."""

    source = Path(path)
    try:
        raw = source.read_bytes()
        payload = json.loads(raw)
    except OSError as error:
        raise ValueError(f"cannot read development manifest: {source}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid development manifest JSON: {source}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("development manifest must be an object")
    if (
        payload.get("record_type") != "firesentinel_frozen_split_manifest"
        or payload.get("split") != "development"
        or payload.get("frozen") is not True
        or payload.get("labels_visible_to_tuning") is not True
    ):
        raise ValueError("baselines require the frozen development manifest")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("development manifest must contain at least one case")
    cases = tuple(_case_from_dict(item) for item in raw_cases)
    case_ids = tuple(case.case_id for case in cases)
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("development manifest repeats case_id")
    return hashlib.sha256(raw).hexdigest(), tuple(
        sorted(cases, key=lambda case: case.case_id)
    )


def run_development_baselines(
    manifest_path: Path,
    artifacts_root: Path,
    parameters: BaselineParameters,
    *,
    source_cache_directory: Path | None = None,
    source_resolver: SourceResolver | None = None,
    timeout_seconds: float = 120.0,
    clock: Callable[[], float] = time.monotonic,
) -> BaselineRunResult:
    """Run both fixed modes over every development case without downloading.

    Source resolution defaults to :meth:`VerifiedSourceCache.require_cached`;
    missing sources are reported per case and never trigger a network fallback.
    ``source_resolver`` exists for local, deterministic integration fixtures.
    """

    manifest_sha256, cases = load_development_manifest(manifest_path)
    if not isinstance(parameters, BaselineParameters):
        raise TypeError("parameters must be BaselineParameters")
    if isinstance(timeout_seconds, bool) or not isinstance(
        timeout_seconds, (int, float)
    ):
        raise ValueError("timeout_seconds must be a finite positive number")
    if not math.isfinite(float(timeout_seconds)) or timeout_seconds <= 0.0:
        raise ValueError("timeout_seconds must be a finite positive number")
    resolver = source_resolver
    if resolver is None:
        if source_cache_directory is None:
            raise ValueError(
                "source_cache_directory is required without source_resolver"
            )
        resolver = _cache_resolver(VerifiedSourceCache(source_cache_directory))

    results: list[BaselineCaseResult] = []
    for mode in BaselineMode:
        for case in cases:
            results.append(
                _run_case(
                    case,
                    mode,
                    artifacts_root,
                    parameters,
                    resolver,
                    timeout_seconds=float(timeout_seconds),
                    clock=clock,
                )
            )
    return BaselineRunResult(
        manifest_sha256,
        parameters.configuration_dict(),
        tuple(results),
    )


def write_baseline_report(result: BaselineRunResult, path: Path) -> Path:
    """Atomically write a human- and machine-comparable baseline report."""

    if not isinstance(result, BaselineRunResult):
        raise TypeError("result must be BaselineRunResult")
    destination = Path(path)
    contents = _canonical_json(result.to_dict()) + b"\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", delete=False, dir=destination.parent, prefix=f".{destination.name}."
    ) as temporary:
        temporary.write(contents)
        temporary_path = Path(temporary.name)
    temporary_path.replace(destination)
    return destination


def _case_from_dict(value: object) -> BaselineCase:
    if not isinstance(value, Mapping):
        raise ValueError("development case must be an object")
    case_id = value.get("case_id")
    anchor = value.get("anchor")
    raw_observations = value.get("observations")
    if not isinstance(case_id, str) or not isinstance(anchor, Mapping):
        raise ValueError("development case must have case_id and anchor")
    if not isinstance(raw_observations, list):
        raise ValueError("development case observations must be an array")
    latitude = anchor.get("latitude")
    longitude = anchor.get("longitude")
    if isinstance(latitude, bool) or not isinstance(latitude, (int, float)):
        raise ValueError("development case anchor latitude must be numeric")
    if isinstance(longitude, bool) or not isinstance(longitude, (int, float)):
        raise ValueError("development case anchor longitude must be numeric")
    sources = tuple(_source_from_dict(item) for item in raw_observations)
    return BaselineCase(case_id, float(latitude), float(longitude), sources)


def _source_from_dict(value: object) -> BaselineSource:
    if not isinstance(value, Mapping):
        raise ValueError("development observation must be an object")
    source = value.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("development observation must have a source")
    role = value.get("role")
    channel = value.get("channel")
    observation_time = value.get("observation_time_utc")
    if not isinstance(role, str) or not isinstance(channel, str):
        raise ValueError("development observation must have role and channel")
    if not isinstance(observation_time, str):
        raise ValueError("development observation must have observation_time_utc")
    try:
        parsed_channel = Channel(channel)
    except ValueError as error:
        raise ValueError(
            "development observation has an unsupported channel"
        ) from error
    return BaselineSource(
        role=role,
        channel=parsed_channel,
        observation_time_utc=observation_time,
        source_id=_required_text(source, "source_id"),
        bucket=_required_text(source, "bucket"),
        object_key=_required_text(source, "object_key"),
        size_bytes=_required_integer(source, "size_bytes"),
        sha256=_required_text(source, "sha256"),
    )


def _run_case(
    case: BaselineCase,
    mode: BaselineMode,
    artifacts_root: Path,
    parameters: BaselineParameters,
    resolver: SourceResolver,
    *,
    timeout_seconds: float,
    clock: Callable[[], float],
) -> BaselineCaseResult:
    started = clock()
    sources = _selected_sources(case, mode)
    selected_bytes = sum(source.size_bytes for source in sources)
    try:
        paths = {
            source.source_id: Path(resolver(case.case_id, source)) for source in sources
        }
        job = _evidence_job(case, mode, paths, parameters)
        evidence = run_evidence_job(
            job,
            Path(artifacts_root) / mode.value,
            timeout_seconds=timeout_seconds,
            clock=clock,
        )
        outcome = _calibrated_outcome(evidence, parameters.outcome_thresholds)
        errors: tuple[BaselineError, ...] = ()
        evidence_results: tuple[EvidenceJobResult, ...] = (evidence,)
    except Exception as error:
        classified = _classify_error(error)
        outcome = CalibratedOutcome(OutcomeState.FAILED, (classified.reason_code,), 0.0)
        errors = (classified,)
        evidence_results = ()
    return BaselineCaseResult(
        case_id=case.case_id,
        mode=mode,
        outcome_state=outcome.state,
        outcome_reason_codes=outcome.reason_codes,
        outcome_confidence=outcome.confidence,
        outcome_explanation=outcome.explanation,
        observations=sources,
        evidence_time_step_count=(
            1 if mode == BaselineMode.ONE_SHOT else len(_CHANNEL7_ROLES)
        ),
        evidence=evidence_results,
        selected_source_bytes=selected_bytes,
        downloaded_bytes=0,
        latency_milliseconds=max(0.0, (clock() - started) * 1000.0),
        errors=errors,
    )


def _selected_sources(
    case: BaselineCase, mode: BaselineMode
) -> tuple[BaselineSource, ...]:
    sources = case.sources_by_role
    return tuple(sources[role] for role in _selection_roles(mode))


def _selection_roles(mode: BaselineMode) -> tuple[str, ...]:
    if mode == BaselineMode.ONE_SHOT:
        return ONE_SHOT_ROLES
    if mode == BaselineMode.FIXED_BUNDLE:
        return FIXED_BUNDLE_ROLES
    raise AssertionError(f"unsupported baseline mode {mode!r}")


def _evidence_job(
    case: BaselineCase,
    mode: BaselineMode,
    paths: Mapping[str, Path],
    parameters: BaselineParameters,
) -> EvidenceJob:
    sources = case.sources_by_role
    alternate = sources["alternate"]
    observations = tuple(
        EvidenceJobObservation(
            role,
            EvidenceJobSource(
                sources[role].catalog_key, paths[sources[role].source_id]
            ),
            EvidenceJobSource(alternate.catalog_key, paths[alternate.source_id]),
        )
        for role in _CHANNEL7_ROLES
        if role in _selection_roles(mode)
    )
    template = parameters.evidence_template
    return EvidenceJob(
        case_id=case.case_id,
        crop_parameters=_crop_parameters(case, parameters),
        tile_parameters=template.tile_parameters,
        observations=observations,
        allow_single_observation=mode == BaselineMode.ONE_SHOT,
        anomaly_parameters=template.anomaly_parameters,
        quality_thresholds=template.quality_thresholds,
        persistence_parameters=template.persistence_parameters,
    )


def _crop_parameters(
    case: BaselineCase, parameters: BaselineParameters
) -> CropParameters:
    south = case.latitude - parameters.crop_half_height_degrees
    north = case.latitude + parameters.crop_half_height_degrees
    west = case.longitude - parameters.crop_half_width_degrees
    east = case.longitude + parameters.crop_half_width_degrees
    if south < -90.0 or north > 90.0 or west < -180.0 or east > 180.0:
        raise ValueError("baseline crop extent crosses the supported WGS84 boundary")
    template = parameters.evidence_template.crop_parameters
    return CropParameters(
        bounds=GeographicBounds(south, west, north, east),
        padding_pixels=template.padding_pixels,
        maximum_usable_dqf=template.maximum_usable_dqf,
        edge_samples=template.edge_samples,
    )


def _cache_resolver(cache: VerifiedSourceCache) -> SourceResolver:
    def resolve(case_id: str, source: BaselineSource) -> Path:
        request = SourceRequest(
            case_id=case_id,
            source_id=source.source_id,
            source_url=(
                f"https://{source.bucket}.s3.amazonaws.com/"
                f"{quote(source.object_key, safe='/')}"
            ),
            source_size_bytes=source.size_bytes,
            expected_sha256=source.sha256,
        )
        return cache.require_cached(request)

    return resolve


def _calibrated_outcome(
    evidence_result: EvidenceJobResult, thresholds: OutcomeThresholds
) -> CalibratedOutcome:
    """Load completed facts and apply the shared cautious outcome calibrator."""

    try:
        payload = json.loads(
            (evidence_result.artifact_directory / "evidence.json").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceJobFailure(
            ReasonCode.SOURCE_CORRUPT, "completed evidence packet is unreadable"
        ) from error
    if not isinstance(payload, Mapping):
        raise EvidenceJobFailure(
            ReasonCode.SOURCE_CORRUPT, "completed evidence packet has an invalid shape"
        )
    try:
        facts = OutcomeEvidence.from_local_evidence(payload)
    except ValueError as error:
        raise EvidenceJobFailure(ReasonCode.SOURCE_CORRUPT, str(error)) from error
    return calibrate_outcome(facts, thresholds)


def _outcome(
    evidence_result: EvidenceJobResult, job: EvidenceJob
) -> tuple[OutcomeState, tuple[ReasonCode, ...], float]:
    """Compatibility wrapper for callers needing only the former tuple fields."""

    del job
    calibrated = _calibrated_outcome(evidence_result, DEVELOPMENT_OUTCOME_THRESHOLDS)
    return calibrated.state, calibrated.reason_codes, calibrated.confidence


def _classify_error(error: Exception) -> BaselineError:
    if isinstance(error, EvidenceJobFailure):
        return BaselineError(error.reason_code, error.detail)
    if isinstance(error, SourceCacheCorruptionError):
        return BaselineError(ReasonCode.SOURCE_CORRUPT, str(error))
    if isinstance(error, SourceCacheError):
        return BaselineError(ReasonCode.SOURCE_MISSING, str(error))
    if isinstance(error, FileNotFoundError):
        return BaselineError(ReasonCode.SOURCE_MISSING, str(error))
    if isinstance(error, (TypeError, ValueError)):
        return BaselineError(ReasonCode.CONFIGURATION_INVALID, str(error))
    return BaselineError(
        ReasonCode.SOURCE_CORRUPT, f"unexpected {type(error).__name__}: {error}"
    )


def _summary(results: tuple[BaselineCaseResult, ...]) -> dict[str, object]:
    outcomes: dict[str, int] = {}
    for result in results:
        outcomes[result.outcome_state.value] = (
            outcomes.get(result.outcome_state.value, 0) + 1
        )
    return {
        "case_count": len(results),
        "completed_case_count": sum(not result.errors for result in results),
        "failed_case_count": sum(bool(result.errors) for result in results),
        "outcomes": dict(sorted(outcomes.items())),
        "observations": sum(len(result.observations) for result in results),
        "channel7_observations": sum(
            sum(source.channel == Channel.C07 for source in result.observations)
            for result in results
        ),
        "selected_source_bytes": sum(
            result.selected_source_bytes for result in results
        ),
        "downloaded_bytes": sum(result.downloaded_bytes for result in results),
        "latency_milliseconds": sum(result.latency_milliseconds for result in results),
        "error_count": sum(len(result.errors) for result in results),
    }


def _required_text(value: Mapping[str, object], field: str) -> str:
    item = value.get(field)
    if not isinstance(item, str):
        raise ValueError(f"development source {field} must be a string")
    return item


def _required_integer(value: Mapping[str, object], field: str) -> int:
    item = value.get(field)
    if isinstance(item, bool) or not isinstance(item, int):
        raise ValueError(f"development source {field} must be an integer")
    return item


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def main(argv: list[str] | None = None) -> int:
    """Run both deterministic baselines against the only permitted split."""

    settings = load_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=settings.root_dir
        / "evaluation-data"
        / "frozen"
        / "development.manifest.json",
        help="frozen development manifest; test and stress are rejected",
    )
    parser.add_argument(
        "--evidence-template",
        type=Path,
        required=True,
        help="Day 17 evidence job whose thresholds and tile settings are reused",
    )
    parser.add_argument(
        "--source-cache",
        type=Path,
        default=settings.source_cache_dir,
        help="verified local source cache; no download fallback is performed",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=settings.artifacts_dir / "baselines",
        help="content-addressed Day 17 evidence artifact root",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.artifacts_dir / "baseline-report.json",
        help="JSON report containing both comparable baseline results",
    )
    parser.add_argument("--crop-half-height-degrees", type=float, default=0.25)
    parser.add_argument("--crop-half-width-degrees", type=float, default=0.25)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    arguments = parser.parse_args(argv)
    try:
        manifest = tuning_manifest_path(
            arguments.manifest, project_root=settings.root_dir
        )
        template = EvidenceJob.from_dict(
            json.loads(arguments.evidence_template.read_text(encoding="utf-8")),
            base_directory=arguments.evidence_template.parent.resolve(),
        )
        parameters = BaselineParameters(
            template,
            crop_half_height_degrees=arguments.crop_half_height_degrees,
            crop_half_width_degrees=arguments.crop_half_width_degrees,
        )
        result = run_development_baselines(
            manifest,
            arguments.artifacts_dir,
            parameters,
            source_cache_directory=arguments.source_cache,
            timeout_seconds=arguments.timeout_seconds,
        )
        report_path = write_baseline_report(result, arguments.output)
    except (OSError, TypeError, ValueError, EvidenceJobFailure) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "report": str(report_path),
                "modes": [mode.value for mode in BaselineMode],
                "case_count": len(result.results) // len(BaselineMode),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BASELINE_SCHEMA_VERSION",
    "FIXED_BUNDLE_ROLES",
    "ONE_SHOT_ROLES",
    "BaselineCase",
    "BaselineCaseResult",
    "BaselineError",
    "BaselineMode",
    "BaselineParameters",
    "BaselineRunResult",
    "BaselineSource",
    "load_development_manifest",
    "run_development_baselines",
    "write_baseline_report",
]
