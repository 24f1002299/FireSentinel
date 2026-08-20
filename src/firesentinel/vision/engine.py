"""One deterministic, local evidence job from cached sources to artifacts.

The engine deliberately receives already selected local source paths plus their
catalog keys.  It performs no network access: catalog selection is preserved as
provenance, while crop, preparation, quality, anomaly, and persistence stages
are replayed from those immutable local inputs.  All outputs are staged before
an atomic directory rename, so a failure cannot appear as a completed packet.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Self, cast

import cv2
import numpy as np
import numpy.typing as npt

from firesentinel.core.records import (
    ReasonCode,
    artifact_directory,
    canonical_content_hash,
)
from firesentinel.data.goes_crop import (
    CalibratedCrop,
    CropArtifactError,
    CropParameters,
    GoesCropError,
    extract_calibrated_crop,
)
from firesentinel.vision.anomalies import (
    DEVELOPMENT_CONTEXTUAL_ANOMALY_PARAMETERS,
    ContextualAnomalyParameters,
    ContextualAnomalyResult,
    extract_contextual_anomalies,
)
from firesentinel.vision.persistence import (
    DEVELOPMENT_PERSISTENCE_PARAMETERS,
    GeospatialGrid,
    PersistenceParameters,
    TemporalObservation,
    TemporalPersistenceResult,
    measure_temporal_persistence,
)
from firesentinel.vision.quality import (
    DEVELOPMENT_QUALITY_THRESHOLDS,
    ObservationQualityThresholds,
)
from firesentinel.vision.tiles import (
    PreparedTile,
    TilePreparationParameters,
    prepare_calibrated_tile,
)

JOB_SCHEMA_VERSION = 1
EVIDENCE_SCHEMA_VERSION = 1
_STAGING_PREFIX = ".evidence-staging-"
_IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9_-]{0,79}\Z")

FloatArray = npt.NDArray[np.float32]
MaskArray = npt.NDArray[np.bool]


class EvidenceJobFailure(RuntimeError):
    """A classified evidence-job failure that never denotes a completed packet."""

    def __init__(self, reason_code: ReasonCode, detail: str) -> None:
        super().__init__(f"{reason_code.value}: {detail}")
        self.reason_code = ReasonCode(reason_code)
        self.detail = detail


class EvidenceJobTimeout(EvidenceJobFailure):
    """The job consumed its finite wall-clock budget before completing."""

    def __init__(self, timeout_seconds: float) -> None:
        super().__init__(
            ReasonCode.TIMEOUT, f"job exceeded {timeout_seconds:.3f} seconds"
        )


class EvidenceJobCancelled(EvidenceJobFailure):
    """A caller cooperatively cancelled work at a stage boundary."""

    def __init__(self) -> None:
        super().__init__(ReasonCode.CANCELLED, "job cancelled before completion")


@dataclass(frozen=True, slots=True)
class EvidenceJobSource:
    """A locally available source selected by a prior catalog lookup."""

    catalog_key: str
    source_path: Path

    def __post_init__(self) -> None:
        if not isinstance(self.catalog_key, str) or not self.catalog_key:
            raise ValueError("catalog_key must be a non-empty string")
        object.__setattr__(self, "source_path", Path(self.source_path))

    @classmethod
    def from_dict(cls, value: object, *, base_directory: Path) -> Self:
        if not isinstance(value, Mapping) or set(value) != {
            "catalog_key",
            "source_path",
        }:
            raise ValueError("evidence source must contain catalog_key and source_path")
        catalog_key = value["catalog_key"]
        source_path = value["source_path"]
        if not isinstance(catalog_key, str) or not isinstance(source_path, str):
            raise ValueError("evidence source values must be strings")
        path = Path(source_path)
        if not path.is_absolute():
            path = base_directory / path
        return cls(catalog_key, path)

    def to_dict(self, *, include_path: bool) -> dict[str, str]:
        payload = {"catalog_key": self.catalog_key}
        if include_path:
            payload["source_path"] = str(self.source_path)
        return payload


@dataclass(frozen=True, slots=True)
class EvidenceJobObservation:
    """One time step's selected C07 and C14 source objects."""

    observation_id: str
    channel7: EvidenceJobSource
    channel14: EvidenceJobSource

    def __post_init__(self) -> None:
        if not isinstance(self.observation_id, str) or not _IDENTIFIER.fullmatch(
            self.observation_id
        ):
            raise ValueError("observation_id must be a safe lowercase identifier")
        if not isinstance(self.channel7, EvidenceJobSource) or not isinstance(
            self.channel14, EvidenceJobSource
        ):
            raise ValueError("channel7 and channel14 must be EvidenceJobSource")

    @classmethod
    def from_dict(cls, value: object, *, base_directory: Path) -> Self:
        if not isinstance(value, Mapping) or set(value) != {
            "observation_id",
            "channel7",
            "channel14",
        }:
            raise ValueError(
                "observation must contain observation_id, channel7, channel14"
            )
        observation_id = value["observation_id"]
        if not isinstance(observation_id, str):
            raise ValueError("observation_id must be a string")
        return cls(
            observation_id,
            EvidenceJobSource.from_dict(
                value["channel7"], base_directory=base_directory
            ),
            EvidenceJobSource.from_dict(
                value["channel14"], base_directory=base_directory
            ),
        )

    def to_dict(self, *, include_path: bool) -> dict[str, object]:
        return {
            "observation_id": self.observation_id,
            "channel7": self.channel7.to_dict(include_path=include_path),
            "channel14": self.channel14.to_dict(include_path=include_path),
        }


@dataclass(frozen=True, slots=True)
class EvidenceJob:
    """Complete immutable configuration for one deterministic local evidence run."""

    case_id: str
    crop_parameters: CropParameters
    tile_parameters: TilePreparationParameters
    observations: tuple[EvidenceJobObservation, ...]
    anomaly_parameters: ContextualAnomalyParameters = (
        DEVELOPMENT_CONTEXTUAL_ANOMALY_PARAMETERS
    )
    quality_thresholds: ObservationQualityThresholds = DEVELOPMENT_QUALITY_THRESHOLDS
    persistence_parameters: PersistenceParameters = DEVELOPMENT_PERSISTENCE_PARAMETERS

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not _IDENTIFIER.fullmatch(self.case_id):
            raise ValueError("case_id must be a safe lowercase identifier")
        if not isinstance(self.crop_parameters, CropParameters):
            raise ValueError("crop_parameters must be CropParameters")
        if not isinstance(self.tile_parameters, TilePreparationParameters):
            raise ValueError("tile_parameters must be TilePreparationParameters")
        if self.tile_parameters.target_shape is not None:
            raise ValueError(
                "tile_parameters.target_shape must be null so tile pixels retain "
                "crop geolocation"
            )
        if len(self.observations) < 2 or not all(
            isinstance(observation, EvidenceJobObservation)
            for observation in self.observations
        ):
            raise ValueError("an evidence job requires at least two observations")
        identifiers = tuple(
            observation.observation_id for observation in self.observations
        )
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("observation identifiers must be unique")
        if not isinstance(self.anomaly_parameters, ContextualAnomalyParameters):
            raise ValueError("anomaly_parameters must be ContextualAnomalyParameters")
        if not isinstance(self.quality_thresholds, ObservationQualityThresholds):
            raise ValueError("quality_thresholds must be ObservationQualityThresholds")
        if not isinstance(self.persistence_parameters, PersistenceParameters):
            raise ValueError("persistence_parameters must be PersistenceParameters")

    @classmethod
    def from_dict(cls, value: object, *, base_directory: Path) -> Self:
        if not isinstance(value, Mapping):
            raise ValueError("evidence job must be an object")
        required = {
            "schema_version",
            "case_id",
            "crop_parameters",
            "tile_parameters",
            "observations",
        }
        optional = {
            "anomaly_parameters",
            "quality_thresholds",
            "persistence_parameters",
        }
        if not required.issubset(value) or not set(value).issubset(required | optional):
            raise ValueError("evidence job has an invalid shape")
        if value["schema_version"] != JOB_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {JOB_SCHEMA_VERSION}")
        case_id = value["case_id"]
        raw_observations = value["observations"]
        if not isinstance(case_id, str) or not isinstance(raw_observations, list):
            raise ValueError(
                "case_id must be a string and observations must be an array"
            )
        return cls(
            case_id=case_id,
            crop_parameters=CropParameters.from_dict(value["crop_parameters"]),
            tile_parameters=_tile_parameters_from_dict(value["tile_parameters"]),
            observations=tuple(
                EvidenceJobObservation.from_dict(
                    observation, base_directory=base_directory
                )
                for observation in raw_observations
            ),
            anomaly_parameters=_anomaly_parameters_from_dict(
                value.get("anomaly_parameters")
            ),
            quality_thresholds=_quality_thresholds_from_dict(
                value.get("quality_thresholds")
            ),
            persistence_parameters=_persistence_parameters_from_dict(
                value.get("persistence_parameters")
            ),
        )

    def to_dict(self, *, include_paths: bool) -> dict[str, object]:
        """Return a JSON-safe job manifest or path-free hash configuration."""

        return {
            "schema_version": JOB_SCHEMA_VERSION,
            "case_id": self.case_id,
            "crop_parameters": self.crop_parameters.to_dict(),
            "tile_parameters": self.tile_parameters.to_dict(),
            "anomaly_parameters": self.anomaly_parameters.to_dict(),
            "quality_thresholds": self.quality_thresholds.to_dict(),
            "persistence_parameters": self.persistence_parameters.to_dict(),
            "observations": [
                observation.to_dict(include_path=include_paths)
                for observation in self.observations
            ],
        }


@dataclass(frozen=True, slots=True)
class EvidenceJobResult:
    """Stable artifact identifiers returned by a successful local job."""

    case_id: str
    content_hash: str
    artifact_directory: Path
    reused_existing_artifact: bool
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "content_hash": self.content_hash,
            "artifact_directory": str(self.artifact_directory),
            "reused_existing_artifact": self.reused_existing_artifact,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class _ObservationRun:
    definition: EvidenceJobObservation
    channel7_crop: CalibratedCrop
    channel14_crop: CalibratedCrop
    channel7_tile: PreparedTile
    channel14_tile: PreparedTile
    anomaly: ContextualAnomalyResult


def run_evidence_job(
    job: EvidenceJob,
    artifacts_root: Path,
    *,
    timeout_seconds: float = 120.0,
    cancellation_requested: Callable[[], bool] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> EvidenceJobResult:
    """Run every local stage and atomically materialize one evidence packet.

    ``clock`` and ``cancellation_requested`` are injectable solely for
    deterministic failure tests.  Runtime timing measurements are written to
    evidence but deliberately excluded from the content-addressed artifact ID.
    """

    if not isinstance(job, EvidenceJob):
        raise TypeError("job must be EvidenceJob")
    timeout = _positive_number(timeout_seconds, "timeout_seconds")
    started = clock()
    timings: dict[str, float] = {}

    def checkpoint() -> None:
        if cancellation_requested is not None and cancellation_requested():
            raise EvidenceJobCancelled()
        if clock() - started > timeout:
            raise EvidenceJobTimeout(timeout)

    def stage[Value](name: str, operation: Callable[[], Value]) -> Value:
        checkpoint()
        stage_started = clock()
        value = operation()
        checkpoint()
        timings[name] = _milliseconds(clock() - stage_started)
        return value

    try:
        runs: list[_ObservationRun] = []
        for observation in job.observations:
            channel7_crop = stage(
                f"crop:{observation.observation_id}:C07",
                partial(
                    extract_calibrated_crop,
                    observation.channel7.source_path,
                    job.crop_parameters,
                ),
            )
            channel14_crop = stage(
                f"crop:{observation.observation_id}:C14",
                partial(
                    extract_calibrated_crop,
                    observation.channel14.source_path,
                    job.crop_parameters,
                ),
            )
            remapped_channel14, remapped_invalid = stage(
                f"align-bands:{observation.observation_id}",
                partial(
                    _resample_band_to_channel7,
                    channel14_crop,
                    channel7_crop,
                    job.persistence_parameters.maximum_resample_distance_kilometres,
                ),
            )
            channel7_tile = stage(
                f"prepare:{observation.observation_id}:C07",
                partial(
                    prepare_calibrated_tile,
                    channel7_crop.calibrated,
                    channel7_crop.invalid_mask,
                    job.tile_parameters,
                    source_crop_checksum=channel7_crop.content_checksum,
                    source_timing=channel7_crop.timing.to_dict(),
                ),
            )
            channel14_tile = stage(
                f"prepare:{observation.observation_id}:C14",
                partial(
                    prepare_calibrated_tile,
                    remapped_channel14,
                    remapped_invalid,
                    job.tile_parameters,
                    source_crop_checksum=channel14_crop.content_checksum,
                    source_timing=channel14_crop.timing.to_dict(),
                ),
            )
            anomaly = stage(
                f"anomaly:{observation.observation_id}",
                partial(
                    extract_contextual_anomalies,
                    channel7_tile.resized_calibrated,
                    channel14_tile.resized_calibrated,
                    channel7_tile.resized_invalid_mask,
                    channel14_tile.resized_invalid_mask,
                    job.anomaly_parameters,
                    quality_thresholds=job.quality_thresholds,
                ),
            )
            runs.append(
                _ObservationRun(
                    observation,
                    channel7_crop,
                    channel14_crop,
                    channel7_tile,
                    channel14_tile,
                    anomaly,
                )
            )
        persistence = stage(
            "persistence",
            lambda: _measure_persistence(runs, job.persistence_parameters),
        )
        checkpoint()
        files = _artifact_files(runs)
        file_manifest = _file_manifest(files)
        warnings = _warnings(runs)
        stable_evidence = _stable_evidence(
            job, runs, persistence, warnings, file_manifest
        )
        content_hash = canonical_content_hash(stable_evidence)
        evidence = dict(stable_evidence)
        evidence["content_hash"] = content_hash
        evidence["timings_milliseconds"] = dict(sorted(timings.items()))
        evidence_bytes = _canonical_json(evidence) + b"\n"
        checkpoint()
        destination, reused = _write_artifacts(
            artifacts_root,
            job.case_id,
            content_hash,
            evidence_bytes,
            files,
            checkpoint,
        )
        return EvidenceJobResult(
            job.case_id,
            content_hash,
            destination,
            reused,
            warnings,
        )
    except EvidenceJobFailure:
        raise
    except Exception as error:
        raise _classify_failure(error) from error


def load_evidence_job(path: Path) -> EvidenceJob:
    """Load a portable JSON job manifest whose paths are relative to itself."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise EvidenceJobFailure(
            ReasonCode.SOURCE_MISSING, f"job manifest missing: {source}"
        ) from error
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceJobFailure(
            ReasonCode.CONFIGURATION_INVALID, f"could not read job manifest {source}"
        ) from error
    try:
        return EvidenceJob.from_dict(payload, base_directory=source.parent.resolve())
    except (TypeError, ValueError, CropArtifactError) as error:
        raise EvidenceJobFailure(
            ReasonCode.CONFIGURATION_INVALID, str(error)
        ) from error


def _measure_persistence(
    runs: list[_ObservationRun], parameters: PersistenceParameters
) -> TemporalPersistenceResult:
    observations = tuple(
        TemporalObservation(
            run.definition.observation_id,
            run.anomaly.candidate_mask,
            run.channel7_tile.resized_calibrated,
            run.channel7_tile.resized_invalid_mask,
            GeospatialGrid(run.channel7_crop.latitude, run.channel7_crop.longitude),
        )
        for run in runs
    )
    return measure_temporal_persistence(observations, parameters)


def _resample_band_to_channel7(
    channel14: CalibratedCrop,
    channel7: CalibratedCrop,
    maximum_distance_kilometres: float,
) -> tuple[FloatArray, MaskArray]:
    """Nearest-resample C14 calibration to the C07 geospatial crop grid."""

    source_latitude = channel14.latitude.reshape(-1)
    source_longitude = channel14.longitude.reshape(-1)
    source_geolocated = np.isfinite(source_latitude) & np.isfinite(source_longitude)
    target_latitude = channel7.latitude.reshape(-1)
    target_longitude = channel7.longitude.reshape(-1)
    target_geolocated = np.isfinite(target_latitude) & np.isfinite(target_longitude)
    values = np.full(channel7.calibrated.shape, np.nan, dtype=np.float32)
    invalid = np.ones(channel7.calibrated.shape, dtype=bool)
    if not np.any(source_geolocated) or not np.any(target_geolocated):
        return values, invalid
    source_indices = np.flatnonzero(source_geolocated)
    latitude = source_latitude[source_indices]
    longitude = source_longitude[source_indices]
    target_indices = np.flatnonzero(target_geolocated)
    source_valid = (~channel14.invalid_mask).reshape(-1)
    source_values = channel14.calibrated.reshape(-1)
    for start in range(0, len(target_indices), 1_024):
        indices = target_indices[start : start + 1_024]
        distance = _haversine_kilometres(
            target_latitude[indices, None],
            target_longitude[indices, None],
            latitude[None, :],
            longitude[None, :],
        )
        nearest_position = np.argmin(distance, axis=1)
        nearest = source_indices[nearest_position]
        target_valid = source_valid[nearest] & (
            distance[np.arange(len(indices)), nearest_position]
            <= maximum_distance_kilometres
        )
        valid_indices = indices[target_valid]
        values.reshape(-1)[valid_indices] = source_values[nearest[target_valid]]
        invalid.reshape(-1)[valid_indices] = False
    return values, invalid


def _artifact_files(runs: list[_ObservationRun]) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    for run in runs:
        prefix = f"observations/{run.definition.observation_id}"
        anomaly = run.anomaly
        arrays: dict[str, npt.NDArray[np.generic]] = {
            "local-contrast-kelvin.npy": anomaly.local_contrast_kelvin,
            "channel-difference-kelvin.npy": anomaly.channel_difference_kelvin,
            "local-contrast-threshold-mask.npy": anomaly.local_contrast_threshold_mask,
            "channel-difference-threshold-mask.npy": (
                anomaly.channel_difference_threshold_mask
            ),
            "morphology-mask.npy": anomaly.morphology_mask,
            "candidate-mask.npy": anomaly.candidate_mask,
        }
        for filename, array in arrays.items():
            files[f"{prefix}/{filename}"] = _npy_bytes(array)
        files[f"{prefix}/overlay.png"] = _png_bytes(anomaly.overlay)
    return files


def _stable_evidence(
    job: EvidenceJob,
    runs: list[_ObservationRun],
    persistence: TemporalPersistenceResult,
    warnings: tuple[str, ...],
    file_manifest: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "record_type": "local_evidence_job",
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "case_id": job.case_id,
        "configuration": job.to_dict(include_paths=False),
        "observations": [
            {
                "observation_id": run.definition.observation_id,
                "catalog": {
                    "channel7_key": run.definition.channel7.catalog_key,
                    "channel14_key": run.definition.channel14.catalog_key,
                },
                "channel7_crop": run.channel7_crop.metadata(),
                "channel14_crop": run.channel14_crop.metadata(),
                "channel7_tile": _stable_tile_metadata(run.channel7_tile),
                "channel14_tile": _stable_tile_metadata(run.channel14_tile),
                "anomaly": run.anomaly.to_dict(),
            }
            for run in runs
        ],
        "persistence": persistence.to_dict(),
        "warnings": list(warnings),
        "artifacts": file_manifest,
    }


def _stable_tile_metadata(tile: PreparedTile) -> dict[str, object]:
    metadata = tile.metadata()
    metadata.pop("timings_milliseconds", None)
    return metadata


def _warnings(runs: list[_ObservationRun]) -> tuple[str, ...]:
    warnings: list[str] = []
    for run in runs:
        for channel, quality in (
            ("C07", run.anomaly.channel7_quality),
            ("C14", run.anomaly.channel14_quality),
        ):
            for reason in quality.reason_codes:
                warnings.append(
                    f"{run.definition.observation_id}:{channel}:{reason.value}"
                )
    return tuple(sorted(set(warnings)))


def _file_manifest(files: Mapping[str, bytes]) -> list[dict[str, object]]:
    return [
        {
            "filename": filename,
            "sha256": hashlib.sha256(contents).hexdigest(),
            "size_bytes": len(contents),
        }
        for filename, contents in sorted(files.items())
    ]


def _write_artifacts(
    artifacts_root: Path,
    case_id: str,
    content_hash: str,
    evidence_bytes: bytes,
    files: Mapping[str, bytes],
    checkpoint: Callable[[], None],
) -> tuple[Path, bool]:
    destination = artifact_directory(Path(artifacts_root), case_id, content_hash)
    if destination.exists():
        _verify_completed_artifact(destination, content_hash)
        return destination, True
    case_directory = destination.parent
    case_directory.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=_STAGING_PREFIX, dir=case_directory))
    try:
        checkpoint()
        for filename, contents in files.items():
            path = stage / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(contents)
        checkpoint()
        (stage / "evidence.json").write_bytes(evidence_bytes)
        completion = {
            "record_type": "evidence_job_completion",
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "content_hash": content_hash,
            "evidence_sha256": hashlib.sha256(evidence_bytes).hexdigest(),
        }
        (stage / "completion.json").write_bytes(_canonical_json(completion) + b"\n")
        checkpoint()
        try:
            os.replace(stage, destination)
        except OSError:
            if destination.exists():
                _verify_completed_artifact(destination, content_hash)
                return destination, True
            raise
        return destination, False
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)


def _verify_completed_artifact(destination: Path, content_hash: str) -> None:
    try:
        completion = json.loads((destination / "completion.json").read_text("utf-8"))
        evidence = json.loads((destination / "evidence.json").read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceJobFailure(
            ReasonCode.SOURCE_CORRUPT,
            f"existing artifact is incomplete or unreadable: {destination}",
        ) from error
    if (
        not isinstance(completion, Mapping)
        or completion.get("content_hash") != content_hash
        or not isinstance(evidence, Mapping)
        or evidence.get("content_hash") != content_hash
    ):
        raise EvidenceJobFailure(
            ReasonCode.SOURCE_CORRUPT,
            f"existing artifact content hash does not match {content_hash}",
        )
    expected_evidence_hash = completion.get("evidence_sha256")
    if not isinstance(expected_evidence_hash, str):
        raise EvidenceJobFailure(
            ReasonCode.SOURCE_CORRUPT, "existing artifact lacks evidence hash"
        )
    actual_evidence_hash = hashlib.sha256(
        (destination / "evidence.json").read_bytes()
    ).hexdigest()
    if actual_evidence_hash != expected_evidence_hash:
        raise EvidenceJobFailure(
            ReasonCode.SOURCE_CORRUPT, "existing evidence JSON hash differs"
        )
    files = evidence.get("artifacts")
    if not isinstance(files, list):
        raise EvidenceJobFailure(
            ReasonCode.SOURCE_CORRUPT, "artifact file list missing"
        )
    for item in files:
        if not isinstance(item, Mapping):
            raise EvidenceJobFailure(
                ReasonCode.SOURCE_CORRUPT, "invalid artifact file entry"
            )
        filename = item.get("filename")
        expected_hash = item.get("sha256")
        if not isinstance(filename, str) or not isinstance(expected_hash, str):
            raise EvidenceJobFailure(
                ReasonCode.SOURCE_CORRUPT, "invalid artifact hash entry"
            )
        path = (destination / filename).resolve()
        if not path.is_relative_to(destination.resolve()):
            raise EvidenceJobFailure(
                ReasonCode.SOURCE_CORRUPT, "artifact filename escapes packet"
            )
        try:
            actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as error:
            raise EvidenceJobFailure(
                ReasonCode.SOURCE_CORRUPT, f"artifact file missing: {filename}"
            ) from error
        if actual_hash != expected_hash:
            raise EvidenceJobFailure(
                ReasonCode.SOURCE_CORRUPT, f"artifact file hash differs: {filename}"
            )


def _classify_failure(error: Exception) -> EvidenceJobFailure:
    if isinstance(error, FileNotFoundError):
        return EvidenceJobFailure(ReasonCode.SOURCE_MISSING, str(error))
    if isinstance(error, GoesCropError) and isinstance(
        error.__cause__, FileNotFoundError
    ):
        return EvidenceJobFailure(ReasonCode.SOURCE_MISSING, str(error))
    if isinstance(error, (GoesCropError, CropArtifactError)):
        return EvidenceJobFailure(ReasonCode.SOURCE_CORRUPT, str(error))
    if isinstance(error, OSError):
        return EvidenceJobFailure(ReasonCode.ARTIFACT_WRITE_FAILED, str(error))
    if isinstance(error, (TypeError, ValueError)):
        return EvidenceJobFailure(ReasonCode.CONFIGURATION_INVALID, str(error))
    return EvidenceJobFailure(
        ReasonCode.SOURCE_CORRUPT, f"unexpected {type(error).__name__}: {error}"
    )


def _tile_parameters_from_dict(value: object) -> TilePreparationParameters:
    if not isinstance(value, Mapping):
        raise ValueError("tile_parameters must be an object")
    fields = {
        "physical_minimum_kelvin",
        "physical_maximum_kelvin",
        "display_lower_quantile",
        "display_upper_quantile",
        "target_shape",
        "minimum_valid_coverage",
        "resize_interpolation",
        "clahe_clip_limit",
        "clahe_tile_grid_size",
    }
    if set(value) != fields:
        raise ValueError("tile_parameters has an invalid shape")
    target_shape = _shape_from_json(
        value["target_shape"], "target_shape", nullable=True
    )
    clahe_shape = _shape_from_json(
        value["clahe_tile_grid_size"], "clahe_tile_grid_size"
    )
    if clahe_shape is None:
        raise ValueError("clahe_tile_grid_size must not be null")
    if value["resize_interpolation"] != "INTER_AREA downsample / INTER_LINEAR upsample":
        raise ValueError("tile resize_interpolation is unsupported")
    return TilePreparationParameters(
        physical_minimum_kelvin=cast(float, value["physical_minimum_kelvin"]),
        physical_maximum_kelvin=cast(float, value["physical_maximum_kelvin"]),
        display_lower_quantile=cast(float, value["display_lower_quantile"]),
        display_upper_quantile=cast(float, value["display_upper_quantile"]),
        target_shape=target_shape,
        minimum_valid_coverage=cast(float, value["minimum_valid_coverage"]),
        clahe_clip_limit=cast(float | None, value["clahe_clip_limit"]),
        clahe_tile_grid_size=clahe_shape,
    )


def _anomaly_parameters_from_dict(value: object) -> ContextualAnomalyParameters:
    if value is None:
        return DEVELOPMENT_CONTEXTUAL_ANOMALY_PARAMETERS
    if not isinstance(value, Mapping) or set(value) != {
        "local_background_kernel_pixels",
        "minimum_local_contrast_kelvin",
        "minimum_channel_difference_kelvin",
        "morphology_kernel_pixels",
        "minimum_component_area_pixels",
        "minimum_edge_distance_pixels",
    }:
        raise ValueError("anomaly_parameters has an invalid shape")
    return ContextualAnomalyParameters(
        local_background_kernel_pixels=cast(
            int, value["local_background_kernel_pixels"]
        ),
        minimum_local_contrast_kelvin=cast(
            float, value["minimum_local_contrast_kelvin"]
        ),
        minimum_channel_difference_kelvin=cast(
            float, value["minimum_channel_difference_kelvin"]
        ),
        morphology_kernel_pixels=cast(int, value["morphology_kernel_pixels"]),
        minimum_component_area_pixels=cast(int, value["minimum_component_area_pixels"]),
        minimum_edge_distance_pixels=cast(int, value["minimum_edge_distance_pixels"]),
    )


def _quality_thresholds_from_dict(value: object) -> ObservationQualityThresholds:
    if value is None:
        return DEVELOPMENT_QUALITY_THRESHOLDS
    if not isinstance(value, Mapping):
        raise ValueError("quality_thresholds must be an object")
    fields = {
        "selection_scope",
        "minimum_usable_coverage_fraction",
        "maximum_saturated_fraction",
        "minimum_contrast_span_kelvin",
        "minimum_texture_standard_deviation_kelvin",
        "blank_maximum_kelvin",
        "saturation_minimum_kelvin",
        "saturation_maximum_kelvin",
    }
    if (
        set(value) != fields
        or value["selection_scope"] != "development_cases_and_synthetic_fixtures_only"
    ):
        raise ValueError("quality_thresholds has an invalid development-only scope")
    return ObservationQualityThresholds(
        minimum_usable_coverage_fraction=cast(
            float, value["minimum_usable_coverage_fraction"]
        ),
        maximum_saturated_fraction=cast(float, value["maximum_saturated_fraction"]),
        minimum_contrast_span_kelvin=cast(float, value["minimum_contrast_span_kelvin"]),
        minimum_texture_standard_deviation_kelvin=cast(
            float, value["minimum_texture_standard_deviation_kelvin"]
        ),
        blank_maximum_kelvin=cast(float, value["blank_maximum_kelvin"]),
        saturation_minimum_kelvin=cast(
            float | None, value["saturation_minimum_kelvin"]
        ),
        saturation_maximum_kelvin=cast(
            float | None, value["saturation_maximum_kelvin"]
        ),
    )


def _persistence_parameters_from_dict(value: object) -> PersistenceParameters:
    if value is None:
        return DEVELOPMENT_PERSISTENCE_PARAMETERS
    if not isinstance(value, Mapping) or set(value) != {
        "maximum_resample_distance_kilometres",
        "maximum_centroid_distance_kilometres",
        "minimum_intersection_over_union",
        "minimum_component_area_pixels",
    }:
        raise ValueError("persistence_parameters has an invalid shape")
    return PersistenceParameters(
        maximum_resample_distance_kilometres=cast(
            float, value["maximum_resample_distance_kilometres"]
        ),
        maximum_centroid_distance_kilometres=cast(
            float, value["maximum_centroid_distance_kilometres"]
        ),
        minimum_intersection_over_union=cast(
            float, value["minimum_intersection_over_union"]
        ),
        minimum_component_area_pixels=cast(int, value["minimum_component_area_pixels"]),
    )


def _shape_from_json(
    value: object, field: str, *, nullable: bool = False
) -> tuple[int, int] | None:
    if value is None and nullable:
        return None
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ValueError(f"{field} must be a two-integer array")
    return value[0], value[1]


def _haversine_kilometres(
    latitude_a: npt.NDArray[np.float64],
    longitude_a: npt.NDArray[np.float64],
    latitude_b: npt.NDArray[np.float64],
    longitude_b: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    latitude_a_radians = np.deg2rad(latitude_a)
    latitude_b_radians = np.deg2rad(latitude_b)
    latitude_delta = latitude_b_radians - latitude_a_radians
    longitude_delta = np.deg2rad(((longitude_b - longitude_a + 180.0) % 360.0) - 180.0)
    haversine = (
        np.sin(latitude_delta / 2.0) ** 2
        + np.cos(latitude_a_radians)
        * np.cos(latitude_b_radians)
        * np.sin(longitude_delta / 2.0) ** 2
    )
    return np.asarray(
        12_742.0176 * np.arcsin(np.sqrt(np.clip(haversine, 0.0, 1.0))),
        dtype=np.float64,
    )


def _npy_bytes(array: npt.NDArray[np.generic]) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(array), allow_pickle=False)
    return buffer.getvalue()


def _png_bytes(image: npt.NDArray[np.uint8]) -> bytes:
    success, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if not success:
        raise OSError("OpenCV could not encode anomaly overlay")
    return bytes(encoded)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _positive_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a positive finite number")
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{field} must be a positive finite number")
    return number


def _milliseconds(duration_seconds: float) -> float:
    return round(max(duration_seconds, 0.0) * 1_000.0, 6)


def main(argv: list[str] | None = None) -> int:
    """Run a JSON-declared local evidence job and print one result record."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True, help="evidence job JSON")
    parser.add_argument(
        "--artifacts-dir", type=Path, required=True, help="local artifact root"
    )
    parser.add_argument(
        "--timeout-seconds", type=float, default=120.0, help="finite local job budget"
    )
    arguments = parser.parse_args(argv)
    try:
        job = load_evidence_job(arguments.job)
        result = run_evidence_job(
            job, arguments.artifacts_dir, timeout_seconds=arguments.timeout_seconds
        )
    except EvidenceJobFailure as error:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "reason_code": error.reason_code.value,
                    "detail": error.detail,
                },
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps({"status": "completed", **result.to_dict()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EVIDENCE_SCHEMA_VERSION",
    "JOB_SCHEMA_VERSION",
    "EvidenceJob",
    "EvidenceJobCancelled",
    "EvidenceJobFailure",
    "EvidenceJobObservation",
    "EvidenceJobResult",
    "EvidenceJobSource",
    "EvidenceJobTimeout",
    "load_evidence_job",
    "main",
    "run_evidence_job",
]
