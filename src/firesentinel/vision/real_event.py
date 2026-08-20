"""Cached-only OpenCV replay for one manually audited historical event.

The Day 9 slice intentionally does not discover or download imagery.  It reads
two already verified GOES Channel 7 source objects named by a checked-in
manifest, makes calibrated regional crops, and writes a compact evidence JSON
record plus an annotated initial/later reviewer panel.  Every value that could
affect a result is recorded or hashed, so ``--verify`` can reject drift.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Self, cast
from urllib.parse import quote

import cv2
import numpy as np
import numpy.typing as npt

from firesentinel.config import load_settings
from firesentinel.core.records import (
    Channel,
    ConfigurationReference,
    Coordinates,
    Measurement,
    Unit,
)
from firesentinel.data.goes_crop import (
    CropParameters,
    extract_calibrated_crop,
)
from firesentinel.data.source_cache import (
    SourceCacheError,
    SourceRequest,
    VerifiedSourceCache,
)

MANIFEST_VERSION: Final = 1
EVIDENCE_SCHEMA_VERSION: Final = 1
CONFIGURATION_ID: Final = "real-event-c07-v1"

MaskArray = npt.NDArray[np.bool]
Uint8Array = npt.NDArray[np.uint8]


class SliceError(RuntimeError):
    """Raised when a real-event manifest or replay cannot be reproduced."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _timestamp(value: object, field: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise SliceError(f"{field} must be an RFC 3339 UTC timestamp ending in Z")
    try:
        result = datetime.fromisoformat(value)
    except ValueError as error:
        raise SliceError(f"{field} must be an RFC 3339 UTC timestamp") from error
    if result.tzinfo is None or result.utcoffset() != UTC.utcoffset(result):
        raise SliceError(f"{field} must use UTC")
    return result.astimezone(UTC)


def _timestamp_text(value: datetime) -> str:
    return (
        value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
    )


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SliceError(f"{field} must be an object")
    return value


def _list(value: object, field: str) -> list[object]:
    if not isinstance(value, list):
        raise SliceError(f"{field} must be an array")
    return value


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SliceError(f"{field} must be a finite number")
    result = float(value)
    if not np.isfinite(result):
        raise SliceError(f"{field} must be a finite number")
    return result


def _integer(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SliceError(
            f"{field} must be an integer greater than or equal to {minimum}"
        )
    return value


def _hash(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SliceError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise SliceError(f"{field} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class SliceConfiguration:
    """The complete, manifest-pinned pixel processing configuration."""

    display_min_kelvin: float
    display_max_kelvin: float
    threshold_kelvin: float
    morphology_kernel_pixels: int
    minimum_component_area_pixels: int

    def __post_init__(self) -> None:
        display_min = _number(self.display_min_kelvin, "display_min_kelvin")
        display_max = _number(self.display_max_kelvin, "display_max_kelvin")
        threshold = _number(self.threshold_kelvin, "threshold_kelvin")
        if display_max <= display_min:
            raise SliceError(
                "display_max_kelvin must be greater than display_min_kelvin"
            )
        if not display_min <= threshold <= display_max:
            raise SliceError("threshold_kelvin must be inside the display range")
        kernel = _integer(
            self.morphology_kernel_pixels, "morphology_kernel_pixels", minimum=1
        )
        if kernel % 2 == 0:
            raise SliceError("morphology_kernel_pixels must be odd")
        object.__setattr__(self, "display_min_kelvin", display_min)
        object.__setattr__(self, "display_max_kelvin", display_max)
        object.__setattr__(self, "threshold_kelvin", threshold)
        object.__setattr__(self, "morphology_kernel_pixels", kernel)
        object.__setattr__(
            self,
            "minimum_component_area_pixels",
            _integer(
                self.minimum_component_area_pixels,
                "minimum_component_area_pixels",
                minimum=1,
            ),
        )

    def to_dict(self) -> dict[str, float | int]:
        return {
            "display_min_kelvin": self.display_min_kelvin,
            "display_max_kelvin": self.display_max_kelvin,
            "threshold_kelvin": self.threshold_kelvin,
            "morphology_kernel_pixels": self.morphology_kernel_pixels,
            "minimum_component_area_pixels": self.minimum_component_area_pixels,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        payload = _mapping(value, "pipeline")
        fields = {
            "display_min_kelvin",
            "display_max_kelvin",
            "threshold_kelvin",
            "morphology_kernel_pixels",
            "minimum_component_area_pixels",
        }
        if set(payload) != fields:
            raise SliceError("pipeline has an invalid shape")
        return cls(
            display_min_kelvin=_number(
                payload["display_min_kelvin"], "display_min_kelvin"
            ),
            display_max_kelvin=_number(
                payload["display_max_kelvin"], "display_max_kelvin"
            ),
            threshold_kelvin=_number(payload["threshold_kelvin"], "threshold_kelvin"),
            morphology_kernel_pixels=_integer(
                payload["morphology_kernel_pixels"], "morphology_kernel_pixels"
            ),
            minimum_component_area_pixels=_integer(
                payload["minimum_component_area_pixels"],
                "minimum_component_area_pixels",
            ),
        )


@dataclass(frozen=True, slots=True)
class EventSource:
    """A fully pinned immutable object in the local verified source cache."""

    source_id: str
    bucket: str
    object_key: str
    size_bytes: int
    sha256: str

    def request_for(self, case_id: str) -> SourceRequest:
        return SourceRequest(
            case_id=case_id,
            source_id=self.source_id,
            source_url=(
                f"https://{self.bucket}.s3.amazonaws.com/"
                f"{quote(self.object_key, safe='/')}"
            ),
            source_size_bytes=self.size_bytes,
            expected_sha256=self.sha256,
        )

    @classmethod
    def from_dict(cls, value: object) -> Self:
        payload = _mapping(value, "source")
        required = {"source_id", "bucket", "object_key", "size_bytes", "sha256"}
        if not required.issubset(payload):
            raise SliceError("source is missing required fields")
        source = cls(
            source_id=_text(payload["source_id"], "source.source_id"),
            bucket=_text(payload["bucket"], "source.bucket"),
            object_key=_text(payload["object_key"], "source.object_key"),
            size_bytes=_integer(payload["size_bytes"], "source.size_bytes", minimum=1),
            sha256=_hash(payload["sha256"], "source.sha256"),
        )
        # Reuse the cache request validation for safe identifiers and URLs.
        source.request_for("slice-validation")
        return source


@dataclass(frozen=True, slots=True)
class EventObservation:
    """One C07 observation paired with exactly one pinned source object."""

    observation_id: str
    source_id: str
    channel: Channel
    observation_time: datetime
    scan_start: datetime
    scan_end: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.observation_id, str) or not self.observation_id:
            raise SliceError("observation_id must be a non-empty string")
        if not isinstance(self.source_id, str) or not self.source_id:
            raise SliceError("source_id must be a non-empty string")
        try:
            channel = Channel(self.channel)
        except ValueError as error:
            raise SliceError("channel must be a supported Channel") from error
        if channel != Channel.C07:
            raise SliceError("the Day 9 real-event slice accepts Channel 7 only")
        observation_time = self.observation_time.astimezone(UTC)
        scan_start = self.scan_start.astimezone(UTC)
        scan_end = self.scan_end.astimezone(UTC)
        if scan_end < scan_start:
            raise SliceError("scan_end must not precede scan_start")
        object.__setattr__(self, "channel", channel)
        object.__setattr__(self, "observation_time", observation_time)
        object.__setattr__(self, "scan_start", scan_start)
        object.__setattr__(self, "scan_end", scan_end)

    @classmethod
    def from_dict(cls, value: object) -> Self:
        payload = _mapping(value, "observation")
        fields = {
            "observation_id",
            "source_id",
            "channel",
            "observation_time",
            "scan_start",
            "scan_end",
        }
        if set(payload) != fields:
            raise SliceError("observation has an invalid shape")
        return cls(
            observation_id=_text(
                payload["observation_id"], "observation.observation_id"
            ),
            source_id=_text(payload["source_id"], "observation.source_id"),
            channel=Channel(_text(payload["channel"], "observation.channel")),
            observation_time=_timestamp(
                payload["observation_time"], "observation_time"
            ),
            scan_start=_timestamp(payload["scan_start"], "scan_start"),
            scan_end=_timestamp(payload["scan_end"], "scan_end"),
        )


@dataclass(frozen=True, slots=True)
class RealEventManifest:
    """The one audited historical case consumed by this vertical slice."""

    case_id: str
    title: str
    location: Coordinates
    crop_parameters: CropParameters
    configuration: SliceConfiguration
    sources: tuple[EventSource, EventSource]
    observations: tuple[EventObservation, EventObservation]
    expected_outputs: Mapping[str, str]

    @property
    def configuration_reference(self) -> ConfigurationReference:
        payload = {
            "configuration_id": CONFIGURATION_ID,
            "crop_parameters": self.crop_parameters.to_dict(),
            "pipeline": self.configuration.to_dict(),
        }
        return ConfigurationReference(
            CONFIGURATION_ID, _sha256(_canonical_json(payload))
        )

    @classmethod
    def from_path(cls, path: Path) -> Self:
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SliceError(f"could not read real-event manifest {path}") from error
        top = _mapping(raw, "manifest")
        if top.get("manifest_version") != MANIFEST_VERSION:
            raise SliceError(f"manifest_version must be {MANIFEST_VERSION}")
        cases = _list(top.get("cases"), "cases")
        if len(cases) != 1:
            raise SliceError("the Day 9 manifest must contain exactly one case")
        case = _mapping(cases[0], "case")
        required = {
            "case_id",
            "title",
            "location",
            "manual_audit",
            "crop_parameters",
            "pipeline",
            "sources",
            "observations",
            "expected_outputs",
        }
        if not required.issubset(case):
            raise SliceError("case is missing required real-event fields")
        case_id = case["case_id"]
        title = case["title"]
        if not isinstance(case_id, str) or not case_id:
            raise SliceError("case_id must be a non-empty string")
        if not isinstance(title, str) or not title:
            raise SliceError("title must be a non-empty string")
        # Coordinates uses the project-wide canonical lat/lon representation.
        try:
            location = Coordinates.from_dict(case["location"])
            crop_parameters = CropParameters.from_dict(case["crop_parameters"])
            configuration = SliceConfiguration.from_dict(case["pipeline"])
        except (TypeError, ValueError) as error:
            raise SliceError(
                "case has invalid location, crop, or pipeline values"
            ) from error
        audit = _mapping(case["manual_audit"], "manual_audit")
        for field in ("audited_at", "reviewer", "event_reference", "selection_note"):
            if field not in audit:
                raise SliceError(f"manual_audit is missing {field}")
        _timestamp(audit["audited_at"], "manual_audit.audited_at")
        if not all(
            isinstance(audit[field], str) and audit[field]
            for field in audit
            if field != "audited_at"
        ):
            raise SliceError("manual_audit text fields must be non-empty strings")
        source_values = _list(case["sources"], "sources")
        observation_values = _list(case["observations"], "observations")
        if len(source_values) != 2 or len(observation_values) != 2:
            raise SliceError(
                "the Day 9 slice requires exactly two sources and observations"
            )
        sources = tuple(EventSource.from_dict(value) for value in source_values)
        observations = tuple(
            EventObservation.from_dict(value) for value in observation_values
        )
        if len({source.source_id for source in sources}) != 2:
            raise SliceError("source IDs must be unique")
        if len({item.observation_id for item in observations}) != 2:
            raise SliceError("observation IDs must be unique")
        if {item.source_id for item in observations} != {
            source.source_id for source in sources
        }:
            raise SliceError("each observation must use one selected source")
        if observations[1].scan_start <= observations[0].scan_start:
            raise SliceError("observations must be ordered initial then later")
        expected_raw = _mapping(case["expected_outputs"], "expected_outputs")
        expected = {
            key: _hash(value, f"expected_outputs.{key}")
            for key, value in expected_raw.items()
        }
        required_expected = {"evidence_content_hash", "reviewer_panel_sha256"}
        if set(expected) != required_expected:
            raise SliceError("expected_outputs has an invalid shape")
        return cls(
            case_id=case_id,
            title=title,
            location=location,
            crop_parameters=crop_parameters,
            configuration=configuration,
            sources=(sources[0], sources[1]),
            observations=(observations[0], observations[1]),
            expected_outputs=expected,
        )


@dataclass(frozen=True, slots=True)
class Component:
    """A connected hot-region measurement in source crop pixel coordinates."""

    label: int
    area_pixels: int
    bounding_box_xywh: tuple[int, int, int, int]
    centroid_xy: tuple[float, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "area_pixels": self.area_pixels,
            "bounding_box_xywh": list(self.bounding_box_xywh),
            "centroid_xy": list(self.centroid_xy),
        }


Contour = tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class FrameResult:
    """All deterministic OpenCV outputs for one calibrated frame."""

    display: Uint8Array
    threshold_mask: Uint8Array
    morphology_mask: Uint8Array
    components: tuple[Component, ...]
    contours: tuple[Contour, ...]
    maximum_kelvin: float

    @property
    def largest_component_pixels(self) -> int:
        return max((component.area_pixels for component in self.components), default=0)

    def contour_hash(self) -> str:
        return _sha256(_canonical_json([list(contour) for contour in self.contours]))

    def to_dict(
        self, *, observation: EventObservation, source: EventSource, crop_checksum: str
    ) -> dict[str, object]:
        return {
            "observation_id": observation.observation_id,
            "source_id": source.source_id,
            "channel": observation.channel.value,
            "scan_start": _timestamp_text(observation.scan_start),
            "scan_end": _timestamp_text(observation.scan_end),
            "source_sha256": source.sha256,
            "crop_content_hash": crop_checksum,
            "display_sha256": _sha256(self.display.tobytes(order="C")),
            "threshold_mask_sha256": _sha256(self.threshold_mask.tobytes(order="C")),
            "morphology_mask_sha256": _sha256(self.morphology_mask.tobytes(order="C")),
            "contour_sha256": self.contour_hash(),
            "maximum_kelvin": self.maximum_kelvin,
            "threshold_pixel_count": int(np.count_nonzero(self.threshold_mask)),
            "morphology_pixel_count": int(np.count_nonzero(self.morphology_mask)),
            "components": [component.to_dict() for component in self.components],
            "contours_xy": [
                [list(point) for point in contour] for contour in self.contours
            ],
        }


def analyse_frame(
    calibrated: npt.NDArray[np.float32],
    invalid_mask: MaskArray,
    configuration: SliceConfiguration,
) -> FrameResult:
    """Run display scaling through contours with an explicit invalid-pixel mask."""
    values = np.asarray(calibrated, dtype=np.float32)
    invalid = np.asarray(invalid_mask, dtype=bool)
    if values.ndim != 2 or invalid.shape != values.shape:
        raise SliceError(
            "calibrated values and invalid mask must be matching 2D arrays"
        )
    valid = np.logical_and(~invalid, np.isfinite(values))
    prepared = np.where(valid, values, np.float32(configuration.display_min_kelvin))
    alpha = 255.0 / (
        configuration.display_max_kelvin - configuration.display_min_kelvin
    )
    display = cv2.convertScaleAbs(
        np.clip(
            prepared, configuration.display_min_kelvin, configuration.display_max_kelvin
        ),
        alpha=alpha,
        beta=-configuration.display_min_kelvin * alpha,
    )
    display[~valid] = 0

    threshold_input = np.where(valid, prepared, np.float32(0.0))
    _, threshold_mask = cv2.threshold(
        threshold_input,
        configuration.threshold_kelvin,
        255.0,
        cv2.THRESH_BINARY,
    )
    threshold_u8 = np.asarray(threshold_mask, dtype=np.uint8)
    threshold_u8[~valid] = 0

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            configuration.morphology_kernel_pixels,
            configuration.morphology_kernel_pixels,
        ),
    )
    opened = cv2.morphologyEx(threshold_u8, cv2.MORPH_OPEN, kernel)
    morphology_mask = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    morphology_mask[~valid] = 0

    count, labels, statistics, centroids = cv2.connectedComponentsWithStats(
        morphology_mask, connectivity=8, ltype=cv2.CV_32S
    )
    kept_labels = [
        label
        for label in range(1, count)
        if int(statistics[label, cv2.CC_STAT_AREA])
        >= configuration.minimum_component_area_pixels
    ]
    component_mask = np.where(np.isin(labels, kept_labels), 255, 0).astype(np.uint8)
    components = tuple(
        Component(
            label=label,
            area_pixels=int(statistics[label, cv2.CC_STAT_AREA]),
            bounding_box_xywh=(
                int(statistics[label, cv2.CC_STAT_LEFT]),
                int(statistics[label, cv2.CC_STAT_TOP]),
                int(statistics[label, cv2.CC_STAT_WIDTH]),
                int(statistics[label, cv2.CC_STAT_HEIGHT]),
            ),
            centroid_xy=(float(centroids[label, 0]), float(centroids[label, 1])),
        )
        for label in kept_labels
    )
    contours_raw, _ = cv2.findContours(
        component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = tuple(
        sorted(
            (
                tuple(
                    (int(point[0]), int(point[1])) for point in contour.reshape(-1, 2)
                )
                for contour in contours_raw
            ),
            key=lambda contour: (
                min(point[1] for point in contour),
                min(point[0] for point in contour),
                len(contour),
                contour,
            ),
        )
    )
    maximum = float(np.max(values[valid])) if np.any(valid) else 0.0
    return FrameResult(
        display=np.ascontiguousarray(display),
        threshold_mask=np.ascontiguousarray(threshold_u8),
        morphology_mask=np.ascontiguousarray(component_mask),
        components=components,
        contours=contours,
        maximum_kelvin=maximum,
    )


def _review_panel(
    initial: FrameResult,
    later: FrameResult,
    manifest: RealEventManifest,
    initial_marker: tuple[int, int],
    later_marker: tuple[int, int],
) -> npt.NDArray[np.uint8]:
    """Create the deterministic initial/later reviewer image with contours."""
    rendered: list[npt.NDArray[np.uint8]] = []
    for title, frame, marker in (
        ("INITIAL C07", initial, initial_marker),
        ("LATER C07", later, later_marker),
    ):
        image = cv2.applyColorMap(frame.display, cv2.COLORMAP_INFERNO)
        contours = [
            np.asarray(contour, dtype=np.int32).reshape(-1, 1, 2)
            for contour in frame.contours
        ]
        if contours:
            cv2.drawContours(image, contours, -1, (0, 255, 255), 1, cv2.LINE_8)
        cv2.drawMarker(
            image,
            marker,
            (255, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=7,
            thickness=1,
            line_type=cv2.LINE_8,
        )
        enlarged = cast(
            Uint8Array,
            np.asarray(
                cv2.resize(image, (512, 512), interpolation=cv2.INTER_NEAREST),
                dtype=np.uint8,
            ),
        )
        header = np.zeros((54, 512, 3), dtype=np.uint8)
        cv2.putText(
            header,
            title,
            (12, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_8,
        )
        cv2.putText(
            header,
            (
                f"threshold {manifest.configuration.threshold_kelvin:.1f} K  "
                f"largest {frame.largest_component_pixels} px"
            ),
            (12, 43),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (220, 220, 220),
            1,
            cv2.LINE_8,
        )
        rendered.append(cast(Uint8Array, cv2.vconcat((header, enlarged))))
    return cast(Uint8Array, cv2.hconcat(rendered))


def _png_bytes(image: npt.NDArray[np.uint8]) -> bytes:
    success, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if not success:
        raise SliceError("OpenCV could not encode reviewer panel PNG")
    return bytes(encoded)


def _atomic_write(path: Path, contents: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, dir=path.parent) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(contents)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _evidence_payload(
    manifest: RealEventManifest,
    initial: FrameResult,
    later: FrameResult,
    initial_crop_hash: str,
    later_crop_hash: str,
    panel_sha256: str,
) -> dict[str, object]:
    configuration = manifest.configuration_reference
    initial_observation, later_observation = manifest.observations
    initial_source, later_source = manifest.sources
    measurements = (
        Measurement(
            "initial_largest_hot_region", initial.largest_component_pixels, Unit.PIXELS
        ),
        Measurement(
            "later_largest_hot_region", later.largest_component_pixels, Unit.PIXELS
        ),
        Measurement(
            "initial_component_count", len(initial.components), Unit.DIMENSIONLESS
        ),
        Measurement("later_component_count", len(later.components), Unit.DIMENSIONLESS),
        Measurement("initial_maximum_temperature", initial.maximum_kelvin, Unit.KELVIN),
        Measurement("later_maximum_temperature", later.maximum_kelvin, Unit.KELVIN),
    )
    payload: dict[str, object] = {
        "record_type": "real_event_evidence",
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "case_id": manifest.case_id,
        "title": manifest.title,
        "created_at": _timestamp_text(later_observation.scan_end),
        "coordinates": manifest.location.to_dict(),
        "configuration": configuration.to_dict(),
        "opencv_version": cv2.__version__,
        "measurements": [measurement.to_dict() for measurement in measurements],
        "frames": [
            initial.to_dict(
                observation=initial_observation,
                source=initial_source,
                crop_checksum=initial_crop_hash,
            ),
            later.to_dict(
                observation=later_observation,
                source=later_source,
                crop_checksum=later_crop_hash,
            ),
        ],
        "reviewer_panel": {
            "filename": "before-after.png",
            "sha256": panel_sha256,
            "width_pixels": 1024,
            "height_pixels": 566,
        },
    }
    payload["content_hash"] = _sha256(_canonical_json(payload))
    return payload


def replay_real_event(
    manifest: RealEventManifest,
    cache: VerifiedSourceCache,
    artifacts_root: Path,
    *,
    verify: bool,
) -> dict[str, object]:
    """Regenerate one evidence packet exclusively from verified cached sources."""
    paths: list[Path] = []
    for source in manifest.sources:
        try:
            paths.append(cache.require_cached(source.request_for(manifest.case_id)))
        except SourceCacheError as error:
            raise SliceError(str(error)) from error
    initial_crop = extract_calibrated_crop(paths[0], manifest.crop_parameters)
    later_crop = extract_calibrated_crop(paths[1], manifest.crop_parameters)
    if initial_crop.source_checksum != manifest.sources[0].sha256:
        raise SliceError("initial crop source checksum differs from the manifest")
    if later_crop.source_checksum != manifest.sources[1].sha256:
        raise SliceError("later crop source checksum differs from the manifest")
    initial = analyse_frame(
        initial_crop.calibrated, initial_crop.invalid_mask, manifest.configuration
    )
    later = analyse_frame(
        later_crop.calibrated, later_crop.invalid_mask, manifest.configuration
    )
    initial_pixel = initial_crop.nearest_pixel(
        manifest.location.latitude, manifest.location.longitude
    )
    later_pixel = later_crop.nearest_pixel(
        manifest.location.latitude, manifest.location.longitude
    )
    panel = _review_panel(
        initial,
        later,
        manifest,
        (initial_pixel.column, initial_pixel.row),
        (later_pixel.column, later_pixel.row),
    )
    panel_bytes = _png_bytes(panel)
    panel_sha256 = _sha256(panel_bytes)
    evidence = _evidence_payload(
        manifest,
        initial,
        later,
        initial_crop.content_checksum,
        later_crop.content_checksum,
        panel_sha256,
    )
    content_hash = evidence["content_hash"]
    assert isinstance(content_hash, str)
    if verify and (
        content_hash != manifest.expected_outputs["evidence_content_hash"]
        or panel_sha256 != manifest.expected_outputs["reviewer_panel_sha256"]
    ):
        raise SliceError(
            "replay outputs differ from the pinned manifest; inspect evidence.json"
        )
    destination = (
        Path(artifacts_root)
        / manifest.case_id
        / manifest.configuration_reference.content_hash
    )
    _atomic_write(destination / "before-after.png", panel_bytes)
    _atomic_write(destination / "evidence.json", _canonical_json(evidence) + b"\n")
    return {
        "artifact_directory": str(destination),
        "evidence_content_hash": content_hash,
        "reviewer_panel_sha256": panel_sha256,
        "initial_contour_sha256": initial.contour_hash(),
        "later_contour_sha256": later.contour_hash(),
        "verified": verify,
    }


def main(argv: list[str] | None = None) -> int:
    settings = load_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=settings.manifests_dir / "park-fire-20240725.json",
        help="checked-in one-case real-event manifest",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=settings.source_cache_dir,
        help="verified source cache directory",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=settings.artifacts_dir,
        help="destination root for generated reviewer artifacts",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="require hashes to match the manifest's manually audited expected outputs",
    )
    arguments = parser.parse_args(argv)
    manifest = RealEventManifest.from_path(arguments.manifest)
    result = replay_real_event(
        manifest,
        VerifiedSourceCache(arguments.cache_dir),
        arguments.artifacts_dir,
        verify=arguments.verify,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
