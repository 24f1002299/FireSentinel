"""Create isolated, evaluation-only event references from FIRMS CSV exports.

The ingester deliberately retains only acquisition time, WGS84 coordinates,
confidence, brightness, and instrument values.  Its outputs are references for
offline scoring; they are not runtime configuration or agent inputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import tempfile
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from typing import Any

DEFAULT_MAXIMUM_DISTANCE_KM = 10.0
DEFAULT_MAXIMUM_TIME_GAP_MINUTES = 60
EVALUATION_DATA_DIRECTORY = "evaluation-data"
FIRMS_OUTPUT_DIRECTORY = Path(EVALUATION_DATA_DIRECTORY) / "firms"
LABELS_FILENAME = "firms-event-labels.json"
AUDIT_FILENAME = "firms-event-labels.audit.json"
SCHEMA_VERSION = 1
COORDINATE_PRECISION_DECIMAL_PLACES = 6
BRIGHTNESS_PRECISION_DECIMAL_PLACES = 3
_COORDINATE_QUANTUM = Decimal("0.000001")
_BRIGHTNESS_QUANTUM = Decimal("0.001")
_CONFIDENCE_QUANTUM = Decimal("0.001")
_INSTRUMENT_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_CONFIDENCE_ALIASES = {
    "l": "low",
    "low": "low",
    "n": "nominal",
    "nominal": "nominal",
    "h": "high",
    "high": "high",
}


@dataclass(frozen=True, slots=True)
class FirmsDetection:
    """One normalized FIRMS detection containing only permitted label fields."""

    acquired_at: datetime
    latitude: float
    longitude: float
    confidence: str
    brightness_kelvin: float
    instrument: str

    def to_dict(self) -> dict[str, str | float]:
        return {
            "acquisition_time_utc": _format_timestamp(self.acquired_at),
            "latitude": self.latitude,
            "longitude": self.longitude,
            "confidence": self.confidence,
            "brightness_kelvin": self.brightness_kelvin,
            "instrument": self.instrument,
        }


@dataclass(frozen=True, slots=True)
class SourceAudit:
    """Hash and row count for a consumed FIRMS source, without retaining its path."""

    sha256: str
    row_count: int

    def to_dict(self) -> dict[str, str | int]:
        return {"sha256": self.sha256, "row_count": self.row_count}


@dataclass(frozen=True, slots=True)
class FirmsEvent:
    """A deterministic single-linkage event window over normalized detections."""

    event_id: str
    start_time: datetime
    end_time: datetime
    centroid_latitude: float
    centroid_longitude: float
    detections: tuple[FirmsDetection, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "start_time_utc": _format_timestamp(self.start_time),
            "end_time_utc": _format_timestamp(self.end_time),
            "centroid_latitude": self.centroid_latitude,
            "centroid_longitude": self.centroid_longitude,
            "detection_count": len(self.detections),
            "detections": [detection.to_dict() for detection in self.detections],
        }


@dataclass(frozen=True, slots=True)
class FirmsIngestion:
    """Normalized input and the event references derived from it."""

    source_audits: tuple[SourceAudit, ...]
    source_row_count: int
    blank_row_count: int
    normalized_detections: tuple[FirmsDetection, ...]
    duplicate_count: int
    events: tuple[FirmsEvent, ...]
    maximum_distance_km: float
    maximum_time_gap_minutes: int

    @property
    def source_hashes(self) -> tuple[str, ...]:
        return tuple(source.sha256 for source in self.source_audits)

    def labels_payload(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "record_type": "firms_event_reference_labels",
            "evaluation_only": True,
            "source_hashes": list(self.source_hashes),
            "clustering": _clustering_payload(
                self.maximum_distance_km, self.maximum_time_gap_minutes
            ),
            "events": [event.to_dict() for event in self.events],
        }

    def audit_payload(self, labels_sha256: str) -> dict[str, object]:
        first_timestamp: str | None = None
        last_timestamp: str | None = None
        if self.normalized_detections:
            first_timestamp = _format_timestamp(
                self.normalized_detections[0].acquired_at
            )
            last_timestamp = _format_timestamp(
                self.normalized_detections[-1].acquired_at
            )
        instruments = Counter(
            detection.instrument for detection in self.normalized_detections
        )
        confidence_values = Counter(
            detection.confidence for detection in self.normalized_detections
        )
        unique_count = len(self.normalized_detections)
        return {
            "schema_version": SCHEMA_VERSION,
            "record_type": "firms_event_reference_audit",
            "evaluation_only": True,
            "labels_filename": LABELS_FILENAME,
            "labels_sha256": labels_sha256,
            "source_hashes": list(self.source_hashes),
            "sources": [source.to_dict() for source in self.source_audits],
            "counts": {
                "source_files": len(self.source_audits),
                "source_rows": self.source_row_count,
                "blank_rows_ignored": self.blank_row_count,
                "normalized_detections": self.source_row_count - self.blank_row_count,
                "duplicate_detections_removed": self.duplicate_count,
                "unique_detections": unique_count,
                "event_windows": len(self.events),
            },
            "date_range": {
                "first_acquisition_time_utc": first_timestamp,
                "last_acquisition_time_utc": last_timestamp,
            },
            "normalization_statistics": {
                "timestamp_timezone": "UTC",
                "timestamp_precision": "minute",
                "coordinate_reference_system": "WGS84",
                "coordinate_precision_decimal_places": (
                    COORDINATE_PRECISION_DECIMAL_PLACES
                ),
                "longitude_range": "[-180, 180]",
                "brightness_unit": "kelvin",
                "brightness_precision_decimal_places": (
                    BRIGHTNESS_PRECISION_DECIMAL_PLACES
                ),
                "instruments": dict(sorted(instruments.items())),
                "confidence_values": dict(sorted(confidence_values.items())),
            },
            "clustering": _clustering_payload(
                self.maximum_distance_km, self.maximum_time_gap_minutes
            ),
        }


def ingest_firms_csvs(
    sources: Iterable[Path],
    *,
    maximum_distance_km: float = DEFAULT_MAXIMUM_DISTANCE_KM,
    maximum_time_gap_minutes: int = DEFAULT_MAXIMUM_TIME_GAP_MINUTES,
) -> FirmsIngestion:
    """Read CSV exports, normalize permitted fields, deduplicate, and cluster.

    Every non-blank source row is strict: malformed values raise ``ValueError``
    with the source location rather than silently producing a partial label set.
    """

    _validate_clustering_parameters(maximum_distance_km, maximum_time_gap_minutes)
    source_paths = tuple(Path(source) for source in sources)
    if not source_paths:
        raise ValueError("At least one FIRMS CSV source is required.")

    detections: list[FirmsDetection] = []
    source_audits: list[SourceAudit] = []
    source_row_count = 0
    blank_row_count = 0
    seen_paths: set[Path] = set()
    for source in source_paths:
        resolved_source = source.resolve()
        if resolved_source in seen_paths:
            raise ValueError(f"FIRMS source was supplied more than once: {source}")
        seen_paths.add(resolved_source)
        source_detections, source_audit, source_rows, source_blanks = _read_source(
            source
        )
        detections.extend(source_detections)
        source_audits.append(source_audit)
        source_row_count += source_rows
        blank_row_count += source_blanks

    sorted_detections = tuple(sorted(detections, key=_detection_sort_key))
    unique_detections = tuple(dict.fromkeys(sorted_detections))
    duplicate_count = len(sorted_detections) - len(unique_detections)
    events = cluster_detections(
        unique_detections,
        maximum_distance_km=maximum_distance_km,
        maximum_time_gap_minutes=maximum_time_gap_minutes,
    )
    return FirmsIngestion(
        source_audits=tuple(sorted(source_audits, key=lambda source: source.sha256)),
        source_row_count=source_row_count,
        blank_row_count=blank_row_count,
        normalized_detections=unique_detections,
        duplicate_count=duplicate_count,
        events=events,
        maximum_distance_km=maximum_distance_km,
        maximum_time_gap_minutes=maximum_time_gap_minutes,
    )


def cluster_detections(
    detections: Sequence[FirmsDetection],
    *,
    maximum_distance_km: float = DEFAULT_MAXIMUM_DISTANCE_KM,
    maximum_time_gap_minutes: int = DEFAULT_MAXIMUM_TIME_GAP_MINUTES,
) -> tuple[FirmsEvent, ...]:
    """Cluster time-ordered detections by temporal and geodesic proximity.

    Clusters use deterministic single linkage.  Thus an event can contain a
    sequence of nearby detections even where its first and last members are
    farther apart than one configured gap; that behavior is recorded in the
    generated audit artifact.
    """

    _validate_clustering_parameters(maximum_distance_km, maximum_time_gap_minutes)
    ordered = tuple(sorted(detections, key=_detection_sort_key))
    if not ordered:
        return ()
    parents = list(range(len(ordered)))
    maximum_gap = timedelta(minutes=maximum_time_gap_minutes)
    bucket_size = _chord_distance(maximum_distance_km)
    active_buckets: dict[tuple[int, int, int], set[int]] = {}
    spatial_coordinates = tuple(
        _unit_sphere_coordinates(detection.latitude, detection.longitude)
        for detection in ordered
    )

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parents[second_root] = first_root

    first_active_index = 0
    for current_index, current in enumerate(ordered):
        while (
            current.acquired_at - ordered[first_active_index].acquired_at > maximum_gap
        ):
            expired_bucket = _spatial_bucket(
                spatial_coordinates[first_active_index], bucket_size
            )
            active_buckets[expired_bucket].remove(first_active_index)
            if not active_buckets[expired_bucket]:
                del active_buckets[expired_bucket]
            first_active_index += 1

        current_bucket = _spatial_bucket(
            spatial_coordinates[current_index], bucket_size
        )
        nearby_indices = _nearby_active_indices(current_bucket, active_buckets)
        for previous_index in sorted(nearby_indices):
            previous = ordered[previous_index]
            if (
                _haversine_distance_km(
                    current.latitude,
                    current.longitude,
                    previous.latitude,
                    previous.longitude,
                )
                <= maximum_distance_km
            ):
                union(previous_index, current_index)
        active_buckets.setdefault(current_bucket, set()).add(current_index)

    groups: dict[int, list[FirmsDetection]] = {}
    for index, detection in enumerate(ordered):
        groups.setdefault(find(index), []).append(detection)

    events = tuple(_event_from_detections(group) for group in groups.values())
    return tuple(sorted(events, key=lambda event: (event.start_time, event.event_id)))


def write_evaluation_references(
    ingestion: FirmsIngestion, output_directory: Path, *, overwrite: bool = False
) -> tuple[Path, Path]:
    """Write labels and a separately hashed audit record to an evaluation path."""

    labels_path = output_directory / LABELS_FILENAME
    audit_path = output_directory / AUDIT_FILENAME
    labels_bytes = _canonical_json_bytes(ingestion.labels_payload())
    labels_sha256 = hashlib.sha256(labels_bytes).hexdigest()
    audit_bytes = _canonical_json_bytes(ingestion.audit_payload(labels_sha256))
    _preflight_output(labels_path, labels_bytes, overwrite=overwrite)
    _preflight_output(audit_path, audit_bytes, overwrite=overwrite)
    _atomic_write(labels_path, labels_bytes)
    _atomic_write(audit_path, audit_bytes)
    return labels_path, audit_path


def default_output_directory() -> Path:
    """Return the repository's deliberately non-configurable label location."""

    return Path(__file__).resolve().parents[3] / FIRMS_OUTPUT_DIRECTORY


def _read_source(
    source: Path,
) -> tuple[list[FirmsDetection], SourceAudit, int, int]:
    if not source.is_file():
        raise ValueError(f"FIRMS source does not exist or is not a file: {source}")
    with source.open("rb") as binary_source:
        source_sha256 = hashlib.file_digest(binary_source, "sha256").hexdigest()

    detections: list[FirmsDetection] = []
    source_rows = 0
    blank_rows = 0
    with source.open("r", encoding="utf-8-sig", newline="") as csv_source:
        reader = csv.DictReader(csv_source)
        header = _normalized_header(reader.fieldnames, source)
        _validate_required_fields(header, source)
        for row_number, raw_row in enumerate(reader, start=2):
            source_rows += 1
            row = {key: value for key, value in raw_row.items() if key is not None}
            if all(value is None or not value.strip() for value in row.values()):
                blank_rows += 1
                continue
            try:
                detections.append(_normalize_row(row, header))
            except ValueError as error:
                raise ValueError(f"{source}:{row_number}: {error}") from error
    return (
        detections,
        SourceAudit(sha256=source_sha256, row_count=source_rows - blank_rows),
        source_rows,
        blank_rows,
    )


def _normalized_header(
    fieldnames: Sequence[str] | None, source: Path
) -> dict[str, str]:
    if fieldnames is None:
        raise ValueError(f"FIRMS source has no CSV header: {source}")
    header: dict[str, str] = {}
    for fieldname in fieldnames:
        normalized = fieldname.strip().lower()
        if not normalized:
            raise ValueError(f"FIRMS source has an empty CSV header name: {source}")
        if normalized in header:
            raise ValueError(
                f"FIRMS source repeats CSV header '{normalized}': {source}"
            )
        header[normalized] = fieldname
    return header


def _validate_required_fields(header: dict[str, str], source: Path) -> None:
    required = {
        "latitude",
        "longitude",
        "acq_date",
        "acq_time",
        "confidence",
        "instrument",
    }
    missing = sorted(required.difference(header))
    if missing:
        raise ValueError(
            f"FIRMS source is missing required columns {missing}: {source}"
        )
    if "brightness" not in header and "bright_ti4" not in header:
        raise ValueError(
            f"FIRMS source needs a 'brightness' or 'bright_ti4' column: {source}"
        )


def _normalize_row(
    row: dict[str, str | None], header: dict[str, str]
) -> FirmsDetection:
    acquisition_date = _required_value(row, header, "acq_date")
    acquisition_time = _required_value(row, header, "acq_time")
    brightness = _optional_value(row, header, "brightness")
    if brightness is None:
        brightness = _optional_value(row, header, "bright_ti4")
    if brightness is None:
        raise ValueError("brightness is required")
    return FirmsDetection(
        acquired_at=_normalize_timestamp(acquisition_date, acquisition_time),
        latitude=_normalize_coordinate(
            _required_value(row, header, "latitude"), "latitude"
        ),
        longitude=_normalize_coordinate(
            _required_value(row, header, "longitude"), "longitude"
        ),
        confidence=_normalize_confidence(_required_value(row, header, "confidence")),
        brightness_kelvin=_normalize_brightness(brightness),
        instrument=_normalize_instrument(_required_value(row, header, "instrument")),
    )


def _required_value(
    row: dict[str, str | None], header: dict[str, str], name: str
) -> str:
    value = _optional_value(row, header, name)
    if value is None:
        raise ValueError(f"{name} is required")
    return value


def _optional_value(
    row: dict[str, str | None], header: dict[str, str], name: str
) -> str | None:
    source_header = header.get(name)
    if source_header is None:
        return None
    value = row.get(source_header)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _normalize_timestamp(acquisition_date: str, acquisition_time: str) -> datetime:
    try:
        parsed_date = datetime.strptime(acquisition_date, "%Y-%m-%d").date()
    except ValueError as error:
        raise ValueError("acq_date must use YYYY-MM-DD") from error
    if not acquisition_time.isascii() or not acquisition_time.isdigit():
        raise ValueError("acq_time must be a one- to four-digit HHMM value")
    padded_time = acquisition_time.zfill(4)
    if len(padded_time) != 4:
        raise ValueError("acq_time must be a one- to four-digit HHMM value")
    hour = int(padded_time[:2])
    minute = int(padded_time[2:])
    if hour > 23 or minute > 59:
        raise ValueError("acq_time must be a valid UTC HHMM value")
    return datetime(
        parsed_date.year, parsed_date.month, parsed_date.day, hour, minute, tzinfo=UTC
    )


def _normalize_coordinate(value: str, field_name: str) -> float:
    coordinate = _decimal(value, field_name)
    lower_bound, upper_bound = (-90, 90) if field_name == "latitude" else (-180, 180)
    if not Decimal(lower_bound) <= coordinate <= Decimal(upper_bound):
        raise ValueError(f"{field_name} must be within [{lower_bound}, {upper_bound}]")
    return _normalized_float(coordinate.quantize(_COORDINATE_QUANTUM, ROUND_HALF_UP))


def _normalize_brightness(value: str) -> float:
    brightness = _decimal(value, "brightness")
    if not Decimal("0") < brightness <= Decimal("2000"):
        raise ValueError("brightness must be greater than 0 and no greater than 2000 K")
    return _normalized_float(brightness.quantize(_BRIGHTNESS_QUANTUM, ROUND_HALF_UP))


def _normalize_confidence(value: str) -> str:
    lowered = value.lower()
    if lowered in _CONFIDENCE_ALIASES:
        return _CONFIDENCE_ALIASES[lowered]
    confidence = _decimal(value, "confidence")
    if not Decimal("0") <= confidence <= Decimal("100"):
        raise ValueError("numeric confidence must be within [0, 100]")
    normalized = confidence.quantize(_CONFIDENCE_QUANTUM, ROUND_HALF_UP)
    return format(normalized.normalize(), "f")


def _normalize_instrument(value: str) -> str:
    if not _INSTRUMENT_PATTERN.fullmatch(value):
        raise ValueError("instrument must be a non-empty identifier")
    return value.upper()


def _decimal(value: str, field_name: str) -> Decimal:
    try:
        decimal = Decimal(value)
    except InvalidOperation as error:
        raise ValueError(f"{field_name} must be numeric") from error
    if not decimal.is_finite():
        raise ValueError(f"{field_name} must be finite")
    return decimal


def _normalized_float(value: Decimal) -> float:
    return 0.0 if value == 0 else float(value)


def _detection_sort_key(
    detection: FirmsDetection,
) -> tuple[datetime, float, float, str, float, str]:
    return (
        detection.acquired_at,
        detection.latitude,
        detection.longitude,
        detection.confidence,
        detection.brightness_kelvin,
        detection.instrument,
    )


def _event_from_detections(detections: Sequence[FirmsDetection]) -> FirmsEvent:
    ordered = tuple(sorted(detections, key=_detection_sort_key))
    event_fingerprint = hashlib.sha256(
        _canonical_json_bytes(
            {"detections": [detection.to_dict() for detection in ordered]}
        )
    ).hexdigest()[:12]
    return FirmsEvent(
        event_id=(
            f"firms-{ordered[0].acquired_at.strftime('%Y%m%dT%H%MZ')}-"
            f"{event_fingerprint}"
        ),
        start_time=ordered[0].acquired_at,
        end_time=ordered[-1].acquired_at,
        centroid_latitude=_mean_coordinate(
            [detection.latitude for detection in ordered]
        ),
        centroid_longitude=_mean_longitude(
            [detection.longitude for detection in ordered]
        ),
        detections=ordered,
    )


def _mean_coordinate(coordinates: Sequence[float]) -> float:
    return _normalize_coordinate(str(sum(coordinates) / len(coordinates)), "latitude")


def _mean_longitude(longitudes: Sequence[float]) -> float:
    angles = [math.radians(longitude) for longitude in longitudes]
    mean_longitude = math.degrees(
        math.atan2(
            sum(math.sin(angle) for angle in angles),
            sum(math.cos(angle) for angle in angles),
        )
    )
    return _normalize_coordinate(str(mean_longitude), "longitude")


def _haversine_distance_km(
    first_latitude: float,
    first_longitude: float,
    second_latitude: float,
    second_longitude: float,
) -> float:
    radius_km = 6371.0088
    first_latitude_radians = math.radians(first_latitude)
    second_latitude_radians = math.radians(second_latitude)
    latitude_delta = second_latitude_radians - first_latitude_radians
    longitude_delta = math.radians(second_longitude - first_longitude)
    haversine = (
        math.sin(latitude_delta / 2) ** 2
        + math.cos(first_latitude_radians)
        * math.cos(second_latitude_radians)
        * math.sin(longitude_delta / 2) ** 2
    )
    return 2 * radius_km * math.asin(math.sqrt(haversine))


def _unit_sphere_coordinates(
    latitude: float, longitude: float
) -> tuple[float, float, float]:
    """Return an Earth-centred unit-vector coordinate for spatial bucketing."""

    latitude_radians = math.radians(latitude)
    longitude_radians = math.radians(longitude)
    latitude_cosine = math.cos(latitude_radians)
    return (
        latitude_cosine * math.cos(longitude_radians),
        latitude_cosine * math.sin(longitude_radians),
        math.sin(latitude_radians),
    )


def _chord_distance(distance_km: float) -> float:
    """Convert a surface distance into its unit-sphere chord distance."""

    radius_km = 6371.0088
    return max(2 * math.sin(min(distance_km / radius_km, math.pi) / 2), math.ulp(1.0))


def _spatial_bucket(
    coordinates: tuple[float, float, float], bucket_size: float
) -> tuple[int, int, int]:
    return (
        math.floor((coordinates[0] + 1) / bucket_size),
        math.floor((coordinates[1] + 1) / bucket_size),
        math.floor((coordinates[2] + 1) / bucket_size),
    )


def _nearby_active_indices(
    bucket: tuple[int, int, int], active_buckets: dict[tuple[int, int, int], set[int]]
) -> set[int]:
    nearby: set[int] = set()
    for x_offset in (-1, 0, 1):
        for y_offset in (-1, 0, 1):
            for z_offset in (-1, 0, 1):
                candidate_indices = active_buckets.get(
                    (
                        bucket[0] + x_offset,
                        bucket[1] + y_offset,
                        bucket[2] + z_offset,
                    )
                )
                if candidate_indices is not None:
                    nearby.update(candidate_indices)
    return nearby


def _validate_clustering_parameters(
    maximum_distance_km: float, maximum_time_gap_minutes: int
) -> None:
    if (
        isinstance(maximum_distance_km, bool)
        or not isinstance(maximum_distance_km, (int, float))
        or not math.isfinite(maximum_distance_km)
        or maximum_distance_km <= 0
    ):
        raise ValueError("maximum_distance_km must be a positive finite number")
    if (
        isinstance(maximum_time_gap_minutes, bool)
        or not isinstance(maximum_time_gap_minutes, int)
        or maximum_time_gap_minutes <= 0
    ):
        raise ValueError("maximum_time_gap_minutes must be a positive integer")


def _clustering_payload(
    maximum_distance_km: float, maximum_time_gap_minutes: int
) -> dict[str, object]:
    return {
        "method": "single_linkage_time_and_geodesic_distance",
        "maximum_distance_km": maximum_distance_km,
        "maximum_time_gap_minutes": maximum_time_gap_minutes,
    }


def _format_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _preflight_output(path: Path, content: bytes, *, overwrite: bool) -> None:
    if not path.exists() or path.read_bytes() == content or overwrite:
        return
    raise FileExistsError(
        f"Refusing to replace existing evaluation reference '{path}'; use --overwrite."
    )


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", delete=False, dir=path.parent, prefix=f".{path.name}."
    ) as temporary:
        temporary.write(content)
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


def main(argv: list[str] | None = None) -> int:
    """Ingest selected local FIRMS exports into the evaluation-only directory."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        action="append",
        required=True,
        help="local FIRMS CSV export; repeat for multiple sources",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_directory(),
        help="evaluation-only output directory (default: evaluation-data/firms)",
    )
    parser.add_argument(
        "--maximum-distance-km",
        type=float,
        default=DEFAULT_MAXIMUM_DISTANCE_KM,
        help=f"event clustering distance (default: {DEFAULT_MAXIMUM_DISTANCE_KM})",
    )
    parser.add_argument(
        "--maximum-time-gap-minutes",
        type=int,
        default=DEFAULT_MAXIMUM_TIME_GAP_MINUTES,
        help=f"event clustering time gap (default: {DEFAULT_MAXIMUM_TIME_GAP_MINUTES})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing label and audit files only when their content changed",
    )
    arguments = parser.parse_args(argv)
    output_directory = arguments.output_dir.resolve()
    allowed_root = default_output_directory().parent.resolve()
    if not output_directory.is_relative_to(allowed_root):
        parser.error(
            f"--output-dir must be inside the evaluation-only directory {allowed_root}"
        )
    ingestion = ingest_firms_csvs(
        arguments.source,
        maximum_distance_km=arguments.maximum_distance_km,
        maximum_time_gap_minutes=arguments.maximum_time_gap_minutes,
    )
    labels_path, audit_path = write_evaluation_references(
        ingestion, output_directory, overwrite=arguments.overwrite
    )
    print(
        json.dumps(
            {
                "labels": str(labels_path),
                "audit": str(audit_path),
                "events": len(ingestion.events),
                "detections": len(ingestion.normalized_detections),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
