"""Deterministic calibrated regional crops from cached GOES ABI NetCDF files.

The ``ABI-L2-CMIPF`` product stores packed ``CMI`` values on the fixed-grid
GOES projection.  This module deliberately decodes those packed values itself,
rather than relying on a NetCDF reader's implicit masking and scaling.  That
makes the calibration, validity policy, crop window, and resulting artifact
fully inspectable and repeatable.

Only local files are read.  In normal use ``source_path`` is the verified
``DownloadReceipt.cache_path`` produced by :mod:`firesentinel.data.source_cache`.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import numpy.typing as npt
from netCDF4 import Dataset, num2date
from pyproj import CRS, Transformer

_SCHEMA_VERSION: Final = 1
_ARCHIVE_NAMES: Final = (
    "calibrated.npy",
    "dqf.npy",
    "invalid_mask.npy",
    "latitude.npy",
    "longitude.npy",
    "metadata.json",
    "x.npy",
    "y.npy",
)
_CALIBRATION_ATTRIBUTES: Final = (
    "scale_factor",
    "add_offset",
    "planck_fk1",
    "planck_fk2",
    "planck_bc1",
    "planck_bc2",
    "kappa0",
)

FloatArray = npt.NDArray[np.float32]
CoordinateArray = npt.NDArray[np.float64]
MaskArray = npt.NDArray[np.bool]
QualityArray = npt.NDArray[np.generic]


class GoesCropError(RuntimeError):
    """Base class for an invalid GOES crop request or source object."""


class GoesCropFormatError(GoesCropError):
    """Raised when a source NetCDF object lacks required GOES metadata."""


class GeographicBoundsError(GoesCropError):
    """Raised when requested WGS84 bounds are malformed or unsupported."""


class RegionOutsideSourceError(GoesCropError):
    """Raised when none of a requested region projects into the source grid."""


class CropArtifactError(GoesCropError):
    """Raised when a persisted crop archive cannot be validated."""


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite number")
    return number


def _integer(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(
            f"{field} must be an integer greater than or equal to {minimum}"
        )
    return value


def _sha256(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _timestamp(value: object, field: str) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(f"{field} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(UTC)


def _timestamp_text(value: datetime) -> str:
    return (
        value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
    )


def _parse_timestamp(value: object, field: str) -> datetime:
    if not isinstance(value, str):
        raise GoesCropFormatError(f"{field} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise GoesCropFormatError(f"{field} must be an ISO-8601 timestamp") from error
    try:
        return _timestamp(parsed, field)
    except ValueError as error:
        raise GoesCropFormatError(str(error)) from error


def _attribute(variable: Any, name: str, *, required: bool = True) -> Any:
    if name not in variable.ncattrs():
        if required:
            raise GoesCropFormatError(
                f"NetCDF variable {variable.name!r} lacks required attribute {name!r}"
            )
        return None
    return variable.getncattr(name)


def _numeric_attribute(
    variable: Any, name: str, *, required: bool = True
) -> float | None:
    value = _attribute(variable, name, required=required)
    if value is None:
        return None
    try:
        return _finite_number(value, f"{variable.name}.{name}")
    except ValueError as error:
        raise GoesCropFormatError(str(error)) from error


def _read_unscaled(variable: Any) -> npt.NDArray[np.generic]:
    """Read a NetCDF variable without automatic masking or scale/offset decode."""
    variable.set_auto_maskandscale(False)
    return np.asarray(variable[:])


def _readonly(array: npt.NDArray[np.generic]) -> npt.NDArray[np.generic]:
    result = np.ascontiguousarray(array).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class GeographicBounds:
    """A non-wrapping WGS84 latitude/longitude rectangle in decimal degrees."""

    south: float
    west: float
    north: float
    east: float

    def __post_init__(self) -> None:
        south = _finite_number(self.south, "south")
        west = _finite_number(self.west, "west")
        north = _finite_number(self.north, "north")
        east = _finite_number(self.east, "east")
        if not -90.0 <= south <= 90.0 or not -90.0 <= north <= 90.0:
            raise GeographicBoundsError("south and north must be in [-90, 90]")
        if not -180.0 <= west <= 180.0 or not -180.0 <= east <= 180.0:
            raise GeographicBoundsError("west and east must be in [-180, 180]")
        if south >= north:
            raise GeographicBoundsError("south must be less than north")
        if west >= east:
            raise GeographicBoundsError(
                "west must be less than east; antimeridian-wrapping bounds "
                "are unsupported"
            )
        object.__setattr__(self, "south", south)
        object.__setattr__(self, "west", west)
        object.__setattr__(self, "north", north)
        object.__setattr__(self, "east", east)

    def to_dict(self) -> dict[str, float]:
        return {
            "south": self.south,
            "west": self.west,
            "north": self.north,
            "east": self.east,
        }

    @classmethod
    def from_dict(cls, value: object) -> GeographicBounds:
        if not isinstance(value, dict) or set(value) != {
            "south",
            "west",
            "north",
            "east",
        }:
            raise CropArtifactError("geographic bounds have an invalid shape")
        return cls(
            south=value["south"],
            west=value["west"],
            north=value["north"],
            east=value["east"],
        )


@dataclass(frozen=True, slots=True)
class CropParameters:
    """Requested regional extent and deterministic source-window policy."""

    bounds: GeographicBounds
    padding_pixels: int = 0
    maximum_usable_dqf: int = 1
    edge_samples: int = 33

    def __post_init__(self) -> None:
        if not isinstance(self.bounds, GeographicBounds):
            raise ValueError("bounds must be GeographicBounds")
        object.__setattr__(
            self, "padding_pixels", _integer(self.padding_pixels, "padding_pixels")
        )
        object.__setattr__(
            self,
            "maximum_usable_dqf",
            _integer(self.maximum_usable_dqf, "maximum_usable_dqf"),
        )
        object.__setattr__(
            self, "edge_samples", _integer(self.edge_samples, "edge_samples", minimum=2)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "bounds": self.bounds.to_dict(),
            "padding_pixels": self.padding_pixels,
            "maximum_usable_dqf": self.maximum_usable_dqf,
            "edge_samples": self.edge_samples,
        }

    @classmethod
    def from_dict(cls, value: object) -> CropParameters:
        if not isinstance(value, dict) or set(value) != {
            "bounds",
            "padding_pixels",
            "maximum_usable_dqf",
            "edge_samples",
        }:
            raise CropArtifactError("crop parameters have an invalid shape")
        return cls(
            bounds=GeographicBounds.from_dict(value["bounds"]),
            padding_pixels=value["padding_pixels"],
            maximum_usable_dqf=value["maximum_usable_dqf"],
            edge_samples=value["edge_samples"],
        )


@dataclass(frozen=True, slots=True)
class SourceWindow:
    """Half-open row/column source indices selected for one regional crop."""

    row_start: int
    row_stop: int
    column_start: int
    column_stop: int

    def __post_init__(self) -> None:
        row_start = _integer(self.row_start, "row_start")
        row_stop = _integer(self.row_stop, "row_stop")
        column_start = _integer(self.column_start, "column_start")
        column_stop = _integer(self.column_stop, "column_stop")
        if row_stop <= row_start or column_stop <= column_start:
            raise ValueError("source window must have positive row and column extent")
        object.__setattr__(self, "row_start", row_start)
        object.__setattr__(self, "row_stop", row_stop)
        object.__setattr__(self, "column_start", column_start)
        object.__setattr__(self, "column_stop", column_stop)

    @property
    def shape(self) -> tuple[int, int]:
        return self.row_stop - self.row_start, self.column_stop - self.column_start

    def to_dict(self) -> dict[str, int]:
        return {
            "row_start": self.row_start,
            "row_stop": self.row_stop,
            "column_start": self.column_start,
            "column_stop": self.column_stop,
        }

    @classmethod
    def from_dict(cls, value: object) -> SourceWindow:
        if not isinstance(value, dict) or set(value) != {
            "row_start",
            "row_stop",
            "column_start",
            "column_stop",
        }:
            raise CropArtifactError("source window has an invalid shape")
        return cls(
            row_start=value["row_start"],
            row_stop=value["row_stop"],
            column_start=value["column_start"],
            column_stop=value["column_stop"],
        )


@dataclass(frozen=True, slots=True)
class ScanTiming:
    """UTC source scan coverage carried forward without adding wall-clock time."""

    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        start = _timestamp(self.start, "start")
        end = _timestamp(self.end, "end")
        if end < start:
            raise ValueError("end must not precede start")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    def to_dict(self) -> dict[str, str]:
        return {"start": _timestamp_text(self.start), "end": _timestamp_text(self.end)}

    @classmethod
    def from_dict(cls, value: object) -> ScanTiming:
        if not isinstance(value, dict) or set(value) != {"start", "end"}:
            raise CropArtifactError("scan timing has an invalid shape")
        return cls(
            start=_parse_timestamp(value["start"], "timing.start"),
            end=_parse_timestamp(value["end"], "timing.end"),
        )


@dataclass(frozen=True, slots=True)
class ProjectionMetadata:
    """GOES fixed-grid projection values required to locate crop pixels."""

    perspective_point_height: float
    longitude_of_projection_origin: float
    latitude_of_projection_origin: float
    semi_major_axis: float
    semi_minor_axis: float
    sweep_angle_axis: str

    def __post_init__(self) -> None:
        for field in (
            "perspective_point_height",
            "longitude_of_projection_origin",
            "latitude_of_projection_origin",
            "semi_major_axis",
            "semi_minor_axis",
        ):
            object.__setattr__(self, field, _finite_number(getattr(self, field), field))
        if self.perspective_point_height <= 0:
            raise ValueError("perspective_point_height must be positive")
        if self.semi_major_axis <= 0 or self.semi_minor_axis <= 0:
            raise ValueError("ellipsoid axes must be positive")
        if self.sweep_angle_axis not in {"x", "y"}:
            raise ValueError("sweep_angle_axis must be 'x' or 'y'")

    def crs(self) -> CRS:
        return CRS.from_cf(
            {
                "grid_mapping_name": "geostationary",
                "perspective_point_height": self.perspective_point_height,
                "longitude_of_projection_origin": self.longitude_of_projection_origin,
                "latitude_of_projection_origin": self.latitude_of_projection_origin,
                "semi_major_axis": self.semi_major_axis,
                "semi_minor_axis": self.semi_minor_axis,
                "sweep_angle_axis": self.sweep_angle_axis,
            }
        )

    def to_dict(self) -> dict[str, float | str]:
        return {
            "perspective_point_height": self.perspective_point_height,
            "longitude_of_projection_origin": self.longitude_of_projection_origin,
            "latitude_of_projection_origin": self.latitude_of_projection_origin,
            "semi_major_axis": self.semi_major_axis,
            "semi_minor_axis": self.semi_minor_axis,
            "sweep_angle_axis": self.sweep_angle_axis,
        }

    @classmethod
    def from_dict(cls, value: object) -> ProjectionMetadata:
        fields = {
            "perspective_point_height",
            "longitude_of_projection_origin",
            "latitude_of_projection_origin",
            "semi_major_axis",
            "semi_minor_axis",
            "sweep_angle_axis",
        }
        if not isinstance(value, dict) or set(value) != fields:
            raise CropArtifactError("projection metadata has an invalid shape")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class CalibrationMetadata:
    """Explicit packed-value calibration coefficients retained with the crop."""

    variable: str
    units: str
    fill_value: float | None
    valid_min: float | None
    valid_max: float | None
    coefficients: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.variable, str) or not self.variable:
            raise ValueError("variable must be a non-empty string")
        if not isinstance(self.units, str):
            raise ValueError("units must be a string")
        for field in ("fill_value", "valid_min", "valid_max"):
            value = getattr(self, field)
            if value is not None:
                object.__setattr__(self, field, _finite_number(value, field))
        if (
            self.valid_min is not None
            and self.valid_max is not None
            and self.valid_min > self.valid_max
        ):
            raise ValueError("valid_min must not exceed valid_max")
        coefficients = tuple(self.coefficients)
        if not coefficients or tuple(name for name, _ in coefficients) != tuple(
            sorted(name for name, _ in coefficients)
        ):
            raise ValueError("coefficients must be a non-empty sorted sequence")
        if len({name for name, _ in coefficients}) != len(coefficients):
            raise ValueError("coefficients must not repeat names")
        normalized: list[tuple[str, float]] = []
        for name, value in coefficients:
            if not isinstance(name, str) or not name:
                raise ValueError("coefficient names must be non-empty strings")
            normalized.append((name, _finite_number(value, f"coefficients.{name}")))
        object.__setattr__(self, "coefficients", tuple(normalized))

    @property
    def scale_factor(self) -> float:
        return dict(self.coefficients)["scale_factor"]

    @property
    def add_offset(self) -> float:
        return dict(self.coefficients)["add_offset"]

    def to_dict(self) -> dict[str, object]:
        return {
            "variable": self.variable,
            "units": self.units,
            "fill_value": self.fill_value,
            "valid_min": self.valid_min,
            "valid_max": self.valid_max,
            "coefficients": {name: value for name, value in self.coefficients},
        }

    @classmethod
    def from_dict(cls, value: object) -> CalibrationMetadata:
        fields = {
            "variable",
            "units",
            "fill_value",
            "valid_min",
            "valid_max",
            "coefficients",
        }
        if not isinstance(value, dict) or set(value) != fields:
            raise CropArtifactError("calibration metadata has an invalid shape")
        coefficients = value["coefficients"]
        if not isinstance(coefficients, dict):
            raise CropArtifactError("calibration coefficients must be an object")
        return cls(
            variable=value["variable"],
            units=value["units"],
            fill_value=value["fill_value"],
            valid_min=value["valid_min"],
            valid_max=value["valid_max"],
            coefficients=tuple(sorted(coefficients.items())),
        )


@dataclass(frozen=True, slots=True)
class QualityMetadata:
    """DQF interpretation used to form the invalid-pixel mask."""

    variable: str
    maximum_usable_dqf: int
    flag_values: tuple[int, ...]
    flag_meanings: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.variable, str) or not self.variable:
            raise ValueError("quality variable must be a non-empty string")
        object.__setattr__(
            self,
            "maximum_usable_dqf",
            _integer(self.maximum_usable_dqf, "maximum_usable_dqf"),
        )
        values = tuple(self.flag_values)
        if any(isinstance(item, bool) or not isinstance(item, int) for item in values):
            raise ValueError("flag_values must contain integers")
        object.__setattr__(self, "flag_values", values)
        if self.flag_meanings is not None and not isinstance(self.flag_meanings, str):
            raise ValueError("flag_meanings must be a string or None")

    def to_dict(self) -> dict[str, object]:
        return {
            "variable": self.variable,
            "maximum_usable_dqf": self.maximum_usable_dqf,
            "flag_values": list(self.flag_values),
            "flag_meanings": self.flag_meanings,
        }

    @classmethod
    def from_dict(cls, value: object) -> QualityMetadata:
        fields = {"variable", "maximum_usable_dqf", "flag_values", "flag_meanings"}
        if not isinstance(value, dict) or set(value) != fields:
            raise CropArtifactError("quality metadata has an invalid shape")
        flags = value["flag_values"]
        if not isinstance(flags, list):
            raise CropArtifactError("quality flag_values must be a list")
        return cls(
            variable=value["variable"],
            maximum_usable_dqf=value["maximum_usable_dqf"],
            flag_values=tuple(flags),
            flag_meanings=value["flag_meanings"],
        )


@dataclass(frozen=True, slots=True)
class CropPixel:
    """The nearest crop/source pixel for a reference geographic coordinate."""

    row: int
    column: int
    source_row: int
    source_column: int
    latitude: float
    longitude: float


@dataclass(frozen=True, slots=True)
class CalibratedCrop:
    """One calibrated, mask-aware regional GOES array and its complete provenance."""

    calibrated: FloatArray
    invalid_mask: MaskArray
    dqf: QualityArray
    latitude: CoordinateArray
    longitude: CoordinateArray
    x: CoordinateArray
    y: CoordinateArray
    requested_bounds: GeographicBounds
    geographic_bounds: GeographicBounds
    parameters: CropParameters
    source_window: SourceWindow
    timing: ScanTiming
    projection: ProjectionMetadata
    calibration: CalibrationMetadata
    quality: QualityMetadata
    source_checksum: str
    source_size_bytes: int
    content_checksum: str

    def __post_init__(self) -> None:
        calibrated = np.asarray(self.calibrated, dtype=np.float32)
        invalid_mask = np.asarray(self.invalid_mask, dtype=bool)
        dqf = np.asarray(self.dqf)
        latitude = np.asarray(self.latitude, dtype=np.float64)
        longitude = np.asarray(self.longitude, dtype=np.float64)
        x = np.asarray(self.x, dtype=np.float64)
        y = np.asarray(self.y, dtype=np.float64)
        if calibrated.ndim != 2:
            raise ValueError("calibrated must be a two-dimensional array")
        if any(
            array.shape != calibrated.shape
            for array in (invalid_mask, dqf, latitude, longitude)
        ):
            raise ValueError("crop arrays must share the calibrated array shape")
        if x.ndim != 1 or y.ndim != 1 or (y.size, x.size) != calibrated.shape:
            raise ValueError("x/y coordinates must match the calibrated array shape")
        if self.source_window.shape != calibrated.shape:
            raise ValueError("source_window must match the calibrated array shape")
        if not isinstance(self.requested_bounds, GeographicBounds) or not isinstance(
            self.geographic_bounds, GeographicBounds
        ):
            raise ValueError("crop bounds must be GeographicBounds")
        for field, record_type in (
            ("parameters", CropParameters),
            ("source_window", SourceWindow),
            ("timing", ScanTiming),
            ("projection", ProjectionMetadata),
            ("calibration", CalibrationMetadata),
            ("quality", QualityMetadata),
        ):
            if not isinstance(getattr(self, field), record_type):
                raise ValueError(f"{field} must be {record_type.__name__}")
        if self.parameters.bounds != self.requested_bounds:
            raise ValueError("parameters.bounds must equal requested_bounds")
        if self.quality.maximum_usable_dqf != self.parameters.maximum_usable_dqf:
            raise ValueError("quality DQF threshold must equal crop parameters")
        object.__setattr__(self, "calibrated", _readonly(calibrated))
        object.__setattr__(self, "invalid_mask", _readonly(invalid_mask))
        object.__setattr__(self, "dqf", _readonly(dqf))
        object.__setattr__(self, "latitude", _readonly(latitude))
        object.__setattr__(self, "longitude", _readonly(longitude))
        object.__setattr__(self, "x", _readonly(x))
        object.__setattr__(self, "y", _readonly(y))
        object.__setattr__(
            self, "source_checksum", _sha256(self.source_checksum, "source_checksum")
        )
        object.__setattr__(
            self,
            "source_size_bytes",
            _integer(self.source_size_bytes, "source_size_bytes"),
        )
        object.__setattr__(
            self, "content_checksum", _sha256(self.content_checksum, "content_checksum")
        )
        if self.content_checksum != self.compute_checksum():
            raise ValueError("content_checksum does not match crop contents")

    @property
    def valid_mask(self) -> MaskArray:
        """Return the immutable complement of ``invalid_mask``."""
        return _readonly(np.logical_not(self.invalid_mask))  # type: ignore[return-value]

    def nearest_pixel(self, latitude: float, longitude: float) -> CropPixel:
        """Return the deterministic nearest saved pixel for a reference coordinate."""
        target_latitude = _finite_number(latitude, "latitude")
        target_longitude = _finite_number(longitude, "longitude")
        distance = (self.latitude - target_latitude) ** 2 + (
            self.longitude - target_longitude
        ) ** 2
        distance[~np.isfinite(distance)] = np.inf
        if not np.isfinite(distance).any():
            raise RegionOutsideSourceError(
                "crop contains no finite geographic coordinates"
            )
        row, column = np.unravel_index(int(np.argmin(distance)), distance.shape)
        return CropPixel(
            row=int(row),
            column=int(column),
            source_row=self.source_window.row_start + int(row),
            source_column=self.source_window.column_start + int(column),
            latitude=float(self.latitude[row, column]),
            longitude=float(self.longitude[row, column]),
        )

    def _metadata(self, *, include_checksum: bool) -> dict[str, object]:
        metadata: dict[str, object] = {
            "schema_version": _SCHEMA_VERSION,
            "requested_bounds": self.requested_bounds.to_dict(),
            "geographic_bounds": self.geographic_bounds.to_dict(),
            "crop_parameters": self.parameters.to_dict(),
            "source_window": self.source_window.to_dict(),
            "timing": self.timing.to_dict(),
            "projection": self.projection.to_dict(),
            "calibration": self.calibration.to_dict(),
            "quality": self.quality.to_dict(),
            "source_checksum": self.source_checksum,
            "source_size_bytes": self.source_size_bytes,
            "arrays": {
                "calibrated": {
                    "dtype": self.calibrated.dtype.str,
                    "shape": list(self.calibrated.shape),
                },
                "invalid_mask": {
                    "dtype": self.invalid_mask.dtype.str,
                    "shape": list(self.invalid_mask.shape),
                },
                "dqf": {"dtype": self.dqf.dtype.str, "shape": list(self.dqf.shape)},
                "latitude": {
                    "dtype": self.latitude.dtype.str,
                    "shape": list(self.latitude.shape),
                },
                "longitude": {
                    "dtype": self.longitude.dtype.str,
                    "shape": list(self.longitude.shape),
                },
                "x": {"dtype": self.x.dtype.str, "shape": list(self.x.shape)},
                "y": {"dtype": self.y.dtype.str, "shape": list(self.y.shape)},
            },
        }
        if include_checksum:
            metadata["content_checksum"] = self.content_checksum
        return metadata

    def metadata(self) -> dict[str, object]:
        """Return JSON-safe artifact metadata, including the verified content hash."""
        return self._metadata(include_checksum=True)

    def compute_checksum(self) -> str:
        """Hash canonical metadata and stored arrays, not ZIP timestamps or paths."""
        return _content_checksum(self._metadata(include_checksum=False), self._arrays())

    def _arrays(self) -> dict[str, npt.NDArray[np.generic]]:
        return {
            "calibrated": self.calibrated,
            "invalid_mask": self.invalid_mask,
            "dqf": self.dqf,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "x": self.x,
            "y": self.y,
        }


def _projection_metadata(dataset: Any, data_variable: Any) -> ProjectionMetadata:
    projection_name = _attribute(data_variable, "grid_mapping")
    if not isinstance(projection_name, str) or projection_name not in dataset.variables:
        raise GoesCropFormatError("CMI grid_mapping must name a projection variable")
    projection_variable = dataset.variables[projection_name]
    mapping_name = _attribute(projection_variable, "grid_mapping_name")
    if mapping_name != "geostationary":
        raise GoesCropFormatError("grid_mapping_name must be 'geostationary'")
    sweep = _attribute(projection_variable, "sweep_angle_axis")
    if not isinstance(sweep, str):
        raise GoesCropFormatError("sweep_angle_axis must be a string")
    height = _numeric_attribute(projection_variable, "perspective_point_height")
    longitude = _numeric_attribute(
        projection_variable, "longitude_of_projection_origin"
    )
    latitude = _numeric_attribute(projection_variable, "latitude_of_projection_origin")
    semi_major_axis = _numeric_attribute(projection_variable, "semi_major_axis")
    semi_minor_axis = _numeric_attribute(projection_variable, "semi_minor_axis")
    assert (
        height is not None
        and longitude is not None
        and latitude is not None
        and semi_major_axis is not None
        and semi_minor_axis is not None
    )
    try:
        return ProjectionMetadata(
            perspective_point_height=height,
            longitude_of_projection_origin=longitude,
            latitude_of_projection_origin=latitude,
            semi_major_axis=semi_major_axis,
            semi_minor_axis=semi_minor_axis,
            sweep_angle_axis=sweep,
        )
    except ValueError as error:
        raise GoesCropFormatError(str(error)) from error


def _coordinate_variables(dataset: Any, data_variable: Any) -> tuple[Any, Any]:
    if len(data_variable.dimensions) != 2:
        raise GoesCropFormatError("CMI must have exactly y/x dimensions")
    y_name, x_name = data_variable.dimensions
    if y_name not in dataset.variables or x_name not in dataset.variables:
        raise GoesCropFormatError("CMI dimensions must have coordinate variables")
    y_variable = dataset.variables[y_name]
    x_variable = dataset.variables[x_name]
    if y_variable.dimensions != (y_name,) or x_variable.dimensions != (x_name,):
        raise GoesCropFormatError("CMI coordinate variables must be one-dimensional")
    return y_variable, x_variable


def _scan_coordinates(variable: Any, expected_size: int, axis: str) -> CoordinateArray:
    values = np.asarray(_read_unscaled(variable), dtype=np.float64)
    if (
        values.ndim != 1
        or values.size != expected_size
        or not np.isfinite(values).all()
    ):
        raise GoesCropFormatError(
            f"{axis} scan coordinates must be finite and one-dimensional"
        )
    differences = np.diff(values)
    if not (np.all(differences > 0) or np.all(differences < 0)):
        raise GoesCropFormatError(f"{axis} scan coordinates must be strictly monotonic")
    units = _attribute(variable, "units", required=False)
    if units is not None and not isinstance(units, str):
        raise GoesCropFormatError(f"{axis} coordinate units must be a string")
    normalized = "rad" if units is None else units.lower().strip()
    if normalized not in {"rad", "radian", "radians"}:
        raise GoesCropFormatError(f"{axis} scan coordinates must use radians")
    return values


def _calibration_metadata(variable: Any) -> CalibrationMetadata:
    scale = _numeric_attribute(variable, "scale_factor")
    offset = _numeric_attribute(variable, "add_offset")
    assert scale is not None and offset is not None
    fill = _numeric_attribute(variable, "_FillValue", required=False)
    valid_min = _numeric_attribute(variable, "valid_min", required=False)
    valid_max = _numeric_attribute(variable, "valid_max", required=False)
    valid_range = _attribute(variable, "valid_range", required=False)
    if valid_range is not None:
        range_values = np.asarray(valid_range).reshape(-1)
        if range_values.size != 2:
            raise GoesCropFormatError("CMI valid_range must contain two values")
        try:
            range_min = _finite_number(range_values[0], "CMI.valid_range[0]")
            range_max = _finite_number(range_values[1], "CMI.valid_range[1]")
        except ValueError as error:
            raise GoesCropFormatError(str(error)) from error
        if valid_min is not None and valid_min != range_min:
            raise GoesCropFormatError("CMI valid_min conflicts with valid_range")
        if valid_max is not None and valid_max != range_max:
            raise GoesCropFormatError("CMI valid_max conflicts with valid_range")
        valid_min, valid_max = range_min, range_max
    coefficients: dict[str, float] = {"add_offset": offset, "scale_factor": scale}
    for name in _CALIBRATION_ATTRIBUTES[2:]:
        coefficient = _numeric_attribute(variable, name, required=False)
        if coefficient is not None:
            coefficients[name] = coefficient
    units = _attribute(variable, "units", required=False)
    if not isinstance(units, str):
        raise GoesCropFormatError("CMI units must be a string")
    try:
        return CalibrationMetadata(
            variable=variable.name,
            units=units,
            fill_value=fill,
            valid_min=valid_min,
            valid_max=valid_max,
            coefficients=tuple(sorted(coefficients.items())),
        )
    except ValueError as error:
        raise GoesCropFormatError(str(error)) from error


def _quality_metadata(variable: Any, maximum_usable_dqf: int) -> QualityMetadata:
    flag_values_attribute = _attribute(variable, "flag_values", required=False)
    if flag_values_attribute is None:
        flag_values: tuple[int, ...] = ()
    else:
        values = np.asarray(flag_values_attribute).reshape(-1)
        try:
            flag_values = tuple(int(value) for value in values)
        except (TypeError, ValueError, OverflowError) as error:
            raise GoesCropFormatError(
                "DQF flag_values must contain integers"
            ) from error
    meanings = _attribute(variable, "flag_meanings", required=False)
    if meanings is not None and not isinstance(meanings, str):
        raise GoesCropFormatError("DQF flag_meanings must be a string")
    try:
        return QualityMetadata(
            variable=variable.name,
            maximum_usable_dqf=maximum_usable_dqf,
            flag_values=flag_values,
            flag_meanings=meanings,
        )
    except ValueError as error:
        raise GoesCropFormatError(str(error)) from error


def _timing(dataset: Any) -> ScanTiming:
    start = _attribute(dataset, "time_coverage_start", required=False)
    end = _attribute(dataset, "time_coverage_end", required=False)
    if (start is None) != (end is None):
        raise GoesCropFormatError(
            "time_coverage_start and time_coverage_end must be present together"
        )
    if start is not None:
        try:
            return ScanTiming(
                start=_parse_timestamp(start, "time_coverage_start"),
                end=_parse_timestamp(end, "time_coverage_end"),
            )
        except ValueError as error:
            raise GoesCropFormatError(str(error)) from error
    if "t" not in dataset.variables and "time" not in dataset.variables:
        raise GoesCropFormatError("source lacks time coverage metadata")
    time_variable = dataset.variables.get("t", dataset.variables.get("time"))
    assert time_variable is not None
    units = _attribute(time_variable, "units")
    calendar = _attribute(time_variable, "calendar", required=False) or "standard"
    values = np.asarray(_read_unscaled(time_variable)).reshape(-1)
    if values.size != 1:
        raise GoesCropFormatError(
            "fallback time variable must contain exactly one value"
        )
    try:
        decoded = num2date(
            values[0],
            units=units,
            calendar=calendar,
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=True,
        )
        if not isinstance(decoded, datetime):
            raise GoesCropFormatError("fallback time variable is not a datetime")
        return ScanTiming(start=decoded, end=decoded)
    except (TypeError, ValueError, OverflowError) as error:
        raise GoesCropFormatError("could not decode source time coordinate") from error


def _edge_points(
    bounds: GeographicBounds, count: int
) -> tuple[CoordinateArray, CoordinateArray]:
    longitudes = np.linspace(bounds.west, bounds.east, count, dtype=np.float64)
    latitudes = np.linspace(bounds.south, bounds.north, count, dtype=np.float64)
    return (
        np.concatenate(
            (
                longitudes,
                longitudes,
                np.full(count, bounds.west),
                np.full(count, bounds.east),
            )
        ),
        np.concatenate(
            (
                np.full(count, bounds.south),
                np.full(count, bounds.north),
                latitudes,
                latitudes,
            )
        ),
    )


def _nearest_indices(
    coordinates: CoordinateArray, requested: CoordinateArray
) -> npt.NDArray[np.intp]:
    ascending = coordinates[0] < coordinates[-1]
    ordered = coordinates if ascending else coordinates[::-1]
    positions = np.searchsorted(ordered, requested, side="left")
    positions = np.clip(positions, 1, ordered.size - 1)
    before = ordered[positions - 1]
    after = ordered[positions]
    selected = np.where(
        np.abs(requested - before) <= np.abs(after - requested),
        positions - 1,
        positions,
    )
    if ascending:
        return selected.astype(np.intp)
    return (coordinates.size - 1 - selected).astype(np.intp)


def _source_window(
    parameters: CropParameters,
    projection: ProjectionMetadata,
    x: CoordinateArray,
    y: CoordinateArray,
) -> SourceWindow:
    longitudes, latitudes = _edge_points(parameters.bounds, parameters.edge_samples)
    transformer = Transformer.from_crs("EPSG:4326", projection.crs(), always_xy=True)
    projected_x, projected_y = transformer.transform(
        longitudes, latitudes, errcheck=False
    )
    requested_x = (
        np.asarray(projected_x, dtype=np.float64) / projection.perspective_point_height
    )
    requested_y = (
        np.asarray(projected_y, dtype=np.float64) / projection.perspective_point_height
    )
    finite = np.isfinite(requested_x) & np.isfinite(requested_y)
    if not finite.any():
        raise RegionOutsideSourceError(
            "requested bounds are outside the GOES projection"
        )
    requested_x = requested_x[finite]
    requested_y = requested_y[finite]
    x_indices = _nearest_indices(x, requested_x)
    y_indices = _nearest_indices(y, requested_y)
    padding = parameters.padding_pixels
    row_start = max(0, int(y_indices.min()) - padding)
    row_stop = min(y.size, int(y_indices.max()) + padding + 1)
    column_start = max(0, int(x_indices.min()) - padding)
    column_stop = min(x.size, int(x_indices.max()) + padding + 1)
    return SourceWindow(row_start, row_stop, column_start, column_stop)


def _coordinate_grids(
    x: CoordinateArray, y: CoordinateArray, projection: ProjectionMetadata
) -> tuple[CoordinateArray, CoordinateArray]:
    x_grid, y_grid = np.meshgrid(
        x * projection.perspective_point_height,
        y * projection.perspective_point_height,
        indexing="xy",
    )
    transformer = Transformer.from_crs(projection.crs(), "EPSG:4326", always_xy=True)
    longitude, latitude = transformer.transform(x_grid, y_grid, errcheck=False)
    return np.asarray(latitude, dtype=np.float64), np.asarray(
        longitude, dtype=np.float64
    )


def _actual_bounds(
    latitude: CoordinateArray, longitude: CoordinateArray
) -> GeographicBounds:
    finite = np.isfinite(latitude) & np.isfinite(longitude)
    if not finite.any():
        raise RegionOutsideSourceError(
            "selected crop contains no valid geographic locations"
        )
    return GeographicBounds(
        south=float(np.min(latitude[finite])),
        west=float(np.min(longitude[finite])),
        north=float(np.max(latitude[finite])),
        east=float(np.max(longitude[finite])),
    )


def _source_digest(path: Path) -> tuple[str, int]:
    try:
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        return digest, path.stat().st_size
    except OSError as error:
        raise GoesCropError(f"cannot read cached GOES source {path}") from error


def _array_payload(array: npt.NDArray[np.generic]) -> bytes:
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, np.ascontiguousarray(array), allow_pickle=False)
    return buffer.getvalue()


def _content_checksum(
    metadata: dict[str, object], arrays: dict[str, npt.NDArray[np.generic]]
) -> str:
    digest = hashlib.sha256()
    digest.update(b"firesentinel-goes-crop-v1\0metadata\0")
    digest.update(
        json.dumps(
            metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    )
    for name in sorted(arrays):
        digest.update(b"\0array\0")
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
        digest.update(_array_payload(arrays[name]))
    return digest.hexdigest()


def extract_calibrated_crop(
    source_path: Path,
    parameters: CropParameters,
    *,
    data_variable_name: str = "CMI",
    quality_variable_name: str = "DQF",
) -> CalibratedCrop:
    """Decode one cached ABI CMIPF object into a clipped, padded calibrated crop.

    ``CMI`` packed values are calibrated as ``raw * scale_factor + add_offset``.
    Invalid pixels include fill, non-finite, outside-valid-range, unusable DQF,
    and off-Earth locations.  Invalid calibrated samples are always ``NaN``;
    callers should preserve ``invalid_mask`` alongside the values.
    """
    if not isinstance(parameters, CropParameters):
        raise TypeError("parameters must be CropParameters")
    if not isinstance(data_variable_name, str) or not data_variable_name:
        raise ValueError("data_variable_name must be a non-empty string")
    if not isinstance(quality_variable_name, str) or not quality_variable_name:
        raise ValueError("quality_variable_name must be a non-empty string")
    source = Path(source_path)
    source_checksum, source_size_bytes = _source_digest(source)
    try:
        with Dataset(source, mode="r") as dataset:
            if data_variable_name not in dataset.variables:
                raise GoesCropFormatError(
                    f"source lacks calibrated data variable {data_variable_name!r}"
                )
            if quality_variable_name not in dataset.variables:
                raise GoesCropFormatError(
                    f"source lacks data-quality variable {quality_variable_name!r}"
                )
            data_variable = dataset.variables[data_variable_name]
            quality_variable = dataset.variables[quality_variable_name]
            projection = _projection_metadata(dataset, data_variable)
            y_variable, x_variable = _coordinate_variables(dataset, data_variable)
            raw = _read_unscaled(data_variable)
            if raw.ndim != 2:
                raise GoesCropFormatError("CMI must be two-dimensional")
            x = _scan_coordinates(x_variable, raw.shape[1], "x")
            y = _scan_coordinates(y_variable, raw.shape[0], "y")
            calibration = _calibration_metadata(data_variable)
            if quality_variable.dimensions != data_variable.dimensions:
                raise GoesCropFormatError("DQF dimensions must match CMI dimensions")
            dqf_raw = _read_unscaled(quality_variable)
            if dqf_raw.shape != raw.shape:
                raise GoesCropFormatError("DQF shape must match CMI shape")
            quality = _quality_metadata(quality_variable, parameters.maximum_usable_dqf)
            timing = _timing(dataset)
            window = _source_window(parameters, projection, x, y)
            rows = slice(window.row_start, window.row_stop)
            columns = slice(window.column_start, window.column_stop)
            crop_raw = np.asarray(raw[rows, columns])
            crop_dqf = np.asarray(dqf_raw[rows, columns])
            crop_x = x[columns]
            crop_y = y[rows]
    except OSError as error:
        raise GoesCropError(f"cannot open cached GOES source {source}") from error

    calibrated = (
        crop_raw.astype(np.float64) * calibration.scale_factor + calibration.add_offset
    ).astype(np.float32)
    invalid = ~np.isfinite(crop_raw) | ~np.isfinite(calibrated)
    if calibration.fill_value is not None:
        invalid |= crop_raw == calibration.fill_value
    if calibration.valid_min is not None:
        invalid |= crop_raw < calibration.valid_min
    if calibration.valid_max is not None:
        invalid |= crop_raw > calibration.valid_max
    dqf_numeric = crop_dqf.astype(np.float64)
    invalid |= ~np.isfinite(dqf_numeric) | (dqf_numeric > quality.maximum_usable_dqf)
    latitude, longitude = _coordinate_grids(crop_x, crop_y, projection)
    invalid |= ~np.isfinite(latitude) | ~np.isfinite(longitude)
    calibrated[invalid] = np.nan
    geographic_bounds = _actual_bounds(latitude, longitude)
    arrays: dict[str, npt.NDArray[np.generic]] = {
        "calibrated": calibrated,
        "invalid_mask": invalid,
        "dqf": crop_dqf,
        "latitude": latitude,
        "longitude": longitude,
        "x": crop_x,
        "y": crop_y,
    }
    metadata = {
        "schema_version": _SCHEMA_VERSION,
        "requested_bounds": parameters.bounds.to_dict(),
        "geographic_bounds": geographic_bounds.to_dict(),
        "crop_parameters": parameters.to_dict(),
        "source_window": window.to_dict(),
        "timing": timing.to_dict(),
        "projection": projection.to_dict(),
        "calibration": calibration.to_dict(),
        "quality": quality.to_dict(),
        "source_checksum": source_checksum,
        "source_size_bytes": source_size_bytes,
        "arrays": {
            name: {"dtype": array.dtype.str, "shape": list(array.shape)}
            for name, array in arrays.items()
        },
    }
    checksum = _content_checksum(metadata, arrays)
    return CalibratedCrop(
        calibrated=calibrated,
        invalid_mask=invalid,
        dqf=crop_dqf,
        latitude=latitude,
        longitude=longitude,
        x=crop_x,
        y=crop_y,
        requested_bounds=parameters.bounds,
        geographic_bounds=geographic_bounds,
        parameters=parameters,
        source_window=window,
        timing=timing,
        projection=projection,
        calibration=calibration,
        quality=quality,
        source_checksum=source_checksum,
        source_size_bytes=source_size_bytes,
        content_checksum=checksum,
    )


def save_calibrated_crop(crop: CalibratedCrop, destination: Path) -> Path:
    """Atomically write a byte-stable ``.npz`` archive and return its path.

    ZIP member timestamps and ordering are fixed, so writing an identical crop
    twice produces identical artifact bytes as well as the same content hash.
    """
    if not isinstance(crop, CalibratedCrop):
        raise TypeError("crop must be CalibratedCrop")
    path = Path(destination)
    if path.suffix.lower() != ".npz":
        raise ValueError("destination must have a .npz suffix")
    metadata = json.dumps(
        crop.metadata(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    payloads = {
        "calibrated.npy": _array_payload(crop.calibrated),
        "dqf.npy": _array_payload(crop.dqf),
        "invalid_mask.npy": _array_payload(crop.invalid_mask),
        "latitude.npy": _array_payload(crop.latitude),
        "longitude.npy": _array_payload(crop.longitude),
        "metadata.json": metadata,
        "x.npy": _array_payload(crop.x),
        "y.npy": _array_payload(crop.y),
    }
    temporary_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(delete=False, dir=path.parent) as temporary:
            temporary_path = Path(temporary.name)
        with zipfile.ZipFile(
            temporary_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for name in _ARCHIVE_NAMES:
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o600 << 16
                archive.writestr(
                    info,
                    payloads[name],
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=9,
                )
        os.replace(temporary_path, path)
    except OSError as error:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise CropArtifactError(f"cannot save crop artifact {path}") from error
    return path


def _archive_array(archive: zipfile.ZipFile, name: str) -> npt.NDArray[np.generic]:
    try:
        with archive.open(name) as member:
            return cast(
                npt.NDArray[np.generic],
                np.load(io.BytesIO(member.read()), allow_pickle=False),
            )
    except (KeyError, OSError, ValueError) as error:
        raise CropArtifactError(f"crop archive has invalid {name}") from error


def load_calibrated_crop(path: Path) -> CalibratedCrop:
    """Load a persisted crop and verify its canonical content checksum."""
    source = Path(path)
    try:
        with zipfile.ZipFile(source, "r") as archive:
            if tuple(sorted(archive.namelist())) != _ARCHIVE_NAMES:
                raise CropArtifactError("crop archive has unexpected members")
            try:
                metadata = json.loads(archive.read("metadata.json").decode("utf-8"))
            except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as error:
                raise CropArtifactError(
                    "crop archive has invalid metadata.json"
                ) from error
            arrays = {
                "calibrated": _archive_array(archive, "calibrated.npy"),
                "dqf": _archive_array(archive, "dqf.npy"),
                "invalid_mask": _archive_array(archive, "invalid_mask.npy"),
                "latitude": _archive_array(archive, "latitude.npy"),
                "longitude": _archive_array(archive, "longitude.npy"),
                "x": _archive_array(archive, "x.npy"),
                "y": _archive_array(archive, "y.npy"),
            }
    except (OSError, zipfile.BadZipFile) as error:
        raise CropArtifactError(f"cannot read crop artifact {source}") from error
    fields = {
        "schema_version",
        "requested_bounds",
        "geographic_bounds",
        "crop_parameters",
        "source_window",
        "timing",
        "projection",
        "calibration",
        "quality",
        "source_checksum",
        "source_size_bytes",
        "content_checksum",
        "arrays",
    }
    if not isinstance(metadata, dict) or set(metadata) != fields:
        raise CropArtifactError("crop metadata has an invalid shape")
    if metadata["schema_version"] != _SCHEMA_VERSION:
        raise CropArtifactError("crop metadata uses an unsupported schema")
    array_metadata = metadata["arrays"]
    if not isinstance(array_metadata, dict) or set(array_metadata) != set(arrays):
        raise CropArtifactError("crop array metadata has an invalid shape")
    for name, array in arrays.items():
        expected = array_metadata[name]
        if not isinstance(expected, dict) or set(expected) != {"dtype", "shape"}:
            raise CropArtifactError(f"crop metadata for {name} has an invalid shape")
        if expected["dtype"] != array.dtype.str or expected["shape"] != list(
            array.shape
        ):
            raise CropArtifactError(f"crop array {name} does not match its metadata")
    try:
        return CalibratedCrop(
            calibrated=np.asarray(arrays["calibrated"], dtype=np.float32),
            invalid_mask=np.asarray(arrays["invalid_mask"], dtype=bool),
            dqf=arrays["dqf"],
            latitude=np.asarray(arrays["latitude"], dtype=np.float64),
            longitude=np.asarray(arrays["longitude"], dtype=np.float64),
            x=np.asarray(arrays["x"], dtype=np.float64),
            y=np.asarray(arrays["y"], dtype=np.float64),
            requested_bounds=GeographicBounds.from_dict(metadata["requested_bounds"]),
            geographic_bounds=GeographicBounds.from_dict(metadata["geographic_bounds"]),
            parameters=CropParameters.from_dict(metadata["crop_parameters"]),
            source_window=SourceWindow.from_dict(metadata["source_window"]),
            timing=ScanTiming.from_dict(metadata["timing"]),
            projection=ProjectionMetadata.from_dict(metadata["projection"]),
            calibration=CalibrationMetadata.from_dict(metadata["calibration"]),
            quality=QualityMetadata.from_dict(metadata["quality"]),
            source_checksum=metadata["source_checksum"],
            source_size_bytes=metadata["source_size_bytes"],
            content_checksum=metadata["content_checksum"],
        )
    except (TypeError, ValueError, CropArtifactError) as error:
        raise CropArtifactError("crop metadata has invalid contents") from error


__all__ = [
    "CalibratedCrop",
    "CalibrationMetadata",
    "CropArtifactError",
    "CropParameters",
    "CropPixel",
    "GeographicBounds",
    "GeographicBoundsError",
    "GoesCropError",
    "GoesCropFormatError",
    "ProjectionMetadata",
    "QualityMetadata",
    "RegionOutsideSourceError",
    "ScanTiming",
    "SourceWindow",
    "extract_calibrated_crop",
    "load_calibrated_crop",
    "save_calibrated_crop",
]
