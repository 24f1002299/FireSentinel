"""Mask-aware conversion of calibrated thermal data into stable OpenCV tiles.

The preparation boundary deliberately keeps calibrated values and display pixels
separate.  Physical clipping is applied only to the derived analysis tile;
the source calibration remains immutable for measurements and provenance.
Invalid source samples are carried through every resize as a mask and are
always black in display-only products, never candidate evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from typing import Final

import cv2
import numpy as np
import numpy.typing as npt

from firesentinel.data.goes_crop import CalibratedCrop

FloatArray = npt.NDArray[np.float32]
MaskArray = npt.NDArray[np.bool]
Uint8Array = npt.NDArray[np.uint8]
_SCHEMA_VERSION: Final = 1
_OPEN_CV_BUILD_INFORMATION = cv2.getBuildInformation()


@dataclass(frozen=True, slots=True)
class TilePreparationParameters:
    """Explicit physical, display, resize, and optional CLAHE settings."""

    physical_minimum_kelvin: float
    physical_maximum_kelvin: float
    display_lower_quantile: float = 0.02
    display_upper_quantile: float = 0.98
    target_shape: tuple[int, int] | None = None
    minimum_valid_coverage: float = 1.0
    clahe_clip_limit: float | None = None
    clahe_tile_grid_size: tuple[int, int] = (8, 8)

    def __post_init__(self) -> None:
        minimum = _finite_number(
            self.physical_minimum_kelvin, "physical_minimum_kelvin"
        )
        maximum = _finite_number(
            self.physical_maximum_kelvin, "physical_maximum_kelvin"
        )
        if maximum <= minimum:
            raise ValueError(
                "physical_maximum_kelvin must exceed physical_minimum_kelvin"
            )
        lower_quantile = _finite_number(
            self.display_lower_quantile, "display_lower_quantile"
        )
        upper_quantile = _finite_number(
            self.display_upper_quantile, "display_upper_quantile"
        )
        if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
            raise ValueError("display quantiles must satisfy 0 <= lower < upper <= 1")
        coverage = _finite_number(self.minimum_valid_coverage, "minimum_valid_coverage")
        if not 0.0 < coverage <= 1.0:
            raise ValueError("minimum_valid_coverage must be within (0, 1]")
        if self.target_shape is not None:
            _shape(self.target_shape, "target_shape")
        if self.clahe_clip_limit is not None:
            clip_limit = _finite_number(self.clahe_clip_limit, "clahe_clip_limit")
            if clip_limit <= 0.0:
                raise ValueError("clahe_clip_limit must be positive when supplied")
        _shape(self.clahe_tile_grid_size, "clahe_tile_grid_size")
        object.__setattr__(self, "physical_minimum_kelvin", minimum)
        object.__setattr__(self, "physical_maximum_kelvin", maximum)
        object.__setattr__(self, "display_lower_quantile", lower_quantile)
        object.__setattr__(self, "display_upper_quantile", upper_quantile)
        object.__setattr__(self, "minimum_valid_coverage", coverage)

    def to_dict(self) -> dict[str, object]:
        """Return a stable, JSON-safe parameter record."""

        return {
            "physical_minimum_kelvin": self.physical_minimum_kelvin,
            "physical_maximum_kelvin": self.physical_maximum_kelvin,
            "display_lower_quantile": self.display_lower_quantile,
            "display_upper_quantile": self.display_upper_quantile,
            "target_shape": None
            if self.target_shape is None
            else list(self.target_shape),
            "minimum_valid_coverage": self.minimum_valid_coverage,
            "resize_interpolation": "INTER_AREA downsample / INTER_LINEAR upsample",
            "clahe_clip_limit": self.clahe_clip_limit,
            "clahe_tile_grid_size": list(self.clahe_tile_grid_size),
        }


@dataclass(frozen=True, slots=True)
class PreparedTile:
    """Calibrated data, masks, and display-only OpenCV inputs for one tile."""

    calibrated: FloatArray
    input_invalid_mask: MaskArray
    effective_invalid_mask: MaskArray
    physical_clipped: FloatArray
    physical_low_clip_mask: MaskArray
    physical_high_clip_mask: MaskArray
    resized_calibrated: FloatArray
    resized_invalid_mask: MaskArray
    display: Uint8Array
    clahe_display: Uint8Array | None
    parameters: TilePreparationParameters
    source_crop_checksum: str | None
    source_timing: dict[str, str] | None
    input_valid_range: tuple[float, float] | None
    display_range: tuple[float, float]
    timings_milliseconds: dict[str, float]
    content_checksum: str

    def __post_init__(self) -> None:
        calibrated = np.asarray(self.calibrated, dtype=np.float32)
        input_invalid = np.asarray(self.input_invalid_mask, dtype=bool)
        effective_invalid = np.asarray(self.effective_invalid_mask, dtype=bool)
        physical = np.asarray(self.physical_clipped, dtype=np.float32)
        low_clipped = np.asarray(self.physical_low_clip_mask, dtype=bool)
        high_clipped = np.asarray(self.physical_high_clip_mask, dtype=bool)
        resized = np.asarray(self.resized_calibrated, dtype=np.float32)
        resized_invalid = np.asarray(self.resized_invalid_mask, dtype=bool)
        display = np.asarray(self.display, dtype=np.uint8)
        clahe = (
            None
            if self.clahe_display is None
            else np.asarray(self.clahe_display, dtype=np.uint8)
        )
        if calibrated.ndim != 2:
            raise ValueError("calibrated must be a two-dimensional array")
        if any(
            array.shape != calibrated.shape
            for array in (
                input_invalid,
                effective_invalid,
                physical,
                low_clipped,
                high_clipped,
            )
        ):
            raise ValueError("input tile arrays must share the calibrated shape")
        if resized.ndim != 2 or any(
            array.shape != resized.shape for array in (resized_invalid, display)
        ):
            raise ValueError("resized values, mask, and display must share a 2D shape")
        if clahe is not None and clahe.shape != display.shape:
            raise ValueError("clahe_display must share the display shape")
        if not isinstance(self.parameters, TilePreparationParameters):
            raise ValueError("parameters must be TilePreparationParameters")
        if (
            self.parameters.target_shape is not None
            and resized.shape != self.parameters.target_shape
        ):
            raise ValueError("resized tile shape must equal parameters.target_shape")
        if np.any(np.isfinite(physical[effective_invalid])):
            raise ValueError("invalid input pixels must be NaN in physical_clipped")
        if np.any(np.isfinite(resized[resized_invalid])):
            raise ValueError("invalid resized pixels must be NaN")
        if np.any(display[resized_invalid] != 0):
            raise ValueError("invalid resized pixels must be black in display")
        if clahe is not None and np.any(clahe[resized_invalid] != 0):
            raise ValueError("invalid resized pixels must be black in clahe_display")
        if np.any(np.logical_and(low_clipped, high_clipped)):
            raise ValueError("a pixel cannot be clipped both low and high")
        if np.any(np.logical_and(low_clipped | high_clipped, effective_invalid)):
            raise ValueError("invalid input pixels cannot be counted as physical clips")
        _range(self.display_range, "display_range")
        if self.input_valid_range is not None:
            _range(self.input_valid_range, "input_valid_range", allow_equal=True)
        if self.source_crop_checksum is not None:
            _sha256_text(self.source_crop_checksum, "source_crop_checksum")
        if self.source_timing is not None and set(self.source_timing) != {
            "start",
            "end",
        }:
            raise ValueError("source_timing must contain start and end")
        if any(
            not isinstance(name, str)
            or not isinstance(duration, float)
            or duration < 0.0
            or not math.isfinite(duration)
            for name, duration in self.timings_milliseconds.items()
        ):
            raise ValueError(
                "timings_milliseconds must contain finite non-negative floats"
            )
        object.__setattr__(self, "calibrated", _readonly(calibrated))
        object.__setattr__(self, "input_invalid_mask", _readonly(input_invalid))
        object.__setattr__(self, "effective_invalid_mask", _readonly(effective_invalid))
        object.__setattr__(self, "physical_clipped", _readonly(physical))
        object.__setattr__(self, "physical_low_clip_mask", _readonly(low_clipped))
        object.__setattr__(self, "physical_high_clip_mask", _readonly(high_clipped))
        object.__setattr__(self, "resized_calibrated", _readonly(resized))
        object.__setattr__(self, "resized_invalid_mask", _readonly(resized_invalid))
        object.__setattr__(self, "display", _readonly(display))
        object.__setattr__(
            self, "clahe_display", None if clahe is None else _readonly(clahe)
        )
        object.__setattr__(
            self, "timings_milliseconds", dict(self.timings_milliseconds)
        )
        expected_checksum = self.compute_checksum()
        if self.content_checksum != expected_checksum:
            raise ValueError("content_checksum does not match prepared tile contents")

    @property
    def valid_mask(self) -> MaskArray:
        """Mask permitted for evidence-producing stages after preparation."""

        return _readonly(np.logical_not(self.resized_invalid_mask))

    def metadata(self) -> dict[str, object]:
        """Emit provenance, masks, ranges, timing, and OpenCV build metadata."""

        metadata = self._metadata(include_checksum=True)
        metadata["timings_milliseconds"] = dict(
            sorted(self.timings_milliseconds.items())
        )
        metadata["opencv"] = _opencv_metadata()
        return metadata

    def compute_checksum(self) -> str:
        """Hash stable inputs and arrays, intentionally excluding runtime timings."""

        return _content_checksum(
            _checksum_metadata(
                {
                    "parameters": self.parameters,
                    "source_crop_checksum": self.source_crop_checksum,
                    "source_timing": self.source_timing,
                    "input_valid_range": self.input_valid_range,
                    "display_range": self.display_range,
                }
            ),
            self._arrays(),
        )

    def _metadata(self, *, include_checksum: bool) -> dict[str, object]:
        valid_count = int(np.count_nonzero(~self.resized_invalid_mask))
        metadata: dict[str, object] = {
            "schema_version": _SCHEMA_VERSION,
            "record_type": "mask_aware_prepared_tile",
            "source_crop_checksum": self.source_crop_checksum,
            "source_timing": self.source_timing,
            "parameters": self.parameters.to_dict(),
            "ranges_kelvin": {
                "input_valid": _range_dict(self.input_valid_range),
                "physical": {
                    "minimum": self.parameters.physical_minimum_kelvin,
                    "maximum": self.parameters.physical_maximum_kelvin,
                },
                "display": _range_dict(self.display_range),
            },
            "masks": {
                "input_invalid_pixel_count": int(
                    np.count_nonzero(self.input_invalid_mask)
                ),
                "effective_invalid_pixel_count": int(
                    np.count_nonzero(self.effective_invalid_mask)
                ),
                "physical_low_clip_pixel_count": int(
                    np.count_nonzero(self.physical_low_clip_mask)
                ),
                "physical_high_clip_pixel_count": int(
                    np.count_nonzero(self.physical_high_clip_mask)
                ),
                "resized_invalid_pixel_count": int(
                    np.count_nonzero(self.resized_invalid_mask)
                ),
                "resized_valid_pixel_count": valid_count,
                "invalid_pixels_are_excluded_from_evidence": True,
            },
            "arrays": {
                name: _array_metadata(array) for name, array in self._arrays().items()
            },
        }
        if include_checksum:
            metadata["content_checksum"] = self.content_checksum
        return metadata

    def _arrays(self) -> dict[str, npt.NDArray[np.generic]]:
        arrays: dict[str, npt.NDArray[np.generic]] = {
            "calibrated": self.calibrated,
            "input_invalid_mask": self.input_invalid_mask,
            "effective_invalid_mask": self.effective_invalid_mask,
            "physical_clipped": self.physical_clipped,
            "physical_low_clip_mask": self.physical_low_clip_mask,
            "physical_high_clip_mask": self.physical_high_clip_mask,
            "resized_calibrated": self.resized_calibrated,
            "resized_invalid_mask": self.resized_invalid_mask,
            "display": self.display,
        }
        if self.clahe_display is not None:
            arrays["clahe_display"] = self.clahe_display
        return arrays


def prepare_tile(
    crop: CalibratedCrop, parameters: TilePreparationParameters
) -> PreparedTile:
    """Prepare one immutable calibrated crop with its source provenance."""

    if not isinstance(crop, CalibratedCrop):
        raise TypeError("crop must be a CalibratedCrop")
    return prepare_calibrated_tile(
        crop.calibrated,
        crop.invalid_mask,
        parameters,
        source_crop_checksum=crop.content_checksum,
        source_timing=crop.timing.to_dict(),
    )


def prepare_calibrated_tile(
    calibrated: npt.NDArray[np.generic],
    invalid_mask: npt.NDArray[np.generic],
    parameters: TilePreparationParameters,
    *,
    source_crop_checksum: str | None = None,
    source_timing: dict[str, str] | None = None,
) -> PreparedTile:
    """Prepare arrays directly when the caller has equivalent provenance.

    The supplied calibrated values are copied unchanged to ``PreparedTile``.
    Clipping, resize, display scaling, and optional CLAHE operate on derived
    arrays only; downstream evidence must consume ``resized_calibrated`` and
    ``valid_mask`` rather than display pixels.
    """

    if not isinstance(parameters, TilePreparationParameters):
        raise TypeError("parameters must be TilePreparationParameters")
    started = time.perf_counter_ns()
    values = np.array(calibrated, dtype=np.float32, copy=True)
    input_invalid = np.array(invalid_mask, dtype=bool, copy=True)
    if values.ndim != 2 or input_invalid.shape != values.shape:
        raise ValueError(
            "calibrated values and invalid_mask must be matching 2D arrays"
        )
    if source_crop_checksum is not None:
        _sha256_text(source_crop_checksum, "source_crop_checksum")
    if source_timing is not None:
        source_timing = dict(source_timing)
        if set(source_timing) != {"start", "end"} or not all(
            isinstance(value, str) and value.endswith("Z")
            for value in source_timing.values()
        ):
            raise ValueError("source_timing must contain UTC start and end strings")

    effective_invalid = np.logical_or(input_invalid, ~np.isfinite(values))
    valid = ~effective_invalid
    input_valid_range = _valid_range(values, valid)
    physical, low_clip_mask, high_clip_mask = _physical_clip(values, valid, parameters)
    clipped_at = time.perf_counter_ns()
    resized, resized_invalid = _mask_aware_resize(physical, valid, parameters)
    resized_at = time.perf_counter_ns()
    display_range = _robust_display_range(resized, ~resized_invalid, parameters)
    display = _display_scale(resized, resized_invalid, display_range)
    scaled_at = time.perf_counter_ns()
    clahe_display = _apply_clahe(display, resized_invalid, parameters)
    finished = time.perf_counter_ns()
    timings = {
        "physical_clip": _milliseconds(clipped_at - started),
        "mask_aware_resize": _milliseconds(resized_at - clipped_at),
        "display_scale": _milliseconds(scaled_at - resized_at),
        "optional_clahe": _milliseconds(finished - scaled_at),
        "total": _milliseconds(finished - started),
    }
    payload = {
        "calibrated": values,
        "input_invalid_mask": input_invalid,
        "effective_invalid_mask": effective_invalid,
        "physical_clipped": physical,
        "physical_low_clip_mask": low_clip_mask,
        "physical_high_clip_mask": high_clip_mask,
        "resized_calibrated": resized,
        "resized_invalid_mask": resized_invalid,
        "display": display,
        "clahe_display": clahe_display,
        "parameters": parameters,
        "source_crop_checksum": source_crop_checksum,
        "source_timing": source_timing,
        "input_valid_range": input_valid_range,
        "display_range": display_range,
    }
    checksum = _content_checksum(_checksum_metadata(payload), _payload_arrays(payload))
    return PreparedTile(
        **payload,
        timings_milliseconds=timings,
        content_checksum=checksum,
    )


def _physical_clip(
    values: FloatArray, valid: MaskArray, parameters: TilePreparationParameters
) -> tuple[FloatArray, MaskArray, MaskArray]:
    physical = np.full(values.shape, np.nan, dtype=np.float32)
    physical[valid] = np.clip(
        values[valid],
        parameters.physical_minimum_kelvin,
        parameters.physical_maximum_kelvin,
    )
    low = np.logical_and(valid, values < parameters.physical_minimum_kelvin)
    high = np.logical_and(valid, values > parameters.physical_maximum_kelvin)
    return physical, low, high


def _mask_aware_resize(
    physical: FloatArray, valid: MaskArray, parameters: TilePreparationParameters
) -> tuple[FloatArray, MaskArray]:
    target_shape = (
        physical.shape if parameters.target_shape is None else parameters.target_shape
    )
    if target_shape == physical.shape:
        return np.array(physical, copy=True), np.array(~valid, copy=True)
    interpolation = _resize_interpolation(physical.shape, target_shape)
    width_height = (target_shape[1], target_shape[0])
    weights = valid.astype(np.float32)
    weighted_values = np.where(valid, physical, np.float32(0.0))
    resized_weights = cv2.resize(weights, width_height, interpolation=interpolation)
    resized_sum = cv2.resize(weighted_values, width_height, interpolation=interpolation)
    resized_weights = np.clip(np.asarray(resized_weights, dtype=np.float32), 0.0, 1.0)
    resized = np.full(target_shape, np.nan, dtype=np.float32)
    supported = resized_weights > 0.0
    resized[supported] = resized_sum[supported] / resized_weights[supported]
    epsilon = np.float32(1e-6)
    invalid = resized_weights + epsilon < parameters.minimum_valid_coverage
    invalid |= ~np.isfinite(resized)
    resized[invalid] = np.nan
    return resized, invalid


def _resize_interpolation(
    source_shape: tuple[int, int], target_shape: tuple[int, int]
) -> int:
    if target_shape[0] <= source_shape[0] and target_shape[1] <= source_shape[1]:
        return cv2.INTER_AREA
    return cv2.INTER_LINEAR


def _robust_display_range(
    values: FloatArray, valid: MaskArray, parameters: TilePreparationParameters
) -> tuple[float, float]:
    valid_values = values[valid]
    if valid_values.size == 0:
        return (
            parameters.physical_minimum_kelvin,
            parameters.physical_maximum_kelvin,
        )
    lower = float(np.quantile(valid_values, parameters.display_lower_quantile))
    upper = float(np.quantile(valid_values, parameters.display_upper_quantile))
    lower = max(lower, parameters.physical_minimum_kelvin)
    upper = min(upper, parameters.physical_maximum_kelvin)
    if not upper > lower:
        lower = parameters.physical_minimum_kelvin
        upper = parameters.physical_maximum_kelvin
    return lower, upper


def _display_scale(
    values: FloatArray, invalid: MaskArray, value_range: tuple[float, float]
) -> Uint8Array:
    lower, upper = value_range
    display = np.zeros(values.shape, dtype=np.uint8)
    valid = ~invalid
    if np.any(valid):
        normalized = np.clip((values[valid] - lower) / (upper - lower), 0.0, 1.0)
        display[valid] = np.rint(normalized * 255.0).astype(np.uint8)
    return display


def _apply_clahe(
    display: Uint8Array, invalid: MaskArray, parameters: TilePreparationParameters
) -> Uint8Array | None:
    if parameters.clahe_clip_limit is None:
        return None
    valid = ~invalid
    if not np.any(valid):
        return np.zeros(display.shape, dtype=np.uint8)
    # OpenCV CLAHE has no mask parameter.  A robust median fill prevents masked
    # zeros from creating a false cold mode; the result remains display-only and
    # masked pixels are restored to black before it can be shown or consumed.
    filled = np.array(display, copy=True)
    fill_value = np.uint8(np.rint(np.median(display[valid])))
    filled[invalid] = fill_value
    clahe = cv2.createCLAHE(
        clipLimit=parameters.clahe_clip_limit,
        tileGridSize=(
            parameters.clahe_tile_grid_size[1],
            parameters.clahe_tile_grid_size[0],
        ),
    )
    enhanced = np.asarray(clahe.apply(filled), dtype=np.uint8)
    enhanced[invalid] = 0
    return enhanced


def _checksum_metadata(payload: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "record_type": "mask_aware_prepared_tile",
        "source_crop_checksum": payload["source_crop_checksum"],
        "source_timing": payload["source_timing"],
        "parameters": _parameters(payload["parameters"]).to_dict(),
        "ranges_kelvin": {
            "input_valid": _range_dict(payload["input_valid_range"]),
            "physical": {
                "minimum": _parameters(payload["parameters"]).physical_minimum_kelvin,
                "maximum": _parameters(payload["parameters"]).physical_maximum_kelvin,
            },
            "display": _range_dict(payload["display_range"]),
        },
    }


def _payload_arrays(payload: dict[str, object]) -> dict[str, npt.NDArray[np.generic]]:
    arrays: dict[str, npt.NDArray[np.generic]] = {}
    for name in (
        "calibrated",
        "input_invalid_mask",
        "effective_invalid_mask",
        "physical_clipped",
        "physical_low_clip_mask",
        "physical_high_clip_mask",
        "resized_calibrated",
        "resized_invalid_mask",
        "display",
    ):
        arrays[name] = _array(payload[name], name)
    if payload["clahe_display"] is not None:
        arrays["clahe_display"] = _array(payload["clahe_display"], "clahe_display")
    return arrays


def _parameters(value: object) -> TilePreparationParameters:
    if not isinstance(value, TilePreparationParameters):
        raise ValueError("parameters must be TilePreparationParameters")
    return value


def _array(value: object, name: str) -> npt.NDArray[np.generic]:
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{name} must be an ndarray")
    return value


def _valid_range(values: FloatArray, valid: MaskArray) -> tuple[float, float] | None:
    if not np.any(valid):
        return None
    return float(np.min(values[valid])), float(np.max(values[valid]))


def _range_dict(value: object) -> dict[str, float] | None:
    if value is None:
        return None
    minimum, maximum = _range(value, "range", allow_equal=True)
    return {"minimum": minimum, "maximum": maximum}


def _range(
    value: object, name: str, *, allow_equal: bool = False
) -> tuple[float, float]:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or isinstance(value[0], bool)
        or isinstance(value[1], bool)
        or not isinstance(value[0], (int, float))
        or not isinstance(value[1], (int, float))
    ):
        raise ValueError(f"{name} must be a two-number tuple")
    minimum = _finite_number(value[0], f"{name} minimum")
    maximum = _finite_number(value[1], f"{name} maximum")
    if maximum < minimum or (not allow_equal and maximum == minimum):
        relation = "not be below" if allow_equal else "exceed"
        raise ValueError(f"{name} maximum must {relation} minimum")
    return minimum, maximum


def _shape(value: object, name: str) -> tuple[int, int]:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 1
            for item in value
        )
    ):
        raise ValueError(f"{name} must be a two-positive-integer tuple")
    return value


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _sha256_text(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _readonly[ArrayScalar: np.generic](
    array: npt.NDArray[ArrayScalar],
) -> npt.NDArray[ArrayScalar]:
    copy = np.ascontiguousarray(array).copy()
    copy.setflags(write=False)
    return copy


def _array_metadata(array: npt.NDArray[np.generic]) -> dict[str, object]:
    contiguous = np.ascontiguousarray(array)
    return {
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
        "sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
    }


def _content_checksum(
    metadata: dict[str, object], arrays: dict[str, npt.NDArray[np.generic]]
) -> str:
    digest = hashlib.sha256()
    digest.update(b"firesentinel-prepared-tile-v1\0metadata\0")
    digest.update(
        json.dumps(
            metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
    )
    for name in sorted(arrays):
        contiguous = np.ascontiguousarray(arrays[name])
        digest.update(b"\0array\0")
        digest.update(name.encode())
        digest.update(b"\0dtype\0")
        digest.update(contiguous.dtype.str.encode())
        digest.update(b"\0shape\0")
        digest.update(
            json.dumps(list(contiguous.shape), separators=(",", ":")).encode()
        )
        digest.update(b"\0bytes\0")
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _opencv_metadata() -> dict[str, str]:
    return {
        "build_information_sha256": hashlib.sha256(
            _OPEN_CV_BUILD_INFORMATION.encode()
        ).hexdigest(),
        "version": cv2.__version__,
    }


def _milliseconds(duration_nanoseconds: int) -> float:
    return round(duration_nanoseconds / 1_000_000.0, 6)


__all__ = [
    "PreparedTile",
    "TilePreparationParameters",
    "prepare_calibrated_tile",
    "prepare_tile",
]
