"""Mask-aware observation-quality measurements and safe evidence gating.

This module intentionally runs before any thermal-anomaly interpretation.  It
measures only calibrated values and their validity mask; display-scaled pixels
and evaluation labels are not inputs.  The checked-in defaults were selected
from the frozen development workflow and the deterministic synthetic fixtures.
They are constants at runtime, so test or stress labels cannot alter an
observation's quality decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from firesentinel.core.records import ReasonCode

if TYPE_CHECKING:
    from firesentinel.vision.tiles import PreparedTile


FloatArray = npt.NDArray[np.float32]
MaskArray = npt.NDArray[np.bool]

THRESHOLD_SELECTION_SCOPE = "development_cases_and_synthetic_fixtures_only"


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{field} must be a finite number")
    return number


def _fraction(value: object, field: str) -> float:
    fraction = _finite_number(value, field)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"{field} must be within [0.0, 1.0]")
    return fraction


@dataclass(frozen=True, slots=True)
class ObservationQualityThresholds:
    """Pinned development-only limits for unusable thermal observations.

    Saturation includes a caller-supplied physical clipping mask and values at
    either configured instrument endpoint.  The default upper endpoint matches
    the controlled clipped-frame fixture; production calibration limits must
    be supplied explicitly when they differ.
    """

    minimum_usable_coverage_fraction: float = 0.75
    maximum_saturated_fraction: float = 0.02
    minimum_contrast_span_kelvin: float = 0.05
    minimum_texture_standard_deviation_kelvin: float = 0.01
    blank_maximum_kelvin: float = 1.0
    saturation_minimum_kelvin: float | None = None
    saturation_maximum_kelvin: float | None = 350.0

    def __post_init__(self) -> None:
        coverage = _fraction(
            self.minimum_usable_coverage_fraction,
            "minimum_usable_coverage_fraction",
        )
        if coverage == 0.0:
            raise ValueError("minimum_usable_coverage_fraction must be positive")
        saturation = _fraction(
            self.maximum_saturated_fraction, "maximum_saturated_fraction"
        )
        if saturation == 0.0:
            raise ValueError("maximum_saturated_fraction must be positive")
        contrast = _finite_number(
            self.minimum_contrast_span_kelvin, "minimum_contrast_span_kelvin"
        )
        texture = _finite_number(
            self.minimum_texture_standard_deviation_kelvin,
            "minimum_texture_standard_deviation_kelvin",
        )
        blank = _finite_number(self.blank_maximum_kelvin, "blank_maximum_kelvin")
        if contrast <= 0.0 or texture <= 0.0:
            raise ValueError("contrast and texture thresholds must be positive")
        minimum = (
            None
            if self.saturation_minimum_kelvin is None
            else _finite_number(
                self.saturation_minimum_kelvin, "saturation_minimum_kelvin"
            )
        )
        maximum = (
            None
            if self.saturation_maximum_kelvin is None
            else _finite_number(
                self.saturation_maximum_kelvin, "saturation_maximum_kelvin"
            )
        )
        if minimum is None and maximum is None:
            raise ValueError("at least one saturation endpoint is required")
        if minimum is not None and maximum is not None and maximum <= minimum:
            raise ValueError("saturation_maximum_kelvin must exceed its minimum")
        object.__setattr__(self, "minimum_usable_coverage_fraction", coverage)
        object.__setattr__(self, "maximum_saturated_fraction", saturation)
        object.__setattr__(self, "minimum_contrast_span_kelvin", contrast)
        object.__setattr__(self, "minimum_texture_standard_deviation_kelvin", texture)
        object.__setattr__(self, "blank_maximum_kelvin", blank)
        object.__setattr__(self, "saturation_minimum_kelvin", minimum)
        object.__setattr__(self, "saturation_maximum_kelvin", maximum)

    def to_dict(self) -> dict[str, float | str | None]:
        """Return all limits and their deliberately restricted selection scope."""

        return {
            "selection_scope": THRESHOLD_SELECTION_SCOPE,
            "minimum_usable_coverage_fraction": self.minimum_usable_coverage_fraction,
            "maximum_saturated_fraction": self.maximum_saturated_fraction,
            "minimum_contrast_span_kelvin": self.minimum_contrast_span_kelvin,
            "minimum_texture_standard_deviation_kelvin": (
                self.minimum_texture_standard_deviation_kelvin
            ),
            "blank_maximum_kelvin": self.blank_maximum_kelvin,
            "saturation_minimum_kelvin": self.saturation_minimum_kelvin,
            "saturation_maximum_kelvin": self.saturation_maximum_kelvin,
        }


DEVELOPMENT_QUALITY_THRESHOLDS = ObservationQualityThresholds()


@dataclass(frozen=True, slots=True)
class ObservationQuality:
    """Measured quality and the deterministic maximum allowed evidence confidence.

    Fraction and score fields are always finite values in ``[0.0, 1.0]``.  Raw
    Kelvin fields are present so a reviewer can understand why a bounded score
    or reason code was emitted.  Any poor-quality reason makes the confidence
    cap zero and clears a candidate mask through :func:`apply_quality_gate`.
    """

    missing_pixel_fraction: float
    usable_coverage_fraction: float
    saturated_pixel_fraction: float
    contrast_span_kelvin: float
    texture_standard_deviation_kelvin: float
    mean_absolute_neighbor_difference_kelvin: float
    coverage_score: float
    saturation_score: float
    contrast_score: float
    texture_score: float
    quality_score: float
    reason_codes: tuple[ReasonCode, ...]
    fire_evidence_confidence_cap: float

    def __post_init__(self) -> None:
        for name in (
            "missing_pixel_fraction",
            "usable_coverage_fraction",
            "saturated_pixel_fraction",
            "coverage_score",
            "saturation_score",
            "contrast_score",
            "texture_score",
            "quality_score",
            "fire_evidence_confidence_cap",
        ):
            object.__setattr__(self, name, _fraction(getattr(self, name), name))
        for name in (
            "contrast_span_kelvin",
            "texture_standard_deviation_kelvin",
            "mean_absolute_neighbor_difference_kelvin",
        ):
            value = _finite_number(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        if not np.isclose(
            self.missing_pixel_fraction + self.usable_coverage_fraction,
            1.0,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("missing and usable coverage fractions must total one")
        reasons = tuple(ReasonCode(reason) for reason in self.reason_codes)
        if len(reasons) != len(set(reasons)):
            raise ValueError("reason_codes must not contain duplicates")
        if reasons and self.fire_evidence_confidence_cap != 0.0:
            raise ValueError(
                "poor-quality observations must have a zero confidence cap"
            )
        object.__setattr__(self, "reason_codes", reasons)

    @property
    def usable_for_fire_evidence(self) -> bool:
        """Whether this observation can contribute non-zero fire evidence."""

        return not self.reason_codes

    def cap_fire_evidence_confidence(self, candidate_confidence: float) -> float:
        """Apply the quality gate to a candidate's bounded confidence."""

        return min(
            _fraction(candidate_confidence, "candidate_confidence"),
            self.fire_evidence_confidence_cap,
        )

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe measurements, bounded scores, and explicit reasons."""

        return {
            "missing_pixel_fraction": self.missing_pixel_fraction,
            "usable_coverage_fraction": self.usable_coverage_fraction,
            "saturated_pixel_fraction": self.saturated_pixel_fraction,
            "contrast_span_kelvin": self.contrast_span_kelvin,
            "texture_standard_deviation_kelvin": self.texture_standard_deviation_kelvin,
            "mean_absolute_neighbor_difference_kelvin": (
                self.mean_absolute_neighbor_difference_kelvin
            ),
            "coverage_score": self.coverage_score,
            "saturation_score": self.saturation_score,
            "contrast_score": self.contrast_score,
            "texture_score": self.texture_score,
            "quality_score": self.quality_score,
            "reason_codes": [reason.value for reason in self.reason_codes],
            "usable_for_fire_evidence": self.usable_for_fire_evidence,
            "fire_evidence_confidence_cap": self.fire_evidence_confidence_cap,
        }


def measure_observation_quality(
    calibrated: npt.NDArray[np.generic],
    invalid_mask: npt.NDArray[np.generic],
    thresholds: ObservationQualityThresholds = DEVELOPMENT_QUALITY_THRESHOLDS,
    *,
    clipped_mask: npt.NDArray[np.generic] | None = None,
) -> ObservationQuality:
    """Measure a calibrated observation before extracting thermal anomalies.

    ``invalid_mask`` and non-finite samples both count as missing.  Clipping is
    accepted as a separate mask because prepared physical tiles may replace the
    original endpoint value.  Invalid samples never contribute to contrast,
    texture, saturation, or potential fire evidence.
    """

    if not isinstance(thresholds, ObservationQualityThresholds):
        raise TypeError("thresholds must be ObservationQualityThresholds")
    values = np.asarray(calibrated, dtype=np.float32)
    invalid = np.asarray(invalid_mask, dtype=bool)
    if values.ndim != 2 or invalid.shape != values.shape:
        raise ValueError(
            "calibrated values and invalid_mask must be matching 2D arrays"
        )
    if values.size == 0:
        raise ValueError("calibrated values must contain at least one pixel")
    clip = np.zeros(values.shape, dtype=bool)
    if clipped_mask is not None:
        clip = np.asarray(clipped_mask, dtype=bool)
        if clip.shape != values.shape:
            raise ValueError("clipped_mask must match calibrated values")

    usable = np.logical_and(~invalid, np.isfinite(values))
    usable_count = int(np.count_nonzero(usable))
    total_count = int(values.size)
    usable_fraction = usable_count / total_count
    missing_fraction = 1.0 - usable_fraction
    usable_values = values[usable]
    blank = (
        usable_count > 0
        and float(np.max(usable_values)) <= thresholds.blank_maximum_kelvin
    )

    saturated = np.logical_and(usable, clip)
    if thresholds.saturation_minimum_kelvin is not None:
        saturated |= np.logical_and(
            usable, values <= thresholds.saturation_minimum_kelvin
        )
    if thresholds.saturation_maximum_kelvin is not None:
        saturated |= np.logical_and(
            usable, values >= thresholds.saturation_maximum_kelvin
        )
    saturated_fraction = (
        float(np.count_nonzero(saturated)) / usable_count if usable_count else 0.0
    )

    contrast_span, texture_standard_deviation, neighbor_difference = (
        _texture_statistics(values, usable)
    )
    coverage_score = min(
        usable_fraction / thresholds.minimum_usable_coverage_fraction, 1.0
    )
    saturation_score = max(
        0.0, 1.0 - saturated_fraction / thresholds.maximum_saturated_fraction
    )
    contrast_score = min(contrast_span / thresholds.minimum_contrast_span_kelvin, 1.0)
    texture_score = min(
        texture_standard_deviation
        / thresholds.minimum_texture_standard_deviation_kelvin,
        1.0,
    )
    quality_score = min(coverage_score, saturation_score, contrast_score, texture_score)

    # Emit one primary poor-quality reason.  Later checks have no useful
    # interpretation once a frame is incomplete, blank, or clipped.
    reasons: tuple[ReasonCode, ...]
    if usable_fraction < thresholds.minimum_usable_coverage_fraction:
        reasons = (ReasonCode.COVERAGE_INSUFFICIENT,)
    elif blank:
        reasons = (ReasonCode.FRAME_BLANK,)
    elif saturated_fraction > thresholds.maximum_saturated_fraction:
        reasons = (ReasonCode.FRAME_SATURATED,)
    elif (
        contrast_span < thresholds.minimum_contrast_span_kelvin
        or texture_standard_deviation
        < thresholds.minimum_texture_standard_deviation_kelvin
    ):
        reasons = (ReasonCode.CONTRAST_LOW,)
    else:
        reasons = ()

    return ObservationQuality(
        missing_pixel_fraction=missing_fraction,
        usable_coverage_fraction=usable_fraction,
        saturated_pixel_fraction=saturated_fraction,
        contrast_span_kelvin=contrast_span,
        texture_standard_deviation_kelvin=texture_standard_deviation,
        mean_absolute_neighbor_difference_kelvin=neighbor_difference,
        coverage_score=coverage_score,
        saturation_score=saturation_score,
        contrast_score=contrast_score,
        texture_score=texture_score,
        quality_score=quality_score,
        reason_codes=reasons,
        fire_evidence_confidence_cap=0.0 if reasons else 1.0,
    )


def measure_prepared_tile_quality(
    tile: PreparedTile,
    thresholds: ObservationQualityThresholds = DEVELOPMENT_QUALITY_THRESHOLDS,
) -> ObservationQuality:
    """Measure a :class:`PreparedTile` using its calibrated pre-resize data."""

    # Importing tiles at module import time would create an avoidable cycle for
    # callers that only need quality measurements from arrays.
    from firesentinel.vision.tiles import PreparedTile

    if not isinstance(tile, PreparedTile):
        raise TypeError("tile must be PreparedTile")
    return measure_observation_quality(
        tile.calibrated,
        tile.effective_invalid_mask,
        thresholds,
        clipped_mask=np.logical_or(
            tile.physical_low_clip_mask, tile.physical_high_clip_mask
        ),
    )


def apply_quality_gate(
    candidate_mask: npt.NDArray[np.generic], quality: ObservationQuality
) -> npt.NDArray[np.generic]:
    """Clear apparent candidates when observation quality forbids evidence."""

    if not isinstance(quality, ObservationQuality):
        raise TypeError("quality must be ObservationQuality")
    mask = np.asarray(candidate_mask)
    if mask.ndim != 2:
        raise ValueError("candidate_mask must be two-dimensional")
    if quality.usable_for_fire_evidence:
        return np.array(mask, copy=True)
    return np.zeros(mask.shape, dtype=mask.dtype)


def _texture_statistics(
    values: FloatArray, usable: MaskArray
) -> tuple[float, float, float]:
    """Return valid-only contrast, standard deviation, and neighbor difference."""

    valid_values = values[usable]
    if valid_values.size == 0:
        return 0.0, 0.0, 0.0
    contrast_span = float(np.max(valid_values) - np.min(valid_values))
    texture_standard_deviation = float(np.std(valid_values, dtype=np.float64))
    horizontal_pairs = np.logical_and(usable[:, :-1], usable[:, 1:])
    vertical_pairs = np.logical_and(usable[:-1, :], usable[1:, :])
    differences: list[npt.NDArray[np.float32]] = []
    if np.any(horizontal_pairs):
        differences.append(np.abs(values[:, 1:] - values[:, :-1])[horizontal_pairs])
    if np.any(vertical_pairs):
        differences.append(np.abs(values[1:, :] - values[:-1, :])[vertical_pairs])
    if not differences:
        return contrast_span, texture_standard_deviation, 0.0
    return (
        contrast_span,
        texture_standard_deviation,
        float(np.mean(np.concatenate(differences), dtype=np.float64)),
    )


__all__ = [
    "DEVELOPMENT_QUALITY_THRESHOLDS",
    "THRESHOLD_SELECTION_SCOPE",
    "ObservationQuality",
    "ObservationQualityThresholds",
    "apply_quality_gate",
    "measure_observation_quality",
    "measure_prepared_tile_quality",
]
