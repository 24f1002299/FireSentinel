"""Interpretable, mask-aware contextual thermal-anomaly extraction.

The stage works on calibrated Channel 7 and Channel 14 arrays, never on a
display rendering.  It first applies the observation-quality gate, then
requires both a locally warm Channel 7 response and a sufficiently large
Channel 7-minus-Channel 14 difference.  Every retained region can therefore
be traced back to explicit source-array measurements rather than a classifier
score.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import cv2
import numpy as np
import numpy.typing as npt

from firesentinel.core.records import ReasonCode
from firesentinel.vision.quality import (
    DEVELOPMENT_QUALITY_THRESHOLDS,
    ObservationQuality,
    ObservationQualityThresholds,
    apply_quality_gate,
    measure_observation_quality,
)

FloatArray = npt.NDArray[np.float32]
MaskArray = npt.NDArray[np.bool]
Uint8Array = npt.NDArray[np.uint8]
Int32Array = npt.NDArray[np.int32]


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{field} must be a finite number")
    return number


def _positive_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _nonnegative_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class ContextualAnomalyParameters:
    """Pinned, auditable thresholds and OpenCV region-filtering settings."""

    local_background_kernel_pixels: int = 5
    minimum_local_contrast_kelvin: float = 2.0
    minimum_channel_difference_kelvin: float = 15.0
    morphology_kernel_pixels: int = 3
    minimum_component_area_pixels: int = 4
    minimum_edge_distance_pixels: int = 1

    def __post_init__(self) -> None:
        background = _positive_integer(
            self.local_background_kernel_pixels, "local_background_kernel_pixels"
        )
        morphology = _positive_integer(
            self.morphology_kernel_pixels, "morphology_kernel_pixels"
        )
        if background % 2 == 0 or morphology % 2 == 0:
            raise ValueError("OpenCV kernel sizes must be odd")
        local_contrast = _finite_number(
            self.minimum_local_contrast_kelvin, "minimum_local_contrast_kelvin"
        )
        channel_difference = _finite_number(
            self.minimum_channel_difference_kelvin,
            "minimum_channel_difference_kelvin",
        )
        if local_contrast <= 0.0 or channel_difference <= 0.0:
            raise ValueError("contextual anomaly thresholds must be positive")
        minimum_area = _positive_integer(
            self.minimum_component_area_pixels, "minimum_component_area_pixels"
        )
        edge_distance = _nonnegative_integer(
            self.minimum_edge_distance_pixels, "minimum_edge_distance_pixels"
        )
        object.__setattr__(self, "local_background_kernel_pixels", background)
        object.__setattr__(self, "minimum_local_contrast_kelvin", local_contrast)
        object.__setattr__(
            self, "minimum_channel_difference_kelvin", channel_difference
        )
        object.__setattr__(self, "morphology_kernel_pixels", morphology)
        object.__setattr__(self, "minimum_component_area_pixels", minimum_area)
        object.__setattr__(self, "minimum_edge_distance_pixels", edge_distance)

    def to_dict(self) -> dict[str, float | int]:
        """Return every parameter that influences the candidate mask."""

        return {
            "local_background_kernel_pixels": self.local_background_kernel_pixels,
            "minimum_local_contrast_kelvin": self.minimum_local_contrast_kelvin,
            "minimum_channel_difference_kelvin": (
                self.minimum_channel_difference_kelvin
            ),
            "morphology_kernel_pixels": self.morphology_kernel_pixels,
            "minimum_component_area_pixels": self.minimum_component_area_pixels,
            "minimum_edge_distance_pixels": self.minimum_edge_distance_pixels,
        }


DEVELOPMENT_CONTEXTUAL_ANOMALY_PARAMETERS = ContextualAnomalyParameters()


@dataclass(frozen=True, slots=True)
class ContextualAnomalyComponent:
    """A source-array-measured retained thermal candidate region."""

    label: int
    area_pixels: int
    bounding_box_xywh: tuple[int, int, int, int]
    centroid_xy: tuple[float, float]
    mean_local_contrast_kelvin: float
    maximum_local_contrast_kelvin: float
    mean_channel_difference_kelvin: float
    maximum_channel_difference_kelvin: float
    edge_distance_pixels: int
    edge_proximity_fraction: float
    touches_edge: bool

    def to_dict(self) -> dict[str, object]:
        """Return reviewer-facing component measurements without image payloads."""

        return {
            "label": self.label,
            "area_pixels": self.area_pixels,
            "bounding_box_xywh": list(self.bounding_box_xywh),
            "centroid_xy": list(self.centroid_xy),
            "mean_local_contrast_kelvin": self.mean_local_contrast_kelvin,
            "maximum_local_contrast_kelvin": self.maximum_local_contrast_kelvin,
            "mean_channel_difference_kelvin": self.mean_channel_difference_kelvin,
            "maximum_channel_difference_kelvin": self.maximum_channel_difference_kelvin,
            "edge_distance_pixels": self.edge_distance_pixels,
            "edge_proximity_fraction": self.edge_proximity_fraction,
            "touches_edge": self.touches_edge,
        }


Contour = tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class ContextualAnomalyResult:
    """All deterministic maps, regions, and quality decisions for one pairing."""

    local_contrast_kelvin: FloatArray
    channel_difference_kelvin: FloatArray
    valid_mask: MaskArray
    local_contrast_threshold_mask: Uint8Array
    channel_difference_threshold_mask: Uint8Array
    morphology_mask: Uint8Array
    candidate_mask: Uint8Array
    labels: Int32Array
    components: tuple[ContextualAnomalyComponent, ...]
    contours: tuple[Contour, ...]
    channel7_quality: ObservationQuality
    channel14_quality: ObservationQuality
    reason_codes: tuple[ReasonCode, ...]
    overlay: Uint8Array

    def __post_init__(self) -> None:
        local = np.asarray(self.local_contrast_kelvin, dtype=np.float32)
        difference = np.asarray(self.channel_difference_kelvin, dtype=np.float32)
        valid = np.asarray(self.valid_mask, dtype=bool)
        masks = tuple(
            np.asarray(mask, dtype=np.uint8)
            for mask in (
                self.local_contrast_threshold_mask,
                self.channel_difference_threshold_mask,
                self.morphology_mask,
                self.candidate_mask,
            )
        )
        labels = np.asarray(self.labels, dtype=np.int32)
        overlay = np.asarray(self.overlay, dtype=np.uint8)
        if local.ndim != 2 or any(
            array.shape != local.shape for array in (difference, valid, *masks, labels)
        ):
            raise ValueError("contextual maps, masks, and labels must share a 2D shape")
        if overlay.shape != (*local.shape, 3):
            raise ValueError(
                "overlay must have the candidate-map height, width, and BGR"
            )
        if np.any(np.isfinite(local[~valid])) or np.any(
            np.isfinite(difference[~valid])
        ):
            raise ValueError("invalid pixels must remain NaN in contextual maps")
        if any(np.any(mask[~valid] != 0) for mask in masks):
            raise ValueError("invalid pixels must not appear in anomaly masks")
        if np.any(labels[~valid] != 0):
            raise ValueError("invalid pixels must not receive component labels")
        if not isinstance(self.channel7_quality, ObservationQuality) or not isinstance(
            self.channel14_quality, ObservationQuality
        ):
            raise ValueError("channel qualities must be ObservationQuality values")
        reasons = tuple(ReasonCode(reason) for reason in self.reason_codes)
        if len(reasons) != len(set(reasons)):
            raise ValueError("reason_codes must not contain duplicates")
        if reasons and np.any(masks[-1]):
            raise ValueError("poor-quality channels must not retain candidate regions")
        object.__setattr__(self, "local_contrast_kelvin", _readonly(local))
        object.__setattr__(self, "channel_difference_kelvin", _readonly(difference))
        object.__setattr__(self, "valid_mask", _readonly(valid))
        object.__setattr__(self, "local_contrast_threshold_mask", _readonly(masks[0]))
        object.__setattr__(
            self, "channel_difference_threshold_mask", _readonly(masks[1])
        )
        object.__setattr__(self, "morphology_mask", _readonly(masks[2]))
        object.__setattr__(self, "candidate_mask", _readonly(masks[3]))
        object.__setattr__(self, "labels", _readonly(labels))
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(self, "overlay", _readonly(overlay))

    @property
    def usable_for_fire_evidence(self) -> bool:
        """Whether both source channels passed quality gating."""

        return not self.reason_codes

    def to_dict(self) -> dict[str, object]:
        """Return measurements and hashes suitable for evidence records."""

        return {
            "channel7_quality": self.channel7_quality.to_dict(),
            "channel14_quality": self.channel14_quality.to_dict(),
            "reason_codes": [reason.value for reason in self.reason_codes],
            "usable_for_fire_evidence": self.usable_for_fire_evidence,
            "local_contrast_threshold_pixel_count": int(
                np.count_nonzero(self.local_contrast_threshold_mask)
            ),
            "channel_difference_threshold_pixel_count": int(
                np.count_nonzero(self.channel_difference_threshold_mask)
            ),
            "morphology_pixel_count": int(np.count_nonzero(self.morphology_mask)),
            "candidate_pixel_count": int(np.count_nonzero(self.candidate_mask)),
            "components": [component.to_dict() for component in self.components],
            "contours_xy": [
                [list(point) for point in contour] for contour in self.contours
            ],
        }


def extract_contextual_anomalies(
    channel7: npt.NDArray[np.generic],
    channel14: npt.NDArray[np.generic],
    channel7_invalid_mask: npt.NDArray[np.generic],
    channel14_invalid_mask: npt.NDArray[np.generic],
    parameters: ContextualAnomalyParameters = DEVELOPMENT_CONTEXTUAL_ANOMALY_PARAMETERS,
    *,
    quality_thresholds: ObservationQualityThresholds = DEVELOPMENT_QUALITY_THRESHOLDS,
) -> ContextualAnomalyResult:
    """Extract quality-gated, interpretable contextual C07/C14 candidate regions."""

    if not isinstance(parameters, ContextualAnomalyParameters):
        raise TypeError("parameters must be ContextualAnomalyParameters")
    channel7_values, channel14_values, channel7_invalid, channel14_invalid = (
        _validated_inputs(
            channel7, channel14, channel7_invalid_mask, channel14_invalid_mask
        )
    )
    channel7_quality = measure_observation_quality(
        channel7_values, channel7_invalid, quality_thresholds
    )
    channel14_quality = measure_observation_quality(
        channel14_values, channel14_invalid, quality_thresholds
    )
    reasons = _reason_codes(channel7_quality, channel14_quality)
    valid = np.logical_and(
        np.logical_and(~channel7_invalid, np.isfinite(channel7_values)),
        np.logical_and(~channel14_invalid, np.isfinite(channel14_values)),
    )
    local_contrast = _local_contrast(
        channel7_values, valid, parameters.local_background_kernel_pixels
    )
    channel_difference = np.full(channel7_values.shape, np.nan, dtype=np.float32)
    channel_difference[valid] = channel7_values[valid] - channel14_values[valid]
    local_threshold = _threshold(
        local_contrast, valid, parameters.minimum_local_contrast_kelvin
    )
    difference_threshold = _threshold(
        channel_difference, valid, parameters.minimum_channel_difference_kelvin
    )
    raw_candidates: Uint8Array = np.asarray(
        cv2.bitwise_and(local_threshold, difference_threshold), dtype=np.uint8
    )
    raw_candidates[~valid] = 0
    morphology = _morphology(raw_candidates, parameters.morphology_kernel_pixels)
    morphology[~valid] = 0
    quality_gated = apply_quality_gate(morphology, channel7_quality)
    quality_gated = apply_quality_gate(quality_gated, channel14_quality)
    candidate_mask, labels, components, contours = _filter_components(
        quality_gated,
        local_contrast,
        channel_difference,
        valid,
        parameters,
    )
    overlay = _overlay(channel7_values, valid, candidate_mask, contours, components)
    return ContextualAnomalyResult(
        local_contrast_kelvin=local_contrast,
        channel_difference_kelvin=channel_difference,
        valid_mask=valid,
        local_contrast_threshold_mask=local_threshold,
        channel_difference_threshold_mask=difference_threshold,
        morphology_mask=morphology,
        candidate_mask=candidate_mask,
        labels=labels,
        components=components,
        contours=contours,
        channel7_quality=channel7_quality,
        channel14_quality=channel14_quality,
        reason_codes=reasons,
        overlay=overlay,
    )


def _validated_inputs(
    channel7: npt.NDArray[np.generic],
    channel14: npt.NDArray[np.generic],
    channel7_invalid_mask: npt.NDArray[np.generic],
    channel14_invalid_mask: npt.NDArray[np.generic],
) -> tuple[FloatArray, FloatArray, MaskArray, MaskArray]:
    channel7_values = np.asarray(channel7, dtype=np.float32)
    channel14_values = np.asarray(channel14, dtype=np.float32)
    channel7_invalid = np.asarray(channel7_invalid_mask, dtype=bool)
    channel14_invalid = np.asarray(channel14_invalid_mask, dtype=bool)
    if channel7_values.ndim != 2 or channel14_values.shape != channel7_values.shape:
        raise ValueError("Channel 7 and Channel 14 must be matching 2D arrays")
    if (
        channel7_invalid.shape != channel7_values.shape
        or channel14_invalid.shape != channel7_values.shape
    ):
        raise ValueError("channel invalid masks must match their calibrated arrays")
    if channel7_values.size == 0:
        raise ValueError("channel arrays must contain at least one pixel")
    return channel7_values, channel14_values, channel7_invalid, channel14_invalid


def _reason_codes(
    channel7_quality: ObservationQuality, channel14_quality: ObservationQuality
) -> tuple[ReasonCode, ...]:
    return tuple(
        dict.fromkeys((*channel7_quality.reason_codes, *channel14_quality.reason_codes))
    )


def _local_contrast(
    values: FloatArray, valid: MaskArray, kernel_pixels: int
) -> FloatArray:
    weights = valid.astype(np.float32)
    weighted_values = np.where(valid, values, np.float32(0.0))
    kernel = (kernel_pixels, kernel_pixels)
    weighted_sum = cv2.GaussianBlur(
        weighted_values, kernel, sigmaX=0.0, borderType=cv2.BORDER_REPLICATE
    )
    support = cv2.GaussianBlur(
        weights, kernel, sigmaX=0.0, borderType=cv2.BORDER_REPLICATE
    )
    local_mean = np.full(values.shape, np.nan, dtype=np.float32)
    supported = np.logical_and(valid, support > np.float32(1e-6))
    local_mean[supported] = weighted_sum[supported] / support[supported]
    local_contrast = np.full(values.shape, np.nan, dtype=np.float32)
    local_contrast[supported] = values[supported] - local_mean[supported]
    return local_contrast


def _threshold(
    values: FloatArray, valid: MaskArray, threshold_kelvin: float
) -> Uint8Array:
    threshold_input = np.where(valid, values, np.float32(0.0))
    _, thresholded = cv2.threshold(
        threshold_input, threshold_kelvin, 255.0, cv2.THRESH_BINARY
    )
    mask = np.asarray(thresholded, dtype=np.uint8)
    mask[~valid] = 0
    return mask


def _morphology(mask: Uint8Array, kernel_pixels: int) -> Uint8Array:
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_pixels, kernel_pixels)
    )
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return np.asarray(cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel), dtype=np.uint8)


def _filter_components(
    mask: npt.NDArray[np.generic],
    local_contrast: FloatArray,
    channel_difference: FloatArray,
    valid: MaskArray,
    parameters: ContextualAnomalyParameters,
) -> tuple[
    Uint8Array, Int32Array, tuple[ContextualAnomalyComponent, ...], tuple[Contour, ...]
]:
    candidate_input = np.asarray(mask, dtype=np.uint8)
    count, raw_labels, statistics, centroids = cv2.connectedComponentsWithStats(
        candidate_input, connectivity=8, ltype=cv2.CV_32S
    )
    retained: list[int] = []
    component_values: list[ContextualAnomalyComponent] = []
    height, width = candidate_input.shape
    for label in range(1, count):
        area = int(statistics[label, cv2.CC_STAT_AREA])
        left = int(statistics[label, cv2.CC_STAT_LEFT])
        top = int(statistics[label, cv2.CC_STAT_TOP])
        component_width = int(statistics[label, cv2.CC_STAT_WIDTH])
        component_height = int(statistics[label, cv2.CC_STAT_HEIGHT])
        edge_distance = min(
            left,
            top,
            width - (left + component_width),
            height - (top + component_height),
        )
        if (
            area < parameters.minimum_component_area_pixels
            or edge_distance < parameters.minimum_edge_distance_pixels
        ):
            continue
        pixels = raw_labels == label
        retained.append(label)
        component_values.append(
            ContextualAnomalyComponent(
                label=label,
                area_pixels=area,
                bounding_box_xywh=(left, top, component_width, component_height),
                centroid_xy=(
                    float(centroids[label, 0]),
                    float(centroids[label, 1]),
                ),
                mean_local_contrast_kelvin=float(np.mean(local_contrast[pixels])),
                maximum_local_contrast_kelvin=float(np.max(local_contrast[pixels])),
                mean_channel_difference_kelvin=float(
                    np.mean(channel_difference[pixels])
                ),
                maximum_channel_difference_kelvin=float(
                    np.max(channel_difference[pixels])
                ),
                edge_distance_pixels=edge_distance,
                edge_proximity_fraction=_edge_proximity(edge_distance, height, width),
                touches_edge=edge_distance == 0,
            )
        )
    labels = np.where(np.isin(raw_labels, retained), raw_labels, 0).astype(np.int32)
    candidate_mask = np.where(labels > 0, 255, 0).astype(np.uint8)
    candidate_mask[~valid] = 0
    labels[~valid] = 0
    contours_raw, _ = cv2.findContours(
        candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
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
    return candidate_mask, labels, tuple(component_values), contours


def _edge_proximity(edge_distance: int, height: int, width: int) -> float:
    maximum_distance = max(min(height, width) // 2, 1)
    return float(1.0 - min(edge_distance / maximum_distance, 1.0))


def _overlay(
    channel7: FloatArray,
    valid: MaskArray,
    candidate_mask: Uint8Array,
    contours: tuple[Contour, ...],
    components: tuple[ContextualAnomalyComponent, ...],
) -> Uint8Array:
    display = _display(channel7, valid)
    overlay = cv2.applyColorMap(display, cv2.COLORMAP_INFERNO)
    overlay[~valid] = 0
    contour_arrays = [
        np.asarray(contour, dtype=np.int32).reshape(-1, 1, 2) for contour in contours
    ]
    if contour_arrays:
        cv2.drawContours(overlay, contour_arrays, -1, (0, 255, 255), 1, cv2.LINE_8)
    for component in components:
        centroid = tuple(int(round(value)) for value in component.centroid_xy)
        cv2.drawMarker(
            overlay,
            centroid,
            (255, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=5,
            thickness=1,
            line_type=cv2.LINE_8,
        )
        cv2.putText(
            overlay,
            str(component.label),
            (centroid[0] + 2, centroid[1] - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_8,
        )
    overlay[candidate_mask > 0] = np.maximum(
        overlay[candidate_mask > 0], np.array((0, 180, 255), dtype=np.uint8)
    )
    return np.asarray(overlay, dtype=np.uint8)


def _display(values: FloatArray, valid: MaskArray) -> Uint8Array:
    display = np.zeros(values.shape, dtype=np.uint8)
    if not np.any(valid):
        return display
    valid_values = values[valid]
    lower, upper = np.quantile(valid_values, (0.02, 0.98))
    if upper <= lower:
        lower = float(np.min(valid_values))
        upper = float(np.max(valid_values))
    if upper <= lower:
        display[valid] = 127
        return display
    scaled = np.clip((values[valid] - lower) / (upper - lower), 0.0, 1.0)
    display[valid] = np.rint(scaled * 255.0).astype(np.uint8)
    return display


def _readonly[ArrayScalar: np.generic](
    array: npt.NDArray[ArrayScalar],
) -> npt.NDArray[ArrayScalar]:
    copied = np.ascontiguousarray(array).copy()
    copied.setflags(write=False)
    return copied


__all__ = [
    "DEVELOPMENT_CONTEXTUAL_ANOMALY_PARAMETERS",
    "ContextualAnomalyComponent",
    "ContextualAnomalyParameters",
    "ContextualAnomalyResult",
    "extract_contextual_anomalies",
]
