"""Contracts for contextual Channel 7 / Channel 14 anomaly extraction."""

from __future__ import annotations

import numpy as np
import pytest

from firesentinel.vision.anomalies import (
    ContextualAnomalyParameters,
    extract_contextual_anomalies,
)


def _channels(shape: tuple[int, int] = (25, 25)) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(20_260_820)
    channel7 = generator.normal(290.0, 0.2, shape).astype(np.float32)
    channel14 = generator.normal(280.0, 0.2, shape).astype(np.float32)
    return channel7, channel14


def _parameters(
    *, minimum_edge_distance_pixels: int = 1
) -> ContextualAnomalyParameters:
    return ContextualAnomalyParameters(
        local_background_kernel_pixels=5,
        minimum_local_contrast_kelvin=2.0,
        minimum_channel_difference_kelvin=25.0,
        morphology_kernel_pixels=3,
        minimum_component_area_pixels=8,
        minimum_edge_distance_pixels=minimum_edge_distance_pixels,
    )


def test_injected_hot_region_is_localized_and_measured_from_source_arrays() -> None:
    channel7, channel14 = _channels()
    channel7[10:15, 11:16] = 330.0
    invalid = np.zeros(channel7.shape, dtype=bool)

    result = extract_contextual_anomalies(
        channel7, channel14, invalid, invalid, _parameters()
    )

    assert result.usable_for_fire_evidence
    assert result.reason_codes == ()
    assert len(result.components) == 1
    component = result.components[0]
    pixels = result.labels == component.label
    assert component.area_pixels == int(np.count_nonzero(pixels))
    assert component.bounding_box_xywh == (11, 10, 5, 5)
    assert component.centroid_xy == pytest.approx((13.0, 12.0))
    assert component.edge_distance_pixels == 9
    assert not component.touches_edge
    assert component.mean_local_contrast_kelvin == pytest.approx(
        float(np.mean(result.local_contrast_kelvin[pixels]))
    )
    assert component.maximum_local_contrast_kelvin == pytest.approx(
        float(np.max(result.local_contrast_kelvin[pixels]))
    )
    assert component.mean_channel_difference_kelvin == pytest.approx(
        float(np.mean(channel7[pixels] - channel14[pixels]))
    )
    assert component.maximum_channel_difference_kelvin == pytest.approx(
        float(np.max(channel7[pixels] - channel14[pixels]))
    )
    assert result.candidate_mask.dtype == np.uint8
    assert result.overlay.shape == (*channel7.shape, 3)
    assert result.overlay.dtype == np.uint8
    assert np.any(result.overlay[pixels])
    assert len(result.contours) == 1


def test_isolated_noise_and_invalid_hot_pixels_are_rejected() -> None:
    channel7, channel14 = _channels()
    channel7[5, 5] = 335.0
    channel7[16:21, 16:21] = 335.0
    channel14[16:21, 16:21] = 280.0
    channel7_invalid = np.zeros(channel7.shape, dtype=bool)
    channel14_invalid = np.zeros(channel7.shape, dtype=bool)
    channel7_invalid[16:21, 16:21] = True

    result = extract_contextual_anomalies(
        channel7,
        channel14,
        channel7_invalid,
        channel14_invalid,
        _parameters(),
    )

    assert result.components == ()
    assert not np.any(result.candidate_mask)
    assert not np.any(result.local_contrast_threshold_mask[16:21, 16:21])
    assert not np.any(result.channel_difference_threshold_mask[16:21, 16:21])
    assert np.all(np.isnan(result.local_contrast_kelvin[16:21, 16:21]))
    assert np.all(np.isnan(result.channel_difference_kelvin[16:21, 16:21]))


def test_local_heat_without_a_channel_difference_is_not_a_contextual_anomaly() -> None:
    channel7, channel14 = _channels()
    channel7[10:15, 11:16] = 330.0
    channel14[10:15, 11:16] = 320.0
    invalid = np.zeros(channel7.shape, dtype=bool)

    result = extract_contextual_anomalies(
        channel7, channel14, invalid, invalid, _parameters()
    )

    assert np.any(result.local_contrast_threshold_mask)
    assert not np.any(result.channel_difference_threshold_mask)
    assert result.components == ()
    assert not np.any(result.candidate_mask)


def test_poor_quality_channel_clears_apparent_contextual_candidates() -> None:
    channel7, channel14 = _channels()
    channel7[10:15, 11:16] = 330.0
    invalid = np.zeros(channel7.shape, dtype=bool)
    channel14_invalid = invalid.copy()
    channel14_invalid[:20, :] = True

    result = extract_contextual_anomalies(
        channel7, channel14, invalid, channel14_invalid, _parameters()
    )

    assert not result.usable_for_fire_evidence
    assert not np.any(result.candidate_mask)
    assert result.components == ()


def test_edge_regions_are_filtered_but_reported_maps_remain_valid() -> None:
    channel7, channel14 = _channels()
    channel7[0:5, 5:10] = 330.0
    invalid = np.zeros(channel7.shape, dtype=bool)

    result = extract_contextual_anomalies(
        channel7,
        channel14,
        invalid,
        invalid,
        _parameters(minimum_edge_distance_pixels=3),
    )

    assert result.components == ()
    assert not np.any(result.candidate_mask)
    assert np.any(result.morphology_mask)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ContextualAnomalyParameters(local_background_kernel_pixels=4),
        lambda: ContextualAnomalyParameters(morphology_kernel_pixels=2),
        lambda: ContextualAnomalyParameters(minimum_component_area_pixels=0),
        lambda: ContextualAnomalyParameters(minimum_edge_distance_pixels=-1),
    ],
)
def test_invalid_anomaly_parameters_fail_fast(factory: object) -> None:
    assert callable(factory)
    with pytest.raises(ValueError):
        factory()


def test_mismatched_channel_shapes_fail_fast() -> None:
    with pytest.raises(ValueError, match="matching 2D arrays"):
        extract_contextual_anomalies(
            np.zeros((4, 4), dtype=np.float32),
            np.zeros((4, 3), dtype=np.float32),
            np.zeros((4, 4), dtype=bool),
            np.zeros((4, 3), dtype=bool),
        )
