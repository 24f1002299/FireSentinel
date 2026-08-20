"""Contracts for Day 14 observation-quality measurements and gating."""

from __future__ import annotations

import numpy as np
import pytest

from firesentinel.core.records import ReasonCode
from firesentinel.vision.fixtures import load_offline_fixture_bundle
from firesentinel.vision.quality import (
    DEVELOPMENT_QUALITY_THRESHOLDS,
    THRESHOLD_SELECTION_SCOPE,
    ObservationQualityThresholds,
    apply_quality_gate,
    measure_observation_quality,
    measure_prepared_tile_quality,
)
from firesentinel.vision.tiles import (
    TilePreparationParameters,
    prepare_calibrated_tile,
)


@pytest.mark.parametrize(
    ("case_name", "reason"),
    [
        ("empty_frame", ReasonCode.FRAME_BLANK),
        ("saturated_pixels", ReasonCode.FRAME_SATURATED),
        ("missing_pixels", ReasonCode.COVERAGE_INSUFFICIENT),
        ("low_contrast", ReasonCode.CONTRAST_LOW),
    ],
)
def test_poor_quality_fixtures_emit_the_expected_reason_and_block_evidence(
    case_name: str, reason: ReasonCode
) -> None:
    case = load_offline_fixture_bundle().case(case_name)
    quality = measure_observation_quality(case.thermal_frames[0], ~case.valid_masks[0])
    apparent_fire = np.full(case.thermal_frames[0].shape, 255, dtype=np.uint8)

    assert reason in quality.reason_codes
    assert quality.fire_evidence_confidence_cap == 0.0
    assert not quality.usable_for_fire_evidence
    assert quality.cap_fire_evidence_confidence(0.99) == 0.0
    assert not np.any(apply_quality_gate(apparent_fire, quality))


def test_quality_fields_are_bounded_and_include_mask_aware_texture_statistics() -> None:
    case = load_offline_fixture_bundle().case("persistent_heat")
    quality = measure_observation_quality(case.thermal_frames[0], ~case.valid_masks[0])
    payload = quality.to_dict()

    assert quality.reason_codes == ()
    assert quality.usable_for_fire_evidence
    assert quality.cap_fire_evidence_confidence(0.82) == 0.82
    assert np.array_equal(
        apply_quality_gate(case.expected_heat_masks[0], quality),
        case.expected_heat_masks[0],
    )
    for field in (
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
        value = payload[field]
        assert isinstance(value, float)
        bounded = value
        assert 0.0 <= bounded <= 1.0
    assert quality.texture_standard_deviation_kelvin > 0.0
    assert quality.mean_absolute_neighbor_difference_kelvin > 0.0


def test_synthetic_quality_expectations_cover_missingness_and_texture() -> None:
    case = load_offline_fixture_bundle().case("persistent_heat")
    expected = case.expected_quality[0]
    actual = measure_observation_quality(case.thermal_frames[0], ~case.valid_masks[0])

    assert actual.usable_coverage_fraction == expected.coverage_fraction
    assert actual.missing_pixel_fraction == expected.missing_fraction
    assert actual.saturated_pixel_fraction == expected.saturated_fraction
    assert actual.contrast_span_kelvin == expected.contrast_span
    assert actual.texture_standard_deviation_kelvin == (
        expected.texture_standard_deviation
    )
    assert actual.mean_absolute_neighbor_difference_kelvin == (
        expected.mean_absolute_neighbor_difference
    )


def test_nonfinite_unmasked_samples_count_as_missing_not_as_thermal_evidence() -> None:
    frame = np.full((4, 4), 325.0, dtype=np.float32)
    frame[:3, :] = np.nan
    quality = measure_observation_quality(frame, np.zeros(frame.shape, dtype=bool))

    assert quality.missing_pixel_fraction == pytest.approx(0.75)
    assert quality.usable_coverage_fraction == pytest.approx(0.25)
    assert quality.reason_codes == (ReasonCode.COVERAGE_INSUFFICIENT,)
    assert quality.fire_evidence_confidence_cap == 0.0


def test_clipped_mask_counts_as_saturation_even_after_physical_values_are_changed() -> (
    None
):
    frame = np.full((10, 10), 300.0, dtype=np.float32)
    clipped = np.zeros(frame.shape, dtype=bool)
    clipped[:2, :] = True

    quality = measure_observation_quality(
        frame, np.zeros(frame.shape, dtype=bool), clipped_mask=clipped
    )

    assert quality.saturated_pixel_fraction == pytest.approx(0.2)
    assert quality.reason_codes == (ReasonCode.FRAME_SATURATED,)


def test_prepared_tile_quality_retains_the_preprocessing_clip_reason() -> None:
    frame = np.full((10, 10), 300.0, dtype=np.float32)
    frame[:2, :] = 390.0
    tile = prepare_calibrated_tile(
        frame,
        np.zeros(frame.shape, dtype=bool),
        TilePreparationParameters(250.0, 340.0),
    )
    thresholds = ObservationQualityThresholds(saturation_maximum_kelvin=400.0)

    quality = measure_prepared_tile_quality(tile, thresholds)

    assert quality.saturated_pixel_fraction == pytest.approx(0.2)
    assert quality.reason_codes == (ReasonCode.FRAME_SATURATED,)


def test_thresholds_are_pinned_to_development_and_synthetic_fixture_scope() -> None:
    assert DEVELOPMENT_QUALITY_THRESHOLDS.to_dict()["selection_scope"] == (
        THRESHOLD_SELECTION_SCOPE
    )
    assert THRESHOLD_SELECTION_SCOPE == "development_cases_and_synthetic_fixtures_only"


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ObservationQualityThresholds(minimum_usable_coverage_fraction=0.0),
        lambda: ObservationQualityThresholds(maximum_saturated_fraction=0.0),
        lambda: ObservationQualityThresholds(saturation_maximum_kelvin=None),
    ],
)
def test_invalid_quality_thresholds_fail_fast(factory: object) -> None:
    assert callable(factory)
    with pytest.raises(ValueError):
        factory()


def test_empty_or_mismatched_quality_arrays_fail_fast() -> None:
    with pytest.raises(ValueError, match="at least one pixel"):
        measure_observation_quality(
            np.empty((0, 0), dtype=np.float32), np.empty((0, 0), dtype=bool)
        )
    with pytest.raises(ValueError, match="matching 2D arrays"):
        measure_observation_quality(
            np.zeros((2, 2), dtype=np.float32), np.zeros((2, 1), dtype=bool)
        )
