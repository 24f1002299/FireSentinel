"""Golden tests for the Day 5 deterministic thermal fixture bundle."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from firesentinel.core.records import ReasonCode
from firesentinel.vision.fixtures import (
    FIXTURE_SEED,
    FIXTURE_SHAPE,
    NUMERIC_TOLERANCES,
    FixtureIntegrityError,
    SyntheticFixtureBundle,
    fixture_bundle_digest,
    generate_synthetic_fixture_bundle,
    load_offline_fixture_bundle,
    offline_fixture_manifest,
    verify_fixture_bundle,
)


def test_repeated_fixture_generation_is_byte_deterministic() -> None:
    first = generate_synthetic_fixture_bundle()
    second = generate_synthetic_fixture_bundle()

    assert first.seed == second.seed == FIXTURE_SEED
    assert first.shape == second.shape == FIXTURE_SHAPE
    assert first.tolerances == second.tolerances == NUMERIC_TOLERANCES
    assert fixture_bundle_digest(first) == fixture_bundle_digest(second)
    for first_case, second_case in zip(first.cases, second.cases, strict=True):
        assert first_case.name == second_case.name
        for first_frame, second_frame in zip(
            first_case.thermal_frames, second_case.thermal_frames, strict=True
        ):
            assert np.array_equal(first_frame, second_frame)
        for first_mask, second_mask in zip(
            first_case.expected_heat_masks,
            second_case.expected_heat_masks,
            strict=True,
        ):
            assert np.array_equal(first_mask, second_mask)


def test_offline_bundle_has_all_required_conditions_and_expectations() -> None:
    bundle = load_offline_fixture_bundle()

    assert {fixture_case.name for fixture_case in bundle.cases} == {
        "persistent_heat",
        "transient_heat",
        "image_shift",
        "missing_pixels",
        "saturated_pixels",
        "low_contrast",
        "empty_frame",
    }
    persistent = bundle.case("persistent_heat")
    assert persistent.expected_components[0][0].area_pixels == 4
    assert persistent.expected_persistence is not None
    assert persistent.expected_persistence.aligned_iou == pytest.approx(
        1.0,
        abs=NUMERIC_TOLERANCES.absolute,
        rel=NUMERIC_TOLERANCES.relative,
    )
    shifted = bundle.case("image_shift")
    assert shifted.expected_persistence is not None
    assert shifted.expected_persistence.translation_xy == (-2, 1)
    assert shifted.expected_persistence.unaligned_iou == 0.0
    assert shifted.expected_persistence.aligned_iou == 1.0
    assert bundle.case("missing_pixels").expected_reason_codes == (
        ReasonCode.COVERAGE_INSUFFICIENT,
    )
    assert bundle.case("saturated_pixels").expected_reason_codes == (
        ReasonCode.FRAME_SATURATED,
    )
    assert bundle.case("low_contrast").expected_reason_codes == (
        ReasonCode.CONTRAST_LOW,
    )
    assert bundle.case("empty_frame").expected_reason_codes == (ReasonCode.FRAME_BLANK,)


def test_checked_in_manifest_pins_bundle_configuration_and_fingerprint() -> None:
    bundle = load_offline_fixture_bundle()
    manifest = offline_fixture_manifest()

    assert manifest["seed"] == FIXTURE_SEED
    assert manifest["shape"] == list(FIXTURE_SHAPE)
    assert manifest["numeric_tolerances"] == {
        "absolute": NUMERIC_TOLERANCES.absolute,
        "relative": NUMERIC_TOLERANCES.relative,
    }
    assert manifest["bundle_digest"] == fixture_bundle_digest(bundle)


def test_deliberate_fixture_corruption_fails_with_source_corrupt_reason() -> None:
    bundle = generate_synthetic_fixture_bundle()
    original = bundle.case("persistent_heat")
    corrupted_frame = original.thermal_frames[0].copy()
    corrupted_frame[0, 0] += np.float32(1.0)
    corrupted_case = replace(
        original,
        thermal_frames=(corrupted_frame, *original.thermal_frames[1:]),
    )
    corrupted_bundle = SyntheticFixtureBundle(
        version=bundle.version,
        seed=bundle.seed,
        shape=bundle.shape,
        tolerances=bundle.tolerances,
        cases=tuple(
            corrupted_case if fixture_case.name == corrupted_case.name else fixture_case
            for fixture_case in bundle.cases
        ),
    )

    with pytest.raises(FixtureIntegrityError) as error:
        verify_fixture_bundle(corrupted_bundle, offline_fixture_manifest())

    assert error.value.reason_code is ReasonCode.SOURCE_CORRUPT
