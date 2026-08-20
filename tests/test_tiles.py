"""Golden contracts for mask-aware calibrated-tile preparation."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from firesentinel.data.goes_crop import extract_calibrated_crop
from firesentinel.vision.tiles import (
    TilePreparationParameters,
    prepare_calibrated_tile,
    prepare_tile,
)
from tests.test_goes_crop import _parameters_at, _source


def _fixture() -> tuple[np.ndarray, np.ndarray]:
    values = np.array(
        [
            [250.0, 260.0, 270.0, 280.0],
            [290.0, np.nan, 310.0, 320.0],
            [330.0, 340.0, 350.0, 360.0],
        ],
        dtype=np.float32,
    )
    invalid = np.zeros(values.shape, dtype=bool)
    invalid[1, 1] = True
    return values, invalid


def test_physical_clip_preserves_calibration_and_golden_ordered_display() -> None:
    values, invalid = _fixture()
    original = values.copy()
    parameters = TilePreparationParameters(
        260.0,
        340.0,
        0.0,
        1.0,
    )

    tile = prepare_calibrated_tile(
        values,
        invalid,
        parameters,
        source_crop_checksum="a" * 64,
        source_timing={
            "start": "2025-01-01T00:00:00Z",
            "end": "2025-01-01T00:09:50Z",
        },
    )

    assert np.array_equal(values, original, equal_nan=True)
    np.testing.assert_allclose(
        tile.physical_clipped,
        np.array(
            [
                [260.0, 260.0, 270.0, 280.0],
                [290.0, np.nan, 310.0, 320.0],
                [330.0, 340.0, 340.0, 340.0],
            ],
            dtype=np.float32,
        ),
        rtol=0.0,
        atol=1e-6,
        equal_nan=True,
    )
    np.testing.assert_array_equal(
        tile.display,
        np.array(
            [
                [0, 0, 32, 64],
                [96, 0, 159, 191],
                [223, 255, 255, 255],
            ],
            dtype=np.uint8,
        ),
    )
    assert tile.content_checksum == (
        "c12080b48b8ef5a750866658911cc3d84f1ec520df2d5e88f37fc745b7b8a832"
    )
    assert tile.physical_low_clip_mask[0, 0]
    assert tile.physical_high_clip_mask[2, 2]
    assert tile.physical_high_clip_mask[2, 3]
    assert not tile.physical_low_clip_mask[1, 1]

    valid_values = tile.physical_clipped[tile.valid_mask]
    display_values = tile.display[tile.valid_mask]
    order = np.argsort(valid_values, kind="stable")
    assert np.all(np.diff(display_values[order]) >= 0)


def test_invalid_pixels_are_never_interpolated_into_evidence() -> None:
    values = np.array([[300.0, 300.0], [300.0, np.nan]], dtype=np.float32)
    invalid = np.array([[False, False], [False, True]])
    tile = prepare_calibrated_tile(
        values,
        invalid,
        TilePreparationParameters(
            250.0,
            350.0,
            0.0,
            1.0,
            target_shape=(4, 4),
            minimum_valid_coverage=0.5,
        ),
    )

    np.testing.assert_allclose(
        tile.resized_calibrated[~tile.resized_invalid_mask],
        300.0,
        rtol=0.0,
        atol=1e-6,
    )
    assert np.all(np.isnan(tile.resized_calibrated[tile.resized_invalid_mask]))
    assert np.all(tile.display[tile.resized_invalid_mask] == 0)
    assert np.all(~tile.valid_mask[tile.resized_invalid_mask])
    np.testing.assert_array_equal(
        tile.resized_invalid_mask,
        np.array(
            [
                [False, False, False, False],
                [False, False, False, False],
                [False, False, True, True],
                [False, False, True, True],
            ]
        ),
    )


def test_optional_clahe_is_display_only_and_metadata_is_complete() -> None:
    values, invalid = _fixture()
    baseline = prepare_calibrated_tile(
        values,
        invalid,
        TilePreparationParameters(260.0, 340.0, 0.1, 0.9, target_shape=(6, 8)),
    )
    tile = prepare_calibrated_tile(
        values,
        invalid,
        TilePreparationParameters(
            260.0,
            340.0,
            0.1,
            0.9,
            target_shape=(6, 8),
            clahe_clip_limit=2.0,
            clahe_tile_grid_size=(2, 2),
        ),
    )

    assert tile.clahe_display is not None
    np.testing.assert_array_equal(tile.display, baseline.display)
    assert np.all(tile.clahe_display[tile.resized_invalid_mask] == 0)
    metadata = tile.metadata()
    assert metadata["parameters"] == tile.parameters.to_dict()
    ranges = metadata["ranges_kelvin"]
    assert isinstance(ranges, dict)
    assert ranges["physical"] == {
        "minimum": 260.0,
        "maximum": 340.0,
    }
    masks = metadata["masks"]
    opencv = metadata["opencv"]
    timings = metadata["timings_milliseconds"]
    assert isinstance(masks, dict)
    assert isinstance(opencv, dict)
    assert isinstance(timings, dict)
    assert masks["invalid_pixels_are_excluded_from_evidence"]
    assert opencv["version"] == cv2.__version__
    build_hash = opencv["build_information_sha256"]
    assert isinstance(build_hash, str)
    assert len(build_hash) == 64
    assert set(timings) == {
        "physical_clip",
        "mask_aware_resize",
        "display_scale",
        "optional_clahe",
        "total",
    }
    assert all(isinstance(value, float) and value >= 0.0 for value in timings.values())


def test_crop_wrapper_carries_calibrated_crop_checksum_and_scan_timing(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.nc"
    _source(source_path)
    crop = extract_calibrated_crop(source_path, _parameters_at(2, 2))

    tile = prepare_tile(crop, TilePreparationParameters(200.0, 240.0, 0.0, 1.0))

    assert tile.source_crop_checksum == crop.content_checksum
    assert tile.source_timing == crop.timing.to_dict()
    assert tile.calibrated.shape == crop.calibrated.shape


@pytest.mark.parametrize(
    "factory",
    [
        lambda: TilePreparationParameters(300.0, 300.0),
        lambda: TilePreparationParameters(250.0, 350.0, 0.9, 0.1),
    ],
)
def test_invalid_tile_parameters_fail_fast(factory: object) -> None:
    assert callable(factory)
    with pytest.raises(ValueError):
        factory()
