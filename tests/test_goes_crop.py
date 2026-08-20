"""Offline contracts for calibrated, deterministic regional GOES crops."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from netCDF4 import Dataset
from pyproj import CRS, Transformer

from firesentinel.data.goes_crop import (
    CropArtifactError,
    CropParameters,
    GeographicBounds,
    extract_calibrated_crop,
    load_calibrated_crop,
    save_calibrated_crop,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array has been deprecated:DeprecationWarning"
)

_HEIGHT = 35_786_023.0
_LON_ORIGIN = -137.2
_SEMI_MAJOR = 6_378_137.0
_SEMI_MINOR = 6_356_752.31414
_X = np.linspace(-0.03, 0.03, 9, dtype=np.float64)
_Y = np.linspace(0.03, -0.03, 7, dtype=np.float64)


def _projection() -> CRS:
    return CRS.from_cf(
        {
            "grid_mapping_name": "geostationary",
            "perspective_point_height": _HEIGHT,
            "longitude_of_projection_origin": _LON_ORIGIN,
            "latitude_of_projection_origin": 0.0,
            "semi_major_axis": _SEMI_MAJOR,
            "semi_minor_axis": _SEMI_MINOR,
            "sweep_angle_axis": "x",
        }
    )


def _latitude_longitude(row: int, column: int) -> tuple[float, float]:
    transformer = Transformer.from_crs(_projection(), "EPSG:4326", always_xy=True)
    longitude, latitude = transformer.transform(_X[column] * _HEIGHT, _Y[row] * _HEIGHT)
    return float(latitude), float(longitude)


def _source(path: Path) -> None:
    raw = np.arange(_X.size * _Y.size, dtype=np.int16).reshape(_Y.size, _X.size)
    raw[0, 0] = -999
    raw[-1, -1] = 61  # Above valid_range and therefore invalid.
    with Dataset(path, "w") as dataset:
        dataset.createDimension("y", _Y.size)
        dataset.createDimension("x", _X.size)
        dataset.time_coverage_start = "2025-01-01T00:00:00Z"
        dataset.time_coverage_end = "2025-01-01T00:09:50Z"
        x = dataset.createVariable("x", "f8", ("x",))
        y = dataset.createVariable("y", "f8", ("y",))
        x.units = "rad"
        y.units = "rad"
        x[:] = _X
        y[:] = _Y
        projection = dataset.createVariable("goes_imager_projection", "i1")
        projection.grid_mapping_name = "geostationary"
        projection.perspective_point_height = _HEIGHT
        projection.longitude_of_projection_origin = _LON_ORIGIN
        projection.latitude_of_projection_origin = 0.0
        projection.semi_major_axis = _SEMI_MAJOR
        projection.semi_minor_axis = _SEMI_MINOR
        projection.sweep_angle_axis = "x"
        cmi = dataset.createVariable("CMI", "i2", ("y", "x"), fill_value=-999)
        cmi.grid_mapping = "goes_imager_projection"
        cmi.units = "K"
        cmi.valid_range = np.array((0, 60), dtype=np.int16)
        cmi.set_auto_maskandscale(False)
        cmi[:, :] = raw.tolist()
        cmi.scale_factor = 0.5
        cmi.add_offset = 200.0
        dqf = dataset.createVariable("DQF", "u1", ("y", "x"), fill_value=255)
        dqf.flag_values = np.array((0, 1, 2, 3), dtype=np.uint8)
        dqf.flag_meanings = "good conditionally_usable out_of_range no_value"
        dqf.set_auto_maskandscale(False)
        quality = np.zeros(raw.shape, dtype=np.uint8)
        quality[2, 3] = 2
        quality[3, 3] = 1
        dqf[:, :] = quality.tolist()


@pytest.fixture
def source_path(tmp_path: Path) -> Path:
    path = tmp_path / "cached-goes.nc"
    _source(path)
    return path


def _parameters_at(row: int, column: int, padding_pixels: int = 1) -> CropParameters:
    latitude, longitude = _latitude_longitude(row, column)
    return CropParameters(
        GeographicBounds(
            south=latitude - 0.08,
            west=longitude - 0.08,
            north=latitude + 0.08,
            east=longitude + 0.08,
        ),
        padding_pixels=padding_pixels,
    )


def test_crop_calibrates_masks_and_locates_reference_pixel(source_path: Path) -> None:
    crop = extract_calibrated_crop(source_path, _parameters_at(2, 2))

    reference_latitude, reference_longitude = _latitude_longitude(2, 2)
    reference = crop.nearest_pixel(reference_latitude, reference_longitude)
    assert (reference.source_row, reference.source_column) == (2, 2)
    assert reference.latitude == pytest.approx(reference_latitude, abs=1e-6)
    assert reference.longitude == pytest.approx(reference_longitude, abs=1e-6)
    assert crop.calibration.scale_factor == 0.5
    assert crop.calibration.add_offset == 200.0
    assert crop.calibrated[reference.row, reference.column] == pytest.approx(210.0)
    unusable = crop.nearest_pixel(*_latitude_longitude(2, 3))
    assert crop.invalid_mask[unusable.row, unusable.column]
    assert np.isnan(crop.calibrated[unusable.row, unusable.column])
    conditional = crop.nearest_pixel(*_latitude_longitude(3, 3))
    assert not crop.invalid_mask[conditional.row, conditional.column]
    assert crop.calibrated[conditional.row, conditional.column] == pytest.approx(215.0)
    assert crop.timing.start == datetime(2025, 1, 1, tzinfo=UTC)
    assert crop.timing.end == datetime(2025, 1, 1, 0, 9, 50, tzinfo=UTC)


def test_crop_clips_padding_at_the_source_edge_without_wrapping(
    source_path: Path,
) -> None:
    crop = extract_calibrated_crop(source_path, _parameters_at(0, 1, padding_pixels=4))

    assert crop.source_window.row_start == 0
    assert crop.source_window.column_start == 0
    assert crop.source_window.row_stop <= _Y.size
    assert crop.source_window.column_stop <= _X.size
    assert crop.x[0] == _X[0]
    assert crop.y[0] == _Y[0]
    assert crop.invalid_mask[0, 0]
    assert np.isnan(crop.calibrated[0, 0])


def test_repeated_crops_and_artifacts_are_identical_and_checksum_verified(
    source_path: Path, tmp_path: Path
) -> None:
    parameters = _parameters_at(3, 5, padding_pixels=2)
    first = extract_calibrated_crop(source_path, parameters)
    second = extract_calibrated_crop(source_path, parameters)

    assert first.content_checksum == second.content_checksum
    assert np.array_equal(first.calibrated, second.calibrated, equal_nan=True)
    np.testing.assert_array_equal(first.invalid_mask, second.invalid_mask)
    first_path = save_calibrated_crop(first, tmp_path / "first.npz")
    second_path = save_calibrated_crop(second, tmp_path / "second.npz")
    assert first_path.read_bytes() == second_path.read_bytes()
    restored = load_calibrated_crop(first_path)
    assert restored.content_checksum == first.content_checksum
    assert np.array_equal(restored.calibrated, first.calibrated, equal_nan=True)
    np.testing.assert_array_equal(restored.latitude, first.latitude)


def test_tampered_artifact_is_rejected(source_path: Path, tmp_path: Path) -> None:
    crop = extract_calibrated_crop(source_path, _parameters_at(3, 5))
    artifact = save_calibrated_crop(crop, tmp_path / "crop.npz")
    contents = artifact.read_bytes()
    artifact.write_bytes(contents[:-15] + b"modified artifact")

    with pytest.raises(CropArtifactError):
        load_calibrated_crop(artifact)
