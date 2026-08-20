"""Contracts for geospatially aligned temporal persistence measurements."""

from __future__ import annotations

import numpy as np
import pytest

from firesentinel.vision.fixtures import load_offline_fixture_bundle
from firesentinel.vision.persistence import (
    GeospatialGrid,
    PersistenceParameters,
    TemporalObservation,
    measure_temporal_persistence,
)


def _grid(
    shape: tuple[int, int] = (12, 12),
    *,
    latitude_offset: int = 0,
    longitude_offset: int = 0,
) -> GeospatialGrid:
    rows, columns = shape
    latitude = np.broadcast_to(
        (np.arange(rows, dtype=np.float64) + latitude_offset)[:, None] * 0.01,
        shape,
    )
    longitude = np.broadcast_to(
        (np.arange(columns, dtype=np.float64) + longitude_offset)[None, :] * 0.01,
        shape,
    )
    return GeospatialGrid(latitude, longitude)


def _observation(
    observation_id: str,
    grid: GeospatialGrid,
    region: tuple[slice, slice] | None,
    *,
    temperature: float = 320.0,
) -> TemporalObservation:
    values = np.full(grid.shape, 290.0, dtype=np.float32)
    mask = np.zeros(grid.shape, dtype=np.uint8)
    if region is not None:
        rows, columns = region
        values[rows, columns] = temperature
        mask[rows, columns] = 255
    return TemporalObservation(
        observation_id,
        mask,
        values,
        np.zeros(grid.shape, dtype=bool),
        grid,
    )


def test_geospatial_alignment_matches_a_displaced_pixel_response_on_common_grid() -> (
    None
):
    first_grid = _grid()
    second_grid = _grid(latitude_offset=-1, longitude_offset=2)
    first = _observation("first", first_grid, (slice(3, 7), slice(4, 8)))
    second = _observation(
        "second",
        second_grid,
        (slice(4, 8), slice(2, 6)),
        temperature=322.0,
    )

    result = measure_temporal_persistence((first, second))

    assert result.persistence_count == 2
    assert result.mean_intersection_over_union == pytest.approx(1.0)
    assert result.confidence == pytest.approx(1.0)
    assert result.area_trend_pixels_per_observation == pytest.approx(0.0)
    assert result.temperature_trend_kelvin_per_observation == pytest.approx(2.0)
    assert not result.disappeared
    assert len(result.matches) == 1
    assert result.matches[0].centroid_distance_kilometres == pytest.approx(0.0)
    assert result.aligned_observations[0] is not None
    assert result.aligned_observations[1] is not None
    np.testing.assert_array_equal(
        result.aligned_observations[0].candidate_mask,
        result.aligned_observations[1].candidate_mask,
    )


def test_persistent_response_outscores_transient_disappearance() -> None:
    grid = _grid()
    first = _observation("first", grid, (slice(3, 7), slice(4, 8)))
    persistent = _observation("persistent", grid, (slice(3, 7), slice(4, 8)))
    transient = _observation("transient", grid, None)

    sustained = measure_temporal_persistence((first, persistent))
    disappeared = measure_temporal_persistence((first, transient))

    assert sustained.confidence > disappeared.confidence
    assert sustained.persistence_count == 2
    assert disappeared.persistence_count == 1
    assert disappeared.disappeared
    assert disappeared.confidence == 0.0


def test_checked_in_persistent_fixture_outscores_the_transient_fixture() -> None:
    bundle = load_offline_fixture_bundle()
    grid = _grid(bundle.shape)

    def observations(case_name: str) -> tuple[TemporalObservation, ...]:
        case = bundle.case(case_name)
        return tuple(
            TemporalObservation(
                f"{case_name}-{index}",
                heat_mask.astype(np.uint8) * 255,
                frame,
                ~valid_mask,
                grid,
            )
            for index, (frame, valid_mask, heat_mask) in enumerate(
                zip(
                    case.thermal_frames,
                    case.valid_masks,
                    case.expected_heat_masks,
                    strict=True,
                )
            )
        )

    persistent = measure_temporal_persistence(observations("persistent_heat"))
    transient = measure_temporal_persistence(observations("transient_heat"))

    assert persistent.persistence_count == 2
    assert persistent.confidence == pytest.approx(1.0)
    assert transient.persistence_count == 1
    assert transient.confidence == 0.0


def test_missing_observation_breaks_continuity_without_declaring_disappearance() -> (
    None
):
    grid = _grid()
    first = _observation("first", grid, (slice(3, 7), slice(4, 8)))
    later = _observation("later", grid, (slice(3, 7), slice(4, 8)))

    result = measure_temporal_persistence((first, None, later))

    assert result.missing_observation_count == 1
    assert result.matches == ()
    assert result.persistence_count == 1
    assert result.confidence == 0.0
    assert not result.disappeared
    assert len(result.tracks) == 2


def test_poor_overlap_produces_low_persistence_confidence() -> None:
    grid = _grid()
    first = _observation("first", grid, (slice(3, 7), slice(3, 7)))
    later = _observation("later", grid, (slice(3, 7), slice(6, 10)))
    parameters = PersistenceParameters(
        maximum_centroid_distance_kilometres=10.0,
        minimum_intersection_over_union=0.01,
    )

    result = measure_temporal_persistence((first, later), parameters)

    assert result.persistence_count == 2
    assert result.mean_intersection_over_union == pytest.approx(4.0 / 28.0)
    assert 0.0 < result.confidence < 0.2


@pytest.mark.parametrize(
    "factory",
    [
        lambda: PersistenceParameters(maximum_resample_distance_kilometres=0.0),
        lambda: PersistenceParameters(maximum_centroid_distance_kilometres=-1.0),
        lambda: PersistenceParameters(minimum_intersection_over_union=1.1),
        lambda: PersistenceParameters(minimum_component_area_pixels=0),
    ],
)
def test_invalid_persistence_parameters_fail_fast(factory: object) -> None:
    assert callable(factory)
    with pytest.raises(ValueError):
        factory()


def test_invalid_candidate_pixels_are_rejected_at_the_observation_boundary() -> None:
    grid = _grid()
    mask = np.zeros(grid.shape, dtype=np.uint8)
    mask[3, 3] = 255
    invalid = np.zeros(grid.shape, dtype=bool)
    invalid[3, 3] = True

    with pytest.raises(ValueError, match="cannot be candidates"):
        TemporalObservation(
            "invalid-candidate",
            mask,
            np.full(grid.shape, 300.0, dtype=np.float32),
            invalid,
            grid,
        )
