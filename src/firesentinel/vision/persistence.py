"""Geospatially aligned, interpretable persistence measurements.

Each observation supplies a candidate mask, calibrated Channel 7 temperature,
and per-pixel latitude/longitude metadata.  Masks and temperatures are first
placed on a declared common grid using nearest geospatial neighbours.  Only
adjacent, actually observed frames can be matched, so a missing observation
cannot manufacture temporal continuity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

import cv2
import numpy as np
import numpy.typing as npt

Float32Array = npt.NDArray[np.float32]
Float64Array = npt.NDArray[np.float64]
MaskArray = npt.NDArray[np.bool]
Uint8Array = npt.NDArray[np.uint8]
Int32Array = npt.NDArray[np.int32]

_EARTH_RADIUS_KM = 6_371.0088


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


@dataclass(frozen=True, slots=True)
class GeospatialGrid:
    """A two-dimensional WGS84 grid used as a common temporal reference."""

    latitude_degrees: Float64Array
    longitude_degrees: Float64Array

    def __post_init__(self) -> None:
        latitude = np.asarray(self.latitude_degrees, dtype=np.float64)
        longitude = np.asarray(self.longitude_degrees, dtype=np.float64)
        if (
            latitude.ndim != 2
            or longitude.shape != latitude.shape
            or latitude.size == 0
        ):
            raise ValueError(
                "latitude and longitude must be matching non-empty 2D arrays"
            )
        finite = np.isfinite(latitude) & np.isfinite(longitude)
        if not np.any(finite):
            raise ValueError("geospatial grid requires at least one finite coordinate")
        if np.any(np.abs(latitude[finite]) > 90.0):
            raise ValueError("latitude_degrees must be within [-90, 90]")
        if np.any(np.abs(longitude[finite]) > 180.0):
            raise ValueError("longitude_degrees must be within [-180, 180]")
        object.__setattr__(self, "latitude_degrees", _readonly(latitude))
        object.__setattr__(self, "longitude_degrees", _readonly(longitude))

    @property
    def shape(self) -> tuple[int, int]:
        """Return the immutable grid's row/column dimensions."""

        return self.latitude_degrees.shape


@dataclass(frozen=True, slots=True)
class TemporalObservation:
    """One candidate observation with calibrated C07 and geospatial metadata."""

    observation_id: str
    candidate_mask: Uint8Array
    channel7_kelvin: Float32Array
    invalid_mask: MaskArray
    grid: GeospatialGrid

    def __post_init__(self) -> None:
        if not isinstance(self.observation_id, str) or not self.observation_id:
            raise ValueError("observation_id must be a non-empty string")
        if not isinstance(self.grid, GeospatialGrid):
            raise ValueError("grid must be GeospatialGrid")
        temperatures = np.asarray(self.channel7_kelvin, dtype=np.float32)
        invalid = np.asarray(self.invalid_mask, dtype=bool)
        raw_mask = np.asarray(self.candidate_mask)
        if (
            temperatures.ndim != 2
            or invalid.shape != temperatures.shape
            or raw_mask.shape != temperatures.shape
            or temperatures.shape != self.grid.shape
        ):
            raise ValueError(
                "observation arrays and geospatial grid must share a 2D shape"
            )
        mask = np.where(raw_mask != 0, 255, 0).astype(np.uint8)
        valid: MaskArray = np.asarray(
            np.logical_and(
                np.logical_and(~invalid, np.isfinite(temperatures)),
                np.isfinite(self.grid.latitude_degrees)
                & np.isfinite(self.grid.longitude_degrees),
            ),
            dtype=bool,
        )
        if np.any(mask[~valid] != 0):
            raise ValueError("invalid or ungeolocated pixels cannot be candidates")
        object.__setattr__(self, "candidate_mask", _readonly(mask))
        object.__setattr__(self, "channel7_kelvin", _readonly(temperatures))
        object.__setattr__(
            self, "invalid_mask", _readonly(np.asarray(~valid, dtype=bool))
        )

    @property
    def valid_mask(self) -> MaskArray:
        """Return pixels eligible for geospatial resampling and matching."""

        return _readonly(np.asarray(~self.invalid_mask, dtype=bool))


@dataclass(frozen=True, slots=True)
class PersistenceParameters:
    """Explicit geospatial alignment and component-match limits."""

    maximum_resample_distance_kilometres: float = 5.0
    maximum_centroid_distance_kilometres: float = 10.0
    minimum_intersection_over_union: float = 0.1
    minimum_component_area_pixels: int = 1

    def __post_init__(self) -> None:
        resample_distance = _finite_number(
            self.maximum_resample_distance_kilometres,
            "maximum_resample_distance_kilometres",
        )
        centroid_distance = _finite_number(
            self.maximum_centroid_distance_kilometres,
            "maximum_centroid_distance_kilometres",
        )
        minimum_iou = _finite_number(
            self.minimum_intersection_over_union,
            "minimum_intersection_over_union",
        )
        if resample_distance <= 0.0 or centroid_distance <= 0.0:
            raise ValueError("geospatial distance limits must be positive")
        if not 0.0 <= minimum_iou <= 1.0:
            raise ValueError("minimum_intersection_over_union must be within [0, 1]")
        object.__setattr__(
            self, "maximum_resample_distance_kilometres", resample_distance
        )
        object.__setattr__(
            self, "maximum_centroid_distance_kilometres", centroid_distance
        )
        object.__setattr__(self, "minimum_intersection_over_union", minimum_iou)
        object.__setattr__(
            self,
            "minimum_component_area_pixels",
            _positive_integer(
                self.minimum_component_area_pixels, "minimum_component_area_pixels"
            ),
        )

    def to_dict(self) -> dict[str, float | int]:
        """Return all matching and alignment thresholds."""

        return {
            "maximum_resample_distance_kilometres": (
                self.maximum_resample_distance_kilometres
            ),
            "maximum_centroid_distance_kilometres": (
                self.maximum_centroid_distance_kilometres
            ),
            "minimum_intersection_over_union": self.minimum_intersection_over_union,
            "minimum_component_area_pixels": self.minimum_component_area_pixels,
        }


DEVELOPMENT_PERSISTENCE_PARAMETERS = PersistenceParameters()


@dataclass(frozen=True, slots=True)
class AlignedObservation:
    """One observation resampled to the result's declared common grid."""

    observation_id: str
    candidate_mask: Uint8Array
    channel7_kelvin: Float32Array
    valid_mask: MaskArray

    def __post_init__(self) -> None:
        mask = np.asarray(self.candidate_mask, dtype=np.uint8)
        temperatures = np.asarray(self.channel7_kelvin, dtype=np.float32)
        valid = np.asarray(self.valid_mask, dtype=bool)
        if (
            mask.ndim != 2
            or temperatures.shape != mask.shape
            or valid.shape != mask.shape
        ):
            raise ValueError("aligned observation arrays must share a 2D shape")
        if np.any(mask[~valid] != 0) or np.any(np.isfinite(temperatures[~valid])):
            raise ValueError(
                "invalid aligned pixels must contain no candidate or value"
            )
        object.__setattr__(self, "candidate_mask", _readonly(mask))
        object.__setattr__(self, "channel7_kelvin", _readonly(temperatures))
        object.__setattr__(self, "valid_mask", _readonly(valid))


@dataclass(frozen=True, slots=True)
class RegionMatch:
    """One one-to-one adjacent-frame component match on the common grid."""

    earlier_observation_index: int
    later_observation_index: int
    earlier_label: int
    later_label: int
    centroid_distance_kilometres: float
    intersection_over_union: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "earlier_observation_index": self.earlier_observation_index,
            "later_observation_index": self.later_observation_index,
            "earlier_label": self.earlier_label,
            "later_label": self.later_label,
            "centroid_distance_kilometres": self.centroid_distance_kilometres,
            "intersection_over_union": self.intersection_over_union,
        }


@dataclass(frozen=True, slots=True)
class PersistenceTrack:
    """A sequence of matched regions and its measured temporal changes."""

    track_id: int
    observation_indexes: tuple[int, ...]
    component_labels: tuple[int, ...]
    persistence_count: int
    mean_intersection_over_union: float
    area_trend_pixels_per_observation: float | None
    temperature_trend_kelvin_per_observation: float | None
    disappeared: bool
    disappearance_observation_index: int | None

    def to_dict(self) -> dict[str, object]:
        return {
            "track_id": self.track_id,
            "observation_indexes": list(self.observation_indexes),
            "component_labels": list(self.component_labels),
            "persistence_count": self.persistence_count,
            "mean_intersection_over_union": self.mean_intersection_over_union,
            "area_trend_pixels_per_observation": self.area_trend_pixels_per_observation,
            "temperature_trend_kelvin_per_observation": (
                self.temperature_trend_kelvin_per_observation
            ),
            "disappeared": self.disappeared,
            "disappearance_observation_index": self.disappearance_observation_index,
        }


@dataclass(frozen=True, slots=True)
class TemporalPersistenceResult:
    """Common-grid regions, tracks, and primary persistence measurements."""

    common_grid: GeospatialGrid
    aligned_observations: tuple[AlignedObservation | None, ...]
    matches: tuple[RegionMatch, ...]
    tracks: tuple[PersistenceTrack, ...]
    missing_observation_count: int
    persistence_count: int
    mean_intersection_over_union: float
    area_trend_pixels_per_observation: float | None
    temperature_trend_kelvin_per_observation: float | None
    disappeared: bool
    confidence: float

    def __post_init__(self) -> None:
        if not isinstance(self.common_grid, GeospatialGrid):
            raise ValueError("common_grid must be GeospatialGrid")
        if not self.aligned_observations:
            raise ValueError("aligned_observations must not be empty")
        if self.missing_observation_count != sum(
            observation is None for observation in self.aligned_observations
        ):
            raise ValueError("missing_observation_count does not match observations")
        if self.persistence_count < 0:
            raise ValueError("persistence_count must be non-negative")
        confidence = _finite_number(self.confidence, "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be within [0, 1]")
        iou = _finite_number(
            self.mean_intersection_over_union, "mean_intersection_over_union"
        )
        if not 0.0 <= iou <= 1.0:
            raise ValueError("mean_intersection_over_union must be within [0, 1]")
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "mean_intersection_over_union", iou)

    def to_dict(self) -> dict[str, object]:
        """Return direct, reviewer-facing persistence measurements."""

        return {
            "missing_observation_count": self.missing_observation_count,
            "persistence_count": self.persistence_count,
            "mean_intersection_over_union": self.mean_intersection_over_union,
            "area_trend_pixels_per_observation": self.area_trend_pixels_per_observation,
            "temperature_trend_kelvin_per_observation": (
                self.temperature_trend_kelvin_per_observation
            ),
            "disappeared": self.disappeared,
            "confidence": self.confidence,
            "matches": [match.to_dict() for match in self.matches],
            "tracks": [track.to_dict() for track in self.tracks],
        }


@dataclass(frozen=True, slots=True)
class _Component:
    observation_index: int
    label: int
    pixels: MaskArray
    area_pixels: int
    centroid_latitude: float
    centroid_longitude: float
    mean_temperature_kelvin: float


@dataclass(slots=True)
class _TrackBuilder:
    track_id: int
    components: list[_Component]
    intersection_over_unions: list[float] = field(default_factory=list)
    disappearance_observation_index: int | None = None


def measure_temporal_persistence(
    observations: tuple[TemporalObservation | None, ...],
    parameters: PersistenceParameters = DEVELOPMENT_PERSISTENCE_PARAMETERS,
    *,
    common_grid: GeospatialGrid | None = None,
) -> TemporalPersistenceResult:
    """Align observations geospatially, match adjacent regions, and measure tracks."""

    if not observations:
        raise ValueError("observations must contain at least one item")
    if not isinstance(parameters, PersistenceParameters):
        raise TypeError("parameters must be PersistenceParameters")
    if any(
        observation is not None and not isinstance(observation, TemporalObservation)
        for observation in observations
    ):
        raise TypeError("observations must contain TemporalObservation or None")
    target_grid = common_grid
    if target_grid is None:
        target_grid = next(
            (
                observation.grid
                for observation in observations
                if observation is not None
            ),
            None,
        )
    if not isinstance(target_grid, GeospatialGrid):
        raise ValueError(
            "at least one available observation or a common_grid is required"
        )
    aligned = tuple(
        None
        if observation is None
        else _align_observation(observation, target_grid, parameters)
        for observation in observations
    )
    components_by_observation = tuple(
        ()
        if aligned_observation is None
        else _components(
            index,
            aligned_observation,
            target_grid,
            parameters.minimum_component_area_pixels,
        )
        for index, aligned_observation in enumerate(aligned)
    )
    matches, builders = _build_tracks(components_by_observation, aligned, parameters)
    tracks = tuple(_freeze_track(builder) for builder in builders)
    primary = _primary_track(tracks)
    available_count = sum(observation is not None for observation in aligned)
    if primary is None:
        persistence_count = 0
        mean_iou = 0.0
        area_trend = None
        temperature_trend = None
        disappeared = False
        confidence = 0.0
    else:
        persistence_count = primary.persistence_count
        mean_iou = primary.mean_intersection_over_union
        area_trend = primary.area_trend_pixels_per_observation
        temperature_trend = primary.temperature_trend_kelvin_per_observation
        disappeared = primary.disappeared
        continuity_fraction = (
            (persistence_count - 1) / max(available_count - 1, 1)
            if persistence_count >= 2
            else 0.0
        )
        confidence = mean_iou * continuity_fraction
    return TemporalPersistenceResult(
        common_grid=target_grid,
        aligned_observations=aligned,
        matches=tuple(matches),
        tracks=tracks,
        missing_observation_count=sum(observation is None for observation in aligned),
        persistence_count=persistence_count,
        mean_intersection_over_union=mean_iou,
        area_trend_pixels_per_observation=area_trend,
        temperature_trend_kelvin_per_observation=temperature_trend,
        disappeared=disappeared,
        confidence=confidence,
    )


def _align_observation(
    observation: TemporalObservation,
    target_grid: GeospatialGrid,
    parameters: PersistenceParameters,
) -> AlignedObservation:
    nearest = _nearest_source_indices(
        observation.grid,
        target_grid,
        parameters.maximum_resample_distance_kilometres,
    )
    mask = np.zeros(target_grid.shape, dtype=np.uint8)
    temperatures = np.full(target_grid.shape, np.nan, dtype=np.float32)
    valid = np.zeros(target_grid.shape, dtype=bool)
    mapped = nearest >= 0
    if np.any(mapped):
        source_flat = nearest[mapped]
        source_valid = observation.valid_mask.reshape(-1)[source_flat]
        target_rows, target_columns = np.nonzero(mapped)
        rows = target_rows[source_valid]
        columns = target_columns[source_valid]
        source_indices = source_flat[source_valid]
        mask[rows, columns] = observation.candidate_mask.reshape(-1)[source_indices]
        temperatures[rows, columns] = observation.channel7_kelvin.reshape(-1)[
            source_indices
        ]
        valid[rows, columns] = True
    return AlignedObservation(observation.observation_id, mask, temperatures, valid)


def _nearest_source_indices(
    source_grid: GeospatialGrid, target_grid: GeospatialGrid, maximum_distance_km: float
) -> Int32Array:
    source_latitude = source_grid.latitude_degrees.reshape(-1)
    source_longitude = source_grid.longitude_degrees.reshape(-1)
    source_geolocated = np.isfinite(source_latitude) & np.isfinite(source_longitude)
    target_latitude = target_grid.latitude_degrees.reshape(-1)
    target_longitude = target_grid.longitude_degrees.reshape(-1)
    target_geolocated = np.isfinite(target_latitude) & np.isfinite(target_longitude)
    result = np.full(target_latitude.shape, -1, dtype=np.int32)
    if not np.any(source_geolocated) or not np.any(target_geolocated):
        return result.reshape(target_grid.shape)
    source_indices = np.flatnonzero(source_geolocated)
    source_latitude = source_latitude[source_indices]
    source_longitude = source_longitude[source_indices]
    target_indices = np.flatnonzero(target_geolocated)
    # Chunking preserves deterministic nearest-neighbour selection without a
    # potentially enormous target-by-source allocation for real crops.
    for chunk_start in range(0, len(target_indices), 1_024):
        chunk_indices = target_indices[chunk_start : chunk_start + 1_024]
        distances = _haversine_kilometres(
            target_latitude[chunk_indices, None],
            target_longitude[chunk_indices, None],
            source_latitude[None, :],
            source_longitude[None, :],
        )
        nearest_positions = np.argmin(distances, axis=1)
        nearest_distances = distances[np.arange(len(chunk_indices)), nearest_positions]
        accepted = nearest_distances <= maximum_distance_km
        result[chunk_indices[accepted]] = source_indices[nearest_positions[accepted]]
    return result.reshape(target_grid.shape)


def _haversine_kilometres(
    latitude_a: npt.NDArray[np.float64],
    longitude_a: npt.NDArray[np.float64],
    latitude_b: npt.NDArray[np.float64],
    longitude_b: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    latitude_a_radians = np.deg2rad(latitude_a)
    latitude_b_radians = np.deg2rad(latitude_b)
    latitude_delta = latitude_b_radians - latitude_a_radians
    longitude_delta = np.deg2rad(((longitude_b - longitude_a + 180.0) % 360.0) - 180.0)
    haversine = (
        np.sin(latitude_delta / 2.0) ** 2
        + np.cos(latitude_a_radians)
        * np.cos(latitude_b_radians)
        * np.sin(longitude_delta / 2.0) ** 2
    )
    return np.asarray(
        2.0 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(haversine, 0.0, 1.0))),
        dtype=np.float64,
    )


def _components(
    observation_index: int,
    observation: AlignedObservation,
    grid: GeospatialGrid,
    minimum_area: int,
) -> tuple[_Component, ...]:
    count, labels, statistics, _ = cv2.connectedComponentsWithStats(
        observation.candidate_mask, connectivity=8, ltype=cv2.CV_32S
    )
    components: list[_Component] = []
    for label in range(1, count):
        pixels = labels == label
        area = int(statistics[label, cv2.CC_STAT_AREA])
        if area < minimum_area:
            continue
        components.append(
            _Component(
                observation_index=observation_index,
                label=label,
                pixels=pixels,
                area_pixels=area,
                centroid_latitude=float(np.mean(grid.latitude_degrees[pixels])),
                centroid_longitude=float(np.mean(grid.longitude_degrees[pixels])),
                mean_temperature_kelvin=float(
                    np.mean(observation.channel7_kelvin[pixels])
                ),
            )
        )
    return tuple(components)


def _build_tracks(
    components_by_observation: tuple[tuple[_Component, ...], ...],
    aligned_observations: tuple[AlignedObservation | None, ...],
    parameters: PersistenceParameters,
) -> tuple[list[RegionMatch], list[_TrackBuilder]]:
    builders: list[_TrackBuilder] = []
    matches: list[RegionMatch] = []
    previous_components: tuple[_Component, ...] = ()
    previous_tracks: dict[int, _TrackBuilder] = {}
    previous_available = False
    for index, components in enumerate(components_by_observation):
        available = aligned_observations[index] is not None
        if not available:
            previous_components = ()
            previous_tracks = {}
            previous_available = False
            continue
        current_tracks: dict[int, _TrackBuilder]
        if not previous_available:
            current_tracks = {
                component.label: _new_track(builders, component)
                for component in components
            }
        else:
            paired = _match_components(previous_components, components, parameters)
            matched_previous = {earlier.label for earlier, _, _ in paired}
            matched_later = {later.label for _, later, _ in paired}
            for component in previous_components:
                if component.label not in matched_previous:
                    previous_tracks[
                        component.label
                    ].disappearance_observation_index = index
            current_tracks = {}
            for earlier, later, match in paired:
                track = previous_tracks[earlier.label]
                track.components.append(later)
                track.intersection_over_unions.append(match.intersection_over_union)
                current_tracks[later.label] = track
                matches.append(match)
            for component in components:
                if component.label not in matched_later:
                    current_tracks[component.label] = _new_track(builders, component)
        previous_components = components
        previous_tracks = current_tracks
        previous_available = True
    return matches, builders


def _new_track(builders: list[_TrackBuilder], component: _Component) -> _TrackBuilder:
    track = _TrackBuilder(track_id=len(builders) + 1, components=[component])
    builders.append(track)
    return track


def _match_components(
    earlier: tuple[_Component, ...],
    later: tuple[_Component, ...],
    parameters: PersistenceParameters,
) -> tuple[tuple[_Component, _Component, RegionMatch], ...]:
    candidates: list[tuple[float, float, _Component, _Component, RegionMatch]] = []
    for earlier_component in earlier:
        for later_component in later:
            intersection = int(
                np.count_nonzero(earlier_component.pixels & later_component.pixels)
            )
            union = int(
                np.count_nonzero(earlier_component.pixels | later_component.pixels)
            )
            iou = intersection / union if union else 0.0
            distance = float(
                _haversine_kilometres(
                    np.asarray(earlier_component.centroid_latitude),
                    np.asarray(earlier_component.centroid_longitude),
                    np.asarray(later_component.centroid_latitude),
                    np.asarray(later_component.centroid_longitude),
                )
            )
            if (
                iou < parameters.minimum_intersection_over_union
                or distance > parameters.maximum_centroid_distance_kilometres
            ):
                continue
            match = RegionMatch(
                earlier_observation_index=earlier_component.observation_index,
                later_observation_index=later_component.observation_index,
                earlier_label=earlier_component.label,
                later_label=later_component.label,
                centroid_distance_kilometres=distance,
                intersection_over_union=iou,
            )
            candidates.append(
                (iou, distance, earlier_component, later_component, match)
            )
    matched_earlier: set[int] = set()
    matched_later: set[int] = set()
    selected: list[tuple[_Component, _Component, RegionMatch]] = []
    for _, _, earlier_component, later_component, match in sorted(
        candidates,
        key=lambda item: (
            -item[0],
            item[1],
            item[2].label,
            item[3].label,
        ),
    ):
        if (
            earlier_component.label in matched_earlier
            or later_component.label in matched_later
        ):
            continue
        matched_earlier.add(earlier_component.label)
        matched_later.add(later_component.label)
        selected.append((earlier_component, later_component, match))
    return tuple(selected)


def _freeze_track(builder: _TrackBuilder) -> PersistenceTrack:
    components = builder.components
    persistence_count = len(components)
    if persistence_count >= 2:
        interval = components[-1].observation_index - components[0].observation_index
        area_trend = (components[-1].area_pixels - components[0].area_pixels) / interval
        temperature_trend = (
            components[-1].mean_temperature_kelvin
            - components[0].mean_temperature_kelvin
        ) / interval
    else:
        area_trend = None
        temperature_trend = None
    return PersistenceTrack(
        track_id=builder.track_id,
        observation_indexes=tuple(
            component.observation_index for component in components
        ),
        component_labels=tuple(component.label for component in components),
        persistence_count=persistence_count,
        mean_intersection_over_union=(
            float(np.mean(builder.intersection_over_unions))
            if builder.intersection_over_unions
            else 0.0
        ),
        area_trend_pixels_per_observation=area_trend,
        temperature_trend_kelvin_per_observation=temperature_trend,
        disappeared=builder.disappearance_observation_index is not None,
        disappearance_observation_index=builder.disappearance_observation_index,
    )


def _primary_track(tracks: tuple[PersistenceTrack, ...]) -> PersistenceTrack | None:
    if not tracks:
        return None
    return min(
        tracks,
        key=lambda track: (
            -track.persistence_count,
            -track.mean_intersection_over_union,
            track.track_id,
        ),
    )


def _readonly[ArrayScalar: np.generic](
    array: npt.NDArray[ArrayScalar],
) -> npt.NDArray[ArrayScalar]:
    copied = np.ascontiguousarray(array).copy()
    copied.setflags(write=False)
    return copied


__all__ = [
    "DEVELOPMENT_PERSISTENCE_PARAMETERS",
    "AlignedObservation",
    "GeospatialGrid",
    "PersistenceParameters",
    "PersistenceTrack",
    "RegionMatch",
    "TemporalObservation",
    "TemporalPersistenceResult",
    "measure_temporal_persistence",
]
