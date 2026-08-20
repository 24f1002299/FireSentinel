"""Build deterministic evaluation cases from FIRMS events and pinned observations.

This module is intentionally evaluation-only.  It turns previously isolated
FIRMS event windows and an offline observation inventory into matched positive
and control cases without creating manual image labels.  Every case includes a
complete C07 initial/later/baseline and C14 alternate observation bundle whose
pinned source references were resolved before it was emitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from firesentinel.core.records import Channel
from firesentinel.data.goes18 import GOES18_BUCKET, parse_scan_times
from firesentinel.evaluation.firms import EVALUATION_DATA_DIRECTORY

AUDIT_FILENAME = "benchmark.audit.json"
BENCHMARK_FILENAME = "benchmark-cases.json"
DEFAULT_CASES_PER_CLASS = 60
DEFAULT_RANDOM_SEED = 20260820
SCHEMA_VERSION = 1
POSITIVE_EVENT_DISTANCE_KM = 25.0
POSITIVE_EVENT_TIME_WINDOW = timedelta(minutes=90)
CONTROL_EXCLUSION_DISTANCE_KM = 50.0
CONTROL_EXCLUSION_TIME_WINDOW = timedelta(hours=24)
MAXIMUM_LOCAL_TIME_DIFFERENCE_HOURS = 1
MAXIMUM_VIEW_ZENITH_DIFFERENCE_DEGREES = 10.0
MAXIMUM_USABLE_FRACTION_DIFFERENCE = 0.10
_IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9_-]{0,79}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SOURCE_CHANNEL = re.compile(r"-M\d(?P<channel>C\d{2})_")
_REQUIRED_ROLES = ("baseline", "initial", "later", "alternate")
_EXPECTED_CHANNELS = {
    "baseline": Channel.C07,
    "initial": Channel.C07,
    "later": Channel.C07,
    "alternate": Channel.C14,
}


@dataclass(frozen=True, slots=True)
class SourceReference:
    """One hash-pinned GOES source that a benchmark observation may reference."""

    source_id: str
    bucket: str
    object_key: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if not _IDENTIFIER.fullmatch(self.source_id):
            raise ValueError("source_id must be a lowercase identifier")
        if self.bucket != GOES18_BUCKET:
            raise ValueError(f"source bucket must be {GOES18_BUCKET!r}")
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int):
            raise ValueError("source size_bytes must be an integer")
        if self.size_bytes <= 0:
            raise ValueError("source size_bytes must be positive")
        if not _SHA256.fullmatch(self.sha256):
            raise ValueError("source sha256 must be a lowercase SHA-256 digest")
        parse_scan_times(self.object_key)

    def to_dict(self) -> dict[str, str | int]:
        return {
            "source_id": self.source_id,
            "bucket": self.bucket,
            "object_key": self.object_key,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class WindowObservation:
    """A role-bound observation whose source was resolved from the inventory."""

    role: str
    channel: Channel
    observation_time: datetime
    source: SourceReference

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "channel": self.channel.value,
            "observation_time_utc": _timestamp(self.observation_time),
            "source": self.source.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ObservationWindow:
    """A candidate image window with the complete four-observation bundle."""

    window_id: str
    anchor_time: datetime
    latitude: float
    longitude: float
    view_zenith_degrees: float
    usable_fraction: float
    observations: tuple[WindowObservation, ...]

    def __post_init__(self) -> None:
        if not _IDENTIFIER.fullmatch(self.window_id):
            raise ValueError("window_id must be a lowercase identifier")
        _coordinate(self.latitude, "latitude", -90.0, 90.0)
        _coordinate(self.longitude, "longitude", -180.0, 180.0)
        if not 0.0 <= self.view_zenith_degrees <= 90.0:
            raise ValueError("view_zenith_degrees must be within [0, 90]")
        if not 0.0 <= self.usable_fraction <= 1.0:
            raise ValueError("usable_fraction must be within [0, 1]")
        expected_roles = set(_REQUIRED_ROLES)
        roles = {observation.role for observation in self.observations}
        if len(self.observations) != len(_REQUIRED_ROLES) or roles != expected_roles:
            raise ValueError("window observations must contain each required role once")
        by_role = {observation.role: observation for observation in self.observations}
        for role, expected_channel in _EXPECTED_CHANNELS.items():
            if by_role[role].channel != expected_channel:
                raise ValueError(f"{role} must use {expected_channel.value}")
        if by_role["initial"].observation_time != self.anchor_time:
            raise ValueError("initial observation_time_utc must equal anchor time")
        baseline_gap = self.anchor_time - by_role["baseline"].observation_time
        later_gap = by_role["later"].observation_time - self.anchor_time
        alternate_gap = abs(by_role["alternate"].observation_time - self.anchor_time)
        if not timedelta(minutes=30) <= baseline_gap <= timedelta(days=7):
            raise ValueError("baseline must be 30 minutes to 7 days before initial")
        if not timedelta(minutes=10) <= later_gap <= timedelta(minutes=90):
            raise ValueError("later must be 10 to 90 minutes after initial")
        if alternate_gap > timedelta(minutes=20):
            raise ValueError("alternate must be within 20 minutes of initial")

    @property
    def matching_variables(self) -> dict[str, str | int | float]:
        return _matching_variables(
            self.anchor_time,
            self.latitude,
            self.longitude,
            self.view_zenith_degrees,
            self.usable_fraction,
        )

    def to_case_observations(self) -> list[dict[str, object]]:
        by_role = {observation.role: observation for observation in self.observations}
        return [by_role[role].to_dict() for role in _REQUIRED_ROLES]


@dataclass(frozen=True, slots=True)
class FirmsDetection:
    """Minimal normalized FIRMS detection used only to exclude controls."""

    acquired_at: datetime
    latitude: float
    longitude: float


@dataclass(frozen=True, slots=True)
class FirmsEvent:
    """One FIRMS event window used to select a positive observation window."""

    event_id: str
    start_time: datetime
    end_time: datetime
    latitude: float
    longitude: float


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """One immutable positive or matched-control benchmark case."""

    case_id: str
    label: str
    anchor_time: datetime
    latitude: float
    longitude: float
    matching_variables: dict[str, str | int | float]
    observations: tuple[WindowObservation, ...]
    event_id: str | None = None
    matched_positive_case_id: str | None = None
    matching_deltas: dict[str, int | float] | None = None

    def _payload_without_hash(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "case_id": self.case_id,
            "label": self.label,
            "anchor": {
                "acquisition_time_utc": _timestamp(self.anchor_time),
                "latitude": self.latitude,
                "longitude": self.longitude,
            },
            "matching_variables": self.matching_variables,
            "observations": [
                observation.to_dict() for observation in self.observations
            ],
        }
        if self.event_id is not None:
            payload["event_id"] = self.event_id
        if self.matched_positive_case_id is not None:
            payload["matched_positive_case_id"] = self.matched_positive_case_id
        if self.matching_deltas is not None:
            payload["matching_deltas"] = self.matching_deltas
        return payload

    def to_dict(self) -> dict[str, object]:
        payload = self._payload_without_hash()
        payload["integrity_sha256"] = _sha256(_canonical_json_bytes(payload))
        return payload


@dataclass(frozen=True, slots=True)
class BenchmarkBuild:
    """The benchmark and its independently hashable input provenance."""

    firms_labels_sha256: str
    observation_inventory_sha256: str
    random_seed: int
    requested_cases_per_class: int
    positive_candidate_count: int
    control_candidate_count: int
    cases: tuple[BenchmarkCase, ...]

    def benchmark_payload(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "record_type": "firms_matched_control_benchmark",
            "evaluation_only": True,
            "random_seed": self.random_seed,
            "inputs": {
                "firms_labels_sha256": self.firms_labels_sha256,
                "observation_inventory_sha256": self.observation_inventory_sha256,
            },
            "cases": [case.to_dict() for case in self.cases],
        }

    def audit_payload(self, benchmark_sha256: str) -> dict[str, object]:
        positive_count = sum(case.label == "positive" for case in self.cases)
        control_count = sum(case.label == "control" for case in self.cases)
        source_hashes = sorted(
            {
                observation.source.sha256
                for case in self.cases
                for observation in case.observations
            }
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "record_type": "firms_matched_control_benchmark_audit",
            "evaluation_only": True,
            "benchmark_filename": BENCHMARK_FILENAME,
            "benchmark_sha256": benchmark_sha256,
            "random_seed": self.random_seed,
            "inputs": {
                "firms_labels_sha256": self.firms_labels_sha256,
                "observation_inventory_sha256": self.observation_inventory_sha256,
            },
            "counts": {
                "requested_cases_per_class": self.requested_cases_per_class,
                "positive_candidates": self.positive_candidate_count,
                "control_candidates_after_firms_exclusion": (
                    self.control_candidate_count
                ),
                "positives": positive_count,
                "controls": control_count,
            },
            "source_reference_hashes": source_hashes,
            "matching_policy": _matching_policy(),
        }


def build_benchmark(
    firms_labels_path: Path,
    observation_inventory_path: Path,
    *,
    cases_per_class: int = DEFAULT_CASES_PER_CLASS,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> BenchmarkBuild:
    """Build exactly ``cases_per_class`` positive/control pairs deterministically."""

    _case_count(cases_per_class)
    _seed(random_seed)
    firms_bytes, firms_payload = _load_json(firms_labels_path, "FIRMS labels")
    inventory_bytes, inventory_payload = _load_json(
        observation_inventory_path, "observation inventory"
    )
    events, detections = _load_firms_events(firms_payload)
    sources, windows = _load_observation_inventory(inventory_payload)
    del sources  # Window observations retain resolved, immutable source references.

    positive_candidates = _positive_candidates(events, windows)
    control_candidates = tuple(
        window
        for window in windows
        if not _has_nearby_firms_detection(window, detections)
    )
    matched_pairs = _matched_pairs(
        positive_candidates, control_candidates, random_seed=random_seed
    )
    if len(matched_pairs) < cases_per_class:
        raise ValueError(
            "insufficient matched benchmark cases: "
            f"need {cases_per_class} pairs, found {len(matched_pairs)}; "
            f"positive candidates={len(positive_candidates)}, "
            f"controls after FIRMS exclusion={len(control_candidates)}"
        )

    chooser = random.Random(random_seed)
    chooser.shuffle(matched_pairs)
    selected_pairs = matched_pairs[:cases_per_class]
    positive_cases: list[BenchmarkCase] = []
    control_cases: list[BenchmarkCase] = []
    for event, positive_window, control_window in selected_pairs:
        positive_case = _positive_case(event, positive_window)
        positive_cases.append(positive_case)
        control_cases.append(_control_case(control_window, positive_case))
    cases = tuple(
        sorted(
            (*positive_cases, *control_cases),
            key=lambda case: (case.label, case.anchor_time, case.case_id),
        )
    )
    return BenchmarkBuild(
        firms_labels_sha256=_sha256(firms_bytes),
        observation_inventory_sha256=_sha256(inventory_bytes),
        random_seed=random_seed,
        requested_cases_per_class=cases_per_class,
        positive_candidate_count=len(positive_candidates),
        control_candidate_count=len(control_candidates),
        cases=cases,
    )


def write_benchmark(
    build: BenchmarkBuild, output_directory: Path, *, overwrite: bool = False
) -> tuple[Path, Path]:
    """Write a benchmark and a separately hash-linked audit record atomically."""

    benchmark_path = output_directory / BENCHMARK_FILENAME
    audit_path = output_directory / AUDIT_FILENAME
    benchmark_bytes = _canonical_json_bytes(build.benchmark_payload())
    audit_bytes = _canonical_json_bytes(build.audit_payload(_sha256(benchmark_bytes)))
    _preflight_output(benchmark_path, benchmark_bytes, overwrite=overwrite)
    _preflight_output(audit_path, audit_bytes, overwrite=overwrite)
    _atomic_write(benchmark_path, benchmark_bytes)
    _atomic_write(audit_path, audit_bytes)
    return benchmark_path, audit_path


def verify_benchmark(
    benchmark_path: Path,
    audit_path: Path,
    *,
    minimum_cases_per_class: int = DEFAULT_CASES_PER_CLASS,
) -> None:
    """Verify case hashes, source references, matching links, and case counts."""

    _case_count(minimum_cases_per_class)
    benchmark_bytes, benchmark = _load_json(benchmark_path, "benchmark")
    _, audit = _load_json(audit_path, "benchmark audit")
    if not isinstance(benchmark, dict) or benchmark.get("evaluation_only") is not True:
        raise ValueError("benchmark must be an evaluation-only JSON object")
    if not isinstance(audit, dict) or audit.get("evaluation_only") is not True:
        raise ValueError("benchmark audit must be an evaluation-only JSON object")
    if audit.get("benchmark_sha256") != _sha256(benchmark_bytes):
        raise ValueError("benchmark audit SHA-256 does not match benchmark bytes")
    cases = benchmark.get("cases")
    if not isinstance(cases, list):
        raise ValueError("benchmark cases must be a list")
    positive_ids: set[str] = set()
    control_count = 0
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("benchmark case must be an object")
        integrity_hash = case.get("integrity_sha256")
        case_without_hash = dict(case)
        case_without_hash.pop("integrity_sha256", None)
        if not isinstance(integrity_hash, str) or integrity_hash != _sha256(
            _canonical_json_bytes(case_without_hash)
        ):
            raise ValueError("benchmark case integrity SHA-256 does not match")
        label = case.get("label")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not _IDENTIFIER.fullmatch(case_id):
            raise ValueError("benchmark case_id must be a lowercase identifier")
        _verify_case_observations(case)
        if label == "positive":
            positive_ids.add(case_id)
        elif label == "control":
            control_count += 1
        else:
            raise ValueError("benchmark case label must be positive or control")

    all_positive_ids = {
        case["case_id"]
        for case in cases
        if isinstance(case, dict) and case.get("label") == "positive"
    }
    for case in cases:
        if (
            isinstance(case, dict)
            and case.get("label") == "control"
            and case.get("matched_positive_case_id") not in all_positive_ids
        ):
            raise ValueError("control does not reference a positive benchmark case")
    if (
        len(positive_ids) < minimum_cases_per_class
        or control_count < minimum_cases_per_class
    ):
        raise ValueError(
            "benchmark requires at least "
            f"{minimum_cases_per_class} positives and controls"
        )


def default_evaluation_directory() -> Path:
    """Return the repository's non-configurable evaluation-data root."""

    return Path(__file__).resolve().parents[3] / EVALUATION_DATA_DIRECTORY


def _load_firms_events(
    payload: object,
) -> tuple[tuple[FirmsEvent, ...], tuple[FirmsDetection, ...]]:
    if not isinstance(payload, dict):
        raise ValueError("FIRMS labels must be a JSON object")
    if payload.get("evaluation_only") is not True:
        raise ValueError("FIRMS labels must be marked evaluation_only")
    if payload.get("record_type") != "firms_event_reference_labels":
        raise ValueError("FIRMS labels have an unexpected record_type")
    raw_events = payload.get("events")
    if not isinstance(raw_events, list):
        raise ValueError("FIRMS labels events must be a list")
    events: list[FirmsEvent] = []
    detections: list[FirmsDetection] = []
    event_ids: set[str] = set()
    for raw_event in raw_events:
        if not isinstance(raw_event, dict):
            raise ValueError("FIRMS event must be an object")
        event_id = _identifier(raw_event.get("event_id"), "FIRMS event_id")
        if event_id in event_ids:
            raise ValueError(f"FIRMS labels repeat event_id {event_id!r}")
        event_ids.add(event_id)
        start_time = _parse_timestamp(raw_event.get("start_time_utc"), "event start")
        end_time = _parse_timestamp(raw_event.get("end_time_utc"), "event end")
        if end_time < start_time:
            raise ValueError("FIRMS event end must not precede start")
        latitude = _coordinate(
            raw_event.get("centroid_latitude"), "event latitude", -90, 90
        )
        longitude = _coordinate(
            raw_event.get("centroid_longitude"), "event longitude", -180, 180
        )
        raw_detections = raw_event.get("detections")
        if not isinstance(raw_detections, list) or not raw_detections:
            raise ValueError("FIRMS event must contain detections")
        for raw_detection in raw_detections:
            if not isinstance(raw_detection, dict):
                raise ValueError("FIRMS detection must be an object")
            detections.append(
                FirmsDetection(
                    acquired_at=_parse_timestamp(
                        raw_detection.get("acquisition_time_utc"), "detection time"
                    ),
                    latitude=_coordinate(
                        raw_detection.get("latitude"), "detection latitude", -90, 90
                    ),
                    longitude=_coordinate(
                        raw_detection.get("longitude"),
                        "detection longitude",
                        -180,
                        180,
                    ),
                )
            )
        events.append(FirmsEvent(event_id, start_time, end_time, latitude, longitude))
    return tuple(
        sorted(events, key=lambda event: (event.start_time, event.event_id))
    ), tuple(sorted(detections, key=lambda detection: detection.acquired_at))


def _load_observation_inventory(
    payload: object,
) -> tuple[dict[str, SourceReference], tuple[ObservationWindow, ...]]:
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "record_type",
        "sources",
        "windows",
    }:
        raise ValueError(
            "observation inventory must contain schema_version, record_type, "
            "sources, windows"
        )
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported observation inventory schema_version")
    if payload["record_type"] != "goes18_observation_window_inventory":
        raise ValueError("unexpected observation inventory record_type")
    raw_sources = payload["sources"]
    raw_windows = payload["windows"]
    if not isinstance(raw_sources, list) or not isinstance(raw_windows, list):
        raise ValueError("observation inventory sources and windows must be lists")
    sources: dict[str, SourceReference] = {}
    for raw_source in raw_sources:
        source = _source_reference(raw_source)
        if source.source_id in sources:
            raise ValueError(
                f"observation inventory repeats source_id {source.source_id!r}"
            )
        sources[source.source_id] = source
    windows = tuple(
        _observation_window(raw_window, sources) for raw_window in raw_windows
    )
    if len({window.window_id for window in windows}) != len(windows):
        raise ValueError("observation inventory repeats window_id")
    return sources, tuple(sorted(windows, key=lambda window: window.window_id))


def _source_reference(raw_source: object) -> SourceReference:
    if not isinstance(raw_source, dict) or set(raw_source) != {
        "source_id",
        "bucket",
        "object_key",
        "size_bytes",
        "sha256",
    }:
        raise ValueError("observation inventory source has an invalid shape")
    return SourceReference(
        source_id=_identifier(raw_source["source_id"], "source_id"),
        bucket=_string(raw_source["bucket"], "source bucket"),
        object_key=_string(raw_source["object_key"], "source object_key"),
        size_bytes=raw_source["size_bytes"],
        sha256=_string(raw_source["sha256"], "source sha256"),
    )


def _observation_window(
    raw_window: object, sources: dict[str, SourceReference]
) -> ObservationWindow:
    if not isinstance(raw_window, dict) or set(raw_window) != {
        "window_id",
        "anchor",
        "view_zenith_degrees",
        "usable_fraction",
        "observations",
    }:
        raise ValueError("observation inventory window has an invalid shape")
    anchor = raw_window["anchor"]
    if not isinstance(anchor, dict) or set(anchor) != {
        "acquisition_time_utc",
        "latitude",
        "longitude",
    }:
        raise ValueError("observation inventory anchor has an invalid shape")
    raw_observations = raw_window["observations"]
    if not isinstance(raw_observations, list):
        raise ValueError("observation inventory observations must be a list")
    observations = tuple(
        _window_observation(raw_observation, sources)
        for raw_observation in raw_observations
    )
    return ObservationWindow(
        window_id=_identifier(raw_window["window_id"], "window_id"),
        anchor_time=_parse_timestamp(anchor["acquisition_time_utc"], "anchor time"),
        latitude=_coordinate(anchor["latitude"], "anchor latitude", -90, 90),
        longitude=_coordinate(anchor["longitude"], "anchor longitude", -180, 180),
        view_zenith_degrees=_coordinate(
            raw_window["view_zenith_degrees"], "view_zenith_degrees", 0, 90
        ),
        usable_fraction=_coordinate(
            raw_window["usable_fraction"], "usable_fraction", 0, 1
        ),
        observations=observations,
    )


def _window_observation(
    raw_observation: object, sources: dict[str, SourceReference]
) -> WindowObservation:
    if not isinstance(raw_observation, dict) or set(raw_observation) != {
        "role",
        "channel",
        "observation_time_utc",
        "source_id",
    }:
        raise ValueError("observation inventory observation has an invalid shape")
    role = _string(raw_observation["role"], "observation role")
    if role not in _EXPECTED_CHANNELS:
        raise ValueError(f"unknown observation role {role!r}")
    try:
        channel = Channel(_string(raw_observation["channel"], "observation channel"))
    except ValueError as error:
        raise ValueError("observation channel must be C07 or C14") from error
    source_id = _identifier(raw_observation["source_id"], "observation source_id")
    try:
        source = sources[source_id]
    except KeyError as error:
        raise ValueError(
            f"observation source_id does not resolve: {source_id}"
        ) from error
    scan_start, _ = parse_scan_times(source.object_key)
    observation_time = _parse_timestamp(
        raw_observation["observation_time_utc"], "observation time"
    )
    if observation_time != scan_start:
        raise ValueError("observation time must resolve exactly to source scan_start")
    if _source_channel(source) != channel:
        raise ValueError("observation channel does not match the resolved source")
    return WindowObservation(role, channel, observation_time, source)


def _positive_candidates(
    events: Sequence[FirmsEvent], windows: Sequence[ObservationWindow]
) -> list[tuple[FirmsEvent, ObservationWindow]]:
    candidates: list[tuple[FirmsEvent, ObservationWindow]] = []
    used_window_ids: set[str] = set()
    for event in events:
        matches = [
            window
            for window in windows
            if _event_matches_window(event, window)
            and window.window_id not in used_window_ids
        ]
        if not matches:
            continue
        selected = min(
            matches,
            key=lambda window: (
                _distance_to_event_window(event, window.anchor_time),
                _haversine_km(
                    event.latitude, event.longitude, window.latitude, window.longitude
                ),
                window.window_id,
            ),
        )
        used_window_ids.add(selected.window_id)
        candidates.append((event, selected))
    return candidates


def _event_matches_window(event: FirmsEvent, window: ObservationWindow) -> bool:
    return (
        _distance_to_event_window(event, window.anchor_time)
        <= POSITIVE_EVENT_TIME_WINDOW
        and _haversine_km(
            event.latitude, event.longitude, window.latitude, window.longitude
        )
        <= POSITIVE_EVENT_DISTANCE_KM
    )


def _distance_to_event_window(event: FirmsEvent, moment: datetime) -> timedelta:
    if event.start_time <= moment <= event.end_time:
        return timedelta(0)
    return min(abs(moment - event.start_time), abs(moment - event.end_time))


def _has_nearby_firms_detection(
    window: ObservationWindow, detections: Iterable[FirmsDetection]
) -> bool:
    return any(
        abs(window.anchor_time - detection.acquired_at) <= CONTROL_EXCLUSION_TIME_WINDOW
        and _haversine_km(
            window.latitude, window.longitude, detection.latitude, detection.longitude
        )
        <= CONTROL_EXCLUSION_DISTANCE_KM
        for detection in detections
    )


def _matched_pairs(
    positive_candidates: Sequence[tuple[FirmsEvent, ObservationWindow]],
    controls: Sequence[ObservationWindow],
    *,
    random_seed: int,
) -> list[tuple[FirmsEvent, ObservationWindow, ObservationWindow]]:
    matcher = random.Random(random_seed)
    control_indices = list(range(len(controls)))
    matcher.shuffle(control_indices)
    matches_by_positive: dict[int, list[int]] = {}
    for positive_index, (_, positive_window) in enumerate(positive_candidates):
        matches_by_positive[positive_index] = [
            control_index
            for control_index in control_indices
            if _variables_match(positive_window, controls[control_index])
        ]
    positive_indices = list(range(len(positive_candidates)))
    matcher.shuffle(positive_indices)
    positive_indices.sort(key=lambda index: len(matches_by_positive[index]))
    matched_control_to_positive: dict[int, int] = {}

    def assign(positive_index: int, visited: set[int]) -> bool:
        for control_index in matches_by_positive[positive_index]:
            if control_index in visited:
                continue
            visited.add(control_index)
            prior_positive = matched_control_to_positive.get(control_index)
            if prior_positive is None or assign(prior_positive, visited):
                matched_control_to_positive[control_index] = positive_index
                return True
        return False

    for positive_index in positive_indices:
        assign(positive_index, set())
    pairs = [
        (
            positive_candidates[positive_index][0],
            positive_candidates[positive_index][1],
            controls[control_index],
        )
        for control_index, positive_index in matched_control_to_positive.items()
    ]
    return sorted(
        pairs,
        key=lambda pair: (pair[0].event_id, pair[1].window_id, pair[2].window_id),
    )


def _variables_match(positive: ObservationWindow, control: ObservationWindow) -> bool:
    positive_variables = positive.matching_variables
    control_variables = control.matching_variables
    return (
        positive_variables["season"] == control_variables["season"]
        and positive_variables["region_cell"] == control_variables["region_cell"]
        and _circular_hour_difference(
            int(positive_variables["local_time_hour"]),
            int(control_variables["local_time_hour"]),
        )
        <= MAXIMUM_LOCAL_TIME_DIFFERENCE_HOURS
        and abs(
            float(positive_variables["view_zenith_degrees"])
            - float(control_variables["view_zenith_degrees"])
        )
        <= MAXIMUM_VIEW_ZENITH_DIFFERENCE_DEGREES
        and abs(
            float(positive_variables["usable_fraction"])
            - float(control_variables["usable_fraction"])
        )
        <= MAXIMUM_USABLE_FRACTION_DIFFERENCE
    )


def _positive_case(event: FirmsEvent, window: ObservationWindow) -> BenchmarkCase:
    case_id = _case_id("positive", event.event_id, window.window_id)
    return BenchmarkCase(
        case_id=case_id,
        label="positive",
        anchor_time=window.anchor_time,
        latitude=window.latitude,
        longitude=window.longitude,
        matching_variables=window.matching_variables,
        observations=tuple(
            sorted(
                window.observations,
                key=lambda observation: _REQUIRED_ROLES.index(observation.role),
            )
        ),
        event_id=event.event_id,
    )


def _control_case(
    window: ObservationWindow, positive_case: BenchmarkCase
) -> BenchmarkCase:
    positive_variables = positive_case.matching_variables
    control_variables = window.matching_variables
    deltas = {
        "local_time_hour_difference": _circular_hour_difference(
            int(positive_variables["local_time_hour"]),
            int(control_variables["local_time_hour"]),
        ),
        "view_zenith_degrees_difference": abs(
            float(positive_variables["view_zenith_degrees"])
            - float(control_variables["view_zenith_degrees"])
        ),
        "usable_fraction_difference": abs(
            float(positive_variables["usable_fraction"])
            - float(control_variables["usable_fraction"])
        ),
    }
    return BenchmarkCase(
        case_id=_case_id("control", window.window_id, positive_case.case_id),
        label="control",
        anchor_time=window.anchor_time,
        latitude=window.latitude,
        longitude=window.longitude,
        matching_variables=control_variables,
        observations=tuple(
            sorted(
                window.observations,
                key=lambda observation: _REQUIRED_ROLES.index(observation.role),
            )
        ),
        matched_positive_case_id=positive_case.case_id,
        matching_deltas=deltas,
    )


def _matching_variables(
    anchor_time: datetime,
    latitude: float,
    longitude: float,
    view_zenith_degrees: float,
    usable_fraction: float,
) -> dict[str, str | int | float]:
    return {
        "season": _season(anchor_time.month),
        "region_cell": _region_cell(latitude, longitude),
        "local_time_hour": int((anchor_time.hour + longitude / 15) % 24),
        "view_zenith_degrees": view_zenith_degrees,
        "view_zenith_bin": int(view_zenith_degrees // 10),
        "usable_fraction": usable_fraction,
        "usable_fraction_bin": int(usable_fraction * 20),
    }


def _matching_policy() -> dict[str, object]:
    return {
        "positive_event_maximum_distance_km": POSITIVE_EVENT_DISTANCE_KM,
        "positive_event_maximum_time_minutes": int(
            POSITIVE_EVENT_TIME_WINDOW.total_seconds() / 60
        ),
        "control_firms_exclusion_distance_km": CONTROL_EXCLUSION_DISTANCE_KM,
        "control_firms_exclusion_time_hours": int(
            CONTROL_EXCLUSION_TIME_WINDOW.total_seconds() / 3600
        ),
        "required_matching_variables": [
            "season",
            "region_cell",
            "local_time_hour",
            "view_zenith_degrees",
            "usable_fraction",
        ],
        "maximum_local_time_difference_hours": MAXIMUM_LOCAL_TIME_DIFFERENCE_HOURS,
        "maximum_view_zenith_difference_degrees": (
            MAXIMUM_VIEW_ZENITH_DIFFERENCE_DEGREES
        ),
        "maximum_usable_fraction_difference": MAXIMUM_USABLE_FRACTION_DIFFERENCE,
    }


def _season(month: int) -> str:
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def _region_cell(latitude: float, longitude: float) -> str:
    latitude_cell = math.floor(latitude / 2) * 2
    longitude_cell = math.floor(longitude / 2) * 2
    return f"lat{latitude_cell:+03d}_lon{longitude_cell:+04d}"


def _circular_hour_difference(first: int, second: int) -> int:
    difference = abs(first - second)
    return min(difference, 24 - difference)


def _case_id(prefix: str, first: str, second: str) -> str:
    digest = _sha256(f"{first}\0{second}".encode())[:20]
    return f"{prefix}-{digest}"


def _haversine_km(
    first_latitude: float,
    first_longitude: float,
    second_latitude: float,
    second_longitude: float,
) -> float:
    radius_km = 6371.0088
    first_latitude_radians = math.radians(first_latitude)
    second_latitude_radians = math.radians(second_latitude)
    latitude_delta = second_latitude_radians - first_latitude_radians
    longitude_delta = math.radians(second_longitude - first_longitude)
    value = (
        math.sin(latitude_delta / 2) ** 2
        + math.cos(first_latitude_radians)
        * math.cos(second_latitude_radians)
        * math.sin(longitude_delta / 2) ** 2
    )
    return 2 * radius_km * math.asin(math.sqrt(value))


def _source_channel(source: SourceReference) -> Channel:
    match = _SOURCE_CHANNEL.search(source.object_key)
    if match is None:
        raise ValueError("resolved source object_key does not contain a channel")
    try:
        return Channel(match["channel"])
    except ValueError as error:  # parse_scan_times makes this defensive only.
        raise ValueError("resolved source uses an unsupported channel") from error


def _verify_case_observations(case: dict[str, object]) -> None:
    observations = case.get("observations")
    if not isinstance(observations, list) or len(observations) != len(_REQUIRED_ROLES):
        raise ValueError("benchmark case must contain four observations")
    roles: set[str] = set()
    for observation in observations:
        if not isinstance(observation, dict):
            raise ValueError("benchmark observation must be an object")
        role = observation.get("role")
        channel = observation.get("channel")
        source = observation.get("source")
        if not isinstance(role, str) or role not in _EXPECTED_CHANNELS:
            raise ValueError("benchmark observation role is invalid")
        if channel != _EXPECTED_CHANNELS[role].value:
            raise ValueError("benchmark observation channel does not match role")
        if not isinstance(source, dict):
            raise ValueError("benchmark observation source is invalid")
        resolved = _source_reference(source)
        observation_time = _parse_timestamp(
            observation.get("observation_time_utc"), "benchmark observation time"
        )
        scan_start, _ = parse_scan_times(resolved.object_key)
        if observation_time != scan_start:
            raise ValueError(
                "benchmark observation does not resolve to source scan_start"
            )
        if _source_channel(resolved).value != channel:
            raise ValueError("benchmark observation channel does not match source")
        roles.add(role)
    if roles != set(_REQUIRED_ROLES):
        raise ValueError("benchmark case observation roles are incomplete")


def _load_json(path: Path, description: str) -> tuple[bytes, object]:
    try:
        payload_bytes = path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read {description}: {path}") from error
    try:
        return payload_bytes, json.loads(payload_bytes)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid {description} JSON: {path}") from error


def _parse_timestamp(value: object, field: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{field} must be a UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(f"{value[:-1]}+00:00")
    except ValueError as error:
        raise ValueError(f"{field} must be an ISO-8601 UTC timestamp") from error
    return parsed.astimezone(UTC)


def _timestamp(value: datetime) -> str:
    return (
        value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
    )


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase identifier")
    return value


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _coordinate(value: object, field: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    converted = float(value)
    if not math.isfinite(converted) or not minimum <= converted <= maximum:
        raise ValueError(f"{field} must be within [{minimum}, {maximum}]")
    return 0.0 if converted == 0 else converted


def _case_count(value: int) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < DEFAULT_CASES_PER_CLASS
    ):
        raise ValueError(
            f"cases_per_class must be an integer of at least {DEFAULT_CASES_PER_CLASS}"
        )


def _seed(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("random_seed must be an integer")


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _preflight_output(path: Path, content: bytes, *, overwrite: bool) -> None:
    if not path.exists() or path.read_bytes() == content or overwrite:
        return
    raise FileExistsError(
        f"Refusing to replace existing evaluation benchmark '{path}'; use --overwrite."
    )


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", delete=False, dir=path.parent, prefix=f".{path.name}."
    ) as temporary:
        temporary.write(content)
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


def main(argv: list[str] | None = None) -> int:
    """Build a hash-audited, evaluation-only FIRMS benchmark."""

    evaluation_directory = default_evaluation_directory()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--firms-labels",
        type=Path,
        default=evaluation_directory / "firms" / "firms-event-labels.json",
        help="evaluation-only FIRMS event labels",
    )
    parser.add_argument(
        "--observation-inventory",
        type=Path,
        default=evaluation_directory / "observation-inventory.json",
        help="pinned GOES observation-window inventory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=evaluation_directory / "benchmark",
        help="directory under evaluation-data/ for the generated benchmark",
    )
    parser.add_argument(
        "--cases-per-class",
        type=int,
        default=DEFAULT_CASES_PER_CLASS,
        help=f"positive and control count (minimum/default: {DEFAULT_CASES_PER_CLASS})",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f"deterministic control-sampling seed (default: {DEFAULT_RANDOM_SEED})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace changed benchmark files",
    )
    arguments = parser.parse_args(argv)
    allowed_root = evaluation_directory.resolve()
    for name, path in {
        "--firms-labels": arguments.firms_labels,
        "--observation-inventory": arguments.observation_inventory,
        "--output-dir": arguments.output_dir,
    }.items():
        if not path.resolve().is_relative_to(allowed_root):
            parser.error(
                f"{name} must be inside the evaluation-only directory {allowed_root}"
            )
    build = build_benchmark(
        arguments.firms_labels,
        arguments.observation_inventory,
        cases_per_class=arguments.cases_per_class,
        random_seed=arguments.random_seed,
    )
    benchmark_path, audit_path = write_benchmark(
        build, arguments.output_dir, overwrite=arguments.overwrite
    )
    verify_benchmark(
        benchmark_path, audit_path, minimum_cases_per_class=arguments.cases_per_class
    )
    print(
        json.dumps(
            {
                "benchmark": str(benchmark_path),
                "audit": str(audit_path),
                "positives": arguments.cases_per_class,
                "controls": arguments.cases_per_class,
                "random_seed": arguments.random_seed,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
