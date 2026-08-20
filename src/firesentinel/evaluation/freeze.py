"""Freeze leakage-safe development, test, and stress benchmark manifests.

The benchmark builder produces immutable cases, but cases alone are not a safe
model-selection split: nearby frames can share an event, a geographic cell, or
a short temporal period.  This module treats each of those axes as a grouping
key, assigns *connected groups* (never individual frames), records the audit,
and emits blind test and stress manifests.  The corresponding labels remain in
separate scoring-only files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from firesentinel.evaluation.benchmark import (
    AUDIT_FILENAME as BENCHMARK_AUDIT_FILENAME,
)
from firesentinel.evaluation.benchmark import (
    BENCHMARK_FILENAME,
    default_evaluation_directory,
    verify_benchmark,
)

SCHEMA_VERSION = 1
SPLITS = ("development", "test", "stress")
SPLIT_RATIOS = {"development": 0.60, "test": 0.20, "stress": 0.20}
FROZEN_AUDIT_FILENAME = "frozen-splits.audit.json"
ASSIGNMENTS_FILENAME = "frozen-split-assignments.json"
MANIFEST_FILENAMES = {
    "development": "development.manifest.json",
    "test": "test.manifest.json",
    "stress": "stress.manifest.json",
}
LABEL_FILENAMES = {
    "test": "test-labels.json",
    "stress": "stress-labels.json",
}
DEFAULT_MANUAL_SAMPLE_SIZE = 6
_IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9_-]{0,79}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_BLIND_CASE_ID = re.compile(r"(?:test|stress)-[0-9a-f]{24}\Z")


@dataclass(frozen=True, slots=True)
class _CaseInfo:
    """Validated benchmark case and the grouping keys used for its split."""

    case: dict[str, object]
    case_id: str
    label: str
    event_id: str
    geographic_cell: str
    time_period: str


@dataclass(frozen=True, slots=True)
class FrozenBenchmark:
    """Canonical payloads ready to be written as one frozen benchmark set."""

    files: dict[str, dict[str, object]]


def freeze_benchmark(
    benchmark_path: Path,
    benchmark_audit_path: Path,
    *,
    reviewer: str,
    review_notes: str,
    manual_sample_size: int = DEFAULT_MANUAL_SAMPLE_SIZE,
) -> FrozenBenchmark:
    """Create split manifests from a verified benchmark without frame leakage.

    A connected component is formed whenever two cases share an event, 2-degree
    geographic cell, or UTC ISO week.  Components rather than cases are then
    deterministically allocated to the three splits.  Fewer than three isolated
    components is an honest failure: a safe dev/test/stress separation is not
    possible with that source population.
    """

    _reviewer(reviewer)
    _review_notes(review_notes)
    _sample_size(manual_sample_size)
    verify_benchmark(benchmark_path, benchmark_audit_path)
    benchmark_bytes, benchmark = _load_object(benchmark_path, "benchmark")
    benchmark_audit_bytes, _ = _load_object(benchmark_audit_path, "benchmark audit")
    cases = _case_infos(benchmark)
    components = _connected_components(cases)
    assignments = _assign_components(components)
    leakage_check = _leakage_check(cases, assignments, components)
    benchmark_sha256 = _sha256(benchmark_bytes)

    public_ids = {
        case.case_id: _public_case_id(
            assignments[case.case_id], case.case_id, benchmark_sha256
        )
        for case in cases
        if assignments[case.case_id] != "development"
    }
    cases_by_split = {
        split: tuple(case for case in cases if assignments[case.case_id] == split)
        for split in SPLITS
    }
    manifest_payloads: dict[str, dict[str, object]] = {}
    for split in SPLITS:
        split_cases = cases_by_split[split]
        if split == "development":
            payload_cases = [case.case for case in split_cases]
            labels_visible_to_tuning = True
        else:
            payload_cases = [
                _blind_case(case.case, public_ids[case.case_id]) for case in split_cases
            ]
            labels_visible_to_tuning = False
        manifest_payloads[MANIFEST_FILENAMES[split]] = _with_integrity(
            {
                "schema_version": SCHEMA_VERSION,
                "record_type": "firesentinel_frozen_split_manifest",
                "evaluation_only": True,
                "frozen": True,
                "split": split,
                "labels_visible_to_tuning": labels_visible_to_tuning,
                "benchmark_sha256": benchmark_sha256,
                "grouping_policy": _grouping_policy(),
                "cases": payload_cases,
            }
        )

    label_payloads: dict[str, dict[str, object]] = {}
    for split, filename in LABEL_FILENAMES.items():
        manifest_bytes = _canonical_json_bytes(
            manifest_payloads[MANIFEST_FILENAMES[split]]
        )
        label_payloads[filename] = _with_integrity(
            {
                "schema_version": SCHEMA_VERSION,
                "record_type": "firesentinel_frozen_split_labels",
                "evaluation_only": True,
                "frozen": True,
                "access": "scoring-only",
                "split": split,
                "manifest_filename": MANIFEST_FILENAMES[split],
                "manifest_sha256": _sha256(manifest_bytes),
                "labels": [
                    {
                        "case_id": public_ids[case.case_id],
                        "label": case.label,
                    }
                    for case in cases_by_split[split]
                ],
            }
        )

    assignment_payload = _with_integrity(
        {
            "schema_version": SCHEMA_VERSION,
            "record_type": "firesentinel_frozen_split_assignments",
            "evaluation_only": True,
            "frozen": True,
            "access": "scoring-only",
            "benchmark_sha256": benchmark_sha256,
            "grouping_policy": _grouping_policy(),
            "assignments": [
                {
                    "benchmark_case_id": case.case_id,
                    "public_case_id": (
                        case.case_id
                        if assignments[case.case_id] == "development"
                        else public_ids[case.case_id]
                    ),
                    "split": assignments[case.case_id],
                    "event_id": case.event_id,
                    "geographic_cell": case.geographic_cell,
                    "time_period": case.time_period,
                }
                for case in cases
            ],
        }
    )

    files: dict[str, dict[str, object]] = {
        **manifest_payloads,
        **label_payloads,
        ASSIGNMENTS_FILENAME: assignment_payload,
    }
    audit_payload = _with_integrity(
        {
            "schema_version": SCHEMA_VERSION,
            "record_type": "firesentinel_frozen_split_audit",
            "evaluation_only": True,
            "frozen": True,
            "source": {
                "benchmark_filename": benchmark_path.name,
                "benchmark_sha256": benchmark_sha256,
                "benchmark_audit_filename": benchmark_audit_path.name,
                "benchmark_audit_sha256": _sha256(benchmark_audit_bytes),
            },
            "grouping_policy": _grouping_policy(),
            "leakage_check": leakage_check,
            "distributions": {
                "all": _distribution(cases),
                **{split: _distribution(cases_by_split[split]) for split in SPLITS},
            },
            "manual_inspection": _manual_inspection(
                cases,
                assignments,
                public_ids,
                reviewer,
                review_notes,
                manual_sample_size,
            ),
            "frozen_file_sha256": {
                filename: _sha256(_canonical_json_bytes(payload))
                for filename, payload in sorted(files.items())
            },
        }
    )
    files[FROZEN_AUDIT_FILENAME] = audit_payload
    return FrozenBenchmark(files=files)


def write_frozen_benchmark(
    frozen: FrozenBenchmark, output_directory: Path, *, overwrite: bool = False
) -> dict[str, Path]:
    """Atomically write the whole manifest set after checking every target."""

    expected_filenames = _expected_filenames()
    if set(frozen.files) != expected_filenames:
        raise ValueError("frozen benchmark does not contain the required files")
    contents = {
        filename: _canonical_json_bytes(payload)
        for filename, payload in frozen.files.items()
    }
    paths = {filename: output_directory / filename for filename in contents}
    for filename in sorted(paths):
        _preflight_output(paths[filename], contents[filename], overwrite=overwrite)
    for filename in sorted(paths):
        _atomic_write(paths[filename], contents[filename])
    return paths


def verify_frozen_benchmark(
    output_directory: Path, *, benchmark_path: Path | None = None
) -> None:
    """Verify frozen hashes, blind-label boundaries, and grouped leakage checks."""

    filenames = _expected_filenames()
    payloads: dict[str, dict[str, object]] = {}
    raw_bytes: dict[str, bytes] = {}
    for filename in sorted(filenames):
        content, payload = _load_object(output_directory / filename, filename)
        raw_bytes[filename] = content
        payloads[filename] = payload
        _verify_integrity(payload, filename)

    audit = payloads[FROZEN_AUDIT_FILENAME]
    if audit.get("record_type") != "firesentinel_frozen_split_audit":
        raise ValueError("frozen split audit has an unexpected record_type")
    hashes = _mapping(audit.get("frozen_file_sha256"), "frozen_file_sha256")
    expected_hashed_files = filenames - {FROZEN_AUDIT_FILENAME}
    if set(hashes) != expected_hashed_files:
        raise ValueError("frozen split audit does not hash every frozen data file")
    for filename in expected_hashed_files:
        expected = hashes[filename]
        if not isinstance(expected, str) or expected != _sha256(raw_bytes[filename]):
            raise ValueError(f"frozen hash mismatch for {filename}")

    manifests = {split: payloads[MANIFEST_FILENAMES[split]] for split in SPLITS}
    for split, manifest in manifests.items():
        _verify_manifest(manifest, split)
    labels = {split: payloads[LABEL_FILENAMES[split]] for split in LABEL_FILENAMES}
    for split, label_payload in labels.items():
        _verify_labels(
            label_payload,
            manifests[split],
            raw_bytes[MANIFEST_FILENAMES[split]],
            split,
        )

    assignment_payload = payloads[ASSIGNMENTS_FILENAME]
    assignments = _assignment_rows(assignment_payload)
    _assert_assignment_leakage(assignments)
    _verify_manifest_membership(manifests, labels, assignments)
    leakage_check = _mapping(audit.get("leakage_check"), "leakage_check")
    if leakage_check.get("status") != "passed":
        raise ValueError("frozen split audit does not record a passing leakage check")

    if benchmark_path is not None:
        benchmark_bytes, benchmark = _load_object(benchmark_path, "benchmark")
        source = _mapping(audit.get("source"), "frozen audit source")
        if source.get("benchmark_sha256") != _sha256(benchmark_bytes):
            raise ValueError("frozen split source benchmark SHA-256 does not match")
        source_cases = _case_infos(benchmark)
        if {case.case_id for case in source_cases} != {
            row["benchmark_case_id"] for row in assignments
        }:
            raise ValueError("frozen assignments do not cover the source benchmark")
        source_by_id = {case.case_id: case for case in source_cases}
        for row in assignments:
            case = source_by_id[row["benchmark_case_id"]]
            if (
                row["event_id"] != case.event_id
                or row["geographic_cell"] != case.geographic_cell
                or row["time_period"] != case.time_period
            ):
                raise ValueError("frozen assignment grouping keys differ from source")


def default_frozen_directory() -> Path:
    """Return the non-configurable directory used for frozen split artifacts."""

    return default_evaluation_directory() / "frozen"


def _case_infos(benchmark: dict[str, object]) -> tuple[_CaseInfo, ...]:
    if benchmark.get("record_type") != "firms_matched_control_benchmark":
        raise ValueError("benchmark has an unexpected record_type")
    raw_cases = benchmark.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError("benchmark cases must be a list")
    raw_by_id: dict[str, dict[str, object]] = {}
    for raw_case in raw_cases:
        case = _mapping(raw_case, "benchmark case")
        case_id = _identifier(case.get("case_id"), "benchmark case_id")
        if case_id in raw_by_id:
            raise ValueError(f"benchmark repeats case_id {case_id!r}")
        raw_by_id[case_id] = case

    positive_event_ids: dict[str, str] = {}
    for case_id, case in raw_by_id.items():
        if case.get("label") == "positive":
            positive_event_ids[case_id] = _identifier(
                case.get("event_id"), "positive event_id"
            )
    infos: list[_CaseInfo] = []
    for case_id, case in raw_by_id.items():
        label = case.get("label")
        if label not in {"positive", "control"}:
            raise ValueError("benchmark case label must be positive or control")
        if label == "positive":
            event_id = positive_event_ids[case_id]
        else:
            matched_id = _identifier(
                case.get("matched_positive_case_id"), "matched_positive_case_id"
            )
            try:
                event_id = positive_event_ids[matched_id]
            except KeyError as error:
                raise ValueError(
                    "control does not resolve to a positive event"
                ) from error
        anchor = _mapping(case.get("anchor"), "benchmark case anchor")
        moment = _timestamp(anchor.get("acquisition_time_utc"), "benchmark anchor time")
        latitude = _coordinate(
            anchor.get("latitude"), "benchmark anchor latitude", -90, 90
        )
        longitude = _coordinate(
            anchor.get("longitude"), "benchmark anchor longitude", -180, 180
        )
        infos.append(
            _CaseInfo(
                case=case,
                case_id=case_id,
                label=label,
                event_id=event_id,
                geographic_cell=_geographic_cell(latitude, longitude),
                time_period=_time_period(moment),
            )
        )
    if not infos:
        raise ValueError("benchmark must contain at least one case")
    return tuple(sorted(infos, key=lambda item: item.case_id))


def _connected_components(
    cases: tuple[_CaseInfo, ...],
) -> tuple[tuple[_CaseInfo, ...], ...]:
    parents = list(range(len(cases)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parents[second_root] = first_root

    first_for_key: dict[tuple[str, str], int] = {}
    for index, case in enumerate(cases):
        for key in (
            ("event", case.event_id),
            ("geographic_cell", case.geographic_cell),
            ("time_period", case.time_period),
        ):
            prior = first_for_key.setdefault(key, index)
            union(index, prior)
    grouped: dict[int, list[_CaseInfo]] = defaultdict(list)
    for index, case in enumerate(cases):
        grouped[find(index)].append(case)
    components = tuple(
        tuple(sorted(component, key=lambda item: item.case_id))
        for component in grouped.values()
    )
    return tuple(
        sorted(
            components,
            key=lambda component: (
                -len(component),
                _sha256("\0".join(case.case_id for case in component).encode()),
            ),
        )
    )


def _assign_components(components: tuple[tuple[_CaseInfo, ...], ...]) -> dict[str, str]:
    if len(components) < len(SPLITS):
        raise ValueError(
            "cannot freeze leakage-safe development, test, and stress splits: "
            "event/geographic-cell/time-period grouping yields fewer than three "
            "independent components"
        )
    assigned_counts = {split: 0 for split in SPLITS}
    assignments: dict[str, str] = {}
    for index, component in enumerate(components):
        if index < len(SPLITS):
            split = SPLITS[index]
        else:
            split = min(
                SPLITS,
                key=lambda name: (
                    assigned_counts[name] / SPLIT_RATIOS[name],
                    SPLITS.index(name),
                ),
            )
        for case in component:
            assignments[case.case_id] = split
            assigned_counts[split] += 1
    if any(count == 0 for count in assigned_counts.values()):
        raise ValueError(
            "each frozen split must contain at least one grouped component"
        )
    return assignments


def _leakage_check(
    cases: tuple[_CaseInfo, ...],
    assignments: dict[str, str],
    components: tuple[tuple[_CaseInfo, ...], ...],
) -> dict[str, object]:
    _assert_case_assignment_coverage(cases, assignments)
    axes = {
        "event_id": [case.event_id for case in cases],
        "geographic_cell": [case.geographic_cell for case in cases],
        "time_period": [case.time_period for case in cases],
    }
    axis_report: dict[str, dict[str, int]] = {}
    for axis, values in axes.items():
        split_by_value: dict[str, set[str]] = defaultdict(set)
        for case, value in zip(cases, values, strict=True):
            split_by_value[value].add(assignments[case.case_id])
        leaking = sorted(
            value for value, splits in split_by_value.items() if len(splits) > 1
        )
        if leaking:
            raise ValueError(f"leakage detected across {axis}: {leaking[0]!r}")
        axis_report[axis] = {
            "unique_groups": len(split_by_value),
            "groups_spanning_splits": 0,
        }
    return {
        "status": "passed",
        "case_count": len(cases),
        "component_count": len(components),
        "largest_component_case_count": max(len(component) for component in components),
        "axes": axis_report,
    }


def _distribution(cases: tuple[_CaseInfo, ...]) -> dict[str, object]:
    season_counts: Counter[str] = Counter()
    hour_counts: Counter[str] = Counter()
    confidence_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    band_sets: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    view_angles: list[float] = []
    missingness: list[float] = []
    complete_bundle_count = 0
    for item in cases:
        case = item.case
        labels[item.label] += 1
        anchor = _mapping(case.get("anchor"), "benchmark case anchor")
        moment = _timestamp(anchor.get("acquisition_time_utc"), "benchmark anchor time")
        longitude = _coordinate(
            anchor.get("longitude"), "benchmark anchor longitude", -180, 180
        )
        matching = _mapping(case.get("matching_variables"), "matching variables")
        view_angle = _number(
            matching.get("view_zenith_degrees"), "view_zenith_degrees", 0, 90
        )
        usable_fraction = _number(
            matching.get("usable_fraction"), "usable_fraction", 0, 1
        )
        season_counts[_season(moment.month)] += 1
        hour_counts[f"{int((moment.hour + longitude / 15) % 24):02d}"] += 1
        view_angles.append(view_angle)
        missingness.append(1.0 - usable_fraction)
        if item.label == "control":
            confidence_counts["not_applicable"] += 1
        else:
            raw_confidences = case.get("firms_confidence_values")
            if not isinstance(raw_confidences, list) or not raw_confidences:
                confidence_counts["unknown"] += 1
            else:
                for value in raw_confidences:
                    if not isinstance(value, str) or not value:
                        raise ValueError("firms_confidence_values must contain strings")
                    confidence_counts[value] += 1
        observations = case.get("observations")
        if not isinstance(observations, list):
            raise ValueError("benchmark case observations must be a list")
        bands: set[str] = set()
        roles: set[str] = set()
        for observation in observations:
            observation_mapping = _mapping(observation, "benchmark observation")
            role = observation_mapping.get("role")
            channel = observation_mapping.get("channel")
            if not isinstance(role, str) or not isinstance(channel, str):
                raise ValueError(
                    "benchmark observation role and channel must be strings"
                )
            role_counts[role] += 1
            roles.add(role)
            bands.add(channel)
        band_sets["+".join(sorted(bands))] += 1
        if roles == {"baseline", "initial", "later", "alternate"} and bands == {
            "C07",
            "C14",
        }:
            complete_bundle_count += 1
    return {
        "case_count": len(cases),
        "label_counts": dict(sorted(labels.items())),
        "season": dict(sorted(season_counts.items())),
        "local_hour": dict(sorted(hour_counts.items())),
        "view_angle_degrees": _numeric_distribution(view_angles, 10.0, 0.0, 90.0),
        "missingness_fraction": _numeric_distribution(missingness, 0.1, 0.0, 1.0),
        "confidence": dict(sorted(confidence_counts.items())),
        "band_availability": {
            "available_band_sets": dict(sorted(band_sets.items())),
            "role_observation_counts": dict(sorted(role_counts.items())),
            "complete_required_bundle_cases": complete_bundle_count,
            "missing_required_bundle_cases": len(cases) - complete_bundle_count,
        },
    }


def _numeric_distribution(
    values: list[float], width: float, minimum: float, maximum: float
) -> dict[str, object]:
    if not values:
        return {"count": 0, "summary": None, "bins": {}}
    counts: Counter[str] = Counter()
    for value in values:
        lower = min(math.floor(value / width) * width, maximum - width)
        upper = lower + width
        counts[f"{lower:.1f}-{upper:.1f}"] += 1
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    median = (
        ordered[midpoint]
        if len(ordered) % 2
        else (ordered[midpoint - 1] + ordered[midpoint]) / 2
    )
    return {
        "count": len(values),
        "summary": {
            "minimum": round(min(values), 6),
            "maximum": round(max(values), 6),
            "mean": round(sum(values) / len(values), 6),
            "median": round(median, 6),
        },
        "bins": dict(sorted(counts.items())),
    }


def _manual_inspection(
    cases: tuple[_CaseInfo, ...],
    assignments: dict[str, str],
    public_ids: dict[str, str],
    reviewer: str,
    review_notes: str,
    sample_size: int,
) -> dict[str, object]:
    selected = sorted(
        cases,
        key=lambda case: _sha256(
            f"{assignments[case.case_id]}\0{case.case_id}".encode()
        ),
    )[:sample_size]
    return {
        "reviewer": reviewer,
        "notes": review_notes,
        "notes_sha256": _sha256(review_notes.encode("utf-8")),
        "selected_case_count": len(selected),
        "selected_cases": [
            {
                "split": assignments[case.case_id],
                "case_id": (
                    case.case_id
                    if assignments[case.case_id] == "development"
                    else public_ids[case.case_id]
                ),
            }
            for case in selected
        ],
    }


def _blind_case(case: dict[str, object], public_case_id: str) -> dict[str, object]:
    payload = {
        "case_id": public_case_id,
        "anchor": case["anchor"],
        "matching_variables": case["matching_variables"],
        "observations": case["observations"],
    }
    return _with_integrity(payload)


def _public_case_id(split: str, case_id: str, benchmark_sha256: str) -> str:
    digest = _sha256(f"{split}\0{case_id}\0{benchmark_sha256}".encode())
    return f"{split}-{digest[:24]}"


def _verify_manifest(manifest: dict[str, object], split: str) -> None:
    if manifest.get("record_type") != "firesentinel_frozen_split_manifest":
        raise ValueError("frozen manifest has an unexpected record_type")
    if (
        manifest.get("evaluation_only") is not True
        or manifest.get("frozen") is not True
    ):
        raise ValueError("frozen manifest must be evaluation-only and frozen")
    if manifest.get("split") != split:
        raise ValueError("frozen manifest split does not match its filename")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("frozen manifest cases must be a non-empty list")
    labels_visible = manifest.get("labels_visible_to_tuning")
    if labels_visible is not (split == "development"):
        raise ValueError("frozen manifest label-access policy is invalid")
    case_ids: set[str] = set()
    for raw_case in cases:
        case = _mapping(raw_case, "frozen manifest case")
        _verify_integrity(case, "frozen manifest case")
        case_id = _identifier(case.get("case_id"), "frozen manifest case_id")
        if case_id in case_ids:
            raise ValueError("frozen manifest repeats case_id")
        case_ids.add(case_id)
        if split == "development":
            if case.get("label") not in {"positive", "control"}:
                raise ValueError("development manifest must contain labels")
        else:
            if (
                "label" in case
                or "event_id" in case
                or "matched_positive_case_id" in case
            ):
                raise ValueError("blind frozen manifests must not contain label fields")
            if not _BLIND_CASE_ID.fullmatch(case_id):
                raise ValueError("blind frozen manifest case_id is not opaque")


def _verify_labels(
    label_payload: dict[str, object],
    manifest: dict[str, object],
    manifest_bytes: bytes,
    split: str,
) -> None:
    if label_payload.get("record_type") != "firesentinel_frozen_split_labels":
        raise ValueError("frozen label file has an unexpected record_type")
    if (
        label_payload.get("access") != "scoring-only"
        or label_payload.get("split") != split
    ):
        raise ValueError("frozen label file access policy is invalid")
    if label_payload.get("manifest_filename") != MANIFEST_FILENAMES[split]:
        raise ValueError("frozen labels refer to the wrong manifest")
    if label_payload.get("manifest_sha256") != _sha256(manifest_bytes):
        raise ValueError("frozen labels do not match their manifest")
    labels = label_payload.get("labels")
    if not isinstance(labels, list):
        raise ValueError("frozen labels must be a list")
    label_ids: set[str] = set()
    for raw_label in labels:
        label = _mapping(raw_label, "frozen label")
        case_id = _identifier(label.get("case_id"), "frozen label case_id")
        if label.get("label") not in {"positive", "control"}:
            raise ValueError("frozen label must be positive or control")
        label_ids.add(case_id)
    manifest_cases = manifest.get("cases")
    assert isinstance(manifest_cases, list)
    manifest_ids = {
        _identifier(_mapping(case, "frozen manifest case").get("case_id"), "case_id")
        for case in manifest_cases
    }
    if label_ids != manifest_ids or len(label_ids) != len(labels):
        raise ValueError("frozen labels do not cover their manifest exactly once")


def _assignment_rows(payload: dict[str, object]) -> list[dict[str, str]]:
    if payload.get("record_type") != "firesentinel_frozen_split_assignments":
        raise ValueError("frozen assignment file has an unexpected record_type")
    if payload.get("access") != "scoring-only":
        raise ValueError("frozen assignments must be scoring-only")
    raw_assignments = payload.get("assignments")
    if not isinstance(raw_assignments, list) or not raw_assignments:
        raise ValueError("frozen assignments must be a non-empty list")
    rows: list[dict[str, str]] = []
    ids: set[str] = set()
    for raw_row in raw_assignments:
        row = _mapping(raw_row, "frozen assignment")
        expected_fields = {
            "benchmark_case_id",
            "public_case_id",
            "split",
            "event_id",
            "geographic_cell",
            "time_period",
        }
        if set(row) != expected_fields:
            raise ValueError("frozen assignment has an invalid shape")
        converted: dict[str, str] = {}
        for field in expected_fields:
            value = row[field]
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"frozen assignment {field} must be a non-empty string"
                )
            converted[field] = value
        if converted["split"] not in SPLITS:
            raise ValueError("frozen assignment split is invalid")
        if converted["benchmark_case_id"] in ids:
            raise ValueError("frozen assignments repeat benchmark_case_id")
        ids.add(converted["benchmark_case_id"])
        rows.append(converted)
    return rows


def _assert_assignment_leakage(rows: list[dict[str, str]]) -> None:
    for field in ("event_id", "geographic_cell", "time_period"):
        split_by_value: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            split_by_value[row[field]].add(row["split"])
        leaking = next(
            (value for value, splits in split_by_value.items() if len(splits) > 1),
            None,
        )
        if leaking is not None:
            raise ValueError(
                f"frozen assignments leak {field} {leaking!r} across splits"
            )


def _verify_manifest_membership(
    manifests: dict[str, dict[str, object]],
    labels: dict[str, dict[str, object]],
    assignments: list[dict[str, str]],
) -> None:
    rows_by_split: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in assignments:
        rows_by_split[row["split"]].append(row)
    for split in SPLITS:
        manifest_cases = manifests[split].get("cases")
        assert isinstance(manifest_cases, list)
        manifest_ids = {
            _identifier(
                _mapping(case, "frozen manifest case").get("case_id"), "case_id"
            )
            for case in manifest_cases
        }
        expected_ids = {row["public_case_id"] for row in rows_by_split.get(split, [])}
        if manifest_ids != expected_ids:
            raise ValueError("frozen manifest membership differs from assignments")
        if split in labels:
            label_values = labels[split].get("labels")
            assert isinstance(label_values, list)
            label_ids = {
                _identifier(_mapping(label, "frozen label").get("case_id"), "case_id")
                for label in label_values
            }
            if label_ids != expected_ids:
                raise ValueError("frozen label membership differs from assignments")


def _assert_case_assignment_coverage(
    cases: tuple[_CaseInfo, ...], assignments: dict[str, str]
) -> None:
    if set(assignments) != {case.case_id for case in cases}:
        raise ValueError("split assignments do not cover every benchmark case")
    if any(split not in SPLITS for split in assignments.values()):
        raise ValueError("split assignment names are invalid")


def _grouping_policy() -> dict[str, object]:
    return {
        "unit": "connected_components_of_grouping_keys",
        "event_key": "positive FIRMS event_id; controls inherit matched positive event",
        "geographic_cell": "2-degree WGS84 latitude/longitude floor cell",
        "time_period": "UTC ISO calendar week of anchor acquisition",
        "split_ratios": SPLIT_RATIOS,
    }


def _expected_filenames() -> set[str]:
    return {
        FROZEN_AUDIT_FILENAME,
        ASSIGNMENTS_FILENAME,
        *MANIFEST_FILENAMES.values(),
        *LABEL_FILENAMES.values(),
    }


def _season(month: int) -> str:
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def _geographic_cell(latitude: float, longitude: float) -> str:
    latitude_cell = math.floor(latitude / 2) * 2
    longitude_cell = math.floor(longitude / 2) * 2
    return f"lat{latitude_cell:+03d}_lon{longitude_cell:+04d}"


def _time_period(moment: datetime) -> str:
    year, week, _ = moment.isocalendar()
    return f"{year:04d}-W{week:02d}"


def _load_object(path: Path, description: str) -> tuple[bytes, dict[str, object]]:
    try:
        content = path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read {description}: {path}") from error
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid {description} JSON: {path}") from error
    return content, _mapping(payload, description)


def _mapping(value: object, description: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def _timestamp(value: object, description: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{description} must be a UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(f"{value[:-1]}+00:00")
    except ValueError as error:
        raise ValueError(f"{description} must be an ISO-8601 UTC timestamp") from error
    return parsed.astimezone(UTC)


def _identifier(value: object, description: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{description} must be a lowercase identifier")
    return value


def _coordinate(
    value: object, description: str, minimum: float, maximum: float
) -> float:
    return _number(value, description, minimum, maximum)


def _number(value: object, description: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{description} must be numeric")
    converted = float(value)
    if not math.isfinite(converted) or not minimum <= converted <= maximum:
        raise ValueError(f"{description} must be within [{minimum}, {maximum}]")
    return converted


def _reviewer(value: str) -> None:
    if not value.strip():
        raise ValueError("reviewer must be non-empty")


def _review_notes(value: str) -> None:
    if not value.strip():
        raise ValueError("review_notes must be non-empty")


def _sample_size(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("manual_sample_size must be a positive integer")


def _with_integrity(payload: dict[str, object]) -> dict[str, object]:
    content = dict(payload)
    content["integrity_sha256"] = _sha256(_canonical_json_bytes(payload))
    return content


def _verify_integrity(payload: dict[str, object], description: str) -> None:
    integrity = payload.get("integrity_sha256")
    without_integrity = dict(payload)
    without_integrity.pop("integrity_sha256", None)
    if not isinstance(integrity, str) or integrity != _sha256(
        _canonical_json_bytes(without_integrity)
    ):
        raise ValueError(f"{description} integrity SHA-256 does not match")


def _canonical_json_bytes(payload: dict[str, object]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _preflight_output(path: Path, content: bytes, *, overwrite: bool) -> None:
    if not path.exists() or path.read_bytes() == content or overwrite:
        return
    raise FileExistsError(
        "Refusing to replace changed frozen evaluation artifact "
        f"'{path}'; use --overwrite."
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
    """Freeze a benchmark after a documented manual sample review."""

    evaluation_directory = default_evaluation_directory()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=evaluation_directory / "benchmark" / BENCHMARK_FILENAME,
        help="hash-audited benchmark cases under evaluation-data/",
    )
    parser.add_argument(
        "--benchmark-audit",
        type=Path,
        default=evaluation_directory / "benchmark" / BENCHMARK_AUDIT_FILENAME,
        help="hash-audited benchmark audit under evaluation-data/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_frozen_directory(),
        help="frozen manifest directory under evaluation-data/",
    )
    parser.add_argument(
        "--reviewer", required=True, help="person who inspected the sample"
    )
    parser.add_argument(
        "--review-notes",
        type=Path,
        required=True,
        help="UTF-8 plain-text notes from the manual sample inspection",
    )
    parser.add_argument(
        "--manual-sample-size",
        type=int,
        default=DEFAULT_MANUAL_SAMPLE_SIZE,
        help=(
            "deterministic number of cases to inspect "
            f"(default: {DEFAULT_MANUAL_SAMPLE_SIZE})"
        ),
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace changed frozen artifacts"
    )
    arguments = parser.parse_args(argv)
    allowed_root = evaluation_directory.resolve()
    for option, path in {
        "--benchmark": arguments.benchmark,
        "--benchmark-audit": arguments.benchmark_audit,
        "--output-dir": arguments.output_dir,
    }.items():
        if not path.resolve().is_relative_to(allowed_root):
            parser.error(f"{option} must be inside {allowed_root}")
    try:
        review_notes = arguments.review_notes.read_text(encoding="utf-8")
    except OSError as error:
        parser.error(f"cannot read --review-notes: {error}")
    frozen = freeze_benchmark(
        arguments.benchmark,
        arguments.benchmark_audit,
        reviewer=arguments.reviewer,
        review_notes=review_notes,
        manual_sample_size=arguments.manual_sample_size,
    )
    paths = write_frozen_benchmark(
        frozen, arguments.output_dir, overwrite=arguments.overwrite
    )
    verify_frozen_benchmark(arguments.output_dir, benchmark_path=arguments.benchmark)
    print(
        json.dumps(
            {
                "audit": str(paths[FROZEN_AUDIT_FILENAME]),
                "frozen_directory": str(arguments.output_dir),
                "leakage_check": "passed",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
