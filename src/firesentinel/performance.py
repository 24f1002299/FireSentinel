"""Small local profiler for the bounded evidence and reviewer replay path.

It profiles local cache/catalog reads, crop and OpenCV stages, residual artifact
work, and reviewer view-model loading.  It is intentionally a one-shot
diagnostic, not a service or telemetry system; no timings feed evidence,
outcomes, or policy decisions.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path

from firesentinel.config import load_settings
from firesentinel.data.goes18 import LocalCatalogCache
from firesentinel.ui.reviewer import discover_reviewer_cases
from firesentinel.vision.engine import EvidenceJob, load_evidence_job, run_evidence_job


def profile_local_replay(
    job: EvidenceJob, *, catalog_cache_directory: Path | None = None
) -> dict[str, object]:
    """Measure one uncached local replay without changing its evidence output."""

    if not isinstance(job, EvidenceJob):
        raise TypeError("job must be EvidenceJob")
    source_access_started = time.perf_counter()
    for source in (
        item
        for observation in job.observations
        for item in (observation.channel7, observation.channel14)
    ):
        source.source_path.stat()
    source_access = _milliseconds(time.perf_counter() - source_access_started)
    catalog_access = _catalog_access(catalog_cache_directory)

    with tempfile.TemporaryDirectory(prefix="firesentinel-profile-") as directory:
        artifacts_root = Path(directory)
        replay_started = time.perf_counter()
        result = run_evidence_job(job, artifacts_root)
        replay_elapsed = _milliseconds(time.perf_counter() - replay_started)
        evidence = _object(
            (result.artifact_directory / "evidence.json").read_bytes(), "evidence"
        )
        timings = _number_mapping(evidence.get("timings_milliseconds"), "timings")
        ui_started = time.perf_counter()
        catalog = discover_reviewer_cases(artifacts_root)
        ui_elapsed = _milliseconds(time.perf_counter() - ui_started)

    crop = _stage_total(timings, "crop:")
    prepare = _stage_total(timings, "prepare:")
    anomaly = _stage_total(timings, "anomaly:")
    align = _stage_total(timings, "align-bands:")
    persistence = timings.get("persistence", 0.0)
    known = crop + prepare + anomaly + align + persistence
    return {
        "record_type": "firesentinel_local_performance_profile",
        "schema_version": 1,
        "scope": "one uncached local evidence replay and reviewer view-model load",
        "catalog_access": catalog_access,
        "source_cache_access_milliseconds": source_access,
        "crop_loading_milliseconds": crop,
        "opencv_stages_milliseconds": {
            "prepare": prepare,
            "anomaly": anomaly,
        },
        "alignment_milliseconds": align,
        "persistence_milliseconds": persistence,
        "artifact_and_metadata_milliseconds": max(0.0, replay_elapsed - known),
        "replay_milliseconds": replay_elapsed,
        "ui_reviewer_model_loading_milliseconds": ui_elapsed,
        "reviewer_case_count": len(catalog.cases),
        "reviewer_warning_count": len(catalog.warnings),
        "stage_timings_milliseconds": dict(sorted(timings.items())),
        "evidence_content_hash": result.content_hash,
    }


def _catalog_access(directory: Path | None) -> dict[str, object]:
    if directory is None:
        return {"status": "not_profiled", "milliseconds": 0.0}
    root = Path(directory)
    snapshots = sorted(root.glob("**/*.json"))
    if not snapshots:
        return {"status": "cache_empty", "milliseconds": 0.0}
    for path in snapshots:
        try:
            payload = _object(path.read_bytes(), "catalog snapshot")
            bucket = payload.get("bucket")
            prefix = payload.get("prefix")
            if not isinstance(bucket, str) or not isinstance(prefix, str):
                continue
            started = time.perf_counter()
            snapshot = LocalCatalogCache(root).load(bucket, prefix)
            elapsed = _milliseconds(time.perf_counter() - started)
            if snapshot is not None:
                return {
                    "status": "local_cache_hit",
                    "milliseconds": elapsed,
                    "object_count": len(snapshot.objects),
                }
        except ValueError:
            continue
    return {"status": "no_valid_snapshot", "milliseconds": 0.0}


def _object(raw: bytes, field: str) -> dict[str, object]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"{field} is invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return value


def _number_mapping(value: object, field: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    result: dict[str, float] = {}
    for key, item in value.items():
        if (
            not isinstance(key, str)
            or isinstance(item, bool)
            or not isinstance(item, (int, float))
            or item < 0
        ):
            raise ValueError(f"{field} contains an invalid duration")
        result[key] = float(item)
    return result


def _stage_total(timings: Mapping[str, float], prefix: str) -> float:
    return round(
        sum(value for key, value in timings.items() if key.startswith(prefix)), 6
    )


def _milliseconds(seconds: float) -> float:
    return round(max(0.0, seconds) * 1000.0, 6)


def main(argv: list[str] | None = None) -> int:
    """Write one local, non-network performance profile for an evidence job."""

    settings = load_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--catalog-cache", type=Path, default=settings.catalog_cache_dir
    )
    arguments = parser.parse_args(argv)
    job = load_evidence_job(arguments.job)
    profile = profile_local_replay(job, catalog_cache_directory=arguments.catalog_cache)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(profile, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"output": str(arguments.output), "status": "profiled"}, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
