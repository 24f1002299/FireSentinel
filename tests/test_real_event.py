"""Deterministic contracts for the cached-only Day 9 OpenCV slice."""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from firesentinel.core.records import Channel, Coordinates
from firesentinel.data.goes_crop import CropParameters, GeographicBounds
from firesentinel.data.source_cache import (
    DownloadResponse,
    VerifiedSourceCache,
)
from firesentinel.vision.real_event import (
    EventObservation,
    EventSource,
    RealEventManifest,
    SliceConfiguration,
    analyse_frame,
    replay_real_event,
)
from tests.test_goes_crop import _latitude_longitude, _source


class BytesOpener:
    def __init__(self, contents: bytes) -> None:
        self.contents = contents

    def __call__(self, _: str, __: float) -> DownloadResponse:
        return io.BytesIO(self.contents)


def test_checked_in_park_fire_manifest_is_a_valid_two_frame_c07_slice() -> None:
    manifest_path = Path(__file__).parents[1] / "manifests" / "park-fire-20240725.json"

    manifest = RealEventManifest.from_path(manifest_path)

    assert manifest.case_id == "park-fire-20240725"
    assert tuple(item.channel for item in manifest.observations) == (
        Channel.C07,
        Channel.C07,
    )
    assert manifest.observations[1].scan_start > manifest.observations[0].scan_start
    assert set(manifest.expected_outputs) == {
        "evidence_content_hash",
        "reviewer_panel_sha256",
    }


def test_opencv_pipeline_scales_thresholds_morphs_and_measures_contours() -> None:
    frame = np.full((9, 9), 300.0, dtype=np.float32)
    frame[3:6, 3:6] = 350.0
    configuration = SliceConfiguration(280.0, 380.0, 330.0, 3, 1)

    first = analyse_frame(frame, np.zeros(frame.shape, dtype=bool), configuration)
    second = analyse_frame(frame, np.zeros(frame.shape, dtype=bool), configuration)

    assert first.display.dtype == np.uint8
    assert first.threshold_mask[4, 4] == 255
    assert first.morphology_mask[4, 4] == 255
    assert first.components[0].area_pixels == 5
    assert first.components[0].bounding_box_xywh == (3, 3, 3, 3)
    assert first.contours == second.contours
    assert first.contour_hash() == second.contour_hash()


def _manifest_and_cache(
    tmp_path: Path,
) -> tuple[RealEventManifest, VerifiedSourceCache]:
    source_path = tmp_path / "source.nc"
    _source(source_path)
    contents = source_path.read_bytes()
    checksum = hashlib.sha256(contents).hexdigest()
    cache = VerifiedSourceCache(tmp_path / "cache", opener=BytesOpener(contents))
    sources = (
        EventSource(
            "c07-initial", "noaa-goes18", "initial.nc", len(contents), checksum
        ),
        EventSource("c07-later", "noaa-goes18", "later.nc", len(contents), checksum),
    )
    for source in sources:
        cache.fetch(source.request_for("slice-case"))
    latitude, longitude = _latitude_longitude(3, 4)
    timing = datetime(2025, 1, 1, tzinfo=UTC)
    observations = (
        EventObservation(
            "initial",
            "c07-initial",
            Channel.C07,
            timing,
            timing,
            timing + timedelta(minutes=9),
        ),
        EventObservation(
            "later",
            "c07-later",
            Channel.C07,
            timing + timedelta(minutes=20),
            timing + timedelta(minutes=20),
            timing + timedelta(minutes=29),
        ),
    )
    manifest = RealEventManifest(
        case_id="slice-case",
        title="Cached test event",
        location=Coordinates(latitude, longitude),
        crop_parameters=CropParameters(
            GeographicBounds(
                latitude - 0.08, longitude - 0.08, latitude + 0.08, longitude + 0.08
            ),
            padding_pixels=1,
        ),
        configuration=SliceConfiguration(200.0, 240.0, 220.0, 1, 1),
        sources=sources,
        observations=observations,
        expected_outputs={
            "evidence_content_hash": "0" * 64,
            "reviewer_panel_sha256": "0" * 64,
        },
    )
    return manifest, cache


def test_cached_replay_writes_repeatable_evidence_and_reviewer_panel(
    tmp_path: Path,
) -> None:
    manifest, cache = _manifest_and_cache(tmp_path)
    artifacts = tmp_path / "artifacts"

    first = replay_real_event(manifest, cache, artifacts, verify=False)
    evidence_content_hash = first["evidence_content_hash"]
    reviewer_panel_sha256 = first["reviewer_panel_sha256"]
    assert isinstance(evidence_content_hash, str)
    assert isinstance(reviewer_panel_sha256, str)
    verified_manifest = replace(
        manifest,
        expected_outputs={
            "evidence_content_hash": evidence_content_hash,
            "reviewer_panel_sha256": reviewer_panel_sha256,
        },
    )
    second = replay_real_event(verified_manifest, cache, artifacts, verify=True)
    artifact_directory = second["artifact_directory"]
    assert isinstance(artifact_directory, str)
    evidence_path = Path(artifact_directory) / "evidence.json"
    panel_path = Path(artifact_directory) / "before-after.png"

    assert second["verified"]
    assert first["evidence_content_hash"] == second["evidence_content_hash"]
    assert (
        hashlib.sha256(panel_path.read_bytes()).hexdigest()
        == second["reviewer_panel_sha256"]
    )
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["record_type"] == "real_event_evidence"
    assert [frame["contour_sha256"] for frame in evidence["frames"]] == [
        second["initial_contour_sha256"],
        second["later_contour_sha256"],
    ]
