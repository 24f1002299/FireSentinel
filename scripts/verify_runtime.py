"""Deterministically verify the local OpenCV 5 runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCKFILE = PROJECT_ROOT / "requirements.lock"

# Filled from the OpenCV 5.0.0.93 / NumPy 2.5.1 clean-install baseline.
EXPECTED_SMOKE_HASHES: dict[str, str] = {
    "connected_components": "848118a2093daef2007a50f85d0ef3b996e6005a4861b159ee127c2b8113c596",
    "contours": "fa5e7722896ce59a8d11c59c1c5afa010a13848649230678c7e0c2e92926fbae",
    "morphology": "723dd2aae1b12f1084fcac96b32ce2cc91ed39a8aec66a347050890a053fb9e8",
    "resize": "f94ea3c3035b45cc71787bf9eeca03dd1c2d0a02d931dcb8e07fb26df774ff98",
    "threshold": "f1ec383ae048869243ba20b29992f5e8149e96456cb817f6490a61c2f66fee68",
}


def _sha256(*arrays: np.ndarray) -> str:
    """Hash array shape, dtype, and contiguous bytes without ambiguity."""
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _fixture() -> np.ndarray:
    """A tiny fixed scene with gradients, two hot targets, and cold background."""
    return np.array(
        [
            [3, 8, 12, 16, 20, 24, 28, 32],
            [7, 14, 22, 30, 38, 46, 54, 62],
            [11, 20, 31, 42, 53, 64, 75, 86],
            [15, 26, 255, 255, 68, 79, 90, 101],
            [19, 32, 255, 255, 83, 94, 105, 116],
            [23, 38, 49, 60, 71, 82, 93, 104],
            [27, 44, 61, 78, 95, 112, 255, 255],
            [31, 50, 69, 88, 107, 126, 255, 255],
        ],
        dtype=np.uint8,
    )


def smoke_hashes() -> dict[str, str]:
    """Exercise the OpenCV operations required by the vision pipeline."""
    source = _fixture()
    resized = cv2.resize(source, (13, 11), interpolation=cv2.INTER_LINEAR)
    _, threshold = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morphology = cv2.morphologyEx(
        threshold,
        cv2.MORPH_CLOSE,
        kernel,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        morphology,
        connectivity=8,
        ltype=cv2.CV_32S,
    )
    contours, hierarchy = cv2.findContours(
        morphology.copy(),
        mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_SIMPLE,
    )
    contour_bytes = np.frombuffer(
        b"".join(np.ascontiguousarray(contour).tobytes() for contour in contours),
        dtype=np.uint8,
    )
    hierarchy_array = (
        np.empty((0,), dtype=np.int32) if hierarchy is None else np.asarray(hierarchy)
    )
    component_count_array = np.asarray([component_count], dtype=np.int32)
    return {
        "resize": _sha256(resized),
        "threshold": _sha256(threshold),
        "morphology": _sha256(morphology),
        "connected_components": _sha256(
            component_count_array, labels, stats, centroids
        ),
        "contours": _sha256(contour_bytes, hierarchy_array),
    }


def runtime_report() -> dict[str, Any]:
    """Return portable runtime provenance and the deterministic smoke evidence."""
    build_information = cv2.getBuildInformation()
    lock_contents = LOCKFILE.read_bytes() if LOCKFILE.exists() else b""
    return {
        "cpu_architecture": {
            "machine": platform.machine(),
            "process_architecture": platform.architecture()[0],
            "processor": platform.processor() or "unreported",
        },
        "dependency_lock": {
            "file": LOCKFILE.name,
            "sha256": hashlib.sha256(lock_contents).hexdigest(),
        },
        "opencv": {
            "build_information_sha256": hashlib.sha256(
                build_information.encode("utf-8")
            ).hexdigest(),
            "version": cv2.__version__,
        },
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "smoke_hashes": smoke_hashes(),
    }


def _assert_baseline(report: dict[str, Any]) -> None:
    if report["opencv"]["version"].split(".", maxsplit=1)[0] != "5":
        raise RuntimeError(
            f"OpenCV 5 is required, found {report['opencv']['version']}."
        )
    if report["smoke_hashes"] != EXPECTED_SMOKE_HASHES:
        raise RuntimeError(
            "Smoke-test hash mismatch. Expected "
            f"{EXPECTED_SMOKE_HASHES}, got {report['smoke_hashes']}."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-report",
        type=Path,
        help="write a JSON runtime report to this path",
    )
    parser.add_argument(
        "--write-build-info",
        type=Path,
        help="write cv2.getBuildInformation() to this path",
    )
    arguments = parser.parse_args()
    report = runtime_report()
    _assert_baseline(report)
    if arguments.write_report:
        arguments.write_report.parent.mkdir(parents=True, exist_ok=True)
        arguments.write_report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if arguments.write_build_info:
        arguments.write_build_info.parent.mkdir(parents=True, exist_ok=True)
        arguments.write_build_info.write_text(
            cv2.getBuildInformation(), encoding="utf-8"
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
