"""Boundary for model-selection commands that consume frozen manifests.

Only the frozen development manifest may be used for tuning.  Test and stress
labels live in scoring-only files and this boundary rejects both their manifests
and their label/assignment artifacts before a tuning implementation can read
them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from firesentinel.evaluation.freeze import default_frozen_directory


def tuning_manifest_path(path: Path, *, project_root: Path) -> Path:
    """Resolve a tuning manifest while allowing only frozen development data."""

    resolved = path.resolve()
    frozen_root = (project_root / "evaluation-data" / "frozen").resolve()
    if not resolved.is_relative_to(frozen_root):
        raise ValueError(
            f"tuning inputs must be the frozen development manifest under {frozen_root}"
        )
    if resolved.name != "development.manifest.json":
        raise ValueError(
            "tuning commands may read only the frozen development manifest; "
            "test and stress labels are scoring-only"
        )
    try:
        payload = json.loads(resolved.read_bytes())
    except OSError as error:
        raise ValueError(f"cannot read tuning manifest: {resolved}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid tuning manifest JSON: {resolved}") from error
    if not isinstance(payload, dict):
        raise ValueError("tuning manifest must be a JSON object")
    if (
        payload.get("record_type") != "firesentinel_frozen_split_manifest"
        or payload.get("split") != "development"
        or payload.get("labels_visible_to_tuning") is not True
        or payload.get("frozen") is not True
    ):
        raise ValueError(
            "tuning commands may read only the frozen development manifest; "
            "test and stress labels are scoring-only"
        )
    return resolved


def main(argv: list[str] | None = None) -> int:
    """Validate the sole permitted input boundary for a future tuning command."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=default_frozen_directory() / "development.manifest.json",
        help="frozen development manifest; test and stress inputs are rejected",
    )
    arguments = parser.parse_args(argv)
    project_root = Path(__file__).resolve().parents[3]
    manifest = tuning_manifest_path(arguments.manifest, project_root=project_root)
    print(json.dumps({"tuning_manifest": str(manifest)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
