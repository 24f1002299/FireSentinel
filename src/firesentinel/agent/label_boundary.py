"""Runtime protection for evaluation-only label artifacts."""

from __future__ import annotations

from pathlib import Path

EVALUATION_LABELS_DIRECTORY_NAME = "evaluation-data"


def runtime_input_path(path: Path, *, project_root: Path) -> Path:
    """Resolve a runtime input while refusing the evaluation label subtree.

    Evaluation labels intentionally live outside ``Settings``.  This narrow
    boundary protects the generic replay input from becoming an accidental path
    around that configuration isolation.
    """

    resolved_path = path.resolve()
    labels_root = (project_root / EVALUATION_LABELS_DIRECTORY_NAME).resolve()
    if resolved_path.is_relative_to(labels_root):
        raise ValueError(
            "agent runtime inputs cannot read evaluation-only labels under "
            f"{labels_root}"
        )
    return resolved_path
