"""Explicit, environment-backed configuration for local development."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

ENV_PREFIX = "FIRE_SENTINEL_"
VALID_LOG_LEVELS = frozenset({"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"})


def project_root() -> Path:
    """Return the repository root without depending on the current directory."""
    return Path(__file__).resolve().parents[2]


def _path_setting(value: str | None, default: Path, root: Path) -> Path:
    if not value:
        return default
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


@dataclass(frozen=True, slots=True)
class Settings:
    """The complete set of configuration values used by this scaffold."""

    root_dir: Path
    data_dir: Path
    artifacts_dir: Path
    manifests_dir: Path
    catalog_cache_dir: Path
    log_level: str


def load_settings(
    environ: Mapping[str, str] | None = None, *, root: Path | None = None
) -> Settings:
    """Load documented optional settings; no configuration file is required."""
    values = os.environ if environ is None else environ
    root_dir = (project_root() if root is None else root).resolve()
    log_level = values.get(f"{ENV_PREFIX}LOG_LEVEL", "INFO").upper()
    if log_level not in VALID_LOG_LEVELS:
        valid = ", ".join(sorted(VALID_LOG_LEVELS))
        raise ValueError(f"{ENV_PREFIX}LOG_LEVEL must be one of: {valid}.")
    return Settings(
        root_dir=root_dir,
        data_dir=_path_setting(
            values.get(f"{ENV_PREFIX}DATA_DIR"), root_dir / "data", root_dir
        ),
        artifacts_dir=_path_setting(
            values.get(f"{ENV_PREFIX}ARTIFACTS_DIR"),
            root_dir / "artifacts",
            root_dir,
        ),
        manifests_dir=_path_setting(
            values.get(f"{ENV_PREFIX}MANIFESTS_DIR"),
            root_dir / "manifests",
            root_dir,
        ),
        catalog_cache_dir=_path_setting(
            values.get(f"{ENV_PREFIX}CATALOG_CACHE_DIR"),
            root_dir / "data" / "catalog",
            root_dir,
        ),
        log_level=log_level,
    )
