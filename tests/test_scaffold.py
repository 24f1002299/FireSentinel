from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from firesentinel.config import load_settings
from firesentinel.data.download import _read_manifest
from firesentinel.logging import JsonFormatter


def test_settings_have_repository_local_defaults(tmp_path: Path) -> None:
    settings = load_settings(root=tmp_path)

    assert settings.root_dir == tmp_path.resolve()
    assert settings.data_dir == tmp_path / "data"
    assert settings.artifacts_dir == tmp_path / "artifacts"
    assert settings.manifests_dir == tmp_path / "manifests"
    assert settings.catalog_cache_dir == tmp_path / "data" / "catalog"
    assert settings.log_level == "INFO"


def test_settings_reject_an_unknown_log_level(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="FIRE_SENTINEL_LOG_LEVEL"):
        load_settings({"FIRE_SENTINEL_LOG_LEVEL": "verbose"}, root=tmp_path)


def test_json_formatter_emits_machine_readable_events() -> None:
    record = logging.LogRecord(
        name="firesentinel.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="ready: %s",
        args=("yes",),
        exc_info=None,
    )

    event = json.loads(JsonFormatter().format(record))

    assert event["level"] == "INFO"
    assert event["logger"] == "firesentinel.test"
    assert event["message"] == "ready: yes"
    assert event["timestamp"].endswith("Z")


def test_empty_dataset_manifest_is_safe(tmp_path: Path) -> None:
    manifest = tmp_path / "datasets.json"
    manifest.write_text('{"datasets": []}\n', encoding="utf-8")

    assert _read_manifest(manifest) == []
