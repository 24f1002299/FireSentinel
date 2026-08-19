"""Download only the datasets explicitly listed in a JSON manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen

from firesentinel.config import load_settings
from firesentinel.logging import configure_logging

LOGGER = logging.getLogger("firesentinel.data.download")


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    raw_manifest = json.loads(path.read_text(encoding="utf-8"))
    datasets = raw_manifest.get("datasets")
    if not isinstance(datasets, list):
        raise ValueError("Manifest must contain a 'datasets' list.")
    return datasets


def _download_dataset(dataset: dict[str, Any], destination: Path) -> None:
    name = dataset.get("name")
    source_url = dataset.get("source_url")
    sha256 = dataset.get("sha256")
    if not isinstance(name, str) or not name:
        raise ValueError("Each dataset needs a non-empty 'name'.")
    if not isinstance(source_url, str) or urlparse(source_url).scheme not in {
        "http",
        "https",
    }:
        raise ValueError(f"Dataset '{name}' needs an http(s) 'source_url'.")

    destination = destination.resolve()
    target = (destination / name).resolve()
    if not target.is_relative_to(destination):
        raise ValueError(f"Dataset '{name}' must stay within the output directory.")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=target.parent) as temporary:
        temporary_path = Path(temporary.name)
        with urlopen(source_url) as response:  # noqa: S310 - manifest is explicit input.
            shutil.copyfileobj(response, temporary)
    if isinstance(sha256, str):
        with temporary_path.open("rb") as downloaded_file:
            actual = hashlib.file_digest(downloaded_file, "sha256").hexdigest()
        if actual != sha256:
            temporary_path.unlink(missing_ok=True)
            raise ValueError(f"Dataset '{name}' failed its SHA-256 check.")
    temporary_path.replace(target)
    LOGGER.info("downloaded dataset: %s", target)


def main(argv: list[str] | None = None) -> int:
    settings = load_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=settings.manifests_dir / "datasets.json",
        help="JSON manifest to download (default: manifests/datasets.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.data_dir,
        help="destination directory (default: data/)",
    )
    arguments = parser.parse_args(argv)
    configure_logging(settings)
    datasets = _read_manifest(arguments.manifest)
    if not datasets:
        LOGGER.info("manifest contains no datasets; nothing to download")
        return 0
    for dataset in datasets:
        _download_dataset(dataset, arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
