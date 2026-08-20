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
from urllib.parse import quote, urlparse
from urllib.request import urlopen

from firesentinel.config import load_settings
from firesentinel.data.source_cache import SourceRequest, VerifiedSourceCache
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


def _source_url(source: dict[str, Any]) -> str:
    value = source.get("source_url")
    if isinstance(value, str):
        return value
    bucket = source.get("bucket")
    object_key = source.get("object_key")
    if not isinstance(bucket, str) or not bucket or not isinstance(object_key, str):
        raise ValueError("Each source needs source_url or bucket and object_key.")
    return f"https://{bucket}.s3.amazonaws.com/{quote(object_key, safe='/')}"


def _read_source_requests(
    path: Path, case_id: str | None = None
) -> list[SourceRequest]:
    """Read selected source objects from the checked-in case manifest format."""
    raw_manifest = json.loads(path.read_text(encoding="utf-8"))
    cases = raw_manifest.get("cases")
    if cases is None and raw_manifest.get("datasets") == []:
        return []
    if not isinstance(cases, list):
        raise ValueError("Source manifest must contain a 'cases' list.")
    requests: list[SourceRequest] = []
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Each case must be a JSON object.")
        selected_case_id = case.get("case_id")
        sources = case.get("sources")
        if not isinstance(selected_case_id, str) or not isinstance(sources, list):
            raise ValueError("Each case needs case_id and a sources list.")
        if case_id is not None and selected_case_id != case_id:
            continue
        source_ids: set[str] = set()
        for source in sources:
            if not isinstance(source, dict):
                raise ValueError("Each source must be a JSON object.")
            size = source.get("size_bytes", source.get("source_size_bytes"))
            checksum = source.get("sha256", source.get("expected_sha256"))
            source_id = source.get("source_id")
            if (
                not isinstance(source_id, str)
                or isinstance(size, bool)
                or not isinstance(size, int)
            ):
                raise ValueError("Each source needs source_id and integer size_bytes.")
            if not isinstance(checksum, str):
                raise ValueError("Each source needs a lowercase SHA-256 checksum.")
            if source_id in source_ids:
                raise ValueError(
                    f"Case '{selected_case_id}' must not repeat "
                    f"source_id '{source_id}'."
                )
            source_ids.add(source_id)
            requests.append(
                SourceRequest(
                    case_id=selected_case_id,
                    source_id=source_id,
                    source_url=_source_url(source),
                    source_size_bytes=size,
                    expected_sha256=checksum,
                )
            )
    return requests


def main(argv: list[str] | None = None) -> int:
    settings = load_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("download", "inspect", "clean-case"),
        nargs="?",
        default="download",
        help="operation to perform (default: download)",
    )
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
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=settings.source_cache_dir,
        help="verified source cache directory (default: data/source-cache/)",
    )
    parser.add_argument("--case-id", help="limit download or inspection to one case")
    parser.add_argument(
        "--timeout-seconds", type=float, default=30.0, help="per-request timeout"
    )
    arguments = parser.parse_args(argv)
    configure_logging(settings)
    cache = VerifiedSourceCache(arguments.cache_dir)
    if arguments.command == "inspect":
        print(json.dumps(cache.inspect(arguments.case_id).to_dict(), sort_keys=True))
        return 0
    if arguments.command == "clean-case":
        if arguments.case_id is None:
            parser.error("clean-case requires --case-id")
        removed = cache.clean_case(arguments.case_id)
        print(json.dumps({"case_id": arguments.case_id, "removed_sources": removed}))
        return 0
    raw_manifest = json.loads(arguments.manifest.read_text(encoding="utf-8"))
    if "datasets" in raw_manifest and "cases" not in raw_manifest:
        datasets = _read_manifest(arguments.manifest)
        if not datasets:
            LOGGER.info("manifest contains no sources; nothing to download")
            return 0
        for dataset in datasets:
            _download_dataset(dataset, arguments.output)
        return 0
    requests = _read_source_requests(arguments.manifest, arguments.case_id)
    if not requests:
        LOGGER.info("manifest contains no selected sources; nothing to download")
        return 0
    for request in requests:
        receipt = cache.fetch(request, timeout_seconds=arguments.timeout_seconds)
        print(json.dumps(receipt.to_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
