"""Offline contracts for verified, immutable source-object caching."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest

from firesentinel.data.download import _read_source_requests
from firesentinel.data.source_cache import (
    DownloadResponse,
    SourceCacheCorruptionError,
    SourceCacheError,
    SourceChecksumError,
    SourceRequest,
    VerifiedSourceCache,
)


class InterruptedResponse:
    def __init__(self, prefix: bytes) -> None:
        self.prefix = prefix
        self.reads = 0

    def __enter__(self) -> InterruptedResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, _: int) -> bytes:
        self.reads += 1
        if self.reads == 1:
            return self.prefix
        raise OSError("connection interrupted")


class FakeOpener:
    def __init__(self, responses: list[io.BytesIO | InterruptedResponse]) -> None:
        self.responses = responses
        self.calls = 0

    def __call__(self, url: str, timeout_seconds: float) -> DownloadResponse:
        assert url == "https://example.test/object.nc"
        assert timeout_seconds == 5.0
        response = self.responses[self.calls]
        self.calls += 1
        return response  # type: ignore[return-value]


def _request(payload: bytes, *, checksum: str | None = None) -> SourceRequest:
    return SourceRequest(
        case_id="pine-creek",
        source_id="c07-001",
        source_url="https://example.test/object.nc",
        source_size_bytes=len(payload),
        expected_sha256=checksum,
    )


def test_interrupted_transfer_is_retried_and_never_published_partially(
    tmp_path: Path,
) -> None:
    payload = b"verified NOAA bytes"
    opener = FakeOpener([InterruptedResponse(payload[:5]), io.BytesIO(payload)])
    cache = VerifiedSourceCache(
        tmp_path, opener=opener, retries=1, sleep=lambda _: None
    )

    receipt = cache.fetch(_request(payload), timeout_seconds=5.0)

    assert receipt.attempts == 2
    assert receipt.downloaded_bytes == len(payload)
    assert receipt.cache_path.read_bytes() == payload
    assert not tuple((tmp_path / "temporary").iterdir())
    assert not tuple((tmp_path / "objects").rglob("*.part"))


def test_checksum_failure_rejects_bytes_without_creating_a_cache_entry(
    tmp_path: Path,
) -> None:
    payload = b"not the expected object"
    expected = hashlib.sha256(b"different object").hexdigest()
    cache = VerifiedSourceCache(tmp_path, opener=FakeOpener([io.BytesIO(payload)]))

    with pytest.raises(SourceChecksumError):
        cache.fetch(_request(payload, checksum=expected), timeout_seconds=5.0)

    assert not (tmp_path / "objects").exists()
    assert not (tmp_path / "sources").exists()
    assert not tuple((tmp_path / "temporary").iterdir())


def test_terminal_interruption_never_appears_as_a_complete_cached_object(
    tmp_path: Path,
) -> None:
    payload = b"only a partial transfer"
    cache = VerifiedSourceCache(
        tmp_path,
        opener=FakeOpener([InterruptedResponse(payload[:4])]),
        retries=0,
    )

    with pytest.raises(SourceCacheError):
        cache.fetch(_request(payload), timeout_seconds=5.0)

    assert not (tmp_path / "objects").exists()
    assert not (tmp_path / "sources").exists()
    assert not tuple((tmp_path / "temporary").iterdir())


def test_second_request_uses_the_verified_cache_and_records_a_case_reference(
    tmp_path: Path,
) -> None:
    payload = b"same immutable object"
    checksum = hashlib.sha256(payload).hexdigest()
    opener = FakeOpener([io.BytesIO(payload)])
    cache = VerifiedSourceCache(tmp_path, opener=opener)
    request = _request(payload, checksum=checksum)

    first = cache.fetch(request, timeout_seconds=5.0)
    second = cache.fetch(request, timeout_seconds=5.0)

    assert not first.cache_hit
    assert second.cache_hit
    assert second.downloaded_bytes == 0
    assert opener.calls == 1
    assert cache.inspect("pine-creek").source_count == 1


def test_corrupt_cached_bytes_are_rejected_not_reported_as_a_hit(
    tmp_path: Path,
) -> None:
    payload = b"immutable object"
    cache = VerifiedSourceCache(tmp_path, opener=FakeOpener([io.BytesIO(payload)]))
    request = _request(payload)
    receipt = cache.fetch(request, timeout_seconds=5.0)
    receipt.cache_path.write_bytes(b"corrupt")

    with pytest.raises(SourceCacheCorruptionError):
        cache.fetch(request, timeout_seconds=5.0)


def test_case_cleanup_keeps_objects_referenced_by_another_case(tmp_path: Path) -> None:
    payload = b"shared bytes"
    opener = FakeOpener([io.BytesIO(payload)])
    cache = VerifiedSourceCache(tmp_path, opener=opener)
    first = cache.fetch(_request(payload), timeout_seconds=5.0)
    other_case = SourceRequest(
        case_id="cedar-ridge",
        source_id="c07-001",
        source_url="https://example.test/object.nc",
        source_size_bytes=len(payload),
    )
    second = cache.fetch(other_case, timeout_seconds=5.0)

    assert second.cache_hit
    assert cache.clean_case("pine-creek") == 1
    assert first.cache_path.exists()
    assert cache.clean_case("cedar-ridge") == 1
    assert not first.cache_path.exists()


def test_case_manifest_converts_anonymous_s3_object_to_a_pinned_request(
    tmp_path: Path,
) -> None:
    checksum = hashlib.sha256(b"object").hexdigest()
    manifest = tmp_path / "cases.json"
    manifest.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "pine-creek",
                        "sources": [
                            {
                                "source_id": "c07-001",
                                "bucket": "noaa-goes18",
                                "object_key": "ABI-L2-CMIPF/a source.nc",
                                "size_bytes": 6,
                                "sha256": checksum,
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    request = _read_source_requests(manifest)

    assert request == [
        SourceRequest(
            case_id="pine-creek",
            source_id="c07-001",
            source_url="https://noaa-goes18.s3.amazonaws.com/ABI-L2-CMIPF/a%20source.nc",
            source_size_bytes=6,
            expected_sha256=checksum,
        )
    ]
