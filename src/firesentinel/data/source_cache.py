"""Verified, content-addressed storage for selected external source objects.

Only a fully downloaded object with a validated size and SHA-256 digest is ever
published to the cache.  Transfer files live in a private temporary directory
and are removed after every failed attempt, so an interrupted transfer cannot
be mistaken for a cached source.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol, Self, cast
from urllib.parse import urlparse
from urllib.request import urlopen

_IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9_-]{0,79}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SCHEMA_VERSION = 1
_CHUNK_SIZE = 1024 * 1024


class SourceCacheError(RuntimeError):
    """Base class for verified source-cache failures."""


class SourceChecksumError(SourceCacheError):
    """Raised when bytes do not match a requested or cached checksum."""


class SourceSizeError(SourceCacheError):
    """Raised when a transfer does not have the declared source size."""


class SourceCacheCorruptionError(SourceCacheError):
    """Raised when an existing cache entry no longer verifies."""


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase identifier")
    return value


def _size(value: object, field: str = "source_size_bytes") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _checksum(value: object | None, field: str = "expected_sha256") -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


def _source_url(value: object) -> str:
    if not isinstance(value, str) or urlparse(value).scheme not in {"http", "https"}:
        raise ValueError("source_url must be an http(s) URL")
    return value


@dataclass(frozen=True, slots=True)
class SourceRequest:
    """One immutable source selected for a pinned case."""

    case_id: str
    source_id: str
    source_url: str
    source_size_bytes: int
    expected_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _identifier(self.case_id, "case_id"))
        object.__setattr__(self, "source_id", _identifier(self.source_id, "source_id"))
        object.__setattr__(self, "source_url", _source_url(self.source_url))
        object.__setattr__(self, "source_size_bytes", _size(self.source_size_bytes))
        object.__setattr__(self, "expected_sha256", _checksum(self.expected_sha256))

    @property
    def cache_key(self) -> str:
        """Return a stable identity for this remote source selection."""
        payload = json.dumps(
            {
                "source_url": self.source_url,
                "source_size_bytes": self.source_size_bytes,
                "expected_sha256": self.expected_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class DownloadReceipt:
    """Measured outcome of one source request, including verified cache hits."""

    case_id: str
    source_id: str
    source_size_bytes: int
    downloaded_bytes: int
    checksum: str
    elapsed_seconds: float
    cache_hit: bool
    attempts: int
    cache_path: Path

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["cache_path"] = str(self.cache_path)
        return result


@dataclass(frozen=True, slots=True)
class CacheInspection:
    """A read-only summary of verified cache state."""

    object_count: int
    object_bytes: int
    case_count: int
    case_id: str | None = None
    source_count: int | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class DownloadResponse(Protocol):
    """The minimal response interface used by the standard-library downloader."""

    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> object: ...

    def read(self, size: int = -1) -> bytes: ...


Opener = Callable[[str, float], DownloadResponse]


class VerifiedSourceCache:
    """Atomic, content-addressed cache with case-scoped reference indexes."""

    def __init__(
        self,
        directory: Path,
        *,
        opener: Opener | None = None,
        retries: int = 2,
        retry_delay_seconds: float = 0.2,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
            raise ValueError("retries must be a non-negative integer")
        if retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds must not be negative")
        self.directory = Path(directory)
        self._opener = self._default_opener if opener is None else opener
        self._retries = retries
        self._retry_delay_seconds = retry_delay_seconds
        self._sleep = sleep
        self._clock = clock

    @staticmethod
    def _default_opener(url: str, timeout_seconds: float) -> DownloadResponse:
        return cast(
            DownloadResponse,
            urlopen(url, timeout=timeout_seconds),  # noqa: S310 - pinned source URL.
        )

    @property
    def objects_dir(self) -> Path:
        return self.directory / "objects"

    @property
    def sources_dir(self) -> Path:
        return self.directory / "sources"

    @property
    def cases_dir(self) -> Path:
        return self.directory / "cases"

    def path_for_checksum(self, checksum: str) -> Path:
        digest = _checksum(checksum, "checksum")
        assert digest is not None
        return self.objects_dir / digest[:2] / digest

    def fetch(
        self, request: SourceRequest, *, timeout_seconds: float = 30.0
    ) -> DownloadReceipt:
        """Return a verified object, downloading it only when no valid hit exists."""
        if not isinstance(request, SourceRequest):
            raise TypeError("request must be a SourceRequest")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        started = self._clock()
        hit = self._load_verified_source(request)
        if hit is not None:
            checksum, path = hit
            self._record_case_reference(request, checksum)
            return DownloadReceipt(
                request.case_id,
                request.source_id,
                request.source_size_bytes,
                0,
                checksum,
                self._clock() - started,
                True,
                0,
                path,
            )

        downloaded_bytes = 0
        attempts = 0
        for attempt in range(self._retries + 1):
            attempts = attempt + 1
            temporary_path: Path | None = None
            try:
                temporary_path, transfer_bytes, checksum = self._transfer(
                    request, timeout_seconds
                )
                downloaded_bytes += transfer_bytes
                path = self.path_for_checksum(checksum)
                self._publish_object(temporary_path, path)
                temporary_path = None
                self._store_source_index(request, checksum)
                self._record_case_reference(request, checksum)
                return DownloadReceipt(
                    request.case_id,
                    request.source_id,
                    request.source_size_bytes,
                    downloaded_bytes,
                    checksum,
                    self._clock() - started,
                    False,
                    attempts,
                    path,
                )
            except (SourceChecksumError, SourceSizeError):
                raise
            except (OSError, TimeoutError) as error:
                if attempt == self._retries:
                    raise SourceCacheError(
                        "could not download "
                        f"{request.source_url} after {attempts} attempts"
                    ) from error
                self._sleep(self._retry_delay_seconds * (2**attempt))
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
        raise AssertionError("download retry loop should always return or raise")

    def require_cached(self, request: SourceRequest) -> Path:
        """Return a verified cached path without performing network I/O.

        Replay stages must be able to prove that they only consumed the exact
        bytes selected by a pinned manifest.  Unlike :meth:`fetch`, this
        method never creates a reference, writes cache metadata, or falls back
        to a download when the requested source has not already been cached.
        """
        if not isinstance(request, SourceRequest):
            raise TypeError("request must be a SourceRequest")
        hit = self._load_verified_source(request)
        if hit is None:
            raise SourceCacheError(
                "required verified source is absent from the local cache: "
                f"{request.source_id}"
            )
        _, path = hit
        return path

    def _transfer(
        self, request: SourceRequest, timeout_seconds: float
    ) -> tuple[Path, int, str]:
        temporary_dir = self.directory / "temporary"
        temporary_dir.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                delete=False, dir=temporary_dir
            ) as temporary:
                temporary_path = Path(temporary.name)
                digest = hashlib.sha256()
                transferred = 0
                with self._opener(request.source_url, timeout_seconds) as response:
                    while chunk := response.read(_CHUNK_SIZE):
                        temporary.write(chunk)
                        digest.update(chunk)
                        transferred += len(chunk)
        except BaseException:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise
        assert temporary_path is not None
        if transferred != request.source_size_bytes:
            temporary_path.unlink(missing_ok=True)
            raise SourceSizeError(
                "source size mismatch: "
                f"expected {request.source_size_bytes}, got {transferred}"
            )
        checksum = digest.hexdigest()
        if request.expected_sha256 is not None and checksum != request.expected_sha256:
            temporary_path.unlink(missing_ok=True)
            raise SourceChecksumError("downloaded source failed its SHA-256 check")
        return temporary_path, transferred, checksum

    def _load_verified_source(self, request: SourceRequest) -> tuple[str, Path] | None:
        path = self.sources_dir / f"{request.cache_key}.json"
        try:
            payload = self._read_json(path)
        except FileNotFoundError:
            return None
        if not isinstance(payload, dict) or set(payload) != {
            "checksum",
            "expected_sha256",
            "schema_version",
            "source_size_bytes",
            "source_url",
        }:
            raise SourceCacheCorruptionError(f"invalid source cache index: {path}")
        if payload["schema_version"] != _SCHEMA_VERSION:
            raise SourceCacheCorruptionError(f"unsupported source cache index: {path}")
        if (
            payload["source_url"] != request.source_url
            or payload["source_size_bytes"] != request.source_size_bytes
            or payload["expected_sha256"] != request.expected_sha256
        ):
            raise SourceCacheCorruptionError(
                f"source cache index does not match: {path}"
            )
        checksum = _checksum(payload["checksum"], "cached checksum")
        assert checksum is not None
        object_path = self.path_for_checksum(checksum)
        self._verify_object(object_path, checksum, request.source_size_bytes)
        return checksum, object_path

    def _verify_object(self, path: Path, checksum: str, size: int) -> None:
        try:
            if path.stat().st_size != size:
                raise SourceCacheCorruptionError(
                    f"cached source has wrong size: {path}"
                )
            with path.open("rb") as cached:
                actual = hashlib.file_digest(cached, "sha256").hexdigest()
        except FileNotFoundError as error:
            raise SourceCacheCorruptionError(
                f"cached source is missing: {path}"
            ) from error
        except OSError as error:
            raise SourceCacheCorruptionError(
                f"cannot read cached source: {path}"
            ) from error
        if actual != checksum:
            raise SourceCacheCorruptionError(
                f"cached source failed SHA-256 check: {path}"
            )

    def _publish_object(self, temporary_path: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            # A concurrent request may have published the same immutable blob.
            temporary_path.unlink(missing_ok=True)
            return
        os.replace(temporary_path, destination)

    def _store_source_index(self, request: SourceRequest, checksum: str) -> None:
        self._write_json_atomic(
            self.sources_dir / f"{request.cache_key}.json",
            {
                "schema_version": _SCHEMA_VERSION,
                "source_url": request.source_url,
                "source_size_bytes": request.source_size_bytes,
                "expected_sha256": request.expected_sha256,
                "checksum": checksum,
            },
        )

    def _record_case_reference(self, request: SourceRequest, checksum: str) -> None:
        path = self.cases_dir / f"{request.case_id}.json"
        try:
            payload = self._read_json(path)
        except FileNotFoundError:
            payload = {
                "schema_version": _SCHEMA_VERSION,
                "case_id": request.case_id,
                "sources": {},
            }
        if not isinstance(payload, dict) or set(payload) != {
            "schema_version",
            "case_id",
            "sources",
        }:
            raise SourceCacheCorruptionError(f"invalid case cache index: {path}")
        if (
            payload["schema_version"] != _SCHEMA_VERSION
            or payload["case_id"] != request.case_id
        ):
            raise SourceCacheCorruptionError(f"case cache index does not match: {path}")
        sources = payload["sources"]
        if not isinstance(sources, dict):
            raise SourceCacheCorruptionError(f"invalid case sources index: {path}")
        sources[request.source_id] = {
            "cache_key": request.cache_key,
            "checksum": checksum,
        }
        self._write_json_atomic(path, payload)

    def inspect(self, case_id: str | None = None) -> CacheInspection:
        """Summarize verified objects, optionally with one case's source count."""
        if case_id is not None:
            case_id = _identifier(case_id, "case_id")
        object_paths = (
            tuple(path for path in self.objects_dir.glob("*/*") if path.is_file())
            if self.objects_dir.exists()
            else ()
        )
        case_paths = (
            tuple(self.cases_dir.glob("*.json")) if self.cases_dir.exists() else ()
        )
        source_count: int | None = None
        if case_id is not None:
            try:
                payload = self._read_json(self.cases_dir / f"{case_id}.json")
            except FileNotFoundError:
                source_count = 0
            else:
                if not isinstance(payload, dict) or not isinstance(
                    payload.get("sources"), dict
                ):
                    raise SourceCacheCorruptionError("invalid case cache index")
                source_count = len(payload["sources"])
        return CacheInspection(
            object_count=len(object_paths),
            object_bytes=sum(path.stat().st_size for path in object_paths),
            case_count=len(case_paths),
            case_id=case_id,
            source_count=source_count,
        )

    def clean_case(self, case_id: str) -> int:
        """Remove references for exactly one case and reclaim unreferenced blobs."""
        case_id = _identifier(case_id, "case_id")
        case_path = self.cases_dir / f"{case_id}.json"
        try:
            payload = self._read_json(case_path)
        except FileNotFoundError:
            return 0
        if not isinstance(payload, dict) or not isinstance(
            payload.get("sources"), dict
        ):
            raise SourceCacheCorruptionError(f"invalid case cache index: {case_path}")
        removed = len(payload["sources"])
        case_path.unlink()
        self._reclaim_unreferenced()
        return removed

    def _reclaim_unreferenced(self) -> None:
        referenced_keys: set[str] = set()
        if self.cases_dir.exists():
            for path in self.cases_dir.glob("*.json"):
                payload = self._read_json(path)
                if not isinstance(payload, dict) or not isinstance(
                    payload.get("sources"), dict
                ):
                    raise SourceCacheCorruptionError(
                        f"invalid case cache index: {path}"
                    )
                for source in payload["sources"].values():
                    if isinstance(source, dict) and isinstance(
                        source.get("cache_key"), str
                    ):
                        referenced_keys.add(source["cache_key"])
        if not self.sources_dir.exists():
            return
        for index_path in self.sources_dir.glob("*.json"):
            if index_path.stem in referenced_keys:
                continue
            payload = self._read_json(index_path)
            if not isinstance(payload, dict):
                raise SourceCacheCorruptionError(
                    f"invalid source cache index: {index_path}"
                )
            checksum = _checksum(payload.get("checksum"), "cached checksum")
            assert checksum is not None
            self.path_for_checksum(checksum).unlink(missing_ok=True)
            index_path.unlink()

    @staticmethod
    def _read_json(path: Path) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise SourceCacheCorruptionError(
                f"invalid JSON cache index: {path}"
            ) from error

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", delete=False, dir=path.parent
        ) as temporary:
            json.dump(payload, temporary, sort_keys=True, separators=(",", ":"))
            temporary.write("\n")
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, path)


__all__ = [
    "CacheInspection",
    "DownloadResponse",
    "DownloadReceipt",
    "SourceCacheCorruptionError",
    "SourceCacheError",
    "SourceChecksumError",
    "SourceRequest",
    "SourceSizeError",
    "VerifiedSourceCache",
]
