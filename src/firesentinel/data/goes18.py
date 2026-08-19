"""Anonymous, cache-backed discovery of GOES-18 ABI full-disk objects.

The NOAA GOES public buckets expose an S3-compatible ``ListObjectsV2`` API.
This module deliberately calls that public HTTP endpoint directly rather than
using an AWS SDK, so discovery neither reads nor needs AWS credentials.

Only the product and bands frozen in the FireSentinel product contract are
accepted: GOES-18 ``ABI-L2-CMIPF`` (full disk) Channels 7 and 14.  A selected
object is represented by its bucket, key, size, scan start/end, and the time
its catalog listing was discovered.  File bytes and content hashes are a later
download-stage responsibility.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import Final
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlencode
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from firesentinel.core.records import Channel

GOES18_BUCKET: Final = "noaa-goes18"
GOES18_PRODUCT: Final = "ABI-L2-CMIPF"
GOES18_PROVIDER: Final = "noaa-goes18"
GOES18_S3_ENDPOINT: Final = f"https://{GOES18_BUCKET}.s3.amazonaws.com"
SUPPORTED_CHANNELS: Final = frozenset((Channel.C07, Channel.C14))
DEFAULT_MAXIMUM_OFFSET: Final = timedelta(minutes=15)
_CACHE_SCHEMA_VERSION: Final = 1
_GOES18_KEY = re.compile(
    r"^ABI-L2-CMIPF/"
    r"(?P<year>\d{4})/(?P<day_of_year>\d{3})/(?P<hour>\d{2})/"
    r"OR_ABI-L2-CMIPF-M\dC(?P<channel>\d{2})_G18_"
    r"s(?P<scan_start>\d{14})_e(?P<scan_end>\d{14})_c\d{14}\.nc$"
)


class Goes18DiscoveryError(RuntimeError):
    """Base class for invalid GOES catalog input or catalog access failures."""


class UnsupportedGoes18ProductError(Goes18DiscoveryError):
    """Raised when a caller asks for a product outside the frozen scope."""


class UnsupportedGoes18ChannelError(Goes18DiscoveryError):
    """Raised when a caller asks for a channel outside the frozen scope."""


class CatalogAccessError(Goes18DiscoveryError):
    """Raised when the public S3 catalog could not be read."""


class CatalogFormatError(Goes18DiscoveryError):
    """Raised when a catalog response, cache, or GOES key is malformed."""


class CatalogCacheError(Goes18DiscoveryError):
    """Raised when an existing local catalog cache cannot be trusted."""


class MissingFrameReason(StrEnum):
    """Closed reasons for a successful discovery request with no usable frame."""

    NO_MATCHING_OBJECTS = "no_matching_objects"
    OUTSIDE_MAXIMUM_OFFSET = "outside_maximum_offset"


def _utc_timestamp(value: datetime, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware UTC")
    return value.astimezone(UTC)


def _timestamp_json(value: datetime) -> str:
    return (
        value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
    )


def _channel(value: Channel | str) -> Channel:
    try:
        channel = Channel(value)
    except ValueError as error:
        raise UnsupportedGoes18ChannelError(
            f"GOES-18 discovery supports only C07 and C14, not {value!r}"
        ) from error
    if channel not in SUPPORTED_CHANNELS:
        raise UnsupportedGoes18ChannelError(
            f"GOES-18 discovery supports only C07 and C14, not {channel.value}"
        )
    return channel


def _maximum_offset(value: timedelta) -> timedelta:
    if not isinstance(value, timedelta):
        raise TypeError("maximum_offset must be a timedelta")
    if value < timedelta(0):
        raise ValueError("maximum_offset must not be negative")
    return value


def _parse_goes_timestamp(value: str, field_name: str) -> datetime:
    """Parse a GOES ``YYYYJJJHHMMSSt`` timestamp into a UTC datetime."""
    if not isinstance(value, str) or not re.fullmatch(r"\d{14}", value):
        raise CatalogFormatError(f"{field_name} must be a 14-digit GOES timestamp")
    year = int(value[:4])
    day_of_year = int(value[4:7])
    hour = int(value[7:9])
    minute = int(value[9:11])
    second = int(value[11:13])
    tenth = int(value[13])
    if not 1 <= day_of_year <= 366 or hour > 23 or minute > 59 or second > 59:
        raise CatalogFormatError(f"{field_name} is not a valid GOES timestamp")
    try:
        timestamp = datetime(year, 1, 1, tzinfo=UTC) + timedelta(
            days=day_of_year - 1,
            hours=hour,
            minutes=minute,
            seconds=second,
            microseconds=tenth * 100_000,
        )
    except ValueError as error:
        raise CatalogFormatError(
            f"{field_name} is not a valid GOES timestamp"
        ) from error
    if timestamp.year != year:
        raise CatalogFormatError(f"{field_name} is not a valid GOES timestamp")
    return timestamp


def parse_scan_times(object_key: str) -> tuple[datetime, datetime]:
    """Return UTC scan start and end timestamps embedded in a GOES object key.

    The function is intentionally useful on its own for validating catalog
    fixtures and cached keys.  It validates the full selected-product key, not
    merely two timestamp-looking fragments in an arbitrary filename.
    """
    match = _GOES18_KEY.fullmatch(object_key)
    if match is None:
        raise CatalogFormatError(
            "object_key must be a GOES-18 ABI-L2-CMIPF NetCDF object key"
        )
    try:
        channel = Channel(f"C{match['channel']}")
    except ValueError as error:
        raise UnsupportedGoes18ChannelError(
            "object_key must use GOES-18 Channel C07 or C14"
        ) from error
    if channel not in SUPPORTED_CHANNELS:
        raise UnsupportedGoes18ChannelError(
            "object_key must use GOES-18 Channel C07 or C14"
        )
    scan_start = _parse_goes_timestamp(match["scan_start"], "scan_start")
    scan_end = _parse_goes_timestamp(match["scan_end"], "scan_end")
    if scan_end < scan_start:
        raise CatalogFormatError("scan_end must not precede scan_start")
    path_start = (int(match["year"]), int(match["day_of_year"]), int(match["hour"]))
    if path_start != (scan_start.year, int(scan_start.strftime("%j")), scan_start.hour):
        raise CatalogFormatError(
            "object_key path must agree with its embedded scan_start timestamp"
        )
    return scan_start, scan_end


@dataclass(frozen=True, slots=True)
class CatalogObject:
    """A raw S3 listing entry retained in a local catalog snapshot."""

    key: str
    size_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key:
            raise CatalogFormatError("catalog key must be a non-empty string")
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int):
            raise CatalogFormatError("catalog object size must be an integer")
        if self.size_bytes < 0:
            raise CatalogFormatError("catalog object size must not be negative")


@dataclass(frozen=True, slots=True)
class CatalogSnapshot:
    """One immutable, cached listing of a product/hour S3 prefix."""

    bucket: str
    prefix: str
    discovered_at: datetime
    objects: tuple[CatalogObject, ...]

    def __post_init__(self) -> None:
        if self.bucket != GOES18_BUCKET:
            raise CatalogFormatError(f"bucket must be {GOES18_BUCKET!r}")
        if not isinstance(self.prefix, str) or not self.prefix:
            raise CatalogFormatError("catalog prefix must be a non-empty string")
        object.__setattr__(
            self, "discovered_at", _utc_timestamp(self.discovered_at, "discovered_at")
        )
        objects = tuple(self.objects)
        if not all(isinstance(item, CatalogObject) for item in objects):
            raise CatalogFormatError(
                "catalog objects must contain CatalogObject values"
            )
        if tuple(sorted(item.key for item in objects)) != tuple(
            item.key for item in objects
        ):
            raise CatalogFormatError("catalog objects must be sorted by key")
        if len({item.key for item in objects}) != len(objects):
            raise CatalogFormatError("catalog objects must not contain duplicate keys")
        object.__setattr__(self, "objects", objects)


@dataclass(frozen=True, slots=True)
class Goes18ObjectReference:
    """A stable, downloadable NOAA object reference selected from the catalog."""

    bucket: str
    object_key: str
    size_bytes: int
    channel: Channel
    scan_start: datetime
    scan_end: datetime
    discovered_at: datetime

    def __post_init__(self) -> None:
        if self.bucket != GOES18_BUCKET:
            raise CatalogFormatError(f"bucket must be {GOES18_BUCKET!r}")
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int):
            raise CatalogFormatError("size_bytes must be an integer")
        if self.size_bytes < 0:
            raise CatalogFormatError("size_bytes must not be negative")
        expected_channel = _channel(self.channel)
        scan_start, scan_end = parse_scan_times(self.object_key)
        key_match = _GOES18_KEY.fullmatch(self.object_key)
        if key_match is None:  # parse_scan_times above makes this defensive only.
            raise CatalogFormatError("object_key must be a selected GOES-18 object")
        key_channel = Channel(f"C{key_match['channel']}")
        if key_channel != expected_channel:
            raise CatalogFormatError("channel must agree with the object key")
        supplied_start = _utc_timestamp(self.scan_start, "scan_start")
        supplied_end = _utc_timestamp(self.scan_end, "scan_end")
        if (supplied_start, supplied_end) != (scan_start, scan_end):
            raise CatalogFormatError("scan timestamps must agree with the object key")
        discovered_at = _utc_timestamp(self.discovered_at, "discovered_at")
        if discovered_at < scan_end:
            raise CatalogFormatError("discovered_at must not be before scan_end")
        object.__setattr__(self, "channel", expected_channel)
        object.__setattr__(self, "scan_start", scan_start)
        object.__setattr__(self, "scan_end", scan_end)
        object.__setattr__(self, "discovered_at", discovered_at)

    @property
    def scan_time(self) -> datetime:
        """Compatibility name for the scan start used by earlier source records."""
        return self.scan_start

    @property
    def key(self) -> str:
        """S3-compatible shorthand for ``object_key``."""
        return self.object_key

    def to_manifest_record(self) -> dict[str, object]:
        """Return JSON-safe source provenance for a pinned case manifest."""
        return {
            "provider": GOES18_PROVIDER,
            "product": GOES18_PRODUCT,
            "channel": self.channel.value,
            "bucket": self.bucket,
            "object_key": self.object_key,
            "size_bytes": self.size_bytes,
            "scan_start": _timestamp_json(self.scan_start),
            "scan_end": _timestamp_json(self.scan_end),
            "discovered_at": _timestamp_json(self.discovered_at),
        }


@dataclass(frozen=True, slots=True)
class MissingFrame:
    """A typed, non-exceptional result when no nearby requested frame exists."""

    requested_time: datetime
    channel: Channel
    maximum_offset: timedelta
    searched_prefixes: tuple[str, ...]
    reason: MissingFrameReason
    candidate_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "requested_time",
            _utc_timestamp(self.requested_time, "requested_time"),
        )
        object.__setattr__(self, "channel", _channel(self.channel))
        object.__setattr__(self, "maximum_offset", _maximum_offset(self.maximum_offset))
        prefixes = tuple(self.searched_prefixes)
        if not all(isinstance(item, str) and item for item in prefixes):
            raise CatalogFormatError("searched_prefixes must contain non-empty strings")
        object.__setattr__(self, "searched_prefixes", prefixes)
        try:
            reason = MissingFrameReason(self.reason)
        except ValueError as error:
            raise CatalogFormatError("reason must be a MissingFrameReason") from error
        object.__setattr__(self, "reason", reason)
        if isinstance(self.candidate_count, bool) or not isinstance(
            self.candidate_count, int
        ):
            raise CatalogFormatError("candidate_count must be an integer")
        if self.candidate_count < 0:
            raise CatalogFormatError("candidate_count must not be negative")


DiscoveryResult = Goes18ObjectReference | MissingFrame


def _catalog_object_from_json(value: object) -> CatalogObject:
    if not isinstance(value, dict) or set(value) != {"key", "size_bytes"}:
        raise CatalogCacheError("catalog cache object entries have an invalid shape")
    return CatalogObject(key=value["key"], size_bytes=value["size_bytes"])


class LocalCatalogCache:
    """Atomic local storage for immutable NOAA product/hour catalog listings."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    def path_for(self, bucket: str, prefix: str) -> Path:
        """Return a deterministic cache path without creating it."""
        if bucket != GOES18_BUCKET:
            raise CatalogCacheError(f"bucket must be {GOES18_BUCKET!r}")
        if not isinstance(prefix, str) or not prefix:
            raise CatalogCacheError("prefix must be a non-empty string")
        digest = sha256(f"{bucket}\0{prefix}".encode()).hexdigest()
        return self.directory / bucket / f"{digest}.json"

    def load(self, bucket: str, prefix: str) -> CatalogSnapshot | None:
        """Load an exact cached listing, or return ``None`` if none exists."""
        path = self.path_for(bucket, prefix)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError) as error:
            raise CatalogCacheError(f"cannot read catalog cache {path}") from error
        if not isinstance(payload, dict) or set(payload) != {
            "bucket",
            "discovered_at",
            "objects",
            "prefix",
            "schema_version",
        }:
            raise CatalogCacheError(f"catalog cache {path} has an invalid shape")
        if payload["schema_version"] != _CACHE_SCHEMA_VERSION:
            raise CatalogCacheError(f"catalog cache {path} has an unsupported schema")
        if payload["bucket"] != bucket or payload["prefix"] != prefix:
            raise CatalogCacheError(f"catalog cache {path} does not match its request")
        if not isinstance(payload["discovered_at"], str):
            raise CatalogCacheError(
                f"catalog cache {path} has an invalid discovery time"
            )
        try:
            discovered_at = datetime.fromisoformat(
                payload["discovered_at"].replace("Z", "+00:00")
            )
            objects_value = payload["objects"]
            if not isinstance(objects_value, list):
                raise CatalogCacheError("catalog objects must be a list")
            return CatalogSnapshot(
                bucket=bucket,
                prefix=prefix,
                discovered_at=discovered_at,
                objects=tuple(
                    _catalog_object_from_json(item) for item in objects_value
                ),
            )
        except (TypeError, ValueError, CatalogFormatError, CatalogCacheError) as error:
            raise CatalogCacheError(
                f"catalog cache {path} has invalid contents"
            ) from error

    def store(self, snapshot: CatalogSnapshot) -> Path:
        """Atomically persist one catalog snapshot and return its cache path."""
        if not isinstance(snapshot, CatalogSnapshot):
            raise TypeError("snapshot must be a CatalogSnapshot")
        path = self.path_for(snapshot.bucket, snapshot.prefix)
        payload = {
            "schema_version": _CACHE_SCHEMA_VERSION,
            "bucket": snapshot.bucket,
            "prefix": snapshot.prefix,
            "discovered_at": _timestamp_json(snapshot.discovered_at),
            "objects": [
                {"key": item.key, "size_bytes": item.size_bytes}
                for item in snapshot.objects
            ],
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", delete=False, dir=path.parent
            ) as temporary:
                temporary_path = Path(temporary.name)
                json.dump(payload, temporary, sort_keys=True, separators=(",", ":"))
                temporary.write("\n")
            os.replace(temporary_path, path)
        except OSError as error:
            temporary_path.unlink(
                missing_ok=True
            ) if "temporary_path" in locals() else None
            raise CatalogCacheError(f"cannot write catalog cache {path}") from error
        return path


HttpGet = Callable[[str, float], bytes]


def _anonymous_http_get(url: str, timeout_seconds: float) -> bytes:
    request = Request(url, headers={"User-Agent": "FireSentinel/0.0 catalog"})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
            return bytes(response.read())
    except (HTTPError, URLError, OSError) as error:
        raise CatalogAccessError(
            f"anonymous NOAA catalog request failed: {url}"
        ) from error


class AnonymousS3Catalog:
    """Read public S3 listings with anonymous HTTP requests only."""

    def __init__(
        self,
        *,
        endpoint: str = GOES18_S3_ENDPOINT,
        timeout_seconds: float = 20.0,
        http_get: HttpGet | None = None,
    ) -> None:
        if not isinstance(endpoint, str) or not endpoint.startswith("https://"):
            raise ValueError("endpoint must be an HTTPS URL")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        self._endpoint = endpoint.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._http_get = _anonymous_http_get if http_get is None else http_get

    def list_objects(self, bucket: str, prefix: str) -> tuple[CatalogObject, ...]:
        """List all S3 objects under ``prefix``, following continuation pages."""
        if bucket != GOES18_BUCKET:
            raise CatalogAccessError(f"bucket must be {GOES18_BUCKET!r}")
        if not isinstance(prefix, str) or not prefix:
            raise CatalogAccessError("prefix must be a non-empty string")

        continuation_token: str | None = None
        seen_tokens: set[str] = set()
        objects: list[CatalogObject] = []
        while True:
            parameters: dict[str, str] = {
                "list-type": "2",
                "prefix": prefix,
                "encoding-type": "url",
            }
            if continuation_token is not None:
                parameters["continuation-token"] = continuation_token
            url = f"{self._endpoint}?{urlencode(parameters)}"
            try:
                body = self._http_get(url, self._timeout_seconds)
                page_objects, is_truncated, next_token = self._parse_page(body)
            except CatalogAccessError:
                raise
            except (OSError, ValueError, ElementTree.ParseError) as error:
                raise CatalogAccessError(
                    f"invalid anonymous NOAA catalog response: {url}"
                ) from error
            objects.extend(page_objects)
            if not is_truncated:
                break
            if next_token is None or next_token in seen_tokens:
                raise CatalogAccessError(
                    "NOAA catalog pagination did not provide a new token"
                )
            seen_tokens.add(next_token)
            continuation_token = next_token

        deduplicated = {item.key: item for item in objects}
        return tuple(deduplicated[key] for key in sorted(deduplicated))

    @staticmethod
    def _parse_page(body: bytes) -> tuple[tuple[CatalogObject, ...], bool, str | None]:
        if not isinstance(body, bytes):
            raise ValueError("catalog response must contain bytes")
        root = ElementTree.fromstring(body)
        objects: list[CatalogObject] = []
        for contents in root.findall("{*}Contents"):
            key = contents.findtext("{*}Key")
            size = contents.findtext("{*}Size")
            if key is None or size is None:
                raise ValueError("catalog Contents needs Key and Size")
            objects.append(CatalogObject(key=unquote(key), size_bytes=int(size)))
        is_truncated_text = root.findtext("{*}IsTruncated")
        if is_truncated_text not in {"true", "false"}:
            raise ValueError("catalog response needs true or false IsTruncated")
        next_token = root.findtext("{*}NextContinuationToken")
        return tuple(objects), is_truncated_text == "true", next_token


def hour_prefixes(
    requested_time: datetime,
    maximum_offset: timedelta = DEFAULT_MAXIMUM_OFFSET,
) -> tuple[str, ...]:
    """Return all hourly product prefixes that could contain an eligible scan."""
    requested = _utc_timestamp(requested_time, "requested_time")
    offset = _maximum_offset(maximum_offset)
    first = (requested - offset).replace(minute=0, second=0, microsecond=0)
    last = (requested + offset).replace(minute=0, second=0, microsecond=0)
    prefixes: list[str] = []
    current = first
    while current <= last:
        prefixes.append(
            f"{GOES18_PRODUCT}/{current.year}/{current.strftime('%j')}/{current:%H}/"
        )
        current += timedelta(hours=1)
    return tuple(prefixes)


def select_nearest_scan(
    objects: Iterable[Goes18ObjectReference],
    requested_time: datetime,
    channel: Channel | str,
    *,
    maximum_offset: timedelta = DEFAULT_MAXIMUM_OFFSET,
    searched_prefixes: Iterable[str] = (),
) -> DiscoveryResult:
    """Select the nearest scan start; an exact tie deterministically picks earlier.

    GOES object keys identify scan *start* times.  Selection therefore measures
    distance to ``scan_start``.  Ordering by start time and key supplies a
    stable earlier-scan tie break at the midpoint between observations.
    """
    requested = _utc_timestamp(requested_time, "requested_time")
    requested_channel = _channel(channel)
    offset = _maximum_offset(maximum_offset)
    prefixes = tuple(searched_prefixes)
    candidates = tuple(
        item
        for item in objects
        if isinstance(item, Goes18ObjectReference) and item.channel == requested_channel
    )
    if not candidates:
        return MissingFrame(
            requested_time=requested,
            channel=requested_channel,
            maximum_offset=offset,
            searched_prefixes=prefixes,
            reason=MissingFrameReason.NO_MATCHING_OBJECTS,
            candidate_count=0,
        )
    selected = min(
        candidates,
        key=lambda item: (
            abs(item.scan_start - requested),
            item.scan_start,
            item.object_key,
        ),
    )
    if abs(selected.scan_start - requested) > offset:
        return MissingFrame(
            requested_time=requested,
            channel=requested_channel,
            maximum_offset=offset,
            searched_prefixes=prefixes,
            reason=MissingFrameReason.OUTSIDE_MAXIMUM_OFFSET,
            candidate_count=len(candidates),
        )
    return selected


class Goes18ObjectDiscovery:
    """Resolve a requested time and supported band to a stable NOAA reference."""

    def __init__(
        self,
        cache: LocalCatalogCache,
        *,
        catalog: AnonymousS3Catalog | None = None,
        now: Callable[[], datetime] | None = None,
        maximum_offset: timedelta = DEFAULT_MAXIMUM_OFFSET,
        product: str = GOES18_PRODUCT,
    ) -> None:
        if product != GOES18_PRODUCT:
            raise UnsupportedGoes18ProductError(
                f"GOES-18 discovery supports only {GOES18_PRODUCT!r}"
            )
        if not isinstance(cache, LocalCatalogCache):
            raise TypeError("cache must be a LocalCatalogCache")
        self._cache = cache
        self._catalog = AnonymousS3Catalog() if catalog is None else catalog
        self._now: Callable[[], datetime] = (
            (lambda: datetime.now(UTC)) if now is None else now
        )
        self._maximum_offset = _maximum_offset(maximum_offset)

    def resolve(
        self,
        requested_time: datetime,
        channel: Channel | str,
        *,
        maximum_offset: timedelta | None = None,
    ) -> DiscoveryResult:
        """Resolve one requested instant and band using cache before public HTTP."""
        requested = _utc_timestamp(requested_time, "requested_time")
        requested_channel = _channel(channel)
        offset = (
            self._maximum_offset
            if maximum_offset is None
            else _maximum_offset(maximum_offset)
        )
        prefixes = hour_prefixes(requested, offset)
        references: list[Goes18ObjectReference] = []
        for prefix in prefixes:
            snapshot = self._cache.load(GOES18_BUCKET, prefix)
            if snapshot is None:
                listed_objects = self._catalog.list_objects(GOES18_BUCKET, prefix)
                discovered_at = _utc_timestamp(self._now(), "now")
                snapshot = CatalogSnapshot(
                    bucket=GOES18_BUCKET,
                    prefix=prefix,
                    discovered_at=discovered_at,
                    objects=listed_objects,
                )
                self._cache.store(snapshot)
            references.extend(
                self._references_for_snapshot(snapshot, requested_channel)
            )
        return select_nearest_scan(
            references,
            requested,
            requested_channel,
            maximum_offset=offset,
            searched_prefixes=prefixes,
        )

    discover = resolve

    @staticmethod
    def _references_for_snapshot(
        snapshot: CatalogSnapshot, channel: Channel
    ) -> tuple[Goes18ObjectReference, ...]:
        references: list[Goes18ObjectReference] = []
        for item in snapshot.objects:
            match = _GOES18_KEY.fullmatch(item.key)
            if match is None:
                raise CatalogFormatError(
                    "catalog object does not match the selected GOES-18 product: "
                    f"{item.key}"
                )
            if f"C{match['channel']}" != channel.value:
                continue
            scan_start, scan_end = parse_scan_times(item.key)
            references.append(
                Goes18ObjectReference(
                    bucket=snapshot.bucket,
                    object_key=item.key,
                    size_bytes=item.size_bytes,
                    channel=channel,
                    scan_start=scan_start,
                    scan_end=scan_end,
                    discovered_at=snapshot.discovered_at,
                )
            )
        return tuple(references)


__all__ = [
    "AnonymousS3Catalog",
    "CatalogAccessError",
    "CatalogCacheError",
    "CatalogFormatError",
    "CatalogObject",
    "CatalogSnapshot",
    "DEFAULT_MAXIMUM_OFFSET",
    "DiscoveryResult",
    "GOES18_BUCKET",
    "GOES18_PRODUCT",
    "GOES18_PROVIDER",
    "Goes18DiscoveryError",
    "Goes18ObjectDiscovery",
    "Goes18ObjectReference",
    "LocalCatalogCache",
    "MissingFrame",
    "MissingFrameReason",
    "SUPPORTED_CHANNELS",
    "UnsupportedGoes18ChannelError",
    "UnsupportedGoes18ProductError",
    "hour_prefixes",
    "parse_scan_times",
    "select_nearest_scan",
]
