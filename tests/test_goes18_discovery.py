"""Offline contracts for anonymous, cache-backed GOES-18 discovery."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from firesentinel.core.records import Channel
from firesentinel.data.goes18 import (
    DEFAULT_MAXIMUM_OFFSET,
    GOES18_BUCKET,
    AnonymousS3Catalog,
    CatalogFormatError,
    Goes18ObjectDiscovery,
    Goes18ObjectReference,
    LocalCatalogCache,
    MissingFrame,
    MissingFrameReason,
    UnsupportedGoes18ChannelError,
    hour_prefixes,
    parse_scan_times,
    select_nearest_scan,
)

DISCOVERED_AT = datetime(2026, 8, 19, 12, 0, tzinfo=UTC)
HOUR_00 = "ABI-L2-CMIPF/2025/001/00/"
HOUR_23 = "ABI-L2-CMIPF/2024/366/23/"


def _object_key(channel: str, start: str, end: str) -> str:
    return (
        f"ABI-L2-CMIPF/{start[:4]}/{start[4:7]}/{start[7:9]}/"
        f"OR_ABI-L2-CMIPF-M6{channel}_G18_s{start}_e{end}_c20250010020000.nc"
    )


KEY_C07_0000 = _object_key("C07", "20250010000000", "20250010009500")
KEY_C07_0010 = _object_key("C07", "20250010010000", "20250010019500")
KEY_C14_0000 = _object_key("C14", "20250010000000", "20250010009500")
KEY_C02_0000 = _object_key("C02", "20250010000000", "20250010009500")


def _listing_xml(
    entries: tuple[tuple[str, int], ...], *, truncated: bool = False, token: str = ""
) -> bytes:
    contents = "".join(
        f"<Contents><Key>{key}</Key><Size>{size}</Size></Contents>"
        for key, size in entries
    )
    next_token = (
        f"<NextContinuationToken>{token}</NextContinuationToken>" if token else ""
    )
    return (
        '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
        f"{contents}<IsTruncated>{str(truncated).lower()}</IsTruncated>{next_token}"
        "</ListBucketResult>"
    ).encode()


class FakeCatalogHttp:
    """A deterministic public-S3 stand-in; it sees only fully anonymous URLs."""

    def __init__(self, pages: dict[tuple[str, str | None], bytes]) -> None:
        self.pages = pages
        self.calls: list[str] = []

    def __call__(self, url: str, timeout_seconds: float) -> bytes:
        assert timeout_seconds == 20.0
        assert "AWS" not in url
        self.calls.append(url)
        query = parse_qs(urlparse(url).query)
        assert query["list-type"] == ["2"]
        assert query["encoding-type"] == ["url"]
        prefix = query["prefix"][0]
        token = query.get("continuation-token", [None])[0]
        return self.pages[(prefix, token)]


def _discovery(tmp_path: Path, http: FakeCatalogHttp) -> Goes18ObjectDiscovery:
    return Goes18ObjectDiscovery(
        LocalCatalogCache(tmp_path),
        catalog=AnonymousS3Catalog(http_get=http),
        now=lambda: DISCOVERED_AT,
    )


def test_parse_scan_times_requires_the_frozen_product_and_supported_channels() -> None:
    scan_start, scan_end = parse_scan_times(KEY_C07_0000)

    assert scan_start == datetime(2025, 1, 1, 0, 0, tzinfo=UTC)
    assert scan_end == datetime(2025, 1, 1, 0, 9, 50, tzinfo=UTC)
    with pytest.raises(UnsupportedGoes18ChannelError):
        parse_scan_times(KEY_C02_0000)
    with pytest.raises(CatalogFormatError, match="path"):
        parse_scan_times(KEY_C07_0000.replace("/00/", "/01/"))


def test_known_event_queries_are_ordered_and_reuse_the_immutable_local_catalog(
    tmp_path: Path,
) -> None:
    http = FakeCatalogHttp(
        {
            (HOUR_23, None): _listing_xml(()),
            (
                HOUR_00,
                None,
            ): _listing_xml(
                ((KEY_C02_0000, 11), (KEY_C07_0000, 101), (KEY_C14_0000, 201)),
                truncated=True,
                token="next-page",
            ),
            (HOUR_00, "next-page"): _listing_xml(((KEY_C07_0010, 102),)),
        }
    )
    discovery = _discovery(tmp_path, http)
    requested = datetime(2025, 1, 1, 0, 5, tzinfo=UTC)

    results = (
        discovery.resolve(requested - timedelta(minutes=1), Channel.C07),
        discovery.resolve(requested + timedelta(minutes=1), Channel.C07),
        discovery.resolve(requested, Channel.C14),
    )

    assert all(isinstance(item, Goes18ObjectReference) for item in results)
    references = tuple(
        item for item in results if isinstance(item, Goes18ObjectReference)
    )
    assert [item.object_key for item in references] == [
        KEY_C07_0000,
        KEY_C07_0010,
        KEY_C14_0000,
    ]
    assert len(http.calls) == 3  # previous hour + both pages in requested hour
    first = results[0]
    assert isinstance(first, Goes18ObjectReference)
    assert first.to_manifest_record() == {
        "provider": "noaa-goes18",
        "product": "ABI-L2-CMIPF",
        "channel": "C07",
        "bucket": GOES18_BUCKET,
        "object_key": KEY_C07_0000,
        "size_bytes": 101,
        "scan_start": "2025-01-01T00:00:00.000000Z",
        "scan_end": "2025-01-01T00:09:50.000000Z",
        "discovered_at": "2026-08-19T12:00:00.000000Z",
    }


def test_nearest_scan_uses_start_time_and_selects_earlier_at_an_exact_midpoint() -> (
    None
):
    scan_start, scan_end = parse_scan_times(KEY_C07_0000)
    next_start, next_end = parse_scan_times(KEY_C07_0010)
    references = (
        Goes18ObjectReference(
            GOES18_BUCKET,
            KEY_C07_0000,
            101,
            Channel.C07,
            scan_start,
            scan_end,
            DISCOVERED_AT,
        ),
        Goes18ObjectReference(
            GOES18_BUCKET,
            KEY_C07_0010,
            102,
            Channel.C07,
            next_start,
            next_end,
            DISCOVERED_AT,
        ),
    )

    before_boundary = select_nearest_scan(
        references,
        datetime(2025, 1, 1, 0, 4, 59, tzinfo=UTC),
        Channel.C07,
    )
    at_boundary = select_nearest_scan(
        references,
        datetime(2025, 1, 1, 0, 5, tzinfo=UTC),
        Channel.C07,
    )
    after_boundary = select_nearest_scan(
        references,
        datetime(2025, 1, 1, 0, 5, 1, tzinfo=UTC),
        Channel.C07,
    )

    assert isinstance(before_boundary, Goes18ObjectReference)
    assert isinstance(at_boundary, Goes18ObjectReference)
    assert isinstance(after_boundary, Goes18ObjectReference)
    assert before_boundary.object_key == KEY_C07_0000
    assert at_boundary.object_key == KEY_C07_0000
    assert after_boundary.object_key == KEY_C07_0010


def test_missing_frame_is_typed_when_the_nearest_scan_exceeds_the_bound() -> None:
    scan_start, scan_end = parse_scan_times(KEY_C07_0000)
    result = select_nearest_scan(
        (
            Goes18ObjectReference(
                GOES18_BUCKET,
                KEY_C07_0000,
                101,
                Channel.C07,
                scan_start,
                scan_end,
                DISCOVERED_AT,
            ),
        ),
        datetime(2025, 1, 1, 0, 20, tzinfo=UTC),
        Channel.C07,
        maximum_offset=timedelta(minutes=5),
        searched_prefixes=(HOUR_00,),
    )

    assert isinstance(result, MissingFrame)
    assert result.reason is MissingFrameReason.OUTSIDE_MAXIMUM_OFFSET
    assert result.maximum_offset == timedelta(minutes=5)
    assert result.candidate_count == 1


def test_hour_prefixes_include_both_sides_of_an_hour_boundary() -> None:
    assert hour_prefixes(
        datetime(2025, 1, 1, 0, 0, tzinfo=UTC), DEFAULT_MAXIMUM_OFFSET
    ) == (HOUR_23, HOUR_00)
