"""Dataset manifests, anonymous GOES-18 discovery, and download utilities."""

from firesentinel.data.goes18 import (
    AnonymousS3Catalog,
    CatalogAccessError,
    CatalogCacheError,
    CatalogFormatError,
    Goes18ObjectDiscovery,
    Goes18ObjectReference,
    LocalCatalogCache,
    MissingFrame,
    MissingFrameReason,
)
from firesentinel.data.source_cache import (
    CacheInspection,
    DownloadReceipt,
    SourceCacheCorruptionError,
    SourceCacheError,
    SourceChecksumError,
    SourceRequest,
    SourceSizeError,
    VerifiedSourceCache,
)

__all__ = [
    "AnonymousS3Catalog",
    "CatalogAccessError",
    "CatalogCacheError",
    "CatalogFormatError",
    "Goes18ObjectDiscovery",
    "Goes18ObjectReference",
    "LocalCatalogCache",
    "MissingFrame",
    "MissingFrameReason",
    "CacheInspection",
    "DownloadReceipt",
    "SourceCacheCorruptionError",
    "SourceCacheError",
    "SourceChecksumError",
    "SourceRequest",
    "SourceSizeError",
    "VerifiedSourceCache",
]
