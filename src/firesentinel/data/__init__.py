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
]
