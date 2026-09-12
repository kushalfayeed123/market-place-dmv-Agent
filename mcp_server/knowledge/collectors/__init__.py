"""Collectors package — one module per source of organization data."""

from .base import Collector, SourceDocument, chunk_id, scrub_untrusted
from .catalog import CatalogCollector
from .documents import DocumentsCollector

__all__ = [
    "Collector",
    "SourceDocument",
    "chunk_id",
    "scrub_untrusted",
    "CatalogCollector",
    "DocumentsCollector",
]
