"""Collector protocol + shared document types.

A Collector turns one *source* of organization data into ``SourceDocument``
objects. Sources are registered in ``sync.py``; each has a stable ``source``
name used in chunk IDs so a sync can delete/rebuild exactly that source.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import AsyncIterator, Protocol


# Strip instruction-like patterns from untrusted text before it enters the index.
_INJECTION_PATTERNS = [
    re.compile(r"ignore (all|any|the)? ?(previous|prior|above) instructions?", re.IGNORECASE),
    re.compile(r"disregard (all|any|the)? ?(previous|prior|above)? ?(instructions|rules|prompts)?", re.IGNORECASE),
    re.compile(r"you are now (a|an|the) ", re.IGNORECASE),
    re.compile(r"system prompt", re.IGNORECASE),
    re.compile(r"</?(system|assistant|tool)_?>", re.IGNORECASE),
]


def scrub_untrusted(text: str) -> str:
    """Remove prompt-injection patterns from untrusted text (defensive, not a guarantee)."""
    out = text
    for pat in _INJECTION_PATTERNS:
        out = pat.sub("[filtered]", out)
    return out


@dataclass
class SourceDocument:
    """One logical document collected from a source, ready to be chunked."""

    source: str                       # collector name, e.g. "documents", "catalog"
    doc_id: str                       # stable id within the source
    text: str                         # full text content
    title: str = ""                   # human-readable title (used for citations)
    kind: str = "document"            # policy | faq | guide | product | category
    topic: str = ""                   # free-form topic tag, e.g. "returns"
    audience: str = "all"             # buyer | merchant | all
    version: int = 1                  # source document version (for citations)
    metadata: dict = field(default_factory=dict)

    @property
    def checksum(self) -> str:
        """Content checksum — unchanged sources are skipped on incremental sync."""
        payload = json.dumps(
            {"text": self.text, "title": self.title, "metadata": self.metadata},
            sort_keys=True, default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


class Collector(Protocol):
    """Port for one source of organization data."""

    @property
    def name(self) -> str: ...

    def collect(self) -> AsyncIterator[SourceDocument]: ...


def chunk_id(source: str, doc_id: str, chunk_n: int) -> str:
    """Deterministic chunk id: {source}:{doc_id}#chunk{nnnn}."""
    return f"{source}:{doc_id}#chunk{chunk_n:04d}"
