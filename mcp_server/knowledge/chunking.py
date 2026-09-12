"""Chunking — split collected documents into embeddable chunks.

Heading-aware splitting for markdown bodies; token-budget windows with
overlap otherwise. Chunk IDs are deterministic so re-syncs overwrite in place.
"""

from __future__ import annotations

import re

from .collectors.base import SourceDocument, chunk_id


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
_WORD_RE = re.compile(r"\S+")


def _approx_tokens(text: str) -> int:
    """Cheap token estimate: whitespace-split words (~1.3 chars/word safety margin)."""
    return len(_WORD_RE.findall(text))


def _split_by_headings(text: str) -> list[tuple[str, str]]:
    """Split markdown text into (heading_path, section_text) pairs."""
    sections: list[tuple[str, str]] = []
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [("", text)]

    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(("", preamble))

    for i, m in enumerate(matches):
        heading = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((heading, f"{heading}\n{body}" if heading else body))
    return sections


def _window(text: str, target: int, overlap: int) -> list[str]:
    """Token-budget windowing with overlap."""
    tokens = _WORD_RE.findall(text)
    if len(tokens) <= target:
        return [text.strip()] if text.strip() else []
    windows: list[str] = []
    step = max(1, target - overlap)
    for i in range(0, len(tokens), step):
        window = tokens[i : i + target]
        windows.append(" ".join(window))
        if i + target >= len(tokens):
            break
    return windows


def chunk_document(doc: SourceDocument, target_tokens: int = 512, overlap_tokens: int = 64) -> list[dict]:
    """Chunk one source document into embeddable records.

    Returns records: {id, text, metadata}. Metadata carries provenance
    (source, doc_id, title, kind, version, checksum) used by the search
    service for citations and by the indexer for change detection.
    """
    chunks: list[dict] = []
    sections = _split_by_headings(doc.text)

    # Merge tiny adjacent sections, then window oversized ones.
    pieces: list[str] = []
    buf = ""
    for _heading, section in sections:
        if _approx_tokens(buf) + _approx_tokens(section) <= target_tokens:
            buf = f"{buf}\n{section}".strip()
        else:
            if buf:
                pieces.append(buf)
            buf = section
    if buf:
        pieces.append(buf)

    for piece in pieces:
        chunks.extend(_window(piece, target_tokens, overlap_tokens))

    records = []
    for n, text in enumerate(chunks):
        records.append({
            "id": chunk_id(doc.source, doc.doc_id, n),
            "text": text,
            "metadata": {
                "source": doc.source,
                "doc_id": doc.doc_id,
                "title": doc.title,
                "kind": doc.kind,
                "topic": doc.topic,
                "audience": doc.audience,
                "version": doc.version,
                "checksum": doc.checksum,
                "chunk": n,
                "chunks_total": len(chunks),
                **doc.metadata,
            },
        })
    return records
