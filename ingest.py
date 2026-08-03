"""
ingest.py
Document loading, cleaning, and chunking.

Supports PDF, TXT, Markdown, and DOCX. Produces a list of chunk dicts,
each carrying the text plus metadata (filename, page, doc type, chunk id)
that downstream retrieval uses for citations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Dict

import fitz  # PyMuPDF
from docx import Document as DocxDocument

from config import Settings
from utils import get_logger, file_content_hash, text_hash

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}


@dataclass
class RawPage:
    """A single unit of extracted text before chunking (a PDF page, or a whole file)."""

    text: str
    page: int | None  # None for non-paginated formats (txt/md/docx)
    filename: str
    doc_type: str
    file_hash: str


class DocumentLoadError(Exception):
    """Raised when a document cannot be parsed."""


def _clean_text(text: str) -> str:
    """Normalize whitespace and strip control characters from extracted text."""
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _load_pdf(path: Path) -> List[RawPage]:
    """Extract text page-by-page from a PDF using PyMuPDF."""
    pages: List[RawPage] = []
    try:
        file_hash = file_content_hash(path)
        with fitz.open(path) as doc:
            if doc.page_count == 0:
                raise DocumentLoadError(f"'{path.name}' has no pages.")
            for i, page in enumerate(doc, start=1):
                raw = page.get_text("text")
                cleaned = _clean_text(raw)
                if cleaned:
                    pages.append(RawPage(cleaned, i, path.name, "pdf", file_hash))
    except DocumentLoadError:
        raise
    except Exception as exc:  # PyMuPDF raises varied exceptions for corrupt PDFs
        raise DocumentLoadError(f"Could not parse PDF '{path.name}': {exc}") from exc

    if not pages:
        raise DocumentLoadError(f"'{path.name}' contains no extractable text (possibly scanned/image-only).")
    return pages


def _load_txt_or_md(path: Path, doc_type: str) -> List[RawPage]:
    """Load a plain text or markdown file as a single unit."""
    try:
        file_hash = file_content_hash(path)
        raw = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        raise DocumentLoadError(f"Could not read '{path.name}': {exc}") from exc

    cleaned = _clean_text(raw)
    if not cleaned:
        raise DocumentLoadError(f"'{path.name}' is empty.")
    return [RawPage(cleaned, None, path.name, doc_type, file_hash)]


def _load_docx(path: Path) -> List[RawPage]:
    """Extract text from a DOCX file's paragraphs."""
    try:
        file_hash = file_content_hash(path)
        doc = DocxDocument(str(path))
        raw = "\n".join(p.text for p in doc.paragraphs)
    except Exception as exc:
        raise DocumentLoadError(f"Could not parse DOCX '{path.name}': {exc}") from exc

    cleaned = _clean_text(raw)
    if not cleaned:
        raise DocumentLoadError(f"'{path.name}' contains no extractable text.")
    return [RawPage(cleaned, None, path.name, "docx", file_hash)]


def load_document(path: Path) -> List[RawPage]:
    """Dispatch to the correct loader based on file extension."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _load_pdf(path)
    if ext == ".txt":
        return _load_txt_or_md(path, "txt")
    if ext == ".md":
        return _load_txt_or_md(path, "markdown")
    if ext == ".docx":
        return _load_docx(path)
    raise DocumentLoadError(f"Unsupported file type: '{path.name}' ({ext})")


def discover_documents(data_dir: Path) -> List[Path]:
    """Recursively find all supported documents under data_dir."""
    return sorted(
        p for p in Path(data_dir).rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split text into overlapping chunks by character count, breaking on
    sentence/paragraph boundaries where possible so chunks stay coherent.

    A simple sliding window over characters (rather than tokens) keeps this
    dependency-free and fast; chunk_size/overlap are tuned for typical
    sentence-transformer context windows.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # Try to end on a paragraph or sentence boundary near `end`.
        if end < text_len:
            boundary = text.rfind("\n\n", start, end)
            if boundary == -1 or boundary <= start:
                boundary = text.rfind(". ", start, end)
            if boundary != -1 and boundary > start:
                end = boundary + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_len:
            break
        start = max(end - overlap, start + 1)  # guarantee forward progress

    return chunks


def build_chunks(pages: List[RawPage], settings: Settings) -> List[Dict]:
    """
    Turn a list of RawPage objects into chunk dicts with full metadata,
    ready for embedding and storage in Chroma.
    """
    results: List[Dict] = []
    for page in pages:
        pieces = chunk_text(page.text, settings.chunk_size, settings.chunk_overlap)
        for idx, piece in enumerate(pieces):
            chunk_id = f"{page.file_hash}_{page.page or 0}_{idx}_{text_hash(piece)}"
            results.append(
                {
                    "id": chunk_id,
                    "text": piece,
                    "metadata": {
                        "filename": page.filename,
                        "page": page.page if page.page is not None else "N/A",
                        "doc_type": page.doc_type,
                        "chunk_index": idx,
                        "file_hash": page.file_hash,
                    },
                }
            )
    return results


def process_directory(data_dir: Path, settings: Settings) -> Iterator[Dict]:
    """
    Generator yielding one result dict per file processed:
    {"filename": ..., "status": "ok"/"error", "chunks": [...], "error": str|None}

    Errors in one file do not stop processing of the rest.
    """
    logger = get_logger("ingest", settings.logs_dir, settings.log_level)
    files = discover_documents(data_dir)
    logger.info(f"Discovered {len(files)} candidate document(s) under {data_dir}")

    for path in files:
        try:
            pages = load_document(path)
            chunks = build_chunks(pages, settings)
            logger.info(f"Processed '{path.name}': {len(chunks)} chunk(s)")
            yield {"filename": path.name, "status": "ok", "chunks": chunks, "error": None}
        except DocumentLoadError as exc:
            logger.warning(f"Skipped '{path.name}': {exc}")
            yield {"filename": path.name, "status": "error", "chunks": [], "error": str(exc)}
        except Exception as exc:  # unexpected failure — log full detail, keep going
            logger.exception(f"Unexpected error processing '{path.name}'")
            yield {"filename": path.name, "status": "error", "chunks": [], "error": f"Unexpected error: {exc}"}
