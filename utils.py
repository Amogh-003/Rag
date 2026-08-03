"""
utils.py
Small shared utilities: logging setup and content hashing used for
duplicate detection during incremental indexing.
"""

from __future__ import annotations

import hashlib
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def get_logger(name: str, logs_dir: Path, level: str = "INFO") -> logging.Logger:
    """
    Return a configured logger that writes to both stdout and a rotating
    log file under `logs_dir`. Safe to call multiple times with the same
    name (handlers are not duplicated).
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if logger.handlers:
        return logger  # already configured

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    Path(logs_dir).mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(
        Path(logs_dir) / "app.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def file_content_hash(path: Path) -> str:
    """
    Compute a stable SHA-256 hash of a file's bytes. Used to detect whether
    a document has changed since it was last indexed (duplicate detection /
    incremental indexing), independent of filename or mtime.
    """
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def text_hash(text: str) -> str:
    """Hash a text chunk's content — used as a deterministic chunk id component."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
