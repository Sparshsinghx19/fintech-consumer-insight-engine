"""Shared utility helpers: logging setup, hashing, text cleaning."""
from __future__ import annotations

import hashlib
import logging
import re
import sys
from typing import Optional


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """Return a configured module-level logger.

    Avoids duplicate handlers when called multiple times (e.g. Streamlit
    reruns the script on every interaction).
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level or "INFO")
    logger.propagate = False
    return logger


def sha256_text(text: str) -> str:
    """Stable hash for a chunk of text (used for duplicate detection)."""
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def sha256_file(path: str) -> str:
    """Stable hash for a file's raw bytes (used to detect duplicate PDFs)."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


_WHITESPACE_RE = re.compile(r"[ \t\u00a0]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_HYPHEN_LINEBREAK_RE = re.compile(r"(\w)-\n(\w)")


def clean_text(text: str) -> str:
    """Normalize whitespace/line-break artifacts from PDF extraction while
    preserving the meaningful research content (sentence structure,
    numbers, punctuation)."""
    if not text:
        return ""
    # Rejoin words that were hyphen-broken across a line wrap.
    text = _HYPHEN_LINEBREAK_RE.sub(r"\1\2", text)
    # Collapse runs of spaces/tabs but keep newlines (paragraph structure).
    text = _WHITESPACE_RE.sub(" ", text)
    # Collapse excessive blank lines.
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def truncate_chars(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " ...[truncated]"
