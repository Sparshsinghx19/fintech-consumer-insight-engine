"""Shared test helpers: text-bearing PDF generation and a deterministic
local embedding function, so indexing tests can run against a real Chroma
collection without extra dependencies or model downloads."""
from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import List

from langchain_core.embeddings import Embeddings


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_text_pdf(path: Path, pages: List[str]) -> Path:
    """Write a minimal, valid PDF whose pages contain extractable text
    (one Helvetica text line per newline-separated line)."""
    page_ids = [4 + 2 * i for i in range(len(pages))]
    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: f"<< /Type /Pages /Kids [{' '.join(f'{p} 0 R' for p in page_ids)}] /Count {len(pages)} >>".encode(),
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    for page_id, text in zip(page_ids, pages):
        lines = " ".join(f"({_pdf_escape(line)}) Tj T*" for line in text.split("\n"))
        stream = f"BT /F1 12 Tf 14 TL 50 750 Td {lines} ET".encode("latin-1")
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {page_id + 1} 0 R >>"
        ).encode()
        objects[page_id + 1] = b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream"

    out = bytearray(b"%PDF-1.4\n")
    offsets = {}
    for obj_id in sorted(objects):
        offsets[obj_id] = len(out)
        out += f"{obj_id} 0 obj\n".encode() + objects[obj_id] + b"\nendobj\n"
    xref_at = len(out)
    size = len(objects) + 1
    out += f"xref\n0 {size}\n0000000000 65535 f \n".encode()
    out += "".join(f"{offsets[i]:010d} 00000 n \n" for i in sorted(objects)).encode()
    out += f"trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))
    return path


class HashingEmbeddings(Embeddings):
    """Deterministic bag-of-words embeddings (hashed into a small vector).
    Similar texts get similar vectors, which is enough for retrieval tests."""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def _embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        for word in re.findall(r"\w+", text.lower()):
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)
