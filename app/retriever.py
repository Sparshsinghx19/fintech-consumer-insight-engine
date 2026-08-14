"""
Semantic search layer over the persisted ChromaDB vector store.

Returns retrieved passages together with source document, page number,
and a relevance/similarity score so the UI and downstream RAG chain can
cite evidence precisely.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from app.config import Settings, settings as default_settings
from app.utils import get_logger
from app.vector_store import VectorStoreManager

logger = get_logger(__name__)


@dataclass
class RetrievedPassage:
    text: str
    source: str
    page: int
    document_type: str
    relevance_score: float  # higher = more relevant, normalized to [0, 1]
    chunk_hash: str = ""


class SemanticRetriever:
    """Retrieves the top-k most relevant chunks for a natural-language query."""

    def __init__(self, vector_store_manager: Optional[VectorStoreManager] = None, settings: Optional[Settings] = None):
        self.settings = settings or default_settings
        self.vsm = vector_store_manager or VectorStoreManager(self.settings)

    def search(self, query: str, k: Optional[int] = None) -> List[RetrievedPassage]:
        if not query or not query.strip():
            return []
        k = k or self.settings.retrieval_k
        store = self.vsm.get_store()

        try:
            # Chroma returns (Document, distance) pairs; smaller distance = more similar.
            results = store.similarity_search_with_score(query, k=k)
        except Exception as exc:  # noqa: BLE001
            logger.error("Semantic search failed: %s", exc)
            return []

        passages: List[RetrievedPassage] = []
        for doc, distance in results:
            # Cosine distance in [0, 2] roughly -> convert to an intuitive
            # similarity score in [0, 1]. This is an approximation intended
            # for display/ranking purposes, not a calibrated probability.
            similarity = max(0.0, 1.0 - (distance / 2.0))
            passages.append(
                RetrievedPassage(
                    text=doc.page_content,
                    source=doc.metadata.get("source", "unknown"),
                    page=doc.metadata.get("page", -1),
                    document_type=doc.metadata.get("document_type", "pdf"),
                    relevance_score=round(similarity, 4),
                    chunk_hash=doc.metadata.get("chunk_hash", ""),
                )
            )
        return passages
