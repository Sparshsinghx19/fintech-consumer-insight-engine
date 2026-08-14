"""
Text chunking for the RAG pipeline.

Wraps LangChain's RecursiveCharacterTextSplitter, converting page-level
`PageRecord` objects into LangChain `Document` objects with metadata
(source file name, page number, document type, chunk index, content
hash) preserved on every chunk.
"""
from __future__ import annotations

from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Settings, settings as default_settings
from app.ingestion import PageRecord, filter_pages_with_content
from app.utils import get_logger, sha256_text

logger = get_logger(__name__)


class DocumentChunker:
    """Splits ingested pages into overlapping, appropriately-sized chunks."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or default_settings
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def chunk_pages(self, pages: List[PageRecord]) -> List[Document]:
        """Convert page records into LangChain Documents ready for
        embedding, one Document per chunk with full source metadata."""
        usable_pages = filter_pages_with_content(pages)
        if not usable_pages:
            logger.warning("No pages with extractable text were provided for chunking.")
            return []

        documents: List[Document] = []
        for page in usable_pages:
            chunks = self.splitter.split_text(page.text)
            for i, chunk_text in enumerate(chunks):
                if not chunk_text.strip():
                    continue
                metadata = {
                    "source": page.source,
                    "source_path": page.source_path,
                    "page": page.page,
                    "document_type": page.document_type,
                    "chunk_index": i,
                    "file_hash": page.file_hash,
                    "chunk_hash": sha256_text(f"{page.source}:{page.page}:{i}:{chunk_text}"),
                }
                documents.append(Document(page_content=chunk_text, metadata=metadata))

        logger.info("Produced %d chunks from %d pages.", len(documents), len(usable_pages))
        return documents
