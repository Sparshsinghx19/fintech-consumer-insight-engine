"""
Local embedding model factory.

Uses HuggingFace / Sentence-Transformers models entirely locally via
`langchain_huggingface.HuggingFaceEmbeddings`. No cloud API is used or
required for embeddings.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings

from app.config import Settings, settings as default_settings
from app.utils import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=4)
def _build_embeddings(model_name: str, device: str, batch_size: int) -> HuggingFaceEmbeddings:
    logger.info("Loading local embedding model '%s' on device '%s'.", model_name, device)
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": batch_size},
    )


def get_embedding_function(settings: Optional[Settings] = None) -> HuggingFaceEmbeddings:
    """Return a cached local embedding function based on configuration.

    Caching avoids re-loading the model weights (which can take several
    seconds) on every Streamlit rerun.
    """
    s = settings or default_settings
    return _build_embeddings(s.embedding_model, s.embedding_device, s.embedding_batch_size)
