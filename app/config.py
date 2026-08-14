"""
Centralized, environment-driven configuration for the FinTech Consumer
Insight Engine.

All values are configurable through environment variables (or a local
`.env` file). Nothing here is hard-coded except sane defaults that make
the app runnable out of the box for local development.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root if present. Does not override variables
# that are already set in the real environment.
load_dotenv(override=False)


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"Environment variable {name}='{raw}' is not a valid integer.")


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # --- Ollama / LLM ---
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "mistral"))
    ollama_temperature: float = field(default_factory=lambda: float(os.getenv("OLLAMA_TEMPERATURE", "0.1")))
    ollama_request_timeout: int = field(default_factory=lambda: _get_int("OLLAMA_REQUEST_TIMEOUT", 120))

    # --- Embeddings ---
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    embedding_device: str = field(default_factory=lambda: os.getenv("EMBEDDING_DEVICE", "cpu"))
    embedding_batch_size: int = field(default_factory=lambda: _get_int("EMBEDDING_BATCH_SIZE", 32))

    # --- Chunking / retrieval ---
    chunk_size: int = field(default_factory=lambda: _get_int("CHUNK_SIZE", 1000))
    chunk_overlap: int = field(default_factory=lambda: _get_int("CHUNK_OVERLAP", 150))
    retrieval_k: int = field(default_factory=lambda: _get_int("RETRIEVAL_K", 5))

    # --- Storage paths ---
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", "data/raw")))
    processed_dir: Path = field(default_factory=lambda: Path(os.getenv("PROCESSED_DIR", "data/processed")))
    vectorstore_dir: Path = field(default_factory=lambda: Path(os.getenv("VECTORSTORE_DIR", "vectorstore")))
    collection_name: str = field(default_factory=lambda: os.getenv("CHROMA_COLLECTION", "fintech_research"))

    # --- Misc ---
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    max_context_chars: int = field(default_factory=lambda: _get_int("MAX_CONTEXT_CHARS", 12000))

    def ensure_directories(self) -> None:
        """Create required directories if they do not already exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.vectorstore_dir.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    """Factory so tests can monkeypatch environment variables cleanly."""
    return Settings()


settings = get_settings()
