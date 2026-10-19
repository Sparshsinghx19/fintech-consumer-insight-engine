"""
Centralized, environment-driven configuration for the FinTech Consumer
Insight Engine.

All values are configurable through environment variables (or a local
`.env` file). Nothing here is hard-coded except sane defaults that make
the app runnable out of the box for local development.
"""
from __future__ import annotations

import os
import sys
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


def _streamlit_secret(name: str) -> str | None:
    """Read `name` from Streamlit secrets (`st.secrets`), which is how
    Streamlit Community Cloud provides configuration. Only consulted when
    Streamlit is already loaded, so tests and scripts never import it, and
    any missing secrets file is treated as "not set"."""
    st = sys.modules.get("streamlit")
    if st is None:
        return None
    try:
        # Without a secrets.toml, plain `st.secrets.get` renders a "No secrets
        # found" error element into the page; load quietly first.
        load_quietly = getattr(st.secrets, "load_if_toml_exists", None)
        if load_quietly is not None and not load_quietly():
            return None
        value = st.secrets.get(name)
    except Exception:  # noqa: BLE001 - no secrets.toml, parse errors, etc.
        return None
    return str(value).strip() if value else None


def _get_env_or_secret(name: str, default: str) -> str:
    """Environment variable (or .env) first, then Streamlit secrets."""
    raw = os.getenv(name)
    if raw is not None and raw.strip():
        return raw.strip()
    return _streamlit_secret(name) or default


def _get_bool(name: str, default: bool) -> bool:
    """Boolean from the environment (or .env), then Streamlit secrets."""
    raw = _get_env_or_secret(name, "")
    if not raw:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _get_int_env_or_secret(name: str, default: int) -> int:
    raw = _get_env_or_secret(name, "")
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"Setting {name}='{raw}' is not a valid integer.")


DEMO_DATA_DIR = "demo_data"


@dataclass(frozen=True)
class Settings:
    # --- LLM provider ---
    # "ollama" (default, fully local) or "gemini" (hosted Google Gemini API).
    llm_provider: str = field(default_factory=lambda: _get_env_or_secret("LLM_PROVIDER", "ollama").lower())

    # --- Gemini (only used when llm_provider == "gemini") ---
    # repr=False keeps the key out of logs, tracebacks, and debug output.
    gemini_api_key: str = field(default_factory=lambda: _get_env_or_secret("GEMINI_API_KEY", ""), repr=False)
    gemini_model: str = field(default_factory=lambda: _get_env_or_secret("GEMINI_MODEL", "gemini-3.8-flash"))
    gemini_request_timeout: int = field(default_factory=lambda: _get_int("GEMINI_REQUEST_TIMEOUT", 120))
    # Thinking level ("minimal", "low", "medium", "high"; support varies by
    # model). Empty means the model's default - except in public demo mode,
    # where "low" is the default to keep hosted responses fast.
    gemini_thinking_level: str = field(
        default_factory=lambda: _get_env_or_secret(
            "GEMINI_THINKING_LEVEL", "low" if _get_bool("PUBLIC_DEMO_MODE", False) else ""
        ).lower()
    )
    # Total attempts per Gemini request (1 = no retries). Retries use the
    # SDK's bounded exponential backoff and only cover transient errors.
    gemini_retry_attempts: int = field(default_factory=lambda: _get_int_env_or_secret("GEMINI_RETRY_ATTEMPTS", 3))

    # --- Public demo mode (off by default) ---
    # When on, visitors can't upload, delete, or reindex documents, and LLM
    # generation is rate-limited per browser session and per server process.
    public_demo_mode: bool = field(default_factory=lambda: _get_bool("PUBLIC_DEMO_MODE", False))
    # Model calls allowed per window (a full report counts as one call per section).
    demo_session_request_limit: int = field(default_factory=lambda: _get_int_env_or_secret("DEMO_SESSION_REQUEST_LIMIT", 20))
    demo_global_request_limit: int = field(default_factory=lambda: _get_int_env_or_secret("DEMO_GLOBAL_REQUEST_LIMIT", 200))
    demo_rate_window_seconds: int = field(default_factory=lambda: _get_int_env_or_secret("DEMO_RATE_WINDOW_SECONDS", 3600))

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
    # In public demo mode the bundled, version-controlled demo corpus is the
    # default source; an explicit DATA_DIR always wins.
    data_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("DATA_DIR") or (DEMO_DATA_DIR if _get_bool("PUBLIC_DEMO_MODE", False) else "data/raw")
        )
    )
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
