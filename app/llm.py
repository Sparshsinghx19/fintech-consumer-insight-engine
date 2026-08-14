"""
Local LLM access via Ollama.

Uses the current (non-deprecated) `langchain_ollama` package. The
application never sends documents, questions, or generated content to a
cloud LLM API - all generation happens through the locally running
Ollama server.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import requests
from langchain_ollama import ChatOllama

from app.config import Settings, settings as default_settings
from app.utils import get_logger

logger = get_logger(__name__)


@dataclass
class OllamaStatus:
    reachable: bool
    model_available: bool
    available_models: list[str]
    error: Optional[str] = None


def check_ollama_status(settings: Optional[Settings] = None, timeout: float = 3.0) -> OllamaStatus:
    """Ping the local Ollama server and check whether the configured model
    is present. Never raises; failures are reported in the returned status
    object so the UI can show a clear, actionable error."""
    s = settings or default_settings
    url = f"{s.ollama_base_url.rstrip('/')}/api/tags"
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        return OllamaStatus(
            reachable=False,
            model_available=False,
            available_models=[],
            error=(
                f"Could not connect to Ollama at {s.ollama_base_url}. "
                "Start Ollama (e.g. run 'ollama serve' or open the Ollama app) and try again."
            ),
        )
    except requests.exceptions.RequestException as exc:
        return OllamaStatus(reachable=False, model_available=False, available_models=[], error=str(exc))

    try:
        payload = resp.json()
        model_names = [m.get("name", "") for m in payload.get("models", [])]
    except ValueError:
        return OllamaStatus(reachable=True, model_available=False, available_models=[], error="Unexpected response from Ollama.")

    # Ollama model tags look like "mistral:latest"; allow matching the
    # configured name with or without an explicit tag.
    configured = s.ollama_model
    model_available = any(
        name == configured or name.split(":")[0] == configured.split(":")[0] for name in model_names
    )

    error = None
    if not model_available:
        error = (
            f"Model '{configured}' was not found in Ollama's local model list. "
            f"Run 'ollama list' to see installed models, or 'ollama pull {configured}' to download it. "
            f"If your installed model uses a different tag, set OLLAMA_MODEL accordingly in your .env file."
        )

    return OllamaStatus(reachable=True, model_available=model_available, available_models=model_names, error=error)


def get_chat_model(settings: Optional[Settings] = None) -> ChatOllama:
    """Build a ChatOllama instance pointed at the configured local model."""
    s = settings or default_settings
    return ChatOllama(
        model=s.ollama_model,
        base_url=s.ollama_base_url,
        temperature=s.ollama_temperature,
        timeout=s.ollama_request_timeout,
    )
