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

import httpx
import ollama
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


def model_is_installed(configured: str, installed: list[str]) -> bool:
    """Whether `configured` names one of the installed Ollama models.

    Ollama treats an untagged name as ":latest", so "mistral" matches
    "mistral:latest" - but "mistral:7b-instruct" does not match
    "mistral:latest", because Ollama would reject that request with a 404.
    """
    wanted = configured if ":" in configured else f"{configured}:latest"
    return wanted in installed


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

    configured = s.ollama_model
    model_available = model_is_installed(configured, model_names)

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
        # ChatOllama has no `timeout` field (it is silently ignored); the
        # timeout must reach the underlying ollama/httpx client instead.
        client_kwargs={"timeout": s.ollama_request_timeout},
    )


def describe_llm_error(exc: Exception, settings: Optional[Settings] = None) -> str:
    """Turn an exception from an LLM call into an actionable, user-facing
    message, distinguishing a missing model and a timeout from Ollama
    being unreachable."""
    s = settings or default_settings
    if isinstance(exc, ollama.ResponseError) and exc.status_code == 404:
        return (
            f"The model '{s.ollama_model}' is not installed in Ollama. Run 'ollama list' to see installed "
            f"models and 'ollama pull {s.ollama_model}' to download it, or set OLLAMA_MODEL to an installed tag."
        )
    if isinstance(exc, httpx.TimeoutException):
        return (
            f"The local model did not respond within {s.ollama_request_timeout} seconds. "
            "Try again, retrieve fewer passages, or increase OLLAMA_REQUEST_TIMEOUT."
        )
    return (
        f"The local Mistral model could not be reached at {s.ollama_base_url}. Please confirm Ollama is running "
        "('ollama list' to check installed models, 'ollama run mistral' to test) and try again."
    )
