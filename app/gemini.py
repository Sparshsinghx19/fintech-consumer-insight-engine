"""
Optional hosted generation via the Google Gemini API.

Uses Google's current `google-genai` SDK directly. The RAG workflows only
need `invoke(prompt) -> message with .content`, so a thin adapter keeps
them provider-agnostic without pulling in a LangChain integration that
would force a LangChain upgrade.

PRIVACY: unlike the default Ollama provider, this sends each prompt -
the user's question/topic plus the retrieved document passages - to
Google's servers. Embeddings and the vector store remain local.
"""
from __future__ import annotations

from typing import Any, Optional

from langchain_core.messages import AIMessage

from app.config import Settings


class LLMConfigurationError(RuntimeError):
    """The selected LLM provider is misconfigured. The message is
    user-facing and never contains secret values."""


class GeminiResponseError(RuntimeError):
    """Gemini answered but returned no usable text (e.g. blocked)."""


class GeminiChatModel:
    """Minimal chat model over `google.genai.Client`, exposing the same
    `invoke(prompt)` interface the RAG chain, insight engine, and report
    generator already use with ChatOllama."""

    def __init__(self, client: Any, model: str):
        self._client = client
        self.model = model

    def __repr__(self) -> str:  # never expose the client (it holds the API key)
        return f"GeminiChatModel(model={self.model!r})"

    def invoke(self, prompt: str) -> AIMessage:
        response = self._client.models.generate_content(model=self.model, contents=prompt)
        text = response.text
        if not text:
            reason = _finish_reason(response)
            raise GeminiResponseError(
                f"Gemini returned no text for this request{f' (finish reason: {reason})' if reason else ''}. "
                "Try rephrasing the question."
            )
        return AIMessage(content=text)


def _finish_reason(response: Any) -> Optional[str]:
    try:
        reason = response.candidates[0].finish_reason
    except (AttributeError, IndexError, TypeError):
        return None
    return getattr(reason, "name", None) or (str(reason) if reason else None)


def build_gemini_chat_model(settings: Settings, client: Any = None) -> GeminiChatModel:
    """Validate Gemini configuration and build the chat model. `client`
    can be injected (tests); otherwise a real client is created."""
    validate_gemini_settings(settings)
    if client is None:
        client = create_gemini_client(settings)
    return GeminiChatModel(client, settings.gemini_model)


def validate_gemini_settings(settings: Settings) -> None:
    """Raise LLMConfigurationError if Gemini cannot be used as configured."""
    if not settings.gemini_api_key:
        raise LLMConfigurationError(
            "LLM_PROVIDER is set to 'gemini' but GEMINI_API_KEY is not set. Add it to your environment or "
            ".env file, or to Streamlit secrets when deployed on Streamlit Community Cloud."
        )
    if not settings.gemini_model:
        raise LLMConfigurationError("LLM_PROVIDER is set to 'gemini' but GEMINI_MODEL is empty.")


def create_gemini_client(settings: Settings) -> Any:
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise LLMConfigurationError(
            "The 'google-genai' package is required for LLM_PROVIDER=gemini. Run 'pip install -r requirements.txt'."
        ) from exc
    # Pass the key explicitly: otherwise the SDK silently prefers any
    # GOOGLE_API_KEY found in the environment.
    return genai.Client(
        api_key=settings.gemini_api_key,
        http_options=types.HttpOptions(timeout=settings.gemini_request_timeout * 1000),  # milliseconds
    )


def describe_gemini_error(exc: Exception, settings: Settings) -> Optional[str]:
    """User-facing message for a Gemini API error, or None if `exc` is not
    one. Messages never include the API key."""
    if isinstance(exc, (LLMConfigurationError, GeminiResponseError)):
        return str(exc)
    try:
        from google.genai import errors
    except ImportError:
        return None
    if not isinstance(exc, errors.APIError):
        return None

    code = getattr(exc, "code", None)
    detail = str(getattr(exc, "message", "") or "")
    if code in (401, 403) or "API key" in detail or "API_KEY" in detail:
        return "Gemini rejected the API key. Check that GEMINI_API_KEY is correct and active."
    if code == 404:
        return (
            f"Gemini model '{settings.gemini_model}' was not found or is not available to this key. "
            "Set GEMINI_MODEL to a current model ID."
        )
    if code == 429:
        return "Gemini rate limit or quota exceeded. Wait a moment and try again, or check your API quota."
    if code is not None and code >= 500:
        return f"The Gemini service returned an error ({code}). Try again shortly."
    return f"The Gemini request failed ({code}): {detail[:200]}"
