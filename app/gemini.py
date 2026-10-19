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

    def __init__(self, client: Any, model: str, config: Any = None):
        self._client = client
        self.model = model
        self._config = config

    def __repr__(self) -> str:  # never expose the client (it holds the API key)
        return f"GeminiChatModel(model={self.model!r})"

    def invoke(self, prompt: str) -> AIMessage:
        kwargs = {"config": self._config} if self._config is not None else {}
        response = self._client.models.generate_content(model=self.model, contents=prompt, **kwargs)
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
    return GeminiChatModel(client, settings.gemini_model, generation_config(settings))


THINKING_LEVELS = ("minimal", "low", "medium", "high")
# Transient server-side errors worth retrying. 429 is excluded: retrying a
# quota/rate-limit error within seconds rarely helps and burns quota.
RETRYABLE_STATUS_CODES = (500, 502, 503, 504)


def generation_config(settings: Settings) -> Any:
    """GenerateContentConfig carrying the thinking level, or None to use the
    model's defaults."""
    if not settings.gemini_thinking_level:
        return None
    from google.genai import types

    level = types.ThinkingLevel(settings.gemini_thinking_level.upper())
    return types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_level=level))


def validate_gemini_settings(settings: Settings) -> None:
    """Raise LLMConfigurationError if Gemini cannot be used as configured."""
    if not settings.gemini_api_key:
        raise LLMConfigurationError(
            "LLM_PROVIDER is set to 'gemini' but GEMINI_API_KEY is not set. Add it to your environment or "
            ".env file, or to Streamlit secrets when deployed on Streamlit Community Cloud."
        )
    if not settings.gemini_model:
        raise LLMConfigurationError("LLM_PROVIDER is set to 'gemini' but GEMINI_MODEL is empty.")
    if settings.gemini_thinking_level and settings.gemini_thinking_level not in THINKING_LEVELS:
        raise LLMConfigurationError(
            f"GEMINI_THINKING_LEVEL '{settings.gemini_thinking_level}' is not valid. "
            f"Use one of: {', '.join(THINKING_LEVELS)}, or leave it empty for the model default."
        )
    if settings.gemini_retry_attempts < 1:
        raise LLMConfigurationError("GEMINI_RETRY_ATTEMPTS must be at least 1.")


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
    #
    # The SDK makes a single attempt unless retry options are given. These
    # enable its own bounded exponential backoff (about 2s, then 4s, capped
    # at 10s) for transient 5xx errors; the SDK also retries timeouts and
    # connection errors under the same attempt limit. This is the only retry
    # layer - callers don't retry again.
    retry = types.HttpRetryOptions(
        attempts=settings.gemini_retry_attempts,
        initial_delay=2.0,
        max_delay=10.0,
        http_status_codes=list(RETRYABLE_STATUS_CODES),
    )
    return genai.Client(
        api_key=settings.gemini_api_key,
        http_options=types.HttpOptions(timeout=settings.gemini_request_timeout * 1000, retry_options=retry),  # ms
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
    if code == 400 and "thinking" in detail.lower():
        return (
            f"Gemini model '{settings.gemini_model}' does not accept GEMINI_THINKING_LEVEL="
            f"'{settings.gemini_thinking_level}'. Choose a level the model supports, or leave it empty."
        )
    if code in (503, 504):
        tries = settings.gemini_retry_attempts
        return (
            f"Gemini is temporarily unavailable or overloaded ({code})"
            + (f" after {tries} attempts" if tries > 1 else "")
            + ". This is usually brief; please try again in a minute."
        )
    if code is not None and code >= 500:
        return f"The Gemini service returned an error ({code}). Try again shortly."
    return f"The Gemini request failed ({code}): {detail[:200]}"
