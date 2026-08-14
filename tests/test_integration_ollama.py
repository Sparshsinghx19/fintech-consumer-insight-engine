"""
Optional integration test that requires a real, locally running Ollama
server with the configured model available.

This test is skipped automatically unless Ollama is reachable, so the
rest of the test suite (and CI) never depends on it. Run explicitly with:

    pytest tests/test_integration_ollama.py -m integration -v
"""
from __future__ import annotations

import pytest

from app.config import settings
from app.llm import check_ollama_status, get_chat_model

pytestmark = pytest.mark.integration


def _ollama_available() -> bool:
    status = check_ollama_status(settings, timeout=2.0)
    return status.reachable and status.model_available


@pytest.mark.skipif(not _ollama_available(), reason="Ollama is not running or the configured model is unavailable.")
def test_local_mistral_generates_a_response():
    llm = get_chat_model(settings)
    response = llm.invoke("Reply with exactly the single word: OK")
    text = response.content if hasattr(response, "content") else str(response)
    assert len(text.strip()) > 0
