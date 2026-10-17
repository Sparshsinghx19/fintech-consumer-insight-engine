"""Tests for Ollama access: model-tag matching, health check, request
timeout wiring, and user-facing error classification. No live Ollama
server is required - HTTP calls are mocked."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import ollama
import pytest
import requests

from app.config import Settings
from app.llm import check_ollama_status, describe_llm_error, get_chat_model, model_is_installed
from app.rag_chain import RAGChain
from app.reporting import ReportGenerator
from app.retriever import SemanticRetriever


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        vectorstore_dir=tmp_path / "vs",
        ollama_model="mistral",
        ollama_request_timeout=42,
    )


@pytest.mark.parametrize(
    "configured, installed, expected",
    [
        ("mistral", ["mistral:latest"], True),
        ("mistral:latest", ["mistral:latest"], True),
        ("mistral:7b-instruct", ["mistral:7b-instruct", "llama3:latest"], True),
        # A different tag of the same model is NOT available - Ollama would 404.
        ("mistral:7b-instruct", ["mistral:latest"], False),
        ("mistral", ["mistral:7b-instruct"], False),
        ("mistral", ["mistral-nemo:latest"], False),
        ("mistral", [], False),
    ],
)
def test_model_is_installed_requires_exact_tag(configured, installed, expected):
    assert model_is_installed(configured, installed) is expected


def _tags_response(names):
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"models": [{"name": n} for n in names]}
    return resp


def test_check_ollama_status_reports_missing_tag(settings: Settings):
    s = Settings(**{**settings.__dict__, "ollama_model": "mistral:7b-instruct"})
    with patch("app.llm.requests.get", return_value=_tags_response(["mistral:latest"])):
        status = check_ollama_status(s)
    assert status.reachable is True
    assert status.model_available is False
    assert "ollama pull mistral:7b-instruct" in status.error


def test_check_ollama_status_available(settings: Settings):
    with patch("app.llm.requests.get", return_value=_tags_response(["mistral:latest"])):
        status = check_ollama_status(settings)
    assert status.reachable and status.model_available and status.error is None


def test_check_ollama_status_unreachable(settings: Settings):
    with patch("app.llm.requests.get", side_effect=requests.exceptions.ConnectionError("refused")):
        status = check_ollama_status(settings)
    assert status.reachable is False
    assert "Could not connect" in status.error


def test_chat_model_passes_request_timeout_to_http_client(settings: Settings):
    llm = get_chat_model(settings)
    assert llm._client._client.timeout == httpx.Timeout(42)


def test_describe_llm_error_missing_model(settings: Settings):
    msg = describe_llm_error(ollama.ResponseError("model 'mistral' not found", 404), settings)
    assert "not installed" in msg
    assert "ollama pull mistral" in msg


def test_describe_llm_error_timeout(settings: Settings):
    msg = describe_llm_error(httpx.ReadTimeout("timed out"), settings)
    assert "42 seconds" in msg
    assert "OLLAMA_REQUEST_TIMEOUT" in msg


def test_describe_llm_error_unreachable(settings: Settings):
    msg = describe_llm_error(httpx.ConnectError("refused"), settings)
    assert "could not be reached" in msg
    assert settings.ollama_base_url in msg


def _failing_llm(exc):
    llm = MagicMock()
    llm.invoke.side_effect = exc
    return llm


def _empty_retriever():
    retriever = MagicMock(spec=SemanticRetriever)
    retriever.search.return_value = []
    return retriever


def test_rag_chain_surfaces_missing_model_message(settings: Settings):
    chain = RAGChain(
        retriever=_empty_retriever(),
        llm=_failing_llm(ollama.ResponseError("not found", 404)),
        settings=settings,
    )
    result = chain.answer("What drives adoption?")
    assert "not installed" in result.answer


def test_report_sections_surface_timeout_message(settings: Settings):
    generator = ReportGenerator(
        retriever=_empty_retriever(),
        llm=_failing_llm(httpx.ReadTimeout("timed out")),
        settings=settings,
    )
    report = generator.generate_report("wallets")
    assert "did not respond within 42 seconds" in report.markdown
