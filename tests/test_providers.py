"""Tests for LLM provider selection (Ollama default, optional Gemini).

No live API calls: the Gemini client is always mocked and no real API key
is used. ChatOllama / requests are patched to fail loudly wherever a test
asserts that Ollama must not be touched."""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from google.genai import errors
from langchain_ollama import ChatOllama

from app.config import Settings
from app.gemini import GeminiChatModel, LLMConfigurationError
from app.insights import InsightEngine
from app.llm import check_llm_status, describe_llm_error, get_chat_model, provider_label
from app.rag_chain import RAGChain
from app.reporting import REPORT_SECTIONS, ReportGenerator
from app.retriever import RetrievedPassage, SemanticRetriever

FAKE_KEY = "test-key-not-real-123"
PASSAGE = RetrievedPassage(
    text="62% of consumers cite privacy concerns.", source="trust.pdf", page=14, document_type="pdf", relevance_score=0.9
)
INSIGHT_RESPONSE = "INSIGHT_TITLE: Privacy gap\nEVIDENCE: 62% cite privacy (trust.pdf, p. 14).\nCONFIDENCE: High"


@pytest.fixture
def base(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        vectorstore_dir=tmp_path / "vs",
        llm_provider="ollama",
        gemini_api_key="",
        gemini_model="gemini-3.8-flash",
    )


@pytest.fixture
def gemini_settings(base: Settings) -> Settings:
    return replace(base, llm_provider="gemini", gemini_api_key=FAKE_KEY)


@pytest.fixture
def genai_client():
    """Patch google.genai.Client so building a Gemini model uses a mock."""
    client = MagicMock()
    client.models.generate_content.return_value = SimpleNamespace(text=INSIGHT_RESPONSE, candidates=[])
    with patch("google.genai.Client", return_value=client) as client_cls:
        client.client_cls = client_cls
        yield client


@pytest.fixture
def no_ollama():
    """Fail the test if anything tries to use Ollama."""
    with patch("app.llm.ChatOllama", side_effect=AssertionError("Ollama must not be used")), patch(
        "app.llm.requests.get", side_effect=AssertionError("Ollama must not be contacted")
    ):
        yield


def _retriever():
    retriever = MagicMock(spec=SemanticRetriever)
    retriever.search.return_value = [PASSAGE]
    return retriever


# -- configuration ---------------------------------------------------------


def test_ollama_is_the_default_provider(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    s = Settings()
    assert s.llm_provider == "ollama"
    assert isinstance(get_chat_model(s), ChatOllama)


def test_provider_name_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "Gemini")
    assert Settings().llm_provider == "gemini"


def test_api_key_falls_back_to_streamlit_secrets(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "streamlit", SimpleNamespace(secrets={"GEMINI_API_KEY": FAKE_KEY}))
    assert Settings().gemini_api_key == FAKE_KEY


def test_environment_takes_precedence_over_streamlit_secrets(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "from-env")
    monkeypatch.setitem(sys.modules, "streamlit", SimpleNamespace(secrets={"GEMINI_API_KEY": FAKE_KEY}))
    assert Settings().gemini_api_key == "from-env"


def test_missing_streamlit_secrets_file_is_not_an_error(monkeypatch):
    class NoSecrets:
        def get(self, name):
            raise FileNotFoundError("no secrets.toml")

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "streamlit", SimpleNamespace(secrets=NoSecrets()))
    assert Settings().gemini_api_key == ""


def test_absent_secrets_file_is_checked_quietly(monkeypatch):
    # Real st.secrets renders an error element if `.get` is called with no
    # secrets.toml; config must use the quiet loader and stop there.
    secrets = MagicMock()
    secrets.load_if_toml_exists.return_value = False
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "streamlit", SimpleNamespace(secrets=secrets))

    assert Settings().gemini_api_key == ""
    secrets.get.assert_not_called()


def test_api_key_never_appears_in_reprs(gemini_settings, genai_client):
    assert FAKE_KEY not in repr(gemini_settings)
    assert FAKE_KEY not in repr(get_chat_model(gemini_settings))


# -- Gemini initialization and errors -------------------------------------


def test_gemini_initializes_with_mocked_client(gemini_settings, genai_client, no_ollama):
    llm = get_chat_model(gemini_settings)

    assert isinstance(llm, GeminiChatModel)
    assert llm.model == "gemini-3.8-flash"
    kwargs = genai_client.client_cls.call_args.kwargs
    assert kwargs["api_key"] == FAKE_KEY
    assert kwargs["http_options"].timeout == gemini_settings.gemini_request_timeout * 1000

    message = llm.invoke("hello")
    assert message.content == INSIGHT_RESPONSE
    genai_client.models.generate_content.assert_called_once_with(model="gemini-3.8-flash", contents="hello")


def test_missing_gemini_api_key_is_reported(base, no_ollama):
    s = replace(base, llm_provider="gemini", gemini_api_key="")

    with pytest.raises(LLMConfigurationError, match="GEMINI_API_KEY is not set"):
        get_chat_model(s)

    status = check_llm_status(s)
    assert status.reachable is False and "GEMINI_API_KEY is not set" in status.error

    result = RAGChain(retriever=_retriever(), settings=s).answer("What drives adoption?")
    assert "GEMINI_API_KEY is not set" in result.answer
    assert result.sources == [PASSAGE]  # retrieval still works and is shown


def test_invalid_provider_is_reported(base):
    s = replace(base, llm_provider="openai")

    with pytest.raises(LLMConfigurationError, match="Unsupported LLM_PROVIDER 'openai'"):
        get_chat_model(s)
    assert "Unsupported LLM_PROVIDER" in check_llm_status(s).error
    insight = InsightEngine(retriever=_retriever(), settings=s).generate_insight("wallets")
    assert "Unsupported LLM_PROVIDER" in insight.evidence


def _api_error(code: int, message: str) -> errors.APIError:
    return errors.APIError(code, {"error": {"code": code, "message": message, "status": "ERR"}})


@pytest.mark.parametrize(
    "code, message, expected",
    [
        (400, "API key not valid. Please pass a valid API key.", "rejected the API key"),
        (403, "Permission denied", "rejected the API key"),
        (404, "models/gemini-x is not found", "GEMINI_MODEL"),
        (429, "Resource has been exhausted", "quota"),
        (500, "Internal error", "Try again shortly"),
        (503, "Service unavailable", "temporarily unavailable or overloaded (503)"),
        (504, "Deadline exceeded", "temporarily unavailable or overloaded (504)"),
    ],
)
def test_gemini_api_errors_have_actionable_messages(gemini_settings, code, message, expected):
    text = describe_llm_error(_api_error(code, message), gemini_settings)
    assert expected in text
    assert FAKE_KEY not in text


def test_gemini_empty_response_is_reported(gemini_settings, genai_client):
    genai_client.models.generate_content.return_value = SimpleNamespace(
        text=None, candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name="SAFETY"))]
    )
    result = RAGChain(retriever=_retriever(), settings=gemini_settings).answer("q")
    assert "no text" in result.answer and "SAFETY" in result.answer


def test_gemini_status_checks_model_without_ollama(gemini_settings, genai_client, no_ollama):
    status = check_llm_status(gemini_settings)
    assert status.reachable and status.model_available and status.provider == "gemini"
    genai_client.models.get.assert_called_once_with(model="gemini-3.8-flash")
    genai_client.models.generate_content.assert_not_called()  # health check sends no content


def test_gemini_status_reports_rejected_key(gemini_settings, genai_client):
    genai_client.models.get.side_effect = _api_error(400, "API key not valid.")
    status = check_llm_status(gemini_settings)
    assert status.model_available is False
    assert "rejected the API key" in status.error


# -- routing every workflow through the configured provider ---------------


def test_all_workflows_route_through_gemini(gemini_settings, genai_client, no_ollama):
    rag = RAGChain(retriever=_retriever(), settings=gemini_settings).answer("Why do consumers avoid wallets?")
    insight = InsightEngine(retriever=_retriever(), settings=gemini_settings).generate_insight("privacy")
    report = ReportGenerator(retriever=_retriever(), settings=gemini_settings).generate_report("wallets")

    assert rag.answer == INSIGHT_RESPONSE
    assert insight.title == "Privacy gap"
    assert "hosted Google Gemini API" in report.markdown
    # 1 RAG + 1 insight + report (executive summary, sections, opportunities, recommendations, limitations)
    expected_calls = 1 + 1 + (1 + len(REPORT_SECTIONS) + 2 + 1)
    calls = genai_client.models.generate_content.call_args_list
    assert len(calls) == expected_calls
    assert all(c.kwargs["model"] == "gemini-3.8-flash" for c in calls)
    # Retrieved passages and citations reach the model unchanged.
    assert "Source: trust.pdf, Page: 14" in calls[0].kwargs["contents"]
    assert "Why do consumers avoid wallets?" in calls[0].kwargs["contents"]


def test_all_workflows_route_through_ollama_without_gemini_key(base):
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = SimpleNamespace(content=INSIGHT_RESPONSE)
    with patch("app.llm.ChatOllama", return_value=fake_llm) as chat_ollama, patch(
        "google.genai.Client", side_effect=AssertionError("Gemini must not be used")
    ):
        RAGChain(retriever=_retriever(), settings=base).answer("q")
        InsightEngine(retriever=_retriever(), settings=base).generate_insight("t")
        report = ReportGenerator(retriever=_retriever(), settings=base).generate_report("t")

    assert chat_ollama.call_count == 3  # one model per workflow instance
    assert chat_ollama.call_args.kwargs["model"] == base.ollama_model
    assert "via Ollama" in report.markdown
    assert fake_llm.invoke.call_count == 1 + 1 + (1 + len(REPORT_SECTIONS) + 2 + 1)


def test_provider_label_is_never_local_for_gemini(base, gemini_settings):
    assert "local" in provider_label(base)
    assert "local" not in provider_label(gemini_settings).lower()
    assert "hosted" in provider_label(gemini_settings)
