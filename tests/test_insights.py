"""Tests for the Consumer Insight Engine: schema validation, response
parsing, and graceful handling of unreachable/malformed LLM output."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.config import Settings
from app.insights import ConsumerInsight, InsightEngine, parse_insight_response
from app.retriever import RetrievedPassage, SemanticRetriever

SAMPLE_MODEL_RESPONSE = """\
INSIGHT_TITLE: Trust Gaps Slow Digital Wallet Adoption
CONSUMER_PROBLEM: Consumers hesitate to link bank accounts to new wallets.
EVIDENCE: 62% of surveyed consumers cited data privacy concerns (trust_survey.pdf, p. 14).
INTERPRETATION: Privacy concerns likely outweigh convenience for cautious segments.
SOURCES: trust_survey.pdf (p. 14)
CONFIDENCE: Medium
BUSINESS_IMPLICATION: Wallet providers may be losing privacy-conscious users at onboarding.
RECOMMENDED_ACTION: Add a clear, prominent data-usage explainer during signup.
POTENTIAL_OPPORTUNITY: Differentiate on privacy-first onboarding messaging.
"""


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "raw", processed_dir=tmp_path / "processed", vectorstore_dir=tmp_path / "vs")


def test_consumer_insight_schema_validates_required_fields():
    insight = ConsumerInsight(
        title="T",
        consumer_problem="P",
        evidence="E",
        interpretation="I",
    )
    assert insight.confidence == "Medium"  # default
    assert insight.topic == ""


def test_parse_insight_response_extracts_all_fields():
    insight = parse_insight_response(SAMPLE_MODEL_RESPONSE, topic="digital wallet adoption")
    assert insight.title == "Trust Gaps Slow Digital Wallet Adoption"
    assert "data privacy" in insight.evidence
    assert insight.confidence == "Medium"
    assert "trust_survey.pdf" in insight.sources
    assert insight.recommended_action.startswith("Add a clear")
    assert insight.topic == "digital wallet adoption"


def test_parse_insight_response_handles_malformed_output_gracefully():
    malformed = "The model just rambled without following the format at all."
    insight = parse_insight_response(malformed, topic="fallback topic")
    assert insight.evidence == malformed
    assert insight.topic == "fallback topic"
    assert isinstance(insight, ConsumerInsight)  # never raises


def test_insight_engine_generates_insight_with_mocked_llm(settings: Settings):
    passages = [RetrievedPassage(text="Evidence text.", source="a.pdf", page=1, document_type="pdf", relevance_score=0.9)]
    retriever = MagicMock(spec=SemanticRetriever)
    retriever.search.return_value = passages

    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = SAMPLE_MODEL_RESPONSE
    mock_llm.invoke.return_value = mock_response

    engine = InsightEngine(retriever=retriever, llm=mock_llm, settings=settings)
    insight = engine.generate_insight("digital wallet adoption")

    assert insight.title == "Trust Gaps Slow Digital Wallet Adoption"
    mock_llm.invoke.assert_called_once()


def test_insight_engine_handles_llm_failure_gracefully(settings: Settings):
    retriever = MagicMock(spec=SemanticRetriever)
    retriever.search.return_value = []

    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = ConnectionError("Ollama not running")

    engine = InsightEngine(retriever=retriever, llm=mock_llm, settings=settings)
    insight = engine.generate_insight("some topic")

    assert "could not be reached" in insight.evidence.lower()
    assert insight.topic == "some topic"


def test_generate_insights_returns_one_per_topic(settings: Settings):
    retriever = MagicMock(spec=SemanticRetriever)
    retriever.search.return_value = []
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = SAMPLE_MODEL_RESPONSE
    mock_llm.invoke.return_value = mock_response

    engine = InsightEngine(retriever=retriever, llm=mock_llm, settings=settings)
    topics = ["Topic A", "Topic B", "Topic C"]
    results = engine.generate_insights(topics)

    assert len(results) == 3
    assert [r.topic for r in results] == topics
