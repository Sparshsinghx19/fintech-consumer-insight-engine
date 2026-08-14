"""Tests for prompt construction: grounding rules present, no unsupported
claim language slips through, context/question interpolate correctly."""
from __future__ import annotations

from app.prompts import (
    build_insight_prompt,
    build_rag_prompt,
    build_report_limitations_prompt,
    build_report_section_prompt,
)


def test_rag_prompt_contains_grounding_rules_and_question():
    prompt = build_rag_prompt(question="What drives adoption?", context="Some retrieved context.")
    assert "Do NOT invent statistics" in prompt
    assert "Do NOT invent citations" in prompt
    assert "What drives adoption?" in prompt
    assert "Some retrieved context." in prompt
    assert "SOURCES USED" in prompt


def test_rag_prompt_instructs_model_to_admit_gaps():
    prompt = build_rag_prompt(question="Q", context="C")
    assert "does not contain enough information" in prompt or "say so explicitly" in prompt


def test_insight_prompt_has_required_structure_labels():
    prompt = build_insight_prompt(topic="digital wallet adoption", context="context text")
    for label in [
        "INSIGHT_TITLE:",
        "CONSUMER_PROBLEM:",
        "EVIDENCE:",
        "INTERPRETATION:",
        "SOURCES:",
        "CONFIDENCE:",
        "BUSINESS_IMPLICATION:",
        "RECOMMENDED_ACTION:",
        "POTENTIAL_OPPORTUNITY:",
    ]:
        assert label in prompt
    assert "digital wallet adoption" in prompt
    assert "context text" in prompt


def test_report_section_prompt_includes_section_and_topic():
    prompt = build_report_section_prompt(section_title="Adoption Barriers", topic="mobile payments", context="ctx")
    assert "Adoption Barriers" in prompt
    assert "mobile payments" in prompt
    assert "ctx" in prompt


def test_report_limitations_prompt_includes_context():
    prompt = build_report_limitations_prompt(topic="mobile payments", limitations_context="Only 2 documents indexed.")
    assert "Only 2 documents indexed." in prompt
    assert "Research Limitations" in prompt or "limitations" in prompt.lower()


def test_grounding_rules_forbid_fabricated_findings_language():
    prompt = build_rag_prompt(question="Q", context="C")
    assert "invent" in prompt.lower()
    assert "EVIDENCE" in prompt.upper() or "INTERPRETATION" in prompt.upper()
