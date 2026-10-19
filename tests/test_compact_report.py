"""Tests for compact (grouped) report generation, partial-failure recovery,
and Gemini thinking/retry configuration. All model calls are mocked."""
from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from google.genai import errors, types

from app.config import Settings
from app.gemini import LLMConfigurationError, build_gemini_chat_model, describe_gemini_error
from app.reporting import (
    REPORT_COMPACT_STEPS,
    REPORT_GROUPS,
    REPORT_SECTIONS,
    ReportGenerator,
    parse_grouped_sections,
)
from app.retriever import RetrievedPassage, SemanticRetriever

ALL_TITLES = (
    ["Executive Summary"]
    + [title for title, _ in REPORT_SECTIONS]
    + ["Actionable Opportunities", "Strategic Recommendations", "Research Limitations"]
)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "raw", processed_dir=tmp_path / "p", vectorstore_dir=tmp_path / "vs",
        llm_provider="gemini", gemini_api_key="test-key-not-real", gemini_thinking_level="",
    )


def _passage(source: str, page: int, text: str, score: float = 0.8) -> RetrievedPassage:
    return RetrievedPassage(text=text, source=source, page=page, document_type="pdf", relevance_score=score,
                            chunk_hash=f"{source}:{page}:{text[:20]}")


SHARED = _passage("trust.pdf", 2, "Shared passage that every query retrieves.", 0.9)


def _retriever() -> MagicMock:
    retriever = MagicMock(spec=SemanticRetriever)
    retriever.search.side_effect = lambda query, k=None: [SHARED, _passage("adoption.pdf", 3, f"Evidence for: {query}", 0.7)]
    return retriever


def _api_error(code: int) -> errors.APIError:
    return errors.APIError(code, {"error": {"code": code, "message": "Service unavailable", "status": "UNAVAILABLE"}})


class FakeLLM:
    """Answers grouped prompts in the requested marker format; can fail on
    chosen call numbers or omit sections."""

    def __init__(self, fail_on=(), omit=()):
        self.prompts: list[str] = []
        self.fail_on, self.omit = set(fail_on), set(omit)

    def invoke(self, prompt: str):
        self.prompts.append(prompt)
        if len(self.prompts) in self.fail_on:
            raise _api_error(503)
        block = prompt.split("Sections to write, in this order:")[1].split("Format:")[0]
        titles = re.findall(r"^- ([^:\n]+):", block, re.MULTILINE)
        body = "\n\n".join(f"### SECTION: {t}\nGrounded text for {t} (trust.pdf, p. 2)." for t in titles if t not in self.omit)
        return MagicMock(content=body)


# -- grouping -------------------------------------------------------------------


def test_groups_cover_all_ten_sections_once():
    titles = [spec.title for _, _, specs in REPORT_GROUPS for spec in specs]
    assert sorted(titles) == sorted(ALL_TITLES) and len(titles) == 10
    assert REPORT_COMPACT_STEPS == 3


def test_compact_report_uses_three_calls_and_keeps_every_section(settings):
    llm, retriever = FakeLLM(), _retriever()
    progress = []

    report = ReportGenerator(retriever=retriever, llm=llm, settings=settings).generate_compact_report(
        "wallet adoption", k=2, progress_callback=lambda d, t, label: progress.append((d, t))
    )

    assert len(llm.prompts) == 3
    assert progress == [(0, 3), (1, 3), (2, 3), (3, 3)]
    assert report.mode == "compact" and report.errors == [] and report.failed_groups == []
    for title in ALL_TITLES:
        assert f"# {title}\nGrounded text for {title}" in report.markdown or f"# {title}\n_The following" in report.markdown
    assert "# Key Evidence" in report.markdown and "# Source References" in report.markdown
    assert report.sources == ["adoption.pdf", "trust.pdf"]
    assert "Incomplete report" not in report.markdown


def test_each_group_retrieves_per_section_and_deduplicates(settings):
    llm, retriever = FakeLLM(), _retriever()
    ReportGenerator(retriever=retriever, llm=llm, settings=settings).generate_compact_report("fees", k=2)

    queries = [c.args[0] for c in retriever.search.call_args_list]
    for title, query in REPORT_SECTIONS:
        assert query in queries
    assert "business opportunities related to fees" in queries
    # The passage every query returns is sent once per group, not once per section.
    for prompt in llm.prompts:
        assert prompt.count("Shared passage that every query retrieves.") == 1


def test_grouped_context_respects_the_length_limit(settings):
    retriever = MagicMock(spec=SemanticRetriever)
    retriever.search.side_effect = lambda q, k=None: [_passage(f"{q[:12]}.pdf", i, "x " * 400) for i in range(1, 6)]
    llm = FakeLLM()
    small = replace(settings, max_context_chars=3000)

    report = ReportGenerator(retriever=retriever, llm=llm, settings=small).generate_compact_report("t", k=5)

    for prompt in llm.prompts:
        context = prompt.split("duplicates removed):")[1].split("Write the following sections")[0]
        context = context.split("CORPUS NOTES")[0]  # limitation notes aren't retrieved context
        assert len(context.strip()) <= 3000 + len(" ...[truncated]")
    # Source references only list passages that fit in a prompt (plus key evidence).
    used = {p.source for g in report.groups for p in g.passages}
    assert all(f"[Passage" in p for p in llm.prompts) and used


# -- failures and recovery ------------------------------------------------------


def test_failed_group_is_marked_and_other_groups_kept(settings):
    llm = FakeLLM(fail_on={2})  # the adoption group fails with a 503

    report = ReportGenerator(retriever=_retriever(), llm=llm, settings=settings).generate_compact_report("t")

    assert len(llm.prompts) == 3  # later groups still ran
    assert [g.key for g in report.failed_groups] == ["adoption"]
    assert "Grounded text for Executive Summary" in report.markdown
    assert "Grounded text for Strategic Recommendations" in report.markdown
    for title in ("Adoption Drivers", "Adoption Barriers", "Trust and Security Factors", "Consumer Segments"):
        assert f"# {title}\n_Not generated: Gemini is temporarily unavailable or overloaded (503)" in report.markdown
    assert "**Incomplete report:** 4 of 10 sections could not be generated" in report.markdown
    assert len(report.errors) == 1 and report.errors[0].startswith("Adoption, trust, and consumer segments:")


def test_retry_regenerates_only_failed_groups(settings):
    generator = ReportGenerator(retriever=_retriever(), llm=FakeLLM(fail_on={2}), settings=settings)
    report = generator.generate_compact_report("t")
    ok_groups = {g.key: g for g in report.groups if g.ok}

    retry_llm = FakeLLM()
    retried = ReportGenerator(retriever=_retriever(), llm=retry_llm, settings=settings).retry_failed_groups(report)

    assert len(retry_llm.prompts) == 1 and "Adoption Drivers" in retry_llm.prompts[0]
    assert retried.errors == [] and retried.failed_groups == []
    for key, group in ok_groups.items():
        assert next(g for g in retried.groups if g.key == key) is group  # reused, not regenerated
    assert "Incomplete report" not in retried.markdown


def test_failed_retry_keeps_previously_generated_sections(settings):
    report = ReportGenerator(
        retriever=_retriever(), llm=FakeLLM(omit={"Consumer Segments"}), settings=settings
    ).generate_compact_report("t")
    assert [g.key for g in report.failed_groups] == ["adoption"]
    assert report.errors == ["Consumer Segments: the model's response did not include this section."]

    retried = ReportGenerator(
        retriever=_retriever(), llm=FakeLLM(fail_on={1}), settings=settings
    ).retry_failed_groups(report)

    adoption = next(g for g in retried.groups if g.key == "adoption")
    assert adoption.error and "Adoption Drivers" in adoption.sections
    assert "Grounded text for Adoption Drivers" in retried.markdown
    assert "# Consumer Segments\n_Not generated" in retried.markdown


def test_parse_grouped_sections_tolerates_marker_variants():
    text = "Intro noise\n### SECTION: Adoption Drivers\nA.\n**SECTION: adoption barriers**\nB.\n## SECTION: Unknown\nC."
    assert parse_grouped_sections(text, ["Adoption Drivers", "Adoption Barriers"]) == {
        "Adoption Drivers": "A.",
        "Adoption Barriers": "B.",
    }


# -- Gemini thinking level and retries -----------------------------------------


def test_thinking_level_is_sent_with_requests(settings):
    client = MagicMock()
    client.models.generate_content.return_value = MagicMock(text="ok")
    llm = build_gemini_chat_model(replace(settings, gemini_thinking_level="low"), client=client)

    llm.invoke("hi")

    config = client.models.generate_content.call_args.kwargs["config"]
    assert config.thinking_config.thinking_level == types.ThinkingLevel.LOW


def test_no_config_when_thinking_level_is_unset(settings):
    client = MagicMock()
    client.models.generate_content.return_value = MagicMock(text="ok")
    build_gemini_chat_model(settings, client=client).invoke("hi")
    assert "config" not in client.models.generate_content.call_args.kwargs


def test_invalid_thinking_level_is_rejected(settings):
    with pytest.raises(LLMConfigurationError, match="GEMINI_THINKING_LEVEL"):
        build_gemini_chat_model(replace(settings, gemini_thinking_level="fast"), client=MagicMock())


def test_public_demo_defaults_to_low_thinking(monkeypatch):
    monkeypatch.delenv("GEMINI_THINKING_LEVEL", raising=False)
    monkeypatch.setenv("PUBLIC_DEMO_MODE", "true")
    assert Settings().gemini_thinking_level == "low"
    monkeypatch.setenv("PUBLIC_DEMO_MODE", "false")
    assert Settings().gemini_thinking_level == ""


def test_client_uses_bounded_sdk_retries_for_transient_errors(settings):
    with patch("google.genai.Client") as client_cls:
        build_gemini_chat_model(settings)
    retry = client_cls.call_args.kwargs["http_options"].retry_options
    assert retry.attempts == 3 and retry.max_delay == 10.0
    assert sorted(retry.http_status_codes) == [500, 502, 503, 504]  # 429 is not retried


def test_transient_error_message_mentions_attempts(settings):
    message = describe_gemini_error(_api_error(503), settings)
    assert "temporarily unavailable or overloaded (503) after 3 attempts" in message
