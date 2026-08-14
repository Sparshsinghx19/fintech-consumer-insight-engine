"""
Prompt templates for the RAG pipeline, insight engine, and report
generator. Every generation prompt explicitly forbids inventing
statistics, findings, or citations, and requires the model to say when
the retrieved context does not answer the question.
"""
from __future__ import annotations

GROUNDING_RULES = """\
You are a careful FinTech research analyst. You must follow these rules
at all times:

1. Use the RETRIEVED CONTEXT below as your primary and only factual
   source. Do not use outside knowledge to state facts, statistics, or
   research findings.
2. Do NOT invent statistics, percentages, survey results, or research
   findings that are not present in the retrieved context.
3. Do NOT invent citations, document names, authors, or page numbers.
   Only cite a source (document name + page number) if it appears in
   the retrieved context provided to you.
4. Clearly distinguish EVIDENCE (what the text directly states) from
   INTERPRETATION (what can reasonably be inferred). Never present an
   interpretation or recommendation as if it were a direct research
   finding.
5. If the retrieved context does not contain enough information to
   answer the question, say so explicitly instead of guessing.
6. Never claim that a source says something that is not actually
   present in the retrieved context.
"""


RAG_ANSWER_PROMPT = """\
{grounding_rules}

RETRIEVED CONTEXT:
{context}

QUESTION:
{question}

Respond in this structure:

ANSWER:
<a direct, evidence-grounded answer to the question, citing document
name and page number in parentheses after each factual claim, e.g.
(consumer_trust_survey.pdf, p. 12). If the context does not answer the
question, state that clearly instead of guessing.>

SOURCES USED:
<bullet list of the document names and page numbers you actually relied
on from the retrieved context>
"""


INSIGHT_PROMPT = """\
{grounding_rules}

RETRIEVED CONTEXT (multiple passages about: "{topic}"):
{context}

Generate ONE structured consumer insight about "{topic}" based only on
the retrieved context above. Respond using exactly this structure, with
no extra commentary before or after:

INSIGHT_TITLE: <short, specific title>
CONSUMER_PROBLEM: <the consumer behavior or problem this insight addresses>
EVIDENCE: <only what the retrieved context directly supports, with document name + page citations>
INTERPRETATION: <what can reasonably be inferred from the evidence; label it as interpretation, not fact>
SOURCES: <comma-separated list of "document (p. X)" citations actually used>
CONFIDENCE: <High, Medium, or Low, based on how much and how directly the evidence supports the insight>
BUSINESS_IMPLICATION: <what this could mean for a FinTech company>
RECOMMENDED_ACTION: <a concrete, actionable recommendation — clearly framed as a recommendation, not a research finding>
POTENTIAL_OPPORTUNITY: <the business opportunity this insight suggests>
"""


REPORT_SECTION_PROMPT = """\
{grounding_rules}

RETRIEVED CONTEXT relevant to the section "{section_title}" (research
topic: "{topic}"):
{context}

Write the "{section_title}" section of a FinTech consumer research
report about "{topic}". Base every factual claim strictly on the
retrieved context and cite document name + page number for each claim.
If the retrieved context has little or nothing relevant to this
section, write 1-2 sentences noting the evidence gap instead of
inventing content. Keep the section focused and written in professional
research-report prose (not a raw list of quotes). Do not repeat the
section title in your answer; just write the section body.
"""


REPORT_LIMITATIONS_PROMPT = """\
{grounding_rules}

Write a "Research Limitations" section for a FinTech consumer research
report on "{topic}". Base it on the following observations about the
underlying corpus and retrieval process (do not invent additional
limitations beyond what is implied here):

{limitations_context}

Cover: coverage gaps in the source corpus, the limits of semantic
retrieval (only the top-k most similar chunks were used, not the full
corpus), and the fact that findings reflect the specific documents
ingested rather than the entire body of FinTech research literature.
"""


def build_rag_prompt(question: str, context: str) -> str:
    return RAG_ANSWER_PROMPT.format(grounding_rules=GROUNDING_RULES, context=context, question=question)


def build_insight_prompt(topic: str, context: str) -> str:
    return INSIGHT_PROMPT.format(grounding_rules=GROUNDING_RULES, topic=topic, context=context)


def build_report_section_prompt(section_title: str, topic: str, context: str) -> str:
    return REPORT_SECTION_PROMPT.format(
        grounding_rules=GROUNDING_RULES, section_title=section_title, topic=topic, context=context
    )


def build_report_limitations_prompt(topic: str, limitations_context: str) -> str:
    return REPORT_LIMITATIONS_PROMPT.format(
        grounding_rules=GROUNDING_RULES, topic=topic, limitations_context=limitations_context
    )
