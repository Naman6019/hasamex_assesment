"""
Tests for Cross-Transcript QA and Synthesis.
"""

import pytest
from core.parser import parse_transcript
from core.qa_engine import CrossTranscriptQA
from core.analyzer import generate_cross_expert_synthesis


class StubLocalSelector:
    """A model that ignores the JSON contract and replies with loose labels."""

    model = "test-local-model"

    def is_configured(self):
        return True

    def query(self, *_args, **_kwargs):
        return {"success": True, "content": "S1, S3, not-a-source"}


def load_all_transcripts():
    files = ["Transcript_1_France.txt", "Transcript_2_Germany.txt", "Transcript_3_UK.txt"]
    transcripts = []
    for f in files:
        with open(f, "r", encoding="utf-8") as file:
            transcripts.append(parse_transcript(file.read(), f))
    return transcripts


def test_cross_expert_synthesis():
    transcripts = load_all_transcripts()
    synthesis = generate_cross_expert_synthesis(transcripts)

    assert len(synthesis.common_themes) >= 3
    assert len(synthesis.disagreements) >= 3
    assert len(synthesis.comparison_matrix) == 6
    assert "adoption" in synthesis.executive_summary.lower()


def test_grounded_qa_retrieval():
    transcripts = load_all_transcripts()
    qa_engine = CrossTranscriptQA(transcripts)

    # Query about training
    resp = qa_engine.answer_query("How does surgeon training affect the business case?")
    assert resp.confidence == "High"
    assert len(resp.citations) > 0
    assert any("training" in c.quote_text.lower() or "surgeon" in c.quote_text.lower() for c in resp.citations)
    assert all(c.timestamp != "" for c in resp.citations)

    # Query about timelines
    resp_timeline = qa_engine.answer_query("What are the purchasing decision timelines in Germany and UK?")
    assert len(resp_timeline.citations) > 0
    assert any("months" in c.quote_text.lower() for c in resp_timeline.citations)


def test_qa_refusal_on_irrelevant_query():
    transcripts = load_all_transcripts()
    qa_engine = CrossTranscriptQA(transcripts)

    resp = qa_engine.answer_query("What is the recipe for chocolate cake with strawberry frosting?")
    assert "refuse" in resp.mode_used.lower() or "refuses" in resp.answer.lower()


def test_unstructured_model_output_never_reaches_the_user():
    transcripts = load_all_transcripts()
    response = CrossTranscriptQA(transcripts, StubLocalSelector()).answer_query(
        "Compare the purchasing decision timelines across France, Germany, and the UK."
    )

    # "S1, S3, not-a-source" is not the required JSON, so the app shows source passages only.
    assert "failed verification" in response.mode_used
    assert {citation.market for citation in response.citations} == {"France", "Germany", "United Kingdom"}
    assert all(citation.is_verified for citation in response.citations)
    assert "not-a-source" not in response.answer
