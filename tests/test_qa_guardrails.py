"""Relevance guardrails for cross-transcript Q&A.

These pin the behaviours that the keyword retriever originally got wrong:
confident answers to out-of-scope questions, substring matches ("any" inside
"Germany"), and a hardcoded "High" confidence.
"""

import pytest

from core.parser import parse_transcript
from core.qa_engine import CrossTranscriptQA, stem_token


def load_transcripts():
    filenames = ["Transcript_1_France.txt", "Transcript_2_Germany.txt", "Transcript_3_UK.txt"]
    return [parse_transcript(open(f, encoding="utf-8").read(), f) for f in filenames]


@pytest.fixture(scope="module")
def qa():
    return CrossTranscriptQA(load_transcripts())


@pytest.mark.parametrize(
    "query",
    [
        "Do any experts mention Intuitive Surgical or da Vinci pricing?",
        "What do experts say about reimbursement?",
        "What is the recipe for chocolate cake with strawberry frosting?",
        "How many robots are installed in Spain?",
    ],
)
def test_out_of_scope_questions_are_refused(qa, query):
    response = qa.answer_query(query)
    assert response.mode_used == "Guardrail refusal"
    assert response.citations == []
    assert response.confidence == "None"
    # The refusal should say what was missing rather than just declining.
    assert response.notes


def test_refusal_names_terms_missing_from_the_transcripts(qa):
    response = qa.answer_query("What do experts say about reimbursement?")
    assert "reimbursement" in response.notes.lower()


def test_single_keyword_coincidence_is_never_high_confidence(qa):
    # "capital" only appears in the transcripts as "capital budget".
    response = qa.answer_query("What is the capital of France?")
    assert response.confidence in {"Low", "None"}
    if response.citations:
        assert "keyword" in response.notes.lower()


def test_confidence_is_computed_not_hardcoded(qa):
    strong = qa.answer_query("How does surgeon training affect the hospital business case?")
    weak = qa.answer_query("Which country has the fastest adoption?")
    assert strong.confidence in {"High", "Medium"}
    assert weak.confidence in {"Low", "None"}
    assert strong.confidence != weak.confidence


def test_words_do_not_match_inside_other_words(qa):
    # "any" is a substring of "Germany"; it must not count as a topical match.
    assert qa.search_relevant_turns("any") == []


def test_named_market_restricts_results_to_that_market(qa):
    response = qa.answer_query("What are the barriers to adoption in the UK?")
    assert response.citations
    assert {c.market for c in response.citations} == {"United Kingdom"}


def test_named_expert_restricts_results_to_that_expert(qa):
    response = qa.answer_query("What did Anna Keller say about training?")
    assert response.citations
    assert {c.market for c in response.citations} == {"Germany"}
    assert any(c.timestamp == "03:05" for c in response.citations)


def test_suggested_prompts_in_the_ui_still_answer(qa):
    prompts = [
        "How does surgeon training affect the economics and hospital business case?",
        "Compare the purchasing decision timelines across France, Germany, and the UK.",
        "How do clinical outcomes weigh against financial ROI when purchasing committees decide?",
        "Why is adoption concentrated in larger hospitals rather than smaller hospitals?",
    ]
    for prompt in prompts:
        response = qa.answer_query(prompt)
        assert response.citations, prompt
        assert response.confidence in {"High", "Medium"}, (prompt, response.confidence)


def test_stemmer_handles_plurals_and_silent_e():
    assert stem_token("timelines") == stem_token("timeline")
    assert stem_token("increasing") == stem_token("increase")
    assert stem_token("purchasing") == stem_token("purchase")
    assert stem_token("economics") == stem_token("economic")
    assert stem_token("process") == "process"
