"""Regression tests for the application's source-grounding contract."""

import re

from core.analyzer import generate_cross_expert_synthesis
from core.extractor import ExpertAnswerExtractor, verify_quote
from core.parser import parse_transcript
from core.qa_engine import CrossTranscriptQA


def load_transcripts():
    filenames = [
        "Transcript_1_France.txt",
        "Transcript_2_Germany.txt",
        "Transcript_3_UK.txt",
    ]
    return [
        parse_transcript(open(filename, encoding="utf-8").read(), filename)
        for filename in filenames
    ]


def test_guide_answers_are_source_turns_with_market_and_timestamp():
    for transcript in load_transcripts():
        answers = ExpertAnswerExtractor().get_grounded_expert_answers(transcript)
        assert len(answers) == 6
        for answer in answers:
            assert answer.citations
            for citation in answer.citations:
                assert citation.market == transcript.profile.market
                assert citation.timestamp != "00:00"
                assert verify_quote(citation.quote_text, transcript.raw_text)[0]


def test_comparative_timeline_query_covers_all_markets_with_verified_quotes():
    transcripts = load_transcripts()
    response = CrossTranscriptQA(transcripts).answer_query(
        "Compare the purchasing decision timelines across France, Germany, and the UK."
    )

    assert {citation.market for citation in response.citations} == {
        "France",
        "Germany",
        "United Kingdom",
    }
    assert all(citation.is_verified for citation in response.citations)
    assert all("month" in citation.quote_text.lower() for citation in response.citations)


def test_synthesis_evidence_is_verbatim_source_text():
    transcripts = load_transcripts()
    source_text = "\n".join(transcript.raw_text for transcript in transcripts)
    synthesis = generate_cross_expert_synthesis(transcripts)
    evidence_sets = [
        theme.supporting_evidence for theme in synthesis.common_themes
    ] + [disagreement.expert_positions for disagreement in synthesis.disagreements]

    for evidence in evidence_sets:
        for labelled_quote in evidence.values():
            quote = re.sub(r"^\[\d{2}:\d{2}\]\s+", "", labelled_quote)
            assert quote in source_text
