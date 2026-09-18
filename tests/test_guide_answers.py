"""Guide answers must surface the turn a reviewer would cite for each question."""

import pytest

from core.extractor import ExpertAnswerExtractor, verify_quote
from core.parser import parse_transcript


FILES = {
    "France": "Transcript_1_France.txt",
    "Germany": "Transcript_2_Germany.txt",
    "United Kingdom": "Transcript_3_UK.txt",
}

# Timestamps that must appear in the cited evidence, read off the transcripts by hand.
GOLD = {
    "France": {1: {"00:18"}, 2: {"01:20"}, 3: {"02:18"}, 4: {"03:10", "04:08"}, 5: {"05:07"}, 6: {"06:08"}},
    "Germany": {1: {"00:16"}, 2: {"01:10"}, 3: {"02:08"}, 4: {"03:05"}, 5: {"05:08"}, 6: {"06:05"}},
    "United Kingdom": {1: {"00:14"}, 2: {"01:05"}, 3: {"02:07"}, 4: {"01:05"}, 5: {"04:06"}, 6: {"05:04"}},
}


@pytest.fixture(scope="module")
def answers():
    extractor = ExpertAnswerExtractor()
    result = {}
    for market, filename in FILES.items():
        transcript = parse_transcript(open(filename, encoding="utf-8").read(), filename)
        result[market] = (transcript, {a.question_id: a for a in extractor.get_grounded_expert_answers(transcript)})
    return result


@pytest.mark.parametrize("market", FILES)
@pytest.mark.parametrize("question_id", [1, 2, 3, 4, 5, 6])
def test_expected_turns_are_cited(answers, market, question_id):
    _, by_question = answers[market]
    cited = {c.timestamp for c in by_question[question_id].citations}
    assert GOLD[market][question_id] <= cited, f"{market} Q{question_id} cited {sorted(cited)}"


@pytest.mark.parametrize("market", FILES)
@pytest.mark.parametrize("question_id", [1, 2, 3, 4, 5, 6])
def test_no_more_than_two_citations_per_answer(answers, market, question_id):
    _, by_question = answers[market]
    assert 1 <= len(by_question[question_id].citations) <= 2


def test_each_citation_has_an_exact_substring_highlight(answers):
    for market, (transcript, by_question) in answers.items():
        for answer in by_question.values():
            for citation in answer.citations:
                assert citation.highlight, f"{market} Q{answer.question_id} {citation.timestamp}"
                assert citation.highlight in citation.quote_text
                assert verify_quote(citation.highlight, transcript.raw_text)[0]


def test_numeric_forecast_and_timeline_highlights_carry_the_figures(answers):
    germany = answers["Germany"][1]
    forecast = next(c for c in germany[5].citations if c.timestamp == "05:08")
    assert "high single digits" in forecast.highlight
    timeline = answers["United Kingdom"][1][6].citations[0]
    assert "six to nine months" in timeline.highlight.lower()
