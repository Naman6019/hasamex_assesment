"""The optional LLM answer step: the model may write, but only verified claims survive."""

import json
import re

import pytest

from core.parser import parse_transcript
from core.qa_engine import CrossTranscriptQA


def load_transcripts():
    filenames = ["Transcript_1_France.txt", "Transcript_2_Germany.txt", "Transcript_3_UK.txt"]
    return [parse_transcript(open(f, encoding="utf-8").read(), f) for f in filenames]


def source_id(prompt: str, market: str, timestamp: str) -> str:
    """Find the S-label the engine gave a (market, timestamp) source."""
    match = re.search(rf"^(S\d+) \| {re.escape(market)} \| {timestamp} \|", prompt, re.MULTILINE)
    assert match, f"{market} {timestamp} not among the candidate sources:\n{prompt}"
    return match.group(1)


class StubModel:
    """Builds its reply from the prompt it receives, like a real model would."""

    model = "stub-model"

    def __init__(self, build_reply):
        self.build_reply = build_reply
        self.calls = 0

    def is_configured(self):
        return True

    def query(self, system_prompt, user_prompt, temperature=0.0):
        self.calls += 1
        reply = self.build_reply(user_prompt)
        if isinstance(reply, dict) and reply.get("__failure__"):
            return {"success": False, "content": "", "error": "boom"}
        return {"success": True, "content": reply if isinstance(reply, str) else json.dumps(reply)}


TIMELINE_QUERY = "What is the typical purchase timeline in Germany?"
FRANCE_TIMELINE_QUERY = "What is the typical purchase timeline in France?"


def ask(build_reply, query=TIMELINE_QUERY):
    model = StubModel(build_reply)
    response = CrossTranscriptQA(load_transcripts(), model).answer_query(query)
    return response, model


def germany_reply(claim_text, quote, market="Germany", timestamp="06:05", source_market=None):
    def build(prompt):
        return {
            "supported": True,
            "claims": [
                {
                    "text": claim_text,
                    "evidence": [{"source": source_id(prompt, source_market or market, timestamp), "quote": quote}],
                }
            ],
        }

    return build


def test_verified_claim_is_shown_with_its_exact_quote_and_timestamp():
    response, model = ask(
        germany_reply("The German expert says purchases commonly take nine to eighteen months.", "Nine to eighteen months is common.")
    )
    assert model.calls == 1
    assert "stub-model" in response.mode_used
    assert "nine to eighteen months" in response.answer.lower()
    assert "Germany · 06:05" in response.answer
    [citation] = response.citations
    assert citation.quote_text == "Nine to eighteen months is common."
    assert citation.timestamp == "06:05" and citation.market == "Germany"
    assert citation.is_verified
    raw = load_transcripts()[1].raw_text
    assert raw[citation.char_start : citation.char_end] == citation.quote_text


def test_fabricated_quote_is_dropped_and_falls_back_to_source_passages():
    response, _ = ask(germany_reply("Purchases take about a year.", "Purchases usually take about a year."))
    assert "verification" in response.mode_used.lower()
    assert "about a year" not in response.answer.lower()
    assert response.citations and all(c.is_verified for c in response.citations)
    assert "Nine to eighteen months" in response.answer  # the deterministic passage instead


def test_real_quote_cited_against_the_wrong_source_is_dropped():
    # The words exist in the Germany transcript, but the claim cites the France source.
    def build(prompt):
        return {
            "supported": True,
            "claims": [
                {
                    "text": "Purchases take nine to eighteen months.",
                    "evidence": [{"source": source_id(prompt, "France", "06:08"), "quote": "Nine to eighteen months is common."}],
                }
            ],
        }

    response, _ = ask(build, "Compare the purchase timeline in Germany and France.")
    assert "verification" in response.mode_used.lower()


def test_number_not_in_the_cited_quote_drops_the_claim():
    response, _ = ask(
        germany_reply("Purchases take 12 to 24 months.", "Nine to eighteen months is common.")
    )
    assert "verification" in response.mode_used.lower()
    assert "24" not in response.answer


def test_claim_naming_a_market_it_does_not_cite_is_dropped():
    # Cites the France source but talks about Germany: a misattribution.
    response, _ = ask(
        germany_reply(
            "In Germany the purchase takes six to twelve months.",
            "Six to twelve months is realistic once the hospital becomes serious.",
            market="France",
            timestamp="06:08",
        ),
        "Compare the purchase timeline in Germany and France.",
    )
    assert "verification" in response.mode_used.lower()
    assert "In Germany the purchase takes" not in response.answer


def test_good_claim_survives_when_another_claim_fails_and_confidence_drops():
    def build(prompt):
        return {
            "supported": True,
            "claims": [
                {
                    "text": "Germany reports nine to eighteen months.",
                    "evidence": [{"source": source_id(prompt, "Germany", "06:05"), "quote": "Nine to eighteen months is common."}],
                },
                {
                    "text": "Germany also says approvals are automatic.",
                    "evidence": [{"source": source_id(prompt, "Germany", "06:05"), "quote": "Approvals are automatic."}],
                },
            ],
        }

    baseline = CrossTranscriptQA(load_transcripts()).answer_query(TIMELINE_QUERY)
    response, _ = ask(build)
    assert "nine to eighteen months" in response.answer.lower()
    assert "automatic" not in response.answer.lower()
    assert "1 claim" in response.notes and "removed" in response.notes.lower()
    order = ["None", "Low", "Medium", "High"]
    assert order.index(response.confidence) == order.index(baseline.confidence) - 1


def test_model_can_judge_the_evidence_insufficient():
    # Keyword retrieval finds "capital"; the model recognises it does not answer the question.
    response, model = ask(
        lambda _p: {"supported": False, "reason": "The passages discuss capital budgets, not a city."},
        "What is the capital of France?",
    )
    assert model.calls == 1
    assert response.citations == []
    assert response.confidence == "None"
    assert "capital budgets" in response.notes


@pytest.mark.parametrize(
    "reply",
    ["I think it is about a year.", "{not json", '{"supported": true}', '{"supported": true, "claims": "x"}', {"__failure__": True}],
)
def test_unusable_model_output_falls_back_to_source_passages(reply):
    response, _ = ask(lambda _p: reply)
    assert response.citations and all(c.is_verified for c in response.citations)
    assert "Nine to eighteen months" in response.answer
    assert "source passages" in response.notes.lower()


def test_json_wrapped_in_a_code_fence_is_accepted():
    def build(prompt):
        payload = germany_reply("Nine to eighteen months is common in Germany.", "Nine to eighteen months is common.")(prompt)
        return "```json\n" + json.dumps(payload) + "\n```"

    response, _ = ask(build)
    assert response.citations[0].quote_text == "Nine to eighteen months is common."
    assert "stub-model" in response.mode_used


def test_model_is_not_called_when_retrieval_already_refuses():
    response, model = ask(lambda _p: {"supported": True, "claims": []}, "What do experts say about reimbursement?")
    assert model.calls == 0
    assert response.mode_used == "Guardrail refusal"


def test_prompt_marks_sources_and_forbids_outside_knowledge():
    seen = {}

    class Recorder(StubModel):
        def query(self, system_prompt, user_prompt, temperature=0.0):
            seen["system"], seen["user"] = system_prompt, user_prompt
            return super().query(system_prompt, user_prompt, temperature)

    model = Recorder(lambda _p: {"supported": False, "reason": "n/a"})
    CrossTranscriptQA(load_transcripts(), model).answer_query(TIMELINE_QUERY)
    assert "only" in seen["system"].lower() and "verbatim" in seen["system"].lower()
    assert "Germany | 06:05" in seen["user"]
