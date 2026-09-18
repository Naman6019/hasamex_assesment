"""Optional model-drafted themes and disagreements: only verified content survives."""

import json
import re

import pytest

from core.analyzer import generate_cross_expert_synthesis
from core.parser import parse_transcript


FILES = ["Transcript_1_France.txt", "Transcript_2_Germany.txt", "Transcript_3_UK.txt"]


def transcripts():
    return [parse_transcript(open(f, encoding="utf-8").read(), f) for f in FILES]


def sid(prompt: str, market: str, timestamp: str) -> str:
    match = re.search(rf"^(S\d+) \| {re.escape(market)} \| {timestamp} \|", prompt, re.MULTILINE)
    assert match, f"{market} {timestamp} not in prompt"
    return match.group(1)


class Stub:
    model = "stub-model"

    def __init__(self, build):
        self.build = build
        self.calls = 0
        self.last_prompt = ""

    def is_configured(self):
        return True

    def query(self, system_prompt, user_prompt, temperature=0.0):
        self.calls += 1
        self.last_prompt = user_prompt
        reply = self.build(user_prompt)
        if isinstance(reply, dict) and reply.get("__failure__"):
            return {"success": False, "content": "", "error": "boom"}
        return {"success": True, "content": reply if isinstance(reply, str) else json.dumps(reply)}


def ev(prompt, market, ts, quote):
    return {"source": sid(prompt, market, ts), "quote": quote}


def good_theme(prompt):
    return {
        "title": "Uneven access",
        "summary": "France and Germany both describe growth that is concentrated in larger hospitals.",
        "evidence": [
            ev(prompt, "France", "00:18", "Smaller regional hospitals are much slower."),
            ev(prompt, "Germany", "00:16", "many smaller hospitals are still waiting"),
        ],
    }


def good_disagreement(prompt):
    return {
        "topic": "Is finance decisive?",
        "type": "Purchasing criteria",
        "description": "Germany says the economic case decides, whereas the UK expert says finance is balanced with clinical strategy.",
        "evidence": [
            ev(prompt, "Germany", "02:08", "the economic case decides whether it gets approved"),
            ev(prompt, "United Kingdom", "03:10", "I would not say finance alone decides the purchase."),
        ],
    }


def run(build):
    stub = Stub(build)
    return generate_cross_expert_synthesis(transcripts(), stub), stub


def test_verified_model_items_replace_computed_ones_and_are_marked():
    synthesis, stub = run(lambda p: {"themes": [good_theme(p)], "disagreements": [good_disagreement(p)]})
    assert stub.calls == 1
    assert [t.title for t in synthesis.common_themes] == ["Uneven access"]
    assert [d.topic for d in synthesis.disagreements] == ["Is finance decisive?"]
    assert "stub-model" in synthesis.method and "verified" in synthesis.method.lower()
    assert len(synthesis.comparison_matrix) == 6  # the matrix is always computed


def test_consensus_level_is_computed_from_cited_calls_not_claimed_by_the_model():
    def build(prompt):
        theme = good_theme(prompt)
        theme["consensus_level"] = "Supported by 3 of 3 calls"
        return {"themes": [theme], "disagreements": []}

    synthesis, _ = run(build)
    assert synthesis.common_themes[0].consensus_level == "Supported by 2 of 3 calls"


def test_evidence_keeps_label_timestamp_and_exact_words():
    synthesis, _ = run(lambda p: {"themes": [good_theme(p)], "disagreements": []})
    evidence = synthesis.common_themes[0].supporting_evidence
    assert evidence["France · Dr. Jean Martin"] == "[00:18] Smaller regional hospitals are much slower."
    assert evidence["Germany · Anna Keller"].startswith("[00:16] many smaller hospitals")


def test_fabricated_quote_is_dropped_and_computed_themes_are_used_instead():
    def build(prompt):
        theme = good_theme(prompt)
        theme["evidence"][1]["quote"] = "Everyone in Germany is waiting for new legislation."
        return {"themes": [theme], "disagreements": [good_disagreement(prompt)]}

    synthesis, _ = run(build)
    # The theme lost one of its two calls, so it no longer qualifies; computed themes fill in.
    assert "Uneven access" not in [t.title for t in synthesis.common_themes]
    assert synthesis.common_themes
    assert [d.topic for d in synthesis.disagreements] == ["Is finance decisive?"]
    assert "removed" in synthesis.notes.lower()


def test_theme_supported_by_only_one_call_is_dropped():
    def build(prompt):
        theme = good_theme(prompt)
        theme["evidence"] = theme["evidence"][:1]
        return {"themes": [theme], "disagreements": []}

    synthesis, _ = run(build)
    assert "Uneven access" not in [t.title for t in synthesis.common_themes]


def test_number_in_summary_that_is_not_in_the_quotes_is_dropped():
    def build(prompt):
        theme = good_theme(prompt)
        theme["summary"] = "Both experts say 40 percent of hospitals are waiting."
        return {"themes": [theme], "disagreements": []}

    synthesis, _ = run(build)
    assert "Uneven access" not in [t.title for t in synthesis.common_themes]


def test_summary_naming_a_market_it_does_not_cite_is_dropped():
    def build(prompt):
        theme = good_theme(prompt)
        theme["summary"] = "The UK expert also sees growth concentrated in larger hospitals."
        return {"themes": [theme], "disagreements": []}

    synthesis, _ = run(build)
    assert "Uneven access" not in [t.title for t in synthesis.common_themes]


@pytest.mark.parametrize("reply", ["not json at all", "{broken", {"themes": "x"}, {"themes": [], "disagreements": []}, {"__failure__": True}])
def test_unusable_model_output_falls_back_to_the_computed_synthesis(reply):
    computed = generate_cross_expert_synthesis(transcripts())
    synthesis, _ = run(lambda _p: reply)
    assert [t.title for t in synthesis.common_themes] == [t.title for t in computed.common_themes]
    assert [d.topic for d in synthesis.disagreements] == [d.topic for d in computed.disagreements]
    assert "computed" in synthesis.method.lower()
    assert synthesis.notes


def test_json_in_a_code_fence_is_accepted():
    synthesis, _ = run(lambda p: "```json\n" + json.dumps({"themes": [good_theme(p)], "disagreements": []}) + "\n```")
    assert [t.title for t in synthesis.common_themes] == ["Uneven access"]


def test_prompt_lists_numbered_expert_turns_only_and_forbids_outside_knowledge():
    _, stub = run(lambda _p: {"themes": [], "disagreements": []})
    assert "France | 00:18" in stub.last_prompt
    assert "Interviewer:" not in stub.last_prompt


def test_no_model_is_called_without_a_configured_service():
    synthesis = generate_cross_expert_synthesis(transcripts(), None)
    assert "computed" in synthesis.method.lower()
