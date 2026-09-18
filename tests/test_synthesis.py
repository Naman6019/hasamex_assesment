"""Cross-call synthesis must be computed from whatever transcripts are loaded."""

import re
from pathlib import Path

import pytest

from core.analyzer import generate_cross_expert_synthesis
from core.extractor import INTERVIEW_QUESTIONS
from core.parser import parse_transcript


FILES = ["Transcript_1_France.txt", "Transcript_2_Germany.txt", "Transcript_3_UK.txt"]


def case_transcripts():
    return [parse_transcript(open(f, encoding="utf-8").read(), f) for f in FILES]


ITALY = """Expert 4 – Dr. Luca Rossi
Role: Chief of Surgery
Market: Italy

00:00
Interviewer: How would you describe robotic surgery adoption in Italy today?

00:12
Dr. Rossi: Adoption is growing, but regional hospitals in the south are far behind the northern university centres.

01:00
Interviewer: What are the main barriers?

01:06
Dr. Rossi: Reimbursement is the main barrier. Regional health authorities decide the budget, and training capacity is thin.

02:00
Interviewer: What outlook do you expect over the next three to five years?

02:05
Dr. Rossi: I expect procedure volumes to grow around 30 percent per year in the north, driven by new regional funding.

03:00
Interviewer: How long does a purchase decision normally take?

03:04
Dr. Rossi: Twelve to twenty-four months is typical because the regional authority must approve the tender.
"""


def italy():
    return parse_transcript(ITALY, "Transcript_4_Italy.txt")


def cell_quote(cell: str) -> str:
    match = re.match(r"^\[(\d\d:\d\d)\]\s+(.*)$", cell, re.DOTALL)
    assert match, cell
    return match.group(2)


# ---------------------------------------------------------------- comparison


def test_matrix_has_a_row_per_guide_question_and_a_column_per_market():
    synthesis = generate_cross_expert_synthesis(case_transcripts())
    assert [row.dimension for row in synthesis.comparison_matrix] == [q["short_label"] for q in INTERVIEW_QUESTIONS]
    for row in synthesis.comparison_matrix:
        assert set(row.cells) == {"France", "Germany", "United Kingdom"}


def test_matrix_cells_are_timestamped_verbatim_quotes():
    transcripts = case_transcripts()
    raw = {t.profile.market: t.raw_text for t in transcripts}
    for row in generate_cross_expert_synthesis(transcripts).comparison_matrix:
        for market, cell in row.cells.items():
            assert cell_quote(cell) in raw[market]


def test_adding_a_transcript_adds_a_column_with_its_own_quotes():
    synthesis = generate_cross_expert_synthesis(case_transcripts() + [italy()])
    for row in synthesis.comparison_matrix:
        assert "Italy" in row.cells
        assert cell_quote(row.cells["Italy"]) in ITALY
    timeline = next(row for row in synthesis.comparison_matrix if row.dimension == "Purchase timeline")
    assert "Twelve to twenty-four months" in timeline.cells["Italy"]


def test_market_names_follow_the_transcript_headers():
    text = open(FILES[0], encoding="utf-8").read().replace("Market: France", "Market: Spain")
    spain = parse_transcript(text, "Transcript_Spain.txt")
    synthesis = generate_cross_expert_synthesis([spain] + case_transcripts()[1:])
    assert set(synthesis.comparison_matrix[0].cells) == {"Spain", "Germany", "United Kingdom"}


# -------------------------------------------------------------------- themes


def test_themes_recur_across_calls_and_counts_match_their_evidence():
    transcripts = case_transcripts()
    synthesis = generate_cross_expert_synthesis(transcripts)
    assert len(synthesis.common_themes) >= 3
    for theme in synthesis.common_themes:
        assert len(theme.supporting_evidence) >= 2
        assert theme.consensus_level == f"Supported by {len(theme.supporting_evidence)} of 3 calls"


def test_case_data_surfaces_training_and_capital_topics():
    synthesis = generate_cross_expert_synthesis(case_transcripts())
    text = " ".join(f"{t.title} {t.summary}" for t in synthesis.common_themes).lower()
    assert "train" in text
    assert "capital" in text


def test_consensus_counts_use_the_number_of_loaded_calls():
    two = case_transcripts()[:2]
    synthesis = generate_cross_expert_synthesis(two)
    assert synthesis.common_themes
    assert all(t.consensus_level.endswith("of 2 calls") for t in synthesis.common_themes)


def test_a_single_call_has_no_cross_call_themes_or_disagreements():
    synthesis = generate_cross_expert_synthesis(case_transcripts()[:1])
    assert synthesis.common_themes == []
    assert synthesis.disagreements == []
    assert "one call" in synthesis.executive_summary.lower()


def test_theme_evidence_is_verbatim_source_text_with_timestamp():
    transcripts = case_transcripts() + [italy()]
    raw = "\n".join(t.raw_text for t in transcripts)
    for theme in generate_cross_expert_synthesis(transcripts).common_themes:
        for label, evidence in theme.supporting_evidence.items():
            assert " · " in label
            assert cell_quote(evidence) in raw


# ------------------------------------------------------------- disagreements


def test_differing_figures_are_reported_as_disagreements():
    synthesis = generate_cross_expert_synthesis(case_transcripts())
    topics = {d.topic for d in synthesis.disagreements}
    assert any("3–5 year outlook" in topic for topic in topics)
    assert any("Purchase timeline" in topic for topic in topics)
    raw = "\n".join(t.raw_text for t in case_transcripts())
    for disagreement in synthesis.disagreements:
        assert len(disagreement.expert_positions) >= 2
        for evidence in disagreement.expert_positions.values():
            assert cell_quote(evidence) in raw


def test_a_new_call_with_new_figures_joins_the_disagreement():
    synthesis = generate_cross_expert_synthesis(case_transcripts() + [italy()])
    timeline = next(d for d in synthesis.disagreements if "Purchase timeline" in d.topic)
    assert any("Italy" in label for label in timeline.expert_positions)
    assert any("twenty-four" in evidence for evidence in timeline.expert_positions.values())


def test_identical_figures_do_not_create_a_disagreement():
    france = case_transcripts()[0]
    clone_text = france.raw_text.replace("Market: France", "Market: Belgium").replace("Jean Martin", "Marc Dupont")
    belgium = parse_transcript(clone_text, "Transcript_Belgium.txt")
    synthesis = generate_cross_expert_synthesis([france, belgium])
    assert synthesis.disagreements == []


# --------------------------------------------------------- executive summary


def test_executive_summary_is_built_from_the_loaded_calls():
    base = generate_cross_expert_synthesis(case_transcripts()).executive_summary
    extended = generate_cross_expert_synthesis(case_transcripts() + [italy()]).executive_summary
    assert "3 expert calls" in base and "Italy" not in base
    assert "4 expert calls" in extended and "Italy" in extended
    assert "outlook" in base.lower() or "timeline" in base.lower()


def test_synthesis_reports_how_it_was_produced():
    synthesis = generate_cross_expert_synthesis(case_transcripts())
    assert "computed" in synthesis.method.lower()


def test_analyzer_source_contains_no_hardcoded_markets_experts_or_quotes():
    source = Path("core/analyzer.py").read_text(encoding="utf-8")
    for needle in ["France", "Germany", "United Kingdom", "Martin", "Keller", "Carter", "capital budget"]:
        assert needle not in source, needle
