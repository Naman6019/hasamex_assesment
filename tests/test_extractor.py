"""
Tests for Question Extractor & Quote Verification.
"""

import pytest
from core.parser import parse_transcript
from core.extractor import ExpertAnswerExtractor, verify_quote, INTERVIEW_QUESTIONS


def load_all_transcripts():
    files = ["Transcript_1_France.txt", "Transcript_2_Germany.txt", "Transcript_3_UK.txt"]
    transcripts = []
    for f in files:
        with open(f, "r", encoding="utf-8") as file:
            transcripts.append(parse_transcript(file.read(), f))
    return transcripts


def test_quote_verification():
    sample_text = "Hospitals want several surgeons trained so utilisation is high enough."
    is_v, score, start, end = verify_quote("surgeons trained so utilisation", sample_text)
    assert is_v is True
    assert score >= 0.95
    assert start is not None

    # Test nonexistent quote
    is_v_fake, _, _, _ = verify_quote("completely fabricated quote about robots", sample_text)
    assert is_v_fake is False


def test_all_six_questions_extracted_and_verified():
    transcripts = load_all_transcripts()
    extractor = ExpertAnswerExtractor()

    assert len(transcripts) == 3
    assert len(INTERVIEW_QUESTIONS) == 6

    for tr in transcripts:
        answers = extractor.get_grounded_expert_answers(tr)
        assert len(answers) == 6, f"Expert {tr.profile.name} did not have all 6 questions answered."

        for ans in answers:
            assert ans.summary_answer, f"Missing summary for Q{ans.question_id}"
            assert len(ans.citations) > 0, f"No citations provided for Q{ans.question_id}"
            
            for cit in ans.citations:
                assert cit.timestamp != "", f"Missing timestamp for citation in Q{ans.question_id}"
                assert cit.quote_text != "", f"Empty quote in citation for Q{ans.question_id}"
                assert cit.is_verified is True, (
                    f"Quote failed verbatim verification: '{cit.quote_text}' in {tr.profile.name}'s transcript"
                )
                assert cit.verification_score >= 0.9
