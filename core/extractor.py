"""Grounded extraction for the interview guide.

The extractor never writes a factual claim of its own. It selects the expert's
timestamped answer and verifies that the displayed text exists in the source.
"""

import re
from typing import Dict, List, Optional, Tuple

from core.models import ExpertTranscript, InterviewAnswer, QuoteCitation, SpeakerTurn
from core.textutil import stem_token, tokenize


INTERVIEW_QUESTIONS = [
    {
        "id": 1,
        "text": "How would you describe current adoption of robotic surgery in your market?",
        "short_label": "Current adoption",
        "category": "Market landscape",
    },
    {
        "id": 2,
        "text": "What are the main barriers to adoption?",
        "short_label": "Adoption barriers",
        "category": "Market obstacles",
    },
    {
        "id": 3,
        "text": "How important are hospital budgets and ROI in purchasing decisions?",
        "short_label": "Budgets and ROI",
        "category": "Economics and purchasing",
    },
    {
        "id": 4,
        "text": "How important are surgeon training and clinical outcomes?",
        "short_label": "Training and outcomes",
        "category": "Clinical operations",
    },
    {
        "id": 5,
        "text": "What adoption trend do you expect over the next 3–5 years?",
        "short_label": "3–5 year outlook",
        "category": "Market forecast",
    },
    {
        "id": 6,
        "text": "What is the typical hospital decision-making timeline for purchasing a new robotic system?",
        "short_label": "Purchase timeline",
        "category": "Procurement",
    },
]


# These concept groups route an interview-guide question to a turn; they do not
# create content. Each group is a set of interchangeable words (matched as whole
# words after light stemming, or as a phrase). The answer shown to users is
# always the original expert text.
QUESTION_SIGNALS: Dict[int, Tuple[Tuple[str, ...], ...]] = {
    1: (("adoption",), ("access",), ("standard",), ("concentrated",), ("growing", "increasing")),
    2: (("barrier",), ("holding",), ("cost",), ("funding",), ("capital",), ("approval",), ("stalls",)),
    3: (("roi",), ("economic", "economics"), ("financial",), ("budget",), ("utilisation",), ("total cost",)),
    4: (("training",), ("surgeon",), ("clinical",), ("outcomes",), ("theatre",), ("sustainable",)),
    5: (("expect", "expectation"), ("outlook",), ("growth",), ("increase", "increasing"), ("accelerate",), ("years",)),
    6: (("how long",), ("timeline",), ("months",), ("purchase",), ("decision",), ("cycle",), ("procurement",)),
}

# Forecast and duration questions are best answered by the passage that gives a figure.
QUANTIFIED = {
    5: re.compile(r"\d|percent|digits", re.IGNORECASE),
    6: re.compile(r"\d|\bmonths?\b|\bweeks?\b|\byears?\b", re.IGNORECASE),
}
QUANTIFIED_BONUS = 3
SECOND_CITATION_RATIO = 0.6  # a second turn is cited only if nearly as relevant as the first
MAX_CITATIONS = 2


def _concepts_matched(text: str, question_id: int) -> int:
    """Count concept groups mentioned in ``text`` (whole words, stemmed; or phrases)."""
    words = tokenize(text)
    stems = {stem_token(word) for word in words}
    phrase_text = " ".join(words)
    matched = 0
    for group in QUESTION_SIGNALS[question_id]:
        for option in group:
            if (" " in option and option in phrase_text) or (" " not in option and stem_token(option) in stems):
                matched += 1
                break
    return matched


def _relevance(text: str, question_id: int) -> int:
    score = _concepts_matched(text, question_id)
    quantified = QUANTIFIED.get(question_id)
    if quantified and quantified.search(text):
        score += QUANTIFIED_BONUS
    return score


def split_sentences(text: str) -> List[str]:
    return [sentence for sentence in re.split(r"(?<=[.!?])\s+", text.strip()) if sentence]


def pick_highlight(text: str, question_id: int) -> str:
    """The sentence(s) of ``text`` that best answer the question, as an exact substring."""
    sentences = split_sentences(text)
    if len(sentences) < 2:
        return text.strip()
    best = max(range(len(sentences)), key=lambda i: (_relevance(sentences[i], question_id), -i))
    return sentences[best]


def verify_quote(quote: str, raw_text: str) -> Tuple[bool, float, Optional[int], Optional[int]]:
    """Return whether ``quote`` is an exact source substring, allowing whitespace changes."""
    clean_quote = quote.strip(' "\'“”’‘')
    if not clean_quote:
        return False, 0.0, None, None

    exact_position = raw_text.find(clean_quote)
    if exact_position != -1:
        return True, 1.0, exact_position, exact_position + len(clean_quote)

    normalised_source = re.sub(r"\s+", " ", raw_text).strip()
    normalised_quote = re.sub(r"\s+", " ", clean_quote).strip()
    if normalised_source.find(normalised_quote) != -1:
        # A normalised offset is not a valid raw-text offset, so omit it.
        return True, 0.98, None, None

    return False, 0.0, None, None


class ExpertAnswerExtractor:
    """Finds evidence-backed answers for the six standard interview questions."""

    def get_grounded_expert_answers(self, transcript: ExpertTranscript) -> List[InterviewAnswer]:
        return [self._answer_for_question(transcript, question) for question in INTERVIEW_QUESTIONS]

    def _answer_for_question(
        self, transcript: ExpertTranscript, question: Dict[str, object]
    ) -> InterviewAnswer:
        question_id = int(question["id"])
        turns = self._rank_expert_turns(transcript, question_id)
        citations = [self._citation_from_turn(turn, transcript, question_id) for turn in turns]

        if citations:
            answer = citations[0].highlight or citations[0].quote_text
            if len(citations) > 1:
                answer += "\n\nAdditional directly relevant evidence is shown below."
        else:
            answer = "No directly relevant answer was found in this transcript."

        return InterviewAnswer(
            question_id=question_id,
            question_text=str(question["text"]),
            expert_id=transcript.profile.id,
            expert_name=transcript.profile.name,
            market=transcript.profile.market,
            summary_answer=answer,
            citations=citations,
        )

    def _rank_expert_turns(self, transcript: ExpertTranscript, question_id: int) -> List[SpeakerTurn]:
        scored: List[Tuple[int, SpeakerTurn]] = []
        for index, turn in enumerate(transcript.turns):
            if turn.is_interviewer:
                continue
            previous_prompt = ""
            if index > 0 and transcript.turns[index - 1].is_interviewer:
                previous_prompt = transcript.turns[index - 1].text

            # An answer to a question about the topic counts for more than an
            # answer that merely mentions it.
            score = _concepts_matched(previous_prompt, question_id) * 4 + _relevance(turn.text, question_id)
            if score:
                scored.append((score, turn))

        scored.sort(key=lambda item: (-item[0], item[1].timestamp_seconds))
        if not scored:
            return []
        top = scored[0][0]
        # A second turn must be nearly as relevant AND mention the topic in its
        # own words, not merely follow an interviewer question about it.
        return [
            turn
            for rank, (score, turn) in enumerate(scored[:MAX_CITATIONS])
            if rank == 0 or (score >= SECOND_CITATION_RATIO * top and _concepts_matched(turn.text, question_id))
        ]

    @staticmethod
    def _citation_from_turn(turn: SpeakerTurn, transcript: ExpertTranscript, question_id: int) -> QuoteCitation:
        verified, score, char_start, char_end = verify_quote(turn.text, transcript.raw_text)
        return QuoteCitation(
            quote_text=turn.text,
            highlight=pick_highlight(turn.text, question_id),
            timestamp=turn.timestamp,
            timestamp_seconds=turn.timestamp_seconds,
            speaker=turn.speaker,
            is_verified=verified,
            verification_score=score,
            char_start=char_start,
            char_end=char_end,
            market=transcript.profile.market,
            transcript_id=transcript.profile.id,
        )
