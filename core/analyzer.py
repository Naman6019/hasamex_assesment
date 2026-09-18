"""Cross-call synthesis computed from whichever transcripts are loaded.

Nothing here is written by hand about a particular call. Every output is derived
from the transcripts and carries verbatim, timestamped quotes:

- comparison matrix: each expert's cited answer to every guide question;
- themes: topics that recur in expert answers across a majority of the calls;
- disagreements: guide questions where experts quote figures that do not match;
- executive summary: a template filled from the results above.

With a configured model, ``core/synthesis_model.py`` can draft richer themes and
disagreements; every quote it returns is verified before it is shown, and the
computed results are the fallback.
"""

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from core.answer_generator import figures
from core.extractor import INTERVIEW_QUESTIONS, ExpertAnswerExtractor, split_sentences
from core.models import (
    ComparisonMetric,
    CrossExpertDisagreement,
    CrossExpertSynthesis,
    CrossExpertTheme,
    ExpertTranscript,
    InterviewAnswer,
)
from core.synthesis_model import ModelDraft, draft_synthesis
from core.textutil import stem_token, tokenize


# Everyday function words plus interview filler. Topic detection ignores these.
STOPWORDS = set(
    """
    a about above after again all also am an and any are as at be because been before being below
    between both but by can could did do does doing down during each either else enough even ever
    few for from further had has have having he her here hers him his how i if in into is it its
    just less like look looks made make many may maybe me might more most much must my need needs
    no nor not now of off often on once one only or other our out over own quite rather really
    same say says see seem seems several she should so some still such take than that the their
    them then there these they think this those though through to too under until up us use used
    very want was we well were what when where whether which while who whom why will with within
    would yes yet you your first second third get gets got going know matter matters important
    probably actually especially typically usually possible able way lot lots kind sort thing things
    """.split()
)
STOP_STEMS = {stem_token(word) for word in STOPWORDS}

MIN_CALL_SHARE = 0.5      # a theme must recur in at least half the calls (and never fewer than two)
MAX_TOPIC_SHARE = 0.5     # a word in over half of all expert turns is background, not a topic
MIN_TERM_LENGTH = 3
CLUSTER_OVERLAP = 0.5     # terms whose supporting turns overlap this much form one theme
MAX_THEMES = 6
MERGE_OVERLAP = 0.6       # themes citing this share of the same passages are merged


def call_labels(transcripts: List[ExpertTranscript]) -> List[str]:
    """Column names: the market, or "market · expert" when two calls share a market."""
    counts = Counter(t.profile.market for t in transcripts)
    return [
        t.profile.market if counts[t.profile.market] == 1 else f"{t.profile.market} · {t.profile.name}"
        for t in transcripts
    ]


def expert_label(transcript: ExpertTranscript) -> str:
    return f"{transcript.profile.market} · {transcript.profile.name}"


def _add_evidence(evidence: Dict[str, str], transcript: ExpertTranscript, timestamp: str, quote: str) -> None:
    key = expert_label(transcript)
    if key in evidence:
        key = f"{key} ({timestamp})"
    evidence[key] = f"[{timestamp}] {quote}"


# ------------------------------------------------------------------ comparison


def build_comparison_matrix(
    transcripts: List[ExpertTranscript], answers: List[Dict[int, InterviewAnswer]]
) -> List[ComparisonMetric]:
    labels = call_labels(transcripts)
    rows = []
    for question in INTERVIEW_QUESTIONS:
        cells = {}
        for label, by_question in zip(labels, answers):
            citations = by_question[question["id"]].citations
            cells[label] = (
                f"[{citations[0].timestamp}] {citations[0].highlight or citations[0].quote_text}"
                if citations
                else "No relevant answer in this transcript."
            )
        rows.append(ComparisonMetric(dimension=question["short_label"], cells=cells))
    return rows


# ---------------------------------------------------------------------- themes


@dataclass
class _ExpertTurn:
    call: int
    timestamp: str
    text: str
    stems: Set[str] = field(default_factory=set)


def _expert_turns(transcripts: List[ExpertTranscript]) -> Tuple[List[_ExpertTurn], Dict[str, Counter]]:
    turns: List[_ExpertTurn] = []
    surface: Dict[str, Counter] = defaultdict(Counter)
    for call, transcript in enumerate(transcripts):
        for turn in transcript.turns:
            if turn.is_interviewer:
                continue
            stems: Set[str] = set()
            for word in tokenize(turn.text):
                stem = stem_token(word)
                if word in STOPWORDS or stem in STOP_STEMS or len(word) < MIN_TERM_LENGTH or word.isdigit():
                    continue
                stems.add(stem)
                surface[stem][word] += 1
            turns.append(_ExpertTurn(call=call, timestamp=turn.timestamp, text=turn.text, stems=stems))
    return turns, surface


def find_common_themes(transcripts: List[ExpertTranscript]) -> List[CrossExpertTheme]:
    """Topics that recur in expert answers across a majority of the loaded calls."""
    call_count = len(transcripts)
    if call_count < 2:
        return []
    turns, surface = _expert_turns(transcripts)
    min_calls = max(2, math.ceil(call_count * MIN_CALL_SHARE))

    turns_by_stem: Dict[str, Set[int]] = defaultdict(set)
    for index, turn in enumerate(turns):
        for stem in turn.stems:
            turns_by_stem[stem].add(index)

    candidates = []
    for stem, indexes in turns_by_stem.items():
        calls = {turns[i].call for i in indexes}
        if len(calls) >= min_calls and len(indexes) / len(turns) <= MAX_TOPIC_SHARE:
            candidates.append((len(calls), len(indexes), stem, indexes))
    candidates.sort(key=lambda c: (-c[0], -c[1], c[2]))

    # Words that recur in the same passages describe one theme, not several.
    clusters: List[Dict[str, Any]] = []
    for _, _, stem, indexes in candidates:
        home = next(
            (c for c in clusters if len(c["seed"] & indexes) / len(c["seed"] | indexes) >= CLUSTER_OVERLAP),
            None,
        )
        if home:
            home["terms"].append(stem)
            home["indexes"] |= indexes
        else:
            clusters.append({"terms": [stem], "seed": set(indexes), "indexes": set(indexes)})

    # Themes that cite mostly the same passages are one theme with several terms.
    merged: List[Dict[str, Any]] = []
    for cluster in clusters:
        chosen = _best_turns(cluster, turns)
        if len(chosen) < 2:
            continue
        signature = set(chosen.values())
        twin = next(
            (m for m in merged if len(signature & m["signature"]) / len(signature | m["signature"]) >= MERGE_OVERLAP),
            None,
        )
        if twin:
            twin["terms"].extend(term for term in cluster["terms"] if term not in twin["terms"])
        else:
            merged.append({"terms": list(cluster["terms"]), "chosen": chosen, "signature": signature})

    return [_render_theme(item, turns, transcripts, surface) for item in merged[:MAX_THEMES]]


def _best_turns(cluster: Dict[str, Any], turns: List[_ExpertTurn]) -> Dict[int, int]:
    """Per call, the index of the passage that mentions the most of the theme's terms."""
    term_set = set(cluster["terms"])
    best: Dict[int, int] = {}
    for index in sorted(cluster["indexes"]):
        call = turns[index].call
        if call not in best or len(turns[index].stems & term_set) > len(turns[best[call]].stems & term_set):
            best[call] = index
    return best


def _render_theme(
    item: Dict[str, Any],
    turns: List[_ExpertTurn],
    transcripts: List[ExpertTranscript],
    surface: Dict[str, Counter],
) -> CrossExpertTheme:
    term_set = set(item["terms"])
    words = [surface[term].most_common(1)[0][0] for term in item["terms"][:5]]

    evidence: Dict[str, str] = {}
    for call in sorted(item["chosen"]):
        turn = turns[item["chosen"][call]]
        sentences = split_sentences(turn.text)
        best = max(
            range(len(sentences)),
            key=lambda i: (len({stem_token(w) for w in tokenize(sentences[i])} & term_set), -i),
        )
        _add_evidence(evidence, transcripts[call], turn.timestamp, sentences[best])

    calls = len(item["chosen"])
    markets = ", ".join(dict.fromkeys(transcripts[call].profile.market for call in sorted(item["chosen"])))
    return CrossExpertTheme(
        theme_id=f"theme-{item['terms'][0]}",
        title=" · ".join(word.capitalize() for word in words[:2]),
        summary=(
            f"Recurs in the expert answers from {calls} of {len(transcripts)} calls ({markets}); "
            f"shared terms: {', '.join(words)}. The quote from each expert is below."
        ),
        consensus_level=f"Supported by {calls} of {len(transcripts)} calls",
        supporting_evidence=evidence,
    )


# --------------------------------------------------------------- disagreements


def find_figure_disagreements(
    transcripts: List[ExpertTranscript], answers: List[Dict[int, InterviewAnswer]]
) -> List[CrossExpertDisagreement]:
    """Guide questions where two or more experts quote figures and the figures differ."""
    disagreements = []
    for question in INTERVIEW_QUESTIONS:
        positions: List[Tuple[ExpertTranscript, str, str, frozenset]] = []
        for transcript, by_question in zip(transcripts, answers):
            citations = by_question[question["id"]].citations
            if not citations:
                continue
            quote = citations[0].highlight or citations[0].quote_text
            numbers = figures(quote)
            if numbers:
                positions.append((transcript, citations[0].timestamp, quote, frozenset(numbers)))

        if len(positions) < 2 or len({numbers for *_, numbers in positions}) < 2:
            continue
        evidence: Dict[str, str] = {}
        for transcript, timestamp, quote, _ in positions:
            _add_evidence(evidence, transcript, timestamp, quote)
        disagreements.append(
            CrossExpertDisagreement(
                topic=f"{question['short_label']}: experts quote different figures",
                description=(
                    f"{len(positions)} of {len(transcripts)} experts give figures for this question, "
                    "and the figures do not match. Compare the quotes below."
                ),
                divergence_type="Quantified answer",
                expert_positions=evidence,
            )
        )
    return disagreements


# ------------------------------------------------------------- executive summary


def compose_executive_summary(
    transcripts: List[ExpertTranscript],
    themes: List[CrossExpertTheme],
    disagreements: List[CrossExpertDisagreement],
) -> str:
    if not transcripts:
        return "No transcripts are loaded."
    markets = ", ".join(dict.fromkeys(t.profile.market for t in transcripts))
    if len(transcripts) == 1:
        return (
            f"Only one call is loaded ({markets}), so there is nothing to compare yet. "
            "Load at least one more transcript to see recurring themes and disagreements."
        )

    parts = [f"Across {len(transcripts)} expert calls ({markets})"]
    if themes:
        topics = "; ".join(theme.title.lower().replace(" · ", " / ") for theme in themes[:4])
        parts.append(f", the topics that recur most are: {topics}.")
    else:
        parts.append(", no topic recurs across enough calls to report as a theme.")
    if disagreements:
        labels = ", ".join(d.topic.split(":")[0].lower() for d in disagreements)
        parts.append(f" The experts quote different figures on {labels}.")
    else:
        parts.append(" The experts do not quote conflicting figures.")
    parts.append(" Every item is backed by timestamped quotes from the transcripts.")
    return "".join(parts)


# ------------------------------------------------------------------ entry point


def generate_cross_expert_synthesis(
    transcripts: List[ExpertTranscript], llm_service: Optional[Any] = None
) -> CrossExpertSynthesis:
    """Synthesise the loaded calls; optionally let a model draft themes and disagreements."""
    extractor = ExpertAnswerExtractor()
    answers = [
        {answer.question_id: answer for answer in extractor.get_grounded_expert_answers(transcript)}
        for transcript in transcripts
    ]
    matrix = build_comparison_matrix(transcripts, answers)
    themes = find_common_themes(transcripts)
    disagreements = find_figure_disagreements(transcripts, answers)
    method = "Computed from the transcripts: recurring topics and differing figures"
    notes = ""

    if llm_service is not None and llm_service.is_configured() and len(transcripts) >= 2:
        priority = {
            (index, citation.timestamp)
            for index, by_question in enumerate(answers)
            for answer in by_question.values()
            for citation in answer.citations
        }
        draft: ModelDraft = draft_synthesis(transcripts, llm_service, priority)
        notes = draft.notes
        if draft.themes or draft.disagreements:
            drafted = [name for name, items in (("themes", draft.themes), ("disagreements", draft.disagreements)) if items]
            computed = [name for name in ("themes", "disagreements") if name not in drafted]
            method = f"{' and '.join(drafted).capitalize()} drafted by {draft.model}; every quote verified verbatim against the transcripts"
            if computed:
                method += f". {' and '.join(computed).capitalize()} computed from the transcripts"
            themes = draft.themes or themes
            disagreements = draft.disagreements or disagreements

    return CrossExpertSynthesis(
        executive_summary=compose_executive_summary(transcripts, themes, disagreements),
        common_themes=themes,
        disagreements=disagreements,
        comparison_matrix=matrix,
        method=method,
        notes=notes,
    )
