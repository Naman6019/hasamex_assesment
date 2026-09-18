"""Hasamex technical-case demo: source-grounded expert-call analysis."""

import html
from pathlib import Path
from typing import List

import streamlit as st
from dotenv import load_dotenv

from core.analyzer import generate_cross_expert_synthesis
from core.extractor import ExpertAnswerExtractor, INTERVIEW_QUESTIONS
from core.llm_service import LocalModelService, get_hosted_model_service, list_local_models
from core.models import ExpertTranscript, QuoteCitation
from core.parser import parse_transcript
from core.qa_engine import CrossTranscriptQA


load_dotenv()

st.set_page_config(
    page_title="Hasamex | Expert Call Analyzer",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp { background: #fbfcfe; }
      .hero { background: linear-gradient(115deg, #0b2745, #1f4c7a); border-radius: 14px;
              color: white; padding: 28px 30px; margin: 4px 0 20px; }
      .hero h1, .hero p { color: white !important; margin: 0; }
      .hero p { opacity: .86; margin-top: 7px; }
      .quote-card { background: #f8fafc; border-left: 4px solid #0f766e; border-radius: 7px;
                    padding: 13px 15px; margin: 8px 0 14px; line-height: 1.55; }
      .source-label { color: #0f766e; font-size: .82rem; font-weight: 700; margin-bottom: 5px; }
      .source-proof { color: #047857; font-size: .76rem; font-weight: 650; }
      .market-chip { display: inline-block; border-radius: 999px; padding: 3px 9px;
                     background: #e8f0fb; color: #1e4b7a; font-size: .78rem; font-weight: 700; }
      .subtle { color: #475569; font-size: .9rem; }
      .quote-card mark { background: #fef3c7; color: inherit; padding: 1px 2px; border-radius: 3px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_case_transcripts() -> List[ExpertTranscript]:
    filenames = (
        "Transcript_1_France.txt",
        "Transcript_2_Germany.txt",
        "Transcript_3_UK.txt",
    )
    transcripts = []
    for filename in filenames:
        path = Path(filename)
        if path.exists():
            transcripts.append(parse_transcript(path.read_text(encoding="utf-8"), filename))
    return transcripts


def build_report(transcripts: List[ExpertTranscript]) -> str:
    synthesis = generate_cross_expert_synthesis(transcripts)
    lines = [
        "# European Robotic Surgery Market — Expert Call Brief",
        "",
        "## Executive summary",
        synthesis.executive_summary,
        "",
        "## Cross-market comparison",
        "",
    ]
    for item in synthesis.comparison_matrix:
        lines.extend(
            [
                f"### {item.dimension}",
                f"- **France:** {item.france}",
                f"- **Germany:** {item.germany}",
                f"- **United Kingdom:** {item.uk}",
                "",
            ]
        )
    lines.extend(["## Source evidence", ""])
    for theme in synthesis.common_themes:
        lines.extend([f"### {theme.title}", theme.summary])
        lines.extend(f"- **{source}:** {quote}" for source, quote in theme.supporting_evidence.items())
        lines.append("")
    return "\n".join(lines)


def citation_card(citation: QuoteCitation) -> None:
    market = html.escape(citation.market or "Source transcript")
    speaker = html.escape(citation.speaker)
    highlight = citation.highlight
    if highlight and highlight != citation.quote_text and highlight in citation.quote_text:
        before, after = citation.quote_text.split(highlight, 1)
        quote = f"{html.escape(before)}<mark>{html.escape(highlight)}</mark>{html.escape(after)}"
    else:
        quote = html.escape(citation.quote_text)
    verification = "Verbatim match verified" if citation.is_verified else "Source match unavailable"
    verification_class = "source-proof" if citation.is_verified else "subtle"
    st.markdown(
        f"""
        <div class="quote-card">
          <div class="source-label">{market} · {speaker} · {citation.timestamp}</div>
          <div>“{quote}”</div>
          <div class="{verification_class}">✓ {verification}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def turn_context(transcript: ExpertTranscript, citation: QuoteCitation) -> None:
    nearby_turns = [
        turn
        for turn in transcript.turns
        if abs(turn.timestamp_seconds - citation.timestamp_seconds) <= 70
    ]
    for turn in nearby_turns:
        speaker = "**" if turn.speaker == citation.speaker else ""
        st.markdown(f"`{turn.timestamp}` {speaker}{turn.speaker}:{speaker} {turn.text}")


if "transcripts" not in st.session_state:
    st.session_state.transcripts = load_case_transcripts()
if "inference_mode" not in st.session_state:
    st.session_state.inference_mode = "Evidence-only"
if "selected_model" not in st.session_state:
    st.session_state.selected_model = ""
if "qa_query" not in st.session_state:
    st.session_state.qa_query = ""
if "run_qa" not in st.session_state:
    st.session_state.run_qa = False


with st.sidebar:
    st.title("Hasamex Intelligence")
    st.caption("Expert-call research, built for reviewable evidence.")
    st.divider()

    st.subheader("Dataset")
    transcript_count = len(st.session_state.transcripts)
    turn_count = sum(transcript.total_turns for transcript in st.session_state.transcripts)
    st.metric("Expert calls", transcript_count)
    st.metric("Timestamped turns", turn_count)
    st.metric("Interview-guide questions", len(INTERVIEW_QUESTIONS))
    st.caption("Every displayed quote is checked against its source transcript.")

    local_models = list_local_models()
    hosted_service = get_hosted_model_service()
    available_modes = ["Evidence-only"]
    if local_models:
        available_modes.append("Private local Ollama")
    if hosted_service:
        available_modes.append("Hosted compatible API")
    if st.session_state.inference_mode not in available_modes:
        st.session_state.inference_mode = "Evidence-only"

    with st.expander("Model & privacy", expanded=False):
        st.caption(
            "Guide answers, themes and the comparison table never use a model. In the two model modes the LLM "
            "writes a short answer to your question, and the app drops any claim whose quote it cannot find "
            "verbatim in the transcript."
        )
        st.session_state.inference_mode = st.radio(
            "Question-answering mode",
            available_modes,
            index=available_modes.index(st.session_state.inference_mode),
        )
        if st.session_state.inference_mode == "Evidence-only":
            st.success("No model call. Answers are the experts' own words, retrieved from the transcripts.")
        elif st.session_state.inference_mode == "Private local Ollama":
            if st.session_state.selected_model not in local_models:
                st.session_state.selected_model = local_models[0]
            st.session_state.selected_model = st.selectbox(
                "Local model",
                local_models,
                index=local_models.index(st.session_state.selected_model),
            )
            st.caption("Retrieved source turns are sent only to your local Ollama server.")
        else:
            st.caption(
                "Retrieved source turns are sent to the configured provider. "
                "Credentials come from server-side environment variables and are never shown."
            )

    st.divider()
    st.download_button(
        "Download cited research brief (.md)",
        data=build_report(st.session_state.transcripts),
        file_name="hasamex_research_brief.md",
        mime="text/markdown",
        use_container_width=True,
    )


if st.session_state.inference_mode == "Private local Ollama":
    llm_service = LocalModelService(model=st.session_state.selected_model)
elif st.session_state.inference_mode == "Hosted compatible API":
    llm_service = hosted_service
else:
    llm_service = None
extractor = ExpertAnswerExtractor()
synthesis = generate_cross_expert_synthesis(st.session_state.transcripts)
qa_engine = CrossTranscriptQA(st.session_state.transcripts, llm_service=llm_service)

st.markdown(
    """
    <div class="hero">
      <h1>Expert Call Analyzer</h1>
      <p>Turn three expert calls into a reviewable market brief—every finding keeps its source quote and timestamp.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.info(
    "Start with the market brief, answer the six guide questions, then inspect the evidence behind any conclusion."
)

overview_tab, guide_tab, themes_tab, qa_tab, sources_tab, architecture_tab = st.tabs(
    [
        "1 · Start here",
        "2 · Guide answers",
        "3 · Themes & differences",
        "4 · Ask the calls",
        "5 · Source transcripts",
        "6 · Method & privacy",
    ]
)


with overview_tab:
    st.subheader("Suggested review path")
    step_one, step_two, step_three = st.columns(3)
    with step_one:
        st.markdown("**1. Read the brief**")
        st.caption("See the shared market picture and side-by-side comparison.")
    with step_two:
        st.markdown("**2. Answer the guide**")
        st.caption("Open one of the six case questions to compare expert answers.")
    with step_three:
        st.markdown("**3. Verify the evidence**")
        st.caption("Use timestamps and surrounding dialogue before relying on a conclusion.")

    st.subheader("Executive readout")
    st.write(synthesis.executive_summary)

    expert_columns = st.columns(len(st.session_state.transcripts))
    for column, transcript in zip(expert_columns, st.session_state.transcripts):
        with column:
            profile = transcript.profile
            with st.container(border=True):
                st.markdown(f"<span class='market-chip'>{profile.country_flag} {profile.market}</span>", unsafe_allow_html=True)
                st.markdown(f"#### {profile.name}")
                st.write(profile.role)
                st.caption(f"{transcript.total_turns} turns · {transcript.duration_str} duration")

    st.subheader("Cross-market comparison")
    matrix_rows = [
        {
            "Dimension": item.dimension,
            "France": item.france,
            "Germany": item.germany,
            "United Kingdom": item.uk,
        }
        for item in synthesis.comparison_matrix
    ]
    st.dataframe(matrix_rows, use_container_width=True, hide_index=True)
    st.caption("The Themes tab supplies the exact supporting source for comparative findings.")


with guide_tab:
    st.subheader("Interview-guide answers")
    selected_question = st.selectbox(
        "Question",
        INTERVIEW_QUESTIONS,
        format_func=lambda question: f"Q{question['id']} · {question['short_label']}",
    )
    st.write(selected_question["text"])
    st.caption(
        "Each card is the expert's verbatim answer with the key sentence highlighted. "
        "Expand a card to read the surrounding dialogue."
    )

    answer_columns = st.columns(len(st.session_state.transcripts))
    for column, transcript in zip(answer_columns, st.session_state.transcripts):
        answer = next(
            item
            for item in extractor.get_grounded_expert_answers(transcript)
            if item.question_id == selected_question["id"]
        )
        with column:
            st.markdown(f"#### {transcript.profile.country_flag} {transcript.profile.market}")
            st.caption(f"{transcript.profile.name} · {transcript.profile.role}")
            if not answer.citations:
                st.warning("No directly relevant answer was found in this transcript.")
            for citation in answer.citations:
                citation_card(citation)
                with st.expander(f"Read context around {citation.timestamp}"):
                    turn_context(transcript, citation)


with themes_tab:
    st.subheader("Consensus and disagreements")
    consensus_column, difference_column = st.columns(2)

    with consensus_column:
        st.markdown("### Where the calls align")
        for theme in synthesis.common_themes:
            with st.container(border=True):
                st.markdown(f"#### {theme.title}")
                st.write(theme.summary)
                st.caption(theme.consensus_level)
                with st.expander("View exact source evidence"):
                    for source, quote in theme.supporting_evidence.items():
                        st.markdown(f"**{source}** · {quote}")

    with difference_column:
        st.markdown("### Where the calls differ")
        for disagreement in synthesis.disagreements:
            with st.container(border=True):
                st.markdown(f"#### {disagreement.topic}")
                st.write(disagreement.description)
                st.caption(disagreement.divergence_type)
                with st.expander("Compare exact source positions"):
                    for source, quote in disagreement.expert_positions.items():
                        st.markdown(f"**{source}** · {quote}")


with qa_tab:
    st.subheader("Ask across all calls")
    if st.session_state.inference_mode == "Evidence-only":
        st.caption(
            "Returns the experts' own words with timestamps. Questions the transcripts cannot answer are refused, "
            "and weak keyword matches are flagged."
        )
    else:
        st.caption(
            "A model drafts a short answer; only claims whose quotes are verified verbatim against the "
            "transcripts are shown. Unsupported questions are refused."
        )

    quick_prompts = [
        ("Training and economics", "How does surgeon training affect the economics and hospital business case?"),
        ("Compare timelines", "Compare the purchasing decision timelines across France, Germany, and the UK."),
        ("ROI versus clinical value", "How do clinical outcomes weigh against financial ROI when purchasing committees decide?"),
        ("Adoption gaps", "Why is adoption concentrated in larger hospitals rather than smaller hospitals?"),
    ]
    prompt_columns = st.columns(2)
    for index, (label, prompt) in enumerate(quick_prompts):
        with prompt_columns[index % 2]:
            if st.button(label, use_container_width=True, key=f"prompt_{index}"):
                st.session_state.qa_query = prompt
                st.session_state.run_qa = True

    query = st.text_input(
        "Question across the transcripts",
        key="qa_query",
        placeholder="e.g. What purchasing factors matter most?",
    )
    should_run = st.button("Find cited evidence", type="primary", use_container_width=True)
    should_run = should_run or st.session_state.run_qa
    st.session_state.run_qa = False

    if should_run:
        if not query.strip():
            st.warning("Enter a question or choose a suggested prompt.")
        else:
            with st.spinner("Retrieving and verifying source passages..."):
                result = qa_engine.answer_query(query)
            st.caption(f"Mode: {result.mode_used} · Confidence: {result.confidence}")
            if result.confidence == "None":
                st.info(f"**Not answerable from these transcripts.** {result.notes}")
            elif result.confidence == "Low":
                st.warning(result.notes)
            elif result.notes:
                st.caption(result.notes)
            st.markdown(result.answer)
            if result.citations:
                st.markdown("#### Verified source passages")
                citation_columns = st.columns(min(3, len(result.citations)))
                for index, citation in enumerate(result.citations):
                    with citation_columns[index % len(citation_columns)]:
                        citation_card(citation)


with sources_tab:
    st.subheader("Source transcripts")
    source_view_tab, upload_tab = st.tabs(["Read and filter", "Upload a transcript"])

    with source_view_tab:
        transcript = st.selectbox(
            "Transcript",
            st.session_state.transcripts,
            format_func=lambda item: f"{item.profile.country_flag} {item.profile.name} — {item.profile.market}",
        )
        keyword = st.text_input("Filter source turns", placeholder="e.g. budget, training, months")
        turns = transcript.turns
        if keyword.strip():
            keyword_lower = keyword.lower()
            turns = [
                turn
                for turn in turns
                if keyword_lower in turn.text.lower() or keyword_lower in turn.speaker.lower()
            ]
        st.caption(f"Showing {len(turns)} of {transcript.total_turns} timestamped turns.")
        for turn in turns:
            with st.container(border=True):
                role = "Interviewer" if turn.is_interviewer else "Expert"
                st.caption(f"{turn.timestamp} · {role} · {turn.speaker}")
                st.write(turn.text)

    with upload_tab:
        st.caption("Uploads stay in the active session and are parsed into timestamped turns.")
        uploaded = st.file_uploader("Transcript (.txt)", type=["txt"])
        if uploaded is not None:
            raw_text = uploaded.getvalue().decode("utf-8")
            parsed = parse_transcript(raw_text, uploaded.name)
            already_loaded = any(item.profile.id == parsed.profile.id for item in st.session_state.transcripts)
            if already_loaded:
                st.info("This transcript is already in the active session.")
            else:
                st.session_state.transcripts.append(parsed)
                st.success(f"Loaded {parsed.profile.name} · {parsed.total_turns} timestamped turns.")
                st.rerun()


with architecture_tab:
    st.subheader("Technical design")
    st.markdown(
        """
        **1. Ingestion.** Transcript headers and timestamps are parsed into structured speaker turns.
        Each turn keeps its market, speaker, timestamp, and raw source text.

        **2. Retrieval.** The local engine uses question-aware keyword retrieval and pairs an expert answer with the preceding interviewer prompt.
        This means a response such as “nine to eighteen months” remains linked to the purchase-timeline question.

        **3. Citations.** The UI renders a quote only after verifying it is present in the raw transcript.
        Each proof card shows market, speaker, and source timestamp; surrounding dialogue is one expansion away.

        **4. Hallucination control.** Retrieval scores how much of the question each passage covers (IDF-weighted) and
        refuses when nothing covers enough, naming the terms the transcripts never mention. Confidence is computed from
        that coverage. In the optional model modes the LLM must return claims with verbatim quotes; code then checks each
        quote against its source turn, checks that figures in the claim appear in the quote, and that any market named
        is a market cited. Claims that fail are dropped, and if none survive the app shows source passages instead.

        **5. Privacy modes.** Evidence-only mode makes no model call. Local Ollama keeps retrieved source turns on the device.
        Hosted mode sends only retrieved candidate turns (never the full transcripts) to the configured provider and keeps
        its API key in server-side configuration. No model is called when retrieval has already refused the question.

        **6. Scaling to 30+ calls.** Persist turns and metadata in Postgres, add hybrid BM25/vector retrieval,
        queue ingestion and embedding jobs, and keep the same citation schema at every stage.
        """
    )
