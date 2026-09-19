# Hasamex Expert Call Analyzer

A Streamlit research app for the Hasamex AI Engineer case study. It analyses the supplied France, Germany, and UK robotic-surgery expert calls and keeps the evidence visible for every material finding.

A static, read-only walkthrough of the same case is hosted at https://hasamex-expert-call-review.reaper2411.chatgpt.site. That site is a separate front-end and does not run the retrieval or guardrail code in this repository.

## What it does

- Parses transcript metadata, speakers, and timestamps into structured turns.
- Answers each of the six interview-guide questions per expert, quoting the expert verbatim with the key sentence highlighted.
- Computes a cross-call comparison, recurring themes, and disagreements from whichever transcripts are loaded, each backed by timestamped quotes. Add a transcript and everything updates.
- **Ask across the calls:** retrieves timestamped source passages, scores how well they cover the question, and refuses questions the transcripts cannot answer.
- Lets reviewers open the surrounding dialogue for any citation.
- Accepts additional `.txt` transcripts for the active browser session.
- Exports a Markdown research brief with source evidence.

## How answers stay grounded

**Every displayed quote is checked against the original transcript** and shown with market, speaker, and timestamp.

**Retrieval refuses instead of guessing.** Each expert answer (plus the interviewer question that prompted it) is scored by IDF-weighted coverage of the question's key terms. Confidence (High / Medium / Low) comes from that coverage, not a constant. If nothing covers enough of the question, the app refuses and names the terms that appear in no transcript (for example "reimbursement", "da Vinci pricing"). A match on a single keyword is capped at Low and flagged. A named market ("in the UK") or expert ("Anna Keller") restricts results to that market or expert.

**Optional model step, verified in code.** With Ollama or a hosted OpenAI-compatible endpoint configured, the model writes a short answer as claims that each carry verbatim quotes from numbered sources. The app then drops any claim that fails these checks:

1. every quote must appear verbatim in the cited source turn;
2. every figure in the claim (digits or spelled-out numbers) must appear in its quotes;
3. any market named in the claim must be a market it cites.

If no claim survives, or the reply is not valid JSON, the app shows source passages instead. If the model judges the passages insufficient, the app refuses. No model is called when retrieval has already refused.

## How the cross-call synthesis works

Nothing in `core/analyzer.py` is written about a particular call. From the loaded transcripts it computes:

- **Comparison table:** each expert's cited answer to every guide question, one column per market (or "market · expert" if two calls share a market).
- **Themes:** words that recur in expert answers across at least half the calls, clustered by the passages they appear in, with one verbatim quote per call. The "supported by k of N calls" count is the number of quotes shown.
- **Disagreements:** guide questions where two or more experts quote figures and the figures differ (for example 15-20% vs. high single digits, or six to twelve vs. nine to eighteen months). Number words and digits are compared as values.
- **Executive summary:** a template filled from the results above.

With a model configured, **"Draft themes and disagreements with the model"** asks it for richer themes and disagreements. Each item must cite quotes from at least two different calls; an item is dropped if any quote is not verbatim in its source turn, or if a figure or market named in its text is not backed by its quotes. Supporting-call counts are recounted from the verified quotes. If nothing survives, the computed results are shown and the app says so.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open `http://localhost:8501`. Run from the repository root; the app reads the three transcripts from the working directory.

No API key is required. Evidence-only mode is the default and makes no model call.

### Optional model modes

The LLM is additive. Every guide answer, quote, timestamp, theme, disagreement and refusal works in Evidence-only mode with no model at all.

**Local Ollama.** Run Ollama; if it listens somewhere other than `http://127.0.0.1:11434`, copy `.env.example` to `.env` and set `OLLAMA_BASE_URL`. The app probes for local models once per 30 seconds, so a hosted deployment does not pay a loopback timeout on every rerun.

**Hosted OpenAI-compatible provider.** There are two ways to configure it, and both are optional:

1. **Server-side (recommended for a deployed app).** Set `CLOUD_MODEL_BASE_URL`, `CLOUD_MODEL_NAME` and `CLOUD_MODEL_API_KEY`. Locally these come from `.env` (git-ignored); on Streamlit Community Cloud paste them into **Advanced settings → Secrets**, where Streamlit exposes them to the app as environment variables. Never commit a key.
2. **Reviewer-supplied in the sidebar.** If no server-side provider is configured, selecting "Hosted compatible API" reveals a provider, base URL, model name and API-key field. A key entered here lives only in that browser session: it is never written to disk, never logged, and never included in the Markdown export. This lets a reviewer try the model path on a hosted deployment without the operator holding a key.

Hosted mode sends only the retrieved candidate turns, never full transcripts. No model is called when retrieval has already refused the question.

## Tests

```powershell
python -m pytest tests -q -p no:cacheprovider
```

The tests cover: parsing; quote verification; a hand-checked gold set of which turn answers each of the six questions for each expert; refusal of out-of-scope questions; single-keyword and substring-match regressions; market and expert filtering; the model step with stub models (fabricated quote, wrong-source quote, invented number, misattributed market, partial failure, malformed JSON, model refusal); that the synthesis follows its inputs (added, removed, renamed and duplicate transcripts, in the running app too), that all synthesis evidence is verbatim source text, and the model-draft verifier with stub models.

## Architecture

```text
Transcript files
    -> parser:     timestamped speaker turns
    -> retrieval:  stemmed, IDF-weighted coverage scoring; market/expert filters; refusal
    -> (optional)  model drafts a cited answer as claims + verbatim quotes
    -> verifier:   quote-in-turn, figures-in-quote, market-attribution checks; failures dropped
    -> Streamlit:  evidence cards, context, synthesis, export
```

Modules: `core/parser.py`, `core/extractor.py` (guide answers), `core/qa_engine.py` (retrieval and answering), `core/answer_generator.py` (model prompt and claim/quote verification), `core/analyzer.py` (computed comparison, themes, disagreements), `core/synthesis_model.py` (optional verified model-drafted themes), `core/llm_service.py` (model clients).

## Deploy

The app runs on Streamlit Community Cloud with no configuration and no secrets, because Evidence-only is the default:

1. Push this repository to GitHub.
2. At [share.streamlit.io](https://share.streamlit.io), choose **Create app → Deploy a public app from GitHub**.
3. Set the main file path to `app.py` and deploy. Community Cloud installs `requirements.txt` automatically.

Nothing else is required. If you want the hosted model mode available to reviewers, either add the `CLOUD_MODEL_*` secrets in **Advanced settings → Secrets**, or leave it unset and let reviewers enter their own key in the sidebar.

## Known limitations

- **Computed themes are lexical.** Themes are recurring words clustered by the passages they appear in, so titles read like "Clinical · Economics" rather than a written headline, and a generic word ("system") can occasionally lead one. Disagreements are detected only where experts quote figures that differ; a qualitative disagreement (finance decides vs. finance is balanced) needs the model draft, whose quotes are verified but whose judgement that two positions really differ is the model's.
- Retrieval is lexical. It cannot tell "capital budget" from "capital of France" on keywords alone; it flags that case as a Low-confidence keyword hit, and the model step (when enabled) can refuse it. Superlative questions ("which country is fastest?") are usually refused rather than answered.
- The automated model-step tests use stub models, never a live provider. The hosted path has been exercised by hand against a real OpenAI-compatible provider, but the test suite itself does not assert on a live model's output.
- Answers are ranked by transcript turn, not sub-turn, and the synonym list is small and specific to this interview guide.

## Scaling beyond three calls

Persist turns and metadata in Postgres; add dense embeddings alongside the lexical index for hybrid retrieval and rerank; queue parsing and embedding jobs; run the model-drafted synthesis map-reduce style per topic (it already falls back to guide-cited turns above 120 expert turns); and keep the citation schema (transcript, speaker, timestamp, character offsets) identical from ingestion through export.
