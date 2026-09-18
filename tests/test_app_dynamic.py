"""End-to-end: the running Streamlit app must update when the loaded transcripts change."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from core.parser import parse_transcript  # noqa: E402
from tests.test_synthesis import ITALY  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def summary_text(at):
    return next(m.value for m in at.markdown if m.value.startswith("Across "))


def test_app_updates_columns_metrics_and_summary_when_a_transcript_is_added():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert list(at.dataframe[0].value.columns) == ["Dimension", "France", "Germany", "United Kingdom"]
    assert at.metric[0].value == "3"
    before = summary_text(at)
    assert "3 expert calls" in before

    at.session_state["transcripts"] = list(at.session_state["transcripts"]) + [
        parse_transcript(ITALY, "Transcript_4_Italy.txt")
    ]
    at.run()
    assert not at.exception
    assert list(at.dataframe[0].value.columns)[-1] == "Italy"
    assert at.metric[0].value == "4"
    after = summary_text(at)
    assert "4 expert calls" in after and "Italy" in after
    assert after != before


def test_app_shows_how_the_synthesis_was_produced():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert any("How this was produced: Computed from the transcripts" in c.value for c in at.caption)
