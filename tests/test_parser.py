"""
Tests for Transcript Parser.
"""

import os
import pytest
from core.parser import parse_transcript, parse_timestamp_to_seconds


def test_parse_timestamp():
    assert parse_timestamp_to_seconds("00:00") == 0
    assert parse_timestamp_to_seconds("01:20") == 80
    assert parse_timestamp_to_seconds("06:08") == 368
    assert parse_timestamp_to_seconds("01:02:03") == 3723


def test_parse_france_transcript():
    with open("Transcript_1_France.txt", "r", encoding="utf-8") as f:
        content = f.read()

    transcript = parse_transcript(content, "Transcript_1_France.txt")
    assert transcript.profile.name == "Dr. Jean Martin"
    assert transcript.profile.market == "France"
    assert "Urology" in transcript.profile.role
    assert transcript.total_turns > 5
    assert any(t.timestamp == "00:18" for t in transcript.turns)
    assert any("academic hospitals" in t.text for t in transcript.turns)


def test_parse_germany_transcript():
    with open("Transcript_2_Germany.txt", "r", encoding="utf-8") as f:
        content = f.read()

    transcript = parse_transcript(content, "Transcript_2_Germany.txt")
    assert transcript.profile.name == "Anna Keller"
    assert transcript.profile.market == "Germany"
    assert "Procurement" in transcript.profile.role
    assert transcript.total_turns > 5
    assert any(t.timestamp == "00:16" for t in transcript.turns)


def test_parse_uk_transcript():
    with open("Transcript_3_UK.txt", "r", encoding="utf-8") as f:
        content = f.read()

    transcript = parse_transcript(content, "Transcript_3_UK.txt")
    assert transcript.profile.name == "Dr. Emily Carter"
    assert "United Kingdom" in transcript.profile.market or "UK" in transcript.profile.market
    assert "Urologist" in transcript.profile.role
    assert transcript.total_turns > 5
    assert any(t.timestamp == "00:14" for t in transcript.turns)


def test_two_experts_from_the_same_market_get_distinct_ids():
    header = "Expert 1 – {name}\nRole: Urologist\nMarket: France\n\n00:00\nInterviewer: Hello there.\n"
    first = parse_transcript(header.format(name="Dr. Jean Martin"), "a.txt")
    second = parse_transcript(header.format(name="Dr. Marc Dupont"), "b.txt")
    assert first.profile.id != second.profile.id


def test_upload_without_a_header_number_is_identified_by_filename():
    text = "Dr. X\nRole: Surgeon\nMarket: Italy\n\n00:00\nInterviewer: Hello there.\n"
    assert parse_transcript(text, "Italy call.txt").profile.id == "uploaded_italy_call"
