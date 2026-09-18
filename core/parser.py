"""
Parser module for Hasamex Expert Call Transcripts.
Extracts metadata, speaker turns, timestamps, and normalized text.
"""

import re
from pathlib import Path
from typing import List, Tuple, Optional
from core.models import ExpertProfile, SpeakerTurn, ExpertTranscript


def parse_timestamp_to_seconds(ts_str: str) -> int:
    """Converts MM:SS or HH:MM:SS string into integer seconds."""
    parts = ts_str.strip().split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return 0


def get_country_flag(market: str) -> str:
    """Returns a flag emoji for common European markets."""
    m = market.lower()
    if "france" in m:
        return "🇫🇷"
    elif "germany" in m:
        return "🇩🇪"
    elif "united kingdom" in m or "uk" in m:
        return "🇬🇧"
    elif "italy" in m:
        return "🇮🇹"
    elif "spain" in m:
        return "🇪🇸"
    elif "europe" in m:
        return "🇪🇺"
    return "🌐"


def parse_transcript(raw_text: str, filename_hint: str = "") -> ExpertTranscript:
    """
    Parses an expert interview transcript into an ExpertTranscript model.
    Handles metadata header (Expert Name, Role, Market) and timestamped speaker turns.
    """
    lines = raw_text.splitlines()
    
    # 1. Parse Header Metadata
    expert_num = 1
    expert_name = "Unknown Expert"
    role = "Expert"
    market = "Europe"
    
    # Regex for header lines
    first_timestamp_idx = -1
    for i, line in enumerate(lines):
        if re.match(r"^\s*\d{1,2}:\d{2}\s*$", line):
            first_timestamp_idx = i
            break
            
    header_lines = lines[:first_timestamp_idx] if first_timestamp_idx != -1 else lines[:5]
    
    for line in header_lines:
        line_clean = line.strip()
        if not line_clean:
            continue
            
        # Match "Expert 1 – Dr. Jean Martin" or "Dr. Jean Martin"
        expert_match = re.search(r"Expert\s*(\d+)?\s*[–\-:]\s*(.+)", line_clean, re.IGNORECASE)
        if expert_match:
            if expert_match.group(1):
                expert_num = int(expert_match.group(1))
            expert_name = expert_match.group(2).strip()
        elif line_clean.lower().startswith("role:"):
            role = line_clean.split(":", 1)[1].strip()
        elif line_clean.lower().startswith("market:"):
            market = line_clean.split(":", 1)[1].strip()
            
    # Fallback to filename hint if market/expert not identified
    if market == "Europe" and filename_hint:
        if "france" in filename_hint.lower():
            market = "France"
        elif "germany" in filename_hint.lower():
            market = "Germany"
        elif "uk" in filename_hint.lower():
            market = "United Kingdom"

    profile_id = f"expert_{expert_num}"
    if "france" in market.lower():
        profile_id = "expert_1_france"
    elif "germany" in market.lower():
        profile_id = "expert_2_germany"
    elif "kingdom" in market.lower() or "uk" in market.lower():
        profile_id = "expert_3_uk"
    elif filename_hint:
        # Uploaded files need a stable, collision-resistant session identifier.
        stem = Path(filename_hint).stem.lower()
        profile_id = "uploaded_" + re.sub(r"[^a-z0-9]+", "_", stem).strip("_")

    flag = get_country_flag(market)
    profile = ExpertProfile(
        id=profile_id,
        expert_id_num=expert_num,
        name=expert_name,
        role=role,
        market=market,
        country_flag=flag,
        summary=f"{expert_name} ({role}) covering the {market} healthcare market."
    )

    # 2. Parse Speaker Turns
    # Structure:
    # 00:00
    # Speaker: Dialogue text...
    turn_blocks = []
    body_text = "\n".join(lines[first_timestamp_idx:]) if first_timestamp_idx != -1 else raw_text
    
    # Split text by timestamps (pattern: newline followed by MM:SS or start of text MM:SS)
    pattern = r"(?:^|\n)\s*(\d{1,2}:\d{2})\s*\n"
    matches = list(re.finditer(pattern, body_text))
    
    turns: List[SpeakerTurn] = []
    for idx, match in enumerate(matches):
        ts = match.group(1).zfill(5)
        start_pos = match.end()
        end_pos = matches[idx + 1].start() if idx + 1 < len(matches) else len(body_text)
        
        turn_content = body_text[start_pos:end_pos].strip()
        if not turn_content:
            continue
            
        # Extract Speaker: Text
        if ":" in turn_content:
            speaker_part, text_part = turn_content.split(":", 1)
            speaker = speaker_part.strip()
            text = text_part.strip()
        else:
            speaker = "Speaker"
            text = turn_content.strip()
            
        is_interviewer = "interviewer" in speaker.lower()
        ts_sec = parse_timestamp_to_seconds(ts)
        
        turns.append(
            SpeakerTurn(
                turn_id=idx + 1,
                timestamp=ts,
                timestamp_seconds=ts_sec,
                speaker=speaker,
                is_interviewer=is_interviewer,
                text=text
            )
        )
        
    duration = turns[-1].timestamp if turns else "00:00"
    
    return ExpertTranscript(
        profile=profile,
        raw_text=raw_text,
        turns=turns,
        total_turns=len(turns),
        duration_str=duration
    )
