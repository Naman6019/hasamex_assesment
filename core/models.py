"""
Core Pydantic models for Hasamex Expert Call Transcript Analyzer.
Defines schemas for transcripts, turns, citations, interview answers, cross-expert synthesis, and QA responses.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class QuoteCitation(BaseModel):
    """Represents a cited quote with exact timestamp and verification status."""
    quote_text: str = Field(..., description="Verbatim quote from the transcript")
    timestamp: str = Field(..., description="Timestamp in MM:SS format")
    timestamp_seconds: int = Field(default=0, description="Timestamp converted to total seconds")
    highlight: Optional[str] = Field(default=None, description="Most relevant sentence(s) within quote_text; always an exact substring")
    speaker: str = Field(..., description="Speaker who uttered the quote")
    is_verified: bool = Field(default=False, description="Whether the quote was verified verbatim in the transcript")
    verification_score: float = Field(default=1.0, description="Verbatim match score (0.0 to 1.0)")
    char_start: Optional[int] = Field(default=None, description="Starting character offset in full text")
    char_end: Optional[int] = Field(default=None, description="Ending character offset in full text")
    market: Optional[str] = Field(default=None, description="Market represented by the source transcript")
    transcript_id: Optional[str] = Field(default=None, description="Stable identifier for the source transcript")


class SpeakerTurn(BaseModel):
    """Represents a single conversational turn in the interview transcript."""
    turn_id: int
    timestamp: str
    timestamp_seconds: int
    speaker: str
    is_interviewer: bool
    text: str


class ExpertProfile(BaseModel):
    """Metadata and profile information for a primary research expert."""
    id: str
    expert_id_num: int
    name: str
    role: str
    market: str
    country_flag: str = ""
    summary: Optional[str] = None


class ExpertTranscript(BaseModel):
    """Complete parsed expert transcript with profile and speaker turns."""
    profile: ExpertProfile
    raw_text: str
    turns: List[SpeakerTurn]
    total_turns: int
    duration_str: str


class InterviewAnswer(BaseModel):
    """Structured answer to an interview guide question for a specific expert."""
    question_id: int
    question_text: str
    expert_id: str
    expert_name: str
    market: str
    summary_answer: str
    key_takeaways: List[str] = Field(default_factory=list)
    citations: List[QuoteCitation] = Field(default_factory=list)


class CrossExpertTheme(BaseModel):
    """A consensus theme identified across experts."""
    theme_id: str
    title: str
    summary: str
    consensus_level: str = "High"  # High, Moderate
    supporting_evidence: Dict[str, str] = Field(default_factory=dict)  # expert_id -> explanation/quote


class CrossExpertDisagreement(BaseModel):
    """A key divergence or disagreement identified across experts."""
    topic: str
    description: str
    divergence_type: str  # Strategic, Outlook, Timeline, Operational
    expert_positions: Dict[str, str] = Field(default_factory=dict)  # expert_id -> stance/quote


class ComparisonMetric(BaseModel):
    """Structured metric for cross-expert side-by-side comparison."""
    dimension: str
    france: str
    germany: str
    uk: str


class CrossExpertSynthesis(BaseModel):
    """Comprehensive synthesis across all transcripts."""
    executive_summary: str
    common_themes: List[CrossExpertTheme]
    disagreements: List[CrossExpertDisagreement]
    comparison_matrix: List[ComparisonMetric]


class QAResponse(BaseModel):
    """Response to an interactive user query across transcripts."""
    query: str
    answer: str
    citations: List[QuoteCitation]
    referenced_experts: List[str]
    confidence: str = "High"  # High, Medium, Low, or None (refusal)
    mode_used: str = "Deterministic grounded retrieval"
    notes: str = ""  # Why the confidence is what it is; shown to the user
