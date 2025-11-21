"""
Pydantic Schemas for Structured LLM Outputs
Ensures type-safe responses from LLM calls.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional
from enum import Enum
import logging


# ==================== Strategic Time Allocation Schemas ====================

class TopicPriority(str, Enum):
    """Priority levels for interview topics"""
    CRITICAL = "critical"      # Must assess thoroughly (core job requirements)
    HIGH = "high"              # Important but some flexibility
    MEDIUM = "medium"          # Good to cover if time permits
    LOW = "low"                # Nice to have


class TopicAllocation(BaseModel):
    """Time and question budget for a single topic"""

    topic_name: str = Field(
        description="Name of the interview topic"
    )

    priority: TopicPriority = Field(
        description="Priority level (critical/high/medium/low)"
    )

    estimated_minutes: float = Field(
        description="Estimated time to allocate (in minutes)",
        ge=2.0,
        le=15.0
    )

    min_questions: int = Field(
        description="Minimum questions before considering topic complete",
        ge=1,
        le=5
    )

    max_questions: int = Field(
        description="Soft maximum questions (warn if exceeded significantly)",
        ge=1,
        le=10
    )

    assessment_goals: list[str] = Field(
        description="What we need to assess in this topic (2-4 goals)",
        min_length=1,
        max_length=5
    )


class InterviewTimeStrategy(BaseModel):
    """V3 interview strategy with strategic time allocation"""

    # Core strategy fields
    key_topics: list[str] = Field(
        description="3-5 focused interview topics",
        min_length=3,
        max_length=5
    )

    # NEW: Strategic time allocation fields
    topic_allocations: list[TopicAllocation] = Field(
        description="Time and question budget per topic (must match key_topics)"
    )

    critical_skills: list[str] = Field(
        description="Must-assess skills from job description (2-5 items)",
        min_length=1,
        max_length=5
    )

    risk_areas: list[str] = Field(
        description="Potential concerns to explore: gaps, mismatches, uncertainties (0-5 items)",
        max_length=5,
        default_factory=list
    )

    @field_validator('risk_areas', mode='before')
    @classmethod
    def truncate_risk_areas(cls, v):
        """
        Auto-truncate risk_areas if LLM provides more than max_length.
        This prevents validation errors while maintaining quality.
        """
        if isinstance(v, list) and len(v) > 5:
            logger = logging.getLogger(__name__)
            logger.warning(f"LLM generated {len(v)} risk areas (max 5). Auto-truncating to first 5: {v[:5]}")
            logger.debug(f"Dropped risk areas: {v[5:]}")
            return v[:5]  # Keep first 5 items
        return v

    def get_total_allocated_time(self) -> float:
        """Calculate total allocated time across all topics"""
        return sum(t.estimated_minutes for t in self.topic_allocations)

    def validate_allocations(self) -> bool:
        """Validate that time allocations are reasonable"""
        total_time = self.get_total_allocated_time()

        # Should allocate ~40 minutes (leaving 5 min buffer from 45 total)
        if total_time > 42.0:
            return False

        # Each topic allocation must match a key topic
        allocated_topics = {t.topic_name for t in self.topic_allocations}
        if allocated_topics != set(self.key_topics):
            return False

        return True


class TopicAssessmentQuality(BaseModel):
    """Real-time assessment of how well a topic has been explored"""

    topic_name: str = Field(
        description="Name of the topic being assessed"
    )

    questions_asked: int = Field(
        description="Number of questions asked on this topic so far",
        ge=0
    )

    time_spent_minutes: float = Field(
        description="Estimated time spent on this topic in minutes",
        ge=0.0
    )

    assessment_coverage: float = Field(
        description="How well assessment goals are met (0.0 = not at all, 1.0 = fully)",
        ge=0.0,
        le=1.0
    )

    confidence_level: float = Field(
        description="Confidence in our assessment of candidate's ability (0.0 = low, 1.0 = high)",
        ge=0.0,
        le=1.0
    )

    needs_more_exploration: bool = Field(
        description="Whether this topic needs additional questions for adequate assessment"
    )



# ==================== Conversational Turn Detection Schemas ====================

class ResponseIntent(str, Enum):
    """Classification of candidate response intent"""
    DIRECT_ANSWER = "direct_answer"              # Answering the question
    CLARIFICATION_REQUEST = "clarification"      # "What do you mean by...?"
    THINKING_ALOUD = "thinking"                  # "Let me think... hmm..."
    ACKNOWLEDGMENT = "acknowledgment"            # "Got it", "Makes sense"
    SMALL_TALK = "small_talk"                    # Off-topic conversation
    PARTIAL_ANSWER = "partial_answer"            # Started answering but incomplete
    MIXED = "mixed"                              # Conversational + answer combined


class ConversationalTurnAnalysis(BaseModel):
    """Analysis of candidate's conversational turn"""

    intent: ResponseIntent = Field(
        description="Primary intent of the response"
    )

    contains_answer: bool = Field(
        description="Whether the response contains substantive answer content"
    )

    needs_response: bool = Field(
        description="Whether interviewer should respond conversationally before proceeding"
    )

    suggested_response: Optional[str] = Field(
        default=None,
        description="Suggested conversational response (if needs_response=True). Max 200 chars.",
        max_length=200
    )


# Example usage for type annotations
__all__ = [
    # Strategic allocation schemas
    'TopicPriority',
    'TopicAllocation',
    'InterviewTimeStrategy',
    'TopicAssessmentQuality',
    # Conversational schemas
    'ResponseIntent',
    'ConversationalTurnAnalysis'
]
