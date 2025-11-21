"""
Data Models Module
Pydantic models for state management and API request/response schemas.
"""

from typing import List, Optional, Dict, Any, Annotated
from datetime import datetime
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class Message(BaseModel):
    """Represents a single message in the conversation."""
    role: str = Field(..., description="Role of the message sender (user, assistant, system)")
    content: str = Field(..., description="Content of the message")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the message was created")


class InterviewState(TypedDict):
    """
    State schema for the LangGraph interview workflow.
    Uses TypedDict for LangGraph compatibility.

    Implements Hybrid Modular State Machine architecture with:
    - Strategic time allocation
    - Conversational turn handling
    - Quality-driven assessments
    """
    # Session identification
    session_id: str

    # Document content
    resume_text: str
    job_description_text: str
    custom_instructions: str  # Custom guidance for interview strategy (empty string if not provided)

    # Conversation tracking
    messages: Annotated[List[Dict[str, Any]], add_messages]

    # V3 Interview strategy and context
    interview_time_strategy: str  # JSON serialized InterviewTimeStrategy
    key_topics: List[str]
    questions_asked: int

    # Current state
    current_question: Optional[str]
    current_topic: Optional[str]

    # Time tracking
    start_time: float
    time_elapsed: float

    # Interview status
    is_concluded: bool
    conclusion_reason: Optional[str]

    # Strategic Time Allocation
    interview_time_strategy: Optional[str]  # JSON serialized InterviewTimeStrategy schema
    topic_statistics: Dict[str, Dict[str, Any]]  # Per-topic tracking: questions, time, coverage, etc.
    # Example topic_statistics:
    # {
    #   "Python Programming": {
    #       "questions_asked": 3,
    #       "time_spent_minutes": 7.5,
    #       "assessment_complete": False,
    #       "coverage": 0.75,
    #       "confidence": 0.8,
    #       "last_quality_assessment": {...}
    #   }
    # }

    # Conversational Turn Handling
    last_turn_analysis: Optional[str]  # JSON serialized ConversationalTurnAnalysis
    pending_clarification: bool  # True if candidate requested clarification
    pending_continuation: bool  # True if waiting for candidate to continue/complete thought

    # Topic Quality Tracking
    current_topic_quality: Optional[str]  # JSON serialized TopicAssessmentQuality
    topics_completed: List[str]  # Topics marked as adequately assessed
    critical_topics_covered: List[str]  # Critical priority topics that have been covered


class SessionData(BaseModel):
    """Data stored for each interview session."""
    session_id: str
    resume_text: str
    job_description_text: str
    custom_instructions: Optional[str] = None
    conversation_history: List[Message] = Field(default_factory=list)
    interview_time_strategy: Optional[str] = None  # V3: JSON serialized InterviewTimeStrategy
    key_topics: List[str] = Field(default_factory=list)
    questions_asked: int = 0
    start_time: Optional[datetime] = None
    is_active: bool = True
    is_concluded: bool = False
    conclusion_reason: Optional[str] = None

    # Current state tracking (persisted between graph invocations)
    current_topic: Optional[str] = None  # Current topic being discussed
    current_question: Optional[str] = None  # Last question asked

    # Strategic fields
    topic_statistics: Dict[str, Dict[str, Any]] = Field(default_factory=dict)  # Per-topic tracking
    topics_completed: List[str] = Field(default_factory=list)  # Completed topics
    critical_topics_covered: List[str] = Field(default_factory=list)  # Critical priority topics covered


class UploadDocumentsRequest(BaseModel):
    """Request model for document upload (not used directly with multipart/form-data)."""
    pass


class UploadDocumentsResponse(BaseModel):
    """Response model for document upload."""
    session_id: str
    message: str
    resume_length: int
    job_description_length: int
    has_custom_instructions: bool = False
    custom_instructions_length: int = 0


class StartInterviewRequest(BaseModel):
    """Request model for starting an interview."""
    session_id: str


class StartInterviewResponse(BaseModel):
    """Response model for starting an interview."""
    session_id: str
    first_question: str
    message: str
    audio_chunks: Optional[List[str]] = Field(default=None, description="Base64-encoded audio chunks for first question")


class TranscribeAudioRequest(BaseModel):
    """Request model for audio transcription (not used directly with multipart/form-data)."""
    pass


class TranscribeAudioResponse(BaseModel):
    """Response model for audio transcription."""
    transcription: str
    session_id: Optional[str] = None


class SubmitAnswerRequest(BaseModel):
    """Request model for submitting an answer."""
    session_id: str
    answer: str


class SubmitAnswerResponse(BaseModel):
    """Response model for submitting an answer."""
    session_id: str
    next_question: Optional[str] = None
    is_concluded: bool = False
    conclusion_message: Optional[str] = None
    time_remaining_seconds: Optional[float] = None


class InterviewStatusResponse(BaseModel):
    """Response model for interview status."""
    session_id: str
    is_active: bool
    is_concluded: bool
    questions_asked: int
    time_elapsed_seconds: Optional[float] = None
    time_remaining_seconds: Optional[float] = None
    conclusion_reason: Optional[str] = None


class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str
    detail: Optional[str] = None


class TranscriptionSegment(BaseModel):
    """Represents a transcription segment from WhisperLive."""
    text: str = Field(..., description="Transcribed text")
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    is_final: bool = Field(..., description="Whether this is a finalized segment or partial")


class TranscriptionMessage(BaseModel):
    """WebSocket message for real-time transcription updates."""
    type: str = Field(..., description="Message type: 'transcript', 'complete', 'error'")
    text: Optional[str] = Field(None, description="Transcribed text (for transcript type)")
    segment: Optional[TranscriptionSegment] = Field(None, description="Full segment details")
    error: Optional[str] = Field(None, description="Error message (for error type)")
