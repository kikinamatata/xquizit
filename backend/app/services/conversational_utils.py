"""
Conversational Handler Utility (V3)
Handles conversational turn responses for non-answer candidate interactions.
"""

from typing import Optional, Dict, Any
from app.prompts.schemas import ResponseIntent, ConversationalTurnAnalysis
from app.prompts.conversation_analysis import get_simple_conversational_response, create_clarification_response_prompt
from app.prompts.optimization import extract_relevant_context, summarize_conversation
from langchain_core.prompts import ChatPromptTemplate
import json
import logging

logger = logging.getLogger(__name__)


class ConversationalHandler:
    """
    Handles generation of conversational responses for various candidate intents.

    Provides both simple template-based responses and LLM-generated clarifications.
    """

    def __init__(self, llm=None):
        """
        Initialize conversational handler.

        Args:
            llm: LangChain LLM instance (optional, only needed for complex clarifications)
        """
        self.llm = llm

    def generate_response(
        self,
        turn_analysis: ConversationalTurnAnalysis,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate appropriate conversational response based on intent.

        Args:
            turn_analysis: Analysis of candidate's conversational turn
            context: Additional context (question, topic, JD snippet, etc.)

        Returns:
            Conversational response string
        """
        intent = turn_analysis.intent

        # Use suggested response if provided by LLM
        if turn_analysis.suggested_response:
            logger.info(f"Using LLM-suggested response for intent: {intent}")
            return turn_analysis.suggested_response

        # Otherwise, generate response based on intent
        if intent == ResponseIntent.CLARIFICATION_REQUEST:
            return self._generate_clarification(turn_analysis, context)

        elif intent == ResponseIntent.THINKING_ALOUD:
            return get_simple_conversational_response("thinking")

        elif intent == ResponseIntent.ACKNOWLEDGMENT:
            return get_simple_conversational_response("acknowledgment")

        elif intent == ResponseIntent.PARTIAL_ANSWER:
            return get_simple_conversational_response("encouragement")

        elif intent == ResponseIntent.SMALL_TALK:
            return get_simple_conversational_response("redirect")

        else:
            # Default encouragement
            logger.warning(f"Unexpected intent: {intent}, using default encouragement")
            return "Please continue."

    def _generate_clarification(
        self,
        turn_analysis: ConversationalTurnAnalysis,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate clarification response with enhanced context.

        IMPROVED: Now passes resume context, expanded JD context, assessment goals,
        and conversation history for more relevant clarifications.

        Args:
            turn_analysis: Analysis with clarification request
            context: Dict with enhanced context (resume_context, jd_context, etc.)

        Returns:
            Contextual clarification response
        """
        if not context:
            context = {}

        original_question = context.get("question", "the previous question")
        clarification_request = "Could you please clarify what you're asking?"

        # If LLM available, generate custom clarification with enhanced context
        if self.llm:
            try:
                logger.info("Generating custom clarification using LLM with enhanced context")

                prompt = create_clarification_response_prompt()
                messages = prompt.format_messages(
                    original_question=original_question,
                    clarification_request=clarification_request,
                    current_topic=context.get("topic", "this topic"),
                    assessment_goals=context.get("assessment_goals", "General exploration"),
                    resume_context=context.get("resume_context", "No resume context available"),
                    jd_context=context.get("jd_context", "No job requirements available"),
                    conversation_history=context.get("conversation_history", "This is the beginning of the interview.")
                )

                response = self.llm.invoke(messages)
                return response.content

            except Exception as e:
                logger.error(f"Error generating LLM clarification: {e}")
                # Fall through to template

        # Template-based clarification (fallback)
        return self._template_clarification(original_question, clarification_request)

    def _template_clarification(self, question: str, request: str) -> str:
        """
        Generate simple template-based clarification.

        Args:
            question: Original question
            request: What they're asking to clarify

        Returns:
            Template-based clarification
        """
        # Check for common clarification types
        if "repeat" in request.lower():
            return f"Of course! The question was: {question}"

        elif "clarify" in request.lower() or "mean" in request.lower():
            return f"Good question! Let me rephrase: {question}"

        else:
            # Generic clarification
            return f"Happy to clarify! {question}"

    @staticmethod
    def from_json(turn_analysis_json: str) -> ConversationalTurnAnalysis:
        """
        Parse ConversationalTurnAnalysis from JSON.

        Args:
            turn_analysis_json: JSON-serialized analysis

        Returns:
            ConversationalTurnAnalysis instance
        """
        data = json.loads(turn_analysis_json)
        return ConversationalTurnAnalysis(**data)

    @staticmethod
    def to_json(turn_analysis: ConversationalTurnAnalysis) -> str:
        """
        Serialize ConversationalTurnAnalysis to JSON.

        Args:
            turn_analysis: Analysis to serialize

        Returns:
            JSON string
        """
        return turn_analysis.model_dump_json()

    def should_wait_for_continuation(self, turn_analysis: ConversationalTurnAnalysis) -> bool:
        """
        Determine if we should wait for candidate to continue.

        Args:
            turn_analysis: Analysis of candidate's turn

        Returns:
            True if should wait for more input before proceeding
        """
        waiting_intents = [
            ResponseIntent.THINKING_ALOUD,
            ResponseIntent.ACKNOWLEDGMENT,
            ResponseIntent.PARTIAL_ANSWER,
            ResponseIntent.CLARIFICATION_REQUEST
        ]

        return turn_analysis.intent in waiting_intents

    def should_process_answer(self, turn_analysis: ConversationalTurnAnalysis) -> bool:
        """
        Determine if response contains answer that should be processed.

        Args:
            turn_analysis: Analysis of candidate's turn

        Returns:
            True if contains substantive answer to process
        """
        return turn_analysis.contains_answer and turn_analysis.intent in [
            ResponseIntent.DIRECT_ANSWER,
            ResponseIntent.MIXED
        ]


# Standalone helper functions

def create_conversational_context(
    question: str,
    topic: str,
    resume_text: str = "",
    job_description_text: str = "",
    interview_strategy: str = "",
    messages: list = None,
    max_resume_chars: int = 1000,
    max_jd_chars: int = 800
) -> Dict[str, Any]:
    """
    Create enhanced context dict for conversational response generation.

    IMPROVED: Now uses extract_relevant_context() for smarter context extraction
    and includes conversation history.

    Args:
        question: The interview question
        topic: Current interview topic
        resume_text: Full resume text (for context extraction)
        job_description_text: Full JD text (for context extraction)
        interview_strategy: JSON interview strategy (for assessment goals)
        messages: Conversation history (list of message objects)
        max_resume_chars: Max chars for resume excerpts (default: 1000)
        max_jd_chars: Max chars for JD excerpts (default: 800)

    Returns:
        Enhanced context dict with resume_context, jd_context, assessment_goals, conversation_history
    """
    # Extract relevant context using smart extraction
    context_data = extract_relevant_context(
        resume_text=resume_text or "",
        job_description_text=job_description_text or "",
        current_topic=topic,
        interview_strategy=interview_strategy or "",
        max_resume_chars=max_resume_chars,
        max_jd_chars=max_jd_chars
    )

    # Get conversation history (last 3-4 exchanges for context)
    conversation_history = "This is the beginning of the interview."
    if messages and len(messages) > 0:
        conversation_history = summarize_conversation(
            messages,
            max_exchanges=4  # Last 4 Q&A for conversational context
        )

    return {
        "question": question,
        "topic": topic,
        "assessment_goals": context_data.get("assessment_goals", "General exploration"),
        "resume_context": context_data.get("resume_context", "No resume context available"),
        "jd_context": context_data.get("jd_context", "No job requirements available"),
        "conversation_history": conversation_history,
        # Legacy field for backward compatibility
        "jd_snippet": context_data.get("jd_context", "")[:200]
    }


def extract_clarification_request(response: str) -> Optional[str]:
    """
    Extract the clarification question from candidate response.

    Args:
        response: Candidate's response text

    Returns:
        Extracted clarification request or None
    """
    # Simple heuristic: if contains '?', extract last sentence with '?'
    if '?' in response:
        sentences = response.split('.')
        for sentence in reversed(sentences):
            if '?' in sentence:
                return sentence.strip()

    return response
