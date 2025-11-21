"""
Conversational Handler Agent (V3)
Detects candidate response intent and determines conversational routing.

Follows LangGraph 2025 best practices:
- Structured output with Pydantic ConversationalTurnAnalysis
- Command-based routing
- Type-safe intent classification
"""

import logging
import time
from typing import Dict, Any, Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.types import Command
from langchain_core.messages import HumanMessage, AIMessage

from app.prompts.schemas import ConversationalTurnAnalysis, ResponseIntent
from app.prompts.conversation_analysis import create_conversational_intent_prompt
from app.prompts.optimization import summarize_conversation, extract_relevant_context
from app.core.models import InterviewState
from app.utils.llm_retry import retry_llm_call
from app.utils.logging_config import log_agent_execution, log_llm_call

logger = logging.getLogger(__name__)


class ConversationalHandlerAgent:
    """
    Specialized agent for conversational turn detection and routing.

    Responsibilities:
    - Detect if candidate is answering vs asking/thinking/acknowledging
    - Classify intent: clarification, thinking, acknowledgment, answer, etc.
    - Route to appropriate handler based on intent

    Uses LLM with structured output for accurate intent classification.
    """

    def __init__(self, llm: ChatGoogleGenerativeAI):
        """
        Initialize conversational handler agent.

        Args:
            llm: LLM instance for intent detection (real-time optimized)
        """
        self.llm = llm
        logger.info("Initialized ConversationalHandlerAgent")

    def __call__(self, state: InterviewState) -> Command[Literal["conversation_responder", "quality_assessor"]]:
        """
        Detect conversational intent and route accordingly.

        Args:
            state: Current interview state with candidate's latest response

        Returns:
            Command routing to conversation_responder (if conversational) or quality_assessor (if answer)
        """
        session_id = state.get("session_id", "unknown")
        start_time = time.time()
        logger.info(f"[{session_id}] Detecting conversational intent...")

        # Get latest message from candidate
        messages = state.get("messages", [])
        if not messages:
            logger.warning(f"[{session_id}] No messages found")
            # Default to quality assessment
            return Command(goto="quality_assessor")

        # Find last human message (support both LangChain objects and dict format)
        last_message = None
        for msg in reversed(messages):
            # Check for LangChain HumanMessage object
            if isinstance(msg, HumanMessage):
                last_message = msg
                break
            # Fallback to dict format (backward compatibility)
            elif isinstance(msg, dict) and msg.get("role") == "user":
                last_message = msg
                break

        if not last_message:
            logger.warning(f"[{session_id}] No user message found")
            return Command(goto="quality_assessor")

        # Extract content from either format
        if isinstance(last_message, HumanMessage):
            candidate_response = last_message.content
        else:
            candidate_response = last_message.get("content", "")
        current_question = state.get("current_question", "")
        current_topic = state.get("current_topic", "General")

        # Get conversation history for context-aware intent detection
        conversation_history = summarize_conversation(
            state.get("messages", []),
            max_exchanges=5  # Last 5 Q&A for pattern detection
        )

        # Get assessment goals for current topic
        resume_text = state.get("resume_text", "")
        jd_text = state.get("job_description_text", "")
        strategy_json = state.get("interview_time_strategy", "")

        context_data = extract_relevant_context(
            resume_text=resume_text,
            job_description_text=jd_text,
            current_topic=current_topic,
            interview_strategy=strategy_json,
            max_resume_chars=500,  # Smaller for intent detection
            max_jd_chars=500
        )

        # Create enhanced prompt for intent detection
        prompt = create_conversational_intent_prompt()
        messages_for_llm = prompt.format_messages(
            question=current_question,
            response=candidate_response,
            current_topic=current_topic,
            assessment_goals=context_data.get("assessment_goals", "General exploration"),
            conversation_history=conversation_history
        )

        try:
            # Track LLM call timing
            llm_start = time.time()
            logger.debug(f"[{session_id}] Invoking LLM for intent detection...")

            # Call LLM with retry logic
            turn_analysis: ConversationalTurnAnalysis = self._call_llm_with_retry(messages_for_llm)

            # Handle None return (validation failures after all retries)
            if turn_analysis is None:
                logger.error(f"[{session_id}] LLM returned None after retries (likely validation error)")
                raise ValueError("ConversationalTurnAnalysis returned None - validation failed after retries")

            llm_duration_ms = (time.time() - llm_start) * 1000
            log_llm_call(
                logger,
                agent_name="conversational_handler",
                latency_ms=llm_duration_ms,
                model=self.llm.model_name if hasattr(self.llm, 'model_name') else "gemini"
            )

            logger.info(
                f"[{session_id}] Intent detected: {turn_analysis.intent.value} "
                f"(contains_answer: {turn_analysis.contains_answer})"
            )

            # Store analysis in state for later use
            state_update = {
                "last_turn_analysis": turn_analysis.model_dump_json(),
                "pending_clarification": turn_analysis.intent == ResponseIntent.CLARIFICATION_REQUEST,
                "pending_continuation": turn_analysis.intent in [
                    ResponseIntent.THINKING_ALOUD,
                    ResponseIntent.ACKNOWLEDGMENT,
                    ResponseIntent.PARTIAL_ANSWER
                ]
            }

            # Routing decision based on intent
            if turn_analysis.needs_response:
                # Conversational turn - needs response before proceeding
                logger.info(f"[{session_id}] Routing to conversation_responder")

                # Record successful execution
                execution_time_ms = (time.time() - start_time) * 1000
                log_agent_execution(
                    logger,
                    agent_name="conversational_handler",
                    execution_time_ms=execution_time_ms,
                    success=True,
                    intent=turn_analysis.intent.value,
                    route="conversation_responder"
                )

                return Command(
                    goto="conversation_responder",
                    update=state_update
                )
            else:
                # Direct answer or mixed - process answer quality
                logger.info(f"[{session_id}] Routing to quality_assessor")

                # Record successful execution
                execution_time_ms = (time.time() - start_time) * 1000
                log_agent_execution(
                    logger,
                    agent_name="conversational_handler",
                    execution_time_ms=execution_time_ms,
                    success=True,
                    intent=turn_analysis.intent.value,
                    route="quality_assessor"
                )

                return Command(
                    goto="quality_assessor",
                    update=state_update
                )

        except Exception as e:
            # Record failed execution
            execution_time_ms = (time.time() - start_time) * 1000
            log_agent_execution(
                logger,
                agent_name="conversational_handler",
                execution_time_ms=execution_time_ms,
                success=False,
                error=str(e)
            )
            logger.error(f"[{session_id}] Error in intent detection: {e}", exc_info=True)
            # On error, default to quality assessment (safe fallback)
            return Command(
                goto="quality_assessor",
                update={"last_turn_analysis": None}
            )

    def _call_llm_with_retry(self, messages) -> ConversationalTurnAnalysis:
        """
        Call LLM with retry logic for intent detection.

        Args:
            messages: Formatted messages for LLM

        Returns:
            Validated ConversationalTurnAnalysis

        Raises:
            Exception: If all retries fail
        """
        @retry_llm_call
        def _invoke_llm():
            llm_with_structure = self.llm.with_structured_output(ConversationalTurnAnalysis)
            return llm_with_structure.invoke(messages)

        return _invoke_llm()

    def get_intent_summary(self, state: InterviewState) -> str:
        """
        Get human-readable summary of last intent detection.

        Args:
            state: Current interview state

        Returns:
            Summary string
        """
        turn_analysis_json = state.get("last_turn_analysis")
        if not turn_analysis_json:
            return "No intent analysis available"

        import json
        try:
            analysis_dict = json.loads(turn_analysis_json)
            intent = analysis_dict.get("intent", "unknown")
            contains_answer = analysis_dict.get("contains_answer", False)

            return (
                f"Intent: {intent}\n"
                f"Contains answer: {contains_answer}"
            )
        except Exception as e:
            logger.error(f"Error parsing turn analysis: {e}")
            return "Error parsing intent analysis"
