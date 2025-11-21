"""
Conversation Responder Agent (V3)
Generates conversational responses for non-answer candidate turns.

Follows LangGraph 2025 best practices:
- Uses ConversationalHandler utility for response generation
- Ends graph execution to wait for user input
- Context-aware clarifications
"""

import logging
import time
from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage

from app.services.conversational_utils import (
    ConversationalHandler,
    create_conversational_context
)
from app.prompts.schemas import ConversationalTurnAnalysis, ResponseIntent
from app.prompts.optimization import _extract_content
from app.core.models import InterviewState
from app.utils.logging_config import log_agent_execution

logger = logging.getLogger(__name__)


class ConversationResponderAgent:
    """
    Specialized agent for generating conversational responses.

    Responsibilities:
    - Generate clarifications when candidate asks for help
    - Provide encouragement when candidate is thinking
    - Handle acknowledgments appropriately
    - Redirect off-topic conversations politely

    Uses ConversationalHandler utility for response logic.
    """

    def __init__(self, llm: ChatGoogleGenerativeAI):
        """
        Initialize conversation responder agent.

        Args:
            llm: LLM instance for clarification generation (real-time optimized)
        """
        self.llm = llm
        self.handler = ConversationalHandler(llm)
        logger.info("Initialized ConversationResponderAgent")

    def __call__(self, state: InterviewState) -> Dict[str, Any]:
        """
        Generate conversational response and end graph execution.

        Args:
            state: Current interview state with last_turn_analysis

        Returns:
            State update with conversational response (graph will end and wait for user input)
        """
        session_id = state.get("session_id", "unknown")
        start_time = time.time()
        logger.info(f"[{session_id}] Generating conversational response...")

        # Get turn analysis from state
        turn_analysis_json = state.get("last_turn_analysis")
        if not turn_analysis_json:
            logger.warning(f"[{session_id}] No turn analysis found, using default response")
            return self._default_response(state)

        try:
            turn_analysis = ConversationalHandler.from_json(turn_analysis_json)
            intent = turn_analysis.intent

            logger.info(f"[{session_id}] Responding to intent: {intent.value}")

            # Create enhanced context for response generation
            context = create_conversational_context(
                question=state.get("current_question", ""),
                topic=state.get("current_topic", ""),
                resume_text=state.get("resume_text", ""),
                job_description_text=state.get("job_description_text", ""),
                interview_strategy=state.get("interview_time_strategy", ""),
                messages=state.get("messages", []),
                max_resume_chars=1000,
                max_jd_chars=800
            )

            # Generate response with retry (handler internally handles LLM calls)
            response = self._generate_response_with_retry(turn_analysis, context, max_retries=3)

            logger.info(f"[{session_id}] Conversational response: {response[:100]}...")

            # Log successful execution
            execution_time_ms = (time.time() - start_time) * 1000
            log_agent_execution(
                logger,
                agent_name="conversation_responder",
                execution_time_ms=execution_time_ms,
                success=True,
                intent=intent.value
            )

            # Return state update and end graph execution (wait for candidate response)
            return {
                "current_question": response,  # Conversational response becomes current question
                "messages": [
                    AIMessage(
                        content=response,
                        additional_kwargs={
                            "conversational": True,
                            "intent": intent.value
                        }
                    )
                ]
            }

        except Exception as e:
            # Log failed execution
            execution_time_ms = (time.time() - start_time) * 1000
            log_agent_execution(
                logger,
                agent_name="conversation_responder",
                execution_time_ms=execution_time_ms,
                success=False,
                error=str(e)
            )
            logger.error(f"[{session_id}] Error generating conversational response: {e}", exc_info=True)
            return self._default_response(state)

    def _generate_response_with_retry(self, turn_analysis, context, max_retries=3):
        """
        Generate response with retry logic.

        Args:
            turn_analysis: Conversational turn analysis
            context: Conversational context
            max_retries: Maximum number of retries

        Returns:
            Generated response string

        Raises:
            Exception: If all retries fail
        """
        last_error = None

        for attempt in range(max_retries):
            try:
                return self.handler.generate_response(turn_analysis, context)
            except Exception as e:
                last_error = e
                logger.warning(f"Response generation attempt {attempt + 1}/{max_retries} failed: {e}")

                if attempt < max_retries - 1:
                    # Exponential backoff
                    import time
                    time.sleep(2 ** attempt)
                    continue
                else:
                    # All retries failed
                    raise

        raise last_error

    def _default_response(self, state: InterviewState) -> Dict[str, Any]:
        """
        Generate default fallback response.

        Args:
            state: Current interview state

        Returns:
            State update with default encouragement
        """
        session_id = state.get("session_id", "unknown")
        logger.info(f"[{session_id}] Using default conversational response")

        default_message = "Please take your time to answer. I'm listening."

        return {
            "current_question": default_message,
            "messages": [{
                "role": "assistant",
                "content": default_message,
                "conversational": True,
                "intent": "default"
            }]
        }

    def get_response_summary(self, state: InterviewState) -> str:
        """
        Get summary of last conversational response for logging.

        Args:
            state: Current interview state

        Returns:
            Summary string
        """
        messages = state.get("messages", [])

        # Find last conversational message (support both dict and LangChain formats)
        for msg in reversed(messages):
            # Check dict format (backward compatibility)
            if isinstance(msg, dict) and msg.get("conversational"):
                intent = msg.get("intent", "unknown")
                content = msg.get("content", "")[:100]
                return f"Last conversational response ({intent}): {content}..."
            # Check LangChain format
            elif hasattr(msg, 'additional_kwargs'):
                if msg.additional_kwargs.get("conversational"):
                    intent = msg.additional_kwargs.get("intent", "unknown")
                    content = _extract_content(msg)[:100]
                    return f"Last conversational response ({intent}): {content}..."

        return "No conversational responses yet"
