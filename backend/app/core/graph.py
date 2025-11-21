"""
Interview Graph Module V3
Hybrid Modular State Machine Architecture

Key improvements over V2:
- Specialized agent modules (modular, testable, independent)
- Conversational turn handling (detects clarifications, thinking, etc.)
- Strategic time allocation (priority-based, no hard limits)
- Quality-driven follow-ups (coverage + confidence, unlimited)
- Intelligent topic selection (multi-factor scoring)

Architecture:
- Explicit State Machine (predictable flow via conditional edges)
- Specialized Agent Nodes (each node is an independent agent)
- Deterministic Routing (no supervisor LLM overhead)
- LangGraph 2025 best practices (Command objects, structured output, async streaming)
"""

import logging
from typing import Literal, Dict, Any, AsyncIterator
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from app.core.models import InterviewState
from app.core.constants import MAX_INTERVIEW_TIME_MINUTES
from app.agents import (
    StrategicPlannerAgent,
    TopicSelectorAgent,
    ConversationalHandlerAgent,
    QualityAssessorAgent,
    QuestionGeneratorAgent,
    ConstraintCheckerAgent,
    ConversationResponderAgent
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InterviewGraphBuilderV3:
    """
    Builds and manages the V3 interview workflow graph.

    V3 Architecture: Hybrid Modular State Machine
    - Combines predictable state machine flow with modular agent design
    - Each node is a specialized agent with clear responsibilities
    - Routing via conditional edges (deterministic, no LLM overhead)
    - Supports conversational intelligence and strategic time allocation
    """

    def __init__(
        self,
        gemini_api_key: str,
        model_name: str = "gemini-2.5-flash",
        thinking_budget: int = 0,
        include_thoughts: bool = False,
        max_output_tokens: int = 1024,
        temperature: float = 0.7
    ):
        """
        Initialize the V3 interview graph builder.

        Args:
            gemini_api_key: Google Gemini API key
            model_name: Gemini model to use (default: gemini-2.5-flash)
            thinking_budget: Thinking budget for LLM (default: 0 - disabled for speed)
            include_thoughts: Include thoughts in LLM response (default: False)
            max_output_tokens: Maximum output tokens (default: 1024)
            temperature: Temperature for randomness (default: 0.7)
        """
        # LLM for document analysis (quality-first, no speed constraints)
        self.llm_analysis = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=gemini_api_key
        )

        # LLM for real-time operations (speed-optimized)
        self.llm_realtime = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=gemini_api_key,
            thinking_budget=thinking_budget,
            include_thoughts=include_thoughts,
            max_output_tokens=max_output_tokens
        )

        # Initialize specialized agents
        self.strategic_planner = StrategicPlannerAgent(self.llm_analysis)
        self.topic_selector = TopicSelectorAgent(max_interview_time=MAX_INTERVIEW_TIME_MINUTES)
        self.conversational_handler = ConversationalHandlerAgent(self.llm_realtime)
        self.quality_assessor = QualityAssessorAgent(self.llm_realtime)
        self.question_generator = QuestionGeneratorAgent(self.llm_realtime)
        self.constraint_checker = ConstraintCheckerAgent(max_interview_time=MAX_INTERVIEW_TIME_MINUTES)
        self.conversation_responder = ConversationResponderAgent(self.llm_realtime)

        self.graph = None
        self._build_graph()

    def _build_graph(self):
        """Build the LangGraph state machine with all specialized agents."""
        builder = StateGraph(InterviewState)

        # Add all agent nodes
        builder.add_node("strategic_planner", self.strategic_planner)
        builder.add_node("topic_selector", self.topic_selector)
        builder.add_node("conversational_handler", self.conversational_handler)
        builder.add_node("quality_assessor", self.quality_assessor)
        builder.add_node("question_generator", self.question_generator)
        builder.add_node("constraint_checker", self.constraint_checker)
        builder.add_node("conversation_responder", self.conversation_responder)
        builder.add_node("conclude_interview", self._conclude_interview)

        # Define graph flow with conditional routing

        # START routing: first-time vs resume
        builder.add_conditional_edges(
            START,
            self._route_start,
            {
                "strategic_planner": "strategic_planner",          # First time: plan strategy
                "conversational_handler": "conversational_handler", # Resume: process answer
                "topic_selector": "topic_selector"                  # Resume: no messages yet
            }
        )

        # After strategic planning → select first topic
        builder.add_edge("strategic_planner", "topic_selector")

        # Topic selector routes to question_generator or conclusion (uses Command in agent)
        # No need for conditional_edges here - agent returns Command with goto

        # Question generator → constraint checker
        builder.add_edge("question_generator", "constraint_checker")

        # Constraint checker: routes to conclude_interview OR ends graph execution (waits for user input)
        # Returns Command(goto="conclude_interview") if should end, otherwise returns dict (graph ends)

        # Conversational handler routes to conversation_responder or quality_assessor (uses Command in agent)
        # No need for conditional_edges here - agent returns Command with goto

        # Conversation responder: ends graph execution after generating conversational response
        # Returns dict (graph ends naturally and waits for user input)

        # Quality assessor routes to question_generator or topic_selector (uses Command in agent)
        # No need for conditional_edges here - agent returns Command with goto

        # Conclude interview → END
        builder.add_edge("conclude_interview", END)

        # Compile the graph
        self.graph = builder.compile()
        logger.info("Interview graph V3 compiled successfully (Hybrid Modular State Machine)")

    def _route_start(self, state: InterviewState) -> str:
        """
        Determine routing from START based on state.

        Returns:
            "strategic_planner" - First time, need to analyze documents
            "conversational_handler" - Resume with messages, process latest answer
            "topic_selector" - Resume without messages, continue interview
        """
        has_strategy = bool(state.get("interview_time_strategy"))
        has_messages = len(state.get("messages", [])) > 1  # More than system message

        if not has_strategy:
            logger.info("START routing: strategic_planner (first time)")
            return "strategic_planner"
        elif has_messages:
            logger.info("START routing: conversational_handler (resume with answer)")
            return "conversational_handler"
        else:
            logger.info("START routing: topic_selector (resume without answer)")
            return "topic_selector"

    def _conclude_interview(self, state: InterviewState) -> Dict[str, Any]:
        """
        Generate conclusion message for the interview.

        Args:
            state: Current interview state

        Returns:
            State update with conclusion message
        """
        session_id = state.get("session_id", "unknown")
        logger.info(f"[{session_id}] Concluding interview (V3)")

        time_elapsed = state.get("time_elapsed", 0)
        time_elapsed_minutes = time_elapsed / 60.0
        questions_asked = state.get("questions_asked", 0)
        topics_completed = state.get("topics_completed", [])
        reason = state.get("conclusion_reason", "completed")

        logger.info(
            f"[{session_id}] Interview concluded: {reason} "
            f"({time_elapsed_minutes:.1f} min, {questions_asked} questions, "
            f"{len(topics_completed)} topics)"
        )

        # Generate appropriate conclusion message based on reason
        if reason in ["time_limit", "time_limit_hard"]:
            conclusion_message = (
                f"Thank you for your time! We've reached the 45-minute mark for this screening interview. "
                f"We covered {len(topics_completed)} topic areas, and I appreciate your detailed responses. "
                f"We'll be in touch soon regarding next steps."
            )
        elif reason == "all_topics_complete":
            conclusion_message = (
                f"Excellent! We've covered all the key areas I wanted to explore today. "
                f"Thank you for your thoughtful answers across {len(topics_completed)} topics. "
                f"We have all the information we need for this screening round. "
                f"We'll review your responses and get back to you soon about next steps."
            )
        elif "critical" in reason.lower():
            conclusion_message = (
                f"Thank you so much for your responses! We've covered all the critical areas for this role. "
                f"I have a good understanding of your background and experience. "
                f"We'll be in touch regarding the next steps in the interview process."
            )
        else:
            conclusion_message = (
                f"Thank you for taking the time to interview with us today. "
                f"We appreciate your interest in the position and will be in touch regarding next steps."
            )

        return {
            "is_concluded": True,
            "conclusion_reason": reason,
            "current_question": conclusion_message,
            "messages": [{
                "role": "assistant",
                "content": conclusion_message,
                "conclusion": True
            }]
        }

    def invoke(self, initial_state: Dict[str, Any]) -> InterviewState:
        """
        Run the interview graph with initial state.

        Args:
            initial_state: Initial state for the interview

        Returns:
            Final interview state
        """
        session_id = initial_state.get("session_id", "unknown")
        try:
            logger.info(f"[{session_id}] Starting interview graph V3 (Hybrid Modular State Machine)")
            result = self.graph.invoke(initial_state)
            logger.info(f"[{session_id}] Interview graph V3 execution completed")
            return result
        except Exception as e:
            logger.error(f"[{session_id}] Error executing interview graph V3: {str(e)}", exc_info=True)
            raise

    async def astream(self, initial_state: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream the interview graph execution asynchronously.

        Args:
            initial_state: Initial state for the interview

        Yields:
            State updates as they occur
        """
        session_id = initial_state.get("session_id", "unknown")
        try:
            logger.info(f"[{session_id}] Starting interview graph V3 stream")
            async for chunk in self.graph.astream(initial_state):
                yield chunk
            logger.info(f"[{session_id}] Interview graph V3 stream completed")
        except Exception as e:
            logger.error(f"[{session_id}] Error streaming interview graph V3: {str(e)}", exc_info=True)
            raise

    def get_question_generator(self) -> QuestionGeneratorAgent:
        """
        Get direct access to question generator for SSE streaming.

        Returns:
            QuestionGeneratorAgent instance
        """
        return self.question_generator

    def get_graph_visualization(self) -> bytes:
        """
        Generate visual representation of the graph.

        Returns:
            PNG bytes of graph visualization
        """
        try:
            return self.graph.get_graph().draw_mermaid_png()
        except Exception as e:
            logger.error(f"Error generating graph visualization: {e}")
            return b""


def create_interview_graph_v3(
    gemini_api_key: str,
    model_name: str = "gemini-2.5-flash",
    thinking_budget: int = 0,
    include_thoughts: bool = False,
    max_output_tokens: int = 1024,
    temperature: float = 0.7
) -> InterviewGraphBuilderV3:
    """
    Factory function to create an interview graph V3.

    Args:
        gemini_api_key: Google Gemini API key
        model_name: Gemini model to use
        thinking_budget: Thinking budget for LLM (default: 0 - disabled)
        include_thoughts: Include thoughts in LLM response (default: False)
        max_output_tokens: Maximum output tokens (default: 1024)
        temperature: Temperature for randomness (default: 0.7)

    Returns:
        Configured InterviewGraphBuilderV3 instance
    """
    return InterviewGraphBuilderV3(
        gemini_api_key=gemini_api_key,
        model_name=model_name,
        thinking_budget=thinking_budget,
        include_thoughts=include_thoughts,
        max_output_tokens=max_output_tokens,
        temperature=temperature
    )
