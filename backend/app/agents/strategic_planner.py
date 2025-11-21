"""
Strategic Planner Agent (V3)
Analyzes resume and job description to create time-allocated interview strategy.

Follows LangGraph 2025 best practices:
- Structured output with Pydantic schemas
- Type-safe state management
- Single responsibility principle
"""

import logging
import time
from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from app.prompts.schemas import InterviewTimeStrategy
from app.prompts.strategic_planning import create_strategic_allocation_prompt
from app.core.models import InterviewState
from app.utils.llm_retry import async_retry_llm_call
from app.utils.validation import call_llm_with_validation_retry_sync
from app.utils.metrics import track_agent_execution, record_llm_tokens
from app.utils.logging_config import log_agent_execution, log_llm_call

logger = logging.getLogger(__name__)


class StrategicPlannerAgent:
    """
    Specialized agent for strategic time allocation and interview planning.

    Responsibilities:
    - Analyze resume and job description
    - Create priority-based topic allocations
    - Identify critical skills and risk areas
    - Allocate time budgets across topics

    Runs once at interview start (not real-time critical).
    """

    def __init__(self, llm: ChatGoogleGenerativeAI):
        """
        Initialize strategic planner agent.

        Args:
            llm: LLM instance for document analysis (no speed constraints for quality)
        """
        self.llm = llm
        logger.info("Initialized StrategicPlannerAgent")

    def __call__(self, state: InterviewState) -> Dict[str, Any]:
        """
        Execute strategic planning analysis.

        Args:
            state: Current interview state with resume, JD, custom instructions

        Returns:
            State update with interview_time_strategy and initialized tracking fields
        """
        session_id = state.get("session_id", "unknown")
        start_time = time.time()

        logger.info(f"[{session_id}] Starting strategic time allocation")

        # Extract documents
        resume_text = state["resume_text"]
        jd_text = state["job_description_text"]
        custom_instructions = state.get("custom_instructions") or "None provided"

        # Create prompt using separated template
        prompt = create_strategic_allocation_prompt()
        messages = prompt.format_messages(
            resume_text=resume_text,
            job_description_text=jd_text,
            custom_instructions=custom_instructions
        )

        try:
            # Track LLM call timing
            llm_start = time.time()
            logger.info(f"[{session_id}] Invoking LLM for strategic allocation...")

            # Use structured output with retry and validation
            # Wrap the LLM call to add retry logic
            strategy: InterviewTimeStrategy = self._call_llm_with_retry(messages)

            llm_duration_ms = (time.time() - llm_start) * 1000
            log_llm_call(
                logger,
                agent_name="strategic_planner",
                latency_ms=llm_duration_ms,
                model=self.llm.model_name if hasattr(self.llm, 'model_name') else "gemini"
            )

            # Validate allocations
            if not strategy.validate_allocations():
                logger.warning(f"[{session_id}] Strategy validation failed (total time > 45 min)")
                # Strategy still usable, just exceeds time budget

            total_time = strategy.get_total_allocated_time()
            logger.info(
                f"[{session_id}] Generated strategy: "
                f"{len(strategy.topic_allocations)} topics, "
                f"{total_time:.1f} min allocated, "
                f"{len(strategy.critical_skills)} critical skills"
            )

            # Log topic breakdown
            for alloc in strategy.topic_allocations:
                logger.debug(
                    f"  - {alloc.topic_name}: {alloc.priority.value} priority, "
                    f"{alloc.estimated_minutes} min, "
                    f"{alloc.min_questions}-{alloc.max_questions} questions"
                )

            # Serialize strategy to JSON for state storage
            strategy_json = strategy.model_dump_json()

            # Initialize V3 tracking fields
            topic_statistics = {
                topic.topic_name: {
                    "questions_asked": 0,
                    "time_spent_minutes": 0.0,
                    "assessment_complete": False,
                    "coverage": 0.0,
                    "confidence": 0.0,
                    "last_quality_assessment": None
                }
                for topic in strategy.topic_allocations
            }

            # Record successful execution
            execution_time_ms = (time.time() - start_time) * 1000
            log_agent_execution(
                logger,
                agent_name="strategic_planner",
                execution_time_ms=execution_time_ms,
                success=True,
                topics_generated=len(strategy.topic_allocations),
                total_time_allocated=total_time
            )

            return {
                # V3 fields
                "interview_time_strategy": strategy_json,
                "topic_statistics": topic_statistics,
                "topics_completed": [],
                "critical_topics_covered": [],
                "key_topics": strategy.key_topics,
                "messages": []
            }

        except Exception as e:
            # Record failed execution
            execution_time_ms = (time.time() - start_time) * 1000
            log_agent_execution(
                logger,
                agent_name="strategic_planner",
                execution_time_ms=execution_time_ms,
                success=False,
                error=str(e)
            )
            logger.error(f"[{session_id}] Error in strategic planning: {e}", exc_info=True)
            raise

    def _call_llm_with_retry(self, messages) -> InterviewTimeStrategy:
        """
        Call LLM with retry logic and validation.

        Args:
            messages: Formatted messages for LLM

        Returns:
            Validated InterviewTimeStrategy

        Raises:
            Exception: If all retries fail or validation fails
        """
        from app.utils.llm_retry import retry_llm_call

        @retry_llm_call
        def _invoke_llm():
            llm_with_structure = self.llm.with_structured_output(InterviewTimeStrategy)
            return llm_with_structure.invoke(messages)

        # Call with retry logic
        return _invoke_llm()
