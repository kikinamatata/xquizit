"""
Quality Assessor Agent (V3)
Evaluates topic assessment quality and determines if more exploration needed.

Follows LangGraph 2025 best practices:
- Structured output with TopicAssessmentQuality Pydantic schema
- Command-based routing (follow-up vs topic complete)
- Dynamic quality assessment (no hard limits)
"""

import logging
import time
from typing import Dict, Any, Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.types import Command

from app.prompts.schemas import TopicAssessmentQuality, TopicAllocation, InterviewTimeStrategy
from app.prompts.strategic_planning import create_topic_quality_prompt
from app.prompts.optimization import extract_topic_context
from app.core.models import InterviewState
from app.utils.llm_retry import retry_llm_call
from app.utils.logging_config import log_agent_execution, log_llm_call
from app.utils.metrics import record_topic_quality

logger = logging.getLogger(__name__)


class QualityAssessorAgent:
    """
    Specialized agent for assessing topic exploration quality.

    Responsibilities:
    - Evaluate how well topic goals have been met
    - Assess coverage and confidence levels
    - Determine if more questions needed (no hard limits!)
    - Update topic statistics

    Uses LLM with structured output for nuanced quality judgments.
    """

    def __init__(self, llm: ChatGoogleGenerativeAI):
        """
        Initialize quality assessor agent.

        Args:
            llm: LLM instance for quality assessment (real-time optimized)
        """
        self.llm = llm
        logger.info("Initialized QualityAssessorAgent")

    def __call__(self, state: InterviewState) -> Command[Literal["question_generator", "topic_selector"]]:
        """
        Assess topic quality and route to follow-up or next topic.

        Args:
            state: Current interview state

        Returns:
            Command routing to question_generator (follow-up) or topic_selector (topic complete)
        """
        session_id = state.get("session_id", "unknown")
        start_time = time.time()
        current_topic = state.get("current_topic")

        if not current_topic:
            logger.warning(f"[{session_id}] No current topic, routing to topic selector")
            return Command(goto="topic_selector")

        logger.info(f"[{session_id}] Assessing quality for topic: {current_topic}")

        # Get strategy and allocation
        strategy_json = state.get("interview_time_strategy")
        if not strategy_json:
            logger.error(f"[{session_id}] No strategy found")
            return Command(goto="topic_selector")

        try:
            import json
            strategy_dict = json.loads(strategy_json)
            strategy = InterviewTimeStrategy(**strategy_dict)

            # Find current topic allocation
            allocation = next(
                (a for a in strategy.topic_allocations if a.topic_name == current_topic),
                None
            )

            if not allocation:
                logger.warning(f"[{session_id}] Topic '{current_topic}' not found in allocations")
                return Command(goto="topic_selector")

        except Exception as e:
            logger.error(f"[{session_id}] Error parsing strategy: {e}")
            return Command(goto="topic_selector")

        # Get topic statistics
        topic_statistics = state.get("topic_statistics", {})
        stats = topic_statistics.get(current_topic, {
            "questions_asked": 0,
            "time_spent_minutes": 0.0
        })

        questions_asked = stats.get("questions_asked", 0)
        time_spent = stats.get("time_spent_minutes", 0.0)

        # Get conversation context for this topic (full user messages preserved)
        messages = state.get("messages", [])
        topic_context = extract_topic_context(
            messages,
            current_topic,
            max_exchanges=6  # Richer context for quality assessment (full user messages)
        )

        # Create assessment prompt
        prompt = create_topic_quality_prompt()

        # Format assessment goals as bulleted list
        assessment_goals_text = "\n".join(
            f"- {goal}" for goal in allocation.assessment_goals
        )

        messages_for_llm = prompt.format_messages(
            topic_name=current_topic,
            topic_priority=allocation.priority.value,
            budget_minutes=allocation.estimated_minutes,
            time_spent_minutes=time_spent,
            min_questions=allocation.min_questions,
            max_questions=allocation.max_questions,
            questions_asked=questions_asked,
            assessment_goals=assessment_goals_text,
            topic_context=topic_context or "No conversation yet on this topic"
        )

        try:
            # Track LLM call timing
            llm_start = time.time()
            logger.debug(f"[{session_id}] Invoking LLM for quality assessment...")

            # Call LLM with retry logic
            quality: TopicAssessmentQuality = self._call_llm_with_retry(messages_for_llm)

            llm_duration_ms = (time.time() - llm_start) * 1000
            log_llm_call(
                logger,
                agent_name="quality_assessor",
                latency_ms=llm_duration_ms,
                model=self.llm.model_name if hasattr(self.llm, 'model_name') else "gemini"
            )

            logger.info(
                f"[{session_id}] Quality assessment: "
                f"coverage={quality.assessment_coverage:.0%}, "
                f"confidence={quality.confidence_level:.0%}, "
                f"needs_more={quality.needs_more_exploration}"
            )

            # Record quality metrics
            record_topic_quality(
                current_topic,
                coverage=quality.assessment_coverage,
                confidence=quality.confidence_level
            )

            # Update topic statistics
            updated_stats = {
                **stats,
                "questions_asked": quality.questions_asked,
                "time_spent_minutes": quality.time_spent_minutes,
                "coverage": quality.assessment_coverage,
                "confidence": quality.confidence_level,
                "assessment_complete": not quality.needs_more_exploration,
                "last_quality_assessment": quality.model_dump()
            }

            topic_statistics[current_topic] = updated_stats

            # Track topic completion
            topics_completed = state.get("topics_completed", [])
            critical_topics_covered = state.get("critical_topics_covered", [])

            if not quality.needs_more_exploration:
                if current_topic not in topics_completed:
                    topics_completed = topics_completed + [current_topic]
                    logger.info(f"[{session_id}] Topic '{current_topic}' marked complete")

                # Check if it's a critical topic
                if current_topic in strategy.critical_skills:
                    if current_topic not in critical_topics_covered:
                        critical_topics_covered = critical_topics_covered + [current_topic]
                        logger.info(f"[{session_id}] Critical topic '{current_topic}' covered")

            # Prepare state update
            state_update = {
                "topic_statistics": topic_statistics,
                "topics_completed": topics_completed,
                "critical_topics_covered": critical_topics_covered,
                "current_topic_quality": quality.model_dump_json()
            }

            # Routing decision
            if quality.needs_more_exploration:
                logger.info(f"[{session_id}] More exploration needed - routing to question_generator")

                # Record successful execution
                execution_time_ms = (time.time() - start_time) * 1000
                log_agent_execution(
                    logger,
                    agent_name="quality_assessor",
                    execution_time_ms=execution_time_ms,
                    success=True,
                    topic=current_topic,
                    coverage=quality.assessment_coverage,
                    confidence=quality.confidence_level,
                    route="question_generator"
                )

                return Command(
                    goto="question_generator",
                    update=state_update
                )
            else:
                logger.info(f"[{session_id}] Topic complete - routing to topic_selector")

                # Record successful execution
                execution_time_ms = (time.time() - start_time) * 1000
                log_agent_execution(
                    logger,
                    agent_name="quality_assessor",
                    execution_time_ms=execution_time_ms,
                    success=True,
                    topic=current_topic,
                    coverage=quality.assessment_coverage,
                    confidence=quality.confidence_level,
                    route="topic_selector"
                )

                return Command(
                    goto="topic_selector",
                    update=state_update
                )

        except Exception as e:
            # Record failed execution
            execution_time_ms = (time.time() - start_time) * 1000
            log_agent_execution(
                logger,
                agent_name="quality_assessor",
                execution_time_ms=execution_time_ms,
                success=False,
                error=str(e)
            )
            logger.error(f"[{session_id}] Error in quality assessment: {e}", exc_info=True)
            # Fallback: mark topic complete and move on
            return Command(
                goto="topic_selector",
                update={
                    "topic_statistics": topic_statistics,
                    "current_topic_quality": None
                }
            )

    def _call_llm_with_retry(self, messages) -> TopicAssessmentQuality:
        """
        Call LLM with retry logic for quality assessment.

        Args:
            messages: Formatted messages for LLM

        Returns:
            Validated TopicAssessmentQuality

        Raises:
            Exception: If all retries fail
        """
        @retry_llm_call
        def _invoke_llm():
            llm_with_structure = self.llm.with_structured_output(TopicAssessmentQuality)
            return llm_with_structure.invoke(messages)

        return _invoke_llm()
