"""
Topic Selector Agent (V3)
Intelligently selects next topic using multi-factor scoring algorithm.

Follows LangGraph 2025 best practices:
- Deterministic algorithm (no LLM calls for efficiency)
- Uses TopicScorer utility for scoring logic
- Single responsibility: topic selection only
"""

import logging
from typing import Dict, Any, Optional, Literal
from langgraph.types import Command

from app.services.topic_scorer import TopicScorer
from app.core.models import InterviewState

logger = logging.getLogger(__name__)


class TopicSelectorAgent:
    """
    Specialized agent for intelligent topic selection.

    Responsibilities:
    - Score topics based on priority, coverage, urgency, quality
    - Select highest-priority topic for next question
    - Determine if interview should conclude

    Uses algorithmic scoring (no LLM calls) for fast, deterministic selection.
    """

    def __init__(self, max_interview_time: float = 45.0):
        """
        Initialize topic selector agent.

        Args:
            max_interview_time: Maximum interview duration in minutes (default: 45)
        """
        self.max_interview_time = max_interview_time
        logger.info("Initialized TopicSelectorAgent")

    def __call__(self, state: InterviewState) -> Command[Literal["question_generator", "conclude_interview"]]:
        """
        Select next topic or conclude interview.

        Args:
            state: Current interview state

        Returns:
            Command object routing to either question_generator or conclusion
        """
        session_id = state.get("session_id", "unknown")
        logger.info(f"[{session_id}] Selecting next topic...")

        # Get strategy and statistics
        strategy_json = state.get("interview_time_strategy")
        if not strategy_json:
            logger.error(f"[{session_id}] No interview strategy found!")
            # Fallback: conclude interview
            return Command(
                goto="conclude_interview",
                update={"conclusion_reason": "missing_strategy"}
            )

        topic_statistics = state.get("topic_statistics", {})
        topics_completed = state.get("topics_completed", [])
        critical_topics_covered = state.get("critical_topics_covered", [])
        time_elapsed = state.get("time_elapsed", 0)
        time_elapsed_minutes = time_elapsed / 60.0

        # Initialize TopicScorer
        try:
            scorer = TopicScorer.from_json(strategy_json, self.max_interview_time)
        except Exception as e:
            logger.error(f"[{session_id}] Error initializing TopicScorer: {e}")
            return Command(
                goto="conclude_interview",
                update={"conclusion_reason": "scorer_error"}
            )

        # Check if interview should conclude
        should_conclude, reason = scorer.should_conclude_interview(
            time_elapsed_minutes,
            topics_completed,
            critical_topics_covered
        )

        if should_conclude:
            logger.info(f"[{session_id}] Interview concluded: {reason}")
            return Command(
                goto="conclude_interview",
                update={"conclusion_reason": reason}
            )

        # Select next topic
        next_topic = scorer.select_next_topic(
            topic_statistics,
            time_elapsed_minutes,
            topics_completed
        )

        if not next_topic:
            logger.info(f"[{session_id}] No topics remaining, concluding interview")
            return Command(
                goto="conclude_interview",
                update={"conclusion_reason": "all_topics_complete"}
            )

        # Get topic allocation for context
        allocation = scorer.get_topic_allocation(next_topic)

        logger.info(
            f"[{session_id}] Selected topic: {next_topic} "
            f"(priority: {allocation.priority.value if allocation else 'unknown'})"
        )

        # Route to question generator with updated topic
        return Command(
            goto="question_generator",
            update={
                "current_topic": next_topic,
                # Store allocation details for question generator
                "current_topic_allocation": allocation.model_dump() if allocation else None
            }
        )

    def get_topic_summary(self, state: InterviewState) -> Optional[str]:
        """
        Get summary of topic coverage for logging/debugging.

        Args:
            state: Current interview state

        Returns:
            Human-readable summary or None if no strategy
        """
        strategy_json = state.get("interview_time_strategy")
        if not strategy_json:
            return None

        topic_statistics = state.get("topic_statistics", {})
        topics_completed = state.get("topics_completed", [])

        lines = ["Topic Coverage Summary:"]
        for topic_name, stats in topic_statistics.items():
            status = "✓ Complete" if topic_name in topics_completed else "⧗ In Progress"
            lines.append(
                f"  {status} {topic_name}: "
                f"{stats['questions_asked']} questions, "
                f"{stats['time_spent_minutes']:.1f} min, "
                f"coverage: {stats['coverage']:.0%}"
            )

        return "\n".join(lines)
