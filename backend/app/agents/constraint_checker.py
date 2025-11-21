"""
Constraint Checker Agent (V3)
Monitors interview constraints and determines continuation vs conclusion.

Follows LangGraph 2025 best practices:
- Command-based routing
- Strategic constraint checking (not just hard limits)
- Integration with TopicScorer for intelligent decisions
"""

import logging
import time
from typing import Dict, Any, Literal
from langgraph.types import Command
from langgraph.graph import END

from app.services.topic_scorer import TopicScorer
from app.core.models import InterviewState

logger = logging.getLogger(__name__)


class ConstraintCheckerAgent:
    """
    Specialized agent for monitoring interview constraints.

    Responsibilities:
    - Check time elapsed vs time remaining
    - Evaluate coverage quality vs goals
    - Determine if interview should continue or conclude
    - Route to appropriate next step

    Uses strategic logic (not hard limits) for intelligent decisions.
    """

    def __init__(self, max_interview_time: float = 45.0):
        """
        Initialize constraint checker agent.

        Args:
            max_interview_time: Maximum interview duration in minutes (default: 45)
        """
        self.max_interview_time = max_interview_time
        self.max_interview_time_seconds = max_interview_time * 60
        logger.info(f"Initialized ConstraintCheckerAgent (max time: {max_interview_time} min)")

    def __call__(self, state: InterviewState) -> Command[Literal["conclude_interview"]] | Dict[str, Any]:
        """
        Check constraints and route to wait (continue) or conclude.

        Args:
            state: Current interview state

        Returns:
            Command routing to wait_for_answer (continue) or conclusion (end)
        """
        session_id = state.get("session_id", "unknown")
        logger.info(f"[{session_id}] Checking interview constraints...")

        # Update time tracking
        current_time = time.time()
        start_time = state.get("start_time", current_time)
        time_elapsed = current_time - start_time
        time_elapsed_minutes = time_elapsed / 60.0
        time_remaining_minutes = self.max_interview_time - time_elapsed_minutes

        logger.info(
            f"[{session_id}] Time: {time_elapsed_minutes:.1f}/{self.max_interview_time} min elapsed, "
            f"{time_remaining_minutes:.1f} min remaining"
        )

        # Get interview progress
        questions_asked = state.get("questions_asked", 0)
        topics_completed = state.get("topics_completed", [])
        critical_topics_covered = state.get("critical_topics_covered", [])

        logger.info(
            f"[{session_id}] Progress: {questions_asked} questions, "
            f"{len(topics_completed)} topics completed, "
            f"{len(critical_topics_covered)} critical topics covered"
        )

        # HARD TIME LIMIT (safety check)
        if time_elapsed >= self.max_interview_time_seconds:
            logger.info(f"[{session_id}] HARD time limit reached (45 min)")
            return Command(
                goto="conclude_interview",
                update={
                    "time_elapsed": time_elapsed,
                    "start_time": start_time,
                    "conclusion_reason": "time_limit_hard",
                    "is_concluded": True
                }
            )

        # STRATEGIC CONCLUSION CHECK (uses TopicScorer logic)
        strategy_json = state.get("interview_time_strategy")
        if strategy_json:
            try:
                scorer = TopicScorer.from_json(strategy_json, self.max_interview_time)
                topic_statistics = state.get("topic_statistics", {})

                should_conclude, reason = scorer.should_conclude_interview(
                    time_elapsed_minutes,
                    topics_completed,
                    critical_topics_covered
                )

                if should_conclude:
                    logger.info(f"[{session_id}] Strategic conclusion: {reason}")
                    return Command(
                        goto="conclude_interview",
                        update={
                            "time_elapsed": time_elapsed,
                            "start_time": start_time,
                            "conclusion_reason": reason,
                            "is_concluded": True
                        }
                    )

            except Exception as e:
                logger.error(f"[{session_id}] Error in strategic conclusion check: {e}")
                # Continue on error (safe fallback)

        # TIME WARNING (if approaching limit)
        if time_remaining_minutes < 5:
            logger.warning(
                f"[{session_id}] TIME WARNING: Only {time_remaining_minutes:.1f} min remaining!"
            )

        # Check if explicitly concluded
        if state.get("is_concluded", False):
            logger.info(f"[{session_id}] Interview explicitly concluded")
            return Command(
                goto="conclude_interview",
                update={
                    "time_elapsed": time_elapsed,
                    "start_time": start_time
                }
            )

        # CONTINUE - Wait for candidate answer (end graph execution)
        logger.info(f"[{session_id}] Constraints satisfied - waiting for answer")
        return {
            "time_elapsed": time_elapsed,
            "start_time": start_time
        }

    def get_status_summary(self, state: InterviewState) -> str:
        """
        Get human-readable status summary for logging/debugging.

        Args:
            state: Current interview state

        Returns:
            Status summary string
        """
        time_elapsed = state.get("time_elapsed", 0)
        time_elapsed_minutes = time_elapsed / 60.0
        time_remaining_minutes = self.max_interview_time - time_elapsed_minutes

        questions_asked = state.get("questions_asked", 0)
        topics_completed = state.get("topics_completed", [])
        critical_topics_covered = state.get("critical_topics_covered", [])

        # Get strategy for total topics
        strategy_json = state.get("interview_time_strategy")
        total_topics = 0
        total_critical = 0

        if strategy_json:
            try:
                import json
                from app.prompts.schemas import InterviewTimeStrategy
                strategy_dict = json.loads(strategy_json)
                strategy = InterviewTimeStrategy(**strategy_dict)
                total_topics = len(strategy.topic_allocations)
                total_critical = len(strategy.critical_skills)
            except:
                pass

        lines = [
            "=== Interview Status ===",
            f"Time: {time_elapsed_minutes:.1f}/{self.max_interview_time} min (remaining: {time_remaining_minutes:.1f} min)",
            f"Questions: {questions_asked}",
            f"Topics: {len(topics_completed)}/{total_topics} completed",
            f"Critical: {len(critical_topics_covered)}/{total_critical} covered"
        ]

        # Add time urgency indicator
        if time_remaining_minutes < 5:
            lines.append("⚠️  TIME CRITICAL")
        elif time_remaining_minutes < 10:
            lines.append("⏰ Time running low")

        return "\n".join(lines)
