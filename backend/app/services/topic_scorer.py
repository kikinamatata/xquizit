"""
Topic Scoring Utility (V3)
Intelligent topic selection algorithm for strategic time allocation.
"""

from typing import Dict, List, Optional, Any
from app.prompts.schemas import TopicPriority, TopicAllocation, InterviewTimeStrategy, TopicAssessmentQuality
from app.prompts.optimization import _extract_role, _extract_content
import json
import logging

logger = logging.getLogger(__name__)


class TopicScorer:
    """
    Intelligent topic scoring and selection for V3 interview graph.

    Implements multi-factor scoring algorithm:
    1. Base priority (CRITICAL > HIGH > MEDIUM > LOW)
    2. Coverage gaps (under-explored topics get boost)
    3. Time urgency (critical topics boosted if time running low)
    4. Assessment quality (low coverage/confidence get boost)
    """

    # Base scores for each priority level
    PRIORITY_SCORES = {
        TopicPriority.CRITICAL: 100,
        TopicPriority.HIGH: 75,
        TopicPriority.MEDIUM: 50,
        TopicPriority.LOW: 25
    }

    def __init__(self, strategy: InterviewTimeStrategy, max_interview_time: float = 45.0):
        """
        Initialize topic scorer.

        Args:
            strategy: The InterviewTimeStrategy with topic allocations
            max_interview_time: Maximum interview duration in minutes (default: 45)
        """
        self.strategy = strategy
        self.max_interview_time = max_interview_time

        # Create lookup dict for quick access
        self.topic_allocations = {
            alloc.topic_name: alloc
            for alloc in strategy.topic_allocations
        }

    @classmethod
    def from_json(cls, strategy_json: str, max_interview_time: float = 45.0) -> 'TopicScorer':
        """
        Create TopicScorer from JSON-serialized InterviewTimeStrategy.

        Args:
            strategy_json: JSON string of InterviewTimeStrategy
            max_interview_time: Maximum interview duration in minutes

        Returns:
            TopicScorer instance
        """
        strategy_dict = json.loads(strategy_json)
        strategy = InterviewTimeStrategy(**strategy_dict)
        return cls(strategy, max_interview_time)

    def score_topics(
        self,
        topic_statistics: Dict[str, Dict[str, Any]],
        time_elapsed_minutes: float,
        topics_completed: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Score all topics and return sorted list.

        Args:
            topic_statistics: Per-topic statistics from InterviewState
            time_elapsed_minutes: Time elapsed so far
            topics_completed: List of topics already completed

        Returns:
            List of dicts with 'topic', 'score', 'allocation', 'stats', sorted by score descending
        """
        time_remaining = self.max_interview_time - time_elapsed_minutes
        scored_topics = []

        for alloc in self.strategy.topic_allocations:
            topic = alloc.topic_name

            # Skip if already completed
            if topic in topics_completed:
                logger.debug(f"Skipping completed topic: {topic}")
                continue

            # Get statistics (default to empty if not found)
            stats = topic_statistics.get(topic, {
                "questions_asked": 0,
                "time_spent_minutes": 0.0,
                "assessment_complete": False,
                "coverage": 0.0,
                "confidence": 0.0
            })

            # Calculate score
            score = self._calculate_topic_score(alloc, stats, time_remaining)

            scored_topics.append({
                "topic": topic,
                "score": score,
                "allocation": alloc,
                "stats": stats
            })

        # Sort by score descending
        scored_topics.sort(key=lambda x: x["score"], reverse=True)

        logger.info(f"Topic scores: {[(t['topic'], t['score']) for t in scored_topics]}")

        return scored_topics

    def _calculate_topic_score(
        self,
        allocation: TopicAllocation,
        stats: Dict[str, Any],
        time_remaining_minutes: float
    ) -> float:
        """
        Calculate score for a single topic.

        Args:
            allocation: TopicAllocation for this topic
            stats: Statistics for this topic
            time_remaining_minutes: Minutes remaining in interview

        Returns:
            Score (higher = higher priority)
        """
        # 1. Base priority score
        score = self.PRIORITY_SCORES[allocation.priority]

        questions_asked = stats.get("questions_asked", 0)
        coverage = stats.get("coverage", 0.0)
        confidence = stats.get("confidence", 0.0)
        time_spent = stats.get("time_spent_minutes", 0.0)

        # 2. Coverage adjustment
        if questions_asked < allocation.min_questions:
            # Haven't met minimum - big boost
            score += 50
            logger.debug(f"{allocation.topic_name}: +50 (under min questions: {questions_asked}/{allocation.min_questions})")

        if questions_asked >= allocation.max_questions:
            # Exceeded max - heavy penalty
            score -= 100
            logger.debug(f"{allocation.topic_name}: -100 (exceeded max questions: {questions_asked}/{allocation.max_questions})")

        # 3. Time urgency factor
        time_needed = allocation.estimated_minutes - time_spent

        if time_needed > time_remaining_minutes:
            # Running out of time for this topic
            if allocation.priority == TopicPriority.CRITICAL:
                score += 75  # URGENT! Critical topic running out of time
                logger.debug(f"{allocation.topic_name}: +75 (URGENT - critical topic, time running out)")
            elif allocation.priority == TopicPriority.HIGH:
                score += 40
                logger.debug(f"{allocation.topic_name}: +40 (HIGH priority, time running out)")

        # 4. Assessment quality factors
        if coverage < 0.5:
            # Less than 50% coverage - needs more attention
            score += 30
            logger.debug(f"{allocation.topic_name}: +30 (low coverage: {coverage})")

        if confidence < 0.6:
            # Low confidence in assessment
            score += 20
            logger.debug(f"{allocation.topic_name}: +20 (low confidence: {confidence})")

        # 5. Critical skill bonus
        if allocation.topic_name in self.strategy.critical_skills:
            score += 15
            logger.debug(f"{allocation.topic_name}: +15 (critical skill)")

        logger.debug(f"{allocation.topic_name}: FINAL SCORE = {score}")

        return score

    def select_next_topic(
        self,
        topic_statistics: Dict[str, Dict[str, Any]],
        time_elapsed_minutes: float,
        topics_completed: List[str]
    ) -> Optional[str]:
        """
        Select the next topic to explore.

        Args:
            topic_statistics: Per-topic statistics
            time_elapsed_minutes: Time elapsed in minutes
            topics_completed: Already completed topics

        Returns:
            Topic name to explore next, or None if all complete/time up
        """
        scored_topics = self.score_topics(topic_statistics, time_elapsed_minutes, topics_completed)

        if not scored_topics:
            logger.info("No topics remaining - all complete or excluded")
            return None

        # Select highest scoring topic
        next_topic = scored_topics[0]["topic"]
        next_score = scored_topics[0]["score"]

        logger.info(f"Selected next topic: {next_topic} (score: {next_score})")

        return next_topic

    def get_topic_allocation(self, topic_name: str) -> Optional[TopicAllocation]:
        """
        Get allocation for a specific topic.

        Args:
            topic_name: Name of the topic

        Returns:
            TopicAllocation or None if not found
        """
        return self.topic_allocations.get(topic_name)

    def estimate_topic_time(self, messages: List[Dict[str, Any]], topic_name: str) -> float:
        """
        Estimate time spent on a topic based on message timestamps.

        Args:
            messages: Conversation messages
            topic_name: Topic to estimate time for

        Returns:
            Estimated time in minutes
        """
        # This is a simplified estimation
        # In practice, would track actual timestamps per topic
        # Use helper functions to handle both LangChain objects and dicts
        topic_messages = [
            msg for msg in messages
            if _extract_role(msg) == "assistant" and topic_name.lower() in _extract_content(msg).lower()
        ]

        # Rough estimate: 1.5 minutes per Q&A exchange
        num_exchanges = len(topic_messages)
        estimated_time = num_exchanges * 1.5

        return estimated_time

    def should_conclude_interview(
        self,
        time_elapsed_minutes: float,
        topics_completed: List[str],
        critical_topics_covered: List[str]
    ) -> tuple[bool, Optional[str]]:
        """
        Determine if interview should conclude.

        Args:
            time_elapsed_minutes: Time elapsed so far
            topics_completed: Completed topics
            critical_topics_covered: Critical topics that have been covered

        Returns:
            (should_conclude: bool, reason: str)
        """
        # Check time limit
        if time_elapsed_minutes >= self.max_interview_time:
            return True, "Time limit reached (45 minutes)"

        # Check if all critical skills covered
        all_critical_covered = all(
            skill in critical_topics_covered or skill in topics_completed
            for skill in self.strategy.critical_skills
        )

        # If all critical covered and most topics complete
        completion_rate = len(topics_completed) / len(self.strategy.topic_allocations)

        if all_critical_covered and completion_rate >= 0.75:
            return True, "All critical topics covered and interview goals met"

        # Continue interview
        return False, None
