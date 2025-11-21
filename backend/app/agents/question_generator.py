"""
Question Generator Agent (V3)
Generates interview questions with streaming support.

Follows LangGraph 2025 best practices:
- Async streaming for real-time question delivery
- Context optimization for performance
- Separated prompt templates
- Type-safe state updates
"""

import logging
import time
from typing import Dict, Any, AsyncIterator, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage

from app.prompts.question_generation import (
    create_intro_question_prompt,
    create_topic_question_prompt,
    create_followup_question_prompt
)
from app.prompts.optimization import (
    summarize_conversation,
    summarize_strategy,
    extract_topic_context,
    extract_relevant_context
)
from app.core.models import InterviewState
from app.utils.llm_retry import async_retry_llm_call, LLMTimeoutError, LLMRateLimitError
from app.utils.logging_config import log_agent_execution, log_llm_call

logger = logging.getLogger(__name__)


class QuestionGeneratorAgent:
    """
    Specialized agent for interview question generation.

    Responsibilities:
    - Generate introductory questions (welcoming, conversational)
    - Generate topic questions (strategic, focused)
    - Generate follow-up questions (depth, clarification)
    - Support streaming for real-time delivery

    Uses optimized context to minimize token usage and latency.
    """

    def __init__(self, llm: ChatGoogleGenerativeAI):
        """
        Initialize question generator agent.

        Args:
            llm: LLM instance for question generation (real-time optimized)
        """
        self.llm = llm
        logger.info("Initialized QuestionGeneratorAgent")

    async def generate_question_stream(self, state: InterviewState) -> AsyncIterator[dict]:
        """
        Generate question with streaming support (async).

        This is the primary method for SSE streaming to the client.

        Args:
            state: Current interview state

        Yields:
            Chunks with 'type' and 'content' for streaming
            Final chunk with 'type': 'question_complete' and metadata
        """
        session_id = state.get("session_id", "unknown")
        questions_asked = state.get("questions_asked", 0)
        current_topic = state.get("current_topic")

        logger.info(
            f"[{session_id}] Generating question #{questions_asked + 1} "
            f"(topic: {current_topic or 'intro'})"
        )

        # Determine question type
        is_first_question = questions_asked == 0
        topic_stats = state.get("topic_statistics", {}).get(current_topic, {}) if current_topic else {}
        topic_questions_asked = topic_stats.get("questions_asked", 0)
        is_follow_up = topic_questions_asked > 0

        # Select appropriate prompt and build context
        if is_first_question:
            prompt, current_topic = self._build_intro_prompt(state)
        elif is_follow_up:
            prompt = self._build_followup_prompt(state, current_topic)
        else:
            prompt = self._build_topic_prompt(state, current_topic)

        # Stream the question with retry and error handling
        stream_start_time = time.time()
        accumulated_content = ""
        chunk_count = 0
        retry_count = 0
        max_retries = 3

        while retry_count < max_retries:
            try:
                async for chunk in self.llm.astream(prompt):
                    if hasattr(chunk, 'content') and chunk.content:
                        accumulated_content += chunk.content
                        chunk_count += 1
                        yield {
                            "type": "text_chunk",
                            "content": chunk.content
                        }

                stream_duration = time.time() - stream_start_time
                logger.info(
                    f"[{session_id}] Question generated in {stream_duration:.2f}s "
                    f"({chunk_count} chunks, {len(accumulated_content)} chars)"
                )

                # Log LLM call metrics
                log_llm_call(
                    logger,
                    agent_name="question_generator",
                    latency_ms=stream_duration * 1000,
                    model=self.llm.model_name if hasattr(self.llm, 'model_name') else "gemini"
                )

                # Clean and finalize question
                question = self._clean_question_response(accumulated_content.strip())

                # Yield completion metadata
                yield {
                    "type": "question_complete",
                    "question": question,
                    "current_topic": current_topic,
                    "questions_asked": questions_asked + 1,
                    "is_first_question": is_first_question,
                    "is_follow_up": is_follow_up,
                    "generation_time": stream_duration
                }
                break  # Success, exit retry loop

            except Exception as e:
                retry_count += 1
                error_msg = str(e).lower()

                # Classify error for appropriate handling
                if "429" in error_msg or "rate limit" in error_msg:
                    logger.warning(f"[{session_id}] Rate limit hit (attempt {retry_count}/{max_retries})")
                    if retry_count < max_retries:
                        import asyncio
                        await asyncio.sleep(2 ** retry_count)  # Exponential backoff
                        continue
                elif "timeout" in error_msg:
                    logger.warning(f"[{session_id}] Timeout (attempt {retry_count}/{max_retries})")
                    if retry_count < max_retries:
                        continue

                # Log error
                logger.error(f"[{session_id}] Error generating question (attempt {retry_count}): {e}", exc_info=True)

                if retry_count >= max_retries:
                    # All retries exhausted, yield error
                    yield {
                        "type": "error",
                        "content": "Error generating question after retries",
                        "error": str(e)
                    }
                    break

    def __call__(self, state: InterviewState) -> Dict[str, Any]:
        """
        Synchronous wrapper for graph execution.

        LangGraph nodes must return Dict, not AsyncIterator.
        This runs async streaming synchronously and collects final result.

        Args:
            state: Current interview state

        Returns:
            State update with current_question, current_topic, questions_asked
        """
        import asyncio
        import nest_asyncio

        session_id = state.get("session_id", "unknown")
        questions_asked = state.get("questions_asked", 0)
        start_time = time.time()

        logger.info(f"[{session_id}] Question generator (sync wrapper) for question #{questions_asked + 1}")

        # Collect final result from async generator
        async def collect_result():
            accumulated_data = {}
            async for chunk in self.generate_question_stream(state):
                if chunk["type"] == "question_complete":
                    accumulated_data = chunk
                elif chunk["type"] == "error":
                    # Return error data
                    accumulated_data = {"error": chunk.get("error")}
            return accumulated_data

        # Run async code
        try:
            nest_asyncio.apply()
            result = asyncio.run(collect_result())
        except RuntimeError:
            # If event loop already running, use it
            try:
                loop = asyncio.get_event_loop()
                result = loop.run_until_complete(collect_result())
            except Exception as e:
                logger.error(f"[{session_id}] Failed to run async collection: {e}")
                result = {}

        if not result or "error" in result:
            error_msg = result.get("error", "Unknown error") if result else "No question generated"
            logger.error(f"[{session_id}] Question generation failed: {error_msg}")

            # Log failed execution
            execution_time_ms = (time.time() - start_time) * 1000
            log_agent_execution(
                logger,
                agent_name="question_generator",
                execution_time_ms=execution_time_ms,
                success=False,
                error=error_msg
            )

            return {
                "current_question": "I apologize, but I'm having trouble generating the next question. Could you please wait a moment?",
                "questions_asked": questions_asked + 1
            }

        # Log successful execution
        execution_time_ms = (time.time() - start_time) * 1000
        current_topic = result.get("current_topic")

        log_agent_execution(
            logger,
            agent_name="question_generator",
            execution_time_ms=execution_time_ms,
            success=True,
            topic=current_topic,
            is_follow_up=result.get("is_follow_up", False)
        )

        # Update topic statistics for time tracking
        if current_topic:
            topic_statistics = state.get("topic_statistics", {})
            if current_topic in topic_statistics:
                # Estimate time spent on this question (will be refined later)
                topic_statistics[current_topic]["questions_asked"] += 1

        return {
            "current_question": result.get("question", ""),
            "current_topic": current_topic,
            "questions_asked": result.get("questions_asked", questions_asked + 1),
            "messages": [
                AIMessage(
                    content=result.get("question", ""),
                    additional_kwargs={"topic": current_topic}
                )
            ]
        }

    def _build_intro_prompt(self, state: InterviewState) -> tuple:
        """Build prompt for introductory question."""
        prompt = create_intro_question_prompt()
        messages = prompt.format_messages()
        return messages, "introduction"

    def _build_topic_prompt(self, state: InterviewState, current_topic: str):
        """
        Build prompt for topic question with enhanced context.

        IMPROVED: Now includes resume/JD excerpts, assessment goals, and better strategy summary.
        """
        import json

        # Get strategy using enhanced summarization (preserves more detail)
        strategy_json = state.get("interview_time_strategy", "")
        strategy_summary = summarize_strategy(strategy_json)  # Now preserves 800 chars

        # Extract relevant resume and JD context for this topic
        resume_text = state.get("resume_text", "")
        jd_text = state.get("job_description_text", "")

        context_data = extract_relevant_context(
            resume_text=resume_text,
            job_description_text=jd_text,
            current_topic=current_topic,
            interview_strategy=strategy_json,
            max_resume_chars=1000,
            max_jd_chars=1000
        )

        # Get expanded conversation summary (8 exchanges, 500 chars each)
        conversation_summary = summarize_conversation(
            state.get("messages", []),
            max_exchanges=8  # IMPROVED: Last 8 Q&A for richer context (up from 2)
        )

        # Build covered topics list
        topics_completed = state.get("topics_completed", [])
        covered_topics = ", ".join(topics_completed) if topics_completed else "None yet"

        # Format prompt with enhanced context
        prompt = create_topic_question_prompt()
        messages = prompt.format_messages(
            strategy_summary=strategy_summary,
            current_topic=current_topic,
            assessment_goals=context_data.get("assessment_goals", "General exploration"),
            covered_topics=covered_topics,
            resume_context=context_data.get("resume_context", "No resume context available"),
            jd_context=context_data.get("jd_context", "No job description context available"),
            conversation_summary=conversation_summary
        )

        return messages

    def _build_followup_prompt(self, state: InterviewState, current_topic: str):
        """
        Build prompt for follow-up question with enhanced context.

        IMPROVED: Now includes resume excerpts and assessment goals for better follow-ups.
        """
        # Get topic-specific conversation (6 exchanges with 500 char messages)
        topic_context = extract_topic_context(
            state.get("messages", []),
            current_topic,
            max_exchanges=6  # IMPROVED: Last 6 exchanges for deeper context (up from 3)
        )

        # Extract relevant resume context for this topic
        resume_text = state.get("resume_text", "")
        jd_text = state.get("job_description_text", "")
        strategy_json = state.get("interview_time_strategy", "")

        context_data = extract_relevant_context(
            resume_text=resume_text,
            job_description_text=jd_text,
            current_topic=current_topic,
            interview_strategy=strategy_json,
            max_resume_chars=1000,
            max_jd_chars=800  # Slightly less for follow-ups
        )

        # Get follow-up number from topic statistics
        topic_stats = state.get("topic_statistics", {}).get(current_topic, {})
        followup_number = topic_stats.get("questions_asked", 0)

        # Format prompt with enhanced context
        prompt = create_followup_question_prompt()
        messages = prompt.format_messages(
            current_topic=current_topic,
            followup_number=followup_number,
            assessment_goals=context_data.get("assessment_goals", "General exploration"),
            resume_context=context_data.get("resume_context", "No resume context available"),
            topic_context=topic_context or "No previous conversation on this topic"
        )

        return messages

    def _clean_question_response(self, response: str) -> str:
        """
        Clean LLM response to extract only the question.

        Args:
            response: Raw LLM output

        Returns:
            Cleaned question text
        """
        # Check for conversation markers
        if "Interviewer:" in response or "Candidate:" in response:
            logger.warning("LLM response contains conversation context, cleaning...")

            lines = response.split('\n')
            question_lines = []
            capture = False

            for line in lines:
                line = line.strip()
                if line.startswith("Interviewer:"):
                    question_lines = [line.replace("Interviewer:", "").strip()]
                    capture = True
                elif capture and line and not line.startswith("Candidate:"):
                    question_lines.append(line)
                elif line.startswith("Candidate:"):
                    capture = False

            if question_lines:
                cleaned = " ".join(question_lines).strip()
                logger.debug(f"Cleaned response from {len(response)} to {len(cleaned)} chars")
                return cleaned

        return response.strip()
