"""
Performance Optimization Utilities for Prompts
Balanced context management to maintain quality while managing token usage.
Emphasis on preserving important context rather than aggressive truncation.
"""

import re
from typing import Any, Dict


def _smart_truncate(text: str, max_length: int) -> str:
    """
    Intelligently truncate text at sentence boundaries when possible.

    Args:
        text: Text to truncate
        max_length: Maximum character length

    Returns:
        Truncated text that ends at a sentence boundary if possible
    """
    if len(text) <= max_length:
        return text

    # Try to find the last sentence boundary within max_length
    truncated = text[:max_length]

    # Look for sentence endings (. ! ?)
    sentence_endings = ['.', '!', '?']
    last_sentence_end = -1

    for ending in sentence_endings:
        pos = truncated.rfind(ending)
        if pos > last_sentence_end:
            last_sentence_end = pos

    # If we found a sentence boundary and it's not too far back (at least 70% of max_length)
    if last_sentence_end > max_length * 0.7:
        return text[:last_sentence_end + 1].strip()

    # Otherwise, try to break at a word boundary
    last_space = truncated.rfind(' ')
    if last_space > max_length * 0.8:
        return text[:last_space].strip() + "..."

    # Last resort: hard truncate
    return truncated.strip() + "..."


def compress_text(text: str | None, max_length: int = 2000) -> str:
    """
    Compress text to maximum length.
    Takes first max_length characters to reduce token usage.

    Performance benefit: 50-60% token reduction for long documents.

    Args:
        text: Text to compress (can be None)
        max_length: Maximum character length (default: 2000)

    Returns:
        Compressed text (first max_length chars), or empty string if None
    """
    if not text:
        return ""

    if len(text) <= max_length:
        return text

    # Take first max_length chars (key info usually at start)
    return text[:max_length] + f"\n\n[Truncated at {max_length} chars for performance]"


def compress_custom_instructions(instructions: str | None, max_length: int = 300) -> str:
    """
    Compress custom instructions to essential points.

    Performance benefit: Variable, up to 90% token reduction.

    Args:
        instructions: Custom instructions text (can be None)
        max_length: Maximum character length (default: 300)

    Returns:
        Compressed instructions or empty string if None
    """
    if not instructions:
        return ""

    # If short enough, return as-is
    if len(instructions) <= max_length:
        return instructions

    # Try to extract bullet points if present
    lines = instructions.split('\n')
    bullet_lines = [line.strip() for line in lines if line.strip().startswith(('-', '*', '•'))]

    if bullet_lines:
        # Join bullet points and truncate if needed
        bullet_text = ' '.join(bullet_lines)
        if len(bullet_text) <= max_length:
            return bullet_text
        return bullet_text[:max_length] + "..."

    # Otherwise just truncate
    return instructions[:max_length] + "..."


def summarize_conversation(messages: list[Any], max_exchanges: int = 8) -> str:
    """
    Create summary of recent conversation with expanded context.
    Includes last N Q&A exchanges with smart truncation to preserve meaning.

    IMPROVED: Keeps full user messages to preserve detailed answers.
    Only truncates assistant messages (questions). This ensures follow-up
    questions can reference specific details from candidate responses.

    Args:
        messages: List of message objects (dict or LangChain message objects)
        max_exchanges: Maximum number of Q&A exchanges to include (default: 8, up from 2)

    Returns:
        Formatted conversation summary (last N exchanges with preserved user responses)
    """
    if not messages:
        return "This is the first question."

    # Filter to only assistant and user messages
    relevant_messages = []
    for msg in messages:
        role = _extract_role(msg)
        content = _extract_content(msg)

        if role in ["assistant", "user"] and content:
            # Keep user messages fully (they contain important answer details)
            # Only truncate assistant messages (questions are usually shorter)
            if role == "assistant" and len(content) > 500:
                content = _smart_truncate(content, 500)
            elif role == "user" and len(content) > 3000:
                # Safety limit for very long user responses
                content = _smart_truncate(content, 3000)

            relevant_messages.append((role, content))

    # Take only last max_exchanges * 2 messages (Q&A pairs)
    recent_messages = relevant_messages[-(max_exchanges * 2):]

    # Format for display
    lines = []
    for role, content in recent_messages:
        speaker = "Interviewer" if role == "assistant" else "Candidate"
        lines.append(f"{speaker}: {content}")

    return "\n".join(lines) if lines else "This is the first question."


def extract_topic_context(messages: list[Any], current_topic: str, max_exchanges: int = 6) -> str:
    """
    Extract conversation relevant to current topic with expanded context.
    Used for generating follow-up questions with better awareness.

    IMPROVED: Keeps full user messages (no truncation) to preserve detailed answers.
    Only truncates assistant messages to save tokens. After 6 exchanges, summarizes
    older messages while keeping last 2-3 exchanges fully.

    Args:
        messages: List of message objects
        current_topic: Current topic to filter by
        max_exchanges: Maximum number of exchanges to include (default: 6, up from 2)

    Returns:
        Topic-filtered conversation context with preserved user responses
    """
    if not messages:
        return "First question on this topic"

    topic_messages = []
    for msg in messages:
        role = _extract_role(msg)
        content = _extract_content(msg)
        msg_topic = _extract_topic(msg)

        # Only include messages for current topic
        if msg_topic == current_topic and role in ["assistant", "user"]:
            # CRITICAL: Keep user messages fully (they contain important answer details)
            # Only truncate assistant messages (questions are usually shorter anyway)
            if role == "assistant" and len(content) > 500:
                content = _smart_truncate(content, 500)
            # User messages: keep entirely (up to 3000 chars max for safety)
            elif role == "user" and len(content) > 3000:
                content = _smart_truncate(content, 3000)

            speaker = "Interviewer" if role == "assistant" else "Candidate"
            topic_messages.append(f"{speaker}: {content}")

    # If more than max_exchanges, summarize older ones and keep recent ones full
    if len(topic_messages) > (max_exchanges * 2):
        # Summarize first half of exchanges
        older_messages = topic_messages[:-(6)]  # All except last 3 exchanges (6 messages)
        recent_messages = topic_messages[-(6):]  # Last 3 exchanges fully

        # Create summary of older exchanges
        summary_lines = ["[Earlier in conversation]"]
        for msg in older_messages[::2]:  # Take every other (candidate responses)
            if "Candidate:" in msg:
                # Extract just first 150 chars of each older answer
                answer = msg.replace("Candidate: ", "")
                summary_lines.append(f"- Candidate mentioned: {_smart_truncate(answer, 150)}")

        summary = "\n".join(summary_lines)
        recent = "\n".join(recent_messages)

        return f"{summary}\n\n[Recent conversation]\n{recent}"

    # If within limit, return all messages
    recent_topic_messages = topic_messages[-(max_exchanges * 2):]

    return "\n".join(recent_topic_messages) if recent_topic_messages else "First question on this topic"


def summarize_strategy(strategy: str, max_length: int = 800) -> str:
    """
    Provide comprehensive strategy context for question generation.

    IMPROVED: Increased from 150 to 800 chars to preserve critical details
    like assessment goals, priorities, and risk areas.

    Args:
        strategy: Full interview strategy text (JSON or formatted text)
        max_length: Maximum character length (default: 800, up from 150)

    Returns:
        Strategy summary with preserved important details
    """
    if not strategy:
        return "General screening interview"

    if len(strategy) <= max_length:
        return strategy

    # Try to parse as JSON to extract structured data
    import json
    try:
        strategy_obj = json.loads(strategy)
        # Extract key fields for summary
        summary_parts = []

        # Add topic allocations if present
        if "topic_allocations" in strategy_obj:
            topics = strategy_obj["topic_allocations"]
            topic_names = [t.get("topic_name", "") for t in topics[:5]]  # First 5 topics
            summary_parts.append(f"Topics: {', '.join(topic_names)}")

            # Add assessment goals for critical topics
            for topic in topics[:3]:  # Top 3 topics
                if topic.get("priority") in ["CRITICAL", "HIGH"]:
                    goals = topic.get("assessment_goals", [])
                    if goals:
                        summary_parts.append(f"{topic['topic_name']} goals: {', '.join(goals[:2])}")

        # Add critical skills
        if "critical_skills" in strategy_obj:
            skills = strategy_obj["critical_skills"][:5]
            summary_parts.append(f"Critical skills: {', '.join(skills)}")

        # Add risk areas
        if "risk_areas" in strategy_obj:
            risks = strategy_obj["risk_areas"][:3]
            summary_parts.append(f"Risk areas: {', '.join(risks)}")

        result = "\n".join(summary_parts)
        if len(result) <= max_length:
            return result

    except json.JSONDecodeError:
        pass  # Not JSON, continue with text-based extraction

    # Fallback: smart truncation
    return _smart_truncate(strategy, max_length)


def extract_relevant_context(
    resume_text: str,
    job_description_text: str,
    current_topic: str,
    interview_strategy: str = "",
    max_resume_chars: int = 1000,
    max_jd_chars: int = 1000
) -> Dict[str, str]:
    """
    Extract relevant excerpts from resume and JD based on current topic.

    This provides document context for question generation, enabling questions
    that cite specific achievements, projects, and requirements.

    CRITICAL IMPROVEMENT: Adds missing document context to question generation.
    Questions can now reference specific resume points and JD requirements.

    Args:
        resume_text: Full resume text
        job_description_text: Full job description text
        current_topic: Current interview topic (e.g., "Python Experience", "System Design")
        interview_strategy: JSON strategy with assessment goals (optional)
        max_resume_chars: Maximum chars to extract from resume (default: 1000)
        max_jd_chars: Maximum chars to extract from JD (default: 1000)

    Returns:
        Dict with 'resume_context', 'jd_context', 'assessment_goals' keys
    """
    import json
    import re

    # Extract topic keywords for matching
    topic_keywords = _extract_topic_keywords(current_topic)

    # Extract relevant resume sections
    resume_context = _extract_matching_sections(
        resume_text,
        topic_keywords,
        max_chars=max_resume_chars,
        section_name="resume"
    )

    # Extract relevant JD sections
    jd_context = _extract_matching_sections(
        job_description_text,
        topic_keywords,
        max_chars=max_jd_chars,
        section_name="job description"
    )

    # Extract assessment goals for this topic from strategy
    assessment_goals = []
    if interview_strategy:
        try:
            strategy_obj = json.loads(interview_strategy)
            topic_allocations = strategy_obj.get("topic_allocations", [])

            for allocation in topic_allocations:
                if allocation.get("topic_name", "").lower() == current_topic.lower():
                    assessment_goals = allocation.get("assessment_goals", [])
                    break
        except json.JSONDecodeError:
            pass

    return {
        "resume_context": resume_context or "No specific resume details found for this topic.",
        "jd_context": jd_context or "No specific job requirements found for this topic.",
        "assessment_goals": ", ".join(assessment_goals) if assessment_goals else "General topic exploration",
        "topic_keywords": ", ".join(topic_keywords[:5])  # For transparency
    }


def _extract_topic_keywords(topic: str) -> list[str]:
    """
    Extract keywords from topic name for matching.

    Examples:
        "Python Experience" -> ["python", "experience"]
        "System Design & Architecture" -> ["system", "design", "architecture"]
    """
    import re

    # Handle None or empty topic
    if not topic:
        return []

    # Remove common filler words
    filler_words = {"and", "or", "the", "a", "an", "of", "in", "on", "at", "to", "for", "with"}

    # Split on non-alphanumeric chars and filter
    words = re.split(r'[^a-zA-Z0-9]+', topic.lower())
    keywords = [w for w in words if w and w not in filler_words and len(w) > 2]

    return keywords


def _extract_matching_sections(
    text: str,
    keywords: list[str],
    max_chars: int,
    section_name: str
) -> str:
    """
    Extract sections of text that contain the given keywords.

    Args:
        text: Full text to search
        keywords: Keywords to match
        max_chars: Maximum characters to extract
        section_name: Name for logging (resume/jd)

    Returns:
        Extracted text with relevant sections
    """
    if not text or not keywords:
        return ""

    import re

    # Split text into paragraphs
    paragraphs = text.split('\n\n')
    if not paragraphs:
        paragraphs = text.split('\n')

    # Score each paragraph by keyword matches
    scored_paragraphs = []
    for para in paragraphs:
        para = para.strip()
        if not para or len(para) < 20:  # Skip very short paragraphs
            continue

        # Count keyword matches (case-insensitive)
        score = sum(1 for keyword in keywords if re.search(r'\b' + re.escape(keyword) + r'\b', para, re.IGNORECASE))

        if score > 0:
            scored_paragraphs.append((score, para))

    # Sort by score (highest first)
    scored_paragraphs.sort(reverse=True, key=lambda x: x[0])

    # Collect top paragraphs until max_chars reached
    selected_paragraphs = []
    total_chars = 0

    for score, para in scored_paragraphs:
        if total_chars + len(para) > max_chars:
            # Try to fit partial paragraph
            remaining_chars = max_chars - total_chars
            if remaining_chars > 200:  # Only if significant space left
                truncated = _smart_truncate(para, remaining_chars)
                selected_paragraphs.append(truncated)
            break

        selected_paragraphs.append(para)
        total_chars += len(para)

        if total_chars >= max_chars:
            break

    if not selected_paragraphs:
        # Fallback: Take first max_chars of document
        return _smart_truncate(text, min(max_chars, 500))

    return "\n\n".join(selected_paragraphs)


def _extract_role(msg: Any) -> str:
    """Extract role from message object (handles both dict and LangChain objects)."""
    if isinstance(msg, dict):
        return msg.get("role", "unknown")
    else:
        msg_type = type(msg).__name__
        if msg_type == "HumanMessage":
            return "user"
        elif msg_type == "AIMessage":
            return "assistant"
        elif msg_type == "SystemMessage":
            return "system"
        return "unknown"


def _extract_content(msg: Any) -> str:
    """Extract content from message object."""
    if isinstance(msg, dict):
        return msg.get("content", "")
    else:
        return msg.content if hasattr(msg, 'content') else ""


def _extract_topic(msg: Any) -> str:
    """Extract topic from message object."""
    if isinstance(msg, dict):
        return msg.get("topic", "")
    else:
        return getattr(msg, 'topic', '') if hasattr(msg, 'topic') else ""


__all__ = [
    'compress_text',
    'compress_custom_instructions',
    'summarize_conversation',
    'extract_topic_context',
    'summarize_strategy',
    'extract_relevant_context'
]
