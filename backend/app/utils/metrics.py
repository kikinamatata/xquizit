"""
Metrics Collection and Monitoring

Provides Prometheus-style metrics for monitoring application performance and behavior.
"""

import time
from typing import Callable, Any
from functools import wraps
from contextlib import contextmanager

from prometheus_client import Counter, Histogram, Gauge, Summary


# LLM Metrics
llm_requests_total = Counter(
    "llm_requests_total",
    "Total number of LLM API requests",
    ["agent_name", "status"]  # status: success, error, rate_limited, timeout
)

llm_latency_seconds = Histogram(
    "llm_latency_seconds",
    "LLM API call latency in seconds",
    ["agent_name"],
    buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)
)

llm_tokens_total = Counter(
    "llm_tokens_total",
    "Total tokens consumed",
    ["agent_name", "token_type"]  # token_type: prompt, completion, total
)

llm_retry_attempts_total = Counter(
    "llm_retry_attempts_total",
    "Total number of LLM retry attempts",
    ["agent_name", "reason"]  # reason: rate_limit, timeout, error
)


# Agent Execution Metrics
agent_executions_total = Counter(
    "agent_executions_total",
    "Total number of agent executions",
    ["agent_name", "status"]  # status: success, error
)

agent_execution_duration_seconds = Histogram(
    "agent_execution_duration_seconds",
    "Agent execution duration in seconds",
    ["agent_name"],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0)
)


# Validation Metrics
validation_failures_total = Counter(
    "validation_failures_total",
    "Total number of validation failures",
    ["schema_name", "resolved"]  # resolved: yes (retry succeeded), no (failed after retries)
)

validation_retry_attempts_total = Counter(
    "validation_retry_attempts_total",
    "Total number of validation retry attempts",
    ["schema_name"]
)


# Session Metrics
active_sessions = Gauge(
    "active_sessions",
    "Number of currently active interview sessions"
)

session_duration_seconds = Histogram(
    "session_duration_seconds",
    "Interview session duration in seconds",
    buckets=(60, 300, 600, 900, 1200, 1800, 2400, 3000)  # 1m to 50m
)

questions_asked_per_session = Histogram(
    "questions_asked_per_session",
    "Number of questions asked per interview session",
    buckets=(5, 10, 15, 20, 25, 30, 40, 50)
)

topics_covered_per_session = Histogram(
    "topics_covered_per_session",
    "Number of topics covered per interview session",
    buckets=(1, 2, 3, 4, 5, 6, 8, 10)
)


# Interview Quality Metrics
topic_coverage_score = Summary(
    "topic_coverage_score",
    "Topic coverage scores (0.0 - 1.0)",
    ["topic_name"]
)

topic_confidence_score = Summary(
    "topic_confidence_score",
    "Topic confidence scores (0.0 - 1.0)",
    ["topic_name"]
)


# API Endpoint Metrics
http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"]
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0)
)


# WebSocket Metrics
websocket_connections_total = Counter(
    "websocket_connections_total",
    "Total WebSocket connections",
    ["endpoint", "status"]  # status: connected, disconnected, error
)

websocket_messages_total = Counter(
    "websocket_messages_total",
    "Total WebSocket messages",
    ["endpoint", "direction"]  # direction: sent, received
)


# TTS Metrics
tts_generations_total = Counter(
    "tts_generations_total",
    "Total TTS audio generations",
    ["status"]  # status: success, error
)

tts_generation_duration_seconds = Histogram(
    "tts_generation_duration_seconds",
    "TTS generation duration in seconds",
    buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
)


# Transcription Metrics
transcription_requests_total = Counter(
    "transcription_requests_total",
    "Total transcription requests",
    ["status"]  # status: success, error
)

transcription_duration_seconds = Histogram(
    "transcription_duration_seconds",
    "Transcription duration in seconds",
    buckets=(0.5, 1.0, 2.0, 3.0, 5.0, 10.0)
)


# Utility Functions

@contextmanager
def track_llm_call(agent_name: str):
    """
    Context manager to track LLM API call metrics.

    Args:
        agent_name: Name of the agent making the call

    Example:
        with track_llm_call("strategic_planner"):
            result = await llm.ainvoke(messages)
            # Metrics automatically recorded
    """
    start_time = time.time()
    status = "success"

    try:
        yield
    except Exception as e:
        error_msg = str(e).lower()
        if "429" in error_msg or "rate limit" in error_msg:
            status = "rate_limited"
            llm_retry_attempts_total.labels(agent_name=agent_name, reason="rate_limit").inc()
        elif "timeout" in error_msg:
            status = "timeout"
            llm_retry_attempts_total.labels(agent_name=agent_name, reason="timeout").inc()
        else:
            status = "error"
            llm_retry_attempts_total.labels(agent_name=agent_name, reason="error").inc()
        raise
    finally:
        duration = time.time() - start_time
        llm_requests_total.labels(agent_name=agent_name, status=status).inc()
        llm_latency_seconds.labels(agent_name=agent_name).observe(duration)


@contextmanager
def track_agent_execution(agent_name: str):
    """
    Context manager to track agent execution metrics.

    Args:
        agent_name: Name of the agent

    Example:
        with track_agent_execution("quality_assessor"):
            result = agent(state)
            # Metrics automatically recorded
    """
    start_time = time.time()
    status = "success"

    try:
        yield
    except Exception:
        status = "error"
        raise
    finally:
        duration = time.time() - start_time
        agent_executions_total.labels(agent_name=agent_name, status=status).inc()
        agent_execution_duration_seconds.labels(agent_name=agent_name).observe(duration)


@contextmanager
def track_http_request(method: str, endpoint: str):
    """
    Context manager to track HTTP request metrics.

    Args:
        method: HTTP method (GET, POST, etc.)
        endpoint: Endpoint path

    Example:
        with track_http_request("POST", "/upload-documents"):
            # Process request
            pass
    """
    start_time = time.time()
    status_code = 200

    try:
        yield lambda code: nonlocal_setter("status_code", code)
    except Exception:
        status_code = 500
        raise
    finally:
        duration = time.time() - start_time
        http_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status_code=str(status_code)
        ).inc()
        http_request_duration_seconds.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)


def nonlocal_setter(var_name: str, value: Any):
    """Helper for modifying nonlocal variables in closures"""
    pass  # Placeholder for closure variable modification


def record_llm_tokens(agent_name: str, prompt_tokens: int = 0, completion_tokens: int = 0):
    """
    Record LLM token usage.

    Args:
        agent_name: Name of the agent
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens

    Example:
        record_llm_tokens("strategic_planner", prompt_tokens=1500, completion_tokens=300)
    """
    if prompt_tokens > 0:
        llm_tokens_total.labels(agent_name=agent_name, token_type="prompt").inc(prompt_tokens)

    if completion_tokens > 0:
        llm_tokens_total.labels(agent_name=agent_name, token_type="completion").inc(completion_tokens)

    total_tokens = prompt_tokens + completion_tokens
    if total_tokens > 0:
        llm_tokens_total.labels(agent_name=agent_name, token_type="total").inc(total_tokens)


def record_validation_failure(schema_name: str, resolved: bool = False):
    """
    Record a validation failure.

    Args:
        schema_name: Name of the Pydantic schema
        resolved: Whether the failure was resolved via retry

    Example:
        record_validation_failure("InterviewTimeStrategy", resolved=True)
    """
    resolved_label = "yes" if resolved else "no"
    validation_failures_total.labels(schema_name=schema_name, resolved=resolved_label).inc()


def record_validation_retry(schema_name: str):
    """
    Record a validation retry attempt.

    Args:
        schema_name: Name of the Pydantic schema

    Example:
        record_validation_retry("TopicAssessmentQuality")
    """
    validation_retry_attempts_total.labels(schema_name=schema_name).inc()


def record_topic_quality(topic_name: str, coverage: float, confidence: float):
    """
    Record topic quality metrics.

    Args:
        topic_name: Name of the topic
        coverage: Coverage score (0.0 - 1.0)
        confidence: Confidence score (0.0 - 1.0)

    Example:
        record_topic_quality("Python programming", coverage=0.85, confidence=0.92)
    """
    topic_coverage_score.labels(topic_name=topic_name).observe(coverage)
    topic_confidence_score.labels(topic_name=topic_name).observe(confidence)


def record_session_metrics(
    duration_seconds: float,
    questions_asked: int,
    topics_covered: int
):
    """
    Record session-level metrics.

    Args:
        duration_seconds: Session duration in seconds
        questions_asked: Total questions asked
        topics_covered: Total topics covered

    Example:
        record_session_metrics(
            duration_seconds=1850.5,
            questions_asked=18,
            topics_covered=5
        )
    """
    session_duration_seconds.observe(duration_seconds)
    questions_asked_per_session.observe(questions_asked)
    topics_covered_per_session.observe(topics_covered)


# Decorators

def track_llm_metrics(agent_name: str):
    """
    Decorator to automatically track LLM call metrics.

    Args:
        agent_name: Name of the agent

    Example:
        @track_llm_metrics("strategic_planner")
        async def call_llm(messages):
            return await llm.ainvoke(messages)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with track_llm_call(agent_name):
                return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with track_llm_call(agent_name):
                return func(*args, **kwargs)

        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def track_agent_metrics(agent_name: str):
    """
    Decorator to automatically track agent execution metrics.

    Args:
        agent_name: Name of the agent

    Example:
        @track_agent_metrics("quality_assessor")
        def assess_quality(state):
            # Agent logic
            return result
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            with track_agent_execution(agent_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator
