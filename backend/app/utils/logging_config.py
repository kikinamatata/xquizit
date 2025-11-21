"""
Structured Logging Configuration

Provides JSON-formatted logging for production environments with rich contextual metadata.
"""

import logging
import json
import sys
from datetime import datetime
from typing import Any, Dict
from logging import LogRecord


class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.

    Outputs logs in JSON format with timestamp, level, logger name, message, and extra fields.
    """

    def format(self, record: LogRecord) -> str:
        """
        Format a log record as JSON.

        Args:
            record: The log record to format

        Returns:
            JSON-formatted log string
        """
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields from record
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        # Add any custom fields passed via extra parameter
        for key, value in record.__dict__.items():
            if key not in [
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "lineno", "module", "msecs", "message", "pathname",
                "process", "processName", "relativeCreated", "thread", "threadName",
                "exc_info", "exc_text", "stack_info", "extra_fields"
            ]:
                # Ensure value is JSON serializable
                try:
                    json.dumps({key: value})
                    log_data[key] = value
                except (TypeError, ValueError):
                    log_data[key] = str(value)

        return json.dumps(log_data)


class ContextFilter(logging.Filter):
    """
    Filter that adds contextual information to log records.
    """

    def __init__(self, **context):
        super().__init__()
        self.context = context

    def filter(self, record: LogRecord) -> bool:
        """
        Add context fields to the log record.

        Args:
            record: The log record to filter

        Returns:
            True to allow the record through
        """
        if not hasattr(record, "extra_fields"):
            record.extra_fields = {}

        record.extra_fields.update(self.context)
        return True


def setup_logging(
    level: str = "INFO",
    json_format: bool = True,
    log_file: str = None,
    transcription_logging_enabled: bool = True
) -> None:
    """
    Configure structured logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: Use JSON formatting (default True)
        log_file: Optional file path to write logs to
        transcription_logging_enabled: Enable verbose transcription logs (default True)

    Example:
        setup_logging(level="DEBUG", json_format=True, log_file="app.log", transcription_logging_enabled=False)
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))

    if json_format:
        console_handler.setFormatter(JSONFormatter())
    else:
        # Use standard format for development
        console_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )

    root_logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(file_handler)

    # Set specific loggers to appropriate levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Configure transcription logging based on flag
    if not transcription_logging_enabled:
        # Suppress verbose RunPod backend logs (intervals, API calls, metrics, segments)
        logging.getLogger("integrations.whisper_live.backend.runpod_backend").setLevel(logging.WARNING)
        logging.getLogger("integrations.whisper_live.backend.base").setLevel(logging.WARNING)
        logging.getLogger("app.services.transcription_service").setLevel(logging.INFO)  # Keep service-level logs
    else:
        # Keep all transcription logs at INFO level for debugging
        logging.getLogger("integrations.whisper_live.backend.runpod_backend").setLevel(logging.INFO)
        logging.getLogger("integrations.whisper_live.backend.base").setLevel(logging.INFO)
        logging.getLogger("app.services.transcription_service").setLevel(logging.INFO)


def get_logger(name: str, **context) -> logging.Logger:
    """
    Get a logger with optional context.

    Args:
        name: Logger name (typically __name__)
        **context: Additional context fields to add to all log records

    Returns:
        Logger instance with context filter

    Example:
        logger = get_logger(__name__, session_id="abc123", agent="strategic_planner")
        logger.info("Planning interview", extra={"topics": 5})
    """
    logger = logging.getLogger(name)

    if context:
        logger.addFilter(ContextFilter(**context))

    return logger


def log_llm_call(
    logger: logging.Logger,
    agent_name: str,
    prompt_tokens: int = None,
    completion_tokens: int = None,
    total_tokens: int = None,
    latency_ms: float = None,
    model: str = None,
    **extra
) -> None:
    """
    Log an LLM API call with standardized fields.

    Args:
        logger: The logger instance
        agent_name: Name of the agent making the call
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens
        total_tokens: Total tokens used
        latency_ms: Latency in milliseconds
        model: Model name
        **extra: Additional fields

    Example:
        log_llm_call(
            logger,
            agent_name="strategic_planner",
            prompt_tokens=1500,
            completion_tokens=300,
            latency_ms=1250.5,
            model="gemini-1.5-flash"
        )
    """
    log_data = {
        "event": "llm_call",
        "agent_name": agent_name,
    }

    if prompt_tokens is not None:
        log_data["prompt_tokens"] = prompt_tokens
    if completion_tokens is not None:
        log_data["completion_tokens"] = completion_tokens
    if total_tokens is not None:
        log_data["total_tokens"] = total_tokens
    if latency_ms is not None:
        log_data["latency_ms"] = latency_ms
    if model is not None:
        log_data["model"] = model

    log_data.update(extra)

    logger.info("LLM call completed", extra=log_data)


def log_agent_execution(
    logger: logging.Logger,
    agent_name: str,
    execution_time_ms: float,
    success: bool = True,
    error: str = None,
    **extra
) -> None:
    """
    Log agent execution with standardized fields.

    Args:
        logger: The logger instance
        agent_name: Name of the agent
        execution_time_ms: Execution time in milliseconds
        success: Whether execution succeeded
        error: Error message if failed
        **extra: Additional fields

    Example:
        log_agent_execution(
            logger,
            agent_name="quality_assessor",
            execution_time_ms=523.2,
            success=True,
            coverage=0.85,
            confidence=0.92
        )
    """
    log_data = {
        "event": "agent_execution",
        "agent_name": agent_name,
        "execution_time_ms": execution_time_ms,
        "success": success,
    }

    if error:
        log_data["error"] = error

    log_data.update(extra)

    if success:
        logger.info("Agent execution completed", extra=log_data)
    else:
        logger.error("Agent execution failed", extra=log_data)


def log_validation_failure(
    logger: logging.Logger,
    schema_name: str,
    error: str,
    attempt: int = 1,
    max_retries: int = 3,
    **extra
) -> None:
    """
    Log validation failure with standardized fields.

    Args:
        logger: The logger instance
        schema_name: Name of the Pydantic schema
        error: Validation error message
        attempt: Current attempt number
        max_retries: Maximum retry attempts
        **extra: Additional fields

    Example:
        log_validation_failure(
            logger,
            schema_name="InterviewTimeStrategy",
            error="Field required: topic_allocations",
            attempt=1,
            max_retries=3
        )
    """
    log_data = {
        "event": "validation_failure",
        "schema_name": schema_name,
        "validation_error": error,
        "attempt": attempt,
        "max_retries": max_retries,
    }

    log_data.update(extra)

    logger.warning("Validation failed", extra=log_data)


def log_session_event(
    logger: logging.Logger,
    session_id: str,
    event_type: str,
    **extra
) -> None:
    """
    Log session-level events.

    Args:
        logger: The logger instance
        session_id: Session identifier
        event_type: Type of event (started, question_asked, answer_received, concluded)
        **extra: Additional fields

    Example:
        log_session_event(
            logger,
            session_id="abc123",
            event_type="question_asked",
            topic="Python programming",
            question_number=5
        )
    """
    log_data = {
        "event": "session_event",
        "session_id": session_id,
        "event_type": event_type,
    }

    log_data.update(extra)

    logger.info(f"Session event: {event_type}", extra=log_data)
