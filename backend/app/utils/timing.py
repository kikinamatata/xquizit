"""
Timing utilities for measuring performance of interview chatbot operations.
"""
import time
import logging
from contextlib import contextmanager
from typing import Dict, Optional, Any
from functools import wraps
from datetime import datetime
import json

# Configure timing logger
timing_logger = logging.getLogger("timing")
timing_logger.setLevel(logging.INFO)

# Console handler with formatted output
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - TIMING - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console_handler.setFormatter(formatter)
timing_logger.addHandler(console_handler)

# Prevent propagation to root logger
timing_logger.propagate = False


class TimingData:
    """Stores timing measurements for an operation."""

    def __init__(self, operation_name: str, start_time: float):
        self.operation_name = operation_name
        self.start_time = start_time
        self.end_time: Optional[float] = None
        self.duration: Optional[float] = None
        self.metadata: Dict[str, Any] = {}

    def complete(self, metadata: Optional[Dict[str, Any]] = None):
        """Mark operation as complete and calculate duration."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        if metadata:
            self.metadata.update(metadata)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "operation": self.operation_name,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None,
            "duration_seconds": round(self.duration, 3) if self.duration else None,
            "metadata": self.metadata
        }

    def log(self):
        """Log timing data."""
        if self.duration is not None:
            metadata_str = f" | {json.dumps(self.metadata)}" if self.metadata else ""
            timing_logger.info(
                f"{self.operation_name}: {self.duration:.3f}s{metadata_str}"
            )


@contextmanager
def time_operation(operation_name: str, log_result: bool = True, metadata: Optional[Dict[str, Any]] = None):
    """
    Context manager for timing operations.

    Usage:
        with time_operation("LLM Call") as timing:
            result = llm.invoke(prompt)
            timing.metadata["tokens"] = len(result)
    """
    timing = TimingData(operation_name, time.time())
    try:
        yield timing
    finally:
        timing.complete(metadata)
        if log_result:
            timing.log()


def timed(operation_name: Optional[str] = None, log_result: bool = True):
    """
    Decorator for timing function execution.

    Usage:
        @timed("Generate Question")
        def generate_question():
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            name = operation_name or f"{func.__module__}.{func.__name__}"
            with time_operation(name, log_result) as timing:
                result = func(*args, **kwargs)
                return result
        return wrapper
    return decorator


class TimingSummary:
    """Accumulates timing data for a session or request."""

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id
        self.timings: list[TimingData] = []
        self.start_time = time.time()

    def add_timing(self, timing: TimingData):
        """Add a timing measurement to the summary."""
        self.timings.append(timing)

    def get_total_duration(self) -> float:
        """Get total elapsed time since summary creation."""
        return time.time() - self.start_time

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all timing measurements."""
        return {
            "session_id": self.session_id,
            "total_duration_seconds": round(self.get_total_duration(), 3),
            "operations": [t.to_dict() for t in self.timings],
            "operation_breakdown": {
                t.operation_name: round(t.duration, 3)
                for t in self.timings if t.duration is not None
            }
        }

    def log_summary(self):
        """Log summary of all timing measurements."""
        summary = self.get_summary()
        timing_logger.info(
            f"\n{'='*60}\n"
            f"TIMING SUMMARY - Session: {self.session_id}\n"
            f"Total Duration: {summary['total_duration_seconds']}s\n"
            f"Operations:\n" +
            "\n".join(f"  - {k}: {v}s" for k, v in summary['operation_breakdown'].items()) +
            f"\n{'='*60}"
        )


@contextmanager
def time_operation_in_summary(summary: TimingSummary, operation_name: str, metadata: Optional[Dict[str, Any]] = None):
    """
    Context manager for timing operations that adds to a TimingSummary.

    Usage:
        summary = TimingSummary(session_id)
        with time_operation_in_summary(summary, "LLM Call") as timing:
            result = llm.invoke(prompt)
    """
    timing = TimingData(operation_name, time.time())
    try:
        yield timing
    finally:
        timing.complete(metadata)
        timing.log()
        summary.add_timing(timing)
