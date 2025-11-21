"""
LLM Retry Utilities

Provides robust error handling and retry logic for LLM API calls.
Implements exponential backoff, rate limit handling, and timeout management.
"""

import logging
import asyncio
from typing import TypeVar, Callable, Any
from functools import wraps

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    after_log,
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


class LLMRateLimitError(Exception):
    """Raised when LLM API rate limit is exceeded (429)"""
    pass


class LLMTimeoutError(Exception):
    """Raised when LLM API call times out"""
    pass


class LLMAPIError(Exception):
    """General LLM API error"""
    pass


# Retry configuration for LLM calls
# 3 attempts with exponential backoff: 2s, 4s, 8s
llm_retry_config = dict(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((LLMRateLimitError, LLMTimeoutError, LLMAPIError, ConnectionError, TimeoutError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    after=after_log(logger, logging.INFO),
    reraise=True,
)


def retry_llm_call(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator for synchronous LLM calls with retry logic.

    Handles:
    - Rate limit errors (429) with exponential backoff
    - Connection errors
    - Timeout errors
    - General API errors

    Example:
        @retry_llm_call
        def call_llm(messages):
            return llm.invoke(messages)
    """
    @retry(**llm_retry_config)
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Classify the exception for appropriate handling
            error_msg = str(e).lower()

            if "429" in error_msg or "rate limit" in error_msg:
                logger.warning(f"Rate limit exceeded in {func.__name__}: {e}")
                raise LLMRateLimitError(f"Rate limit exceeded: {e}") from e
            elif "timeout" in error_msg:
                logger.warning(f"Timeout in {func.__name__}: {e}")
                raise LLMTimeoutError(f"Request timed out: {e}") from e
            elif any(keyword in error_msg for keyword in ["connection", "network", "unavailable"]):
                logger.warning(f"Connection error in {func.__name__}: {e}")
                raise ConnectionError(f"Connection failed: {e}") from e
            else:
                logger.error(f"LLM API error in {func.__name__}: {e}")
                raise LLMAPIError(f"API error: {e}") from e

    return wrapper


def async_retry_llm_call(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator for async LLM calls with retry logic.

    Handles:
    - Rate limit errors (429) with exponential backoff
    - Connection errors
    - Timeout errors
    - General API errors

    Example:
        @async_retry_llm_call
        async def call_llm_async(messages):
            return await llm.ainvoke(messages)
    """
    @retry(**llm_retry_config)
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            # Classify the exception for appropriate handling
            error_msg = str(e).lower()

            if "429" in error_msg or "rate limit" in error_msg:
                logger.warning(f"Rate limit exceeded in {func.__name__}: {e}")
                raise LLMRateLimitError(f"Rate limit exceeded: {e}") from e
            elif "timeout" in error_msg:
                logger.warning(f"Timeout in {func.__name__}: {e}")
                raise LLMTimeoutError(f"Request timed out: {e}") from e
            elif any(keyword in error_msg for keyword in ["connection", "network", "unavailable"]):
                logger.warning(f"Connection error in {func.__name__}: {e}")
                raise ConnectionError(f"Connection failed: {e}") from e
            else:
                logger.error(f"LLM API error in {func.__name__}: {e}")
                raise LLMAPIError(f"API error: {e}") from e

    return wrapper


async def call_llm_with_timeout(
    llm_call: Callable,
    timeout_seconds: int = 60,
    *args,
    **kwargs
) -> Any:
    """
    Execute an LLM call with a timeout.

    Args:
        llm_call: The LLM function to call
        timeout_seconds: Maximum time to wait (default 60s)
        *args, **kwargs: Arguments to pass to llm_call

    Returns:
        The result of the LLM call

    Raises:
        LLMTimeoutError: If the call exceeds the timeout

    Example:
        result = await call_llm_with_timeout(
            llm.ainvoke,
            timeout_seconds=30,
            messages=[HumanMessage(content="Hello")]
        )
    """
    try:
        return await asyncio.wait_for(
            llm_call(*args, **kwargs),
            timeout=timeout_seconds
        )
    except asyncio.TimeoutError as e:
        logger.error(f"LLM call timed out after {timeout_seconds}s")
        raise LLMTimeoutError(f"LLM call exceeded {timeout_seconds}s timeout") from e


def with_fallback(
    primary_func: Callable[..., T],
    fallback_func: Callable[..., T],
    fallback_exceptions: tuple = (Exception,)
) -> Callable[..., T]:
    """
    Decorator that provides a fallback function if the primary function fails.

    Args:
        primary_func: The main function to try
        fallback_func: The fallback function to use on failure
        fallback_exceptions: Tuple of exceptions that trigger fallback

    Example:
        def get_complex_answer(question):
            # Use advanced LLM with extended thinking
            pass

        def get_simple_answer(question):
            # Use basic LLM without thinking
            pass

        get_answer = with_fallback(
            get_complex_answer,
            get_simple_answer,
            fallback_exceptions=(LLMTimeoutError, LLMRateLimitError)
        )
    """
    @wraps(primary_func)
    def wrapper(*args, **kwargs):
        try:
            return primary_func(*args, **kwargs)
        except fallback_exceptions as e:
            logger.warning(
                f"Primary function {primary_func.__name__} failed with {type(e).__name__}: {e}. "
                f"Using fallback {fallback_func.__name__}"
            )
            return fallback_func(*args, **kwargs)

    return wrapper


async def with_fallback_async(
    primary_func: Callable[..., T],
    fallback_func: Callable[..., T],
    fallback_exceptions: tuple = (Exception,),
    *args,
    **kwargs
) -> T:
    """
    Async version of with_fallback.

    Args:
        primary_func: The main async function to try
        fallback_func: The fallback async function to use on failure
        fallback_exceptions: Tuple of exceptions that trigger fallback

    Example:
        result = await with_fallback_async(
            get_complex_answer_async,
            get_simple_answer_async,
            fallback_exceptions=(LLMTimeoutError,),
            question="What is AI?"
        )
    """
    try:
        return await primary_func(*args, **kwargs)
    except fallback_exceptions as e:
        logger.warning(
            f"Primary function {primary_func.__name__} failed with {type(e).__name__}: {e}. "
            f"Using fallback {fallback_func.__name__}"
        )
        return await fallback_func(*args, **kwargs)
