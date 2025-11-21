"""
Validation Utilities with Retry Logic

Provides validation retry loops for LLM structured outputs.
Automatically retries LLM calls when validation fails, passing error messages back for correction.
"""

import logging
from typing import TypeVar, Type, Callable, Any, Optional
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


async def call_llm_with_validation_retry(
    llm_call: Callable,
    schema: Type[T],
    max_retries: int = 3,
    *args,
    **kwargs
) -> T:
    """
    Call LLM with automatic validation retry on structured output failure.

    Args:
        llm_call: The LLM function to call (should return structured output)
        schema: The Pydantic schema to validate against
        max_retries: Maximum number of retry attempts (default 3)
        *args, **kwargs: Arguments to pass to llm_call

    Returns:
        Validated Pydantic model instance

    Raises:
        ValidationError: If validation fails after all retries

    Example:
        result = await call_llm_with_validation_retry(
            llm.with_structured_output(InterviewTimeStrategy).ainvoke,
            InterviewTimeStrategy,
            max_retries=3,
            messages=[HumanMessage(content="Create strategy")]
        )
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            # Call the LLM
            if attempt == 0:
                result = await llm_call(*args, **kwargs)
            else:
                # On retry, add validation error feedback to messages
                # Assuming kwargs['messages'] exists
                if 'messages' in kwargs:
                    error_message = {
                        "role": "system",
                        "content": f"Validation failed: {last_error}\n\n"
                                 f"Please correct the output to match the required schema and try again."
                    }
                    kwargs['messages'].append(error_message)
                    result = await llm_call(*args, **kwargs)
                else:
                    # If no messages, just retry without feedback
                    result = await llm_call(*args, **kwargs)

            # Validate the result
            if isinstance(result, schema):
                # Already validated by with_structured_output
                logger.info(f"Validation successful on attempt {attempt + 1}")
                return result
            else:
                # Manual validation needed
                validated = schema.model_validate(result)
                logger.info(f"Validation successful on attempt {attempt + 1}")
                return validated

        except ValidationError as e:
            last_error = e
            logger.warning(
                f"Validation failed on attempt {attempt + 1}/{max_retries}: {e}",
                extra={
                    "attempt": attempt + 1,
                    "max_retries": max_retries,
                    "schema": schema.__name__,
                    "error": str(e)
                }
            )

            if attempt == max_retries - 1:
                # Last attempt failed
                logger.error(
                    f"Validation failed after {max_retries} attempts",
                    extra={
                        "schema": schema.__name__,
                        "final_error": str(e)
                    }
                )
                raise

        except Exception as e:
            # Non-validation error (e.g., LLM API error)
            logger.error(f"Unexpected error during validation retry: {e}")
            raise

    # Should not reach here
    raise ValidationError(f"Validation failed after {max_retries} attempts")


def call_llm_with_validation_retry_sync(
    llm_call: Callable,
    schema: Type[T],
    max_retries: int = 3,
    *args,
    **kwargs
) -> T:
    """
    Synchronous version of call_llm_with_validation_retry.

    Args:
        llm_call: The LLM function to call (should return structured output)
        schema: The Pydantic schema to validate against
        max_retries: Maximum number of retry attempts (default 3)
        *args, **kwargs: Arguments to pass to llm_call

    Returns:
        Validated Pydantic model instance

    Raises:
        ValidationError: If validation fails after all retries

    Example:
        result = call_llm_with_validation_retry_sync(
            llm.with_structured_output(InterviewTimeStrategy).invoke,
            InterviewTimeStrategy,
            max_retries=3,
            messages=[HumanMessage(content="Create strategy")]
        )
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            # Call the LLM
            if attempt == 0:
                result = llm_call(*args, **kwargs)
            else:
                # On retry, add validation error feedback to messages
                if 'messages' in kwargs:
                    error_message = {
                        "role": "system",
                        "content": f"Validation failed: {last_error}\n\n"
                                 f"Please correct the output to match the required schema and try again."
                    }
                    kwargs['messages'].append(error_message)
                    result = llm_call(*args, **kwargs)
                else:
                    result = llm_call(*args, **kwargs)

            # Validate the result
            if isinstance(result, schema):
                logger.info(f"Validation successful on attempt {attempt + 1}")
                return result
            else:
                validated = schema.model_validate(result)
                logger.info(f"Validation successful on attempt {attempt + 1}")
                return validated

        except ValidationError as e:
            last_error = e
            logger.warning(
                f"Validation failed on attempt {attempt + 1}/{max_retries}: {e}",
                extra={
                    "attempt": attempt + 1,
                    "max_retries": max_retries,
                    "schema": schema.__name__,
                    "error": str(e)
                }
            )

            if attempt == max_retries - 1:
                logger.error(
                    f"Validation failed after {max_retries} attempts",
                    extra={
                        "schema": schema.__name__,
                        "final_error": str(e)
                    }
                )
                raise

        except Exception as e:
            logger.error(f"Unexpected error during validation retry: {e}")
            raise

    raise ValidationError(f"Validation failed after {max_retries} attempts")


def validate_with_fallback(
    value: Any,
    schema: Type[T],
    fallback: Optional[T] = None
) -> T:
    """
    Validate a value against a schema with an optional fallback.

    Args:
        value: The value to validate
        schema: The Pydantic schema to validate against
        fallback: Optional fallback value if validation fails

    Returns:
        Validated value or fallback

    Raises:
        ValidationError: If validation fails and no fallback provided

    Example:
        strategy = validate_with_fallback(
            json_data,
            InterviewTimeStrategy,
            fallback=default_strategy
        )
    """
    try:
        if isinstance(value, schema):
            return value
        return schema.model_validate(value)
    except ValidationError as e:
        if fallback is not None:
            logger.warning(
                f"Validation failed, using fallback: {e}",
                extra={"schema": schema.__name__, "error": str(e)}
            )
            return fallback
        else:
            raise


def validate_field_range(
    value: float,
    field_name: str,
    min_value: float = 0.0,
    max_value: float = 1.0
) -> float:
    """
    Validate that a numeric field is within a specified range.

    Args:
        value: The value to validate
        field_name: Name of the field (for error messages)
        min_value: Minimum allowed value (default 0.0)
        max_value: Maximum allowed value (default 1.0)

    Returns:
        The value if valid, clamped value if out of range

    Example:
        confidence = validate_field_range(llm_confidence, "confidence", 0.0, 1.0)
    """
    if value < min_value:
        logger.warning(
            f"{field_name} value {value} below minimum {min_value}, clamping to {min_value}"
        )
        return min_value
    elif value > max_value:
        logger.warning(
            f"{field_name} value {value} above maximum {max_value}, clamping to {max_value}"
        )
        return max_value
    return value


def semantic_validation(
    response: str,
    expected_keywords: list[str],
    min_keywords: int = 1
) -> tuple[bool, list[str]]:
    """
    Perform semantic validation by checking for expected keywords in the response.

    Args:
        response: The text response to validate
        expected_keywords: List of keywords to check for
        min_keywords: Minimum number of keywords required (default 1)

    Returns:
        Tuple of (is_valid, found_keywords)

    Example:
        is_valid, found = semantic_validation(
            question,
            expected_keywords=["Python", "experience", "project"],
            min_keywords=2
        )
    """
    response_lower = response.lower()
    found_keywords = [
        keyword for keyword in expected_keywords
        if keyword.lower() in response_lower
    ]

    is_valid = len(found_keywords) >= min_keywords

    if not is_valid:
        logger.warning(
            f"Semantic validation failed: found {len(found_keywords)}/{min_keywords} required keywords",
            extra={
                "expected_keywords": expected_keywords,
                "found_keywords": found_keywords,
                "response_preview": response[:200]
            }
        )

    return is_valid, found_keywords


def validate_conversation_turn(
    turn_data: dict,
    required_fields: list[str]
) -> bool:
    """
    Validate that a conversation turn contains all required fields.

    Args:
        turn_data: The conversation turn dictionary
        required_fields: List of required field names

    Returns:
        True if all required fields present, False otherwise

    Example:
        is_valid = validate_conversation_turn(
            message_dict,
            required_fields=["role", "content", "timestamp"]
        )
    """
    missing_fields = [
        field for field in required_fields
        if field not in turn_data or turn_data[field] is None
    ]

    if missing_fields:
        logger.warning(
            f"Conversation turn validation failed: missing fields {missing_fields}",
            extra={
                "missing_fields": missing_fields,
                "turn_data": turn_data
            }
        )
        return False

    return True
