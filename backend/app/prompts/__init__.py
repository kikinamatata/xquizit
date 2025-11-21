"""
Prompts Module
Centralized LLM prompts with LangChain templates and structured output.

Provides:
- Pydantic schemas for type-safe LLM outputs
- Token optimization utilities for performance
- ChatPromptTemplates for all LLM call locations
"""

# Optimization utilities
from .optimization import (
    compress_text,
    compress_custom_instructions,
    summarize_conversation,
    extract_topic_context,
    summarize_strategy
)

# Prompt templates
from .question_generation import (
    create_intro_question_prompt,
    create_topic_question_prompt,
    create_followup_question_prompt
)


__all__ = [
    # Optimization
    'compress_text',
    'compress_custom_instructions',
    'summarize_conversation',
    'extract_topic_context',
    'summarize_strategy',

    # Prompts
    'create_intro_question_prompt',
    'create_topic_question_prompt',
    'create_followup_question_prompt',
]


# Module version
__version__ = '2.0.0'
