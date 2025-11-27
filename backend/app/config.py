"""
Application Configuration Module
Handles environment variable loading and application settings.
"""

from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    gemini_api_key: str
    gemini_model: str = "gemini-2.5-flash"

    # Chatterbox TTS Configuration (Real-Time Streaming with Voice Cloning)
    chatterbox_reference_audio: str  # Path to reference audio for voice cloning (WAV recommended)
    chatterbox_device: str = "cuda"  # Device: "cuda" or "cpu"
    chatterbox_chunk_size: int = 25  # Tokens per streaming chunk (15-30, lower=faster TTFA)
    chatterbox_context_window: int = 50  # Tokens of context for continuity
    chatterbox_fade_duration: float = 0.02  # Audio fade transition in seconds (prevents clicks)
    chatterbox_cfm_steps: int = 7  # CFM inference steps (7 for real-time, 10+ for quality)
    chatterbox_use_fp16: bool = True  # Use half-precision for 2x speed and 50% memory reduction
    chatterbox_optimize_gpu: bool = True  # Enable CUDA graph compilation for 30-50% speedup
    chatterbox_preload: bool = True  # Preload model at startup for instant TTS

    # LLM Optimization Parameters
    gemini_thinking_budget: int = 0
    gemini_include_thoughts: bool = False
    gemini_max_output_tokens: int = 1024
    gemini_temperature: float = 0.7

    # Embedded Whisper Transcription Settings
    whisper_model: str = "distil-large-v3"  # Model: tiny, small, medium, large-v3, distil-large-v3
    whisper_device: str = "cuda"  # Device: cuda or cpu
    whisper_compute_type: str = "float16"  # Precision: float16, float32, int8
    whisper_language: str = "en"  # Language code for transcription
    whisper_use_vad: bool = True  # Enable Voice Activity Detection
    whisper_no_speech_thresh: float = 0.45  # Silence detection threshold (0.0-1.0)
    whisper_chunk_interval: float = 1.0  # Minimum seconds of audio before transcription
    whisper_same_output_threshold: int = 5  # Repeated outputs before finalizing segment
    transcription_logging_enabled: bool = True  # Enable/disable verbose transcription logs
    whisper_preload: bool = True  # Preload model at startup for instant transcription

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
