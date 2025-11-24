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

    # Indic Parler-TTS Configuration (ai4bharat/indic-parler-tts)
    parler_model_id: str = "ai4bharat/indic-parler-tts"  # Model: "ai4bharat/indic-parler-tts" (gated, Indian English) or "parler-tts/parler-tts-mini-v1" (public, standard English)
    parler_device: str = "auto"  # "cuda", "cpu", or "auto"
    parler_voice_description: str = "Thoma speaks with a clear, moderate pace in a close recording with minimal background noise and a slightly expressive tone"  # Natural language voice description
    parler_play_steps_in_s: float = 0.3  # Target seconds for first audio chunk (0.3s = 300ms TTFA)

    # Parler-TTS Generation Parameters
    parler_temperature: float = 1.0  # Sampling temperature (1.0 recommended)
    parler_top_p: float = 1.0  # Nucleus sampling threshold
    parler_repetition_penalty: float = 1.0  # Token repetition penalty
    parler_min_new_tokens: int = 10  # Minimum tokens to generate
    parler_max_new_tokens: int = 2000  # Maximum tokens (~22s audio)

    # Parler-TTS Performance Optimization
    parler_enable_compile: bool = True  # Enable torch.compile() for 4x speedup (requires warmup)

    # [DEPRECATED] Chatterbox TTS Configuration (kept for rollback)
    # chatterbox_reference_voice: str = ""
    # chatterbox_device: str = "auto"
    # chatterbox_exaggeration: float = 0.5
    # chatterbox_cfg_weight: float = 0.5
    # chatterbox_chunk_ms: int = 50
    # chatterbox_temperature: float = 0.8
    # chatterbox_top_p: float = 1.0
    # chatterbox_repetition_penalty: float = 1.2
    # chatterbox_synthesis_timeout: int = 30
    # chatterbox_retry_attempts: int = 3
    # chatterbox_audio_postprocess: bool = True
    # chatterbox_fade_ms: int = 10
    # chatterbox_trim_silence: bool = True
    # chatterbox_min_text_chars: int = 20
    # chatterbox_stream_on_clause: bool = True
    # chatterbox_skip_postprocess_streaming: bool = True
    # chatterbox_use_token_streaming: bool = True
    # chatterbox_token_chunk_size: int = 50
    # chatterbox_first_chunk_only_streaming: bool = True
    # chatterbox_streaming_chunk_fades: bool = True

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

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
