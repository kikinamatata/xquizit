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

    # TTS Backend Selection
    tts_backend: str = "kokoro"  # "kokoro" or "websocket"

    # Kokoro TTS Configuration (used when tts_backend="kokoro")
    tts_device: str = "auto"
    kokoro_voice: str = "af_bella"
    kokoro_speed: float = 1.0

    # WebSocket TTS Configuration (used when tts_backend="websocket")
    tts_server_url: str = "ws://localhost:8765"

    # LLM Optimization Parameters
    gemini_thinking_budget: int = 0
    gemini_include_thoughts: bool = False
    gemini_max_output_tokens: int = 1024
    gemini_temperature: float = 0.7

    # Transcription Backend Selection
    transcription_backend: str = "runpod"  # "runpod" or "local"

    # RunPod Serverless Transcription (used when transcription_backend="runpod")
    runpod_endpoint_id: Optional[str] = None
    runpod_api_key: Optional[str] = None
    runpod_max_buffer_seconds: int = 30

    # Local WhisperLive Server Configuration (used when transcription_backend="local")
    # Option 1: Provide complete URL (e.g., "ws://192.168.1.100:9090" or "wss://example.com:443")
    whisperlive_server_url: Optional[str] = None
    # Option 2: Provide host, port, and SSL separately (used if whisperlive_server_url is not set)
    whisperlive_server_host: Optional[str] = None  # e.g., "192.168.1.100" or "localhost"
    whisperlive_server_port: int = 9090
    whisperlive_use_ssl: bool = False  # Use wss:// instead of ws://

    # Transcription Settings (shared by both backends)
    whisperlive_language: str = "en"
    whisperlive_model: str = "large-v3"
    whisperlive_use_vad: bool = True
    whisperlive_no_speech_thresh: float = 0.45  # Silence detection threshold
    whisperlive_same_output_threshold: int = 3  # Repeated outputs threshold (used by both backends)
    whisperlive_transcription_interval: float = 3.0  # RunPod-specific: seconds between API calls
    transcription_logging_enabled: bool = True  # Enable/disable verbose transcription logs

    # Advanced Transcription Settings (local WhisperLive only)
    whisperlive_send_last_n_segments: int = 10  # Number of recent segments to send for context
    whisperlive_clip_audio: bool = False  # Clip audio with no valid segments
    whisperlive_chunking_mode: str = "vad"  # Audio chunking strategy: "vad" or "time_based"
    whisperlive_chunk_interval: float = 2.0  # Interval for time_based chunking (seconds)

    # Translation Settings (local WhisperLive only)
    whisperlive_enable_translation: bool = False  # Enable live translation
    whisperlive_target_language: str = "en"  # Target language for translation

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
