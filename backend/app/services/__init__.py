"""
Services module for xquizit backend.
"""

from app.services.embedded_whisper_service import (
    EmbeddedWhisperService,
    EmbeddedWhisperSession,
    initialize_embedded_whisper_service,
    get_embedded_whisper_service,
    cleanup_embedded_whisper_service,
)
from app.services.tts_service import (
    initialize_tts_service,
    get_tts_service,
    cleanup_tts_service,
)
from app.services.document_processor import (
    extract_text_from_document,
    DocumentProcessingError,
)

__all__ = [
    # Embedded Whisper Transcription
    "EmbeddedWhisperService",
    "EmbeddedWhisperSession",
    "initialize_embedded_whisper_service",
    "get_embedded_whisper_service",
    "cleanup_embedded_whisper_service",
    # TTS
    "initialize_tts_service",
    "get_tts_service",
    "cleanup_tts_service",
    # Document Processing
    "extract_text_from_document",
    "DocumentProcessingError",
]
