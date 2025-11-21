"""
Base TTS Service Interface
Abstract base class for TTS service implementations.
"""

from abc import ABC, abstractmethod
from typing import AsyncIterator


class BaseTTSService(ABC):
    """
    Abstract base class for TTS service implementations.

    All TTS services must implement this interface to ensure consistency
    across different backends (KokoroTTS, WebSocket TTS, etc.)
    """

    @abstractmethod
    async def generate_stream(
        self,
        text: str,
        audio_index: int = 0
    ) -> AsyncIterator[dict]:
        """
        Generate audio from text using TTS engine.

        Args:
            text: Text to convert to speech
            audio_index: Index number for this audio chunk (for ordering)

        Yields:
            Dictionary with the following structure:
            {
                "type": "audio_chunk",
                "audio": str,  # Base64-encoded audio data (WAV format)
                "index": int,  # Audio index (for ordering multiple sentences)
                "chunk_index": int  # Optional: chunk number within the audio
            }
        """
        pass

    @abstractmethod
    def cleanup(self):
        """
        Clean up resources used by the TTS service.

        This method should release any resources, close connections,
        and perform any necessary cleanup operations.
        """
        pass
