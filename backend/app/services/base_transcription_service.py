"""
Abstract Base Transcription Service Module
Defines the interface for transcription services (RunPod, WhisperLive, etc.)
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional, Dict, Any


class TranscriptionSession(ABC):
    """
    Abstract base class for transcription sessions.
    Implementations: RunPodTranscriptionSession, LocalWhisperLiveSession
    """

    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to transcription backend.

        Returns:
            True if connection successful, False otherwise
        """
        pass

    @abstractmethod
    async def send_audio(self, audio_data: bytes):
        """
        Send audio data to transcription backend.

        Args:
            audio_data: Raw audio bytes (int16 PCM, 16kHz, mono)
        """
        pass

    @abstractmethod
    async def trigger_final_flush(self):
        """
        Trigger final flush to process remaining buffered audio.
        Used when user stops recording.
        """
        pass

    @abstractmethod
    async def close(self):
        """Gracefully close the transcription session."""
        pass

    @abstractmethod
    def get_transcript(self) -> str:
        """
        Get the full transcript of completed segments.

        Returns:
            Complete transcript text
        """
        pass


class TranscriptionService(ABC):
    """
    Abstract base class for transcription services.
    Implementations: RunPodTranscriptionService, LocalWhisperLiveService
    """

    @abstractmethod
    def create_session(
        self,
        session_id: str,
        fastapi_websocket,
        callback: Callable[[Dict[str, Any]], None],
        lang: Optional[str] = None,
        model: Optional[str] = None
    ) -> TranscriptionSession:
        """
        Create a new transcription session.

        Args:
            session_id: Unique session identifier
            fastapi_websocket: FastAPI WebSocket instance
            callback: Function to call when transcription segments arrive
            lang: Override default language (optional)
            model: Override default model (optional)

        Returns:
            TranscriptionSession instance

        Raises:
            ValueError: If session_id already exists
        """
        pass

    @abstractmethod
    def get_session(self, session_id: str) -> Optional[TranscriptionSession]:
        """
        Get an existing transcription session.

        Args:
            session_id: Session identifier

        Returns:
            TranscriptionSession if found, None otherwise
        """
        pass

    @abstractmethod
    async def close_session(self, session_id: str):
        """
        Close and remove a transcription session.

        Args:
            session_id: Session identifier
        """
        pass

    @abstractmethod
    async def close_all(self):
        """Close all active transcription sessions."""
        pass
