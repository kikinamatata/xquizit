"""
TTS Service Module
RealtimeTTS with Kokoro engine for real-time audio generation with streaming.
"""

import logging
import base64
import io
import time
from typing import Optional, AsyncIterator
import asyncio
from queue import Queue, Empty

from .base_tts_service import BaseTTSService

logger = logging.getLogger(__name__)


class KokoroTTS(BaseTTSService):
    """
    RealtimeTTS wrapper with Kokoro engine for streaming text-to-speech conversion.

    Uses RealtimeTTS with KokoroEngine for high-quality, low-latency audio.
    Requires Python < 3.13 and realtimetts[kokoro] package.
    """

    def __init__(
        self,
        device: str = "auto",
        voice: str = "af_bella",
        speed: float = 1.0
    ):
        """
        Initialize the Kokoro TTS service using RealtimeTTS.

        Args:
            device: Device to run the model on ('cuda', 'cpu', or 'auto')
            voice: Kokoro voice to use (af_bella, af_sarah, am_adam, am_michael)
            speed: Speech speed multiplier (default: 1.0)
        """
        self.device = device
        self.voice = voice
        self.speed = speed
        self.engine = None

        self._load_model()

    def _load_model(self):
        """Load the Kokoro TTS engine for direct audio generation (no playback)."""
        try:
            from RealtimeTTS.engines import KokoroEngine
            import torch

            logger.info(f"Loading Kokoro TTS engine (direct synthesis mode)...")

            # Determine device
            if self.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.device
                if device == "cuda" and not torch.cuda.is_available():
                    logger.warning("CUDA not available, falling back to CPU")
                    device = "cpu"

            logger.info(f"Using device: {device}")

            # Initialize Kokoro engine (no parameters in __init__)
            self.engine = KokoroEngine()

            # Set voice and speed via methods
            self.engine.set_voice(self.voice)
            self.engine.set_speed(self.speed)

            logger.info(f"Kokoro TTS engine loaded successfully with voice: {self.voice}, speed: {self.speed}")
            logger.info(f"Using direct synthesis mode (no audio device required)")

        except ImportError as e:
            logger.error("RealtimeTTS library not installed")
            logger.error(f"Error: {str(e)}")
            logger.error("Please install with: pip install realtimetts[kokoro]")
            raise ImportError(
                "RealtimeTTS with Kokoro is not installed. "
                "Please install with: pip install realtimetts[kokoro]"
            ) from e
        except Exception as e:
            logger.error(f"Failed to load Kokoro TTS engine: {str(e)}")
            raise

    async def generate_stream(
        self,
        text: str,
        audio_index: int = 0
    ) -> AsyncIterator[dict]:
        """
        Generate audio from text using direct engine synthesis (no playback).

        Args:
            text: Text to convert to speech
            audio_index: Index number for this audio chunk (for ordering)

        Yields:
            Dictionary with 'type', 'audio' (base64), and 'index' keys
        """
        if not self.engine:
            raise RuntimeError("TTS engine not initialized")

        if not text or not text.strip():
            logger.warning("Empty text provided to TTS")
            return

        try:
            logger.info(f"Generating audio for text: '{text[:50]}...' (index: {audio_index})")

            # Start timing
            start_time = time.time()
            chunk_count = 0

            # Run direct synthesis in thread pool to avoid blocking
            loop = asyncio.get_event_loop()

            # Clear any existing items in the queue before synthesis
            while not self.engine.queue.empty():
                try:
                    self.engine.queue.get_nowait()
                except Empty:
                    break

            # Call engine.synthesize() directly (bypasses playback layer)
            synthesis_start = time.time()
            synthesis_success = await loop.run_in_executor(
                None,
                lambda: self.engine.synthesize(text)
            )
            synthesis_duration = time.time() - synthesis_start

            if not synthesis_success:
                logger.error(f"Synthesis failed for text: '{text[:50]}...'")
                return

            # Extract all audio chunks from engine's internal queue
            audio_chunks = []
            while not self.engine.queue.empty():
                try:
                    chunk = self.engine.queue.get_nowait()
                    if chunk and len(chunk) > 0:
                        audio_chunks.append(chunk)
                except Empty:
                    break

            # Yield collected audio chunks
            for chunk in audio_chunks:
                audio_bytes = self._audio_to_bytes(chunk)
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

                chunk_count += 1
                yield {
                    "type": "audio_chunk",
                    "audio": audio_base64,
                    "index": audio_index,
                    "chunk_index": chunk_count
                }

            total_duration = time.time() - start_time

            # Log timing
            from app.utils.timing import timing_logger
            timing_logger.info(
                f"TTS - Audio Generation: {total_duration:.3f}s | "
                f"{{\"text_length\": {len(text)}, \"synthesis_duration\": {synthesis_duration:.3f}, "
                f"\"chunks\": {chunk_count}, \"index\": {audio_index}}}"
            )

            logger.info(f"Generated {chunk_count} audio chunks for index {audio_index}")

        except Exception as e:
            logger.error(f"Error generating audio: {str(e)}", exc_info=True)
            raise

    def _audio_to_bytes(self, audio_data) -> bytes:
        """
        Convert audio data to bytes for transmission.

        Args:
            audio_data: Audio bytes or numpy array from RealtimeTTS

        Returns:
            Audio data as bytes (WAV format)
        """
        try:
            import wave
            import numpy as np

            # Convert to numpy array if needed
            if isinstance(audio_data, bytes):
                audio_np = np.frombuffer(audio_data, dtype=np.int16)
            elif isinstance(audio_data, np.ndarray):
                audio_np = audio_data
            else:
                audio_np = np.array(audio_data, dtype=np.int16)

            # Ensure proper shape
            if audio_np.ndim > 1:
                audio_np = audio_np.flatten()

            # Create WAV file in memory
            buffer = io.BytesIO()
            with wave.open(buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(24000)  # Kokoro uses 24kHz
                wav_file.writeframes(audio_np.tobytes())

            return buffer.getvalue()

        except Exception as e:
            logger.error(f"Error converting audio to bytes: {str(e)}")
            raise

    def cleanup(self):
        """Clean up resources."""
        try:
            if self.engine:
                try:
                    # Clear the queue
                    while not self.engine.queue.empty():
                        try:
                            self.engine.queue.get_nowait()
                        except Empty:
                            break

                    # Shutdown engine
                    self.engine.shutdown()
                except Exception as e:
                    logger.debug(f"Error during engine shutdown: {e}")
                    pass  # Ignore errors during shutdown

                del self.engine
                self.engine = None

            # Clear GPU cache if using CUDA
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

            logger.info("TTS service cleaned up successfully")
        except Exception as e:
            logger.warning(f"Error during cleanup: {str(e)}")


# Global TTS service instance (initialized on startup)
_tts_service: Optional[BaseTTSService] = None


def initialize_tts_service(
    backend: str = "kokoro",
    # Kokoro-specific parameters
    device: str = "auto",
    voice: str = "af_bella",
    speed: float = 1.0,
    # WebSocket-specific parameters
    server_url: str = "ws://localhost:8765"
) -> BaseTTSService:
    """
    Initialize the global TTS service instance using factory pattern.

    Args:
        backend: TTS backend to use ("kokoro" or "websocket")
        device: Device to run Kokoro on ('cuda', 'cpu', or 'auto') [Kokoro only]
        voice: Kokoro voice to use (af_bella, af_sarah, am_adam, am_michael) [Kokoro only]
        speed: Speech speed multiplier [Kokoro only]
        server_url: WebSocket server URL (e.g., ws://localhost:8765) [WebSocket only]

    Returns:
        Initialized TTS service instance (KokoroTTS or WebSocketTTSService)

    Raises:
        ValueError: If backend is not recognized
        RuntimeError: If TTS service initialization fails
    """
    global _tts_service

    if _tts_service is not None:
        logger.warning("TTS service already initialized, returning existing instance")
        return _tts_service

    try:
        if backend == "kokoro":
            logger.info(f"Initializing Kokoro TTS backend (device={device}, voice={voice}, speed={speed})")
            _tts_service = KokoroTTS(
                device=device,
                voice=voice,
                speed=speed
            )

        elif backend == "websocket":
            logger.info(f"Initializing WebSocket TTS backend (server={server_url})")
            from .websocket_tts_service import WebSocketTTSService

            _tts_service = WebSocketTTSService(server_url=server_url)

            # Connect to server asynchronously
            # Note: Connection will happen on first use if not connected
            # This allows the app to start even if TTS server is temporarily unavailable
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Schedule connection for later
                    asyncio.create_task(_tts_service.connect())
                else:
                    # Connect immediately
                    loop.run_until_complete(_tts_service.connect())
            except Exception as e:
                logger.warning(f"Could not connect to TTS server during initialization: {e}")
                logger.warning("TTS service will attempt to connect on first use")

        else:
            raise ValueError(f"Unknown TTS backend: {backend}. Must be 'kokoro' or 'websocket'")

        logger.info(f"TTS service initialized successfully with backend: {backend}")
        return _tts_service

    except Exception as e:
        logger.error(f"Failed to initialize TTS service: {e}", exc_info=True)
        _tts_service = None
        raise


def get_tts_service() -> BaseTTSService:
    """
    Get the global TTS service instance.

    Returns:
        TTS service instance (KokoroTTS or WebSocketTTSService)

    Raises:
        RuntimeError: If TTS service not initialized
    """
    if _tts_service is None:
        raise RuntimeError("TTS service not initialized. Call initialize_tts_service() first.")

    return _tts_service


def cleanup_tts_service():
    """Clean up the global TTS service instance."""
    global _tts_service

    if _tts_service:
        _tts_service.cleanup()
        _tts_service = None
        logger.info("TTS service cleaned up")
