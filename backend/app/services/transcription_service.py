"""
RunPod Transcription Service Module
Manages real-time streaming transcription sessions using RunPod serverless backend.
Direct integration with FastAPI - no external server required.
"""

import asyncio
import json
import logging
import threading
import time
import uuid
from typing import Callable, Optional, Dict, Any
from queue import Queue
import numpy as np

# Import RunPod backend from whisper_live
# NOTE: Conditional import - only fails if you actually try to use RunPod backend
try:
    from runpod_backend import ServeClientRunPod
    RUNPOD_AVAILABLE = True
except ImportError:
    RUNPOD_AVAILABLE = False
    logger_import = logging.getLogger(__name__)
    logger_import.warning(
        "RunPod backend not available. If you need RunPod transcription, "
        "place runpod_backend.py in the backend folder. "
        "Using TRANSCRIPTION_BACKEND=local will work without it."
    )

# Import abstract base classes
from app.services.base_transcription_service import TranscriptionSession, TranscriptionService

logger = logging.getLogger(__name__)

# Flag to control transcription logging (set during service initialization)
_transcription_logging_enabled = True


class WebSocketAdapter:
    """
    Adapter to make FastAPI's async WebSocket compatible with ServeClientRunPod's sync expectations.

    ServeClientRunPod expects a synchronous WebSocket with a send() method.
    This adapter wraps FastAPI's async WebSocket and provides a sync interface
    by queuing messages and processing them in the async context.
    """

    def __init__(self, fastapi_websocket, session_id: str):
        """
        Initialize the WebSocket adapter.

        Args:
            fastapi_websocket: FastAPI WebSocket instance (async)
            session_id: Session identifier for logging
        """
        self.fastapi_ws = fastapi_websocket
        self.session_id = session_id
        self.message_queue = Queue()
        self.is_running = True

        # Start async message sender task
        self._sender_task = None

    async def start_sender(self):
        """Start the async message sender task."""
        self._sender_task = asyncio.create_task(self._send_queued_messages())

    async def _send_queued_messages(self):
        """Process queued messages and send them via FastAPI WebSocket."""
        while self.is_running:
            try:
                # Check queue every 50ms
                if not self.message_queue.empty():
                    message = self.message_queue.get_nowait()

                    # Parse JSON and transform for frontend
                    try:
                        data = json.loads(message)

                        # Transform RunPod format → Frontend format
                        if "segments" in data:
                            # RunPod sends: {"uid": "...", "segments": [{...}, {...}]}
                            # Frontend expects: {"type": "transcript", "segment": {...}}
                            segments = data.get("segments", [])
                            if _transcription_logging_enabled:
                                logger.debug(f"Session {self.session_id}: Transforming {len(segments)} segment(s) from RunPod format")

                            for segment in segments:
                                # Convert to frontend format
                                frontend_message = {
                                    "type": "transcript",
                                    "segment": {
                                        "text": segment.get("text", ""),
                                        "is_final": segment.get("completed", False),
                                        "start": segment.get("start", "0.000"),
                                        "end": segment.get("end", "0.000")
                                    }
                                }

                                # Send transformed message
                                await self.fastapi_ws.send_json(frontend_message)

                                # Log transformation with timestamp types
                                if _transcription_logging_enabled:
                                    is_final = frontend_message["segment"]["is_final"]
                                    text_preview = frontend_message["segment"]["text"][:50]
                                    start_ts = frontend_message["segment"]["start"]
                                    end_ts = frontend_message["segment"]["end"]
                                    logger.debug(f"Session {self.session_id}: ✉️ Sent transcript (is_final={is_final}, start={start_ts}, end={end_ts}): '{text_preview}...'")

                        elif "message" in data:
                            # Pass through SERVER_READY and other system messages
                            await self.fastapi_ws.send_json(data)
                            if _transcription_logging_enabled:
                                logger.debug(f"Session {self.session_id}: Sent system message: {data.get('message')}")

                        else:
                            # Unknown format - log and skip
                            logger.warning(f"Session {self.session_id}: Unknown message format, keys: {list(data.keys())}")

                    except json.JSONDecodeError:
                        logger.error(f"Session {self.session_id}: Failed to parse queued message as JSON")
                    except Exception as e:
                        logger.error(f"Session {self.session_id}: Error sending message: {e}")

                await asyncio.sleep(0.05)  # 50ms poll interval

            except Exception as e:
                logger.error(f"Session {self.session_id}: Error in message sender: {e}")
                await asyncio.sleep(0.1)

    def send(self, message: str):
        """
        Synchronous send method (called from RunPod backend thread).
        Queues message for async processing.

        Args:
            message: JSON string to send
        """
        if self.is_running:
            self.message_queue.put(message)
        else:
            logger.warning(f"Session {self.session_id}: Attempted to send message after shutdown")

    async def close(self):
        """Close the adapter and stop message processing."""
        logger.info(f"Session {self.session_id}: Closing WebSocket adapter")
        self.is_running = False

        # Wait for sender task to finish
        if self._sender_task:
            try:
                await asyncio.wait_for(self._sender_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning(f"Session {self.session_id}: Sender task did not finish in time")
                self._sender_task.cancel()


class RunPodTranscriptionSession(TranscriptionSession):
    """
    Per-user transcription session using RunPod serverless backend.
    Manages a single RunPod client instance for real-time audio streaming.
    """

    # Audio configuration (must match expected format)
    RATE = 16000  # 16kHz sample rate
    CHUNK_SIZE = 4096  # Samples per chunk (~256ms)
    CHANNELS = 1

    def __init__(
        self,
        session_id: str,
        fastapi_websocket,
        runpod_endpoint_id: str,
        runpod_api_key: str,
        lang: str,
        model: str,
        use_vad: bool,
        no_speech_thresh: float,
        same_output_threshold: int,
        transcription_interval: float,
        max_buffer_seconds: int,
        callback: Callable[[Dict[str, Any]], None]
    ):
        """
        Initialize a RunPod transcription session.

        Args:
            session_id: Unique session identifier
            fastapi_websocket: FastAPI WebSocket instance
            runpod_endpoint_id: RunPod endpoint ID
            runpod_api_key: RunPod API key
            lang: Language code (e.g., 'en', 'es', 'fr')
            model: Whisper model size (small, medium, large-v3)
            use_vad: Enable Voice Activity Detection
            no_speech_thresh: Silence detection threshold (0.0-1.0)
            same_output_threshold: Number of repeated outputs before finalizing
            transcription_interval: Seconds between API calls
            max_buffer_seconds: Maximum audio buffer size in seconds
            callback: Function to call when transcription segments arrive
        """
        self.session_id = session_id
        self.callback = callback
        self.is_connected = False
        self.shutdown = False

        # Generate unique client ID
        self.client_uid = str(uuid.uuid4())

        # Create WebSocket adapter
        self.ws_adapter = WebSocketAdapter(fastapi_websocket, session_id)

        # Initialize RunPod client
        try:
            if not RUNPOD_AVAILABLE:
                raise ImportError(
                    "RunPod backend is not available. Please add runpod_backend.py to the backend folder "
                    "or use TRANSCRIPTION_BACKEND=local in your .env file."
                )

            logger.info(f"Session {session_id}: Initializing RunPod client...")

            self.runpod_client = ServeClientRunPod(
                websocket=self.ws_adapter,
                task="transcribe",
                language=lang,
                client_uid=self.client_uid,
                model=model,
                use_vad=use_vad,
                no_speech_thresh=no_speech_thresh,
                same_output_threshold=same_output_threshold,
                runpod_endpoint_id=runpod_endpoint_id,
                runpod_api_key=runpod_api_key,
                transcription_interval=transcription_interval,
                max_buffer_seconds=max_buffer_seconds
            )

            self.is_connected = True
            logger.info(f"Session {session_id}: RunPod client initialized successfully")

        except Exception as e:
            logger.error(f"Session {session_id}: Failed to initialize RunPod client - {str(e)}")
            raise

    async def connect(self) -> bool:
        """
        Establish connection (for RunPod, this just starts the adapter).

        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Start the WebSocket adapter's message sender
            await self.ws_adapter.start_sender()
            logger.info(f"Session {self.session_id}: Connection established")
            return True

        except Exception as e:
            logger.error(f"Session {self.session_id}: Connection failed - {str(e)}")
            return False

    async def send_audio(self, audio_data: bytes):
        """
        Send audio data to RunPod backend for transcription.

        Args:
            audio_data: Raw audio bytes (int16 PCM, 16kHz, mono)

        Note:
            Expected format is int16 PCM. This method converts to float32 normalized [-1, 1]
            as required by RunPod backend.
        """
        if not self.is_connected or self.shutdown:
            logger.warning(f"Session {self.session_id}: Cannot send audio - not ready")
            return

        try:
            # Convert int16 -> float32 normalized to [-1, 1]
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            # Log audio reception
            if _transcription_logging_enabled:
                duration_ms = len(audio_array) / self.RATE * 1000
                logger.debug(f"Session {self.session_id}: Received {len(audio_data)} bytes ({len(audio_array)} samples, ~{duration_ms:.1f}ms)")

            # Send to RunPod backend (calls add_frames in the background)
            self.runpod_client.add_frames(audio_array)

            # Log buffer status
            if _transcription_logging_enabled and hasattr(self.runpod_client, 'frames_np') and self.runpod_client.frames_np is not None:
                buffer_samples = self.runpod_client.frames_np.shape[0]
                buffer_seconds = buffer_samples / self.RATE
                logger.debug(f"Session {self.session_id}: Buffer now has {buffer_seconds:.2f}s of audio")

        except Exception as e:
            logger.error(f"Session {self.session_id}: Error sending audio - {str(e)}")

    async def trigger_final_flush(self):
        """Trigger final flush without closing WebSocket - used when user stops recording."""
        import time
        start_time = time.time()

        logger.info(f"Session {self.session_id}: Triggering final flush...")
        self.shutdown = True

        # Set exit flag to trigger final flush in RunPod client thread
        try:
            self.runpod_client.exit = True

            # Wait for the transcription thread to complete final flush
            # The thread will process remaining buffered audio and send final segments
            logger.info(f"Session {self.session_id}: Waiting for final flush to complete...")

            # Give the thread time to process and send final segments (max 5 seconds)
            max_wait = 5.0
            elapsed = 0.0
            while elapsed < max_wait:
                # Check if thread is still running
                if hasattr(self.runpod_client, 't_speech_to_text'):
                    if not self.runpod_client.t_speech_to_text.is_alive():
                        logger.info(f"Session {self.session_id}: Transcription thread finished")
                        break
                await asyncio.sleep(0.1)
                elapsed += 0.1

            flush_duration = time.time() - start_time
            logger.info(f"Session {self.session_id}: Final flush completed in {flush_duration:.2f}s")

            # Send "complete" message to frontend
            try:
                complete_message = json.dumps({
                    "type": "complete",
                    "message": "Transcription complete, all segments sent"
                })
                self.ws_adapter.send(complete_message)
                logger.info(f"Session {self.session_id}: ✅ Sent 'complete' message to frontend after final flush")

                # Wait briefly for message to be sent
                await asyncio.sleep(0.2)
            except Exception as e:
                logger.error(f"Session {self.session_id}: Error sending complete message: {e}")

        except Exception as e:
            logger.error(f"Session {self.session_id}: Error during final flush - {str(e)}")

        # DO NOT close WebSocket - frontend will close it after receiving "complete" message
        logger.info(f"Session {self.session_id}: Final flush complete, WebSocket remains open for frontend to close")

    async def close(self):
        """Gracefully close the transcription session."""
        import time
        start_time = time.time()

        logger.info(f"Session {self.session_id}: Closing session...")

        # Check if final flush was already triggered
        if not self.shutdown:
            self.shutdown = True

            # Cleanup RunPod client
            try:
                logger.info(f"Session {self.session_id}: Cleaning up RunPod client...")
                self.runpod_client.exit = True

                # Wait for cleanup to complete (this now includes final flush)
                # No artificial delay - await the async cleanup directly
                cleanup_start = time.time()
                logger.info(f"Session {self.session_id}: Waiting for cleanup to complete (includes final flush)...")
                await self.runpod_client.cleanup()

                cleanup_duration = time.time() - cleanup_start
                logger.info(f"Session {self.session_id}: Cleanup completed in {cleanup_duration:.2f}s")

                # Send "complete" message to frontend BEFORE closing WebSocket
                try:
                    complete_message = json.dumps({
                        "type": "complete",
                        "message": "Transcription complete, all segments sent"
                    })
                    self.ws_adapter.send(complete_message)
                    logger.info(f"Session {self.session_id}: ✅ Sent 'complete' message to frontend")

                    # Wait briefly for message to be sent (200ms)
                    await asyncio.sleep(0.2)
                except Exception as e:
                    logger.error(f"Session {self.session_id}: Error sending complete message: {e}")

            except Exception as e:
                logger.error(f"Session {self.session_id}: Error cleaning up RunPod client - {str(e)}")
        else:
            logger.info(f"Session {self.session_id}: Final flush already completed, skipping cleanup")

        # Close WebSocket adapter (now frontend knows all segments were sent)
        try:
            await self.ws_adapter.close()
            logger.info(f"Session {self.session_id}: WebSocket closed")
        except Exception as e:
            logger.warning(f"Session {self.session_id}: Error closing WebSocket adapter: {e}")

        self.is_connected = False
        total_duration = time.time() - start_time
        logger.info(f"Session {self.session_id}: Session closed (total time: {total_duration:.2f}s)")

    def get_transcript(self) -> str:
        """
        Get the full transcript of completed segments.

        Returns:
            Complete transcript text
        """
        # RunPod client maintains transcript in base class
        if hasattr(self.runpod_client, 'transcript'):
            return "".join([seg.get("text", "") for seg in self.runpod_client.transcript])
        return ""


class RunPodTranscriptionService(TranscriptionService):
    """
    Singleton service for managing RunPod transcription sessions.
    Handles configuration and session lifecycle management.
    """

    def __init__(
        self,
        runpod_endpoint_id: str,
        runpod_api_key: str,
        lang: str = "en",
        model: str = "small",
        use_vad: bool = True,
        no_speech_thresh: float = 0.45,
        same_output_threshold: int = 7,
        transcription_interval: float = 3.0,
        max_buffer_seconds: int = 30
    ):
        """
        Initialize the RunPod transcription service.

        Args:
            runpod_endpoint_id: RunPod endpoint ID (required)
            runpod_api_key: RunPod API key (required)
            lang: Default language code (e.g., 'en', 'es', 'fr')
            model: Whisper model size (small, medium, large-v3)
            use_vad: Enable Voice Activity Detection
            no_speech_thresh: Silence detection threshold (0.0-1.0)
            same_output_threshold: Number of repeated outputs before finalizing
            transcription_interval: Seconds between API calls
            max_buffer_seconds: Maximum audio buffer size in seconds
        """
        if not runpod_endpoint_id:
            raise ValueError("runpod_endpoint_id is required")
        if not runpod_api_key:
            raise ValueError("runpod_api_key is required")

        self.runpod_endpoint_id = runpod_endpoint_id
        self.runpod_api_key = runpod_api_key
        self.lang = lang
        self.model = model
        self.use_vad = use_vad
        self.no_speech_thresh = no_speech_thresh
        self.same_output_threshold = same_output_threshold
        self.transcription_interval = transcription_interval
        self.max_buffer_seconds = max_buffer_seconds

        # Track active sessions
        self.sessions: Dict[str, RunPodTranscriptionSession] = {}

        logger.info(f"RunPod transcription service initialized (model={model}, lang={lang}, endpoint={runpod_endpoint_id[:8]}...)")

    def create_session(
        self,
        session_id: str,
        fastapi_websocket,
        callback: Callable[[Dict[str, Any]], None],
        lang: Optional[str] = None,
        model: Optional[str] = None
    ) -> RunPodTranscriptionSession:
        """
        Create a new transcription session.

        Args:
            session_id: Unique session identifier
            fastapi_websocket: FastAPI WebSocket instance
            callback: Function to call when transcription segments arrive
            lang: Override default language (optional)
            model: Override default model (optional)

        Returns:
            RunPodTranscriptionSession instance

        Raises:
            ValueError: If session_id already exists
        """
        if session_id in self.sessions:
            raise ValueError(f"Session {session_id} already exists")

        session = RunPodTranscriptionSession(
            session_id=session_id,
            fastapi_websocket=fastapi_websocket,
            runpod_endpoint_id=self.runpod_endpoint_id,
            runpod_api_key=self.runpod_api_key,
            lang=lang or self.lang,
            model=model or self.model,
            use_vad=self.use_vad,
            no_speech_thresh=self.no_speech_thresh,
            same_output_threshold=self.same_output_threshold,
            transcription_interval=self.transcription_interval,
            max_buffer_seconds=self.max_buffer_seconds,
            callback=callback
        )

        self.sessions[session_id] = session
        logger.info(f"Created RunPod transcription session: {session_id}")
        return session

    def get_session(self, session_id: str) -> Optional[RunPodTranscriptionSession]:
        """
        Get an existing transcription session.

        Args:
            session_id: Session identifier

        Returns:
            RunPodTranscriptionSession if found, None otherwise
        """
        return self.sessions.get(session_id)

    async def close_session(self, session_id: str):
        """
        Close and remove a transcription session.

        Args:
            session_id: Session identifier
        """
        session = self.sessions.get(session_id)
        if session:
            await session.close()
            del self.sessions[session_id]
            logger.info(f"Closed and removed session: {session_id}")

    async def close_all(self):
        """Close all active transcription sessions."""
        logger.info(f"Closing {len(self.sessions)} active sessions...")
        for session_id in list(self.sessions.keys()):
            await self.close_session(session_id)
        logger.info("All sessions closed")


# Global service instance
_runpod_service: Optional[RunPodTranscriptionService] = None


def initialize_runpod_service(
    runpod_endpoint_id: str,
    runpod_api_key: str,
    lang: str = "en",
    model: str = "small",
    use_vad: bool = True,
    no_speech_thresh: float = 0.45,
    same_output_threshold: int = 7,
    transcription_interval: float = 3.0,
    max_buffer_seconds: int = 30,
    transcription_logging_enabled: bool = True
) -> RunPodTranscriptionService:
    """
    Initialize the global RunPod transcription service instance.

    Args:
        runpod_endpoint_id: RunPod endpoint ID (required)
        runpod_api_key: RunPod API key (required)
        lang: Default language code
        model: Whisper model size
        use_vad: Enable Voice Activity Detection
        no_speech_thresh: Silence detection threshold
        same_output_threshold: Number of repeated outputs before finalizing
        transcription_interval: Seconds between API calls
        max_buffer_seconds: Maximum audio buffer size in seconds
        transcription_logging_enabled: Enable/disable verbose transcription logs

    Returns:
        RunPodTranscriptionService instance
    """
    global _runpod_service, _transcription_logging_enabled

    # Set the module-level logging flag
    _transcription_logging_enabled = transcription_logging_enabled

    if transcription_logging_enabled:
        logger.info("Transcription logging is ENABLED")
    else:
        logger.info("Transcription logging is DISABLED (set TRANSCRIPTION_LOGGING_ENABLED=true to enable)")

    _runpod_service = RunPodTranscriptionService(
        runpod_endpoint_id=runpod_endpoint_id,
        runpod_api_key=runpod_api_key,
        lang=lang,
        model=model,
        use_vad=use_vad,
        no_speech_thresh=no_speech_thresh,
        same_output_threshold=same_output_threshold,
        transcription_interval=transcription_interval,
        max_buffer_seconds=max_buffer_seconds
    )
    return _runpod_service


def get_runpod_service() -> RunPodTranscriptionService:
    """
    Get the global RunPod transcription service instance.

    Returns:
        RunPodTranscriptionService instance

    Raises:
        RuntimeError: If service not initialized
    """
    if _runpod_service is None:
        raise RuntimeError("RunPod transcription service not initialized. Call initialize_runpod_service() first.")
    return _runpod_service


async def cleanup_runpod_service():
    """Cleanup the global RunPod transcription service instance."""
    global _runpod_service
    if _runpod_service:
        await _runpod_service.close_all()
        _runpod_service = None
        logger.info("RunPod transcription service cleaned up")
