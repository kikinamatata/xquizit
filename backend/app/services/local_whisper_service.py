"""
Local WhisperLive Transcription Service Module
Manages real-time streaming transcription sessions using local WhisperLive server.
Connects to WhisperLive server running on separate GPU system via WebSocket.
"""

import asyncio
import json
import logging
import numpy as np
import websockets
from typing import Callable, Optional, Dict, Any
from queue import Queue

from app.services.base_transcription_service import TranscriptionSession, TranscriptionService

logger = logging.getLogger(__name__)

# Flag to control transcription logging (set during service initialization)
_transcription_logging_enabled = True


class LocalWhisperLiveSession(TranscriptionSession):
    """
    Per-user transcription session using local WhisperLive server.
    Manages a WebSocket connection to remote WhisperLive server for real-time audio streaming.
    """

    # Audio configuration (must match expected format)
    RATE = 16000  # 16kHz sample rate
    CHUNK_SIZE = 4096  # Samples per chunk (~256ms)
    CHANNELS = 1

    def __init__(
        self,
        session_id: str,
        fastapi_websocket,
        server_url: str,
        lang: str,
        model: str,
        use_vad: bool,
        no_speech_thresh: float,
        send_last_n_segments: int,
        same_output_threshold: int,
        clip_audio: bool,
        chunking_mode: str,
        chunk_interval: float,
        enable_translation: bool,
        target_language: str,
        callback: Callable[[Dict[str, Any]], None]
    ):
        """
        Initialize a local WhisperLive transcription session.

        Args:
            session_id: Unique session identifier
            fastapi_websocket: FastAPI WebSocket instance (for sending results to frontend)
            server_url: WhisperLive server WebSocket URL (e.g., ws://192.168.1.100:9090)
            lang: Language code (e.g., 'en', 'es', 'fr')
            model: Whisper model size (small, medium, large-v3)
            use_vad: Enable Voice Activity Detection
            no_speech_thresh: Silence detection threshold (0.0-1.0)
            send_last_n_segments: Number of recent segments to send for context
            same_output_threshold: Number of repeated outputs before finalizing
            clip_audio: Remove audio segments with no valid speech
            chunking_mode: Audio chunking strategy ("vad" or "time_based")
            chunk_interval: Interval for time-based chunking (seconds)
            enable_translation: Enable live translation
            target_language: Target language for translation
            callback: Function to call when transcription segments arrive
        """
        self.session_id = session_id
        self.fastapi_websocket = fastapi_websocket
        self.server_url = server_url
        self.lang = lang
        self.model = model
        self.use_vad = use_vad
        self.no_speech_thresh = no_speech_thresh
        self.send_last_n_segments = send_last_n_segments
        self.same_output_threshold = same_output_threshold
        self.clip_audio = clip_audio
        self.chunking_mode = chunking_mode
        self.chunk_interval = chunk_interval
        self.enable_translation = enable_translation
        self.target_language = target_language
        self.callback = callback

        self.is_connected = False
        self.shutdown = False

        # WebSocket connection to WhisperLive server
        self.whisper_ws: Optional[websockets.WebSocketClientProtocol] = None

        # Background task for receiving transcription results
        self._receiver_task: Optional[asyncio.Task] = None

        # Transcript accumulator
        self.transcript_segments = []

        # Event to track final flush completion
        self.flush_complete_event = asyncio.Event()

        logger.info(f"Session {session_id}: Initialized local WhisperLive session (server={server_url})")

    async def connect(self) -> bool:
        """
        Establish WebSocket connection to WhisperLive server.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info(f"Session {self.session_id}: Connecting to WhisperLive server at {self.server_url}...")

            # Connect to WhisperLive server
            self.whisper_ws = await websockets.connect(
                self.server_url,
                ping_interval=20,
                ping_timeout=10
            )

            logger.info(f"Session {self.session_id}: WebSocket connected to WhisperLive server")

            # Send handshake configuration
            handshake = {
                "uid": self.session_id,
                "language": self.lang,
                "task": "transcribe",
                "model": self.model,
                "use_vad": self.use_vad,
                "send_last_n_segments": self.send_last_n_segments,
                "no_speech_thresh": self.no_speech_thresh,
                "same_output_threshold": self.same_output_threshold,
                "clip_audio": self.clip_audio,
                "chunking_mode": self.chunking_mode,
                "chunk_interval": self.chunk_interval,
                "enable_translation": self.enable_translation,
                "target_language": self.target_language if self.enable_translation else None
            }

            await self.whisper_ws.send(json.dumps(handshake))
            logger.info(f"Session {self.session_id}: Sent handshake: {handshake}")

            # Wait for SERVER_READY response
            response = await asyncio.wait_for(self.whisper_ws.recv(), timeout=10.0)
            response_data = json.loads(response)

            if response_data.get("message") == "SERVER_READY":
                logger.info(f"Session {self.session_id}: Server ready (backend={response_data.get('backend')})")
                self.is_connected = True

                # Start background receiver task
                self._receiver_task = asyncio.create_task(self._receive_transcriptions())

                # Send ready message to frontend
                await self.fastapi_websocket.send_json({
                    "type": "ready",
                    "message": "Transcription service ready"
                })

                return True
            else:
                logger.error(f"Session {self.session_id}: Unexpected server response: {response_data}")
                return False

        except asyncio.TimeoutError:
            logger.error(f"Session {self.session_id}: Timeout waiting for server ready")
            return False
        except Exception as e:
            logger.error(f"Session {self.session_id}: Connection failed - {str(e)}")
            return False

    async def _receive_transcriptions(self):
        """
        Background task to receive transcription results from WhisperLive server.
        Transforms WhisperLive format to application format and sends to frontend.
        """
        try:
            while self.is_connected and not self.shutdown:
                try:
                    # Receive message from WhisperLive server
                    message = await asyncio.wait_for(self.whisper_ws.recv(), timeout=1.0)

                    # Parse JSON response
                    data = json.loads(message)

                    # Check for FLUSH_COMPLETE signal
                    if data.get("message") == "FLUSH_COMPLETE":
                        logger.info(f"Session {self.session_id}: Received FLUSH_COMPLETE signal from server")
                        self.flush_complete_event.set()
                        continue

                    if "segments" in data:
                        segments = data.get("segments", [])

                        # Log received segments for debugging
                        logger.info(f"Session {self.session_id}: 📨 Received {len(segments)} segment(s) from trans_server")

                        # Transform each segment and send to frontend
                        for i, segment in enumerate(segments):
                            completed_value = segment.get("completed", False)
                            text_preview = segment.get("text", "")[:60]

                            # Log each segment with its completed status
                            logger.info(
                                f"  Segment {i+1}/{len(segments)}: "
                                f"completed={completed_value} (type={type(completed_value).__name__}) "
                                f"start={segment.get('start')} end={segment.get('end')} "
                                f"text='{text_preview}...'"
                            )
                            # WhisperLive format: {"text": "...", "start": "0.000", "end": "1.500", "completed": false}
                            # Application format: {"type": "transcript", "segment": {"text": "...", "is_final": ..., "start": ..., "end": ...}}

                            frontend_message = {
                                "type": "transcript",
                                "segment": {
                                    "text": segment.get("text", ""),
                                    "is_final": segment.get("completed", False),
                                    "start": segment.get("start", "0.000"),
                                    "end": segment.get("end", "0.000")
                                }
                            }

                            # Log what's being sent to frontend
                            is_final_value = frontend_message["segment"]["is_final"]
                            logger.info(
                                f"    → Sending to frontend: is_final={is_final_value} "
                                f"(type={type(is_final_value).__name__})"
                            )

                            # Send to frontend via FastAPI WebSocket
                            await self.fastapi_websocket.send_json(frontend_message)

                            # Store in transcript
                            self.transcript_segments.append(segment)

                            if _transcription_logging_enabled:
                                is_final = frontend_message["segment"]["is_final"]
                                text_preview = frontend_message["segment"]["text"][:50]
                                logger.debug(f"Session {self.session_id}: ✉️ Sent transcript (is_final={is_final}): '{text_preview}...'")

                except asyncio.TimeoutError:
                    # No message received, continue listening
                    continue
                except json.JSONDecodeError as e:
                    logger.error(f"Session {self.session_id}: Failed to parse server response: {e}")
                except Exception as e:
                    if self.shutdown:
                        break
                    logger.error(f"Session {self.session_id}: Error receiving transcription: {e}")

        except Exception as e:
            logger.error(f"Session {self.session_id}: Receiver task error: {e}")
        finally:
            logger.info(f"Session {self.session_id}: Receiver task stopped")

    async def send_audio(self, audio_data: bytes):
        """
        Send audio data to WhisperLive server for transcription.

        Args:
            audio_data: Raw audio bytes (int16 PCM, 16kHz, mono)
        """
        if not self.is_connected or self.shutdown or not self.whisper_ws:
            logger.warning(f"Session {self.session_id}: Cannot send audio - not connected")
            return

        try:
            # Convert int16 PCM → float32 normalized to [-1.0, 1.0]
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            if _transcription_logging_enabled:
                duration_ms = len(audio_array) / self.RATE * 1000
                logger.debug(f"Session {self.session_id}: Sending {len(audio_data)} bytes ({len(audio_array)} samples, ~{duration_ms:.1f}ms)")

            # Send as binary float32 frames to WhisperLive server
            audio_bytes = audio_array.tobytes()
            await self.whisper_ws.send(audio_bytes)

        except Exception as e:
            logger.error(f"Session {self.session_id}: Error sending audio - {str(e)}")

    async def trigger_final_flush(self):
        """
        Trigger final flush to process remaining buffered audio.
        Sends END_OF_AUDIO signal to WhisperLive server and waits for FLUSH_COMPLETE confirmation.
        """
        logger.info(f"Session {self.session_id}: Triggering final flush...")
        # DON'T set shutdown yet - receiver needs to stay alive to receive final segments

        try:
            if self.whisper_ws:
                # Send END_OF_AUDIO signal
                await self.whisper_ws.send(b"END_OF_AUDIO")
                logger.info(f"Session {self.session_id}: Sent END_OF_AUDIO signal")

                # Wait for FLUSH_COMPLETE signal from server (with timeout)
                timeout = 2.5  # Max 2.5 seconds (reduced for speed optimization)
                try:
                    await asyncio.wait_for(self.flush_complete_event.wait(), timeout=timeout)
                    logger.info(f"Session {self.session_id}: Flush complete confirmed by server")
                except asyncio.TimeoutError:
                    logger.warning(f"Session {self.session_id}: Timeout waiting for FLUSH_COMPLETE after {timeout}s, proceeding anyway")

            logger.info(f"Session {self.session_id}: Final flush completed")

            # Wait for final segments to be forwarded to frontend (prevent race condition)
            await asyncio.sleep(0.5)

            # NOW set shutdown to stop the receiver (after final segments received)
            self.shutdown = True
            logger.info(f"Session {self.session_id}: Set shutdown=True, receiver will stop")

            # Send "complete" message to frontend
            try:
                await self.fastapi_websocket.send_json({
                    "type": "complete",
                    "message": "Transcription complete, all segments sent"
                })
                logger.info(f"Session {self.session_id}: ✅ Sent 'complete' message to frontend")
                await asyncio.sleep(0.2)
            except Exception as e:
                logger.error(f"Session {self.session_id}: Error sending complete message: {e}")

        except Exception as e:
            logger.error(f"Session {self.session_id}: Error during final flush - {str(e)}")

    async def close(self):
        """Gracefully close the transcription session and WebSocket connection."""
        logger.info(f"Session {self.session_id}: Closing session...")

        if not self.shutdown:
            self.shutdown = True

            # Trigger final flush if not already done
            try:
                if self.whisper_ws:
                    await self.whisper_ws.send(b"END_OF_AUDIO")
                    await asyncio.sleep(1.0)
            except Exception as e:
                logger.error(f"Session {self.session_id}: Error sending END_OF_AUDIO: {e}")

        # Stop receiver task
        if self._receiver_task:
            try:
                self._receiver_task.cancel()
                await asyncio.wait_for(self._receiver_task, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass

        # Close WebSocket connection to WhisperLive server
        if self.whisper_ws:
            try:
                await self.whisper_ws.close()
                logger.info(f"Session {self.session_id}: WebSocket closed")
            except Exception as e:
                logger.warning(f"Session {self.session_id}: Error closing WebSocket: {e}")

        self.is_connected = False
        logger.info(f"Session {self.session_id}: Session closed")

    def get_transcript(self) -> str:
        """
        Get the full transcript of completed segments.

        Returns:
            Complete transcript text
        """
        return "".join([seg.get("text", "") for seg in self.transcript_segments])


class LocalWhisperLiveService(TranscriptionService):
    """
    Service for managing local WhisperLive transcription sessions.
    Connects to WhisperLive server running on separate GPU system.
    """

    def __init__(
        self,
        server_url: Optional[str] = None,
        server_host: Optional[str] = None,
        server_port: int = 9090,
        use_ssl: bool = False,
        lang: str = "en",
        model: str = "large-v3",
        use_vad: bool = True,
        no_speech_thresh: float = 0.45,
        send_last_n_segments: int = 10,
        same_output_threshold: int = 3,
        clip_audio: bool = False,
        chunking_mode: str = "vad",
        chunk_interval: float = 2.0,
        enable_translation: bool = False,
        target_language: str = "en"
    ):
        """
        Initialize the local WhisperLive transcription service.

        Args:
            server_url: Complete WebSocket URL (e.g., "ws://192.168.1.100:9090" or "wss://example.com:443")
                       If provided, this takes precedence over server_host/server_port/use_ssl
            server_host: WhisperLive server host/IP (e.g., "192.168.1.100" or "localhost")
                        Used only if server_url is not provided
            server_port: WhisperLive server port (default: 9090)
                        Used only if server_url is not provided
            use_ssl: Use wss:// instead of ws:// (default: False)
                    Used only if server_url is not provided
            lang: Default language code (e.g., 'en', 'es', 'fr')
            model: Whisper model size (small, medium, large-v3)
            use_vad: Enable Voice Activity Detection
            no_speech_thresh: Silence detection threshold (0.0-1.0)
            send_last_n_segments: Number of recent segments to send for context
            same_output_threshold: Number of repeated outputs before finalizing
            clip_audio: Remove audio segments with no valid speech
            chunking_mode: Audio chunking strategy ("vad" or "time_based")
            chunk_interval: Interval for time-based chunking (seconds)
            enable_translation: Enable live translation
            target_language: Target language for translation
        """
        # Determine WebSocket URL
        if server_url:
            # Option 1: Use provided complete URL
            self.server_url = server_url.rstrip('/')
            logger.info(f"Using provided server URL: {self.server_url}")
        elif server_host:
            # Option 2: Build URL from components
            protocol = "wss" if use_ssl else "ws"
            self.server_url = f"{protocol}://{server_host}:{server_port}"
            logger.info(f"Built server URL from components: {self.server_url}")
        else:
            raise ValueError("Either server_url or server_host must be provided")

        self.server_host = server_host
        self.server_port = server_port
        self.use_ssl = use_ssl
        self.lang = lang
        self.model = model
        self.use_vad = use_vad
        self.no_speech_thresh = no_speech_thresh
        self.send_last_n_segments = send_last_n_segments
        self.same_output_threshold = same_output_threshold
        self.clip_audio = clip_audio
        self.chunking_mode = chunking_mode
        self.chunk_interval = chunk_interval
        self.enable_translation = enable_translation
        self.target_language = target_language

        # Track active sessions
        self.sessions: Dict[str, LocalWhisperLiveSession] = {}

        logger.info(f"Local WhisperLive service initialized (server={self.server_url}, model={model}, lang={lang})")

    def create_session(
        self,
        session_id: str,
        fastapi_websocket,
        callback: Callable[[Dict[str, Any]], None],
        lang: Optional[str] = None,
        model: Optional[str] = None
    ) -> LocalWhisperLiveSession:
        """
        Create a new transcription session.

        Args:
            session_id: Unique session identifier
            fastapi_websocket: FastAPI WebSocket instance
            callback: Function to call when transcription segments arrive
            lang: Override default language (optional)
            model: Override default model (optional)

        Returns:
            LocalWhisperLiveSession instance

        Raises:
            ValueError: If session_id already exists
        """
        if session_id in self.sessions:
            raise ValueError(f"Session {session_id} already exists")

        session = LocalWhisperLiveSession(
            session_id=session_id,
            fastapi_websocket=fastapi_websocket,
            server_url=self.server_url,
            lang=lang or self.lang,
            model=model or self.model,
            use_vad=self.use_vad,
            no_speech_thresh=self.no_speech_thresh,
            send_last_n_segments=self.send_last_n_segments,
            same_output_threshold=self.same_output_threshold,
            clip_audio=self.clip_audio,
            chunking_mode=self.chunking_mode,
            chunk_interval=self.chunk_interval,
            enable_translation=self.enable_translation,
            target_language=self.target_language,
            callback=callback
        )

        self.sessions[session_id] = session
        logger.info(f"Created local WhisperLive session: {session_id}")
        return session

    def get_session(self, session_id: str) -> Optional[LocalWhisperLiveSession]:
        """
        Get an existing transcription session.

        Args:
            session_id: Session identifier

        Returns:
            LocalWhisperLiveSession if found, None otherwise
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
_local_whisper_service: Optional[LocalWhisperLiveService] = None


def initialize_local_whisper_service(
    server_url: Optional[str] = None,
    server_host: Optional[str] = None,
    server_port: int = 9090,
    use_ssl: bool = False,
    lang: str = "en",
    model: str = "large-v3",
    use_vad: bool = True,
    no_speech_thresh: float = 0.45,
    send_last_n_segments: int = 10,
    same_output_threshold: int = 3,
    clip_audio: bool = False,
    chunking_mode: str = "vad",
    chunk_interval: float = 2.0,
    enable_translation: bool = False,
    target_language: str = "en",
    transcription_logging_enabled: bool = True
) -> LocalWhisperLiveService:
    """
    Initialize the global local WhisperLive transcription service instance.

    Args:
        server_url: Complete WebSocket URL (e.g., "ws://192.168.1.100:9090")
                   If provided, takes precedence over server_host/server_port/use_ssl
        server_host: WhisperLive server host/IP (used if server_url not provided)
        server_port: WhisperLive server port (used if server_url not provided)
        use_ssl: Use wss:// instead of ws:// (used if server_url not provided)
        lang: Default language code
        model: Whisper model size
        use_vad: Enable Voice Activity Detection
        no_speech_thresh: Silence detection threshold
        send_last_n_segments: Number of recent segments to send for context
        same_output_threshold: Number of repeated outputs before finalizing
        clip_audio: Remove audio segments with no valid speech
        chunking_mode: Audio chunking strategy ("vad" or "time_based")
        chunk_interval: Interval for time-based chunking (seconds)
        enable_translation: Enable live translation
        target_language: Target language for translation
        transcription_logging_enabled: Enable/disable verbose transcription logs

    Returns:
        LocalWhisperLiveService instance
    """
    global _local_whisper_service, _transcription_logging_enabled

    # Set the module-level logging flag
    _transcription_logging_enabled = transcription_logging_enabled

    if transcription_logging_enabled:
        logger.info("Transcription logging is ENABLED")
    else:
        logger.info("Transcription logging is DISABLED")

    _local_whisper_service = LocalWhisperLiveService(
        server_url=server_url,
        server_host=server_host,
        server_port=server_port,
        use_ssl=use_ssl,
        lang=lang,
        model=model,
        use_vad=use_vad,
        no_speech_thresh=no_speech_thresh,
        send_last_n_segments=send_last_n_segments,
        same_output_threshold=same_output_threshold,
        clip_audio=clip_audio,
        chunking_mode=chunking_mode,
        chunk_interval=chunk_interval,
        enable_translation=enable_translation,
        target_language=target_language
    )
    return _local_whisper_service


def get_local_whisper_service() -> LocalWhisperLiveService:
    """
    Get the global local WhisperLive transcription service instance.

    Returns:
        LocalWhisperLiveService instance

    Raises:
        RuntimeError: If service not initialized
    """
    if _local_whisper_service is None:
        raise RuntimeError("Local WhisperLive service not initialized. Call initialize_local_whisper_service() first.")
    return _local_whisper_service


async def cleanup_local_whisper_service():
    """Cleanup the global local WhisperLive transcription service instance."""
    global _local_whisper_service
    if _local_whisper_service:
        await _local_whisper_service.close_all()
        _local_whisper_service = None
        logger.info("Local WhisperLive service cleaned up")
