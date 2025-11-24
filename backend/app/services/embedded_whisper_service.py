"""
Embedded Whisper Transcription Service
Direct in-process transcription using faster-whisper with CTranslate2.

Eliminates network latency by running transcription in the same process as the FastAPI backend.
Each session gets its own model instance for parallel processing.
"""

import os
import logging
import threading
import asyncio
import time
import numpy as np
import torch
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass

# Lazy imports for faster startup
_faster_whisper_model = None
_vad_model = None

logger = logging.getLogger(__name__)

# Constants
SAMPLE_RATE = 16000
MAX_BUFFER_SECONDS = 45  # Rolling buffer limit
TRIM_SECONDS = 30  # Amount to trim when buffer exceeds limit


@dataclass
class TranscriptionSegment:
    """Represents a transcription segment with timing information."""
    text: str
    start: float
    end: float
    is_final: bool


class EmbeddedWhisperSession:
    """
    A transcription session that processes audio in real-time using faster-whisper.

    Each session maintains:
    - Its own model instance (for parallel processing)
    - Audio buffer with rolling window
    - Background transcription thread
    - Callback mechanism for streaming results
    """

    def __init__(
        self,
        session_id: str,
        callback: Callable[[Dict[str, Any]], None],
        model_name: str = "distil-large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        language: str = "en",
        use_vad: bool = True,
        no_speech_thresh: float = 0.45,
        chunk_interval: float = 1.0,
        same_output_threshold: int = 5,
        send_last_n_segments: int = 10,
    ):
        """
        Initialize an embedded whisper transcription session.

        Args:
            session_id: Unique identifier for this session
            callback: Function to call with transcription results
            model_name: Whisper model to use (default: distil-large-v3)
            device: Device to run inference on (cuda/cpu)
            compute_type: Compute precision (float16/float32/int8)
            language: Language code for transcription
            use_vad: Whether to use Voice Activity Detection
            no_speech_thresh: Threshold for filtering silent segments
            chunk_interval: Minimum seconds of audio before transcription
            same_output_threshold: Repeated outputs before finalizing segment
            send_last_n_segments: Number of recent segments to track
        """
        self.session_id = session_id
        self.callback = callback
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.use_vad = use_vad
        self.no_speech_thresh = no_speech_thresh
        self.chunk_interval = chunk_interval
        self.same_output_threshold = same_output_threshold
        self.send_last_n_segments = send_last_n_segments

        # State
        self.transcriber = None
        self.is_connected = False
        self.exit = False
        self.shutdown = False

        # Audio buffer
        self.frames_np: Optional[np.ndarray] = None
        self.frames_offset = 0.0
        self.timestamp_offset = 0.0
        self.lock = threading.Lock()

        # Transcription state
        self.transcript: List[Dict[str, Any]] = []
        self.current_out = ""
        self.prev_out = ""
        self.same_output_count = 0
        self.last_transcription_time: Optional[float] = None
        self.last_interim_segment: Optional[Dict[str, Any]] = None  # Track pending interim for final flush

        # Threading
        self.trans_thread: Optional[threading.Thread] = None

        # For final flush synchronization
        self.flush_complete_event = asyncio.Event()

        # Store reference to main event loop for thread-safe callbacks
        self.main_loop: Optional[asyncio.AbstractEventLoop] = None

        logger.info(f"[{session_id}] EmbeddedWhisperSession initialized")

    async def connect(self) -> bool:
        """
        Initialize the model and start the transcription thread.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Store reference to main event loop for thread-safe callbacks
            self.main_loop = asyncio.get_running_loop()

            logger.info(f"[{self.session_id}] Loading Whisper model: {self.model_name}")

            # Import faster_whisper here to avoid slow startup
            from faster_whisper import WhisperModel

            # Determine compute type based on device
            if self.device == "cuda" and torch.cuda.is_available():
                device = "cuda"
                # Check GPU compute capability
                major, _ = torch.cuda.get_device_capability(0)
                compute_type = "float16" if major >= 7 else "float32"
                logger.info(f"[{self.session_id}] Using CUDA with {compute_type}")
            else:
                device = "cpu"
                compute_type = "int8"
                logger.info(f"[{self.session_id}] Using CPU with int8 quantization")

            # Load the model
            self.transcriber = WhisperModel(
                self.model_name,
                device=device,
                compute_type=compute_type,
                local_files_only=False,
            )

            logger.info(f"[{self.session_id}] Model loaded successfully")

            # Start transcription thread
            self.trans_thread = threading.Thread(
                target=self._speech_to_text_loop,
                daemon=True
            )
            self.trans_thread.start()

            self.is_connected = True
            return True

        except Exception as e:
            logger.error(f"[{self.session_id}] Failed to load model: {e}", exc_info=True)
            return False

    async def send_audio(self, audio_data: bytes):
        """
        Add audio data to the buffer for transcription.

        Args:
            audio_data: Raw audio bytes (int16 PCM, 16kHz, mono)
        """
        if not self.is_connected or self.exit:
            return

        try:
            # Convert int16 PCM to float32 normalized [-1.0, 1.0]
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            self._add_frames(audio_array)
        except Exception as e:
            logger.error(f"[{self.session_id}] Error processing audio: {e}")

    def _add_frames(self, frame_np: np.ndarray):
        """Add audio frames to the rolling buffer."""
        with self.lock:
            # Trim buffer if it exceeds maximum
            if self.frames_np is not None and self.frames_np.shape[0] > MAX_BUFFER_SECONDS * SAMPLE_RATE:
                self.frames_offset += TRIM_SECONDS
                self.frames_np = self.frames_np[int(TRIM_SECONDS * SAMPLE_RATE):]
                # Ensure timestamp offset is at least frames_offset
                if self.timestamp_offset < self.frames_offset:
                    self.timestamp_offset = self.frames_offset

            # Initialize or append to buffer
            if self.frames_np is None:
                self.frames_np = frame_np.copy()
            else:
                self.frames_np = np.concatenate((self.frames_np, frame_np), axis=0)

    def _get_audio_chunk(self) -> tuple[np.ndarray, float]:
        """Get the current audio chunk for processing."""
        with self.lock:
            if self.frames_np is None:
                return np.array([], dtype=np.float32), 0.0

            samples_to_skip = max(0, int((self.timestamp_offset - self.frames_offset) * SAMPLE_RATE))
            chunk = self.frames_np[samples_to_skip:].copy()

        duration = chunk.shape[0] / SAMPLE_RATE if chunk.shape[0] > 0 else 0.0
        return chunk, duration

    def _speech_to_text_loop(self):
        """Background thread that continuously processes audio and generates transcriptions."""
        logger.info(f"[{self.session_id}] Transcription thread started")

        while not self.exit:
            try:
                # Check if we have audio to process
                if self.frames_np is None:
                    time.sleep(0.1)
                    continue

                # Get audio chunk
                audio_chunk, duration = self._get_audio_chunk()

                # Wait for minimum audio duration
                if duration < self.chunk_interval:
                    time.sleep(0.1)
                    continue

                # Transcribe
                result = self._transcribe_chunk(audio_chunk)

                if result is None:
                    # No speech detected, update offset
                    with self.lock:
                        self.timestamp_offset += duration
                    time.sleep(0.1)
                    continue

                # Process transcription output
                self._handle_transcription_output(result, duration)

            except Exception as e:
                logger.error(f"[{self.session_id}] Transcription error: {e}", exc_info=True)
                time.sleep(0.1)

        logger.info(f"[{self.session_id}] Transcription thread exiting")

        # Signal flush complete
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.call_soon_threadsafe(self.flush_complete_event.set)
        except Exception:
            pass

    def _transcribe_chunk(self, audio_chunk: np.ndarray):
        """Transcribe an audio chunk using faster-whisper."""
        if self.transcriber is None:
            return None

        try:
            segments, info = self.transcriber.transcribe(
                audio_chunk,
                language=self.language,
                task="transcribe",
                vad_filter=self.use_vad,
                vad_parameters={"threshold": 0.5} if self.use_vad else None,
            )

            # Convert generator to list
            segment_list = list(segments)

            if not segment_list:
                return None

            return segment_list

        except Exception as e:
            logger.error(f"[{self.session_id}] Transcription failed: {e}")
            return None

    def _handle_transcription_output(self, segments: list, duration: float):
        """Process transcription segments and send to callback."""
        if not segments:
            return

        offset = None
        self.current_out = ""
        last_segment_data = None

        # Process complete segments (all except last)
        if len(segments) > 1:
            last_seg = segments[-1]
            if getattr(last_seg, 'no_speech_prob', 0) <= self.no_speech_thresh:
                for seg in segments[:-1]:
                    if getattr(seg, 'no_speech_prob', 0) > self.no_speech_thresh:
                        continue

                    start = self.timestamp_offset + seg.start
                    end = self.timestamp_offset + min(duration, seg.end)

                    if start >= end:
                        continue

                    # Create completed segment
                    completed_segment = {
                        'start': f"{start:.3f}",
                        'end': f"{end:.3f}",
                        'text': seg.text,
                        'completed': True
                    }
                    self.transcript.append(completed_segment)

                    # Send to callback
                    self._send_segment(completed_segment)
                    self.last_interim_segment = None  # Clear interim when completed segment sent

                    offset = min(duration, seg.end)

        # Process last segment (incomplete)
        last_seg = segments[-1]
        if getattr(last_seg, 'no_speech_prob', 0) <= self.no_speech_thresh:
            self.current_out = last_seg.text

            start = self.timestamp_offset + last_seg.start
            end = self.timestamp_offset + min(duration, last_seg.end)

            last_segment_data = {
                'start': f"{start:.3f}",
                'end': f"{end:.3f}",
                'text': self.current_out,
                'completed': False
            }

        # Handle repeated output (finalize if same text repeated multiple times)
        if self.current_out.strip() == self.prev_out.strip() and self.current_out:
            self.same_output_count += 1
            time.sleep(0.1)
        else:
            self.same_output_count = 0

        # Finalize if repeated too many times
        if self.same_output_count > self.same_output_threshold:
            if last_segment_data:
                last_segment_data['completed'] = True
                self.transcript.append(last_segment_data)
                self._send_segment(last_segment_data)
                last_segment_data = None
                self.last_interim_segment = None  # Clear tracked interim

            self.current_out = ""
            offset = duration
            self.same_output_count = 0
        else:
            self.prev_out = self.current_out

        # Send incomplete segment and track it for final flush
        if last_segment_data:
            self.last_interim_segment = last_segment_data.copy()  # Track for final flush
            self._send_segment(last_segment_data)

        # Update timestamp offset
        if offset is not None:
            with self.lock:
                self.timestamp_offset += offset

    def _send_segment(self, segment: Dict[str, Any]):
        """Send a transcription segment via the callback."""
        try:
            data = {
                "type": "transcript",
                "text": segment.get('text', ''),
                "start": float(segment.get('start', 0)),
                "end": float(segment.get('end', 0)),
                "is_final": segment.get('completed', False)
            }

            # Check if callback is a coroutine function
            result = self.callback(data)
            if asyncio.iscoroutine(result):
                # Schedule coroutine on the main event loop from this thread
                if self.main_loop and self.main_loop.is_running():
                    asyncio.run_coroutine_threadsafe(result, self.main_loop)
                else:
                    # Fallback: close the unawaited coroutine
                    result.close()
        except Exception as e:
            logger.error(f"[{self.session_id}] Callback error: {e}")

    async def trigger_final_flush(self):
        """
        Process any remaining audio in the buffer before stopping.
        Called when user stops recording.
        """
        logger.info(f"[{self.session_id}] Triggering final flush")

        # Signal transcription thread to exit
        self.exit = True

        # Wait for transcription thread to finish (with timeout)
        try:
            await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.trans_thread.join(timeout=3.0) if self.trans_thread else None
                ),
                timeout=5.0
            )
        except asyncio.TimeoutError:
            logger.warning(f"[{self.session_id}] Final flush timed out")

        # Finalize any pending interim segment before sending complete
        if self.last_interim_segment and self.last_interim_segment.get('text'):
            logger.info(f"[{self.session_id}] Finalizing pending interim segment: {self.last_interim_segment.get('text', '')[:50]}...")
            final_segment = {
                **self.last_interim_segment,
                'completed': True
            }
            self.transcript.append(final_segment)
            try:
                result = self.callback({
                    "type": "transcript",
                    "text": final_segment.get('text', ''),
                    "start": float(final_segment.get('start', 0)),
                    "end": float(final_segment.get('end', 0)),
                    "is_final": True
                })
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"[{self.session_id}] Error sending final interim segment: {e}")
            self.last_interim_segment = None

        # Send completion signal
        try:
            result = self.callback({
                "type": "complete",
                "message": "Transcription complete"
            })
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            logger.error(f"[{self.session_id}] Error sending completion: {e}")

        self.shutdown = True
        logger.info(f"[{self.session_id}] Final flush complete")

    async def close(self):
        """Close the session and release resources."""
        logger.info(f"[{self.session_id}] Closing session")

        self.exit = True
        self.is_connected = False

        # Wait for transcription thread
        if self.trans_thread and self.trans_thread.is_alive():
            self.trans_thread.join(timeout=2.0)

        # Release model and clear GPU memory
        if self.transcriber is not None:
            del self.transcriber
            self.transcriber = None

            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Clear audio buffer
        self.frames_np = None

        logger.info(f"[{self.session_id}] Session closed")

    def get_transcript(self) -> str:
        """Get the accumulated transcript text."""
        return " ".join(seg.get('text', '') for seg in self.transcript if seg.get('completed', False))


class EmbeddedWhisperService:
    """
    Service for managing embedded whisper transcription sessions.

    Provides the same interface as the previous TranscriptionService implementations
    but runs transcription in-process without network overhead.
    """

    def __init__(
        self,
        model_name: str = "distil-large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        language: str = "en",
        use_vad: bool = True,
        no_speech_thresh: float = 0.45,
        chunk_interval: float = 1.0,
        same_output_threshold: int = 5,
    ):
        """
        Initialize the embedded whisper transcription service.

        Args:
            model_name: Whisper model to use
            device: Device for inference (cuda/cpu)
            compute_type: Compute precision
            language: Default language for transcription
            use_vad: Enable Voice Activity Detection
            no_speech_thresh: Silence filtering threshold
            chunk_interval: Minimum audio duration before transcription
            same_output_threshold: Repeated outputs before finalizing
        """
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.use_vad = use_vad
        self.no_speech_thresh = no_speech_thresh
        self.chunk_interval = chunk_interval
        self.same_output_threshold = same_output_threshold

        self.sessions: Dict[str, EmbeddedWhisperSession] = {}

        logger.info(f"EmbeddedWhisperService initialized with model={model_name}, device={device}")

    def create_session(
        self,
        session_id: str,
        fastapi_websocket,  # Not used but kept for interface compatibility
        callback: Callable[[Dict[str, Any]], None],
        lang: Optional[str] = None,
        model: Optional[str] = None,
    ) -> EmbeddedWhisperSession:
        """
        Create a new transcription session.

        Args:
            session_id: Unique identifier for the session
            fastapi_websocket: WebSocket instance (kept for compatibility)
            callback: Function to call with transcription results
            lang: Override language for this session
            model: Override model for this session

        Returns:
            EmbeddedWhisperSession instance

        Raises:
            ValueError: If session_id already exists
        """
        if session_id in self.sessions:
            raise ValueError(f"Session {session_id} already exists")

        session = EmbeddedWhisperSession(
            session_id=session_id,
            callback=callback,
            model_name=model or self.model_name,
            device=self.device,
            compute_type=self.compute_type,
            language=lang or self.language,
            use_vad=self.use_vad,
            no_speech_thresh=self.no_speech_thresh,
            chunk_interval=self.chunk_interval,
            same_output_threshold=self.same_output_threshold,
        )

        self.sessions[session_id] = session
        logger.info(f"Created transcription session: {session_id}")

        return session

    def get_session(self, session_id: str) -> Optional[EmbeddedWhisperSession]:
        """Get an existing session by ID."""
        return self.sessions.get(session_id)

    async def close_session(self, session_id: str):
        """Close and remove a session."""
        session = self.sessions.pop(session_id, None)
        if session:
            await session.close()
            logger.info(f"Closed and removed session: {session_id}")

    async def close_all(self):
        """Close all active sessions."""
        session_ids = list(self.sessions.keys())
        for session_id in session_ids:
            await self.close_session(session_id)
        logger.info("All transcription sessions closed")


# Global service instance
_embedded_whisper_service: Optional[EmbeddedWhisperService] = None


def initialize_embedded_whisper_service(
    model_name: str = "distil-large-v3",
    device: str = "cuda",
    compute_type: str = "float16",
    language: str = "en",
    use_vad: bool = True,
    no_speech_thresh: float = 0.45,
    chunk_interval: float = 1.0,
    same_output_threshold: int = 5,
) -> EmbeddedWhisperService:
    """
    Initialize the global embedded whisper transcription service.

    Returns:
        EmbeddedWhisperService instance
    """
    global _embedded_whisper_service

    _embedded_whisper_service = EmbeddedWhisperService(
        model_name=model_name,
        device=device,
        compute_type=compute_type,
        language=language,
        use_vad=use_vad,
        no_speech_thresh=no_speech_thresh,
        chunk_interval=chunk_interval,
        same_output_threshold=same_output_threshold,
    )

    return _embedded_whisper_service


def get_embedded_whisper_service() -> EmbeddedWhisperService:
    """
    Get the global embedded whisper transcription service.

    Returns:
        EmbeddedWhisperService instance

    Raises:
        RuntimeError: If service not initialized
    """
    if _embedded_whisper_service is None:
        raise RuntimeError("Embedded whisper service not initialized. Call initialize_embedded_whisper_service() first.")
    return _embedded_whisper_service


def cleanup_embedded_whisper_service():
    """Cleanup the global service on shutdown."""
    global _embedded_whisper_service
    if _embedded_whisper_service is not None:
        # Note: This should be called from an async context
        # For sync cleanup, sessions should be closed individually
        _embedded_whisper_service = None
        logger.info("Embedded whisper service cleaned up")
