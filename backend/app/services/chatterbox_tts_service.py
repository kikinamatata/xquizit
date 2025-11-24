"""
Chatterbox TTS Service Module
Direct integration of Chatterbox TTS for real-time audio generation with voice cloning.

Uses the chatterbox-tts pip package directly without RealtimeTTS wrapper.

Features:
- Zero-shot voice cloning from reference audio
- CUDA/CPU device selection with auto-detection
- Emotion control via exaggeration parameter
- Chunked streaming for low-latency audio delivery
- Timeout and retry logic for reliability
- Audio post-processing (fade, silence trimming)
- Voice embedding caching for performance
"""

import logging
import base64
import hashlib
import io
import time
import wave
import threading
from pathlib import Path
from typing import Optional, AsyncIterator, Dict
import asyncio

import numpy as np

from .base_tts_service import BaseTTSService

logger = logging.getLogger(__name__)


class ChatterboxTTSService(BaseTTSService):
    """
    Direct Chatterbox TTS integration for streaming text-to-speech with voice cloning.

    Uses chatterbox-tts pip package directly for synthesis.
    """

    # Audio format constants
    SAMPLE_RATE = 24000  # Chatterbox outputs 24kHz audio
    CHANNELS = 1  # Mono
    SAMPLE_WIDTH = 2  # 16-bit (2 bytes per sample)

    def __init__(
        self,
        reference_voice_path: Optional[str] = None,
        device: str = "auto",
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        chunk_duration_ms: int = 100,
        # Generation parameters
        temperature: float = 0.8,
        top_p: float = 1.0,
        repetition_penalty: float = 1.2,
        # Reliability parameters
        synthesis_timeout: int = 30,
        retry_attempts: int = 3,
        # Audio post-processing
        audio_postprocess: bool = True,
        fade_duration_ms: int = 10,
        trim_silence: bool = True,
        silence_threshold: float = 0.01,
        # Model loading
        preload: bool = True
    ):
        """
        Initialize the Chatterbox TTS service.

        Args:
            reference_voice_path: Path to reference audio file for voice cloning
            device: Device to run model on ('cuda', 'cpu', or 'auto')
            exaggeration: Emotion exaggeration level (0.0-1.0)
            cfg_weight: Classifier-free guidance weight
            chunk_duration_ms: Duration of each audio chunk in milliseconds
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling probability threshold
            repetition_penalty: Penalty for repeated tokens
            synthesis_timeout: Max seconds to wait for synthesis
            retry_attempts: Number of retry attempts on failure
            audio_postprocess: Enable audio post-processing
            fade_duration_ms: Fade in/out duration in milliseconds
            trim_silence: Enable silence trimming
            silence_threshold: Threshold for silence detection
            preload: If True, load model immediately
        """
        # Core parameters
        self.reference_voice_path = reference_voice_path
        self.device = device
        self.exaggeration = max(0.0, min(1.0, exaggeration))
        self.cfg_weight = cfg_weight
        self.chunk_duration_ms = chunk_duration_ms

        # Generation parameters
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty

        # Reliability parameters
        self.synthesis_timeout = synthesis_timeout
        self.retry_attempts = max(1, retry_attempts)
        self.retry_delays = [1.0, 2.0, 4.0]  # Exponential backoff

        # Audio post-processing
        self.audio_postprocess = audio_postprocess
        self.fade_duration_ms = fade_duration_ms
        self.trim_silence = trim_silence
        self.silence_threshold = silence_threshold

        # Model state
        self.model = None
        self._model_loaded = False
        self._synthesis_in_progress = False
        self._stop_event = threading.Event()

        # Voice embedding cache
        self._voice_cache: Dict[str, bool] = {}
        self._current_voice_hash: Optional[str] = None

        if preload:
            self._load_model()

    def _get_voice_hash(self, voice_path: str) -> str:
        """Generate a hash for voice path for caching."""
        return hashlib.md5(voice_path.encode()).hexdigest()[:16]

    def _load_model(self):
        """Load the Chatterbox TTS model."""
        if self._model_loaded:
            return

        try:
            from chatterbox.tts import ChatterboxTTS
            import torch

            logger.info("Loading Chatterbox TTS model...")
            load_start = time.time()

            # Determine device
            if self.device == "auto":
                actual_device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                actual_device = self.device
                if actual_device == "cuda" and not torch.cuda.is_available():
                    logger.warning("CUDA requested but not available, falling back to CPU")
                    actual_device = "cpu"

            self._actual_device = actual_device
            logger.info(f"Using device: {actual_device}")

            # Enable CUDA optimizations
            if actual_device == "cuda":
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.enabled = True
                logger.info("CUDA optimizations enabled (cuDNN benchmark mode)")

            # Load the model
            logger.info("Loading Chatterbox model (this may take 30-60 seconds on first run)...")
            self.model = ChatterboxTTS.from_pretrained(device=actual_device)

            load_duration = time.time() - load_start
            self._model_loaded = True

            logger.info(f"Chatterbox TTS model loaded successfully in {load_duration:.1f}s")
            logger.info(f"  - Device: {actual_device}")
            logger.info(f"  - Sample rate: {self.SAMPLE_RATE} Hz")
            logger.info(f"  - Chunk duration: {self.chunk_duration_ms}ms")
            logger.info(f"  - Exaggeration: {self.exaggeration}")
            logger.info(f"  - Audio post-processing: {self.audio_postprocess}")
            if self.reference_voice_path:
                logger.info(f"  - Voice: {self.reference_voice_path}")

        except ImportError as e:
            logger.error("chatterbox-tts package not installed")
            logger.error(f"Error: {str(e)}")
            logger.error("Please install with: pip install chatterbox-tts")
            raise ImportError(
                "chatterbox-tts is not installed. "
                "Please install with: pip install chatterbox-tts"
            ) from e
        except Exception as e:
            logger.error(f"Failed to load Chatterbox TTS model: {str(e)}")
            raise

    def set_voice(self, voice_path: str):
        """
        Set the voice for synthesis from a reference audio file.

        Args:
            voice_path: Path to reference audio file (WAV recommended, 6-10 seconds)
        """
        # Validate path exists
        if not Path(voice_path).exists():
            raise ValueError(f"Reference voice file not found: {voice_path}")

        voice_hash = self._get_voice_hash(voice_path)

        # Check if voice is already set (cached)
        if self._current_voice_hash == voice_hash:
            logger.debug(f"Voice already set (cached): {voice_path}")
            return

        self.reference_voice_path = voice_path
        self._current_voice_hash = voice_hash
        self._voice_cache[voice_hash] = True

        logger.info(f"Voice set to: {voice_path}")

    def set_voice_parameters(
        self,
        exaggeration: Optional[float] = None,
        cfg_weight: Optional[float] = None,
        chunk_duration_ms: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None
    ):
        """Update voice synthesis parameters."""
        if exaggeration is not None:
            self.exaggeration = max(0.0, min(1.0, exaggeration))
        if cfg_weight is not None:
            self.cfg_weight = cfg_weight
        if chunk_duration_ms is not None:
            self.chunk_duration_ms = chunk_duration_ms
        if temperature is not None:
            self.temperature = temperature
        if top_p is not None:
            self.top_p = top_p
        if repetition_penalty is not None:
            self.repetition_penalty = repetition_penalty

    def stop(self):
        """Stop any ongoing synthesis."""
        self._stop_event.set()
        logger.info("Synthesis stop requested")

    def _check_stopped(self) -> bool:
        """Check if stop was requested."""
        return self._stop_event.is_set()

    async def _synthesize_with_timeout(self, text: str) -> np.ndarray:
        """
        Run synthesis with timeout protection.

        Returns:
            Audio as int16 numpy array

        Raises:
            asyncio.TimeoutError: If synthesis exceeds timeout
        """
        import torch

        loop = asyncio.get_event_loop()

        def _generate():
            with torch.inference_mode():
                audio_tensor = self.model.generate(
                    text=text,
                    audio_prompt_path=self.reference_voice_path,
                    exaggeration=self.exaggeration,
                    cfg_weight=self.cfg_weight
                )
            # Convert to int16 numpy array
            audio_np = audio_tensor.cpu().numpy()
            audio_int16 = (audio_np * 32767).astype(np.int16)
            return audio_int16

        try:
            audio = await asyncio.wait_for(
                loop.run_in_executor(None, _generate),
                timeout=self.synthesis_timeout
            )
            return audio
        except asyncio.TimeoutError:
            logger.error(f"Synthesis timeout after {self.synthesis_timeout}s")
            self.stop()
            raise

    async def _synthesize_with_retry(self, text: str) -> np.ndarray:
        """
        Run synthesis with retry logic.

        Returns:
            Audio as int16 numpy array

        Raises:
            Exception: If all retry attempts fail
        """
        last_error = None

        for attempt in range(self.retry_attempts):
            try:
                self._stop_event.clear()
                audio = await self._synthesize_with_timeout(text)
                return audio
            except asyncio.TimeoutError as e:
                last_error = e
                logger.warning(f"Synthesis timeout (attempt {attempt + 1}/{self.retry_attempts})")
            except Exception as e:
                last_error = e
                logger.warning(f"Synthesis error (attempt {attempt + 1}/{self.retry_attempts}): {e}")

            # Wait before retry (if not last attempt)
            if attempt < self.retry_attempts - 1:
                delay = self.retry_delays[min(attempt, len(self.retry_delays) - 1)]
                logger.info(f"Retrying in {delay}s...")
                await asyncio.sleep(delay)

        # All attempts failed
        if last_error:
            raise last_error
        raise RuntimeError("Synthesis failed after all retry attempts")

    def _apply_fade_in(self, audio: np.ndarray) -> np.ndarray:
        """Apply linear fade-in to audio."""
        fade_samples = int(self.SAMPLE_RATE * self.fade_duration_ms / 1000)
        if fade_samples == 0 or len(audio) < fade_samples:
            return audio

        audio = audio.copy()
        fade_in = np.linspace(0.0, 1.0, fade_samples)
        audio[:fade_samples] = (audio[:fade_samples] * fade_in).astype(audio.dtype)
        return audio

    def _apply_fade_out(self, audio: np.ndarray) -> np.ndarray:
        """Apply linear fade-out to audio."""
        fade_samples = int(self.SAMPLE_RATE * self.fade_duration_ms / 1000)
        if fade_samples == 0 or len(audio) < fade_samples:
            return audio

        audio = audio.copy()
        fade_out = np.linspace(1.0, 0.0, fade_samples)
        audio[-fade_samples:] = (audio[-fade_samples:] * fade_out).astype(audio.dtype)
        return audio

    def _trim_silence_start(self, audio: np.ndarray) -> np.ndarray:
        """Trim leading silence from audio."""
        threshold = self.silence_threshold * 32767
        non_silent = np.where(np.abs(audio) > threshold)[0]
        if len(non_silent) > 0:
            start_index = non_silent[0]
            if start_index > 0:
                return audio[start_index:]
        return audio

    def _trim_silence_end(self, audio: np.ndarray) -> np.ndarray:
        """Trim trailing silence from audio."""
        threshold = self.silence_threshold * 32767
        non_silent = np.where(np.abs(audio) > threshold)[0]
        if len(non_silent) > 0:
            end_index = non_silent[-1] + 1
            if end_index < len(audio):
                return audio[:end_index]
        return audio

    def _postprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply audio post-processing (fade, silence trimming).

        Args:
            audio: Full audio as int16 numpy array

        Returns:
            Processed audio
        """
        if not self.audio_postprocess:
            return audio

        # Trim silence
        if self.trim_silence:
            audio = self._trim_silence_start(audio)
            audio = self._trim_silence_end(audio)

        # Apply fades
        audio = self._apply_fade_in(audio)
        audio = self._apply_fade_out(audio)

        return audio

    def _chunk_audio(self, audio: np.ndarray) -> list:
        """
        Split audio into chunks for streaming.

        Args:
            audio: Full audio as int16 numpy array

        Returns:
            List of audio chunks (bytes)
        """
        chunk_size = int(self.SAMPLE_RATE * self.chunk_duration_ms / 1000)
        chunks = []

        for i in range(0, len(audio), chunk_size):
            if self._check_stopped():
                break
            chunk = audio[i:i + chunk_size]
            chunks.append(chunk.tobytes())

        return chunks

    def _pcm_to_wav(self, pcm_bytes: bytes) -> bytes:
        """Convert raw PCM bytes to WAV format."""
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(self.CHANNELS)
            wav_file.setsampwidth(self.SAMPLE_WIDTH)
            wav_file.setframerate(self.SAMPLE_RATE)
            wav_file.writeframes(pcm_bytes)
        return buffer.getvalue()

    async def generate_stream(
        self,
        text: str,
        audio_index: int = 0,
        skip_postprocess: bool = False,
        use_token_streaming: bool = True,
        token_chunk_size: int = 50,
        first_chunk_only_streaming: bool = False,
        apply_chunk_fades: bool = True
    ) -> AsyncIterator[dict]:
        """
        Generate audio from text using Chatterbox TTS with token-level streaming.

        Args:
            text: Text to convert to speech
            audio_index: Index number for this audio segment (for ordering)
            skip_postprocess: Skip silence trimming for faster streaming (fade still applied)
            use_token_streaming: Use true token-level streaming (recommended for low latency)
            token_chunk_size: Number of tokens per audio chunk (lower = lower latency, default 50)
            first_chunk_only_streaming: If True, only use token streaming for audio_index=0,
                                        subsequent chunks use batch synthesis (reduces overhead)
            apply_chunk_fades: Apply fade-in/fade-out to streamed chunks for smoother transitions

        Yields:
            Dictionary with 'type', 'audio' (base64 WAV), 'index', and 'chunk_index' keys
        """
        # Lazy load if not preloaded
        if not self._model_loaded:
            self._load_model()

        if not self.model:
            raise RuntimeError("TTS model not initialized")

        if not text or not text.strip():
            logger.warning("Empty text provided to TTS")
            return

        # Prevent concurrent synthesis
        if self._synthesis_in_progress:
            logger.warning("Synthesis already in progress, waiting...")
            while self._synthesis_in_progress:
                await asyncio.sleep(0.1)

        self._synthesis_in_progress = True
        self._stop_event.clear()

        try:
            text_preview = text[:50] + ('...' if len(text) > 50 else '')

            # Determine if we should use token streaming for this chunk
            # If first_chunk_only_streaming is enabled, only stream for audio_index=0
            should_stream = use_token_streaming
            if first_chunk_only_streaming:
                should_stream = (audio_index == 0)

            logger.info(f"Generating audio for: '{text_preview}' (index: {audio_index}, streaming: {should_stream}, first_only: {first_chunk_only_streaming})")

            start_time = time.time()

            if should_stream:
                # Use true token-level streaming for low latency
                async for chunk_data in self._generate_stream_realtime(
                    text, audio_index, token_chunk_size, apply_chunk_fades
                ):
                    yield chunk_data
            else:
                # Fall back to batch synthesis + post-hoc chunking
                synthesis_start = time.time()
                audio = await self._synthesize_with_retry(text)
                synthesis_duration = time.time() - synthesis_start

                if skip_postprocess:
                    audio = self._apply_fade_in(audio)
                    audio = self._apply_fade_out(audio)
                else:
                    audio = self._postprocess_audio(audio)

                chunk_count = 0
                async for chunk_data in self._stream_audio_chunks(audio, audio_index):
                    chunk_count += 1
                    yield chunk_data

                total_duration = time.time() - start_time
                try:
                    from app.utils.timing import timing_logger
                    timing_logger.info(
                        f"TTS - Chatterbox Batch Generation: {total_duration:.3f}s | "
                        f"{{\"text_length\": {len(text)}, \"synthesis_duration\": {synthesis_duration:.3f}, "
                        f"\"chunks\": {chunk_count}, \"index\": {audio_index}}}"
                    )
                except ImportError:
                    pass

        except asyncio.TimeoutError:
            logger.error(f"Synthesis timeout for text: '{text[:50]}...'")
            raise
        except Exception as e:
            logger.error(f"Error generating audio: {str(e)}", exc_info=True)
            raise
        finally:
            self._synthesis_in_progress = False

    async def _generate_stream_realtime(
        self,
        text: str,
        audio_index: int,
        token_chunk_size: int = 50,
        apply_chunk_fades: bool = True
    ) -> AsyncIterator[dict]:
        """
        True token-level streaming using Chatterbox's generate_stream API.
        Uses context window approach for smooth audio transitions.

        Args:
            apply_chunk_fades: Apply fade-in/fade-out to chunks for smoother transitions
        """
        import torch

        loop = asyncio.get_event_loop()
        start_time = time.time()
        chunk_count = 0
        first_chunk_latency = None

        def _run_streaming_generator():
            """Run the synchronous generator in executor."""
            return list(self.model.generate_stream(
                text=text,
                chunk_size=token_chunk_size,
                context_window=50,  # Use context window for smooth transitions
                fade_duration=0.02,  # 20ms fade-in for chunk boundaries
                audio_prompt_path=self.reference_voice_path,
                exaggeration=self.exaggeration,
                cfg_weight=self.cfg_weight,
                temperature=getattr(self, 'temperature', 0.8),
                repetition_penalty=getattr(self, 'repetition_penalty', 1.2),
            ))

        try:
            # Run streaming generation in executor to not block event loop
            results = await loop.run_in_executor(None, _run_streaming_generator)

            for audio_chunk, metrics in results:
                if self._check_stopped():
                    logger.info("Streaming synthesis stopped by request")
                    break

                chunk_count += 1

                # Record first chunk latency (only set once in metrics)
                if first_chunk_latency is None and metrics.latency_to_first_chunk:
                    first_chunk_latency = metrics.latency_to_first_chunk
                    logger.info(f"First chunk latency: {first_chunk_latency:.3f}s")

                # Convert tensor to WAV bytes
                audio_np = audio_chunk.squeeze(0).numpy()
                audio_int16 = (audio_np * 32767).astype(np.int16)

                # Apply fades to smooth chunk boundaries
                is_final = getattr(metrics, 'is_final', False)
                if apply_chunk_fades:
                    # Apply fade-in only to first chunk
                    if chunk_count == 1:
                        audio_int16 = self._apply_fade_in(audio_int16)
                    # Apply fade-out to all chunks except final for smooth transitions
                    if not is_final:
                        audio_int16 = self._apply_fade_out(audio_int16)

                wav_bytes = self._pcm_to_wav(audio_int16.tobytes())
                audio_base64 = base64.b64encode(wav_bytes).decode('utf-8')

                yield {
                    "type": "audio_chunk",
                    "audio": audio_base64,
                    "index": audio_index,
                    "chunk_index": chunk_count,
                    "is_final": is_final
                }

                # Yield control to event loop
                await asyncio.sleep(0)

            total_duration = time.time() - start_time
            try:
                from app.utils.timing import timing_logger
                timing_logger.info(
                    f"TTS - Chatterbox Streaming Generation: {total_duration:.3f}s | "
                    f"{{\"text_length\": {len(text)}, \"first_chunk_latency\": {first_chunk_latency or 0:.3f}, "
                    f"\"chunks\": {chunk_count}, \"index\": {audio_index}}}"
                )
            except ImportError:
                pass

            logger.info(f"Streaming generated {chunk_count} audio chunks for index {audio_index}")

        except Exception as e:
            logger.error(f"Error in streaming generation: {str(e)}", exc_info=True)
            raise

    async def _stream_audio_chunks(
        self,
        audio: np.ndarray,
        audio_index: int
    ) -> AsyncIterator[dict]:
        """
        Stream audio chunks immediately without pre-building list.
        Yields chunks as soon as they're encoded for minimum latency.
        """
        chunk_size = int(self.SAMPLE_RATE * self.chunk_duration_ms / 1000)
        chunk_index = 0

        for i in range(0, len(audio), chunk_size):
            if self._check_stopped():
                logger.info("Synthesis stopped by request")
                break

            chunk = audio[i:i + chunk_size]
            chunk_bytes = chunk.tobytes()
            wav_bytes = self._pcm_to_wav(chunk_bytes)
            audio_base64 = base64.b64encode(wav_bytes).decode('utf-8')

            chunk_index += 1
            yield {
                "type": "audio_chunk",
                "audio": audio_base64,
                "index": audio_index,
                "chunk_index": chunk_index
            }

            # Yield control to event loop after each chunk for responsive streaming
            await asyncio.sleep(0)

    def cleanup(self):
        """Clean up resources."""
        try:
            # Stop any ongoing synthesis
            self.stop()

            if self.model:
                del self.model
                self.model = None
                self._model_loaded = False

            # Clear caches
            self._voice_cache.clear()
            self._current_voice_hash = None

            # Clear GPU cache if using CUDA
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

            logger.info("Chatterbox TTS service cleaned up successfully")
        except Exception as e:
            logger.warning(f"Error during cleanup: {str(e)}")
