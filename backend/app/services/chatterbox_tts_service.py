"""
Chatterbox TTS Service Module
Real-time streaming TTS with voice cloning using Chatterbox.

Features:
- Ultra-low latency streaming (<250ms time-to-first-audio)
- Progressive chunk generation for immediate playback
- Voice cloning via reference audio
- CUDA graph compilation for optimal performance
- Async streaming interface compatible with BaseTTSService
"""

import logging
import base64
import io
import time
import wave
import asyncio
from typing import AsyncIterator
from pathlib import Path

import numpy as np
import torch

from .base_tts_service import BaseTTSService

logger = logging.getLogger(__name__)


class ChatterboxTTSService(BaseTTSService):
    """
    Chatterbox TTS service for real-time interview TTS with voice cloning.

    Uses Chatterbox TTS with progressive streaming for sub-250ms latency.
    """

    def __init__(
        self,
        reference_audio_path: str,
        device: str = "cuda",
        chunk_size: int = 25,
        context_window: int = 50,
        fade_duration: float = 0.02,
        cfm_steps: int = 7,
        use_fp16: bool = True,
        optimize_gpu: bool = True,
        preload: bool = True
    ):
        """
        Initialize the Chatterbox TTS service.

        Args:
            reference_audio_path: Path to reference audio for voice cloning (WAV recommended)
            device: Device to run model on ('cuda' or 'cpu')
            chunk_size: Tokens per streaming chunk (15-30, lower=faster TTFA)
            context_window: Tokens of context for continuity (default: 50)
            fade_duration: Audio fade transition in seconds (prevents clicks)
            cfm_steps: CFM inference steps (7 for real-time, 10+ for quality)
            use_fp16: Use half-precision for 2x speed and 50% memory reduction
            optimize_gpu: Enable CUDA graph compilation for 30-50% speedup
            preload: If True, load model immediately at initialization
        """
        # Store parameters
        self.reference_audio_path = reference_audio_path
        self.device = device
        self.chunk_size = chunk_size
        self.context_window = context_window
        self.fade_duration = fade_duration
        self.cfm_steps = cfm_steps
        self.use_fp16 = use_fp16
        self.optimize_gpu = optimize_gpu

        self.model = None
        self.sample_rate = None
        self._model_loaded = False

        if preload:
            self._load_model()

    def _load_model(self):
        """Load ChatterboxTTS model and prepare voice conditionals."""
        if self._model_loaded:
            return

        try:
            from chatterbox import ChatterboxTTS

            logger.info("Loading Chatterbox TTS model...")
            load_start = time.time()

            # Initialize model with from_pretrained factory
            logger.info(f"  Device: {self.device}")
            logger.info(f"  FP16: {self.use_fp16}")
            logger.info(f"  GPU Optimization: {self.optimize_gpu}")

            self.model = ChatterboxTTS.from_pretrained(
                device=self.device,
                use_fp16=self.use_fp16,
                optimize_gpu=self.optimize_gpu
            )

            # Get sample rate from model
            self.sample_rate = self.model.sr  # Usually 24000 Hz
            logger.info(f"  Sample rate: {self.sample_rate} Hz")

            # Prepare voice conditionals for cloning (REQUIRED before generation)
            if not Path(self.reference_audio_path).exists():
                raise FileNotFoundError(
                    f"Reference audio not found: {self.reference_audio_path}"
                )

            logger.info(f"  Loading reference audio: {self.reference_audio_path}")
            self.model.prepare_conditionals(self.reference_audio_path)
            logger.info("  Voice conditionals prepared")

            # Run warmup to compile CUDA graphs (critical for performance)
            logger.info("  Running warmup pass to compile CUDA graphs...")
            warmup_start = time.time()
            self._warmup()
            warmup_duration = time.time() - warmup_start
            logger.info(f"  Warmup complete ({warmup_duration:.1f}s)")

            load_duration = time.time() - load_start
            self._model_loaded = True

            logger.info(f"Chatterbox TTS model loaded successfully in {load_duration:.1f}s")
            logger.info(f"  - Device: {self.device}")
            logger.info(f"  - Sample rate: {self.sample_rate} Hz")
            logger.info(f"  - Chunk size: {self.chunk_size} tokens")
            logger.info(f"  - Context window: {self.context_window} tokens")
            logger.info(f"  - CFM steps: {self.cfm_steps}")
            logger.info(f"  - Optimizations: FP16={self.use_fp16}, GPU={self.optimize_gpu}")

        except ImportError as e:
            logger.error("Chatterbox TTS not found. Please install it first.")
            logger.error(f"Error: {e}")
            logger.error("Install with: pip install chatterbox-tts")
            raise ImportError(
                "Chatterbox TTS is not installed. "
                "Please install with: pip install chatterbox-tts"
            ) from e

    def _warmup(self):
        """Warmup generation to compile CUDA graphs."""
        warmup_text = "Hello, this is a warmup."
        warmup_chunks = 0
        for _, _ in self.model.generate_streaming(
            text=warmup_text,
            chunk_size=self.chunk_size,
            cfm_steps=self.cfm_steps
        ):
            warmup_chunks += 1
        logger.debug(f"  Warmup generated {warmup_chunks} chunks")

    async def generate_stream(
        self,
        text: str,
        audio_index: int = 0
    ) -> AsyncIterator[dict]:
        """
        Generate streaming audio from text.

        Wraps synchronous generate_streaming() in async generator.
        Chatterbox streaming is CPU-bound sync code, so we use asyncio.sleep(0)
        to yield control back to event loop between chunks.

        Args:
            text: Text to synthesize
            audio_index: Audio segment index for ordering

        Yields:
            dict with 'type', 'audio' (base64 WAV), 'index', 'chunk_index'
        """
        # Lazy load if not preloaded
        if not self._model_loaded:
            self._load_model()

        if not text or not text.strip():
            logger.warning("Empty text provided to TTS")
            return

        text_preview = text[:50] + ('...' if len(text) > 50 else '')
        logger.info(f"Generating audio for: '{text_preview}' (index: {audio_index})")

        chunk_count = 0
        first_chunk_latency = None
        start_time = time.time()

        try:
            # Create streaming generator (synchronous)
            stream = self.model.generate_streaming(
                text=text,
                chunk_size=self.chunk_size,
                context_window=self.context_window,
                fade_duration=self.fade_duration,
                cfm_steps=self.cfm_steps,
            )

            # Iterate over chunks - Chatterbox yields (audio_chunk, metrics) tuples
            for audio_chunk, metrics in stream:
                chunk_count += 1

                # Log first chunk latency (important metric)
                if first_chunk_latency is None:
                    first_chunk_latency = metrics.latency_to_first_chunk
                    logger.info(f"  First chunk latency: {first_chunk_latency:.1f}ms")

                # Convert numpy float32 array to int16 WAV
                # audio_chunk is float32 numpy array in range [-1, 1]
                audio_int16 = (audio_chunk * 32767).astype(np.int16)
                wav_bytes = self._numpy_to_wav(audio_int16)

                # Encode to base64
                audio_base64 = base64.b64encode(wav_bytes).decode('utf-8')

                # Yield in BaseTTSService format
                yield {
                    "type": "audio_chunk",
                    "audio": audio_base64,
                    "index": audio_index,
                    "chunk_index": chunk_count
                }

                # Yield control to event loop (allows concurrent requests)
                await asyncio.sleep(0)

            total_duration = time.time() - start_time

            # Log timing metrics
            try:
                from app.utils.timing import timing_logger
                timing_logger.info(
                    f"TTS - Chatterbox Streaming: {total_duration:.3f}s | "
                    f"{{\"text_length\": {len(text)}, \"first_chunk_latency\": {first_chunk_latency or 0:.3f}, "
                    f"\"chunks\": {chunk_count}, \"rtf\": {metrics.rtf:.2f}, \"index\": {audio_index}}}"
                )
            except ImportError:
                pass

            logger.info(
                f"Generated {chunk_count} chunks in {total_duration:.2f}s "
                f"(RTF: {metrics.rtf:.2f}, First chunk: {first_chunk_latency:.1f}ms)"
            )

        except Exception as e:
            logger.error(f"Error generating audio: {str(e)}", exc_info=True)
            raise

    def _numpy_to_wav(self, audio_int16: np.ndarray) -> bytes:
        """Convert int16 numpy array to WAV bytes."""
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        return buffer.getvalue()

    def cleanup(self):
        """Release model resources and clear CUDA cache."""
        try:
            if self.model:
                del self.model
                self.model = None

            self._model_loaded = False

            # Clear GPU cache if using CUDA
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Chatterbox TTS service cleaned up successfully")
        except Exception as e:
            logger.warning(f"Error during cleanup: {str(e)}")
