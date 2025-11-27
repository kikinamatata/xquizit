"""
TTS Service Module
Factory pattern for TTS service initialization using Chatterbox TTS engine.
"""

import logging
from typing import Optional

from .base_tts_service import BaseTTSService
from .chatterbox_tts_service import ChatterboxTTSService

logger = logging.getLogger(__name__)


# Global TTS service instance (initialized on startup)
_tts_service: Optional[BaseTTSService] = None


def initialize_tts_service(
    # Voice cloning
    reference_audio_path: str,
    # Core parameters
    device: str = "cuda",
    chunk_size: int = 25,
    context_window: int = 50,
    fade_duration: float = 0.02,
    # Generation quality
    cfm_steps: int = 7,
    # Performance optimization
    use_fp16: bool = True,
    optimize_gpu: bool = True,
    preload: bool = True
) -> BaseTTSService:
    """
    Initialize the global TTS service instance using Chatterbox TTS.

    Args:
        reference_audio_path: Path to reference audio for voice cloning (WAV recommended)
        device: Device to run model on ('cuda' or 'cpu')
        chunk_size: Tokens per streaming chunk (15-30, lower=faster TTFA)
        context_window: Tokens of context for continuity
        fade_duration: Audio fade transition in seconds (prevents clicks)
        cfm_steps: CFM inference steps (7 for real-time, 10+ for quality)
        use_fp16: Use half-precision for 2x speed and 50% memory reduction
        optimize_gpu: Enable CUDA graph compilation for 30-50% speedup
        preload: If True, load model immediately at startup

    Returns:
        Initialized ChatterboxTTSService instance

    Raises:
        RuntimeError: If TTS service initialization fails
    """
    global _tts_service

    if _tts_service is not None:
        logger.warning("TTS service already initialized, returning existing instance")
        return _tts_service

    try:
        logger.info(f"Initializing Chatterbox TTS service (device={device}, preload={preload})")

        _tts_service = ChatterboxTTSService(
            reference_audio_path=reference_audio_path,
            device=device,
            chunk_size=chunk_size,
            context_window=context_window,
            fade_duration=fade_duration,
            cfm_steps=cfm_steps,
            use_fp16=use_fp16,
            optimize_gpu=optimize_gpu,
            preload=preload
        )

        logger.info("TTS service initialized successfully with Chatterbox TTS")
        return _tts_service

    except Exception as e:
        logger.error(f"Failed to initialize TTS service: {e}", exc_info=True)
        _tts_service = None
        raise


def get_tts_service() -> BaseTTSService:
    """
    Get the global TTS service instance.

    Returns:
        TTS service instance (ChatterboxTTSService)

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
