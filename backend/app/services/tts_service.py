"""
TTS Service Module
Factory pattern for TTS service initialization using Indic Parler-TTS engine.
"""

import logging
from typing import Optional

from .base_tts_service import BaseTTSService
from .parler_tts_service import IndicParlerTTSService
# Keep Chatterbox import for backward compatibility (if rollback needed)
# from .chatterbox_tts_service import ChatterboxTTSService

logger = logging.getLogger(__name__)


# Global TTS service instance (initialized on startup)
_tts_service: Optional[BaseTTSService] = None


def initialize_tts_service(
    # Model selection
    model_id: str = "ai4bharat/indic-parler-tts",
    # Core parameters
    device: str = "auto",
    voice_description: str = "Thoma speaks with a clear, moderate pace in a close recording with minimal background noise and a slightly expressive tone",
    play_steps_in_s: float = 0.3,  # 300ms target TTFA
    # Generation parameters
    temperature: float = 1.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    min_new_tokens: int = 10,
    max_new_tokens: int = 2000,
    # Performance optimization
    enable_compile: bool = True,
    # Model loading
    preload: bool = True
) -> BaseTTSService:
    """
    Initialize the global TTS service instance using Indic Parler-TTS.

    Args:
        model_id: HuggingFace model ID (default: "ai4bharat/indic-parler-tts")
        device: Device to run model on ('cuda', 'cpu', or 'auto')
        voice_description: Natural language description of desired voice
        play_steps_in_s: Target seconds for first audio chunk (lower = faster TTFA)
        temperature: Sampling temperature (1.0 recommended)
        top_p: Nucleus sampling probability threshold
        repetition_penalty: Penalty for repeated tokens
        min_new_tokens: Minimum tokens to generate
        max_new_tokens: Maximum tokens to generate (~22s audio at 2000)
        enable_compile: Enable torch.compile() for 4x speedup (requires warmup)
        preload: If True, load model immediately at startup

    Returns:
        Initialized IndicParlerTTSService instance

    Raises:
        RuntimeError: If TTS service initialization fails
    """
    global _tts_service

    if _tts_service is not None:
        logger.warning("TTS service already initialized, returning existing instance")
        return _tts_service

    try:
        logger.info(f"Initializing Indic Parler-TTS service (model={model_id}, device={device}, preload={preload})")

        _tts_service = IndicParlerTTSService(
            model_id=model_id,
            device=device,
            voice_description=voice_description,
            play_steps_in_s=play_steps_in_s,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            min_new_tokens=min_new_tokens,
            max_new_tokens=max_new_tokens,
            enable_compile=enable_compile,
            preload=preload
        )

        logger.info("TTS service initialized successfully with Indic Parler-TTS")
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
