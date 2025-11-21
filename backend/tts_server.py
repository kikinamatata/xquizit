#!/usr/bin/env python3
"""
WebSocket TTS Server for RealtimeTTS ChatterboxEngine

This server receives text via WebSocket and streams back synthesized audio chunks.
Designed for deployment on RunPod with GPU support.

Protocol:
  Client -> Server (JSON):
    {"action": "set_voice", "voice": "default"}
    {"action": "synthesize", "text": "Hello world"}
    {"action": "stop"}

  Server -> Client:
    JSON: {"type": "ready", "sampleRate": 24000, "channels": 1, "format": "int16"}
    Binary: WAV audio chunks (PCM with WAV headers, 24kHz, mono, int16)
    Binary: Empty (0 bytes) = end of synthesis
"""

# IMPORTANT: Print before any imports to confirm file is loaded
print("=" * 70)
print(">>> TTS_SERVER.PY LOADING - VERSION WITH DEBUG LOGGING <<<")
print("=" * 70)
import sys
sys.stdout.flush()

import asyncio
import io
import json
import logging
import os
import queue
import struct
import sys
import threading
from pathlib import Path
from typing import Optional

import websockets

# Add parent directory to path to import RealtimeTTS
sys.path.insert(0, str(Path(__file__).parent.parent))

from RealtimeTTS import ChatterboxEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def log_and_flush(message):
    """Log message and flush to ensure immediate output"""
    print(f"[LOG] {message}")  # Direct print for visibility
    logger.info(message)
    sys.stdout.flush()
    sys.stderr.flush()


def pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 24000, channels: int = 1, sample_width: int = 2) -> bytes:
    """
    Convert raw PCM audio bytes to WAV format with headers.

    Args:
        pcm_bytes: Raw PCM audio data (int16 format)
        sample_rate: Sample rate in Hz (default: 24000)
        channels: Number of audio channels (default: 1 for mono)
        sample_width: Sample width in bytes (default: 2 for int16)

    Returns:
        WAV formatted audio bytes with headers
    """
    # Calculate sizes
    data_size = len(pcm_bytes)
    file_size = data_size + 36  # 44 byte header - 8 bytes (RIFF header)

    # Build WAV header
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',           # ChunkID
        file_size,         # ChunkSize (file size - 8)
        b'WAVE',           # Format
        b'fmt ',           # Subchunk1ID
        16,                # Subchunk1Size (16 for PCM)
        1,                 # AudioFormat (1 for PCM)
        channels,          # NumChannels
        sample_rate,       # SampleRate
        sample_rate * channels * sample_width,  # ByteRate
        channels * sample_width,  # BlockAlign
        sample_width * 8,  # BitsPerSample
        b'data',           # Subchunk2ID
        data_size          # Subchunk2Size
    )

    return header + pcm_bytes


class TTSServer:
    """WebSocket TTS Server using ChatterboxEngine"""

    def __init__(
        self,
        voice_config_path: str = "voice_config.json",
        device: str = "cuda",
        chunk_duration_ms: int = 100,
        port: int = 8765,
        host: str = "0.0.0.0"
    ):
        """
        Initialize TTS Server

        Args:
            voice_config_path: Path to voice configuration JSON
            device: "cuda" or "cpu" (default: "cuda")
            chunk_duration_ms: Audio chunk size in milliseconds
            port: WebSocket server port
            host: WebSocket server host
        """
        try:
            log_and_flush(f"Starting TTSServer initialization...")

            self.voice_config_path = voice_config_path
            self.device = device
            self.chunk_duration_ms = chunk_duration_ms
            self.port = port
            self.host = host

            # Load voice configuration
            log_and_flush("Loading voice configuration...")
            self.voices = self._load_voice_config()
            log_and_flush(f"Loaded {len(self.voices)} voices: {list(self.voices.keys())}")

            # Initialize ChatterboxEngine (lazy loading on first synthesis)
            log_and_flush(f"Initializing ChatterboxEngine (device: {device or 'auto'})...")
            self.engine = ChatterboxEngine(
                device=device,
                chunk_duration_ms=chunk_duration_ms,
                level=logging.INFO
            )
            log_and_flush("ChatterboxEngine created successfully")

            # Set default voice if available
            if "default" in self.voices:
                log_and_flush(f"Setting default voice: {self.voices['default']}")
                self.engine.set_voice(self.voices["default"])
                log_and_flush(f"Default voice set successfully")

            # Preload model at startup (warm-up)
            log_and_flush("=" * 60)
            log_and_flush("Preloading Chatterbox model (this may take 30-60 seconds)...")
            log_and_flush("=" * 60)
            try:
                # Trigger model loading by calling get_stream_info()
                stream_info = self.engine.get_stream_info()
                log_and_flush("=" * 60)
                log_and_flush("✓ Model preloaded successfully!")
                log_and_flush(f"  - Sample rate: {stream_info.get('sample_rate', 24000)}Hz")
                log_and_flush(f"  - Channels: {stream_info.get('channels', 1)}")
                log_and_flush("=" * 60)
            except Exception as e:
                log_and_flush("=" * 60)
                log_and_flush(f"⚠ Warning: Failed to preload model: {e}")
                log_and_flush("  Model will load on first synthesis request")
                log_and_flush("=" * 60)

            # Track synthesis state
            log_and_flush("Initializing synthesis state tracking...")
            self.synthesis_thread: Optional[threading.Thread] = None
            self.is_synthesizing = False
            self.stop_requested = False

            log_and_flush(f"TTS Server initialized on {host}:{port}")

        except Exception as e:
            logger.error(f"FATAL ERROR during initialization: {e}", exc_info=True)
            sys.stdout.flush()
            sys.stderr.flush()
            raise

    def _load_voice_config(self) -> dict:
        """Load voice configuration from JSON file"""
        config_path = Path(self.voice_config_path)

        if not config_path.exists():
            logger.warning(f"Voice config not found: {config_path}")
            return {}

        try:
            with open(config_path, 'r') as f:
                voices = json.load(f)

            # Validate voice files exist
            for name, path in list(voices.items()):
                if not os.path.exists(path):
                    logger.warning(f"Voice file not found: {name} -> {path}")
                    del voices[name]

            return voices

        except Exception as e:
            logger.error(f"Failed to load voice config: {e}")
            return {}

    def _synthesize_worker(self, text: str):
        """Background worker that synthesizes text and queues audio chunks"""
        try:
            logger.info(f"Synthesizing: {text[:50]}...")
            self.is_synthesizing = True
            self.stop_requested = False

            # Synthesize (fills engine.queue with chunks)
            success = self.engine.synthesize(text)

            if success:
                logger.info(f"Synthesis complete ({self.engine.audio_duration:.2f}s audio)")
            else:
                logger.error("Synthesis failed")

        except Exception as e:
            logger.error(f"Synthesis error: {e}", exc_info=True)

        finally:
            self.is_synthesizing = False

    async def handle_client(self, websocket):
        """Handle WebSocket client connection"""
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logger.info(f"Client connected: {client_id}")

        try:
            # Send audio configuration immediately (don't wait for model loading)
            # Note: ChatterboxEngine always uses 24kHz, mono, int16 - no need to query
            config_message = {
                "type": "ready",
                "sampleRate": 24000,
                "channels": 1,
                "format": "int16"
            }
            await websocket.send(json.dumps(config_message))
            logger.info(f"Sent audio config to {client_id} (model will load on first synthesis)")

            # Handle messages from client
            async for message in websocket:
                await self.handle_message(websocket, message)

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {client_id}")

        except Exception as e:
            logger.error(f"Error handling client {client_id}: {e}", exc_info=True)

        finally:
            # Stop any ongoing synthesis
            if self.is_synthesizing:
                self.stop_requested = True
                self.engine.stop_synthesis_event.set()
                logger.info(f"Stopped synthesis for disconnected client: {client_id}")

    async def handle_message(self, websocket, message):
        """Handle incoming message from client"""
        try:
            # Parse JSON message
            data = json.loads(message)
            action = data.get("action")

            if action == "set_voice":
                # Set voice
                voice_name = data.get("voice", "default")
                if voice_name in self.voices:
                    self.engine.set_voice(self.voices[voice_name])
                    logger.info(f"Voice changed to: {voice_name}")
                    await websocket.send(json.dumps({
                        "type": "voice_changed",
                        "voice": voice_name
                    }))
                else:
                    logger.warning(f"Unknown voice: {voice_name}")
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": f"Voice not found: {voice_name}"
                    }))

            elif action == "synthesize":
                # Synthesize text
                text = data.get("text", "")
                if not text:
                    logger.warning("Empty text received")
                    return

                # Start synthesis in background thread
                self.synthesis_thread = threading.Thread(
                    target=self._synthesize_worker,
                    args=(text,)
                )
                self.synthesis_thread.start()

                # Stream audio chunks to client
                await self.stream_audio(websocket)

            elif action == "stop":
                # Stop synthesis
                if self.is_synthesizing:
                    self.stop_requested = True
                    self.engine.stop_synthesis_event.set()
                    logger.info("Synthesis stopped by client")

        except json.JSONDecodeError:
            logger.error(f"Invalid JSON: {message}")

        except Exception as e:
            logger.error(f"Error handling message: {e}", exc_info=True)

    async def stream_audio(self, websocket):
        """Stream audio chunks from engine queue to WebSocket (as WAV format)"""
        try:
            while True:
                # Check if synthesis is still running
                if not self.is_synthesizing and self.engine.queue.empty():
                    break

                # Get chunk from queue (non-blocking with timeout)
                try:
                    chunk = self.engine.queue.get(timeout=0.1)

                    # Convert raw PCM chunk to WAV format
                    wav_chunk = pcm_to_wav(chunk, sample_rate=24000, channels=1, sample_width=2)

                    # Send WAV chunk as binary WebSocket message
                    await websocket.send(wav_chunk)

                except queue.Empty:
                    # Queue empty, check if synthesis still running
                    if self.synthesis_thread and not self.synthesis_thread.is_alive():
                        # Synthesis complete and queue empty
                        break

                    # Wait a bit and try again
                    await asyncio.sleep(0.01)

                # Check if stop requested
                if self.stop_requested:
                    # Clear remaining queue
                    while not self.engine.queue.empty():
                        try:
                            self.engine.queue.get_nowait()
                        except queue.Empty:
                            break
                    break

            # Send end-of-stream marker (empty binary message)
            await websocket.send(b'')
            logger.info("Audio streaming complete")

        except Exception as e:
            logger.error(f"Error streaming audio: {e}", exc_info=True)

    async def start(self):
        """Start the WebSocket server"""
        try:
            log_and_flush(f"Starting WebSocket server on {self.host}:{self.port}")

            async with websockets.serve(self.handle_client, self.host, self.port):
                log_and_flush(f"Server ready and listening on ws://{self.host}:{self.port}")
                await asyncio.Future()  # Run forever

        except Exception as e:
            logger.error(f"FATAL ERROR starting server: {e}", exc_info=True)
            sys.stdout.flush()
            sys.stderr.flush()
            raise


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="WebSocket TTS Server")
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to bind to (default: 8765)"
    )
    parser.add_argument(
        "--voice-config",
        default="voice_config.json",
        help="Path to voice configuration JSON (default: voice_config.json)"
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device to use (default: cuda)"
    )
    parser.add_argument(
        "--chunk-duration",
        type=int,
        default=100,
        help="Audio chunk duration in ms (default: 100)"
    )

    args = parser.parse_args()

    try:
        log_and_flush("=" * 60)
        log_and_flush("WebSocket TTS Server Starting...")
        log_and_flush("=" * 60)

        # Create server
        log_and_flush("Creating TTSServer instance...")
        server = TTSServer(
            voice_config_path=args.voice_config,
            device=args.device,
            chunk_duration_ms=args.chunk_duration,
            port=args.port,
            host=args.host
        )

        log_and_flush("TTSServer instance created successfully!")
        log_and_flush("=" * 60)

        # Run server
        asyncio.run(server.start())

    except KeyboardInterrupt:
        log_and_flush("\nServer stopped by user")
    except Exception as e:
        logger.error(f"\nFATAL ERROR in main: {e}", exc_info=True)
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(1)


if __name__ == "__main__":
    print("[DEBUG] About to call main()")
    sys.stdout.flush()
    main()
