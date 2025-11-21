"""
WebSocket TTS Service Module
Connects to an external WebSocket TTS server for real-time audio generation.
Adapted from tts_client.py for server-side use (no audio playback).
"""

import asyncio
import json
import logging
import base64
from typing import Optional, AsyncIterator

import websockets

from .base_tts_service import BaseTTSService

logger = logging.getLogger(__name__)


class WebSocketTTSService(BaseTTSService):
    """
    WebSocket TTS Service for connecting to an external TTS server.

    This service connects to a WebSocket TTS server (e.g., RealtimeTTS server),
    sends text for synthesis, and streams back the generated audio chunks.

    Unlike the original client, this service does NOT play audio locally.
    It's designed for server-side use where audio is forwarded to clients.
    """

    def __init__(self, server_url: str):
        """
        Initialize WebSocket TTS service.

        Args:
            server_url: WebSocket server URL (e.g., ws://localhost:8765)
        """
        self.server_url = server_url
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.is_connected = False

        # Audio configuration (received from server)
        self.sample_rate = 24000
        self.channels = 1

        logger.info(f"WebSocketTTSService initialized for {server_url}")

    async def connect(self):
        """Connect to WebSocket TTS server."""
        if self.is_connected:
            logger.warning("Already connected to TTS server")
            return

        try:
            logger.info(f"Connecting to TTS server at {self.server_url}...")
            self.websocket = await websockets.connect(self.server_url)
            self.is_connected = True

            # Wait for audio configuration from server
            # Longer timeout for initial model loading (Chatterbox can take 30+ seconds)
            logger.info("Waiting for TTS server to finish loading model...")
            config_msg = await asyncio.wait_for(self.websocket.recv(), timeout=60.0)
            config = json.loads(config_msg)

            if config.get("type") == "ready":
                self.sample_rate = config.get("sampleRate", 24000)
                self.channels = config.get("channels", 1)
                logger.info(
                    f"✓ Connected to TTS server! Audio config: {self.sample_rate}Hz, {self.channels}ch"
                )
            else:
                logger.warning(f"Unexpected first message from server: {config}")

        except asyncio.TimeoutError:
            logger.error("Timeout waiting for TTS server configuration")
            self.is_connected = False
            raise RuntimeError("TTS server connection timeout")
        except Exception as e:
            logger.error(f"Failed to connect to TTS server: {e}", exc_info=True)
            self.is_connected = False
            raise

    async def close(self):
        """Close connection to TTS server."""
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception as e:
                logger.debug(f"Error closing websocket: {e}")
            finally:
                self.websocket = None

        self.is_connected = False
        logger.info("TTS server connection closed")

    async def generate_stream(
        self,
        text: str,
        audio_index: int = 0
    ) -> AsyncIterator[dict]:
        """
        Generate audio from text using the WebSocket TTS server.

        Args:
            text: Text to convert to speech
            audio_index: Index number for this audio chunk (for ordering)

        Yields:
            Dictionary with 'type', 'audio' (base64), 'index', and 'chunk_index' keys
        """
        if not self.is_connected:
            logger.warning("Not connected to TTS server, attempting to connect...")
            await self.connect()

        if not text or not text.strip():
            logger.warning("Empty text provided to TTS")
            return

        try:
            logger.info(f"Generating audio for text: '{text[:50]}...' (index: {audio_index})")

            # Send synthesis request to server
            message = {
                "action": "synthesize",
                "text": text
            }
            await self.websocket.send(json.dumps(message))
            logger.debug(f"Sent synthesis request for: {text[:50]}...")

            # Receive and yield audio chunks
            chunk_count = 0
            async for audio_chunk in self._receive_audio_chunks():
                # Convert binary audio to base64
                audio_base64 = base64.b64encode(audio_chunk).decode('utf-8')

                chunk_count += 1
                yield {
                    "type": "audio_chunk",
                    "audio": audio_base64,
                    "index": audio_index,
                    "chunk_index": chunk_count
                }

            logger.info(f"Generated {chunk_count} audio chunks for index {audio_index}")

        except Exception as e:
            logger.error(f"Error generating audio via WebSocket: {str(e)}", exc_info=True)
            # Attempt to reconnect for next request
            self.is_connected = False
            raise

    async def _receive_audio_chunks(self):
        """
        Generator that yields audio chunks from the TTS server.

        Yields:
            Audio chunks as bytes
        """
        try:
            while True:
                # Receive message (binary or JSON)
                message = await asyncio.wait_for(self.websocket.recv(), timeout=30.0)

                # Check if it's JSON (control message)
                if isinstance(message, str):
                    data = json.loads(message)
                    msg_type = data.get("type")

                    if msg_type == "error":
                        logger.error(f"TTS server error: {data.get('message')}")
                        break

                    elif msg_type == "voice_changed":
                        logger.info(f"Voice changed: {data.get('voice')}")
                        continue

                    else:
                        logger.debug(f"Control message from TTS server: {data}")
                        continue

                # Binary message (audio chunk)
                elif isinstance(message, bytes):
                    # Empty message = end of stream
                    if len(message) == 0:
                        logger.debug("End of audio stream")
                        break

                    # Yield audio chunk
                    yield message

        except asyncio.TimeoutError:
            logger.warning("Timeout waiting for audio chunks from TTS server")

        except websockets.exceptions.ConnectionClosed:
            logger.warning("TTS server connection closed during audio reception")
            self.is_connected = False

        except Exception as e:
            logger.error(f"Error receiving audio chunks: {e}", exc_info=True)
            self.is_connected = False

    def cleanup(self):
        """Clean up resources (synchronous wrapper for async close)."""
        try:
            # Create event loop if needed for cleanup
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is running, schedule cleanup
                    asyncio.create_task(self.close())
                else:
                    # If loop is not running, run cleanup
                    loop.run_until_complete(self.close())
            except RuntimeError:
                # No event loop, create new one for cleanup
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.close())
                loop.close()

            logger.info("WebSocket TTS service cleaned up successfully")
        except Exception as e:
            logger.warning(f"Error during WebSocket TTS cleanup: {str(e)}")
