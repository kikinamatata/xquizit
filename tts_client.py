#!/usr/bin/env python3
"""
WebSocket TTS Client for RealtimeTTS

This client connects to a TTS WebSocket server, sends text, and plays audio.
Supports streaming LLM output for real-time TTS conversion.

Usage:
    # Basic usage
    client = TTSClient("ws://localhost:8765")
    await client.connect()
    await client.synthesize("Hello world!")
    await client.close()

    # Stream LLM output
    async for chunk in llm_generator():
        await client.synthesize(chunk)
"""

import asyncio
import json
import logging
import queue
import threading
from typing import Optional, Generator

import pyaudio
import websockets

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AudioPlayer:
    """Plays PCM audio chunks using PyAudio"""

    def __init__(self, sample_rate: int = 24000, channels: int = 1, format=pyaudio.paInt16):
        """
        Initialize audio player

        Args:
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels (1=mono, 2=stereo)
            format: PyAudio format (paInt16 for 16-bit PCM)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.format = format

        self.pyaudio = pyaudio.PyAudio()
        self.stream: Optional[pyaudio.Stream] = None
        self.is_playing = False

        # Audio buffer queue
        self.audio_queue = queue.Queue()
        self.playback_thread: Optional[threading.Thread] = None

        logger.info(f"AudioPlayer initialized: {sample_rate}Hz, {channels}ch, format={format}")

    def start(self):
        """Start audio playback"""
        if self.is_playing:
            logger.warning("AudioPlayer already playing")
            return

        # Open PyAudio stream
        self.stream = self.pyaudio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            output=True,
            frames_per_buffer=1024
        )

        # Start playback thread
        self.is_playing = True
        self.playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self.playback_thread.start()

        logger.info("Audio playback started")

    def stop(self):
        """Stop audio playback"""
        if not self.is_playing:
            return

        self.is_playing = False

        # Wait for playback thread to finish
        if self.playback_thread:
            self.playback_thread.join(timeout=2.0)

        # Close stream
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

        # Clear queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        logger.info("Audio playback stopped")

    def add_chunk(self, chunk: bytes):
        """Add audio chunk to playback queue"""
        if not self.is_playing:
            logger.warning("Cannot add chunk: player not started")
            return

        self.audio_queue.put(chunk)

    def _playback_worker(self):
        """Worker thread that plays audio chunks from queue"""
        try:
            while self.is_playing:
                try:
                    # Get chunk from queue with timeout
                    chunk = self.audio_queue.get(timeout=0.1)

                    # Play chunk
                    if self.stream and len(chunk) > 0:
                        self.stream.write(chunk)

                except queue.Empty:
                    continue

        except Exception as e:
            logger.error(f"Playback worker error: {e}", exc_info=True)

    def close(self):
        """Close audio player and release resources"""
        self.stop()
        self.pyaudio.terminate()
        logger.info("AudioPlayer closed")


class TTSClient:
    """WebSocket TTS Client"""

    def __init__(self, server_url: str, auto_play: bool = True):
        """
        Initialize TTS client

        Args:
            server_url: WebSocket server URL (e.g., ws://localhost:8765)
            auto_play: Automatically play received audio
        """
        self.server_url = server_url
        self.auto_play = auto_play

        self.websocket = None
        self.audio_player: Optional[AudioPlayer] = None
        self.is_connected = False

        # Audio configuration (received from server)
        self.sample_rate = 24000
        self.channels = 1
        self.format = pyaudio.paInt16

        logger.info(f"TTSClient initialized for {server_url}")

    async def connect(self):
        """Connect to WebSocket server"""
        if self.is_connected:
            logger.warning("Already connected")
            return

        try:
            logger.info(f"Connecting to {self.server_url}...")
            self.websocket = await websockets.connect(self.server_url)
            self.is_connected = True

            # Wait for audio configuration from server
            config_msg = await self.websocket.recv()
            config = json.loads(config_msg)

            if config.get("type") == "ready":
                self.sample_rate = config.get("sampleRate", 24000)
                self.channels = config.get("channels", 1)
                logger.info(f"Connected! Audio config: {self.sample_rate}Hz, {self.channels}ch")

                # Initialize audio player
                if self.auto_play:
                    self.audio_player = AudioPlayer(
                        sample_rate=self.sample_rate,
                        channels=self.channels,
                        format=self.format
                    )
                    self.audio_player.start()

            else:
                logger.warning(f"Unexpected first message: {config}")

        except Exception as e:
            logger.error(f"Connection failed: {e}", exc_info=True)
            self.is_connected = False
            raise

    async def close(self):
        """Close connection and clean up resources"""
        if self.audio_player:
            self.audio_player.close()
            self.audio_player = None

        if self.websocket:
            await self.websocket.close()
            self.websocket = None

        self.is_connected = False
        logger.info("Connection closed")

    async def set_voice(self, voice: str = "default"):
        """
        Set voice on server

        Args:
            voice: Voice name (must exist in server's voice_config.json)
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to server")

        message = {
            "action": "set_voice",
            "voice": voice
        }

        await self.websocket.send(json.dumps(message))
        logger.info(f"Voice set to: {voice}")

    async def synthesize(self, text: str, stream_audio: bool = True):
        """
        Synthesize text to speech

        Args:
            text: Text to synthesize
            stream_audio: If True, receive and play audio. If False, just send request.

        Returns:
            List of audio chunks (if stream_audio=False), otherwise None
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to server")

        if not text.strip():
            logger.warning("Empty text, skipping synthesis")
            return

        # Send synthesis request
        message = {
            "action": "synthesize",
            "text": text
        }

        await self.websocket.send(json.dumps(message))
        logger.info(f"Sent text: {text[:50]}...")

        # Receive and process audio chunks
        if stream_audio:
            await self._receive_audio()
        else:
            # Return chunks without playing
            chunks = []
            async for chunk in self._receive_audio_chunks():
                chunks.append(chunk)
            return chunks

    async def _receive_audio(self):
        """Receive audio chunks and play them"""
        try:
            async for chunk in self._receive_audio_chunks():
                if self.audio_player:
                    self.audio_player.add_chunk(chunk)

            logger.info("Audio reception complete")

        except Exception as e:
            logger.error(f"Error receiving audio: {e}", exc_info=True)

    async def _receive_audio_chunks(self):
        """
        Generator that yields audio chunks from server

        Yields:
            Audio chunks as bytes
        """
        try:
            while True:
                # Receive message (binary or JSON)
                message = await self.websocket.recv()

                # Check if it's JSON (control message)
                if isinstance(message, str):
                    data = json.loads(message)
                    msg_type = data.get("type")

                    if msg_type == "error":
                        logger.error(f"Server error: {data.get('message')}")
                        break

                    elif msg_type == "voice_changed":
                        logger.info(f"Voice changed: {data.get('voice')}")
                        continue

                    else:
                        logger.debug(f"Control message: {data}")
                        continue

                # Binary message (audio chunk)
                elif isinstance(message, bytes):
                    # Empty message = end of stream
                    if len(message) == 0:
                        logger.debug("End of audio stream")
                        break

                    # Yield audio chunk
                    yield message

        except websockets.exceptions.ConnectionClosed:
            logger.warning("Connection closed during audio reception")

        except Exception as e:
            logger.error(f"Error in audio chunk generator: {e}", exc_info=True)

    async def stop(self):
        """Stop current synthesis"""
        if not self.is_connected:
            return

        message = {"action": "stop"}
        await self.websocket.send(json.dumps(message))
        logger.info("Stop request sent")


async def example_basic():
    """Basic usage example"""
    client = TTSClient("ws://localhost:8765")

    try:
        await client.connect()
        await client.synthesize("Hello! This is a test of the TTS system.")
        await asyncio.sleep(2)  # Wait for audio to finish playing

    finally:
        await client.close()


async def example_llm_streaming():
    """Example: Streaming LLM output"""

    async def fake_llm_generator():
        """Simulates LLM output"""
        sentences = [
            "Hello! This is a simulated LLM response. ",
            "I'm generating text one sentence at a time. ",
            "Each sentence is synthesized as it arrives. ",
            "This creates a more natural, real-time conversation experience."
        ]
        for sentence in sentences:
            await asyncio.sleep(0.5)  # Simulate LLM delay
            yield sentence

    client = TTSClient("ws://localhost:8765")

    try:
        await client.connect()

        # Stream each sentence as it arrives from LLM
        async for text in fake_llm_generator():
            logger.info(f"LLM generated: {text}")
            await client.synthesize(text)
            # Note: Audio will be played as it's received

        # Wait for final audio to finish
        await asyncio.sleep(3)

    finally:
        await client.close()


async def example_voice_selection():
    """Example: Voice selection"""
    client = TTSClient("ws://localhost:8765")

    try:
        await client.connect()

        # Try different voices
        await client.set_voice("default")
        await client.synthesize("This is the default voice.")
        await asyncio.sleep(2)

        await client.set_voice("voice1")
        await client.synthesize("This is voice number one.")
        await asyncio.sleep(2)

    finally:
        await client.close()


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="WebSocket TTS Client")
    parser.add_argument(
        "--server",
        default="ws://localhost:8765",
        help="WebSocket server URL (default: ws://localhost:8765)"
    )
    parser.add_argument(
        "--text",
        default="Hello world! This is a test.",
        help="Text to synthesize"
    )
    parser.add_argument(
        "--voice",
        default="default",
        help="Voice to use (default: default)"
    )
    parser.add_argument(
        "--example",
        choices=["basic", "llm", "voice"],
        help="Run example (basic, llm, or voice)"
    )

    args = parser.parse_args()

    # Run example
    if args.example == "llm":
        asyncio.run(example_llm_streaming())
    elif args.example == "voice":
        asyncio.run(example_voice_selection())
    elif args.example == "basic":
        asyncio.run(example_basic())
    else:
        # Basic synthesis
        async def run():
            client = TTSClient(args.server)
            try:
                await client.connect()
                if args.voice != "default":
                    await client.set_voice(args.voice)
                await client.synthesize(args.text)
                await asyncio.sleep(3)  # Wait for playback
            finally:
                await client.close()

        asyncio.run(run())


if __name__ == "__main__":
    main()
