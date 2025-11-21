import os
import json
import logging
import threading
import time
import torch
import ctranslate2
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from huggingface_hub import snapshot_download

from whisper_live.transcriber.transcriber_faster_whisper import WhisperModel
from whisper_live.backend.base import ServeClientBase


class ServeClientFasterWhisper(ServeClientBase):
    SINGLE_MODEL = None
    SINGLE_MODEL_LOCK = threading.Lock()

    def __init__(
        self,
        websocket,
        task="transcribe",
        device=None,
        language=None,
        client_uid=None,
        model="small.en",
        initial_prompt=None,
        vad_parameters=None,
        use_vad=True,
        single_model=False,
        send_last_n_segments=10,
        no_speech_thresh=0.45,
        clip_audio=False,
        same_output_threshold=7,
        chunking_mode="vad",
        chunk_interval=2.0,
        cache_path="~/.cache/whisper-live/",
        translation_queue=None,
    ):
        """
        Initialize a ServeClient instance.
        The Whisper model is initialized based on the client's language and device availability.
        The transcription thread is started upon initialization. A "SERVER_READY" message is sent
        to the client to indicate that the server is ready.

        Args:
            websocket (WebSocket): The WebSocket connection for the client.
            task (str, optional): The task type, e.g., "transcribe". Defaults to "transcribe".
            device (str, optional): The device type for Whisper, "cuda" or "cpu". Defaults to None.
            language (str, optional): The language for transcription. Defaults to None.
            client_uid (str, optional): A unique identifier for the client. Defaults to None.
            model (str, optional): The whisper model size. Defaults to 'small.en'
            initial_prompt (str, optional): Prompt for whisper inference. Defaults to None.
            single_model (bool, optional): Whether to instantiate a new model for each client connection. Defaults to False.
            send_last_n_segments (int, optional): Number of most recent segments to send to the client. Defaults to 10.
            no_speech_thresh (float, optional): Segments with no speech probability above this threshold will be discarded. Defaults to 0.45.
            clip_audio (bool, optional): Whether to clip audio with no valid segments. Defaults to False.
            same_output_threshold (int, optional): Number of repeated outputs before considering it as a valid segment. Defaults to 10.
            chunking_mode (str, optional): Audio chunking mode - "vad" or "time_based". Defaults to "vad".
            chunk_interval (float, optional): Interval in seconds for time-based chunking. Defaults to 2.0.

        """
        super().__init__(
            client_uid,
            websocket,
            send_last_n_segments,
            no_speech_thresh,
            clip_audio,
            same_output_threshold,
            chunking_mode,
            chunk_interval,
            translation_queue
        )
        self.cache_path = cache_path
        self.model_sizes = [
            "tiny", "tiny.en", "base", "base.en", "small", "small.en",
            "medium", "medium.en", "large-v2", "large-v3", "distil-small.en",
            "distil-medium.en", "distil-large-v2", "distil-large-v3",
            "large-v3-turbo", "turbo"
        ]

        self.model_size_or_path = model
        self.language = "en" if self.model_size_or_path.endswith("en") else language
        self.task = task
        self.initial_prompt = initial_prompt
        self.vad_parameters = vad_parameters or {"onset": 0.5}

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            major, _ = torch.cuda.get_device_capability(device)
            self.compute_type = "float16" if major >= 7 else "float32"
        else:
            self.compute_type = "int8"

        if self.model_size_or_path is None:
            return
        logging.info(f"Using Device={device} with precision {self.compute_type}")
    
        try:
            if single_model:
                if ServeClientFasterWhisper.SINGLE_MODEL is None:
                    self.create_model(device)
                    ServeClientFasterWhisper.SINGLE_MODEL = self.transcriber
                else:
                    self.transcriber = ServeClientFasterWhisper.SINGLE_MODEL
            else:
                self.create_model(device)
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            self.websocket.send(json.dumps({
                "uid": self.client_uid,
                "status": "ERROR",
                "message": f"Failed to load model: {str(self.model_size_or_path)}"
            }))
            self.websocket.close()
            return

        self.use_vad = use_vad

        # threading
        self.trans_thread = threading.Thread(target=self.speech_to_text)
        self.trans_thread.start()
        self.websocket.send(
            json.dumps(
                {
                    "uid": self.client_uid,
                    "message": self.SERVER_READY,
                    "backend": "faster_whisper"
                }
            )
        )

    def create_model(self, device):
        """
        Instantiates a new model, sets it as the transcriber. If model is a huggingface model_id
        then it is automatically converted to ctranslate2(faster_whisper) format.
        """
        model_ref = self.model_size_or_path

        if model_ref in self.model_sizes:
            model_to_load = model_ref
        else:
            logging.info(f"Model not in model_sizes")
            if os.path.isdir(model_ref) and ctranslate2.contains_model(model_ref):
                model_to_load = model_ref
            else:
                local_snapshot = snapshot_download(
                    repo_id = model_ref,
                    repo_type = "model",
                )
                if ctranslate2.contains_model(local_snapshot):
                    model_to_load = local_snapshot
                else:
                    cache_root = os.path.expanduser(os.path.join(self.cache_path, "whisper-ct2-models/"))
                    os.makedirs(cache_root, exist_ok=True)
                    safe_name = model_ref.replace("/", "--")
                    ct2_dir = os.path.join(cache_root, safe_name)

                    if not ctranslate2.contains_model(ct2_dir):
                        logging.info(f"Converting '{model_ref}' to CTranslate2 @ {ct2_dir}")
                        ct2_converter = ctranslate2.converters.TransformersConverter(
                            local_snapshot, 
                            copy_files=["tokenizer.json", "preprocessor_config.json"]
                        )
                        ct2_converter.convert(
                            output_dir=ct2_dir,
                            quantization=self.compute_type,
                            force=False,  # skip if already up-to-date
                        )
                    model_to_load = ct2_dir

        logging.info(f"Loading model: {model_to_load}")
        self.transcriber = WhisperModel(
            model_to_load,
            device=device,
            compute_type=self.compute_type,
            local_files_only=False,
        )

    @classmethod
    def preload_model(cls, model="distil-large-v3", device=None, cache_path="~/.cache/whisper-live/"):
        """
        Preload the Whisper model for single_model mode.

        Args:
            model (str): Model size or path (e.g., "distil-large-v3"). Default is "distil-large-v3".
            device (str, optional): Device to use (auto-detected if None).
            cache_path (str): Cache directory for models. Default is "~/.cache/whisper-live/".

        Returns:
            bool: True if successful, False otherwise.
        """
        import torch
        import os
        from huggingface_hub import snapshot_download

        try:
            with cls.SINGLE_MODEL_LOCK:
                # Check if model already loaded
                if cls.SINGLE_MODEL is not None:
                    logging.info(f"Model already loaded: {model}")
                    return True

                logging.info(f"Preloading model: {model}")

                # Auto-detect device
                if device is None:
                    device = "cuda" if torch.cuda.is_available() else "cpu"

                # Determine compute type based on device
                if device == "cuda":
                    major, _ = torch.cuda.get_device_capability(device)
                    compute_type = "float16" if major >= 7 else "float32"
                else:
                    compute_type = "int8"

                logging.info(f"Using Device={device} with precision {compute_type}")

                # Determine model path
                model_to_load = model
                model_sizes = [
                    "tiny", "tiny.en", "base", "base.en", "small", "small.en",
                    "medium", "medium.en", "large-v2", "large-v3", "distil-small.en",
                    "distil-medium.en", "distil-large-v2", "distil-large-v3",
                    "large-v3-turbo", "turbo"
                ]

                # Handle HuggingFace models
                if model not in model_sizes:
                    if "/" in model:  # HuggingFace model ID
                        cache_path = os.path.expanduser(cache_path)
                        os.makedirs(cache_path, exist_ok=True)

                        try:
                            model_path = snapshot_download(
                                repo_id=model,
                                allow_patterns=["*.json", "*.bin", "*.txt", "*.model", "vocabulary.*"],
                                cache_dir=cache_path
                            )
                            model_to_load = model_path
                            logging.info(f"Downloaded model from HuggingFace: {model}")
                        except Exception as e:
                            logging.error(f"Failed to download model from HuggingFace: {e}")
                            return False

                # Load the model
                cls.SINGLE_MODEL = WhisperModel(
                    model_to_load,
                    device=device,
                    compute_type=compute_type,
                    local_files_only=False,
                )

                logging.info(f"Model preloading complete: {model}")
                return True

        except Exception as e:
            logging.error(f"Failed to preload model: {e}")
            return False

    def set_language(self, info):
        """
        Updates the language attribute based on the detected language information.

        Args:
            info (object): An object containing the detected language and its probability. This object
                        must have at least two attributes: `language`, a string indicating the detected
                        language, and `language_probability`, a float representing the confidence level
                        of the language detection.
        """
        if info.language_probability > 0.5:
            self.language = info.language
            logging.info(f"Detected language {self.language} with probability {info.language_probability}")
            self.websocket.send(json.dumps(
                {"uid": self.client_uid, "language": self.language, "language_prob": info.language_probability}))

    def transcribe_audio(self, input_sample):
        """
        Transcribes the provided audio sample using the configured transcriber instance.

        If the language has not been set, it updates the session's language based on the transcription
        information.

        Args:
            input_sample (np.array): The audio chunk to be transcribed. This should be a NumPy
                                    array representing the audio data.

        Returns:
            The transcription result from the transcriber. The exact format of this result
            depends on the implementation of the `transcriber.transcribe` method but typically
            includes the transcribed text.
        """
        if ServeClientFasterWhisper.SINGLE_MODEL:
            ServeClientFasterWhisper.SINGLE_MODEL_LOCK.acquire()
        result, info = self.transcriber.transcribe(
            input_sample,
            initial_prompt=self.initial_prompt,
            language=self.language,
            task=self.task,
            vad_filter=self.use_vad,
            vad_parameters=self.vad_parameters if self.use_vad else None)
        if ServeClientFasterWhisper.SINGLE_MODEL:
            ServeClientFasterWhisper.SINGLE_MODEL_LOCK.release()

        if self.language is None and info is not None:
            self.set_language(info)
        return result

    def handle_transcription_output(self, result, duration):
        """
        Handle the transcription output, updating the transcript and sending data to the client.

        Args:
            result (str): The result from whisper inference i.e. the list of segments.
            duration (float): Duration of the transcribed audio chunk.
        """
        segments = []
        if len(result):
            self.t_start = None
            last_segment = self.update_segments(result, duration)
            segments = self.prepare_segments(last_segment)

        if len(segments):
            self.send_transcription_to_client(segments)

    def flush_audio_buffer(self):
        """
        Process all remaining buffered audio and finalize pending segments.
        Override base implementation to perform final transcription on FasterWhisper backend.
        """
        logging.info(f"Client {self.client_uid}: FasterWhisper backend flushing buffer...")

        try:
            # Get remaining audio from buffer
            with self.lock:
                if self.frames_np is None or self.frames_np.shape[0] == 0:
                    logging.info(f"Client {self.client_uid}: No audio in buffer to flush")
                    # Still finalize current_out if it exists
                    super().flush_audio_buffer()
                    return

                # Get remaining audio (limit to last 10 seconds for speed optimization)
                samples_take = max(0, (self.timestamp_offset - self.frames_offset) * self.RATE)
                remaining_audio_full = self.frames_np[int(samples_take):].copy()

                # Limit to last 10 seconds to reduce transcription time
                max_samples = int(10.0 * self.RATE)  # 10 seconds
                if len(remaining_audio_full) > max_samples:
                    logging.info(f"Client {self.client_uid}: Limiting final flush to last 10s (was {len(remaining_audio_full) / self.RATE:.1f}s)")
                    remaining_audio = remaining_audio_full[-max_samples:]
                else:
                    remaining_audio = remaining_audio_full

                duration = remaining_audio.shape[0] / self.RATE

            # Only transcribe if we have significant audio (> 0.2 seconds)
            if duration > 0.2:
                logging.info(f"Client {self.client_uid}: Transcribing remaining {duration:.2f}s of audio...")

                # Define transcription task for timeout protection
                def transcribe_with_lock():
                    if ServeClientFasterWhisper.SINGLE_MODEL:
                        ServeClientFasterWhisper.SINGLE_MODEL_LOCK.acquire()
                    try:
                        # Use fastest settings for final flush: beam_size=1 (greedy), no VAD
                        result, info = self.transcriber.transcribe(
                            remaining_audio,
                            initial_prompt=self.initial_prompt,
                            language=self.language,
                            task=self.task,
                            vad_filter=False,  # Disable VAD for speed
                            beam_size=1  # Greedy decoding (fastest)
                        )
                        return result, info
                    finally:
                        if ServeClientFasterWhisper.SINGLE_MODEL:
                            ServeClientFasterWhisper.SINGLE_MODEL_LOCK.release()

                # Run final transcription with timeout protection
                try:
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(transcribe_with_lock)
                        result, info = future.result(timeout=3.0)  # 3 second timeout (aggressive for speed)
                except FuturesTimeoutError:
                    logging.error(f"Client {self.client_uid}: Final transcription timed out after 3s")
                    # Fall back to parent implementation
                    super().flush_audio_buffer()

                    # Send flush completion signal even on timeout
                    try:
                        self.websocket.send(json.dumps({
                            "uid": self.client_uid,
                            "message": "FLUSH_COMPLETE"
                        }))
                        logging.info(f"Client {self.client_uid}: Sent FLUSH_COMPLETE signal (timeout fallback)")
                    except Exception as send_error:
                        logging.error(f"Client {self.client_uid}: Error sending FLUSH_COMPLETE: {send_error}")
                    return

                # Process the final transcription result
                if result is not None:
                    # Convert result to list if it's a generator
                    result_list = list(result)
                    if len(result_list) > 0:
                        logging.info(f"Client {self.client_uid}: Got {len(result_list)} segments from final transcription")
                        last_segment = self.update_segments(result_list, duration)

                        # Mark last segment as completed during final flush
                        if last_segment is not None:
                            last_segment['completed'] = True
                            logging.info(f"Client {self.client_uid}: Marked last segment as completed for final flush")

                        segments = self.prepare_segments(last_segment)

                        # Log final flush segments for debugging
                        if segments:
                            logging.info(f"Client {self.client_uid}: === FINAL FLUSH SEGMENTS ===")
                            for i, seg in enumerate(segments):
                                logging.info(
                                    f"  Segment {i+1}/{len(segments)}: "
                                    f"completed={seg.get('completed', False)} "
                                    f"start={seg.get('start')} end={seg.get('end')} "
                                    f"text='{seg.get('text', '')[:80]}...'"
                                )

                        if segments:
                            self.send_transcription_to_client(segments)
                            logging.info(f"Client {self.client_uid}: Sent {len(segments)} final segments after flush")

                            # Clear current_out to prevent duplicate segment in base flush
                            with self.lock:
                                self.current_out = ""
                                self.prev_out = ""
                            logging.info(f"Client {self.client_uid}: Cleared current_out to prevent duplicate")
                    else:
                        logging.info(f"Client {self.client_uid}: Final transcription returned no segments")

            # Call parent implementation to handle any remaining current_out
            super().flush_audio_buffer()

            logging.info(f"Client {self.client_uid}: Buffer flush complete")

            # Send flush completion signal to client
            try:
                self.websocket.send(json.dumps({
                    "uid": self.client_uid,
                    "message": "FLUSH_COMPLETE"
                }))
                logging.info(f"Client {self.client_uid}: Sent FLUSH_COMPLETE signal")
            except Exception as send_error:
                logging.error(f"Client {self.client_uid}: Error sending FLUSH_COMPLETE: {send_error}")

        except Exception as e:
            logging.error(f"Client {self.client_uid}: Error during FasterWhisper flush: {e}", exc_info=True)
            # Try base implementation as fallback
            try:
                super().flush_audio_buffer()
            except Exception as fallback_error:
                logging.error(f"Client {self.client_uid}: Fallback flush also failed: {fallback_error}")
