"""
Integrated Audio Controller with VAD and STT

Handles audio input/output with:
- Voice Activity Detection (VAD) for interruption
- Audio buffering and transcription
- Real-time speech-to-text with Whisper
- Interrupt capability for AI responses
"""

import asyncio
import queue
import threading
import sounddevice as sd
import numpy as np
from utils.vad_handler import InterruptionHandler

try:
    import whisper

    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False
    print(
        "WARNING: openai-whisper not installed. Install with: pip install openai-whisper"
    )


def play_audio(audio_data: np.ndarray, samplerate: int = 24000) -> None:
    """
    Simple audio playback without interruption handling.

    Args:
        audio_data: Audio samples as numpy array
        samplerate: Sample rate in Hz (default 24000)
    """
    # Ensure audio is float32
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)

    # Clip to [-1, 1] range to prevent distortion
    audio_data = np.clip(audio_data, -1.0, 1.0)

    print("🔊 Playing audio...")
    try:
        sd.play(audio_data, samplerate=samplerate)
        sd.wait()  # Wait for playback to complete
        print("✓ Playback complete")
    except Exception as e:
        print(f"✗ Playback error: {e}")


def play_ai_response(audio_data, handler, samplerate=24000):
    """
    Plays audio data in chunks, with immediate interruption capability.
    `audio_data` is expected to be a numpy array.
    """
    chunk_size = 1024
    print("AI Speaking...")
    for i in range(0, len(audio_data), chunk_size):
        if handler.is_interrupted:
            print("AI Silenced (Interrupted).")
            sd.stop()  # Kill audio output immediately
            break

        chunk = audio_data[i : i + chunk_size]
        # Ensure chunk is float32 for sounddevice
        if chunk.dtype != np.float32:
            chunk = chunk.astype(np.float32)

        sd.play(chunk, samplerate=samplerate)
        sd.wait()  # Wait for the current chunk to finish playing
    if not handler.is_interrupted:
        print("AI Finished Speaking.")


class IntegratedAudioController:
    """
    Manages audio input/output with VAD detection and STT transcription.

    Features:
    - Real-time speech-to-text with Whisper
    - Voice activity detection for interruption
    - Audio buffering and queue management
    - Transcription in background thread
    """

    def __init__(self, handler: InterruptionHandler, whisper_model="base"):
        """
        Initialize the audio controller.

        Args:
            handler: InterruptionHandler instance for VAD
            whisper_model: Whisper model size (tiny, base, small, medium, large)
        """
        self.handler = handler
        self.fs = 16000  # VAD standard rate
        self.output_fs = 24000  # PersonaPlex/Qwen3 rate, though can be resampled
        self.audio_queue = queue.Queue()  # To buffer AI responses (thread-safe)

        # Audio input buffer for transcription
        self.input_buffer = []
        self.buffer_duration = 2.0  # Transcribe every 2 seconds
        self.buffer_size = int(self.fs * self.buffer_duration)

        # Transcription queue
        self.transcription_queue = queue.Queue()

        # Load Whisper model if available
        self.whisper_model = None
        if HAS_WHISPER:
            print(f"Loading Whisper ({whisper_model}) for STT...")
            try:
                self.whisper_model = whisper.load_model(whisper_model, device="cpu")
                print("✓ Whisper model loaded successfully")
            except Exception as e:
                print(f"Failed to load Whisper model: {e}")
        else:
            print("STT will run in stub mode (no transcription)")

        # Thread for background transcription
        self.transcription_thread = None
        self.stop_transcription = threading.Event()

    def _input_callback(self, indata, frames, time, status):
        """
        Continuous stream handling Input (Mic).
        sounddevice callbacks must be synchronous (not async).

        Buffers audio and detects speech for interruption.
        """
        if status:
            print("Input callback status:", status)

        # Convert indata to numpy array and check for speech
        audio_chunk = np.frombuffer(indata, dtype=np.int16).astype(np.float32) / 32768.0

        # Check for speech interruption
        if self.handler.check_for_speech(audio_chunk, sampling_rate=self.fs):
            self.handler.is_interrupted = True
            # In a real system, you might also signal the AI to stop generating

        # Buffer audio for transcription
        self.input_buffer.extend(audio_chunk)

        # When buffer reaches threshold, transcribe
        if len(self.input_buffer) >= self.buffer_size:
            buffer_copy = np.array(
                self.input_buffer[: self.buffer_size], dtype=np.float32
            )
            # Keep overlap for context (shift by half)
            self.input_buffer = self.input_buffer[self.buffer_size // 2 :]

            # Queue for transcription in background thread
            if self.whisper_model is not None:
                self.transcription_queue.put(buffer_copy)

    def _output_callback(self, outdata, frames, time, status):
        """
        Continuous stream handling Output (AI Voice).
        sounddevice callbacks must be synchronous (not async).
        """
        if status:
            print("Output callback status:", status)

        if self.handler.is_interrupted:
            outdata.fill(0)  # Immediately mute the speaker buffer
            return

        try:
            # Try to get audio from the queue, non-blocking
            audio_chunk = self.audio_queue.get_nowait()
            # Ensure chunk matches output size
            if len(audio_chunk) < frames:
                # Pad with zeros if chunk is too short
                audio_chunk = np.pad(audio_chunk, (0, frames - len(audio_chunk)))
            elif len(audio_chunk) > frames:
                # Truncate if chunk is too long
                audio_chunk = audio_chunk[:frames]

            outdata[:, 0] = audio_chunk
        except queue.Empty:
            outdata.fill(0)  # No audio to play, fill with silence

    def _transcription_worker(self):
        """
        Background worker thread for transcription.
        Processes buffered audio chunks with Whisper.
        """
        while not self.stop_transcription.is_set():
            try:
                # Get audio chunk with timeout to allow clean shutdown
                audio_chunk = self.transcription_queue.get(timeout=1.0)
                if audio_chunk is None:
                    break

                # Transcribe with Whisper
                result = self.whisper_model.transcribe(
                    audio_chunk, language="en", verbose=False
                )

                transcription = result.get("text", "").strip()
                if transcription:
                    print(f"User (transcribed): {transcription}")
                    # Put transcription in a queue for the main loop
                    # This would be consumed by the main application

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Transcription error: {e}")

    def start_transcription_worker(self):
        """Start the background transcription thread."""
        if self.whisper_model is None:
            return

        self.stop_transcription.clear()
        self.transcription_thread = threading.Thread(
            target=self._transcription_worker, daemon=True
        )
        self.transcription_thread.start()

    def stop_transcription_worker(self):
        """Stop the background transcription thread."""
        if self.transcription_thread is None:
            return

        self.stop_transcription.set()
        self.transcription_queue.put(None)  # Signal to stop
        if self.transcription_thread.is_alive():
            self.transcription_thread.join(timeout=2.0)

    async def start_session(self):
        """
        Start audio input/output session.

        Creates two streams: one for input (with VAD and transcription),
        one for output (AI responses with interruption capability).
        """
        print("Starting audio session...")

        # Find default input and output devices
        input_device = sd.default.device[0]  # Default input device
        output_device = sd.default.device[1]  # Default output device

        input_device_info = sd.query_devices(input_device)
        output_device_info = sd.query_devices(output_device)

        print(f"Using input device: {input_device_info['name']}")
        print(f"Using output device: {output_device_info['name']}")

        # Start background transcription worker
        self.start_transcription_worker()

        try:
            with sd.Stream(
                callback=self._input_callback,
                channels=1,
                samplerate=self.fs,
                blocksize=512,
                dtype="int16",
                device=input_device,
                kind="input",
            ) as input_stream, sd.Stream(
                callback=self._output_callback,
                channels=1,
                samplerate=self.output_fs,
                blocksize=int(self.output_fs * 0.05),  # 50ms chunks
                dtype="float32",
                device=output_device,
                kind="output",
            ) as output_stream:
                print("Session Active. Speak to the Agent...")
                while True:  # Keep session alive until explicitly stopped
                    await asyncio.sleep(0.1)  # Prevent busy-waiting
        finally:
            self.stop_transcription_worker()

    def queue_audio_output(self, audio_data):
        """
        Queue audio for playback.

        Args:
            audio_data: numpy array of audio samples
        """
        try:
            self.audio_queue.put(audio_data, block=False)
        except queue.Full:
            print("Warning: Audio output queue full, dropping chunk")

    def get_transcription(self, timeout=1.0):
        """
        Get transcribed text (if available).

        Args:
            timeout: Timeout in seconds

        Returns:
            Transcribed text or None
        """
        try:
            # This is a placeholder - in a real implementation,
            # transcriptions would be pushed to a separate queue
            return None
        except queue.Empty:
            return None


if __name__ == "__main__":
    print("IntegratedAudioController with STT integration created.")
    # Example usage:
    # handler = InterruptionHandler()
    # controller = IntegratedAudioController(handler, whisper_model="base")
    # asyncio.run(controller.start_session())
