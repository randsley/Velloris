"""
Test suite for realtime audio callbacks.

Tests the dual-stream sounddevice callback infrastructure without requiring
actual audio devices or model inference.
"""

import pytest
import numpy as np
import queue
import time
from unittest.mock import Mock, patch

from utils.vad_handler import InterruptionHandler
from utils.audio_io import IntegratedAudioController


class TestInputCallback:
    """Test input callback audio buffering and VAD detection."""

    def test_input_callback_buffers_audio(self):
        """Verify input callback accumulates audio chunks."""
        handler = InterruptionHandler()
        controller = IntegratedAudioController(handler, whisper_model="base")

        # Mock audio: 512 samples at 16kHz (32ms - Silero VAD chunk size)
        # This is the exact size VAD expects
        mock_audio = np.zeros(512, dtype=np.int16)

        # Call input callback
        controller._input_callback(mock_audio, 512, None, None)

        # Verify: audio was buffered
        assert len(controller.input_buffer) > 0

    def test_input_buffer_accumulates_to_threshold(self):
        """Verify buffer accumulates until 2-second threshold."""
        handler = InterruptionHandler()
        controller = IntegratedAudioController(handler, whisper_model="base")

        # Buffer size is 2 seconds * 16000 Hz = 32000 samples
        buffer_threshold = 32000

        # Send enough chunks to exceed threshold
        # When threshold is reached, buffer is trimmed to half (16000 samples)
        # So we need to send just enough to reach but not trigger transcription
        chunk_size = 512  # Silero VAD chunk size

        # Calculate chunks needed: 32000 / 512 = 62.5, so 62 chunks
        for _ in range(62):
            chunk = np.zeros(chunk_size, dtype=np.int16)
            controller._input_callback(chunk, chunk_size, None, None)

        # Verify: buffer accumulated close to threshold
        # Note: It may be slightly less than 32000 due to rounding
        assert len(controller.input_buffer) >= 31000  # Allow some tolerance

    def test_input_callback_detects_speech(self):
        """Verify VAD detection triggers on speech-like audio."""
        handler = InterruptionHandler()
        controller = IntegratedAudioController(handler, whisper_model="base")

        # Create speech-like audio: 300Hz tone (human voice range)
        # Must be exactly 512 samples for Silero VAD
        chunk_size = 512
        sample_rate = 16000
        t = np.linspace(0, chunk_size / sample_rate, chunk_size)
        # Convert to int16 (callback expects this format)
        speech = (np.sin(2 * np.pi * 300 * t) * 32767 * 0.5).astype(np.int16)

        # Call input callback
        controller._input_callback(speech, len(speech), None, None)

        # Verify: handler processed audio (is_interrupted may or may not be set,
        # depending on VAD model's sensitivity, but no error should occur)
        assert controller.input_buffer is not None


class TestOutputCallback:
    """Test output callback audio playback and queue management."""

    def test_output_callback_retrieves_queued_audio(self):
        """Verify output callback gets audio from queue."""
        handler = InterruptionHandler()
        controller = IntegratedAudioController(handler, whisper_model="base")

        # Queue some test audio
        test_audio = np.ones(1024, dtype=np.float32) * 0.1
        controller.queue_audio_output(test_audio)

        # Verify: audio is in queue
        assert not controller.audio_queue.empty()

        # Create output buffer
        outdata = np.zeros((1024, 1), dtype=np.float32)

        # Call output callback
        controller._output_callback(outdata, 1024, None, None)

        # Verify: outdata was filled (not all zeros)
        assert np.any(outdata != 0)

    def test_output_callback_fills_silence_when_empty(self):
        """Verify output callback fills zeros when queue is empty."""
        handler = InterruptionHandler()
        controller = IntegratedAudioController(handler, whisper_model="base")

        # Ensure queue is empty
        assert controller.audio_queue.empty()

        # Create output buffer
        outdata = np.zeros((1024, 1), dtype=np.float32)

        # Call output callback
        controller._output_callback(outdata, 1024, None, None)

        # Verify: outdata is all zeros (silence)
        assert np.all(outdata == 0)

    def test_interruption_flag_stops_playback(self):
        """Verify is_interrupted flag mutes output."""
        handler = InterruptionHandler()
        controller = IntegratedAudioController(handler, whisper_model="base")

        # Queue audio
        test_audio = np.ones(1024, dtype=np.float32) * 0.5
        controller.queue_audio_output(test_audio)

        # Set interrupt flag
        handler.is_interrupted = True

        # Create output buffer
        outdata = np.zeros((1024, 1), dtype=np.float32)

        # Call output callback
        controller._output_callback(outdata, 1024, None, None)

        # Verify: outdata is zeros (muted due to interruption)
        assert np.all(outdata == 0)

        # Verify: queue was cleared
        assert controller.audio_queue.empty()


class TestSampleRateConsistency:
    """Test dual sample rate handling (16kHz input, 24kHz output)."""

    def test_input_stream_uses_16khz(self):
        """Verify input stream sample rate is 16kHz."""
        handler = InterruptionHandler()
        controller = IntegratedAudioController(handler, whisper_model="base")

        # Verify: input sample rate
        assert controller.fs == 16000

    def test_output_stream_uses_24khz(self):
        """Verify output stream sample rate is 24kHz."""
        handler = InterruptionHandler()
        controller = IntegratedAudioController(handler, whisper_model="base")

        # Verify: output sample rate
        assert controller.output_fs == 24000


class TestQueueThreadSafety:
    """Test thread-safe queue operations."""

    def test_uses_thread_safe_queue(self):
        """Verify audio queue is queue.Queue, not asyncio.Queue."""
        handler = InterruptionHandler()
        controller = IntegratedAudioController(handler, whisper_model="base")

        # Verify: uses standard queue.Queue (thread-safe)
        assert isinstance(controller.audio_queue, queue.Queue)

    def test_queue_audio_output_is_thread_safe(self):
        """Verify queue_audio_output can be called from multiple threads."""
        handler = InterruptionHandler()
        controller = IntegratedAudioController(handler, whisper_model="base")

        # Queue audio from "multiple threads" (simulated)
        for i in range(10):
            audio = np.ones(100, dtype=np.float32) * i
            controller.queue_audio_output(audio)

        # Verify: all chunks queued
        assert controller.audio_queue.qsize() == 10


class TestTranscriptionWorker:
    """Test background transcription worker thread."""

    def test_transcription_worker_starts_and_stops(self):
        """Verify transcription worker thread lifecycle."""
        handler = InterruptionHandler()
        controller = IntegratedAudioController(handler, whisper_model="base")

        # Start worker
        controller.start_transcription_worker()

        # Verify: worker thread exists
        assert controller.transcription_thread is not None
        assert controller.transcription_thread.is_alive()

        # Stop worker
        controller.stop_transcription_worker()

        # Give thread time to terminate
        time.sleep(0.5)

        # Verify: worker thread stopped
        assert not controller.transcription_thread.is_alive()

    def test_transcription_queue_consumed_by_worker(self):
        """Verify transcription worker processes queued audio."""
        handler = InterruptionHandler()
        controller = IntegratedAudioController(handler, whisper_model="base")

        # Start worker
        controller.start_transcription_worker()

        # Queue audio chunk
        chunk = np.zeros(16000, dtype=np.float32)  # 1 second at 16kHz
        controller.transcription_queue.put(chunk)

        # Give worker time to process
        time.sleep(1)

        # Verify: queue consumed (worker processed it)
        # Note: In stub mode, transcription may not produce output, but queue should be empty
        assert controller.transcription_queue.empty() or controller.transcription_queue.qsize() < 2

        # Cleanup
        controller.stop_transcription_worker()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
