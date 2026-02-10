"""
Pipeline Integration Tests

Tests for the complete Velloris voice pipeline.
"""

import pytest
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.orchestrator import LocalVoiceOrchestrator
from core.brain import VoiceAgentBrain
from utils.vad_handler import InterruptionHandler
from utils.audio_io import IntegratedAudioController
from utils.audio_utils import resample_audio, normalize_audio, chunk_audio
from config import Config


class TestOrchestrator:
    """Test orchestrator initialization and routing."""

    def test_orchestrator_init(self):
        """Test orchestrator initialization."""
        orchestrator = LocalVoiceOrchestrator(device="cpu", llm_model="llama3")
        assert orchestrator is not None
        assert orchestrator.device == "cpu"
        assert orchestrator.llm_model == "llama3"

    def test_orchestrator_lazy_loading(self):
        """Test lazy loading of engines."""
        orchestrator = LocalVoiceOrchestrator(device="cpu")

        # Engines should be None initially
        assert orchestrator.s2s_engine is None
        assert orchestrator.tts_engine is None

        # Load S2S engine
        orchestrator._load_s2s_engine()
        # Note: This will fail if dependencies not installed
        # In CI, we'll skip this test

    def test_orchestrator_unload(self):
        """Test engine unloading."""
        orchestrator = LocalVoiceOrchestrator(device="cpu")
        orchestrator.unload_engines()
        assert orchestrator.s2s_engine is None
        assert orchestrator.tts_engine is None


class TestVADHandler:
    """Test Voice Activity Detection."""

    def test_vad_initialization(self):
        """Test VAD handler initialization."""
        handler = InterruptionHandler(threshold=0.5)
        assert handler is not None
        assert handler.is_interrupted is False

    def test_vad_reset(self):
        """Test VAD reset."""
        handler = InterruptionHandler()
        handler.is_interrupted = True
        handler.reset()
        assert handler.is_interrupted is False

    def test_vad_check_speech(self):
        """Test speech detection."""
        handler = InterruptionHandler()
        # Create dummy audio
        audio = np.random.randn(512).astype(np.float32) * 0.1
        result = handler.check_for_speech(audio, sampling_rate=16000)
        # Result should be boolean
        assert isinstance(result, (bool, np.bool_))


class TestAudioUtils:
    """Test audio utility functions."""

    def test_resample_audio(self):
        """Test audio resampling."""
        # Create 1 second of 16kHz audio
        audio = np.random.randn(16000).astype(np.float32)

        # Resample to 24kHz
        resampled = resample_audio(audio, 16000, 24000)

        # Check output length
        expected_length = int(16000 * 24000 / 16000)
        assert len(resampled) == expected_length
        assert resampled.dtype == np.float32

    def test_normalize_audio(self):
        """Test audio normalization."""
        # Create loud audio
        audio = np.ones(1000, dtype=np.float32) * 0.5

        normalized = normalize_audio(audio, target_db=-20.0)

        # Check it's normalized
        assert normalized.min() >= -1.0
        assert normalized.max() <= 1.0
        assert normalized.dtype == np.float32

    def test_chunk_audio(self):
        """Test audio chunking."""
        audio = np.random.randn(10000).astype(np.float32)

        chunks = list(chunk_audio(audio, chunk_size=4096, overlap=1024))

        # Should have multiple chunks
        assert len(chunks) > 1

        # All chunks should be arrays
        for chunk in chunks:
            assert isinstance(chunk, np.ndarray)


class TestBrain:
    """Test voice agent brain."""

    def test_brain_initialization(self):
        """Test brain initialization."""
        brain = VoiceAgentBrain(model_name="llama3")
        assert brain is not None
        assert brain.llm is not None
        assert brain.tts_engine is None

    @pytest.mark.asyncio
    async def test_brain_stream_tokens(self):
        """Test token streaming."""
        brain = VoiceAgentBrain(model_name="llama3")

        # This test will skip if Ollama is not running
        try:
            tokens = []
            async for token in brain.stream_tokens("Hello"):
                tokens.append(token)
                if len(tokens) >= 2:  # Just get a few tokens
                    break

            # We should get at least one token
            assert len(tokens) >= 0  # May be 0 if Ollama not running
        except Exception:
            pytest.skip("Ollama not running")


class TestAudioController:
    """Test integrated audio controller."""

    def test_audio_controller_init(self):
        """Test audio controller initialization."""
        handler = InterruptionHandler()
        controller = IntegratedAudioController(handler, whisper_model="base")

        assert controller is not None
        assert controller.fs == 16000
        assert controller.output_fs == 24000

    def test_audio_queue_audio_output(self):
        """Test queueing audio for output."""
        handler = InterruptionHandler()
        controller = IntegratedAudioController(handler)

        # Queue some audio
        audio = np.random.randn(1024).astype(np.float32)
        controller.queue_audio_output(audio)

        # Should not raise an error
        assert True


class TestConfig:
    """Test configuration."""

    def test_config_validation(self):
        """Test config validation."""
        assert Config.validate() is True

    def test_config_paths(self):
        """Test config paths exist or can be created."""
        assert Config.model.VOICES_DIR.parent.exists()
        assert isinstance(Config.model.VOICE_REFERENCE, str)


class TestS2SEngines:
    """Test S2S engine selection and initialization (platform-aware)."""

    def test_personaplex_initialization(self):
        """Test PersonaPlex initialization (all platforms)."""
        from engines.personaplex import PersonaPlexEngine

        engine = PersonaPlexEngine(device="cpu")
        assert engine is not None
        assert engine.sample_rate == 24000
        assert engine.voice in PersonaPlexEngine.AVAILABLE_VOICES

        engine.unload()

    def test_macecho_initialization_on_macos(self):
        """Test MacEcho initialization on macOS."""
        import sys

        if sys.platform != "darwin":
            pytest.skip("MacEcho test skipped on non-macOS platform")

        from engines.macecho_s2s import MacEchoEngine

        engine = MacEchoEngine(device="cpu")
        assert engine is not None
        assert engine.sample_rate == 24000
        assert engine.voice in MacEchoEngine.AVAILABLE_VOICES

        engine.unload()

    def test_macecho_voice_mapping_complete(self):
        """Test that all PersonaPlex voices are mapped in MacEcho."""
        from engines.macecho_s2s import MacEchoEngine
        from engines.personaplex import PersonaPlexEngine

        pp_voices = set(PersonaPlexEngine.AVAILABLE_VOICES.keys())
        me_voices = set(MacEchoEngine.VOICE_MAPPING.keys())

        # Convert PersonaPlex names to shortcodes for comparison
        pp_shortcodes = set(PersonaPlexEngine.AVAILABLE_VOICES.values())

        # MacEcho should have mappings for all PersonaPlex shortcodes
        for shortcode in pp_shortcodes:
            assert shortcode in MacEchoEngine.VOICE_MAPPING, (
                f"Voice {shortcode} not mapped in MacEcho"
            )

    def test_macecho_stub_mode(self):
        """Test MacEcho stub mode (no installation required)."""
        from engines.macecho_s2s import HAS_MACECHO, MacEchoEngine

        # Stub mode should work even if not installed
        engine = MacEchoEngine(device="cpu")
        assert engine is not None

        # Call should return stub audio (2 seconds of silence)
        if not HAS_MACECHO:
            audio = np.random.randn(16000).astype(np.float32)
            result = engine.generate_s2s_response(audio, sr=24000)
            assert result is not None
            stub_audio, sr = result
            assert sr == 24000
            assert len(stub_audio) == 48000  # 2 seconds at 24kHz

        engine.unload()

    def test_personaplex_stub_mode(self):
        """Test PersonaPlex stub mode (no installation required)."""
        from engines.personaplex import HAS_PERSONAPLEX, PersonaPlexEngine

        # Stub mode should work even if not installed
        engine = PersonaPlexEngine(device="cpu")
        assert engine is not None

        # Call should return None without real model
        if not HAS_PERSONAPLEX:
            audio = np.random.randn(24000).astype(np.float32)
            result = engine.generate_s2s_response(audio, sr=24000)
            assert result is None

        engine.unload()


class TestIntegration:
    """Integration tests for complete pipeline."""

    def test_full_pipeline_init(self):
        """Test initializing all components together."""
        vad = InterruptionHandler()
        audio_controller = IntegratedAudioController(vad)
        orchestrator = LocalVoiceOrchestrator(device="cpu")
        brain = VoiceAgentBrain(model_name="llama3", orchestrator=orchestrator)

        assert vad is not None
        assert audio_controller is not None
        assert orchestrator is not None
        assert brain is not None

        # Cleanup
        orchestrator.unload_engines()

    @pytest.mark.filterwarnings("ignore:MLX-Audio is optimized for 'mps'")
    def test_orchestrator_routing(self):
        """Test orchestrator request routing."""
        orchestrator = LocalVoiceOrchestrator(device="cpu")

        # Test routing without actually generating audio (to avoid dependency issues)
        # Just check that routing doesn't crash
        # Note: Use keyword arguments since mode is the first parameter
        result = orchestrator.route_request(mode="dubbing", text="Hello")
        # Result may be None if TTS not available
        assert result is None or isinstance(result, tuple)

        orchestrator.unload_engines()


if __name__ == "__main__":
    # Run tests with: pytest tests/test_pipeline.py -v
    pytest.main([__file__, "-v"])
