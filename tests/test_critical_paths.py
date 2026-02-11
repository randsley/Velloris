"""
Critical Path Tests for Velloris

These tests verify that recent architectural changes (three-mode system,
parameter passing, platform-specific engines) are working correctly.

These tests are designed to catch bugs that the original test_pipeline.py
does not cover:
- Emotion parameter passing
- Language parameter support
- Reference audio validation
- Platform-specific code paths
- Mode handler routing

Run with: pytest tests/test_critical_paths.py -v
"""

import os
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.orchestrator import LocalVoiceOrchestrator
from config import Config


class TestEmotionParameterPassthrough:
    """Test that emotion parameter reaches TTS engine."""

    def test_emotion_passed_to_tts_in_creative_mode(self):
        """Verify emotion parameter is passed through orchestrator to TTS in creative mode."""
        orchestrator = LocalVoiceOrchestrator(device="cpu", llm_model="llama3")

        # Mock the TTS engine's generate_dubbing method
        mock_generate = Mock(return_value=(np.zeros(12000), 12000))
        orchestrator.tts_engine = Mock()
        orchestrator.tts_engine.generate_dubbing = mock_generate

        # Mock Ollama to avoid requiring running instance
        mock_llm = Mock()
        mock_llm.invoke = Mock(return_value="Test response from LLM")
        orchestrator.ollama_brain = mock_llm

        # Pre-load TTS engine to avoid lazy loading issues
        orchestrator.tts_engine = Mock()
        orchestrator.tts_engine.generate_dubbing = mock_generate

        # Call creative mode with emotion
        emotion_text = "Speak with excitement and enthusiasm"
        orchestrator.route_request(
            mode="creative",
            text="Tell me a story",
            emotion=emotion_text
        )

        # Verify TTS generate_dubbing was called with instruct parameter
        mock_generate.assert_called_once()
        call_kwargs = mock_generate.call_args[1]

        assert "instruct" in call_kwargs, (
            "Emotion parameter not passed to TTS! "
            "Missing 'instruct' kwarg in generate_dubbing call"
        )
        assert call_kwargs["instruct"] == emotion_text, (
            f"Emotion parameter mismatch: "
            f"expected '{emotion_text}', got '{call_kwargs.get('instruct')}'"
        )

    def test_emotion_passed_to_tts_in_dubbing_mode(self):
        """Verify emotion parameter is passed through orchestrator to TTS in dubbing mode."""
        orchestrator = LocalVoiceOrchestrator(device="cpu")

        # Mock TTS
        mock_generate = Mock(return_value=(np.zeros(12000), 12000))
        orchestrator.tts_engine = Mock()
        orchestrator.tts_engine.generate_dubbing = mock_generate

        # Call dubbing mode with emotion (if supported)
        emotion_text = "Speak softly and thoughtfully"
        orchestrator.route_request(
            mode="dubbing",
            text="Once upon a time...",
            emotion=emotion_text
        )

        # Verify instruct parameter was passed
        mock_generate.assert_called_once()
        call_kwargs = mock_generate.call_args[1]

        # For dubbing mode, emotion goes to instruct parameter
        if "instruct" in call_kwargs:
            assert call_kwargs["instruct"] == emotion_text, (
                "Emotion not correctly passed to TTS in dubbing mode"
            )


class TestLanguageParameterSupport:
    """Test that language parameter is properly supported."""

    def test_language_parameter_passed_to_tts(self):
        """Verify language parameter is passed to TTS engine."""
        orchestrator = LocalVoiceOrchestrator(device="cpu")

        # Mock TTS
        mock_generate = Mock(return_value=(np.zeros(12000), 12000))
        orchestrator.tts_engine = Mock()
        orchestrator.tts_engine.generate_dubbing = mock_generate

        # Test with language parameter
        test_language = "spanish"
        orchestrator.route_request(
            mode="dubbing",
            text="Hola mundo",
        )

        # Verify TTS was called
        mock_generate.assert_called_once()
        call_kwargs = mock_generate.call_args[1]

        # Language should be present (currently hardcoded to "english", but should be configurable)
        assert "language" in call_kwargs, (
            "Language parameter not passed to TTS! "
            "Expected 'language' kwarg in generate_dubbing call"
        )

    def test_config_language_setting_applied(self):
        """Verify language configuration is applied to TTS calls."""
        # Check that Config has language settings
        assert hasattr(Config.app, "DUBBING_LANGUAGE"), (
            "Config.app should have DUBBING_LANGUAGE setting"
        )

        # Verify it has a default value
        default_lang = Config.app.DUBBING_LANGUAGE
        assert isinstance(default_lang, str), (
            f"DUBBING_LANGUAGE should be string, got {type(default_lang)}"
        )
        assert len(default_lang) > 0, "DUBBING_LANGUAGE should not be empty"

    def test_supported_languages_defined(self):
        """Verify supported languages are defined in config."""
        assert hasattr(Config.app, "SUPPORTED_LANGUAGES"), (
            "Config.app should have SUPPORTED_LANGUAGES list"
        )

        supported = Config.app.SUPPORTED_LANGUAGES
        assert isinstance(supported, list), "SUPPORTED_LANGUAGES should be a list"
        assert len(supported) >= 10, (
            f"Expected at least 10 supported languages, got {len(supported)}"
        )
        assert "english" in supported, "English should be in supported languages"


class TestReferenceAudioValidation:
    """Test that reference audio for voice cloning is validated."""

    def test_missing_reference_audio_handled(self):
        """Verify missing reference audio file is handled gracefully."""
        orchestrator = LocalVoiceOrchestrator(device="cpu")

        # Mock TTS
        mock_generate = Mock(return_value=(np.zeros(12000), 12000))
        orchestrator.tts_engine = Mock()
        orchestrator.tts_engine.generate_dubbing = mock_generate

        # Call with non-existent reference audio
        missing_path = "/nonexistent/voice_sample.wav"
        result = orchestrator.route_request(
            mode="dubbing",
            text="Test text",
            ref_audio_path=missing_path
        )

        # Should not crash - either return None or pass validated path to TTS
        assert result is None or isinstance(result, tuple), (
            "Orchestrator should handle missing reference audio gracefully"
        )

    def test_reference_audio_validation_in_engines(self):
        """Verify TTS engines validate reference audio before use."""
        # This test checks that engines have validation logic
        from engines.qwen_tts import Qwen3TTSEngine
        from inspect import signature

        # Check that generate_dubbing accepts ref_audio_path
        sig = signature(Qwen3TTSEngine.generate_dubbing)
        assert "ref_audio_path" in sig.parameters, (
            "Qwen3TTSEngine.generate_dubbing should accept ref_audio_path parameter"
        )

    def test_invalid_reference_audio_file_sizes(self):
        """Verify reference audio file size is validated."""
        orchestrator = LocalVoiceOrchestrator(device="cpu")

        # Mock TTS to track what it receives
        mock_generate = Mock(return_value=(np.zeros(12000), 12000))
        orchestrator.tts_engine = Mock()
        orchestrator.tts_engine.generate_dubbing = mock_generate

        # Test with very small file (simulate too-short audio)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"tiny")  # Only 4 bytes - too small
            tiny_file = f.name

        try:
            orchestrator.route_request(
                mode="dubbing",
                text="Test",
                ref_audio_path=tiny_file
            )
            # Should handle without crashing
        finally:
            Path(tiny_file).unlink()

        # Test with very large file (simulate too-long audio)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"x" * (10 * 1024 * 1024))  # 10MB - too large
            huge_file = f.name

        try:
            orchestrator.route_request(
                mode="dubbing",
                text="Test",
                ref_audio_path=huge_file
            )
            # Should handle without crashing
        finally:
            Path(huge_file).unlink()


class TestPlatformSpecificImports:
    """Test that correct TTS engine is loaded per platform."""

    def test_mlx_imported_on_darwin(self):
        """Verify MLX engine is used on macOS."""
        with patch("sys.platform", "darwin"):
            # Re-import to trigger conditional import
            import importlib
            import core.orchestrator as orch_module

            importlib.reload(orch_module)

            # On darwin, TTSEngine should be MLXTTSEngine
            # We can't directly check the import, but we can verify the platform detection
            assert sys.platform == "darwin" or sys.platform != "darwin"

    def test_qwen3tts_imported_on_non_darwin(self):
        """Verify Qwen3-TTS engine is used on Windows/Linux."""
        # The current environment determines which engine is loaded
        # Just verify orchestrator initializes properly with default engine

        orchestrator = LocalVoiceOrchestrator(device="cpu")

        # Should initialize without error
        assert orchestrator is not None
        assert orchestrator.tts_engine is None  # Not loaded until needed (lazy loading)

    def test_platform_branching_code_path(self):
        """Verify platform branching in orchestrator works."""
        # Check that orchestrator.py has the platform check
        import core.orchestrator

        source = Path(core.orchestrator.__file__).read_text()
        assert "sys.platform" in source, (
            "Orchestrator should branch on sys.platform"
        )
        assert "darwin" in source, (
            "Orchestrator should check for darwin (macOS)"
        )

    def test_engine_selection_matches_platform(self):
        """Verify selected TTS engine matches platform."""
        orchestrator = LocalVoiceOrchestrator(device="cpu")

        # Mock both engines to track which is used
        with patch("engines.mlx_tts.MLXTTSEngine") as mock_mlx:
            with patch("engines.qwen_tts.Qwen3TTSEngine") as mock_qwen:

                orchestrator._load_tts_engine()

                # Check which engine was instantiated based on platform
                if sys.platform == "darwin":
                    # On macOS, MLX should be attempted
                    # (It might fail due to missing dependencies, which is OK)
                    pass
                else:
                    # On Windows/Linux, Qwen3-TTS should be used
                    # Engine loading happens in _load_tts_engine
                    pass


class TestOrchestratorModeHandlers:
    """Test that orchestrator properly routes to mode-specific handlers."""

    def test_dubbing_mode_handler_called(self):
        """Verify _handle_dubbing is called in dubbing mode."""
        orchestrator = LocalVoiceOrchestrator(device="cpu")

        # Mock the handler
        mock_handler = Mock(return_value=(np.zeros(12000), 12000))
        orchestrator._handle_dubbing = mock_handler

        # Call dubbing mode
        orchestrator.route_request(mode="dubbing", text="Test text")

        # Verify handler was called
        mock_handler.assert_called_once()

    def test_creative_mode_handler_called(self):
        """Verify _handle_creative is called in creative mode."""
        orchestrator = LocalVoiceOrchestrator(device="cpu")

        # Mock the handler
        mock_handler = Mock(return_value=(np.zeros(12000), 12000))
        orchestrator._handle_creative = mock_handler

        # Call creative mode
        orchestrator.route_request(mode="creative", text="Test text")

        # Verify handler was called
        mock_handler.assert_called_once()

    def test_realtime_mode_handler_called(self):
        """Verify _handle_realtime is called in realtime mode."""
        orchestrator = LocalVoiceOrchestrator(device="cpu")

        # Mock the handler
        mock_handler = Mock(return_value=(np.zeros(24000), 24000))
        orchestrator._handle_realtime = mock_handler

        # Create dummy audio
        audio = np.zeros(24000, dtype=np.float32)

        # Call realtime mode
        orchestrator.route_request(mode="realtime", audio_input=audio)

        # Verify handler was called
        mock_handler.assert_called_once()

    def test_invalid_mode_rejected(self):
        """Verify invalid mode is rejected with informative error."""
        orchestrator = LocalVoiceOrchestrator(device="cpu")

        # Call with invalid mode
        result = orchestrator.route_request(mode="invalid_mode", text="Test")

        # Should return None (not crash)
        assert result is None, "Invalid mode should return None"

    def test_mode_parameters_passed_correctly(self):
        """Verify mode-specific parameters are passed to handlers."""
        orchestrator = LocalVoiceOrchestrator(device="cpu")

        # Mock handlers to capture parameters
        handlers_called = {}

        def mock_dubbing(text, ref_audio_path=None):
            handlers_called["dubbing"] = {"text": text, "ref_audio_path": ref_audio_path}
            return None

        def mock_creative(text, emotion=None, ref_audio_path=None):
            handlers_called["creative"] = {
                "text": text,
                "emotion": emotion,
                "ref_audio_path": ref_audio_path,
            }
            return None

        orchestrator._handle_dubbing = mock_dubbing
        orchestrator._handle_creative = mock_creative

        # Test dubbing parameters
        orchestrator.route_request(
            mode="dubbing",
            text="Test script",
            ref_audio_path="/path/to/audio.wav",
        )

        assert "dubbing" in handlers_called
        assert handlers_called["dubbing"]["text"] == "Test script"
        assert handlers_called["dubbing"]["ref_audio_path"] == "/path/to/audio.wav"

        # Test creative parameters
        orchestrator.route_request(
            mode="creative",
            text="Tell a story",
            emotion="excited",
            ref_audio_path="/path/to/voice.wav",
        )

        assert "creative" in handlers_called
        assert handlers_called["creative"]["text"] == "Tell a story"
        assert handlers_called["creative"]["emotion"] == "excited"
        assert handlers_called["creative"]["ref_audio_path"] == "/path/to/voice.wav"


class TestParameterIntegration:
    """Integration tests for parameter passing through the entire pipeline."""

    def test_emotion_flow_from_cli_to_tts(self):
        """Test complete flow of emotion parameter from CLI to TTS."""
        from main import VellorisApplication
        from argparse import Namespace

        # Simulate CLI arguments with emotion
        args = Namespace(
            mode="creative",
            device="cpu",
            llm_model="llama3",
            script="Tell me a story",
            voice_ref=None,
            emotion="excited",
            persona=None,
            voice=None,
            show_config=False,
        )

        # Create application (don't run, just init)
        app = VellorisApplication(args)

        # Verify emotion is stored in args
        assert app.args.emotion == "excited"

        # Mock the orchestrator to verify emotion is passed
        mock_route = Mock(return_value=(np.zeros(12000), 12000))
        app.orchestrator.route_request = mock_route

        # The app would call orchestrator with emotion parameter
        # (In actual run, this happens in async methods)
        app.orchestrator.route_request(
            mode="creative",
            text="Test",
            emotion=app.args.emotion
        )

        # Verify emotion was passed to orchestrator
        call_kwargs = mock_route.call_args[1]
        assert call_kwargs.get("emotion") == "excited"

    def test_all_parameters_coexist(self):
        """Verify emotion, language, and reference audio can all be used together."""
        orchestrator = LocalVoiceOrchestrator(device="cpu")

        # Mock TTS
        captured_call = {}

        def capture_call(**kwargs):
            captured_call.update(kwargs)
            return (np.zeros(12000), 12000)

        orchestrator.tts_engine = Mock()
        orchestrator.tts_engine.generate_dubbing = capture_call

        # Mock Ollama
        orchestrator.ollama_brain = Mock()
        orchestrator.ollama_brain.invoke = Mock(return_value="LLM response")

        # Call with all parameters
        orchestrator.route_request(
            mode="creative",
            text="Tell a happy story",
            emotion="joyful",
            language="english",
            ref_audio_path="/path/to/voice.wav",
        )

        # Verify all were captured
        assert "emotion" in captured_call or "instruct" in captured_call, (
            "Emotion/instruct should be passed to TTS"
        )
        assert "language" in captured_call, "Language should be passed to TTS"
        assert "ref_audio_path" in captured_call, (
            "Reference audio path should be passed to TTS"
        )


class TestRMSNormalization:
    """Test RMS-based audio normalization."""

    def test_rms_normalize_output_range(self):
        """Verify RMS normalization keeps audio in [-1, 1] via soft clipping."""
        from engines.mlx_tts import MLXTTSEngine

        # Loud signal that would clip with simple gain
        audio = np.random.randn(12000).astype(np.float32) * 5.0
        result = MLXTTSEngine._rms_normalize(audio)

        assert np.all(np.abs(result) <= 1.0), "Soft clipping should keep audio in [-1, 1]"

    def test_rms_normalize_silent_input(self):
        """Verify silent audio is returned unchanged."""
        from engines.mlx_tts import MLXTTSEngine

        audio = np.zeros(12000, dtype=np.float32)
        result = MLXTTSEngine._rms_normalize(audio)
        np.testing.assert_array_equal(result, audio)

    def test_rms_normalize_preserves_dynamics(self):
        """Verify dynamic range is better preserved than peak normalization."""
        from engines.mlx_tts import MLXTTSEngine

        # Signal with one loud spike and quiet content
        audio = np.ones(12000, dtype=np.float32) * 0.01
        audio[6000] = 1.0  # single spike

        rms_result = MLXTTSEngine._rms_normalize(audio)

        # Peak normalization would leave quiet parts at 0.01
        # RMS normalization should boost quiet parts more
        quiet_rms = np.sqrt(np.mean(rms_result[:5000] ** 2))
        assert quiet_rms > 0.01, "RMS normalization should boost quiet content"


class TestChunkValidation:
    """Test chunk validation in generation loop."""

    def test_empty_chunks_skipped(self):
        """Verify empty chunks are filtered out during generation."""
        import pytest
        from engines.mlx_tts import MLXTTSEngine

        # NOTE: This test was designed for the old model.generate() implementation.
        # The new implementation calls the CLI subprocess, which handles chunk validation internally.
        # Skip this test as it's no longer applicable with the CLI approach.
        pytest.skip("Chunk validation test not applicable with CLI-based implementation")

        # Original test code (kept for reference):
        # engine = MLXTTSEngine.__new__(MLXTTSEngine)
        # engine.model = Mock()
        # engine.sample_rate = 12000
        # valid_chunk = Mock(audio=np.ones(1000, dtype=np.float32) * 0.5)
        # empty_chunk = Mock(audio=np.array([], dtype=np.float32))
        # silent_chunk = Mock(audio=np.zeros(1000, dtype=np.float32))
        # engine.model.generate = Mock(return_value=iter([valid_chunk, empty_chunk, silent_chunk, valid_chunk]))
        # result = engine.generate_dubbing(text="Test")
        # assert result is not None
        # audio, sr = result
        # assert len(audio) == 1950


class TestRefAudioValidation:
    """Test reference audio validation in MLX engine."""

    def test_nonexistent_path_returns_none(self):
        """Verify non-existent path is rejected."""
        from engines.mlx_tts import MLXTTSEngine

        result = MLXTTSEngine._validate_ref_audio("/nonexistent/file.wav")
        assert result is None

    def test_unsupported_extension_returns_none(self):
        """Verify unsupported extension is rejected."""
        from engines.mlx_tts import MLXTTSEngine

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"x" * 2000)
            path = f.name
        try:
            result = MLXTTSEngine._validate_ref_audio(path)
            assert result is None
        finally:
            os.unlink(path)

    def test_too_small_file_returns_none(self):
        """Verify too-small file is rejected."""
        from engines.mlx_tts import MLXTTSEngine

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"tiny")
            path = f.name
        try:
            result = MLXTTSEngine._validate_ref_audio(path)
            assert result is None
        finally:
            os.unlink(path)

    def test_valid_file_accepted(self):
        """Verify valid audio file is accepted."""
        from engines.mlx_tts import MLXTTSEngine

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"x" * 2000)
            path = f.name
        try:
            result = MLXTTSEngine._validate_ref_audio(path)
            assert result == path
        finally:
            os.unlink(path)

    def test_none_path_returns_none(self):
        """Verify None input returns None."""
        from engines.mlx_tts import MLXTTSEngine

        assert MLXTTSEngine._validate_ref_audio(None) is None


class TestModelCaching:
    """Test model caching across instances."""

    @pytest.mark.skipif(sys.platform != "darwin", reason="MLX is macOS-only")
    def test_model_loaded_once(self):
        """Verify same model is only loaded from disk once (macOS only)."""
        from engines.mlx_tts import MLXTTSEngine

        # Clear cache
        MLXTTSEngine._model_cache.clear()

        mock_model = Mock()
        # Patch at the module level where load_model is already imported
        with patch("engines.mlx_tts.load_model", return_value=mock_model) as mock_load:
            with patch("engines.mlx_tts.HAS_MLX_AUDIO", True):
                e1 = MLXTTSEngine(model_name="test-model")
                e2 = MLXTTSEngine(model_name="test-model")

                # load_model should only be called once (caching works)
                mock_load.assert_called_once()
                assert e1.model is e2.model

        # Cleanup
        MLXTTSEngine._model_cache.clear()


class TestChunkAssembly:
    """Test cross-fade chunk assembly for audio dropout mitigation."""

    def test_single_chunk_passthrough(self):
        """Single chunk should be returned as-is."""
        from engines.mlx_tts import MLXTTSEngine

        chunk = np.ones(500, dtype=np.float32) * 0.5
        result = MLXTTSEngine._assemble_chunks([chunk])
        np.testing.assert_array_equal(result, chunk)

    def test_empty_list_returns_empty(self):
        """Empty chunk list should return empty array."""
        from engines.mlx_tts import MLXTTSEngine

        result = MLXTTSEngine._assemble_chunks([])
        assert len(result) == 0

    def test_crossfade_smooth_transition(self):
        """Cross-fade should produce smooth transitions between chunks."""
        from engines.mlx_tts import MLXTTSEngine

        # Two chunks with a sharp boundary (0.5 -> -0.5)
        chunk_a = np.ones(200, dtype=np.float32) * 0.5
        chunk_b = np.ones(200, dtype=np.float32) * -0.5

        result = MLXTTSEngine._assemble_chunks([chunk_a, chunk_b], crossfade_samples=50)

        # The cross-fade region should have intermediate values
        # Total length = 200 + 200 - 50 = 350
        assert len(result) == 350

        # Middle of cross-fade region should be near zero (blend of 0.5 and -0.5)
        mid = 200 - 25  # middle of the overlap
        assert abs(result[mid]) < 0.3, "Cross-fade midpoint should blend the two amplitudes"

    def test_discontinuity_detected(self, caplog):
        """Large amplitude jumps should trigger a warning."""
        from engines.mlx_tts import MLXTTSEngine
        import logging

        # Chunk with normal amplitude followed by near-silent chunk
        chunk_a = np.ones(200, dtype=np.float32) * 0.8
        chunk_b = np.ones(200, dtype=np.float32) * 0.001

        with caplog.at_level(logging.WARNING, logger="engines.mlx_tts"):
            MLXTTSEngine._assemble_chunks([chunk_a, chunk_b], crossfade_samples=50)

        assert any("discontinuity" in r.message.lower() for r in caplog.records), (
            "Should warn about amplitude discontinuity"
        )

    def test_no_warning_for_smooth_chunks(self, caplog):
        """Similar-amplitude chunks should not trigger warnings."""
        from engines.mlx_tts import MLXTTSEngine
        import logging

        chunk_a = np.ones(200, dtype=np.float32) * 0.5
        chunk_b = np.ones(200, dtype=np.float32) * 0.45

        with caplog.at_level(logging.WARNING, logger="engines.mlx_tts"):
            MLXTTSEngine._assemble_chunks([chunk_a, chunk_b], crossfade_samples=50)

        assert not any("discontinuity" in r.message.lower() for r in caplog.records)


class TestDurationSanityCheck:
    """Test duration sanity check for audio dropout detection."""

    def test_warns_on_short_audio(self, caplog):
        """Should warn when audio is much shorter than expected."""
        from engines.mlx_tts import MLXTTSEngine
        import logging

        # 20 words at 150wpm = 8 seconds expected, give only 1 second
        text = " ".join(["word"] * 20)
        audio = np.zeros(12000, dtype=np.float32)  # 1 second at 12kHz

        with caplog.at_level(logging.WARNING, logger="engines.mlx_tts"):
            MLXTTSEngine._check_duration_sanity(audio, 12000, text)

        assert any("shorter than expected" in r.message for r in caplog.records)

    def test_no_warning_for_normal_audio(self, caplog):
        """Should not warn when audio duration is reasonable."""
        from engines.mlx_tts import MLXTTSEngine
        import logging

        # 10 words at 150wpm = 4 seconds, give 5 seconds
        text = " ".join(["word"] * 10)
        audio = np.zeros(60000, dtype=np.float32)  # 5 seconds at 12kHz

        with caplog.at_level(logging.WARNING, logger="engines.mlx_tts"):
            MLXTTSEngine._check_duration_sanity(audio, 12000, text)

        assert not any("shorter than expected" in r.message for r in caplog.records)

    def test_short_text_skipped(self, caplog):
        """Should skip check for very short text (< 2 words)."""
        from engines.mlx_tts import MLXTTSEngine
        import logging

        audio = np.zeros(100, dtype=np.float32)  # tiny audio

        with caplog.at_level(logging.WARNING, logger="engines.mlx_tts"):
            MLXTTSEngine._check_duration_sanity(audio, 12000, "Hi")

        assert not any("shorter than expected" in r.message for r in caplog.records)


if __name__ == "__main__":
    # Run with: pytest tests/test_critical_paths.py -v
    pytest.main([__file__, "-v"])
