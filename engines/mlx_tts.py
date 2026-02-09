"""
MLX-Audio Engine for Apple Silicon

This engine uses the mlx-audio library, which is a native
implementation of various audio models for Apple's MLX framework.
It is used on macOS to avoid the dependency conflicts between
qwen-tts and other MLX-based libraries.

Repo: https://github.com/Blaizzy/mlx-audio
"""

import logging
import numpy as np
import os
import threading
from typing import Optional, Tuple
import warnings

from utils.device_utils import get_optimal_device

logger = logging.getLogger(__name__)

try:
    from mlx_audio.tts.utils import load_model

    HAS_MLX_AUDIO = True
except ImportError:
    HAS_MLX_AUDIO = False
    warnings.warn("mlx-audio not installed. On macOS, run: pip install mlx-audio")

# Supported audio extensions for reference audio
_SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

# Minimum file size for valid audio (1KB)
_MIN_AUDIO_FILE_SIZE = 1024


class MLXTTSEngine:
    """
    MLX-Audio Engine for high-quality speech synthesis on Apple Silicon.
    """

    # Class-level model cache and lock
    _model_cache: dict = {}
    _cache_lock = threading.Lock()

    def __init__(
        self,
        model_name: str = "Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit",
        device: str = "mps",
    ):
        """
        Initialize the MLX-Audio engine.

        Args:
            model_name: The MLX model to use from Hugging Face.
            device: The device to use (should be 'mps' for Apple Silicon).
        """
        self.device = get_optimal_device(device)
        if self.device != "mps":
            warnings.warn(
                f"MLX-Audio is optimized for 'mps', but device is set to '{self.device}'."
            )

        self.model_name = f"mlx-community/{model_name}"
        self.sample_rate = 12000  # Qwen3-TTS native sample rate
        self.model = None

        if not HAS_MLX_AUDIO:
            raise ImportError("mlx-audio library is not installed.")

        try:
            self.model = self._get_cached_model(self.model_name)
            print(f"[OK] MLXTTSEngine initialized for device '{self.device}'")
            print(f"   Model: {self.model_name}")
        except Exception as e:
            print(f"[X] Failed to load MLX TTS model: {e}")
            self.model = None

    @classmethod
    def _get_cached_model(cls, model_name: str):
        """Load model from cache or disk, thread-safe."""
        with cls._cache_lock:
            if model_name in cls._model_cache:
                logger.info(f"Reusing cached model: {model_name}")
                return cls._model_cache[model_name]
            model = load_model(model_name)
            cls._model_cache[model_name] = model
            return model

    @staticmethod
    def _validate_ref_audio(ref_audio_path: Optional[str]) -> Optional[str]:
        """Validate reference audio file. Returns path if valid, None otherwise."""
        if ref_audio_path is None:
            return None

        if not os.path.isfile(ref_audio_path):
            logger.warning(f"Reference audio not found: {ref_audio_path}")
            return None

        ext = os.path.splitext(ref_audio_path)[1].lower()
        if ext not in _SUPPORTED_AUDIO_EXTENSIONS:
            logger.warning(
                f"Unsupported audio format '{ext}'. Supported: {_SUPPORTED_AUDIO_EXTENSIONS}"
            )
            return None

        size = os.path.getsize(ref_audio_path)
        if size < _MIN_AUDIO_FILE_SIZE:
            logger.warning(
                f"Reference audio too small ({size} bytes, min {_MIN_AUDIO_FILE_SIZE})"
            )
            return None

        return ref_audio_path

    @staticmethod
    def _assemble_chunks(chunks: list, crossfade_samples: int = 50) -> np.ndarray:
        """Assemble audio chunks with cross-fade to smooth discontinuities.

        Detects amplitude jumps at chunk boundaries (suspected dropouts from
        mlx-audio #464) and applies overlap-add cross-fading to reduce audible
        artifacts.

        Args:
            chunks: List of audio arrays (MLX or numpy).
            crossfade_samples: Number of samples for the cross-fade window.

        Returns:
            Assembled audio array (numpy, float32).
        """
        if not chunks:
            return np.array([], dtype=np.float32)

        # Helper: convert MLX arrays to numpy
        def to_numpy(chunk):
            if hasattr(chunk, 'tolist'):
                # MLX array - convert via tolist() then to numpy
                return np.asarray(chunk.tolist(), dtype=np.float32)
            else:
                # Already numpy or convertible
                return np.asarray(chunk, dtype=np.float32)

        if len(chunks) == 1:
            return to_numpy(chunks[0])

        discontinuities = 0
        assembled = to_numpy(chunks[0])

        for chunk in chunks[1:]:
            chunk = to_numpy(chunk)
            fade_len = min(crossfade_samples, len(assembled), len(chunk))

            if fade_len > 0:
                # Detect amplitude discontinuity at boundary
                tail_rms = np.sqrt(np.mean(assembled[-fade_len:] ** 2)) + 1e-10
                head_rms = np.sqrt(np.mean(chunk[:fade_len] ** 2)) + 1e-10
                ratio = max(tail_rms, head_rms) / min(tail_rms, head_rms)
                if ratio > 10.0:
                    discontinuities += 1
                    logger.warning(
                        f"Amplitude discontinuity detected (ratio={ratio:.1f}x) — "
                        "possible audio dropout (mlx-audio #464)"
                    )

                # Cross-fade: linear ramp
                fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
                fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)

                overlap = assembled[-fade_len:] * fade_out + chunk[:fade_len] * fade_in
                assembled = np.concatenate([assembled[:-fade_len], overlap, chunk[fade_len:]])
            else:
                assembled = np.concatenate([assembled, chunk])

        if discontinuities > 0:
            logger.warning(
                f"{discontinuities} chunk discontinuities detected — "
                "audio may have dropouts (mlx-audio #464)"
            )

        return assembled

    @staticmethod
    def _check_duration_sanity(audio: np.ndarray, sample_rate: int, text: str) -> None:
        """Warn if generated audio is suspiciously short for the input text.

        Estimates expected duration at ~150 words/min and warns if actual
        audio is less than 50% of that estimate.
        """
        word_count = len(text.split())
        if word_count < 2:
            return
        expected_seconds = word_count / 150.0 * 60.0  # 150 wpm
        actual_seconds = len(audio) / sample_rate
        if actual_seconds < expected_seconds * 0.5:
            logger.warning(
                f"Audio duration ({actual_seconds:.1f}s) is much shorter than "
                f"expected (~{expected_seconds:.1f}s for {word_count} words) — "
                "possible audio dropout (mlx-audio #464)"
            )

    @staticmethod
    def _rms_normalize(audio: np.ndarray, target_dbfs: float = -20.0) -> np.ndarray:
        """RMS-based normalization preserving dynamic range.

        Targets a specific loudness level (dBFS) without aggressive compression.
        The generated audio from CosyVoice is already well-formed; hard clipping
        is rarely needed and tanh compression destroys quality.
        """
        rms = np.sqrt(np.mean(audio**2))
        if rms < 1e-8:
            return audio

        target_rms = 10 ** (target_dbfs / 20.0)
        gain = target_rms / rms
        audio = audio * gain

        # Clip only extreme peaks (values > 1.0) to prevent actual hard clipping
        # but preserve all dynamic range and quality
        audio = np.clip(audio, -1.0, 1.0)
        return audio

    def generate_dubbing(
        self,
        text: str,
        ref_audio_path: Optional[str] = None,
        language: str = "",
        speaker: Optional[str] = None,
        instruct: str = "",
    ) -> Optional[Tuple[np.ndarray, int]]:
        """
        Generate speech from text by calling mlx-audio CLI tool.
        This guarantees identical output to running the CLI manually.

        Args:
            text: Text to synthesize.
            ref_audio_path: Path to reference audio for voice cloning.
            language: Language of the text.
            speaker: Speaker name (voice).
            instruct: Instruction for tone/emotion/style.

        Returns:
            Tuple of (audio_array, sample_rate) or None.
        """
        import subprocess
        import tempfile
        import soundfile as sf
        import glob

        try:
            print(f"Generating speech with mlx-audio CLI: {text[:50]}...")

            # Create temporary directory for output
            with tempfile.TemporaryDirectory() as tmpdir:
                # Build CLI command
                cmd = [
                    "python3", "-m", "mlx_audio.tts.generate",
                    "--model", "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit",
                    "--text", text,
                    "--voice", speaker or "Aiden",
                    "--temperature", "0.0",
                    "--cfg_scale", "1.5",
                    "--speed", "1.0",
                    "--lang_code", language or "en",
                    "--output_path", tmpdir,
                    "--file_prefix", "generated",
                ]

                # Add optional parameters
                if instruct:
                    cmd.extend(["--instruct", instruct])
                if ref_audio_path:
                    cmd.extend(["--ref_audio", ref_audio_path])

                print(f"  [DEBUG] Calling CLI...")

                # Run CLI command (suppress output)
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

                if result.returncode != 0:
                    print(f"[X] CLI error: {result.stderr[:200]}")
                    return None

                # Find generated audio file
                audio_files = glob.glob(f"{tmpdir}/generated_*.wav")
                if not audio_files:
                    print("[X] No audio file generated")
                    return None

                # Load the audio file
                audio_file = audio_files[0]
                audio, sr = sf.read(audio_file)
                audio = audio.astype(np.float32)

                duration = len(audio) / sr
                print(f"[OK] Generated {duration:.2f}s of audio at {sr}Hz (via CLI)")

                return audio, sr

        except subprocess.TimeoutExpired:
            print("[X] CLI generation timed out")
            return None
        except Exception as e:
            print(f"[X] Error during generation: {e}")
            import traceback
            traceback.print_exc()
            return None

    def unload(self):
        """Unload model to free memory (MLX handles this automatically)."""
        self.model = None
        print("[OK] MLX-Audio engine unloaded (memory managed by MLX).")
