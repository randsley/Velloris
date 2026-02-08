"""
MLX-Audio Engine for Apple Silicon

This engine uses the mlx-audio library, which is a native
implementation of various audio models for Apple's MLX framework.
It is used on macOS to avoid the dependency conflicts between
qwen-tts and other MLX-based libraries.

Repo: https://github.com/Blaizzy/mlx-audio
"""

import numpy as np
from typing import Optional, Tuple
import warnings

from utils.device_utils import get_optimal_device

try:
    from mlx_audio.tts.utils import load_model

    HAS_MLX_AUDIO = True
except ImportError:
    HAS_MLX_AUDIO = False
    warnings.warn("mlx-audio not installed. On macOS, run: pip install mlx-audio")


class MLXTTSEngine:
    """
    MLX-Audio Engine for high-quality speech synthesis on Apple Silicon.
    """

    def __init__(
        self,
        model_name: str = "Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
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
            self.model = load_model(self.model_name)
            print(f"[OK] MLXTTSEngine initialized for device '{self.device}'")
            print(f"   Model: {self.model_name}")
        except Exception as e:
            print(f"[X] Failed to load MLX TTS model: {e}")
            self.model = None


    def generate_dubbing(
        self,
        text: str,
        ref_audio_path: Optional[str] = None,
        language: str = "english",
        speaker: Optional[str] = None,
        instruct: str = "",
    ) -> Optional[Tuple[np.ndarray, int]]:
        """
        Generate speech from text using MLX-Audio.

        Args:
            text: Text to synthesize.
            ref_audio_path: Path to reference audio for voice cloning.
            language: Language of the text.
            speaker: Speaker name (not directly used, for API compatibility).
            instruct: Instruction for tone/emotion/style.

        Returns:
            Tuple of (audio_array, sample_rate) or None.
        """
        if self.model is None:
            print("[X] MLX TTS model not loaded.")
            return None

        try:
            print(f"Generating speech with MLX-Audio: {text[:50]}...")

            results = list(self.model.generate(
                text=text,
                language=language,
                instruction=instruct,
                voice=speaker,
                ref_audio_path=ref_audio_path,
            ))
            
            if not results:
                print("[X] MLX-Audio generation failed to produce audio.")
                return None

            # Concatenate audio chunks from the generator
            full_audio = np.concatenate([res.audio for res in results])

            if full_audio is None or full_audio.size == 0:
                print("[X] MLX-Audio generation failed to produce audio.")
                return None

            # Ensure float32 and normalize
            audio = full_audio.astype(np.float32)
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))

            duration = len(audio) / self.sample_rate
            print(
                f"[OK] Generated {duration:.2f}s of audio at {self.sample_rate}Hz with MLX-Audio"
            )

            return audio, self.sample_rate

        except Exception as e:
            print(f"[X] Error during MLX-Audio generation: {e}")
            import traceback

            traceback.print_exc()
            return None

    def unload(self):
        """Unload model to free memory (MLX handles this automatically)."""
        self.model = None
        print("[OK] MLX-Audio engine unloaded (memory managed by MLX).")
