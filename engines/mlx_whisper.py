"""
MLX Whisper Engine for Apple Silicon

This engine uses the mlx-audio library for a native implementation
of Whisper on Apple's MLX framework.
"""

import numpy as np
import warnings

from utils.device_utils import get_optimal_device

try:
    from mlx_audio.stt.generate import generate_transcription

    HAS_MLX_AUDIO = True
except ImportError:
    HAS_MLX_AUDIO = False
    warnings.warn("mlx-audio not installed. On macOS, run: pip install mlx-audio")


class MLXWhisperEngine:
    """
    MLX-Audio Whisper Engine for speech-to-text on Apple Silicon.
    """

    def __init__(
        self,
        model_name: str = "whisper-large-v3-turbo-asr-fp16",
        device: str = "mps",
    ):
        """
        Initialize the MLX-Audio Whisper engine.

        Args:
            model_name: The MLX Whisper model to use from Hugging Face.
            device: The device to use (should be 'mps' for Apple Silicon).
        """
        self.device = get_optimal_device(device)
        if self.device != "mps":
            warnings.warn(
                f"MLX-Audio is optimized for 'mps', but device is set to '{self.device}'."
            )

        self.model_name = f"mlx-community/{model_name}"

        if not HAS_MLX_AUDIO:
            raise ImportError("mlx-audio library is not installed.")

        print(f"[OK] MLXWhisperEngine initialized for device '{self.device}'")
        print(f"   Model: {self.model_name}")

    def transcribe(self, audio: np.ndarray, language: str = "en", **kwargs) -> dict:
        """
        Transcribe audio using MLX-Audio Whisper.

        Args:
            audio: Audio samples as a numpy array.
            language: The language of the audio.

        Returns:
            A dictionary containing the transcribed text.
        """
        try:
            result = generate_transcription(
                model=self.model_name,
                audio=audio,
                language=language,
            )

            return {"text": result.text}

        except Exception as e:
            print(f"[X] Error during MLX-Whisper transcription: {e}")
            import traceback

            traceback.print_exc()
            return {"text": ""}

    def unload(self):
        """Unload model to free memory (MLX handles this automatically)."""
        print("[OK] MLX-Whisper engine unloaded (memory managed by MLX).")
