"""
Qwen3-TTS Engine Implementation

Official Qwen3-TTS from Alibaba's Qwen team.
Models available from Hugging Face:
  - Qwen3-TTS-12Hz-1.7B-CustomVoice (voice cloning)
  - Qwen3-TTS-12Hz-1.7B-Base (baseline)
  - Qwen3-TTS-12Hz-1.7B-VoiceDesign (voice design)
  - Qwen3-TTS-12Hz-0.6B-CustomVoice (lightweight with voice cloning)
  - Qwen3-TTS-12Hz-0.6B-Base (lightweight baseline)

Installation:
  pip install qwen-tts

Features:
  - High-quality voice synthesis
  - Voice cloning with reference audio
  - Voice design with natural language descriptions
  - Multiple model sizes for efficiency
  - Multilingual support (English, Mandarin, etc.)
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import warnings

from utils.device_utils import (
    get_optimal_device,
    get_optimal_dtype,
    validate_flash_attention,
)

try:
    from qwen_tts import Qwen3TTSModel

    HAS_QWEN3_TTS = True
except ImportError:
    HAS_QWEN3_TTS = False
    print("WARNING: qwen-tts not installed. Install with: pip install qwen-tts")

try:
    import soundfile as sf

    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False


class Qwen3TTSEngine:
    """
    Official Qwen3-TTS Engine for high-quality speech synthesis.

    Supports:
    - Text-to-speech synthesis
    - Voice cloning from reference audio
    - Voice design with natural language descriptions
    - Multiple model sizes (0.6B, 1.7B)
    - Efficient inference with optional FlashAttention 2
    """

    # Available models on Hugging Face
    AVAILABLE_MODELS = {
        "1.7B-CustomVoice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "1.7B-Base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "1.7B-VoiceDesign": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        "0.6B-CustomVoice": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "0.6B-Base": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    }

    def __init__(
        self,
        model_size: str = "1.7B-CustomVoice",
        device: str = "auto",
        dtype: str = "auto",
        use_flash_attention: bool = True,
    ):
        """
        Initialize Qwen3-TTS engine.

        Args:
            model_size: Model variant to use
                - "1.7B-CustomVoice" (recommended, voice cloning)
                - "1.7B-Base" (baseline, no voice cloning)
                - "1.7B-VoiceDesign" (design-oriented voices)
                - "0.6B-CustomVoice" (lightweight, voice cloning)
                - "0.6B-Base" (lightweight, no voice cloning)
            device: Device to use ('cuda', 'mps', 'cpu', or 'auto' for auto-detection)
            dtype: Data type ('float32', 'float16', 'bfloat16', or 'auto' for device-optimal)
            use_flash_attention: Whether to use FlashAttention 2 (faster, less memory)
        """
        # Auto-detect optimal device
        self.device = get_optimal_device(device)

        # Auto-detect optimal dtype if not specified
        if dtype == "auto":
            self.dtype = get_optimal_dtype(self.device)
        else:
            self.dtype = self._parse_dtype(dtype)
            # Warn if dtype may not be optimal for device
            if self.dtype == torch.bfloat16 and self.device == "mps":
                print("⚠ bfloat16 not well-supported on MPS, using float32 instead")
                self.dtype = torch.float32

        self.model_size = model_size
        self.use_flash_attention = use_flash_attention and validate_flash_attention(
            self.device
        )

        # Verify model size is valid
        if model_size not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Unknown model size: {model_size}. Available: {list(self.AVAILABLE_MODELS.keys())}"
            )

        self.model_name = self.AVAILABLE_MODELS[model_size]
        self.model = None
        self.sample_rate = 12000  # Qwen3-TTS native sample rate

        if HAS_QWEN3_TTS:
            self._load_model()
        else:
            warnings.warn(
                "Qwen3-TTS will not function without qwen-tts library. "
                "Install with: pip install qwen-tts"
            )

    def _parse_dtype(self, dtype_str: str):
        """Parse dtype string to torch dtype."""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if dtype_str not in dtype_map:
            raise ValueError(f"Unsupported dtype: {dtype_str}")
        return dtype_map[dtype_str]

    def _load_model(self):
        """Load Qwen3-TTS model from Hugging Face."""
        try:
            print(f"Loading Qwen3-TTS ({self.model_size}) from Hugging Face...")
            print(f"Model: {self.model_name}")

            # Device-specific warnings
            if self.device == "mps":
                print("⚠ Qwen3-TTS on MPS: Slower than CUDA, limited dtype support")
            elif self.device == "cpu":
                print("⚠ Qwen3-TTS on CPU: Very slow. GPU is recommended.")

            # Determine attention implementation
            # FlashAttention 2 only works on CUDA
            attn_impl = "flash_attention_2" if self.use_flash_attention else None

            # Load model
            self.model = Qwen3TTSModel.from_pretrained(
                self.model_name,
                device_map=self.device,
                dtype=self.dtype,
                attn_implementation=attn_impl if self.device == "cuda" else None,
            )

            print("✓ Qwen3-TTS model loaded successfully")
            print(f"   Device: {self.device}, Dtype: {str(self.dtype).split('.')[-1]}")
            if self.use_flash_attention:
                print("   FlashAttention 2: ✅ Enabled")

        except Exception as e:
            print(f"✗ Failed to load Qwen3-TTS: {e}")
            print("\nMake sure you have:")
            print("  1. Installed qwen-tts: pip install qwen-tts")
            print("  2. Internet connection for downloading from Hugging Face")
            print("  3. Sufficient disk space (~3-5GB)")
            print("  4. Python 3.12+ (recommended)")
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
        Generate speech from text using Qwen3-TTS.

        Args:
            text: Text to synthesize
            ref_audio_path: Path to reference audio for voice cloning
            language: Language code (lowercase). Supported: 'auto', 'chinese', 'english',
                     'french', 'german', 'italian', 'japanese', 'korean', 'portuguese',
                     'russian', 'spanish'
            speaker: Speaker name (for CustomVoice models like "Ryan", "Lisa", etc.)
            instruct: Instruction for tone/emotion/style

        Returns:
            Tuple of (audio_array, sample_rate) or None
        """
        if self.model is None:
            print("[STUB MODE] Qwen3-TTS not loaded. Run: pip install qwen-tts")
            return None

        # Normalize language to lowercase
        language = language.lower()
        # Map common language codes to full names
        language_map = {
            "en": "english",
            "zh": "chinese",
            "fr": "french",
            "de": "german",
            "it": "italian",
            "ja": "japanese",
            "ko": "korean",
            "pt": "portuguese",
            "ru": "russian",
            "es": "spanish",
        }
        language = language_map.get(language, language)

        try:
            print(f"Generating speech: {text[:50]}...")

            with torch.no_grad():
                # Determine which generation method to use
                if "CustomVoice" in self.model_size:
                    # Custom voice generation (use provided speaker or default to "Ryan")
                    default_speaker = "Ryan"
                    wavs, sr = self.model.generate_custom_voice(
                        text=text,
                        language=language,
                        speaker=speaker or default_speaker,
                        instruct=instruct if instruct else None,
                    )

                elif "VoiceDesign" in self.model_size:
                    # Voice design generation (use provided instruct or default)
                    default_instruct = "A clear and professional voice"
                    wavs, sr = self.model.generate_voice_design(
                        text=text,
                        language=language,
                        instruct=instruct or default_instruct,
                    )

                elif ref_audio_path:
                    # Voice cloning with reference audio
                    ref_path = Path(ref_audio_path)
                    if not ref_path.exists():
                        print(f"Warning: Reference audio not found: {ref_audio_path}")
                        # Fallback to default speaker for CustomVoice or default instruct for VoiceDesign
                        wavs, sr = self.model.generate_custom_voice(
                            text=text,
                            language=language,
                            speaker=speaker or "Ryan",
                        )
                    else:
                        # For voice clone, we need the reference text
                        # For now, use the generation method available
                        wavs, sr = self.model.generate_voice_clone(
                            text=text,
                            language=language,
                            ref_audio=str(ref_audio_path),
                            ref_text=text,  # Simplified: use same text as reference
                        )

                else:
                    # Fallback: use default speaker for CustomVoice models
                    if "CustomVoice" in self.model_size:
                        wavs, sr = self.model.generate_custom_voice(
                            text=text,
                            language=language,
                            speaker="Ryan",
                        )
                    else:
                        print(
                            "Note: Model requires either speaker, instruct, or ref_audio"
                        )
                        return None

            # Convert to numpy
            if isinstance(wavs, torch.Tensor):
                audio = wavs[0].cpu().numpy()
            else:
                audio = np.array(wavs[0]) if isinstance(wavs, list) else np.array(wavs)

            # Ensure float32
            audio = audio.astype(np.float32)

            # Normalize if needed
            if audio.max() > 1.0:
                audio = audio / (audio.max() + 1e-6)

            duration = len(audio) / sr
            print(f"✓ Generated {duration:.2f}s of audio at {sr}Hz")

            return audio, sr

        except Exception as e:
            print(f"✗ Error during Qwen3-TTS generation: {e}")
            import traceback

            traceback.print_exc()
            return None

    async def stream_text_to_speech(
        self,
        text_iterator,
        ref_audio_path: Optional[str] = None,
        language: str = "English",
        speaker: Optional[str] = None,
    ):
        """
        Stream text chunks to Qwen3-TTS for continuous speech generation.

        Args:
            text_iterator: Async iterator yielding text chunks
            ref_audio_path: Path to reference audio for voice cloning
            language: Language code
            speaker: Speaker name (for CustomVoice models)

        Yields:
            (audio_chunk, sample_rate) tuples
        """
        if self.model is None:
            print("[STUB MODE] Qwen3-TTS not available")
            return

        try:
            async for text_chunk in text_iterator:
                if not text_chunk.strip():
                    continue

                result = self.generate_dubbing(
                    text=text_chunk,
                    ref_audio_path=ref_audio_path,
                    language=language,
                    speaker=speaker,
                )

                if result:
                    yield result
                else:
                    yield np.array([], dtype=np.float32), self.sample_rate

        except Exception as e:
            print(f"Error in stream_text_to_speech: {e}")

    def set_model(self, model_size: str):
        """
        Switch to a different Qwen3-TTS model size.

        Args:
            model_size: Model variant to switch to
        """
        if model_size not in self.AVAILABLE_MODELS:
            print(f"Unknown model size: {model_size}")
            return

        self.model_size = model_size
        self.model_name = self.AVAILABLE_MODELS[model_size]
        self._load_model()

    def unload(self):
        """Unload model to free memory."""
        self.model = None
        print("✓ Qwen3-TTS model unloaded")

    @classmethod
    def get_available_models(cls):
        """Get list of available Qwen3-TTS models."""
        return list(cls.AVAILABLE_MODELS.keys())


# Backward compatibility alias
Qwen3TTSStreamer = Qwen3TTSEngine


if __name__ == "__main__":
    print("Available Qwen3-TTS models:")
    for model in Qwen3TTSEngine.get_available_models():
        print(f"  - {model}")

    print("\nTo install Qwen3-TTS:")
    print("  pip install qwen-tts")
    print("\nThen initialize:")
    print("  engine = Qwen3TTSEngine(model_size='1.7B-CustomVoice')")
