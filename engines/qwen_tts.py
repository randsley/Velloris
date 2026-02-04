"""
Qwen3-TTS Engine Implementation

Qwen3-TTS is a state-of-the-art text-to-speech model from Alibaba's Qwen team.
Models available from Hugging Face:
  - Qwen3-TTS-12Hz-1.7B-CustomVoice (voice cloning)
  - Qwen3-TTS-12Hz-1.7B-Base (baseline)
  - Qwen3-TTS-12Hz-0.6B-CustomVoice (lightweight with voice cloning)
  - Qwen3-TTS-12Hz-0.6B-Base (lightweight baseline)

Features:
  - High-quality voice synthesis
  - Voice cloning with reference audio
  - Multiple model sizes for efficiency
  - Multilingual support
"""

import torch
import numpy as np
import torchaudio
from pathlib import Path
from typing import Optional, Tuple
import warnings

try:
    from transformers import AutoModel, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("WARNING: transformers not installed. Install with: pip install transformers")


class Qwen3TTSEngine:
    """
    Qwen3-TTS Engine for high-quality speech synthesis.

    Supports:
    - Text-to-speech synthesis
    - Voice cloning from reference audio (3-5 seconds)
    - Multiple model sizes (0.6B, 1.7B)
    - Efficient inference
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
        device: str = "cuda",
        use_auth_token: Optional[str] = None,
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
            device: 'cuda' or 'cpu'
            use_auth_token: Hugging Face API token if needed
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_size = model_size
        self.use_auth_token = use_auth_token

        # Verify model size is valid
        if model_size not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Unknown model size: {model_size}. Available: {list(self.AVAILABLE_MODELS.keys())}"
            )

        self.model_name = self.AVAILABLE_MODELS[model_size]
        self.model = None
        self.tokenizer = None
        self.sample_rate = 12000  # Qwen3-TTS native sample rate

        if HAS_TRANSFORMERS:
            self._load_model()
        else:
            warnings.warn("Qwen3-TTS will not function without transformers library")

    def _load_model(self):
        """Load Qwen3-TTS model and tokenizer from Hugging Face."""
        try:
            print(f"Loading Qwen3-TTS ({self.model_size}) from Hugging Face...")
            print(f"Model: {self.model_name}")

            # Load model
            self.model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device,
                trust_remote_code=True,
                token=self.use_auth_token,
            )
            self.model.eval()

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                token=self.use_auth_token,
            )

            print("✓ Qwen3-TTS model loaded successfully")

        except Exception as e:
            print(f"✗ Failed to load Qwen3-TTS: {e}")
            print("Make sure you have:")
            print("  1. Internet connection for downloading from Hugging Face")
            print("  2. Sufficient disk space (~2-5GB)")
            print("  3. If private model: Hugging Face API token")
            self.model = None
            self.tokenizer = None

    def generate_dubbing(
        self,
        text: str,
        ref_audio_path: Optional[str] = None,
        emotion: str = "",
        language: str = "en",
    ) -> Optional[Tuple[np.ndarray, int]]:
        """
        Generate speech from text using Qwen3-TTS.

        Args:
            text: Text to synthesize
            ref_audio_path: Path to reference audio for voice cloning (3-5 seconds)
            emotion: Emotion/style prompt (e.g., "happy", "sad", "neutral")
            language: Language code (en, zh, etc.)

        Returns:
            Tuple of (audio_array, sample_rate) or None
        """
        if self.model is None or self.tokenizer is None:
            print("[STUB MODE] Qwen3-TTS not loaded. Install transformers and download model.")
            return None

        try:
            # Prepare input text with optional emotion/style
            if emotion:
                prompt = f"[{emotion}] {text}"
            else:
                prompt = text

            print(f"Generating speech: {text[:50]}...")

            # Load reference audio if provided
            speaker_audio = None
            if ref_audio_path:
                ref_path = Path(ref_audio_path)
                if ref_path.exists():
                    try:
                        speaker_audio, sr = torchaudio.load(ref_audio_path)
                        # Resample to 12kHz if needed
                        if sr != self.sample_rate:
                            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                            speaker_audio = resampler(speaker_audio)
                        speaker_audio = speaker_audio.to(self.device)
                        print(f"Using voice reference: {ref_audio_path}")
                    except Exception as e:
                        print(f"Warning: Could not load reference audio: {e}")
                        speaker_audio = None
                else:
                    print(f"Warning: Reference audio not found: {ref_audio_path}")

            # Generate speech
            with torch.no_grad():
                # Tokenize input
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Generate audio
                if speaker_audio is not None and "CustomVoice" in self.model_size:
                    # Voice cloning mode
                    outputs = self.model.generate(
                        **inputs,
                        speaker_audio=speaker_audio,
                        max_new_tokens=1024,
                    )
                else:
                    # Standard generation
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=1024,
                    )

            # Extract audio from outputs
            if isinstance(outputs, tuple):
                audio = outputs[0] if len(outputs) > 0 else outputs
            else:
                audio = outputs

            # Convert to numpy
            if hasattr(audio, "cpu"):
                audio = audio.cpu().numpy()
            else:
                audio = np.array(audio)

            # Ensure float32
            audio = audio.astype(np.float32)

            # Flatten if needed
            if len(audio.shape) > 1:
                audio = audio.squeeze()

            # Normalize
            if audio.max() > 1.0:
                audio = audio / (audio.max() + 1e-6)

            duration = len(audio) / self.sample_rate
            print(f"✓ Generated {duration:.2f}s of audio")

            return audio, self.sample_rate

        except Exception as e:
            print(f"✗ Error during Qwen3-TTS generation: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def stream_text_to_speech(
        self,
        text_iterator,
        ref_audio_path: Optional[str] = None,
        emotion: str = "",
        language: str = "en",
    ):
        """
        Stream text chunks to Qwen3-TTS for continuous speech generation.

        Args:
            text_iterator: Async iterator yielding text chunks
            ref_audio_path: Path to reference audio for voice cloning
            emotion: Emotion/style prompt
            language: Language code

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
                    text_chunk,
                    ref_audio_path=ref_audio_path,
                    emotion=emotion,
                    language=language,
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
        self.tokenizer = None
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

    print("\nInitializing Qwen3-TTS engine...")
    engine = Qwen3TTSEngine(model_size="1.7B-CustomVoice", device="cpu")

    if engine.model is not None:
        print("\n✓ Engine ready for speech synthesis")
        # Test: result = engine.generate_dubbing("Hello, this is Qwen3-TTS.")
    else:
        print("\nNote: Download model from Hugging Face for full functionality")
