"""
Qwen3-TTS Engine Implementation

Currently uses Coqui TTS (XTTS-v2) as a placeholder until Qwen3-TTS becomes available.
Coqui XTTS-v2 supports voice cloning and is a proven, stable TTS solution.

When Qwen3-TTS stabilizes, replace the model loading logic while keeping the interface.
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path

try:
    from TTS.api import TTS
    HAS_COQUI_TTS = True
except ImportError:
    HAS_COQUI_TTS = False
    print("WARNING: Coqui TTS not installed. Install with: pip install TTS")


class Qwen3TTSEngine:
    """
    TTS Engine using Coqui XTTS-v2 (placeholder for Qwen3-TTS).

    Supports:
    - Voice cloning from reference audio (3-5 seconds)
    - Multiple output formats
    - GPU acceleration
    - Fallback to CPU mode
    """

    def __init__(self, model_size="v2", device="cuda"):
        """
        Initialize the TTS engine.

        Args:
            model_size: Model variant (v2 for XTTS-v2)
            device: 'cuda' or 'cpu'
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_size = model_size

        if not HAS_COQUI_TTS:
            print("TTS engine will run in stub mode (generate_dubbing returns None)")
            self.model = None
            return

        print(f"Loading Coqui TTS (XTTS-v2) on {self.device}...")
        try:
            # Load Coqui XTTS-v2 model
            # This model supports multilingual and voice cloning
            self.model = TTS(
                model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                device=self.device,
                progress_bar=True,
                gpu=self.device == "cuda",
            )
            print("✓ Coqui TTS model loaded successfully")
        except Exception as e:
            print(f"Failed to load TTS model: {e}")
            self.model = None

    def generate_dubbing(self, text, ref_audio_path=None, emotion_prompt="", language="en"):
        """
        Generates high-fidelity speech from text with optional voice cloning.

        Args:
            text: Text to synthesize
            ref_audio_path: Path to 3-5 second reference audio for voice cloning
            emotion_prompt: Optional style instruction (future Qwen3 feature)
            language: Language code (en, es, fr, de, etc.)

        Returns:
            Tuple of (audio_array, sample_rate) or None if TTS not available
        """
        if self.model is None:
            print(f"[STUB MODE] Would generate TTS for: {text}")
            return None

        try:
            # Use reference audio for voice cloning if provided
            speaker_wav = None
            if ref_audio_path:
                ref_path = Path(ref_audio_path)
                if ref_path.exists():
                    speaker_wav = str(ref_audio_path)
                    print(f"Using voice reference from: {ref_audio_path}")
                else:
                    print(f"Warning: Reference audio not found at {ref_audio_path}, using default voice")

            # Generate speech
            # Note: emotion_prompt is a placeholder for future Qwen3 features
            output_path = "/tmp/tts_output.wav"

            self.model.tts_to_file(
                text=text,
                speaker_wav=speaker_wav,
                language=language,
                file_path=output_path,
            )

            # Load generated audio
            audio, sr = torchaudio.load(output_path)
            audio = audio.cpu().numpy().astype(np.float32)

            # If stereo, convert to mono
            if audio.shape[0] > 1:
                audio = audio.mean(axis=0)
            else:
                audio = audio.squeeze()

            print(f"✓ Generated {len(audio) / sr:.2f}s of audio at {sr}Hz")
            return audio, sr

        except Exception as e:
            print(f"Error during TTS generation: {e}")
            return None

    async def stream_text_to_speech(
        self, text_iterator, ref_audio_path=None, emotion_prompt="", language="en"
    ):
        """
        Streams text chunks to TTS for continuous speech generation.

        Args:
            text_iterator: Iterator yielding text chunks
            ref_audio_path: Path to reference audio for voice cloning
            emotion_prompt: Optional style instruction
            language: Language code

        Yields:
            (audio_chunk, sample_rate) tuples
        """
        if self.model is None:
            print("[STUB MODE] Stream TTS not available")
            return

        speaker_wav = None
        if ref_audio_path:
            ref_path = Path(ref_audio_path)
            if ref_path.exists():
                speaker_wav = str(ref_audio_path)

        for text_chunk in text_iterator:
            if not text_chunk.strip():
                continue

            try:
                output_path = "/tmp/tts_chunk_output.wav"

                self.model.tts_to_file(
                    text=text_chunk,
                    speaker_wav=speaker_wav,
                    language=language,
                    file_path=output_path,
                )

                audio, sr = torchaudio.load(output_path)
                audio = audio.cpu().numpy().astype(np.float32)

                if audio.shape[0] > 1:
                    audio = audio.mean(axis=0)
                else:
                    audio = audio.squeeze()

                yield audio, sr

            except Exception as e:
                print(f"Error streaming TTS chunk: {e}")

    def _resample_audio(self, audio, sr_from, sr_to):
        """Resample audio to target sample rate."""
        if sr_from == sr_to:
            return audio

        resampler = torchaudio.transforms.Resample(sr_from, sr_to)
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)
        resampled = resampler(audio_tensor)
        return resampled.squeeze().numpy()


# Backward compatibility alias
Qwen3TTSStreamer = Qwen3TTSEngine


if __name__ == "__main__":
    # Test the TTS engine
    engine = Qwen3TTSEngine(device="cpu")

    # Test without reference audio
    result = engine.generate_dubbing("Hello, this is a test of the Qwen3 TTS engine.")
    if result:
        audio, sr = result
        print(f"Generated audio shape: {audio.shape}, sample rate: {sr}")
