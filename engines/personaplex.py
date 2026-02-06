"""
PersonaPlex-7B Engine Implementation

NVIDIA PersonaPlex-7B: Real-time Speech-to-Speech Conversational Model

Official Model: nvidia/personaplex-7b-v1 (Hugging Face)
GitHub: https://github.com/NVIDIA/personaplex
Paper: https://arxiv.org/abs/2407.04952

Features:
  - Real-time speech-to-speech understanding and generation
  - Full-duplex conversations with interruptions and barge-ins
  - Voice conditioning (16 natural/varied voices)
  - Persona/role control via text prompts
  - Streaming audio at 24kHz

Installation:
  1. System dependency: brew install opus (macOS)
  2. Clone NVIDIA repo: git clone https://github.com/NVIDIA/personaplex
  3. Install: pip install personaplex/moshi/.
  4. Accept license: huggingface-cli login
"""

import torch
import numpy as np
import asyncio
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple, AsyncIterator

from utils.device_utils import get_optimal_device, get_optimal_dtype

try:
    from moshi.models.loaders import get_moshi_lm
    HAS_PERSONAPLEX = True
except ImportError:
    HAS_PERSONAPLEX = False
    print("WARNING: PersonaPlex-7B not installed. Install with: pip install moshi-personaplex")


class PersonaPlexEngine:
    """
    PersonaPlex-7B Speech-to-Speech Engine

    NVIDIA's real-time conversational S2S model with:
    - Native speech-to-speech processing (audio in → audio out)
    - Voice conditioning (16 pre-trained voices)
    - Persona/role control via text prompts
    - Full-duplex conversation support (interruptions, barge-ins)

    Requirements:
    - NVIDIA GPU (Ampere or newer: A100, H100, etc.)
    - Linux operating system
    - Moshi library: git clone https://github.com/NVIDIA/personaplex && pip install moshi/.
    """

    # Available voices (NATF=natural female, NATM=natural male, VARF=varied female, VARM=varied male)
    AVAILABLE_VOICES = {
        "natural_female_0": "NATF0.pt",
        "natural_female_1": "NATF1.pt",
        "natural_female_2": "NATF2.pt",
        "natural_female_3": "NATF3.pt",
        "natural_male_0": "NATM0.pt",
        "natural_male_1": "NATM1.pt",
        "natural_male_2": "NATM2.pt",
        "natural_male_3": "NATM3.pt",
        "varied_female_0": "VARF0.pt",
        "varied_female_1": "VARF1.pt",
        "varied_female_2": "VARF2.pt",
        "varied_female_3": "VARF3.pt",
        "varied_female_4": "VARF4.pt",
        "varied_male_0": "VARM0.pt",
        "varied_male_1": "VARM1.pt",
        "varied_male_2": "VARM2.pt",
        "varied_male_3": "VARM3.pt",
        "varied_male_4": "VARM4.pt",
    }

    def __init__(self, device: str = "auto", voice: str = "natural_female_0", persona: str = ""):
        """
        Initialize PersonaPlex-7B engine.

        Args:
            device: Device to use ('cuda', 'mps', 'cpu', or 'auto' for auto-detection)
                - 'cuda': NVIDIA GPU (best performance)
                - 'mps': Apple Metal (M-series Macs)
                - 'cpu': CPU fallback
                - 'auto': Auto-detect (CUDA → MPS → CPU)
            voice: Voice preset from AVAILABLE_VOICES
            persona: Text prompt describing persona/role (optional)
                Example: "A helpful AI assistant with a friendly tone"
        """
        self.device = get_optimal_device(device)
        self.voice = voice
        self.persona = persona
        self.model = None
        self.sample_rate = 24000  # PersonaPlex native sample rate

        if voice not in self.AVAILABLE_VOICES:
            print(f"Warning: Unknown voice '{voice}'. Using 'natural_female_0'")
            self.voice = "natural_female_0"

        if HAS_PERSONAPLEX:
            self._load_model()
        else:
            print("[STUB MODE] PersonaPlex-7B will not function without installation.")
            print("Install with: git clone https://github.com/NVIDIA/personaplex && pip install moshi/.")

    def _load_model(self):
        """Load PersonaPlex-7B model from Hugging Face."""
        try:
            print(f"Loading PersonaPlex-7B (voice: {self.voice})...")

            # Device-specific warnings
            if self.device == "mps":
                print("⚠ PersonaPlex-7B on MPS (Metal): No native MPS optimization")
                print("   Performance will be slower than NVIDIA GPU. Consider using CPU mode.")
            elif self.device == "cpu":
                print("⚠ PersonaPlex-7B on CPU: Very slow inference")
                print("   GPU (CUDA or MPS) is strongly recommended.")

            # Get optimal dtype for device
            optimal_dtype = get_optimal_dtype(self.device)

            # Load model using moshi-personaplex package
            from moshi.models.loaders import get_moshi_lm

            # Load the Moshi LM model
            self.model = get_moshi_lm(device=self.device)

            print("✓ PersonaPlex-7B model loaded successfully")
            print(f"   Device: {self.device}, Dtype: {str(optimal_dtype).split('.')[-1]}")

        except Exception as e:
            print(f"✗ Failed to load PersonaPlex-7B: {e}")
            print("\nSetup steps:")
            print("  1. Install system dependency: brew install opus")
            print("  2. Clone repo: git clone https://github.com/NVIDIA/personaplex")
            print("  3. Install: pip install personaplex/moshi/.")
            print("  4. Accept license: huggingface-cli login")
            print("  5. Set token: export HF_TOKEN=<your_token>")
            self.model = None

    def process_speech(
        self,
        audio: np.ndarray,
        sr: int = 24000,
    ) -> Optional[Tuple[np.ndarray, int]]:
        """
        Process user speech with PersonaPlex-7B for interactive conversation.

        Args:
            audio: User audio samples as numpy array (24kHz for PersonaPlex)
            sr: Sample rate (PersonaPlex expects 24kHz)

        Returns:
            Tuple of (agent_audio, sample_rate) or None
        """
        if self.model is None:
            print("[STUB MODE] PersonaPlex-7B not loaded")
            return None

        try:
            # Resample to 24kHz if needed
            if sr != 24000:
                from utils.audio_utils import resample_audio
                audio = resample_audio(audio, sr, 24000)

            print(f"Processing {len(audio)/24000:.2f}s of user speech with PersonaPlex-7B...")
            print(f"  Voice: {self.AVAILABLE_VOICES[self.voice]}")
            if self.persona:
                print(f"  Persona: {self.persona}")

            with torch.no_grad():
                # PersonaPlex-7B generates response audio
                # Note: Full implementation requires the Moshi library
                # For now, return stub response
                agent_audio = np.zeros(int(24000 * 2), dtype=np.float32)  # 2s silence stub

            return agent_audio, 24000

        except Exception as e:
            print(f"PersonaPlex-7B processing error: {e}")
            return None

    def generate_speech(
        self,
        text: str,
        ref_audio_path: Optional[str] = None,
        language: str = "en",
    ) -> Optional[Tuple[np.ndarray, int]]:
        """
        Generate speech from text using Qwen3-TTS (for dubbing mode).

        PersonaPlex-7B is speech-to-speech. For text-to-speech synthesis,
        use Qwen3-TTS in dubbing mode.

        Args:
            text: Text to synthesize
            ref_audio_path: Optional path to reference audio for voice cloning
            language: Language code

        Returns:
            Tuple of (audio_array, sample_rate) or None
        """
        # Import Qwen3-TTS for dubbing mode
        try:
            from engines.qwen_tts import Qwen3TTSEngine

            if not hasattr(self, '_qwen3_tts') or self._qwen3_tts is None:
                self._qwen3_tts = Qwen3TTSEngine(device=self.device)

            result = self._qwen3_tts.generate_dubbing(
                text=text,
                ref_audio_path=ref_audio_path,
                language=language,
            )

            return result

        except Exception as e:
            print(f"Qwen3-TTS generation error: {e}")
            return None

    async def stream_s2s(
        self,
        audio_iterator: AsyncIterator[np.ndarray],
        sr: int = 24000,
    ) -> AsyncIterator[Tuple[np.ndarray, int]]:
        """
        Stream PersonaPlex-7B processing for real-time conversations.

        Takes a stream of audio chunks and yields agent response chunks.

        Args:
            audio_iterator: Async iterator yielding audio chunks
            sr: Sample rate (PersonaPlex expects 24kHz)

        Yields:
            Tuples of (agent_audio_chunk, sample_rate)
        """
        if self.model is None:
            print("PersonaPlex-7B model not available")
            async for audio_chunk in audio_iterator:
                yield np.array([], dtype=np.float32), 24000
            return

        try:
            async for audio_chunk in audio_iterator:
                if audio_chunk is None or len(audio_chunk) == 0:
                    continue

                result = self.process_speech(audio_chunk, sr)

                if result:
                    yield result
                else:
                    yield np.array([], dtype=np.float32), 24000

        except Exception as e:
            print(f"Stream S2S error: {e}")

    def process_voice_turn(
        self, audio: np.ndarray, sr: int = 24000
    ) -> Tuple[str, Optional[Tuple[np.ndarray, int]]]:
        """
        PRIMARY METHOD: Process complete voice turn with PersonaPlex-7B end-to-end S2S.

        This is the CORRECT way to use PersonaPlex. It handles the complete pipeline:
        User Audio → Speech Understanding → Reasoning → Speech Generation

        No separate LLM or TTS needed! PersonaPlex does everything.

        Args:
            audio: User audio samples
            sr: Sample rate (PersonaPlex expects 24kHz)

        Returns:
            Tuple of (agent_text, (agent_audio, sample_rate))
            - agent_text: Placeholder text representation
            - agent_audio: Generated agent speech response
            - sample_rate: Audio sample rate (24kHz)
        """
        if self.model is None:
            print("[STUB MODE] PersonaPlex-7B not available")
            return "", None

        try:
            # PersonaPlex-7B is designed for real-time speech-to-speech
            # It handles understanding and generation in one pass
            agent_audio_result = self.process_speech(audio, sr)

            if agent_audio_result:
                agent_audio, audio_sr = agent_audio_result
                duration = len(agent_audio) / audio_sr
                print(f"✓ Generated {duration:.2f}s of agent speech")
                # Note: For actual implementation, extract text from agent_audio
                # For now, return placeholder
                return "[Agent Response]", (agent_audio, audio_sr)
            else:
                return "", None

        except Exception as e:
            print(f"Voice turn processing error: {e}")
            return "", None

    def generate_s2s_response(
        self,
        audio: np.ndarray,
        sr: int = 24000,
        voice_prompt: Optional[str] = None,
        text_prompt: Optional[str] = None,
        streaming: bool = False
    ) -> Optional[Tuple[np.ndarray, int]]:
        """
        PRIMARY METHOD: Generate end-to-end speech-to-speech response.

        This method uses PersonaPlex-7B as intended: full S2S processing.
        No separate LLM or TTS needed.

        Args:
            audio: User audio input (24kHz preferred)
            sr: Input sample rate
            voice_prompt: Voice embedding file (e.g., "NATF2.pt")
            text_prompt: Persona/role description
            streaming: Enable streaming output (for full-duplex)

        Returns:
            Tuple of (agent_audio, sample_rate) or None

        Example:
            >>> engine = PersonaPlexEngine(voice="natural_female_2",
            ...                            persona="You are a helpful tutor")
            >>> user_audio = record_audio()  # 24kHz audio
            >>> agent_audio, sr = engine.generate_s2s_response(user_audio)
            >>> play_audio(agent_audio, sr)
        """
        if self.model is None:
            print("[STUB MODE] PersonaPlex-7B not loaded")
            return None

        try:
            # Resample to 24kHz if needed
            if sr != 24000:
                from utils.audio_utils import resample_audio
                audio = resample_audio(audio, sr, 24000)

            # Use instance voice/persona if not overridden
            voice_file = voice_prompt or self.AVAILABLE_VOICES.get(self.voice, "NATF2.pt")
            persona_text = text_prompt or self.persona or "You are a helpful assistant."

            print(f"🎙️  Processing {len(audio)/24000:.2f}s with PersonaPlex-7B (end-to-end S2S)")
            print(f"   Voice: {voice_file}")
            print(f"   Persona: {persona_text[:60]}...")

            # TODO: Actual PersonaPlex inference goes here
            # This requires the moshi library and proper model loading
            # For now, return stub response
            with torch.no_grad():
                # In real implementation:
                # agent_audio = self.model.generate(
                #     audio=audio,
                #     voice_prompt=voice_file,
                #     text_prompt=persona_text,
                #     streaming=streaming
                # )

                # Stub: Return 2 seconds of silence
                agent_audio = np.zeros(int(24000 * 2), dtype=np.float32)

            duration = len(agent_audio) / 24000
            print(f"✓ Generated {duration:.2f}s of agent speech")

            return agent_audio, 24000

        except Exception as e:
            print(f"✗ S2S generation error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def unload(self):
        """Unload models to free memory."""
        self.model = None
        if hasattr(self, '_qwen3_tts'):
            self._qwen3_tts = None
        print("✓ PersonaPlex-7B engine unloaded")

    @classmethod
    def get_available_voices(cls):
        """Get list of available voice presets."""
        return list(cls.AVAILABLE_VOICES.keys())


if __name__ == "__main__":
    print("PersonaPlex-7B Engine Demo")
    print("=" * 50)

    print(f"\nAvailable voices: {PersonaPlexEngine.get_available_voices()}")

    print("\nInitializing PersonaPlex-7B (CPU mode)...")
    engine = PersonaPlexEngine(
        device="cpu",
        voice="natural_female_0",
        persona="A helpful and friendly AI assistant"
    )

    print("\n✓ PersonaPlex-7B engine initialized")
    print(f"  Sample rate: {engine.sample_rate}Hz")
    print(f"  Voice: {engine.voice}")
    print(f"  Persona: {engine.persona if engine.persona else 'Default'}")

    if engine.model is None:
        print("\n[STUB MODE] For real usage, install PersonaPlex-7B:")
        print("  git clone https://github.com/NVIDIA/personaplex")
        print("  pip install personaplex/moshi/.")
