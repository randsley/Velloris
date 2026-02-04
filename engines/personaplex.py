"""
PersonaPlex Engine Implementation

PersonaPlex is a unified Speech-to-Speech (S2S) model for real-time voice interactions.

Current Implementation:
- Uses Whisper for STT (speech-to-text)
- Uses LangChain/Ollama for LLM reasoning
- Uses Coqui TTS for TTS (text-to-speech)

This is a placeholder until the official PersonaPlex model becomes available.
The interface remains the same for easy migration.
"""

import torch
import numpy as np
import asyncio
from typing import Optional, Tuple, AsyncIterator

try:
    import whisper
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False

try:
    from transformers import AutoModel, AutoTokenizer
    HAS_QWEN3_TTS = True
except ImportError:
    HAS_QWEN3_TTS = False

from langchain_community.llms import Ollama


class PersonaPlexEngine:
    """
    PersonaPlex Speech-to-Speech Engine

    Implements a complete voice interaction pipeline:
    Audio Input -> Transcription -> LLM -> TTS -> Audio Output

    Features:
    - Real-time speech recognition with Whisper
    - Language model reasoning with Ollama
    - High-fidelity speech synthesis with Coqui TTS
    - Voice cloning support
    - Barge-in (interruption) ready architecture
    """

    def __init__(self, device="cuda", llm_model="llama3", whisper_model="base", tts_model="v2"):
        """
        Initialize PersonaPlex engine.

        Args:
            device: 'cuda' or 'cpu'
            llm_model: Ollama model name (e.g., 'llama3', 'neural-chat')
            whisper_model: Whisper model size (tiny, base, small, medium, large)
            tts_model: TTS model variant (v2 for XTTS-v2)
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.llm_model = llm_model
        self.whisper_model_name = whisper_model
        self.tts_model_name = tts_model

        # Initialize components
        self.whisper_model = None
        self.llm = None
        self.tts_model = None

        self._load_models()

    def _load_models(self):
        """Load all required models."""
        self._load_whisper()
        self._load_llm()
        self._load_tts()

    def _load_whisper(self):
        """Load Whisper STT model."""
        if not HAS_WHISPER:
            print("WARNING: Whisper not available. STT will not work.")
            return

        try:
            print(f"Loading Whisper ({self.whisper_model_name}) for STT...")
            self.whisper_model = whisper.load_model(self.whisper_model_name, device=self.device)
            print("✓ Whisper loaded successfully")
        except Exception as e:
            print(f"Failed to load Whisper: {e}")

    def _load_llm(self):
        """Load LLM through Ollama."""
        try:
            print(f"Loading LLM ({self.llm_model}) through Ollama...")
            self.llm = Ollama(model=self.llm_model)
            print("✓ Ollama LLM loaded successfully")
        except Exception as e:
            print(f"Failed to load Ollama LLM: {e}")
            print("Make sure Ollama is running: ollama serve")

    def _load_tts(self):
        """Load Qwen3-TTS model."""
        if not HAS_QWEN3_TTS:
            print("WARNING: transformers not available. TTS will not work.")
            return

        try:
            print(f"Loading Qwen3-TTS for speech synthesis...")
            # Import here to avoid circular dependency
            from engines.qwen_tts import Qwen3TTSEngine

            self.tts_model = Qwen3TTSEngine(
                model_size="1.7B-CustomVoice",
                device=self.device,
            )
            print("✓ Qwen3-TTS loaded successfully")
        except Exception as e:
            print(f"Failed to load Qwen3-TTS: {e}")
            self.tts_model = None

    def transcribe_audio(self, audio: np.ndarray, sr: int = 16000) -> str:
        """
        Transcribe audio to text using Whisper.

        Args:
            audio: Audio samples as numpy array
            sr: Sample rate

        Returns:
            Transcribed text
        """
        if self.whisper_model is None:
            print("[STUB] Would transcribe audio")
            return ""

        try:
            # Convert to audio format expected by Whisper
            # Whisper expects 16kHz audio
            if sr != 16000:
                from utils.audio_utils import resample_audio
                audio = resample_audio(audio, sr, 16000)

            # Whisper can work with numpy arrays directly
            result = self.whisper_model.transcribe(audio, language="en", verbose=False)
            text = result.get("text", "").strip()
            return text
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""

    def generate_speech(
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
            ref_audio_path: Optional path to reference audio for voice cloning
            emotion: Optional emotion/style prompt (e.g., "happy", "sad", "neutral")
            language: Language code

        Returns:
            Tuple of (audio_array, sample_rate) or None
        """
        if self.tts_model is None:
            print(f"[STUB] Would generate speech: {text}")
            return None

        try:
            result = self.tts_model.generate_dubbing(
                text=text,
                ref_audio_path=ref_audio_path,
                emotion=emotion,
                language=language,
            )

            return result

        except Exception as e:
            print(f"TTS generation error: {e}")
            return None

    async def stream_s2s(
        self,
        text_iterator: AsyncIterator[str],
        ref_audio_path: Optional[str] = None,
        language: str = "en",
    ) -> AsyncIterator[Tuple[np.ndarray, int]]:
        """
        Stream speech-to-speech processing using Qwen3-TTS.

        Takes a stream of text chunks and yields audio chunks.

        Args:
            text_iterator: Async iterator yielding text chunks
            ref_audio_path: Optional reference audio for voice cloning
            language: Language code

        Yields:
            Tuples of (audio_chunk, sample_rate)
        """
        if self.tts_model is None:
            print("TTS model not available")
            async for text_chunk in text_iterator:
                yield np.array([], dtype=np.float32), 12000
            return

        try:
            async for text_chunk in text_iterator:
                if not text_chunk.strip():
                    continue

                result = self.tts_model.generate_dubbing(
                    text=text_chunk,
                    ref_audio_path=ref_audio_path,
                    language=language,
                )

                if result:
                    yield result
                else:
                    yield np.array([], dtype=np.float32), 12000

        except Exception as e:
            print(f"Stream S2S error: {e}")

    def process_voice_turn(
        self, audio: np.ndarray, sr: int = 16000, ref_audio_path: Optional[str] = None
    ) -> Tuple[str, Optional[np.ndarray]]:
        """
        Process a complete voice turn: transcribe, generate LLM response, synthesize speech.

        Args:
            audio: Audio samples
            sr: Sample rate
            ref_audio_path: Optional reference for voice cloning

        Returns:
            Tuple of (text_response, audio_response)
        """
        # Step 1: Transcribe
        user_text = self.transcribe_audio(audio, sr)
        print(f"Transcribed: {user_text}")

        # Step 2: Generate LLM response
        if self.llm is None:
            llm_response = "I'm not ready to respond yet."
        else:
            try:
                llm_response = self.llm.invoke(user_text)
            except Exception as e:
                print(f"LLM error: {e}")
                llm_response = f"I encountered an error: {e}"

        print(f"LLM Response: {llm_response}")

        # Step 3: Synthesize speech
        audio_response = self.generate_speech(llm_response, ref_audio_path)

        return llm_response, audio_response

    def unload(self):
        """Unload all models to free memory."""
        self.whisper_model = None
        self.llm = None
        self.tts_model = None
        print("✓ PersonaPlex engine unloaded")


if __name__ == "__main__":
    # Test initialization
    engine = PersonaPlexEngine(device="cpu", llm_model="llama3")
    print("\n✓ PersonaPlex engine initialized successfully")

    # Test generate_speech (without loading heavy models if not needed)
    result = engine.generate_speech("Hello, this is a test.")
    if result:
        audio, sr = result
        print(f"Generated {len(audio) / sr:.2f}s of audio at {sr}Hz")
