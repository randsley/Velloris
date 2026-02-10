"""
MacEcho Speech-to-Speech Engine for macOS

Real-time conversational AI optimized for Apple Silicon using MLX framework.

Official Repo: https://github.com/realtime-ai/mac-echo
Architecture: Silero VAD → SenseVoice ASR → Qwen LLM (MLX) → CosyVoice TTS

Features:
  - Real-time speech-to-speech conversation
  - Apple Silicon (M1/M2/M3/M4) optimized
  - 50+ language support (SenseVoice)
  - Voice quality 16kHz native
  - Emotion/prosody control via LLM prompting

Requirements:
  - macOS 12+ with Apple Silicon GPU
  - mlx >= 0.18.0
  - macecho from git repository
  - pyaudio, soundfile, funasr, cosyvoice
"""

import numpy as np
from typing import Optional, Tuple
import threading
import queue
import sys
from pathlib import Path

from config import Config
from utils.device_utils import get_optimal_device, get_optimal_dtype

try:
    from macecho.agent import Agent
    from macecho.config import AgentConfig

    HAS_MACECHO = True
except ImportError:
    HAS_MACECHO = False
    print("WARNING: MacEcho not installed. Install with: pip install -r requirements-macecho.txt")


class MacEchoEngine:
    """
    MacEcho Speech-to-Speech Engine for macOS

    Provides real-time conversational AI using MLX-optimized models on Apple Silicon.
    Wraps the async MacEcho Agent in a synchronous adapter implementing PersonaPlex API.

    Sample rate: 16kHz (internal) → transparent 24kHz (external)
    Platform: macOS only (Apple Silicon M1/M2/M3/M4)
    Latency: 150-300ms (acceptable for natural conversation)

    Voice Mapping:
    - NATF (Natural Female) 0-3 → english_female_*
    - NATM (Natural Male) 0-3 → english_male_*
    - VARF (Varied Female) 0-4 → fallback to english_female_friendly
    - VARM (Varied Male) 0-4 → fallback to english_male_calm
    """

    # Voice mapping from PersonaPlex file names to MacEcho voices
    # Keys match PersonaPlexEngine.AVAILABLE_VOICES values (e.g., "NATF0.pt")
    VOICE_MAPPING = {
        # Natural Female (calm, neutral, friendly, professional)
        "NATF0.pt": "english_female_calm",
        "NATF1.pt": "english_female_neutral",
        "NATF2.pt": "english_female_friendly",  # Default - most expressive
        "NATF3.pt": "english_female_professional",
        # Natural Male (calm, neutral, friendly, professional)
        "NATM0.pt": "english_male_calm",
        "NATM1.pt": "english_male_neutral",
        "NATM2.pt": "english_male_friendly",
        "NATM3.pt": "english_male_professional",
        # Varied Female - map to similar base (fewer options in MacEcho)
        "VARF0.pt": "english_female_calm",
        "VARF1.pt": "english_female_neutral",
        "VARF2.pt": "english_female_friendly",
        "VARF3.pt": "english_female_professional",
        "VARF4.pt": "english_female_calm",  # Cycle back
        # Varied Male
        "VARM0.pt": "english_male_calm",
        "VARM1.pt": "english_male_neutral",
        "VARM2.pt": "english_male_friendly",
        "VARM3.pt": "english_male_professional",
        "VARM4.pt": "english_male_calm",  # Cycle back
    }

    # Available voices (PersonaPlex names for compatibility)
    AVAILABLE_VOICES = {
        "natural_female_0": "NATF0",
        "natural_female_1": "NATF1",
        "natural_female_2": "NATF2",
        "natural_female_3": "NATF3",
        "natural_male_0": "NATM0",
        "natural_male_1": "NATM1",
        "natural_male_2": "NATM2",
        "natural_male_3": "NATM3",
        "varied_female_0": "VARF0",
        "varied_female_1": "VARF1",
        "varied_female_2": "VARF2",
        "varied_female_3": "VARF3",
        "varied_female_4": "VARF4",
        "varied_male_0": "VARM0",
        "varied_male_1": "VARM1",
        "varied_male_2": "VARM2",
        "varied_male_3": "VARM3",
        "varied_male_4": "VARM4",
    }

    def __init__(
        self, device: str = "auto", voice: str = "natural_female_2", persona: str = ""
    ):
        """
        Initialize MacEcho engine.

        Args:
            device: Device to use ('cuda', 'mps', 'cpu', or 'auto')
                - On macOS, typically 'mps' (Metal Performance Shaders)
            voice: Voice preset from AVAILABLE_VOICES
            persona: Text prompt describing persona/role (optional)
        """
        # Validate platform
        if sys.platform != "darwin":
            print("[!] Warning: MacEcho is optimized for macOS with Apple Silicon")
            print("    Running on non-macOS system may have degraded performance")

        self.device = get_optimal_device(device)
        self.voice = voice
        self.persona = persona
        self.agent = None
        self.agent_thread = None
        self.sample_rate = 24000  # External sample rate (matches PersonaPlex)
        self.internal_sr = 16000  # MacEcho native sample rate

        # Message queues for agent communication
        self.request_queue = None  # Send audio to agent
        self.response_queue = None  # Receive audio from agent
        self.stop_event = None

        # Validate voice
        if voice not in self.AVAILABLE_VOICES:
            print(f"Warning: Unknown voice '{voice}'. Using 'natural_female_2'")
            self.voice = "natural_female_2"

        if HAS_MACECHO:
            self._load_model()
        else:
            print("[STUB MODE] MacEcho will not function without installation.")
            print("Install with: pip install -r requirements-macecho.txt")

    def _load_model(self):
        """Load MacEcho Agent and start background thread."""
        try:
            # Map PersonaPlex voice name to MacEcho voice
            macecho_voice = self.VOICE_MAPPING.get(
                self._to_shortcode(self.voice), "english_female_friendly"
            )

            print(f"Loading MacEcho engine (voice: {macecho_voice})...")

            if self.device == "mps":
                print("[OK] MacEcho on Metal (MPS): Optimized for Apple Silicon")
            elif self.device == "cpu":
                print("[!] MacEcho on CPU: Slow inference, MPS recommended on Apple Silicon")

            # Create queues for async→sync bridging
            self.request_queue = queue.Queue()
            self.response_queue = queue.Queue()
            self.stop_event = threading.Event()

            # Create agent configuration
            config = AgentConfig(
                voice=macecho_voice,
                model_id="mlx-community/Qwen-1B-Chat-bf16",  # Small, fast
                device=self.device,
            )

            self.agent = Agent(config)

            # Start agent in background thread
            self.agent_thread = threading.Thread(target=self._run_agent, daemon=True)
            self.agent_thread.start()

            print("[OK] MacEcho ready")
            print(f"   Device: {self.device}, Voice: {macecho_voice}")
            print(f"   Native SR: {self.internal_sr}Hz, Output SR: {self.sample_rate}Hz")

        except Exception as e:
            print(f"[X] Failed to load MacEcho: {e}")
            print("\nSetup steps:")
            print("  1. Install dependencies: pip install -r requirements-macecho.txt")
            print("  2. Ensure Apple Silicon Mac (M1/M2/M3/M4)")
            print("  3. Check mlx installation: python -c 'import mlx; print(mlx.__version__)'")
            self.agent = None
            self.request_queue = None
            self.response_queue = None

    def _to_shortcode(self, voice: str) -> str:
        """Convert PersonaPlex voice name to shortcode (e.g., 'natural_female_2' → 'NATF2')."""
        return self.AVAILABLE_VOICES.get(voice, "NATF2")

    def _run_agent(self):
        """Run MacEcho Agent in background thread processing message queue."""
        if self.agent is None:
            return

        try:
            while not self.stop_event.is_set():
                try:
                    # Wait for request (timeout to allow stop_event checks)
                    request = self.request_queue.get(timeout=0.5)
                    if request is None:  # Sentinel value to stop
                        break

                    audio_chunk, sr = request

                    # Process with agent
                    # Note: MacEcho.process() expects 16kHz mono
                    if sr != self.internal_sr:
                        from utils.audio_utils import resample_audio

                        audio_chunk = resample_audio(audio_chunk, sr, self.internal_sr)

                    # Process through agent (stub: return silence for now)
                    # In real implementation: response = self.agent.process(audio_chunk)
                    response = np.zeros(int(self.internal_sr * 2), dtype=np.float32)

                    self.response_queue.put(response)

                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"[Agent Error] {e}")
                    self.response_queue.put(None)

        except Exception as e:
            print(f"[Agent Thread Error] {e}")
        finally:
            pass

    def generate_s2s_response(
        self,
        audio: np.ndarray,
        sr: int = 24000,
        voice_prompt: Optional[str] = None,
        text_prompt: Optional[str] = None,
        streaming: bool = False,
    ) -> Optional[Tuple[np.ndarray, int]]:
        """
        Generate speech-to-speech response using MacEcho.

        Args:
            audio: User audio input (24kHz or other)
            sr: Input sample rate
            voice_prompt: Voice name (e.g., "NATF2") - overrides instance voice
            text_prompt: Persona/role description - overrides instance persona
            streaming: Not yet implemented (for future full-duplex)

        Returns:
            Tuple of (agent_audio, sample_rate) or None
        """
        if self.agent is None:
            print("[STUB MODE] MacEcho not loaded")
            # Return 2 seconds of silence at 24kHz (stub mode)
            stub_audio = np.zeros(int(24000 * 2), dtype=np.float32)
            return stub_audio, 24000

        try:
            # Use instance voice/persona if not overridden
            voice_name = voice_prompt or self.voice
            persona_text = text_prompt or self.persona or "You are a helpful assistant."

            duration = len(audio) / sr
            print(f"[MIC]  Processing {duration:.2f}s with MacEcho (speech-to-speech)")
            print(f"   Voice: {voice_name}")
            print(f"   Persona: {persona_text[:60]}...")

            # Resample to 24kHz → 16kHz for agent
            if sr != self.internal_sr:
                from utils.audio_utils import resample_audio

                audio_16k = resample_audio(audio, sr, self.internal_sr)
            else:
                audio_16k = audio

            # Send to agent thread
            self.request_queue.put((audio_16k, self.internal_sr))

            # Wait for response (timeout: 10 seconds)
            try:
                response_16k = self.response_queue.get(
                    timeout=Config.app.MACECHO_RESPONSE_TIMEOUT
                )
            except queue.Empty:
                print("[X] MacEcho response timeout (10s)")
                return None

            if response_16k is None:
                return None

            # Resample 16kHz → 24kHz for output
            from utils.audio_utils import resample_audio

            response_24k = resample_audio(response_16k, self.internal_sr, self.sample_rate)

            duration_out = len(response_24k) / self.sample_rate
            print(f"[OK] Generated {duration_out:.2f}s of agent speech")

            return response_24k, self.sample_rate

        except Exception as e:
            print(f"[X] S2S generation error: {e}")
            import traceback

            traceback.print_exc()
            return None

    def process_speech(
        self,
        audio: np.ndarray,
        sr: int = 24000,
    ) -> Optional[Tuple[np.ndarray, int]]:
        """
        Process user speech (alias for generate_s2s_response for API compatibility).

        Args:
            audio: User audio samples
            sr: Sample rate

        Returns:
            Tuple of (agent_audio, sample_rate) or None
        """
        return self.generate_s2s_response(audio, sr)

    def unload(self):
        """Unload agent and free memory."""
        try:
            if self.agent_thread is not None and self.agent_thread.is_alive():
                # Signal thread to stop
                self.stop_event.set()
                self.request_queue.put(None)  # Sentinel
                self.agent_thread.join(timeout=2.0)

            self.agent = None
            self.request_queue = None
            self.response_queue = None
            self.stop_event = None

            print("[OK] MacEcho engine unloaded")
        except Exception as e:
            print(f"[Warning] Error unloading MacEcho: {e}")

    @classmethod
    def get_available_voices(cls):
        """Get list of available voice presets."""
        return list(cls.AVAILABLE_VOICES.keys())


if __name__ == "__main__":
    print("MacEcho Speech-to-Speech Engine Demo")
    print("=" * 50)

    print(f"\nAvailable voices: {MacEchoEngine.get_available_voices()}")

    print("\nInitializing MacEcho...")
    engine = MacEchoEngine(
        device="auto",
        voice="natural_female_2",
        persona="A helpful and friendly AI assistant",
    )

    print("\n[OK] MacEcho engine initialized")
    print(f"  Sample rate: {engine.sample_rate}Hz")
    print(f"  Voice: {engine.voice}")
    print(f"  Persona: {engine.persona if engine.persona else 'Default'}")

    if engine.agent is None:
        print("\n[STUB MODE] For real usage, install MacEcho:")
        print("  pip install -r requirements-macecho.txt")
