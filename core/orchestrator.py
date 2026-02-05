"""
Local Voice Orchestrator - Dual-Engine Architecture

By orchestrating the real-time interaction of PersonaPlex-7B with the expressive
"Voice Design" of Qwen3-TTS, Velloris achieves human-level conversation without the cloud.

Routes voice requests to specialized engines:
- Interactive Mode: PersonaPlex-7B (NVIDIA's real-time speech-to-speech)
  * Input: User audio + persona text
  * Output: Agent audio response
  * Features: Full-duplex conversations, interruptions, voice conditioning

- Dubbing Mode: Qwen3-TTS with Voice Design (Alibaba's expressive synthesis)
  * Input: Script text
  * Output: High-fidelity audio with voice cloning support
  * Features: Emotion control, voice design, multilingual

Manages engine lifecycle with lazy loading for memory efficiency.
"""

import torch
import numpy as np
from typing import Optional, Tuple
from pathlib import Path

from engines.personaplex import PersonaPlexEngine
from engines.qwen_tts import Qwen3TTSEngine
from utils.device_utils import get_optimal_device, get_platform_info


class LocalVoiceOrchestrator:
    """
    Orchestrates voice processing across multiple engines.

    Routes requests to PersonaPlex (interactive/real-time) or
    Qwen3-TTS (dubbing/high-fidelity) based on mode.

    Features:
    - Lazy loading of models (only load when needed)
    - Memory management (unload when not in use)
    - Mode-based routing
    - Voice cloning support
    """

    def __init__(self, device: str = "auto", llm_model: str = "llama3"):
        """
        Initialize the orchestrator.

        Args:
            device: Device to use ('cuda', 'mps', 'cpu', or 'auto' for auto-detection)
            llm_model: Ollama model name
        """
        self.device = get_optimal_device(device)
        self.llm_model = llm_model
        self.platform_info = get_platform_info()

        # Engine instances (lazy-loaded)
        self.personaplex_engine: Optional[PersonaPlexEngine] = None
        self.qwen3_engine: Optional[Qwen3TTSEngine] = None

        print(f"🔧 Orchestrator initialized on {self.device.upper()}")
        print(f"   Platform: {self.platform_info['os']} ({self.platform_info['machine']})")
        print(f"   LLM: {self.llm_model}")
        print(f"   Interactive mode: PersonaPlex (Whisper + Ollama + Qwen3-TTS)")
        print(f"   Dubbing mode: Qwen3-TTS (from Hugging Face)")

    def _load_personaplex(self):
        """Lazy-load PersonaPlex engine if not already loaded."""
        if self.personaplex_engine is not None:
            return

        print("\n📦 Loading PersonaPlex engine for interactive mode...")
        try:
            self.personaplex_engine = PersonaPlexEngine(
                device=self.device, llm_model=self.llm_model
            )
            print("✓ PersonaPlex ready")
        except Exception as e:
            print(f"✗ Failed to load PersonaPlex: {e}")

    def _load_qwen3(self):
        """Lazy-load Qwen3-TTS engine if not already loaded."""
        if self.qwen3_engine is not None:
            return

        print("\n📦 Loading Qwen3-TTS engine for dubbing mode...")
        try:
            self.qwen3_engine = Qwen3TTSEngine(device=self.device)
            print("✓ Qwen3-TTS ready")
        except Exception as e:
            print(f"✗ Failed to load Qwen3-TTS: {e}")

    def unload_engines(self):
        """Unload all engines to free memory."""
        if self.personaplex_engine is not None:
            self.personaplex_engine.unload()
            self.personaplex_engine = None

        if self.qwen3_engine is not None:
            # Coqui TTS doesn't have explicit unload, just set to None
            self.qwen3_engine = None

        print("✓ All engines unloaded")

    def route_request(
        self, text: str, mode: str = "interactive", ref_audio_path: Optional[str] = None
    ) -> Optional[Tuple[np.ndarray, int]]:
        """
        Route a request to the appropriate engine.

        Args:
            text: Input text or script
            mode: 'interactive' for real-time, 'dubbing' for high-fidelity
            ref_audio_path: Optional reference audio for voice cloning

        Returns:
            Tuple of (audio_array, sample_rate) or None
        """
        if mode == "interactive":
            return self._run_personaplex(text, ref_audio_path)
        elif mode == "dubbing":
            return self._run_qwen3(text, ref_audio_path)
        else:
            print(f"Unknown mode: {mode}")
            return None

    def _run_personaplex(
        self, text: str, ref_audio_path: Optional[str] = None
    ) -> Optional[Tuple[np.ndarray, int]]:
        """
        Run PersonaPlex engine for interactive mode.

        Args:
            text: Text/script to process
            ref_audio_path: Optional reference for voice cloning

        Returns:
            Tuple of (audio, sample_rate) or None
        """
        self._load_personaplex()

        if self.personaplex_engine is None:
            print("✗ PersonaPlex engine not available")
            return None

        print(f"\n🎯 [INTERACTIVE MODE] Running PersonaPlex")
        print(f"   Input: {text[:100]}{'...' if len(text) > 100 else ''}")

        try:
            # For now, just generate speech from the text directly
            # In a real scenario with audio input, we'd transcribe first
            result = self.personaplex_engine.generate_speech(text, ref_audio_path)

            if result:
                audio, sr = result
                print(f"   Generated: {len(audio) / sr:.2f}s of audio")
                return audio, sr
            else:
                print("   ✗ Failed to generate speech")
                return None

        except Exception as e:
            print(f"   ✗ Error: {e}")
            return None

    def _run_qwen3(
        self, text: str, ref_audio_path: Optional[str] = None
    ) -> Optional[Tuple[np.ndarray, int]]:
        """
        Run Qwen3-TTS engine for dubbing mode.

        Args:
            text: Script to narrate
            ref_audio_path: Reference audio for voice cloning

        Returns:
            Tuple of (audio, sample_rate) or None
        """
        self._load_qwen3()

        if self.qwen3_engine is None:
            print("✗ Qwen3-TTS engine not available")
            return None

        print(f"\n🎯 [DUBBING MODE] Running Qwen3-TTS")
        print(f"   Script: {text[:100]}{'...' if len(text) > 100 else ''}")

        if ref_audio_path:
            ref_path = Path(ref_audio_path)
            if ref_path.exists():
                print(f"   Voice reference: {ref_audio_path}")
            else:
                print(f"   Warning: Reference not found, using default voice")

        try:
            result = self.qwen3_engine.generate_dubbing(
                text, ref_audio_path, language="en"
            )

            if result:
                audio, sr = result
                print(f"   Generated: {len(audio) / sr:.2f}s of high-fidelity audio")
                return audio, sr
            else:
                print("   ✗ Failed to generate dubbing")
                return None

        except Exception as e:
            print(f"   ✗ Error: {e}")
            return None


if __name__ == "__main__":
    # Test the orchestrator
    orchestrator = LocalVoiceOrchestrator(device="cpu", llm_model="llama3")

    # Test interactive mode
    print("\n" + "=" * 50)
    print("Test 1: Interactive Mode")
    print("=" * 50)
    result = orchestrator.route_request(
        "Hello, how are you today?", mode="interactive"
    )

    # Test dubbing mode
    print("\n" + "=" * 50)
    print("Test 2: Dubbing Mode")
    print("=" * 50)
    script = "Once upon a time in a digital landscape, models lived in harmony."
    result = orchestrator.route_request(script, mode="dubbing")

    # Cleanup
    orchestrator.unload_engines()
