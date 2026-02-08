"""
Local Voice Orchestrator - Three-Mode Architecture

Velloris achieves versatile voice AI through three specialized modes:

1. Real-Time Mode: PersonaPlex-7B (end-to-end S2S)
   * Input: User audio
   * Output: Agent audio response (70-170ms latency)
   * Features: Full-duplex, interruptions, 16 voices, persona control
   * Best for: Interactive conversations, customer service, tutoring

2. Dubbing Mode: Qwen3-TTS (high-fidelity synthesis)
   * Input: Script text
   * Output: Professional-quality audio
   * Features: 10 languages, emotion control, voice cloning, voice design
   * Best for: Content creation, narration, audiobooks, multilingual

3. Creative Mode: Ollama LLM + Qwen3-TTS (emotional synthesis)
   * Input: User text
   * Output: Emotionally expressive audio
   * Features: LLM reasoning, emotion control, multilingual
   * Best for: Storytelling, creative content, brainstorming

Manages engine lifecycle with lazy loading for memory efficiency.
"""

import numpy as np
from typing import Optional, Tuple
from pathlib import Path
import sys

from engines.personaplex import PersonaPlexEngine
from utils.device_utils import get_optimal_device, get_platform_info

# Conditional import for TTS engine based on platform
if sys.platform == "darwin":
    from engines.mlx_tts import MLXTTSEngine as TTSEngine
else:
    from engines.qwen_tts import Qwen3TTSEngine as TTSEngine


class LocalVoiceOrchestrator:
    """
    Orchestrates voice processing across three specialized modes.

    Modes:
    - realtime: PersonaPlex end-to-end S2S (no LLM needed)
    - dubbing: Qwen3-TTS/MLX-TTS high-fidelity synthesis
    - creative: Ollama + Qwen3-TTS/MLX-TTS (emotional content)

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
            llm_model: Ollama model name (only used in creative mode)
        """
        self.device = get_optimal_device(device)
        self.llm_model = llm_model
        self.platform_info = get_platform_info()

        # Engine instances (lazy-loaded)
        self.personaplex_engine: Optional[PersonaPlexEngine] = None
        self.tts_engine: Optional[TTSEngine] = None
        self.ollama_brain = None  # Lazy-loaded for creative mode

        print(f"[CFG] Orchestrator initialized on {self.device.upper()}")
        print(
            f"   Platform: {self.platform_info['os']} ({self.platform_info['machine']})"
        )
        tts_backend = "MLX-Audio" if sys.platform == "darwin" else "Qwen3-TTS"
        print(f"   TTS Backend: {tts_backend}")
        print("   Modes available:")
        print("     • realtime: PersonaPlex-7B (end-to-end S2S, no LLM)")
        print(f"     • dubbing: {tts_backend} (high-fidelity narration)")
        print(
            f"     • creative: {self.llm_model} + {tts_backend} (emotional synthesis)"
        )

    def _load_personaplex(self):
        """Lazy-load PersonaPlex engine if not already loaded."""
        if self.personaplex_engine is not None:
            return

        print("\n[LOAD] Loading PersonaPlex engine for realtime mode...")
        try:
            self.personaplex_engine = PersonaPlexEngine(
                device=self.device, llm_model=self.llm_model
            )
            print("[OK] PersonaPlex ready")
        except Exception as e:
            print(f"[X] Failed to load PersonaPlex: {e}")

    def _load_tts_engine(self):
        """Lazy-load the appropriate TTS engine for the platform."""
        if self.tts_engine is not None:
            return

        tts_backend = "MLX-Audio" if sys.platform == "darwin" else "Qwen3-TTS"
        print(f"\n[LOAD] Loading {tts_backend} engine...")
        try:
            self.tts_engine = TTSEngine(device=self.device)
            print(f"[OK] {tts_backend} ready")
        except Exception as e:
            print(f"[X] Failed to load {tts_backend}: {e}")

    def _load_ollama(self):
        """Lazy-load Ollama brain for creative mode."""
        if self.ollama_brain is not None:
            return

        print(f"\n[LOAD] Loading Ollama ({self.llm_model}) for creative mode...")
        try:
            from langchain_ollama import OllamaLLM

            # Test if Ollama is accessible
            test_llm = OllamaLLM(model=self.llm_model)
            # Simple test to see if it responds
            test_llm.invoke("Hi")
            self.ollama_brain = test_llm
            print(f"[OK] Ollama {self.llm_model} ready")
        except Exception as e:
            print(f"[X] Failed to connect to Ollama: {e}")
            print("   Make sure Ollama is running: ollama serve")
            print(f"   And model is available: ollama pull {self.llm_model}")
            self.ollama_brain = None

    def unload_engines(self):
        """Unload all engines to free memory."""
        if self.personaplex_engine is not None:
            self.personaplex_engine.unload()
            self.personaplex_engine = None

        if self.tts_engine is not None:
            self.tts_engine.unload()
            self.tts_engine = None

        if self.ollama_brain is not None:
            self.ollama_brain = None

        print("[OK] All engines unloaded")

    def route_request(
        self,
        mode: str = "realtime",
        text: Optional[str] = None,
        audio_input: Optional[np.ndarray] = None,
        ref_audio_path: Optional[str] = None,
        voice_prompt: Optional[str] = None,
        text_prompt: Optional[str] = None,
        emotion: Optional[str] = None,
        **kwargs,
    ) -> Optional[Tuple[np.ndarray, int]]:
        """
        Route a request to the appropriate engine based on mode.

        Args:
            mode: Operating mode
                - 'realtime': PersonaPlex end-to-end S2S (requires audio_input)
                - 'dubbing': Qwen3-TTS/MLX-TTS high-fidelity narration (requires text)
                - 'creative': Ollama + Qwen3-TTS/MLX-TTS emotional synthesis (requires text)
            text: Input text (for dubbing/creative modes)
            audio_input: User audio (for realtime mode)
            ref_audio_path: Optional reference audio for voice cloning
            voice_prompt: Voice file for PersonaPlex (e.g., "NATF2.pt")
            text_prompt: Persona/role description for PersonaPlex
            emotion: Emotion instruction for TTS (creative mode)

        Returns:
            Tuple of (audio_array, sample_rate) or None

        Examples:
            # Real-time conversation
            >>> audio, sr = orchestrator.route_request(
            ...     mode="realtime",
            ...     audio_input=user_audio,
            ...     voice_prompt="NATF2.pt",
            ...     text_prompt="You are a helpful tutor"
            ... )

            # Dubbing/narration
            >>> audio, sr = orchestrator.route_request(
            ...     mode="dubbing",
            ...     text="Once upon a time..."
            ... )

            # Creative storytelling
            >>> audio, sr = orchestrator.route_request(
            ...     mode="creative",
            ...     text="Tell me a story about space",
            ...     emotion="Speak with excitement"
            ... )
        """
        if mode == "realtime":
            if audio_input is None:
                print("[X] realtime mode requires audio_input")
                return None
            return self._handle_realtime(audio_input, voice_prompt, text_prompt)

        elif mode == "dubbing":
            if text is None:
                print("[X] dubbing mode requires text")
                return None
            return self._handle_dubbing(text, ref_audio_path)

        elif mode == "creative":
            if text is None:
                print("[X] creative mode requires text")
                return None
            return self._handle_creative(text, emotion, ref_audio_path)

        else:
            print(f"[X] Unknown mode: {mode}")
            print("   Available modes: 'realtime', 'dubbing', 'creative'")
            return None

    def _handle_realtime(
        self,
        audio_input: np.ndarray,
        voice_prompt: Optional[str] = None,
        text_prompt: Optional[str] = None,
    ) -> Optional[Tuple[np.ndarray, int]]:
        """
        Handle real-time mode: PersonaPlex end-to-end S2S.

        PersonaPlex does EVERYTHING:
        - Listens to user speech
        - Understands meaning
        - Generates intelligent response
        - Speaks response naturally

        No separate LLM or TTS needed!

        Args:
            audio_input: User audio (24kHz preferred)
            voice_prompt: Voice file (e.g., "NATF2.pt")
            text_prompt: Persona/role description

        Returns:
            Tuple of (agent_audio, sample_rate) or None
        """
        self._load_personaplex()

        if self.personaplex_engine is None:
            print("[X] PersonaPlex engine not available")
            return None

        print("\n🎯 [REALTIME MODE] PersonaPlex end-to-end S2S")
        print(f"   Input: {len(audio_input)/24000:.2f}s of user audio")

        try:
            # PersonaPlex handles the complete S2S pipeline
            result = self.personaplex_engine.generate_s2s_response(
                audio=audio_input,
                sr=24000,
                voice_prompt=voice_prompt,
                text_prompt=text_prompt,
                streaming=False,  # TODO: Enable streaming for full-duplex
            )

            if result:
                audio, sr = result
                print(f"[OK] Generated: {len(audio) / sr:.2f}s of agent speech")
                return audio, sr
            else:
                print("[X] Failed to generate S2S response")
                return None

        except Exception as e:
            print(f"[X] Error: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _handle_creative(
        self,
        text: str,
        emotion: Optional[str] = None,
        ref_audio_path: Optional[str] = None,
    ) -> Optional[Tuple[np.ndarray, int]]:
        """
        Handle creative mode: Ollama LLM + TTS emotional synthesis.

        Pipeline:
        1. User text → Ollama LLM (reasoning/creativity)
        2. LLM response → TTS (emotional synthesis)
        3. Audio output

        Args:
            text: User input text
            emotion: Emotion instruction for TTS
            ref_audio_path: Optional voice reference

        Returns:
            Tuple of (audio, sample_rate) or None
        """
        self._load_ollama()
        self._load_tts_engine()

        if self.ollama_brain is None:
            print("[X] Ollama not available. Is ollama running?")
            print("   Start with: ollama serve")
            return None

        if self.tts_engine is None:
            print("[X] TTS engine not available")
            return None

        tts_backend = "MLX-Audio" if sys.platform == "darwin" else "Qwen3-TTS"
        print(f"\n🎯 [CREATIVE MODE] Ollama + {tts_backend}")
        print(f"   Input: {text[:100]}{'...' if len(text) > 100 else ''}")

        try:
            # Step 1: Generate creative response with Ollama
            print(f"   🧠 Generating response with {self.llm_model}...")
            from langchain_ollama import OllamaLLM

            llm = OllamaLLM(model=self.llm_model)
            response_text = llm.invoke(text)
            print(
                f"   LLM response: {response_text[:100]}{'...' if len(response_text) > 100 else ''}"
            )

            # Step 2: Synthesize with emotion control
            print(f"   [MIC]  Synthesizing with {tts_backend}...")
            if emotion:
                print(f"   Emotion: {emotion}")

            result = self.tts_engine.generate_dubbing(
                text=response_text,
                ref_audio_path=ref_audio_path,
                language="english",
                instruct=emotion or "",
            )

            if result:
                audio, sr = result
                print(f"[OK] Generated: {len(audio) / sr:.2f}s of emotional speech")
                return audio, sr
            else:
                print("[X] Failed to generate emotional speech")
                return None

        except Exception as e:
            print(f"[X] Error: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _handle_dubbing(
        self, text: str, ref_audio_path: Optional[str] = None
    ) -> Optional[Tuple[np.ndarray, int]]:
        """
        Handle dubbing mode: TTS high-fidelity narration.

        Direct text-to-speech synthesis without LLM.
        Best for: Audiobooks, podcasts, video narration, content creation.

        Args:
            text: Script to narrate
            ref_audio_path: Reference audio for voice cloning

        Returns:
            Tuple of (audio, sample_rate) or None
        """
        self._load_tts_engine()

        if self.tts_engine is None:
            print("[X] TTS engine not available")
            return None

        tts_backend = "MLX-Audio" if sys.platform == "darwin" else "Qwen3-TTS"
        print(f"\n🎯 [DUBBING MODE] {tts_backend} High-Fidelity")
        print(f"   Script: {text[:100]}{'...' if len(text) > 100 else ''}")

        if ref_audio_path:
            ref_path = Path(ref_audio_path)
            if ref_path.exists():
                print(f"   Voice reference: {ref_audio_path}")
            else:
                print("   [!]  Reference not found, using default voice")

        try:
            result = self.tts_engine.generate_dubbing(
                text, ref_audio_path, language="english"
            )

            if result:
                audio, sr = result
                print(f"[OK] Generated: {len(audio) / sr:.2f}s of high-fidelity audio")
                return audio, sr
            else:
                print("[X] Failed to generate dubbing")
                return None

        except Exception as e:
            print(f"[X] Error: {e}")
            import traceback

            traceback.print_exc()
            return None


if __name__ == "__main__":
    # Test the orchestrator
    orchestrator = LocalVoiceOrchestrator(device="cpu", llm_model="llama3")

    # Test dubbing mode
    print("\n" + "=" * 50)
    print("Test 1: Dubbing Mode")
    print("=" * 50)
    script = "Once upon a time in a digital landscape, models lived in harmony."
    result = orchestrator.route_request(mode="dubbing", text=script)

    # Test creative mode
    print("\n" + "=" * 50)
    print("Test 2: Creative Mode")
    print("=" * 50)
    result = orchestrator.route_request(
        mode="creative", text="Hello, how are you today?"
    )

    # Cleanup
    orchestrator.unload_engines()
