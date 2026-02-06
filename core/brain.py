"""
Voice Agent Brain - Mode-Based Architecture

Processes voice turns based on operating mode:
- realtime: PersonaPlex end-to-end S2S (no brain needed - PersonaPlex has built-in reasoning)
- creative: Ollama LLM + Qwen3-TTS (brain coordinates LLM + TTS)
- dubbing: Direct Qwen3-TTS (no brain needed - just synthesis)

The brain is primarily used for creative mode.
"""

import asyncio
import numpy as np
from typing import Optional, AsyncIterator, Tuple


class VoiceAgentBrain:
    """
    Core brain for voice agent reasoning and response generation.

    Mode-based operation:
    - realtime: NOT USED (PersonaPlex handles everything)
    - creative: USED (coordinates Ollama + Qwen3-TTS)
    - dubbing: NOT USED (direct Qwen3-TTS)
    """

    def __init__(
        self,
        mode: str = "creative",
        model_name: str = "llama3",
        tts_engine=None,
        orchestrator=None
    ):
        """
        Initialize the brain.

        Args:
            mode: Operating mode ('realtime', 'creative', 'dubbing')
            model_name: Ollama model name (only used in creative mode)
            tts_engine: Optional TTS engine for speech generation
            orchestrator: Orchestrator for routing (required for all modes)
        """
        self.mode = mode
        self.model_name = model_name
        self.tts_engine = tts_engine
        self.orchestrator = orchestrator

        # Ollama only needed for creative mode
        self.llm = None
        if mode == "creative":
            try:
                from langchain_ollama import OllamaLLM
                self.llm = OllamaLLM(model=model_name)
                print(f"ℹ️  Brain initialized for creative mode with {model_name}")
            except Exception as e:
                print(f"⚠️  Failed to initialize Ollama: {e}")
                print(f"   Make sure Ollama is running: ollama serve")
        elif mode == "realtime":
            print(f"ℹ️  Brain initialized for realtime mode (PersonaPlex handles reasoning)")
        else:
            print(f"ℹ️  Brain initialized for {mode} mode")

    async def process_voice_turn(self, user_text: str) -> Tuple[str, Optional[np.ndarray]]:
        """
        Takes user text, generates LLM response, and optionally synthesizes audio.

        Only used in creative mode. For realtime mode, use PersonaPlex directly.

        Args:
            user_text: Transcribed user input

        Returns:
            Tuple of (response_text, audio_response) where audio_response may be None
        """
        if self.mode == "realtime":
            print("⚠️  WARNING: Brain.process_voice_turn() not needed in realtime mode!")
            print("   Use PersonaPlex.generate_s2s_response() directly for realtime S2S.")
            return user_text, None

        if self.mode != "creative":
            print(f"⚠️  WARNING: Brain only used in creative mode, not '{self.mode}'")
            return user_text, None

        if self.llm is None:
            print("❌ LLM not available. Is Ollama running?")
            return "Error: LLM not available", None

        print(f"🧠 Agent Thinking...")

        # Step 1: Generate LLM response
        try:
            # Use streaming to get tokens one-by-one for progressive TTS
            full_response = ""
            async for token in self.stream_tokens(user_text):
                full_response += token

                # Send tokens to TTS buffer immediately if engine available
                # Qwen3 starts synthesizing audio once it has enough context (usually 1-2 words)
                if self.tts_engine is not None:
                    try:
                        await self.tts_engine.push_text(token)
                    except Exception:
                        # TTS might not support streaming
                        pass

        except Exception as e:
            print(f"❌ LLM error: {e}")
            full_response = "I encountered an error processing your request."

        print(f"💬 Response: {full_response[:100]}{'...' if len(full_response) > 100 else ''}")

        # Step 2: Finalize TTS if streaming
        if self.tts_engine is not None:
            try:
                await self.tts_engine.finalize()
            except Exception:
                pass

        # Step 3: Generate audio if TTS or orchestrator available
        audio_response = None
        if self.tts_engine is not None:
            try:
                result = self.tts_engine.generate_dubbing(full_response)
                if result:
                    audio_response = result[0]  # Extract audio array from (audio, sr)
            except Exception as e:
                pass
        elif self.orchestrator is not None:
            # Use orchestrator's Qwen3-TTS for dubbing
            try:
                result = self.orchestrator.route_request(text=full_response, mode="dubbing")
                if result:
                    audio_response = result[0]  # Extract audio array from (audio, sr)
            except Exception as e:
                print(f"❌ TTS error: {e}")

        return full_response, audio_response

    async def stream_tokens(self, user_text: str) -> AsyncIterator[str]:
        """
        Stream LLM tokens one-by-one for progressive generation.

        Only used in creative mode with Ollama.

        Args:
            user_text: User input

        Yields:
            Individual tokens from the LLM
        """
        if self.llm is None:
            print("❌ LLM not available for streaming")
            yield ""
            return

        try:
            # LangChain's astream returns individual tokens
            async for token in self.llm.astream(user_text):
                yield token
        except Exception as e:
            print(f"❌ Streaming error: {e}")
            yield ""

    async def process_audio_turn(
        self, audio: np.ndarray, sr: int = 16000, ref_audio_path: Optional[str] = None
    ) -> Tuple[str, Optional[np.ndarray]]:
        """
        DEPRECATED: Use orchestrator.route_request() instead.

        For realtime mode: Use PersonaPlex end-to-end S2S directly
        For creative mode: This method can still be used, but orchestrator is preferred

        Args:
            audio: Audio samples
            sr: Sample rate
            ref_audio_path: Optional reference for voice cloning

        Returns:
            Tuple of (response_text, audio_response)
        """
        import warnings
        warnings.warn(
            "VoiceAgentBrain.process_audio_turn() is deprecated. "
            "Use orchestrator.route_request() with mode='realtime' or 'creative' instead.",
            DeprecationWarning,
            stacklevel=2
        )

        if self.orchestrator is None:
            print("❌ No orchestrator available for audio processing")
            return "", None

        if self.mode == "realtime":
            print("⚠️  For realtime mode, use orchestrator.route_request(mode='realtime', audio_input=...)")
            print("   This bypasses unnecessary transcription and uses PersonaPlex S2S directly.")
            # Use orchestrator's realtime mode
            result = self.orchestrator.route_request(
                mode="realtime",
                audio_input=audio
            )
            if result:
                audio_out, sr_out = result
                return "[PersonaPlex S2S Response]", audio_out
            return "", None

        # For creative mode, we need transcription first
        print("⚠️  Using deprecated transcription path. Consider using Whisper for STT.")
        user_text = self.orchestrator.personaplex_engine.transcribe_audio(audio, sr)
        print(f"Transcribed: {user_text}")

        if not user_text:
            return "", None

        # Process the transcribed text
        response_text, audio_response = await self.process_voice_turn(user_text)

        return response_text, audio_response

    def interrupt(self):
        """Signal the brain to stop processing (for barge-in support)."""
        if self.tts_engine is not None:
            try:
                self.tts_engine.interrupt()
            except Exception:
                pass
        print("Agent interrupted")


if __name__ == "__main__":
    # Test the brain
    print("VoiceAgentBrain initialized.")
    brain = VoiceAgentBrain(model_name="llama3")

    # Test process_voice_turn (requires Ollama running)
    # asyncio.run(brain.process_voice_turn("Tell me a short story about a space pirate."))
