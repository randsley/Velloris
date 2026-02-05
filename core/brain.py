"""
Voice Agent Brain

Processes voice turns and coordinates between transcription, LLM reasoning, and speech synthesis.

The brain integrates with:
- Orchestrator for engine routing
- LLM (Ollama) for reasoning
- Optional TTS engine for speech generation
"""

import asyncio
import numpy as np
from typing import Optional, AsyncIterator, Tuple
from langchain_community.llms import Ollama


class VoiceAgentBrain:
    """
    Core brain for voice agent reasoning and response generation.

    Coordinates:
    - Audio transcription (via orchestrator)
    - LLM reasoning (via Ollama)
    - Speech synthesis (via TTS engine)
    """

    def __init__(self, model_name: str = "llama3", tts_engine=None, orchestrator=None):
        """
        Initialize the brain.

        Args:
            model_name: Ollama model name
            tts_engine: Optional TTS engine for speech generation
            orchestrator: Optional orchestrator for complex routing
        """
        self.llm = Ollama(model=model_name)
        self.tts_engine = tts_engine  # Optional TTS engine (Qwen3 or Coqui)
        self.orchestrator = orchestrator  # Optional orchestrator for routing

    async def process_voice_turn(self, user_text: str) -> Tuple[str, Optional[np.ndarray]]:
        """
        Takes user text, generates LLM response, and optionally synthesizes audio.

        Args:
            user_text: Transcribed user input

        Returns:
            Tuple of (response_text, audio_response) where audio_response may be None
        """
        print(f"Agent Thinking...")

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
            print(f"LLM error: {e}")
            full_response = "I encountered an error processing your request."

        print(f"Response: {full_response}")

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
                result = self.orchestrator.route_request(full_response, mode="dubbing")
                if result:
                    audio_response = result[0]  # Extract audio array from (audio, sr)
            except Exception as e:
                print(f"TTS error: {e}")

        return full_response, audio_response

    async def stream_tokens(self, user_text: str) -> AsyncIterator[str]:
        """
        Stream LLM tokens one-by-one for progressive generation.

        Args:
            user_text: User input

        Yields:
            Individual tokens from the LLM
        """
        try:
            # LangChain's astream returns individual tokens
            async for token in self.llm.astream(user_text):
                yield token
        except Exception as e:
            print(f"Streaming error: {e}")
            yield ""

    async def process_audio_turn(
        self, audio: np.ndarray, sr: int = 16000, ref_audio_path: Optional[str] = None
    ) -> Tuple[str, Optional[np.ndarray]]:
        """
        Process a complete audio turn: transcribe, reason, synthesize.

        Args:
            audio: Audio samples
            sr: Sample rate
            ref_audio_path: Optional reference for voice cloning

        Returns:
            Tuple of (response_text, audio_response)
        """
        if self.orchestrator is None:
            print("No orchestrator available for audio processing")
            return "", None

        # Step 1: Transcribe audio using PersonaPlex
        user_text = self.orchestrator.personaplex_engine.transcribe_audio(audio, sr)
        print(f"Transcribed: {user_text}")

        if not user_text:
            return "", None

        # Step 2: Process the transcribed text
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
