import asyncio
from langchain_community.llms import Ollama
# from qwen3_tts import Qwen3TTSStreamer # Specialized streaming wrapper

class VoiceAgentBrain:
    def __init__(self, model_name="llama3"):
        self.llm = Ollama(model=model_name)
        # self.tts_engine = Qwen3TTSStreamer(model="1.7B-VoiceDesign")
        self.tts_engine = None # Placeholder for now

    async def process_voice_turn(self, user_text):
        """
        Takes user text, generates LLM response, and streams audio simultaneously.
        """
        print(f"Agent Thinking...")
        
        # We use stream() to get tokens one-by-one
        full_response = ""
        async for token in self.llm.astream(user_text):
            full_response += token
            # Send tokens to Qwen3-TTS buffer immediately
            # Qwen3 starts synthesizing audio once it has enough context (usually 1-2 words)
            if self.tts_engine:
                await self.tts_engine.push_text(token)
            
        print(f"Response complete: {full_response}")
        if self.tts_engine:
            await self.tts_engine.finalize()

# Usage with the Model Router logic from previous steps
if __name__ == "__main__":
    brain = VoiceAgentBrain()
    asyncio.run(brain.process_voice_turn("Tell me a short story about a space pirate."))
