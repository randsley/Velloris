import asyncio
import argparse
import sounddevice as sd
import numpy as np

from core.orchestrator import LocalVoiceOrchestrator
from core.brain import VoiceAgentBrain
from utils.vad_handler import InterruptionHandler
from utils.audio_io import IntegratedAudioController, play_ai_response

async def main():
    parser = argparse.ArgumentParser(description="Velloris: A Local-First, High-Fidelity Voice Agent Engine.")
    parser.add_argument("--mode", type=str, default="interactive",
                        help="Operating mode: 'interactive' for real-time agent, 'dubbing' for content generation.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for models (e.g., 'cuda', 'cpu', 'mps').")
    args = parser.parse_args()

    print(f"🚀 Starting Velloris in {args.mode} mode on {args.device}...")

    # Initialize components
    interruption_handler = InterruptionHandler(threshold=0.5)
    audio_controller = IntegratedAudioController(handler=interruption_handler)
    orchestrator = LocalVoiceOrchestrator(device=args.device)
    brain = VoiceAgentBrain(model_name="llama3") # Assuming Ollama 'llama3' is running

    if args.mode == "interactive":
        print("
--- Interactive Agent Mode ---")
        print("Say something to start the conversation. The AI will listen and respond.")
        print("You can interrupt the AI at any time.")

        # Main interactive loop
        try:
            # Start the audio stream
            # The IntegratedAudioController's start_session is blocking (sd.Stream),
            # so we'll need to run it in a separate thread or use a different async approach
            # For simplicity, this boilerplate will focus on the logic flow.
            # A full implementation would involve async handling of audio callbacks.

            # Mocking user input for demonstration
            user_input_queue = asyncio.Queue()
            
            # This is a highly simplified loop.
            # In a real app, `audio_controller.start_session()` would manage microphone input
            # and push transcribed text to user_input_queue.
            async def mock_user_input():
                while True:
                    text = await asyncio.to_thread(input, "You (type to speak): ")
                    if text.lower() == "exit":
                        break
                    await user_input_queue.put(text)
            
            # Start the mock user input in the background
            asyncio.create_task(mock_user_input())

            while True:
                user_text = await user_input_queue.get()
                if user_text.lower() == "exit":
                    break
                
                interruption_handler.reset() # Reset interruption status for new turn
                print(f"
User: {user_text}")

                # Process user input with the brain
                # The brain's process_voice_turn will use Qwen3-TTS for streaming response
                # We need to integrate the audio_controller's output callback with the brain's TTS streaming.
                
                # For this boilerplate, we'll simulate a response and use play_ai_response
                # In a real setup, brain.tts_engine would push audio to audio_controller.audio_queue
                
                # Simulate LLM thinking and generating text
                llm_response_text = f"That's an interesting thought, {user_text}. Let me elaborate..."
                print(f"Velloris (thinking): {llm_response_text}")

                # Simulate TTS generation and playing with barge-in
                # In a real scenario, the brain's streamer would feed chunks directly
                # to the audio_controller's output buffer or queue.
                
                # For demo, generate a dummy audio for play_ai_response
                # This would come from qwen3_tts.py generate_dubbing or stream_text_to_speech
                dummy_audio = np.random.rand(48000).astype(np.float32) # 2 seconds of dummy audio
                
                # Use play_ai_response to simulate playing the AI's speech with interrupt capability
                play_ai_response(dummy_audio, interruption_handler, samplerate=audio_controller.output_fs)

                if interruption_handler.is_interrupted:
                    print("Velloris was interrupted, waiting for your next input.")
                    interruption_handler.reset() # Prepare for next user input
                
        except KeyboardInterrupt:
            print("
Exiting interactive mode.")
        finally:
            print("Cleanup...")
            sd.stop()
            orchestrator = None
            brain = None
            interruption_handler = None
            audio_controller = None

    elif args.mode == "dubbing":
        print("
--- Dubbing Mode ---")
        # Example for dubbing:
        script = "Once upon a time in a digital landscape, models lived in harmony. One day, a curious user embarked on a quest to build a local-first voice agent, named Velloris."
        ref_audio_path = "voices/reference.wav" # Ensure this file exists
        emotion = "narrative and engaging"

        print(f"Generating high-fidelity dubbing for script: '{script}'")
        print(f"Using voice reference: '{ref_audio_path}' with emotion: '{emotion}'")

        # Call the orchestrator to use Qwen3-TTS for dubbing
        # This will return an audio tensor from qwen_tts.py's generate_dubbing
        # For boilerplate, we'll just print a message.
        orchestrator._run_qwen3(script, ref_audio_path) # Call the internal method for demo
        
        # Simulate playing the generated audio
        print("
Dubbing generation complete. Playing audio (simulated)...")
        dummy_audio = np.random.rand(96000).astype(np.float32) # 4 seconds of dummy audio
        sd.play(dummy_audio, samplerate=audio_controller.output_fs)
        sd.wait()
        print("Dubbing playback complete.")

if __name__ == "__main__":
    asyncio.run(main())
