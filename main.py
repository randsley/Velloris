"""
Velloris: A Local-First, High-Fidelity Voice Agent Engine

Main entry point for the dual-engine voice agent system.

Usage:
    python main.py --mode interactive
    python main.py --mode dubbing --script "Your script here"
"""

import asyncio
import argparse
import signal
import sys
import torch
import sounddevice as sd
import numpy as np
from pathlib import Path

from core.orchestrator import LocalVoiceOrchestrator
from core.brain import VoiceAgentBrain
from utils.vad_handler import InterruptionHandler
from utils.audio_io import IntegratedAudioController, play_ai_response
from config import Config


class VellorisApplication:
    """
    Main Velloris application.

    Manages lifecycle and coordinates all components:
    - Audio I/O controller
    - Voice Activity Detection
    - Voice orchestrator (PersonaPlex/Qwen3)
    - Agent brain (LLM reasoning)
    """

    def __init__(self, args):
        """
        Initialize the application.

        Args:
            args: Parsed command-line arguments
        """
        self.args = args
        self.running = True

        # Initialize components
        self.interruption_handler = InterruptionHandler(threshold=Config.vad.THRESHOLD)
        self.audio_controller = IntegratedAudioController(
            handler=self.interruption_handler, whisper_model=Config.model.WHISPER_MODEL
        )
        self.orchestrator = LocalVoiceOrchestrator(
            device=args.device, llm_model=args.llm_model
        )
        self.brain = VoiceAgentBrain(
            model_name=args.llm_model,
            orchestrator=self.orchestrator,
        )

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, sig, frame):
        """Handle graceful shutdown on Ctrl+C."""
        print("\n\n⏸ Shutting down Velloris...")
        self.cleanup()
        sys.exit(0)

    def cleanup(self):
        """Clean up resources."""
        self.running = False
        if hasattr(self, "orchestrator"):
            self.orchestrator.unload_engines()
        if hasattr(self, "audio_controller"):
            self.audio_controller.stop_transcription_worker()
        print("✓ Cleanup complete")

    async def run_interactive(self):
        """
        Run Velloris in interactive mode.

        Real-time voice conversation with the agent.
        User can interrupt the agent at any time.
        """
        print("\n" + "=" * 60)
        print("🎤 VELLORIS - INTERACTIVE MODE")
        print("=" * 60)
        print("Starting voice agent in real-time mode...")
        print("The agent will listen and respond to your voice.")
        print("Press Ctrl+C to exit.\n")

        try:
            # For now, show a demo with text input (since audio setup is complex)
            # In production, this would use the audio_controller.start_session()
            await self._demo_interactive_mode()

        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        except Exception as e:
            print(f"\n✗ Error in interactive mode: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()

    async def _demo_interactive_mode(self):
        """
        Demo interactive mode with text input.

        In a real implementation, this would use:
        - self.audio_controller.start_session()
        - Real audio transcription with Whisper
        - Real-time interruption handling
        """
        print("Demo Mode: Type your input (type 'quit' to exit)\n")

        while self.running:
            try:
                # Get user input asynchronously
                user_input = await asyncio.to_thread(
                    input, "You: "
                )

                if user_input.lower() in ["quit", "exit", "bye"]:
                    print("Goodbye!")
                    break

                if not user_input.strip():
                    continue

                # Process with brain
                print("\nVelloris: Processing...")
                response_text, audio_response = await self.brain.process_voice_turn(
                    user_input
                )

                print(f"Velloris: {response_text}\n")

                # If audio was generated, simulate playing it
                if audio_response is not None and len(audio_response) > 0:
                    print("[Audio would play here]")

                # Reset interruption status for next turn
                self.interruption_handler.reset()

            except EOFError:
                # Ctrl+D pressed
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error processing input: {e}")

    async def run_dubbing(self):
        """
        Run Velloris in dubbing mode.

        High-fidelity speech generation from script.
        Supports voice cloning with reference audio.
        """
        print("\n" + "=" * 60)
        print("🎬 VELLORIS - DUBBING MODE")
        print("=" * 60)

        script = self.args.script
        if not script:
            script = (
                "Once upon a time in a digital landscape, "
                "models lived in harmony. One day, a curious user "
                "embarked on a quest to build a local-first voice agent, "
                "named Velloris."
            )

        voice_ref = self.args.voice_ref
        if voice_ref and not Path(voice_ref).exists():
            print(f"⚠ Voice reference not found: {voice_ref}")
            voice_ref = None

        print(f"Script: {script[:100]}...")
        if voice_ref:
            print(f"Voice reference: {voice_ref}")
        print()

        try:
            # Process with orchestrator in dubbing mode
            print("Generating high-fidelity audio...")
            result = self.orchestrator.route_request(script, mode="dubbing", ref_audio_path=voice_ref)

            if result:
                audio, sr = result
                print(f"✓ Generated {len(audio) / sr:.2f} seconds of audio")
                print("[Audio would play here]")
            else:
                print("✗ Failed to generate audio")

        except Exception as e:
            print(f"✗ Error in dubbing mode: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()

    async def run(self):
        """Run the application based on mode."""
        print("\n🚀 Starting Velloris...")
        print(f"   Mode: {self.args.mode}")
        print(f"   Device: {self.args.device}")
        print(f"   LLM: {self.args.llm_model}")

        # Show platform info if verbose
        from utils.device_utils import get_platform_info
        platform_info = get_platform_info()
        print(f"   Platform: {platform_info['os']} ({platform_info['machine']})")

        if self.args.mode == "interactive":
            await self.run_interactive()
        elif self.args.mode == "dubbing":
            await self.run_dubbing()
        else:
            print(f"Unknown mode: {self.args.mode}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Velloris: A Local-First, High-Fidelity Voice Agent Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (real-time voice conversation)
  python main.py --mode interactive

  # Dubbing mode (high-fidelity speech generation)
  python main.py --mode dubbing --script "Your script here"

  # With custom voice reference
  python main.py --mode dubbing --script "Story" --voice-ref voices/my_voice.wav

  # With CPU device
  python main.py --mode interactive --device cpu

For full documentation, see README.md
        """,
    )

    # Mode arguments
    parser.add_argument(
        "--mode",
        type=str,
        default="interactive",
        choices=["interactive", "dubbing"],
        help="Operating mode: 'interactive' for real-time, 'dubbing' for content generation",
    )

    # Device arguments
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["cuda", "cpu", "mps", "auto"],
        help="Device to use: cuda (NVIDIA GPU), mps (Apple Metal), cpu (CPU), or auto (detect)",
    )

    # Model arguments
    parser.add_argument(
        "--llm-model",
        type=str,
        default=Config.model.OLLAMA_MODEL,
        help="Ollama model name (requires Ollama running)",
    )

    parser.add_argument(
        "--whisper-model",
        type=str,
        default=Config.model.WHISPER_MODEL,
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper STT model size",
    )

    # Dubbing mode arguments
    parser.add_argument(
        "--script",
        type=str,
        help="Script/text to narrate in dubbing mode",
    )

    parser.add_argument(
        "--voice-ref",
        type=str,
        help="Path to voice reference audio for cloning (3-5 seconds)",
    )

    # Configuration arguments
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config file",
    )

    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Show configuration and exit",
    )

    args = parser.parse_args()

    # Show config if requested
    if args.show_config:
        Config.print_config()
        return

    # Validate configuration
    if not Config.validate():
        print("\n✗ Configuration validation failed")
        return

    # Create and run application
    try:
        app = VellorisApplication(args)
        asyncio.run(app.run())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
