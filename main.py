"""
Velloris: A Local-First, High-Fidelity Voice Agent Engine

Main entry point for the three-mode voice agent system.

Usage:
    python main.py --mode realtime
    python main.py --mode dubbing --script "Your script here"
    python main.py --mode creative --emotion "Speak with excitement"
"""

import asyncio
import argparse
import signal
import sys
from pathlib import Path

from core.orchestrator import LocalVoiceOrchestrator
from core.brain import VoiceAgentBrain
from utils.vad_handler import InterruptionHandler
from utils.audio_io import IntegratedAudioController, play_audio
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

        # Initialize components based on mode
        self.orchestrator = LocalVoiceOrchestrator(
            device=args.device, llm_model=args.llm_model
        )

        # Brain only needed for creative mode
        if args.mode == "creative":
            self.brain = VoiceAgentBrain(
                mode=args.mode,
                model_name=args.llm_model,
                orchestrator=self.orchestrator,
            )
        else:
            self.brain = None  # Not needed for realtime or dubbing

        # Audio controller for realtime mode
        if args.mode == "realtime":
            self.interruption_handler = InterruptionHandler(
                threshold=Config.vad.THRESHOLD
            )
            self.audio_controller = IntegratedAudioController(
                handler=self.interruption_handler,
                whisper_model=Config.model.WHISPER_MODEL,
            )
        else:
            self.interruption_handler = None
            self.audio_controller = None

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
        if hasattr(self, "audio_controller") and self.audio_controller:
            self.audio_controller.stop_transcription_worker()
        print("✓ Cleanup complete")

    async def run_realtime(self):
        """
        Run Velloris in realtime mode.

        Real-time voice conversation with the agent using PersonaPlex S2S.
        User can interrupt the agent at any time.
        """
        print("\n" + "=" * 60)
        print("🎤 VELLORIS - REALTIME MODE")
        print("=" * 60)
        print("Starting voice agent in real-time mode...")
        print("The agent will listen and respond to your voice.")
        print("Press Ctrl+C to exit.\n")

        try:
            # For now, show a demo with text input (since audio setup is complex)
            # In production, this would use the audio_controller.start_session()
            await self._demo_realtime_mode()

        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        except Exception as e:
            print(f"\n[X] Error in realtime mode: {e}")
            import traceback

            traceback.print_exc()
        finally:
            self.cleanup()

    async def _demo_realtime_mode(self):
        """
        Demo realtime mode with text input.

        In a real implementation, this would use:
        - self.audio_controller.start_session()
        - Real audio transcription with Whisper
        - Real-time interruption handling
        """
        print("Demo Mode: Type your input (type 'quit' to exit)\n")

        while self.running:
            try:
                # Get user input asynchronously
                user_input = await asyncio.to_thread(input, "You: ")

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

                # If audio was generated, play it
                if audio_response is not None and len(audio_response) > 0:
                    print()
                    play_audio(audio_response, samplerate=24000)

                # Reset interruption status for next turn
                if self.interruption_handler:
                    self.interruption_handler.reset()

            except EOFError:
                # Ctrl+D pressed
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error processing input: {e}")

    async def run_creative(self):
        """
        Run Velloris in creative mode.

        LLM-powered emotional synthesis using Ollama + Qwen3-TTS.
        """
        print("\n" + "=" * 60)
        print("🎨 VELLORIS - CREATIVE MODE")
        print("=" * 60)
        print("Starting voice agent in creative mode...")
        print("Type your prompts and get emotional voice responses.")
        print("Press Ctrl+C to exit.\n")

        try:
            await self._demo_realtime_mode()  # Reuse the demo loop
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        except Exception as e:
            print(f"\n[X] Error in creative mode: {e}")
            import traceback

            traceback.print_exc()
        finally:
            self.cleanup()

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
            print(f"[!] Voice reference not found: {voice_ref}")
            voice_ref = None

        print(f"Script: {script[:100]}...")
        if voice_ref:
            print(f"Voice reference: {voice_ref}")
        print()

        try:
            # Process with orchestrator in dubbing mode
            print("Generating high-fidelity audio...")
            result = self.orchestrator.route_request(
                mode="dubbing", text=script, ref_audio_path=voice_ref
            )

            if result:
                audio, sr = result
                print(f"✓ Generated {len(audio) / sr:.2f} seconds of audio")

                # Play the generated audio
                print()
                play_audio(audio, samplerate=sr)
            else:
                print("[X] Failed to generate audio")

        except Exception as e:
            print(f"[X] Error in dubbing mode: {e}")
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

        # Show platform info
        from utils.device_utils import get_platform_info

        platform_info = get_platform_info()
        print(f"   Platform: {platform_info['os']} ({platform_info['machine']})")

        if self.args.mode == "realtime":
            await self.run_realtime()
        elif self.args.mode == "creative":
            await self.run_creative()
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
  # Realtime mode (PersonaPlex S2S, ultra-low latency)
  python main.py --mode realtime

  # Creative mode (Ollama + Qwen3-TTS, emotional synthesis)
  python main.py --mode creative --emotion "Speak with excitement"

  # Dubbing mode (high-fidelity speech generation)
  python main.py --mode dubbing --script "Your script here"

  # With custom voice reference
  python main.py --mode dubbing --script "Story" --voice-ref voices/my_voice.wav

  # With specific device
  python main.py --mode realtime --device cuda

For full documentation, see README.md
        """,
    )

    # Mode arguments
    parser.add_argument(
        "--mode",
        type=str,
        default=Config.app.DEFAULT_MODE,
        choices=Config.app.MODES,
        help="""Operating mode:
        - realtime: PersonaPlex end-to-end S2S (ultra-low latency, full-duplex)
        - dubbing: Qwen3-TTS high-fidelity narration (multilingual, professional quality)
        - creative: Ollama + Qwen3-TTS (emotional synthesis, storytelling)
        """,
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
        help="Ollama model name (only used in creative mode, requires 'ollama serve')",
    )

    # Real-time mode arguments (PersonaPlex)
    parser.add_argument(
        "--persona",
        type=str,
        default=Config.app.REALTIME_PERSONA,
        help="Persona/role description for PersonaPlex (realtime mode)",
    )

    parser.add_argument(
        "--voice",
        type=str,
        default=Config.app.REALTIME_VOICE,
        choices=[
            "NATF0",
            "NATF1",
            "NATF2",
            "NATF3",
            "NATM0",
            "NATM1",
            "NATM2",
            "NATM3",
            "VARF0",
            "VARF1",
            "VARF2",
            "VARF3",
            "VARF4",
            "VARM0",
            "VARM1",
            "VARM2",
            "VARM3",
            "VARM4",
        ],
        help="Voice selection for PersonaPlex (realtime mode): NATF=natural female, NATM=natural male, VARF=varied female, VARM=varied male",
    )

    # Dubbing mode arguments
    parser.add_argument(
        "--script",
        type=str,
        help="Script/text to narrate (dubbing mode) or process (creative mode)",
    )

    parser.add_argument(
        "--voice-ref",
        type=str,
        help="Path to voice reference audio for voice cloning (3-5 seconds, dubbing/creative modes)",
    )

    # Creative mode arguments
    parser.add_argument(
        "--emotion",
        type=str,
        default=Config.app.CREATIVE_DEFAULT_EMOTION,
        help="Emotion instruction for Qwen3-TTS (creative mode), e.g., 'Speak with excitement'",
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
        print("\n[X] Configuration validation failed")
        return

    # Create and run application
    try:
        app = VellorisApplication(args)
        asyncio.run(app.run())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n[X] Fatal error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
