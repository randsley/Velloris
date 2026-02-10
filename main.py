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


# Voice format mapping: convert long names to short codes
VOICE_MAP = {
    # Long names -> Short codes
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


def voice_converter(voice_input):
    """Convert voice input (long name or short code) to short code."""
    # Check if it's a long name
    if voice_input in VOICE_MAP:
        return VOICE_MAP[voice_input]
    # Already a short code
    if voice_input in VOICE_MAP.values():
        return voice_input
    # Invalid voice
    valid_short = list(VOICE_MAP.values())
    valid_long = list(VOICE_MAP.keys())
    raise argparse.ArgumentTypeError(
        f"Invalid voice '{voice_input}'. Use short codes ({', '.join(valid_short)}) "
        f"or long names ({', '.join(valid_long)})"
    )


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

        Real-time voice conversation with the agent using S2S engine.
        User can interrupt the agent at any time with barge-in.
        """
        print("\n" + "=" * 60)
        print("🎤 VELLORIS - REALTIME MODE")
        print("=" * 60)
        print("Starting voice agent in real-time mode...")
        print("The agent will listen and respond to your voice.")
        print("Press Ctrl+C to exit.\n")

        try:
            # Run the actual audio processing loop
            await self._realtime_audio_loop()

        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        except Exception as e:
            print(f"\n[X] Error in realtime mode: {e}")
            import traceback

            traceback.print_exc()
        finally:
            self.cleanup()

    async def _realtime_audio_loop(self):
        """
        Main realtime audio processing loop.

        Captures audio from microphone, processes through S2S engine,
        and plays responses through speakers with barge-in capability.
        """
        import sounddevice as sd

        # Find audio devices
        input_device = sd.default.device[0]
        output_device = sd.default.device[1]

        input_device_info = sd.query_devices(input_device, kind="input")
        output_device_info = sd.query_devices(output_device, kind="output")

        print(f"🎙️  Input: {input_device_info['name']}")
        print(f"🔊 Output: {output_device_info['name']}")
        print()

        # Start background transcription worker
        self.audio_controller.start_transcription_worker()

        try:
            # Open audio streams
            with sd.InputStream(
                callback=self.audio_controller._input_callback,
                channels=1,
                samplerate=self.audio_controller.fs,  # 16kHz for VAD
                blocksize=512,  # 32ms chunks for Silero VAD
                dtype="int16",
                device=input_device,
            ), sd.OutputStream(
                callback=self.audio_controller._output_callback,
                channels=1,
                samplerate=self.audio_controller.output_fs,  # 24kHz for TTS
                blocksize=int(self.audio_controller.output_fs * 0.05),  # 50ms chunks
                dtype="float32",
                device=output_device,
            ):
                print("✅ Session active. Listening...\n")
                print("💬 Speak naturally - I'll respond when you pause.")
                print("✋ You can interrupt me at any time (barge-in enabled).\n")

                # Main processing loop
                while self.running:
                    # Check for audio input ready for S2S processing
                    if self.audio_controller.has_audio_input():
                        # Get raw audio chunk (16kHz, 2 seconds)
                        audio_chunk_16k = self.audio_controller.get_audio_input(
                            timeout=0.1
                        )

                        if audio_chunk_16k is not None:
                            # Resample to 24kHz for S2S engine
                            from utils.audio_utils import resample_audio

                            audio_chunk_24k = resample_audio(
                                audio_chunk_16k, 16000, 24000
                            )

                            # Process through S2S engine
                            result = self.orchestrator.route_request(
                                mode="realtime",
                                audio_input=audio_chunk_24k,
                                voice_prompt=self.args.voice,
                                text_prompt=self.args.persona,
                            )

                            if result:
                                agent_audio, sr = result

                                # Queue response for playback
                                if agent_audio is not None and len(agent_audio) > 0:
                                    self.audio_controller.queue_audio_output(
                                        agent_audio
                                    )

                                # Reset interruption flag for next turn
                                self.interruption_handler.reset()

                    # Prevent busy-waiting
                    await asyncio.sleep(0.05)  # Check every 50ms

        finally:
            print("\n🛑 Stopping audio session...")
            self.audio_controller.stop_transcription_worker()

    async def _demo_realtime_mode(self):
        """
        Demo realtime mode with text input.

        DEPRECATED: Use _realtime_audio_loop() for actual audio processing.
        Kept for backwards compatibility and testing without audio devices.
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

                # Process with brain (creative mode needs brain)
                if self.brain:
                    print("\nVelloris: Processing...")
                    result = await self.brain.process_voice_turn(user_input)

                    if isinstance(result, tuple) and len(result) == 3:
                        # New format: (response_text, audio, sample_rate)
                        response_text, audio_response, sr = result
                    else:
                        # Legacy format: (response_text, audio)
                        response_text, audio_response = result
                        sr = 24000  # Default sample rate

                    print(f"Velloris: {response_text}\n")

                    # If audio was generated, play it
                    if audio_response is not None and len(audio_response) > 0:
                        print()
                        play_audio(audio_response, samplerate=sr)

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
        type=voice_converter,
        default=Config.app.REALTIME_VOICE,
        help="Voice selection for PersonaPlex (realtime mode). "
        "Use short codes (NATF0-3, NATM0-3, VARF0-4, VARM0-4) "
        "or long names (natural_female_0, natural_male_0, varied_female_0, varied_male_0, etc.)",
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
