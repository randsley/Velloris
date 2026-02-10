#!/usr/bin/env python3
"""
Simple Text-to-Speech with Qwen3-TTS

This example demonstrates direct usage of the Qwen3TTSEngine for
high-fidelity speech synthesis with support for 10 languages.

Features:
- Professional-quality speech synthesis (12kHz)
- 10 languages supported
- Voice cloning from reference audio
- Emotion/style control via natural language

Languages:
  - Chinese, English, Japanese, Korean, German,
    French, Russian, Portuguese, Spanish, Italian

Usage:
    python examples/text_to_speech_simple.py
    python examples/text_to_speech_simple.py --text "Hello world"
    python examples/text_to_speech_simple.py --language en --emotion "Speak with excitement"
"""

import argparse
import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engines.qwen_tts import Qwen3TTSEngine
from utils.audio_io import play_audio


def main():
    parser = argparse.ArgumentParser(description="Simple text-to-speech synthesis")
    parser.add_argument(
        "--text",
        default="Hello, this is a test of the Qwen3 text-to-speech system.",
        help="Text to synthesize",
    )
    parser.add_argument(
        "--language",
        default="en",
        choices=["zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it"],
        help="Language code",
    )
    parser.add_argument(
        "--emotion",
        default="normal",
        help="Emotion/style instruction (e.g., 'speak with excitement')",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use",
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Play audio after generation",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Qwen3-TTS: Simple Text-to-Speech Synthesis")
    print("=" * 70)

    # Initialize engine
    print(f"\n[*] Initializing Qwen3-TTS engine...")
    print(f"    Device: {args.device}")
    print(f"    Language: {args.language}")

    engine = Qwen3TTSEngine(device=args.device)

    if engine.model is None:
        print("[X] Failed to load Qwen3-TTS model")
        print("    Ensure dependencies are installed:")
        print("      pip install qwen-tts")
        return 1

    print("[OK] Engine ready\n")

    # Generate speech
    print(f"[*] Synthesizing: '{args.text}'")
    print(f"    Language: {args.language}")
    print(f"    Emotion: {args.emotion}")

    result = engine.generate_dubbing(
        text=args.text, language=args.language, instruct=args.emotion
    )

    if result is None:
        print("[X] Synthesis failed")
        engine.unload()
        return 1

    audio, sr = result
    duration = len(audio) / sr

    print(f"\n[OK] Audio generated!")
    print(f"    Duration: {duration:.2f}s")
    print(f"    Sample rate: {sr}Hz")
    print(f"    Samples: {len(audio)}")
    print(f"    Min/max: [{np.min(audio):.3f}, {np.max(audio):.3f}]")

    # Play audio if requested
    if args.play:
        print("\n[*] Playing audio...")
        play_audio(audio, sr)
        print("[OK] Playback complete")

    # Save to file example
    output_file = Path("output_tts.wav")
    print(f"\n[*] Example: Save to file")
    print(f"    import soundfile as sf")
    print(f"    sf.write('{output_file}', audio, sr)")

    # Cleanup
    engine.unload()
    print("\n[OK] Engine cleaned up")

    print("\n" + "=" * 70)
    print("For more control, use the Orchestrator:")
    print("  from core.orchestrator import LocalVoiceOrchestrator")
    print("  orchestrator = LocalVoiceOrchestrator()")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
