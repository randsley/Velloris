#!/usr/bin/env python3
"""
Real-Time Speech-to-Speech Conversation with PersonaPlex-7B

This example demonstrates direct usage of the PersonaPlexEngine for
end-to-end speech understanding, reasoning, and generation with custom voices.

Features:
- Real-time speech-to-speech conversations
- 18 pre-trained voices (natural and varied accents)
- Custom persona/role control via text prompts
- No LLM required (PersonaPlex handles everything)

Usage:
    python examples/realtime_conversation.py
    python examples/realtime_conversation.py --voice natural_female_2
    python examples/realtime_conversation.py --device cuda
"""

import argparse
import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engines.personaplex import PersonaPlexEngine


def main():
    parser = argparse.ArgumentParser(
        description="Real-time S2S conversation with PersonaPlex-7B"
    )
    parser.add_argument(
        "--voice",
        default="natural_female_2",
        help="Voice preset (see available voices below)",
    )
    parser.add_argument(
        "--persona",
        default="You are a helpful and friendly AI assistant.",
        help="Persona/role description for the agent",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use for inference",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Velloris Real-Time S2S Conversation")
    print("=" * 70)

    # List available voices
    print("\nAvailable Voices:")
    voices = PersonaPlexEngine.get_available_voices()
    for i, voice in enumerate(voices, 1):
        print(f"  {i:2d}. {voice}")
    print()

    # Initialize engine
    print(f"[*] Initializing PersonaPlex-7B engine...")
    print(f"    Device: {args.device}")
    print(f"    Voice: {args.voice}")
    print(f"    Persona: {args.persona}")

    engine = PersonaPlexEngine(
        device=args.device, voice=args.voice, persona=args.persona
    )

    if engine.model is None:
        print("[X] Failed to load PersonaPlex-7B model")
        print("    Installation required:")
        print("      git clone https://github.com/NVIDIA/personaplex")
        print("      pip install personaplex/moshi/")
        return 1

    print("[OK] Engine ready\n")

    # Example: Generate response to test audio
    print("=" * 70)
    print("Demo: Processing sample audio")
    print("=" * 70)

    # Create test audio (1 second of silence with small noise)
    sample_rate = 24000
    duration = 1.0
    test_audio = np.random.randn(int(sample_rate * duration)).astype(np.float32) * 0.05
    test_audio = np.clip(test_audio, -1.0, 1.0)

    print(f"\n[*] Input: {len(test_audio)} samples ({duration:.1f}s) of test audio")
    print(f"    Sample rate: {sample_rate}Hz")

    # Generate S2S response
    print("\n[*] Running S2S inference...")
    result = engine.generate_s2s_response(
        audio=test_audio,
        sr=sample_rate,
        voice_prompt=None,  # Use engine's default voice
        text_prompt=args.persona,
    )

    if result is None:
        print("[X] S2S inference failed")
        engine.unload()
        return 1

    agent_audio, output_sr = result
    print(f"\n[OK] Response generated!")
    print(f"    Output: {len(agent_audio)} samples ({len(agent_audio)/output_sr:.2f}s)")
    print(f"    Sample rate: {output_sr}Hz")
    print(f"    Min/max amplitude: [{np.min(agent_audio):.3f}, {np.max(agent_audio):.3f}]")

    # Cleanup
    engine.unload()
    print("\n[OK] Engine cleaned up")

    print("\n" + "=" * 70)
    print("For real-time audio I/O and playback, use main.py:")
    print("  python main.py --mode realtime --voice natural_female_2")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
