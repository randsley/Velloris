#!/usr/bin/env python3
"""
Orchestrator API: Unified Interface for All Three Modes

This example demonstrates how to use LocalVoiceOrchestrator to route
requests to the appropriate engine based on mode, with automatic
device detection and lazy loading.

Modes:
  - realtime: PersonaPlex-7B S2S (speech-to-speech)
  - dubbing: Qwen3-TTS (text-to-speech)
  - creative: Ollama LLM + Qwen3-TTS (reasoning + synthesis)

Usage:
    python examples/orchestrator_api.py --mode realtime
    python examples/orchestrator_api.py --mode dubbing
    python examples/orchestrator_api.py --mode creative
"""

import argparse
import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.orchestrator import LocalVoiceOrchestrator


def demo_realtime(orchestrator):
    """Demonstrate real-time S2S mode."""
    print("\n" + "=" * 70)
    print("MODE: Real-Time Speech-to-Speech (PersonaPlex-7B)")
    print("=" * 70)

    # Create test audio
    sample_rate = 24000
    duration = 0.5
    test_audio = np.random.randn(int(sample_rate * duration)).astype(np.float32) * 0.05
    test_audio = np.clip(test_audio, -1.0, 1.0)

    print(f"\n[*] Input audio: {len(test_audio)} samples ({duration:.1f}s)")

    result = orchestrator.route_request(
        mode="realtime",
        audio_input=test_audio,
        voice_prompt="natural_female_2",
        text_prompt="You are a helpful assistant",
    )

    if result is None:
        print("[X] Real-time mode failed")
        return False

    audio, sr = result
    print(f"[OK] Generated {len(audio)} samples ({len(audio)/sr:.2f}s)")
    return True


def demo_dubbing(orchestrator):
    """Demonstrate dubbing (TTS) mode."""
    print("\n" + "=" * 70)
    print("MODE: High-Fidelity Dubbing (Qwen3-TTS)")
    print("=" * 70)

    text = "Welcome to Velloris, the local-first voice agent engine. This system runs entirely on your hardware."

    print(f"\n[*] Text: '{text}'")
    print(f"    Language: en")

    result = orchestrator.route_request(
        mode="dubbing",
        text=text,
        language="en",
        instruct="Speak clearly and professionally",
    )

    if result is None:
        print("[X] Dubbing mode failed")
        return False

    audio, sr = result
    print(f"[OK] Generated {len(audio)} samples ({len(audio)/sr:.2f}s)")
    return True


def demo_creative(orchestrator):
    """Demonstrate creative mode (Ollama + TTS)."""
    print("\n" + "=" * 70)
    print("MODE: Creative (Ollama LLM + Qwen3-TTS)")
    print("=" * 70)

    prompt = "Tell me a short joke about artificial intelligence in one sentence."

    print(f"\n[*] Prompt: '{prompt}'")
    print(f"    LLM model: llama3 (via Ollama)")
    print(f"    Emotion: humorous")

    print("\n[!] Note: This mode requires Ollama running.")
    print("    Start with: ollama serve")
    print("    Then pull model: ollama pull llama3")

    result = orchestrator.route_request(
        mode="creative",
        text=prompt,
        emotion="humorous",
        llm_model="llama3",
    )

    if result is None:
        print("[X] Creative mode failed (check if Ollama is running)")
        return False

    audio, sr = result
    print(f"[OK] Generated {len(audio)} samples ({len(audio)/sr:.2f}s)")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Orchestrator API: Unified voice agent interface"
    )
    parser.add_argument(
        "--mode",
        choices=["realtime", "dubbing", "creative", "all"],
        default="dubbing",
        help="Which mode(s) to demonstrate",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Velloris Orchestrator API Demo")
    print("=" * 70)

    # Initialize orchestrator
    print(f"\n[*] Initializing orchestrator...")
    print(f"    Device: {args.device}")

    orchestrator = LocalVoiceOrchestrator(device=args.device)
    print("[OK] Orchestrator ready")
    print("    - Models loaded on-demand (lazy loading)")
    print("    - Device auto-detection enabled")
    print("    - All three modes available")

    # Run demonstrations
    success_count = 0
    total_count = 0

    if args.mode in ["realtime", "all"]:
        total_count += 1
        if demo_realtime(orchestrator):
            success_count += 1

    if args.mode in ["dubbing", "all"]:
        total_count += 1
        if demo_dubbing(orchestrator):
            success_count += 1

    if args.mode in ["creative", "all"]:
        total_count += 1
        if demo_creative(orchestrator):
            success_count += 1

    # Cleanup
    print("\n[*] Cleaning up...")
    orchestrator.unload_engines()
    print("[OK] All engines unloaded")

    # Summary
    print("\n" + "=" * 70)
    print(f"Results: {success_count}/{total_count} modes successful")
    print("=" * 70)

    print("\nUsage Examples:")
    print("  # Real-time mode")
    print("  result = orchestrator.route_request(mode='realtime', audio_input=audio)")
    print()
    print("  # Dubbing mode")
    print("  result = orchestrator.route_request(mode='dubbing', text='Your text')")
    print()
    print("  # Creative mode")
    print("  result = orchestrator.route_request(mode='creative', text='Your prompt')")

    return 0 if success_count == total_count else 1


if __name__ == "__main__":
    sys.exit(main())
