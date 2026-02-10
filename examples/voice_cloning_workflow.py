#!/usr/bin/env python3
"""
Voice Cloning Workflow: Create Custom Voices from Audio Samples

This example demonstrates how to use voice cloning with Qwen3-TTS
to create narration in custom voices from reference audio.

Voice Cloning Steps:
  1. Prepare reference audio (3-5 seconds of clear speech)
  2. Validate audio format and duration
  3. Use engine.generate_dubbing() with ref_audio_path parameter
  4. Synthesize narration in the cloned voice

Audio Requirements:
  - Duration: 3-5 seconds (can be longer, but 3-5s is optimal)
  - Format: WAV, MP3, or other common formats
  - Content: Clear speech with minimal background noise
  - Quality: 16kHz or higher sample rate recommended

Usage:
    # With an existing reference audio file
    python examples/voice_cloning_workflow.py --voice-ref path/to/sample.wav

    # Create sample voice reference (for demo)
    python examples/voice_cloning_workflow.py --demo
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engines.qwen_tts import Qwen3TTSEngine
from utils.audio_utils import save_audio


def create_demo_voice_sample():
    """Create a demo voice reference file for testing."""
    import numpy as np

    print("\n[*] Creating demo voice sample...")

    # Create a 4-second speech-like audio sample
    # (In practice, this should be real speech)
    sample_rate = 16000
    duration = 4.0
    num_samples = int(sample_rate * duration)

    # Create synthetic audio with varying pitch/frequency
    t = np.linspace(0, duration, num_samples)
    # Mix of frequencies to simulate speech
    audio = (
        0.3 * np.sin(2 * np.pi * 200 * t)  # Low frequency
        + 0.2 * np.sin(2 * np.pi * 500 * t)  # Mid frequency
        + 0.15 * np.sin(2 * np.pi * 1000 * t)  # High frequency
    )
    audio = audio * 0.5  # Normalize amplitude

    # Save to file
    demo_path = Path("demo_voice_sample.wav")
    save_audio(audio, sample_rate, str(demo_path))
    print(f"    Saved demo voice sample: {demo_path}")
    print(f"    Duration: {duration:.1f}s")
    print(f"    Sample rate: {sample_rate}Hz")

    return str(demo_path)


def validate_reference_audio(audio_path):
    """Validate reference audio file."""
    path = Path(audio_path)

    if not path.exists():
        print(f"[X] Audio file not found: {audio_path}")
        return False

    print(f"\n[*] Validating reference audio: {path.name}")

    # Try to load the audio
    try:
        import soundfile as sf

        audio, sr = sf.read(str(path))
        duration = len(audio) / sr

        print(f"    Sample rate: {sr}Hz")
        print(f"    Duration: {duration:.2f}s")
        print(f"    Samples: {len(audio)}")

        # Check duration (3-5s optimal, but accept up to 10s)
        if duration < 1:
            print(f"    [!] Warning: Audio too short (<1s). Aim for 3-5s.")
            return False
        elif duration < 3:
            print(f"    [!] Warning: Audio shorter than recommended. 3-5s is optimal.")
        elif duration > 10:
            print(f"    [!] Warning: Audio longer than typical. May reduce quality.")

        print(f"    [OK] Audio valid")
        return True

    except ImportError:
        print(f"    [!] soundfile not available. Skipping validation.")
        print(f"        Install with: pip install soundfile")
        return True
    except Exception as e:
        print(f"    [X] Error reading audio: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Voice cloning workflow with Qwen3-TTS"
    )
    parser.add_argument(
        "--voice-ref",
        help="Path to reference voice audio file",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Create a demo voice sample for testing",
    )
    parser.add_argument(
        "--script",
        default="Velloris enables professional-quality voice synthesis with custom voice cloning.",
        help="Text to synthesize in cloned voice",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Voice Cloning Workflow: Create Custom Voices")
    print("=" * 70)

    # Get reference audio path
    voice_ref = args.voice_ref
    if args.demo:
        voice_ref = create_demo_voice_sample()
    elif not voice_ref:
        print("\n[!] Usage:")
        print("    python examples/voice_cloning_workflow.py --voice-ref path/to/sample.wav")
        print("    OR")
        print("    python examples/voice_cloning_workflow.py --demo")
        return 1

    # Validate reference audio
    if not validate_reference_audio(voice_ref):
        return 1

    # Initialize engine
    print(f"\n[*] Initializing Qwen3-TTS engine for voice cloning...")
    print(f"    Device: {args.device}")

    engine = Qwen3TTSEngine(device=args.device)

    if engine.model is None:
        print("[X] Failed to load model")
        return 1

    print("[OK] Engine ready\n")

    # Generate speech with voice cloning
    print("=" * 70)
    print("Voice Cloning Synthesis")
    print("=" * 70)

    print(f"\n[*] Reference voice: {voice_ref}")
    print(f"    Script: '{args.script}'")
    print(f"    Language: en")

    print("\n[*] Synthesizing with cloned voice...")

    result = engine.generate_dubbing(
        text=args.script,
        ref_audio_path=voice_ref,
        language="en",
    )

    if result is None:
        print("[X] Voice cloning synthesis failed")
        engine.unload()
        return 1

    audio, sr = result
    duration = len(audio) / sr

    print(f"\n[OK] Synthesis complete!")
    print(f"    Duration: {duration:.2f}s")
    print(f"    Sample rate: {sr}Hz")
    print(f"    Samples: {len(audio)}")

    # Save result
    output_path = Path("cloned_voice_output.wav")
    try:
        import soundfile as sf

        sf.write(str(output_path), audio, sr)
        print(f"\n[OK] Saved to: {output_path}")
    except ImportError:
        print(f"\n[!] soundfile not available for saving")
        print(f"    Install with: pip install soundfile")

    # Show usage guide
    print("\n" + "=" * 70)
    print("Voice Cloning Guide")
    print("=" * 70)

    print("\n1. Prepare Reference Audio:")
    print("   - Record 3-5 seconds of clear speech")
    print("   - Minimize background noise")
    print("   - Use high sample rate (16kHz or higher)")

    print("\n2. Use Voice Cloning:")
    print("   result = engine.generate_dubbing(")
    print("       text='Your script',")
    print("       ref_audio_path='path/to/voice_sample.wav'")
    print("   )")

    print("\n3. Process Multiple Scripts:")
    print("   for script in scripts:")
    print("       audio, sr = engine.generate_dubbing(")
    print("           text=script,")
    print("           ref_audio_path=voice_sample")
    print("       )")

    print("\n4. Quality Tips:")
    print("   - Longer reference audio (3-5s) = better cloning")
    print("   - Multiple speakers = blend of voices")
    print("   - Clear articulation = better results")

    # Cleanup
    engine.unload()
    print("\n[OK] Engine cleaned up")

    return 0


if __name__ == "__main__":
    sys.exit(main())
