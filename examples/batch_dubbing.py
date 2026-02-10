#!/usr/bin/env python3
"""
Batch Dubbing: Process Multiple Scripts with Qwen3-TTS

This example demonstrates batch processing of multiple text scripts
for content creation workflows like audiobooks, video narration, etc.

Features:
- Process multiple scripts efficiently
- Language and emotion control per script
- Progress tracking
- Optional audio playback

Usage:
    python examples/batch_dubbing.py
    python examples/batch_dubbing.py --save-audio
    python examples/batch_dubbing.py --language en --emotion "Speak professionally"
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engines.qwen_tts import Qwen3TTSEngine


def main():
    parser = argparse.ArgumentParser(description="Batch TTS synthesis")
    parser.add_argument(
        "--language",
        default="en",
        choices=["zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it"],
        help="Language for all scripts",
    )
    parser.add_argument(
        "--emotion",
        default="",
        help="Emotion/style for all scripts",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use",
    )
    parser.add_argument(
        "--save-audio",
        action="store_true",
        help="Save generated audio files",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Velloris Batch Dubbing: Multiple Scripts")
    print("=" * 70)

    # Example batch of scripts for a podcast/audiobook
    scripts = [
        {
            "title": "Introduction",
            "text": "Welcome to our podcast series on artificial intelligence. In this episode, we'll explore how local AI systems are transforming voice technology.",
            "emotion": "enthusiastic" if not args.emotion else args.emotion,
        },
        {
            "title": "Main Content",
            "text": "Unlike cloud-based solutions, local AI runs entirely on your hardware. This means faster responses, better privacy, and complete control over your data.",
            "emotion": "informative" if not args.emotion else args.emotion,
        },
        {
            "title": "Technical Details",
            "text": "Velloris combines three specialized engines: PersonaPlex for real-time conversations, Qwen3-TTS for high-quality narration, and Ollama for creative reasoning.",
            "emotion": "technical" if not args.emotion else args.emotion,
        },
        {
            "title": "Conclusion",
            "text": "The future of AI is local, private, and under your control. Thank you for listening, and we'll see you in the next episode.",
            "emotion": "warm" if not args.emotion else args.emotion,
        },
    ]

    # Initialize engine
    print(f"\n[*] Initializing Qwen3-TTS engine...")
    print(f"    Device: {args.device}")
    print(f"    Language: {args.language}")
    print(f"    Emotion: {args.emotion or '(per-script)'}")

    engine = Qwen3TTSEngine(device=args.device)

    if engine.model is None:
        print("[X] Failed to load model")
        return 1

    print(f"[OK] Engine ready\n")

    # Process batch
    print("=" * 70)
    print("Processing Batch")
    print("=" * 70)

    total_duration = 0
    results = []

    for i, script in enumerate(scripts, 1):
        title = script["title"]
        text = script["text"]
        emotion = script["emotion"]

        print(
            f"\n[{i}/{len(scripts)}] {title}"
        )
        print(f"      Length: {len(text)} characters")
        print(f"      Emotion: {emotion}")

        result = engine.generate_dubbing(
            text=text,
            language=args.language,
            instruct=emotion,
        )

        if result is None:
            print(f"      [X] Failed")
            continue

        audio, sr = result
        duration = len(audio) / sr
        total_duration += duration

        print(f"      [OK] Generated {duration:.2f}s of audio")

        results.append(
            {
                "title": title,
                "audio": audio,
                "sr": sr,
                "duration": duration,
            }
        )

    # Summary
    print("\n" + "=" * 70)
    print("Batch Processing Complete")
    print("=" * 70)

    print(f"\nResults:")
    print(f"  Scripts processed: {len(results)}/{len(scripts)}")
    print(f"  Total audio duration: {total_duration:.2f}s")
    print(f"  Average per script: {total_duration/len(results):.2f}s" if results else "  N/A")

    # Show results
    print(f"\nGenerated Audio Files:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result['title']}: {result['duration']:.2f}s")

    # Save to file example
    if args.save_audio:
        print(f"\n[*] Example: Save audio files")
        print(f"    import soundfile as sf")
        for i, result in enumerate(results, 1):
            filename = f"output_{i:02d}_{result['title'].lower().replace(' ', '_')}.wav"
            print(f"    sf.write('{filename}', audio, sr)")

    # Concatenation example
    if results:
        print(f"\n[*] Example: Concatenate all audio")
        print(f"    import numpy as np")
        print(f"    combined = np.concatenate([r['audio'] for r in results])")
        print(f"    sf.write('full_episode.wav', combined, {results[0]['sr']})")

    # Cleanup
    engine.unload()
    print("\n[OK] Engine cleaned up")

    return 0


if __name__ == "__main__":
    sys.exit(main())
