#!/usr/bin/env python3
"""
Velloris Model Downloader

Downloads all required models for offline/air-gapped use.

Usage:
    python download_models.py              # Download all models
    python download_models.py --model tts  # Download only TTS model
    python download_models.py --list       # List available models
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import os

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Import Config directly to avoid triggering utils/__init__.py
from config import Config


def download_qwen3_tts(force: bool = False) -> bool:
    """Download Qwen3-TTS model from Hugging Face."""
    model_dir = Config.model.QWEN3_TTS_DIR
    model_id = Config.model.QWEN3_TTS_MODEL_ID

    if not force and Config.model.is_model_downloaded("qwen3-tts"):
        print(f"[OK] Qwen3-TTS already downloaded at {model_dir}")
        return True

    print(f"Downloading Qwen3-TTS ({model_id})...")
    print(f"  Destination: {model_dir}")

    try:
        from huggingface_hub import snapshot_download

        model_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=model_id,
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
        )
        print("[OK] Qwen3-TTS downloaded successfully")
        return True
    except ImportError:
        print("[X] huggingface_hub not installed. Run: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"[X] Failed to download Qwen3-TTS: {e}")
        return False


def download_personaplex(force: bool = False) -> bool:
    """Download PersonaPlex-7B model from Hugging Face."""
    model_dir = Config.model.PERSONAPLEX_DIR
    model_id = Config.model.PERSONAPLEX_MODEL_ID

    if not force and Config.model.is_model_downloaded("personaplex"):
        print(f"[OK] PersonaPlex already downloaded at {model_dir}")
        return True

    print(f"Downloading PersonaPlex-7B ({model_id})...")
    print(f"  Destination: {model_dir}")
    print("  Note: This model requires accepting the license on Hugging Face")
    print("        Run 'huggingface-cli login' first if you haven't")

    try:
        from huggingface_hub import snapshot_download

        model_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=model_id,
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
        )
        print("[OK] PersonaPlex downloaded successfully")
        return True
    except ImportError:
        print("[X] huggingface_hub not installed. Run: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"[X] Failed to download PersonaPlex: {e}")
        if "401" in str(e) or "403" in str(e):
            print("  This may be a license issue. Please:")
            print("  1. Visit https://huggingface.co/nvidia/personaplex-7b-v1")
            print("  2. Accept the license agreement")
            print("  3. Run: huggingface-cli login")
        return False


def download_silero_vad(force: bool = False) -> bool:
    """Download Silero VAD model."""
    model_dir = Config.model.SILERO_VAD_DIR

    if not force and Config.model.is_model_downloaded("silero-vad"):
        print(f"[OK] Silero VAD already downloaded at {model_dir}")
        return True

    print("Downloading Silero VAD...")
    print(f"  Destination: {model_dir}")

    try:
        import torch

        model_dir.mkdir(parents=True, exist_ok=True)

        # Download via torch.hub
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=force,
        )

        # Save the model locally
        model_path = model_dir / "silero_vad.jit"
        torch.jit.save(model, str(model_path))

        print("[OK] Silero VAD downloaded successfully")
        return True
    except Exception as e:
        print(f"[X] Failed to download Silero VAD: {e}")
        return False


def download_whisper(force: bool = False) -> bool:
    """Download Whisper model."""
    model_dir = Config.model.WHISPER_DIR
    model_name = Config.model.WHISPER_MODEL

    if not force and Config.model.is_model_downloaded("whisper"):
        print(f"[OK] Whisper already downloaded at {model_dir}")
        return True

    print(f"Downloading Whisper ({model_name})...")
    print(f"  Destination: {model_dir}")

    try:
        import whisper

        model_dir.mkdir(parents=True, exist_ok=True)

        # Set download root to our local directory
        os.environ["XDG_CACHE_HOME"] = str(model_dir.parent)

        # Download the model
        whisper.load_model(model_name, download_root=str(model_dir))

        print("[OK] Whisper downloaded successfully")
        return True
    except ImportError:
        print("[X] openai-whisper not installed. Run: pip install openai-whisper")
        return False
    except Exception as e:
        print(f"[X] Failed to download Whisper: {e}")
        return False


def download_mlx_tts(force: bool = False) -> bool:
    """Download MLX-compatible Qwen3-TTS model."""
    model_dir = Config.model.MLX_TTS_DIR
    model_id = Config.model.MLX_TTS_MODEL_ID

    if not force and Config.model.is_model_downloaded("mlx-tts"):
        print(f"[OK] MLX TTS already downloaded at {model_dir}")
        return True

    print(f"Downloading MLX TTS ({model_id})...")
    print(f"  Destination: {model_dir}")

    try:
        from huggingface_hub import snapshot_download

        model_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=model_id,
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
        )
        print("[OK] MLX TTS downloaded successfully")
        return True
    except ImportError:
        print("[X] huggingface_hub not installed. Run: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"[X] Failed to download MLX TTS: {e}")
        return False


def download_mlx_whisper(force: bool = False) -> bool:
    """Download MLX-compatible Whisper model."""
    model_dir = Config.model.MLX_WHISPER_DIR
    model_id = Config.model.MLX_WHISPER_MODEL_ID

    if not force and Config.model.is_model_downloaded("mlx-whisper"):
        print(f"[OK] MLX Whisper already downloaded at {model_dir}")
        return True

    print(f"Downloading MLX Whisper ({model_id})...")
    print(f"  Destination: {model_dir}")

    try:
        from huggingface_hub import snapshot_download

        model_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=model_id,
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
        )
        print("[OK] MLX Whisper downloaded successfully")
        return True
    except ImportError:
        print("[X] huggingface_hub not installed. Run: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"[X] Failed to download MLX Whisper: {e}")
        return False


def list_models():
    """List all models and their status."""
    print("\n=== Velloris Model Status ===\n")

    models = [
        (
            "qwen3-tts",
            "Qwen3-TTS-1.7B",
            "~3-5 GB",
            "Dubbing, Creative modes (non-macOS)",
        ),
        ("personaplex", "PersonaPlex-7B", "~14 GB", "Realtime mode"),
        ("silero-vad", "Silero VAD", "~2 MB", "Voice activity detection"),
        (
            "whisper",
            f"Whisper ({Config.model.WHISPER_MODEL})",
            "~150 MB - 3 GB",
            "Speech-to-text (non-macOS)",
        ),
        ("mlx-tts", "MLX Qwen3-TTS", "~3-5 GB", "Dubbing, Creative modes (macOS only)"),
        ("mlx-whisper", "MLX Whisper", "~3 GB", "Speech-to-text (macOS only)"),
    ]

    for model_key, name, size, usage in models:
        status = (
            "[OK] Downloaded"
            if Config.model.is_model_downloaded(model_key)
            else "[X] Not found"
        )
        path = Config.model.get_local_model_path(model_key)
        print(f"{name}")
        print(f"  Status: {status}")
        print(f"  Size: {size}")
        print(f"  Usage: {usage}")
        print(f"  Path: {path}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Download Velloris models for offline use",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_models.py              # Download all models
  python download_models.py --model tts  # Download only Qwen3-TTS
  python download_models.py --model mlx  # Download all MLX models for macOS
  python download_models.py --model vad  # Download only Silero VAD
  python download_models.py --list       # List model status
  python download_models.py --force      # Re-download all models
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "all",
            "tts",
            "personaplex",
            "vad",
            "whisper",
            "mlx",
            "mlx-tts",
            "mlx-whisper",
        ],
        default="all",
        help="Which model to download (default: all)",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models and their status",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if model exists",
    )

    args = parser.parse_args()

    if args.list:
        list_models()
        return

    print("=" * 50)
    print("Velloris Model Downloader")
    print("=" * 50)
    print(f"\nModels directory: {Config.model.MODELS_DIR}")
    print()

    results = {}
    is_darwin = sys.platform == "darwin"
    is_linux = sys.platform == "linux"

    # TTS models (platform-specific)
    if args.model in ["all", "tts"]:
        if is_darwin:
            # macOS uses MLX TTS, not Qwen3-TTS
            if args.model == "tts":
                print(
                    "[i] Qwen3-TTS is not available on macOS. Use '--model mlx-tts' instead."
                )
        else:
            results["qwen3-tts"] = download_qwen3_tts(force=args.force)

    # PersonaPlex: NVIDIA GPU + Linux/Windows only (NOT macOS)
    if args.model in ["all", "personaplex"]:
        if is_darwin:
            print(
                "[i] PersonaPlex is not available on macOS (requires NVIDIA GPU and Linux/Windows)."
            )
        else:
            results["personaplex"] = download_personaplex(force=args.force)

    # Silero VAD (works on all platforms)
    if args.model in ["all", "vad"]:
        results["silero-vad"] = download_silero_vad(force=args.force)

    # Whisper: Standard openai-whisper for non-macOS
    if args.model in ["all", "whisper"]:
        if is_darwin:
            # macOS uses MLX Whisper, not openai-whisper
            if args.model == "whisper":
                print(
                    "[i] Openai-Whisper is not available on macOS. Use '--model mlx-whisper' instead."
                )
        else:
            results["whisper"] = download_whisper(force=args.force)

    # MLX models (macOS only)
    if args.model in ["all", "mlx", "mlx-tts"]:
        if is_darwin:
            results["mlx-tts"] = download_mlx_tts(force=args.force)
        elif args.model == "mlx-tts":
            print("[i] MLX models are only available on macOS.")

    if args.model in ["all", "mlx", "mlx-whisper"]:
        if is_darwin:
            results["mlx-whisper"] = download_mlx_whisper(force=args.force)
        elif args.model == "mlx-whisper":
            print("[i] MLX models are only available on macOS.")

    # Summary
    print("\n" + "=" * 50)
    print("Download Summary")
    print("=" * 50)

    if not results:
        print("\n[i] No models to download for this platform/selection.")
        print("    (Models may have been skipped due to platform incompatibility)")
        sys.exit(0)

    success = all(results.values())
    for model, status in results.items():
        icon = "[OK]" if status else "[X]"
        print(f"  {icon} {model}")

    if success:
        print("\n[OK] All requested models downloaded successfully!")
        print("\nTo enable offline mode, set environment variable:")
        print("  export VELLORIS_OFFLINE=true  # Linux/macOS")
        print("  set VELLORIS_OFFLINE=true     # Windows")
    else:
        print("\n[X] Some models failed to download")
        sys.exit(1)


if __name__ == "__main__":
    main()
