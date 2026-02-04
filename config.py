"""
Velloris Configuration

Centralized configuration for audio settings, model settings, and VAD parameters.
"""

import os
from pathlib import Path
from typing import Optional


class AudioConfig:
    """Audio input/output configuration."""

    # Sample rates
    INPUT_SAMPLE_RATE = 16000  # VAD standard, Whisper input
    OUTPUT_SAMPLE_RATE = 24000  # PersonaPlex/Qwen3 output

    # Buffer settings
    BUFFER_DURATION = 2.0  # Transcribe every N seconds
    CHUNK_SIZE = 512  # Audio frames per callback
    OVERLAP_RATIO = 0.5  # Overlapping regions for smooth transcription

    # Audio quality
    MONO = True  # Use mono audio for efficiency
    BIT_DEPTH = 16  # 16-bit audio

    # Device configuration
    INPUT_DEVICE = None  # None = default input device
    OUTPUT_DEVICE = None  # None = default output device


class ModelConfig:
    """Model configuration."""

    # Whisper STT settings
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
    # Options: tiny, base, small, medium, large
    # tiny = fastest, large = most accurate

    # LLM settings (Ollama)
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    # TTS settings (Coqui XTTS-v2)
    TTS_MODEL = "v2"  # XTTS-v2 variant
    TTS_LANGUAGE = "en"  # Default language

    # Device settings
    DEVICE = os.getenv("DEFAULT_DEVICE", "cuda")  # cuda, cpu, mps
    DTYPE = "float16" if DEVICE == "cuda" else "float32"

    # Paths
    MODELS_DIR = Path.home() / ".cache" / "velloris" / "models"
    VOICES_DIR = Path(__file__).parent / "voices"
    VOICE_REFERENCE = os.getenv("VOICE_REFERENCE", str(VOICES_DIR / "reference.wav"))


class VADConfig:
    """Voice Activity Detection configuration."""

    # Silero VAD settings
    THRESHOLD = 0.5  # VAD confidence threshold (0.0-1.0)
    MIN_SPEECH_DURATION = 0.3  # Minimum speech duration in seconds
    MIN_SILENCE_DURATION = 0.3  # Minimum silence duration in seconds

    # Speech detection sensitivity
    # Higher = more sensitive, may pick up background noise
    SENSITIVITY = 0.5

    # Interruption detection
    ENABLE_BARGE_IN = True  # Allow user to interrupt AI
    INTERRUPT_THRESHOLD = 0.6  # How quickly user input interrupts AI


class ApplicationConfig:
    """Application-level configuration."""

    # Mode settings
    DEFAULT_MODE = "interactive"  # interactive, dubbing
    MODES = ["interactive", "dubbing"]

    # Interactive mode settings
    INTERACTIVE_TIMEOUT = 30.0  # Timeout for user input (seconds)
    RESPONSE_TIMEOUT = 120.0  # Timeout for AI response (seconds)

    # Dubbing mode settings
    DUBBING_CHUNK_SIZE = 256  # Process script in chunks

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = None  # None = stdout only

    # Error handling
    RETRY_ATTEMPTS = 3
    RETRY_DELAY = 1.0  # seconds

    # Performance
    MAX_WORKERS = 4  # Thread pool size
    ENABLE_PROFILING = False


class Config:
    """Combined configuration object."""

    audio = AudioConfig()
    model = ModelConfig()
    vad = VADConfig()
    app = ApplicationConfig()

    @classmethod
    def from_env(cls):
        """Load configuration from environment variables."""
        # This is a placeholder for loading from .env files
        # In production, use python-dotenv
        pass

    @classmethod
    def validate(cls) -> bool:
        """Validate configuration settings."""
        errors = []

        # Check model names
        valid_whisper = ["tiny", "base", "small", "medium", "large"]
        if cls.model.WHISPER_MODEL not in valid_whisper:
            errors.append(f"Invalid WHISPER_MODEL: {cls.model.WHISPER_MODEL}")

        # Check device
        valid_devices = ["cuda", "cpu", "mps"]
        if cls.model.DEVICE not in valid_devices:
            errors.append(f"Invalid DEVICE: {cls.model.DEVICE}")

        # Check VAD threshold
        if not 0.0 <= cls.vad.THRESHOLD <= 1.0:
            errors.append(f"VAD THRESHOLD must be between 0.0 and 1.0")

        if errors:
            print("Configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False

        return True

    @classmethod
    def print_config(cls):
        """Print current configuration."""
        print("\n=== Velloris Configuration ===")
        print("\nAudio:")
        print(f"  Input SR: {cls.audio.INPUT_SAMPLE_RATE} Hz")
        print(f"  Output SR: {cls.audio.OUTPUT_SAMPLE_RATE} Hz")
        print(f"  Buffer: {cls.audio.BUFFER_DURATION}s")

        print("\nModels:")
        print(f"  Whisper: {cls.model.WHISPER_MODEL}")
        print(f"  LLM: {cls.model.OLLAMA_MODEL}")
        print(f"  Device: {cls.model.DEVICE}")
        print(f"  Ollama: {cls.model.OLLAMA_HOST}")

        print("\nVAD:")
        print(f"  Threshold: {cls.vad.THRESHOLD}")
        print(f"  Barge-in: {cls.vad.ENABLE_BARGE_IN}")

        print("\nApplication:")
        print(f"  Mode: {cls.app.DEFAULT_MODE}")
        print(f"  Log Level: {cls.app.LOG_LEVEL}\n")


if __name__ == "__main__":
    Config.print_config()
    Config.validate()
