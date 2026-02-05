"""
Velloris Configuration

Centralized configuration for audio settings, model settings, and VAD parameters.
"""

import os
from pathlib import Path
from typing import Optional
from utils.device_utils import get_optimal_device, get_optimal_dtype, get_platform_info


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

    # Device settings (auto-detect if not specified)
    _requested_device = os.getenv("DEFAULT_DEVICE", "auto")
    DEVICE = get_optimal_device(_requested_device)
    _dtype_obj = get_optimal_dtype(DEVICE)
    DTYPE = str(_dtype_obj).split('.')[-1]  # "float32", "float16", or "bfloat16"

    # Platform info
    PLATFORM_INFO = get_platform_info()

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
    DEFAULT_MODE = os.getenv("DEFAULT_MODE", "realtime")  # realtime, dubbing, creative
    MODES = ["realtime", "dubbing", "creative"]

    # Real-time mode settings (PersonaPlex end-to-end S2S)
    REALTIME_VOICE = os.getenv("REALTIME_VOICE", "NATF2")  # Default voice
    REALTIME_PERSONA = os.getenv(
        "REALTIME_PERSONA",
        "You are a helpful and friendly AI assistant."
    )
    REALTIME_STREAMING = True  # Enable streaming for full-duplex
    REALTIME_TIMEOUT = 30.0  # Timeout for user input (seconds)
    REALTIME_SAMPLE_RATE = 24000  # PersonaPlex native sample rate

    # Creative mode settings (Ollama + Qwen3-TTS)
    CREATIVE_LLM = os.getenv("CREATIVE_LLM", "llama3")
    CREATIVE_DEFAULT_EMOTION = os.getenv("CREATIVE_EMOTION", "")
    CREATIVE_TIMEOUT = 120.0  # Timeout for LLM response (seconds)

    # Dubbing mode settings (Qwen3-TTS high-fidelity)
    DUBBING_CHUNK_SIZE = 256  # Process script in chunks
    DUBBING_TIMEOUT = 60.0  # Timeout for TTS generation

    # Backward compatibility (deprecated)
    INTERACTIVE_TIMEOUT = 30.0  # DEPRECATED: Use REALTIME_TIMEOUT or CREATIVE_TIMEOUT
    RESPONSE_TIMEOUT = 120.0  # DEPRECATED: Use CREATIVE_TIMEOUT

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

        # Check device (DEVICE is already validated and resolved by get_optimal_device)
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

        # Platform info
        print("\nPlatform:")
        print(f"  OS: {cls.model.PLATFORM_INFO['os']} ({cls.model.PLATFORM_INFO['machine']})")
        print(f"  Python: {cls.model.PLATFORM_INFO['python_version']}")
        print(f"  CUDA Available: {cls.model.PLATFORM_INFO['cuda_available']}")
        print(f"  MPS Available: {cls.model.PLATFORM_INFO['mps_available']}")

        print("\nAudio:")
        print(f"  Input SR: {cls.audio.INPUT_SAMPLE_RATE} Hz")
        print(f"  Output SR: {cls.audio.OUTPUT_SAMPLE_RATE} Hz")
        print(f"  Buffer: {cls.audio.BUFFER_DURATION}s")

        print("\nModels:")
        print(f"  Whisper: {cls.model.WHISPER_MODEL}")
        print(f"  LLM: {cls.model.OLLAMA_MODEL}")
        print(f"  Device: {cls.model.DEVICE}")
        print(f"  Dtype: {cls.model.DTYPE}")
        print(f"  Ollama: {cls.model.OLLAMA_HOST}")

        print("\nVAD:")
        print(f"  Threshold: {cls.vad.THRESHOLD}")
        print(f"  Barge-in: {cls.vad.ENABLE_BARGE_IN}")

        print("\nApplication:")
        print(f"  Default Mode: {cls.app.DEFAULT_MODE}")
        print(f"  Available Modes: {', '.join(cls.app.MODES)}")
        print(f"  Log Level: {cls.app.LOG_LEVEL}")

        print("\nMode Settings:")
        print(f"  Real-Time:")
        print(f"    Voice: {cls.app.REALTIME_VOICE}")
        print(f"    Persona: {cls.app.REALTIME_PERSONA[:50]}...")
        print(f"    Streaming: {cls.app.REALTIME_STREAMING}")
        print(f"  Creative:")
        print(f"    LLM: {cls.app.CREATIVE_LLM}")
        print(f"    Emotion: {cls.app.CREATIVE_DEFAULT_EMOTION or 'None'}")
        print(f"  Dubbing:")
        print(f"    Chunk Size: {cls.app.DUBBING_CHUNK_SIZE}\n")


if __name__ == "__main__":
    Config.print_config()
    Config.validate()
