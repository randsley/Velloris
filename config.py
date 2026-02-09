"""
Velloris Configuration

Centralized configuration for audio settings, model settings, and VAD parameters.
"""

import os
from pathlib import Path
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

    # Project root directory
    PROJECT_ROOT = Path(__file__).parent

    # Model storage - project-local for offline/air-gapped use
    MODELS_DIR = PROJECT_ROOT / "models"
    OFFLINE_MODE = os.getenv("VELLORIS_OFFLINE", "false").lower() == "true"

    # Model-specific directories
    QWEN3_TTS_DIR = MODELS_DIR / "qwen3-tts"
    PERSONAPLEX_DIR = MODELS_DIR / "personaplex"
    SILERO_VAD_DIR = MODELS_DIR / "silero-vad"
    WHISPER_DIR = MODELS_DIR / "whisper"
    MLX_TTS_DIR = MODELS_DIR / "mlx-tts"
    MLX_WHISPER_DIR = MODELS_DIR / "mlx-whisper"
    MACECHO_DIR = MODELS_DIR / "macecho"  # macOS only
    MACECHO_SENSEVOICE_DIR = MODELS_DIR / "macecho-sensevoice"  # ASR model
    MACECHO_COSYVOICE_DIR = MODELS_DIR / "macecho-cosyvoice"  # TTS model

    # Hugging Face model identifiers
    QWEN3_TTS_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    PERSONAPLEX_MODEL_ID = "nvidia/personaplex-7b-v1"
    MLX_TTS_MODEL_ID = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"
    MLX_WHISPER_MODEL_ID = "mlx-community/whisper-large-v3-turbo-asr-fp16"

    # Whisper STT settings
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
    # Options: tiny, base, small, medium, large
    # tiny = fastest, large = most accurate

    # LLM settings (Ollama)
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    # TTS settings
    TTS_MODEL = "1.7B-CustomVoice"  # Qwen3-TTS variant
    TTS_LANGUAGE = "en"  # Default language

    # Device settings (auto-detect if not specified)
    _requested_device = os.getenv("DEFAULT_DEVICE", "auto")
    DEVICE = get_optimal_device(_requested_device)
    _dtype_obj = get_optimal_dtype(DEVICE)
    DTYPE = str(_dtype_obj).split(".")[-1]  # "float32", "float16", or "bfloat16"

    # Platform info
    PLATFORM_INFO = get_platform_info()

    # Voice paths
    VOICES_DIR = PROJECT_ROOT / "voices"
    VOICE_REFERENCE = os.getenv("VOICE_REFERENCE", str(VOICES_DIR / "reference.wav"))

    @classmethod
    def get_local_model_path(cls, model_type: str) -> Path:
        """Get local path for a model type."""
        paths = {
            "qwen3-tts": cls.QWEN3_TTS_DIR,
            "personaplex": cls.PERSONAPLEX_DIR,
            "silero-vad": cls.SILERO_VAD_DIR,
            "whisper": cls.WHISPER_DIR,
            "mlx-tts": cls.MLX_TTS_DIR,
            "mlx-whisper": cls.MLX_WHISPER_DIR,
            "macecho": cls.MACECHO_DIR,
            "macecho-sensevoice": cls.MACECHO_SENSEVOICE_DIR,
            "macecho-cosyvoice": cls.MACECHO_COSYVOICE_DIR,
        }
        return paths.get(model_type, cls.MODELS_DIR)

    @classmethod
    def is_model_downloaded(cls, model_type: str) -> bool:
        """Check if a model is downloaded locally."""
        model_path = cls.get_local_model_path(model_type)
        if not model_path.exists():
            return False
        # Check if directory has content
        return any(model_path.iterdir()) if model_path.is_dir() else False


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

    # Real-time mode settings (PersonaPlex/MacEcho end-to-end S2S)
    REALTIME_VOICE = os.getenv("REALTIME_VOICE", "NATF2")  # Default voice
    REALTIME_PERSONA = os.getenv(
        "REALTIME_PERSONA", "You are a helpful and friendly AI assistant."
    )
    REALTIME_STREAMING = True  # Enable streaming for full-duplex
    REALTIME_TIMEOUT = 30.0  # Timeout for user input (seconds)
    REALTIME_SAMPLE_RATE = 24000  # Native sample rate for output
    MACECHO_RESPONSE_TIMEOUT = float(
        os.getenv("MACECHO_RESPONSE_TIMEOUT", "10.0")
    )  # Timeout for MacEcho response (seconds)

    # Creative mode settings (Ollama + Qwen3-TTS)
    CREATIVE_LLM = os.getenv("CREATIVE_LLM", "llama3")
    CREATIVE_DEFAULT_EMOTION = os.getenv("CREATIVE_EMOTION", "")
    CREATIVE_TIMEOUT = 120.0  # Timeout for LLM response (seconds)

    # Dubbing mode settings (Qwen3-TTS/MLX-Audio high-fidelity)
    DUBBING_CHUNK_SIZE = 256  # Process script in chunks
    DUBBING_TIMEOUT = 60.0  # Timeout for TTS generation
    DUBBING_LANGUAGE = os.getenv("DUBBING_LANGUAGE", "")  # Default language for TTS (empty = auto-detect)
    SUPPORTED_LANGUAGES = [
        "english",
        "chinese",
        "japanese",
        "korean",
        "german",
        "french",
        "russian",
        "portuguese",
        "spanish",
        "italian",
    ]  # Languages supported by Qwen3-TTS and MLX-Audio

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
            errors.append("VAD THRESHOLD must be between 0.0 and 1.0")

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
        print(
            f"  OS: {cls.model.PLATFORM_INFO['os']} ({cls.model.PLATFORM_INFO['machine']})"
        )
        print(f"  Python: {cls.model.PLATFORM_INFO['python_version']}")
        print(f"  CUDA Available: {cls.model.PLATFORM_INFO['cuda_available']}")
        print(f"  MPS Available: {cls.model.PLATFORM_INFO['mps_available']}")

        print("\nAudio:")
        print(f"  Input SR: {cls.audio.INPUT_SAMPLE_RATE} Hz")
        print(f"  Output SR: {cls.audio.OUTPUT_SAMPLE_RATE} Hz")
        print(f"  Buffer: {cls.audio.BUFFER_DURATION}s")

        print("\nModels:")
        print(f"  Models Dir: {cls.model.MODELS_DIR}")
        print(f"  Offline Mode: {cls.model.OFFLINE_MODE}")
        print(f"  Whisper: {cls.model.WHISPER_MODEL}")
        print(f"  LLM: {cls.model.OLLAMA_MODEL}")
        print(f"  Device: {cls.model.DEVICE}")
        print(f"  Dtype: {cls.model.DTYPE}")
        print(f"  Ollama: {cls.model.OLLAMA_HOST}")

        print("\nModel Status:")
        models_to_check = ["qwen3-tts", "personaplex", "silero-vad", "whisper"]
        # Add MacEcho models on macOS
        import sys

        if sys.platform == "darwin":
            models_to_check.append("macecho")
        for model in models_to_check:
            status = (
                "[OK] Downloaded"
                if cls.model.is_model_downloaded(model)
                else "[X] Not found"
            )
            print(f"  {model}: {status}")

        print("\nVAD:")
        print(f"  Threshold: {cls.vad.THRESHOLD}")
        print(f"  Barge-in: {cls.vad.ENABLE_BARGE_IN}")

        print("\nApplication:")
        print(f"  Default Mode: {cls.app.DEFAULT_MODE}")
        print(f"  Available Modes: {', '.join(cls.app.MODES)}")
        print(f"  Log Level: {cls.app.LOG_LEVEL}")

        print("\nMode Settings:")
        print("  Real-Time:")
        print(f"    Voice: {cls.app.REALTIME_VOICE}")
        print(f"    Persona: {cls.app.REALTIME_PERSONA[:50]}...")
        print(f"    Streaming: {cls.app.REALTIME_STREAMING}")
        print("  Creative:")
        print(f"    LLM: {cls.app.CREATIVE_LLM}")
        print(f"    Emotion: {cls.app.CREATIVE_DEFAULT_EMOTION or 'None'}")
        print("  Dubbing:")
        print(f"    Chunk Size: {cls.app.DUBBING_CHUNK_SIZE}\n")


if __name__ == "__main__":
    Config.print_config()
    Config.validate()
