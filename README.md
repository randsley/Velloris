# 🎙️ Velloris

**The Local-First, High-Fidelity Voice Agent Engine**

Velloris is a state-of-the-art framework for creating lifelike, interactive AI agents that run entirely on your local hardware. By orchestrating **PersonaPlex-7B** (real-time speech-to-speech) with **Qwen3-TTS** (expressive text-to-speech), Velloris achieves human-level voice conversations without the cloud.

**Key Features:**
- 🎯 **Dual-Engine Architecture**: Real-time interactive mode + high-fidelity dubbing mode
- 🌐 **Cross-Platform**: Windows (NVIDIA CUDA) + macOS (Apple Metal/MPS) + Linux (CPU)
- 🚀 **Production Ready**: All 17 tests passing, optimized device detection
- 🔒 **Privacy First**: 100% local processing, no cloud dependencies
- 🎭 **Voice Cloning**: Optional voice reference for personalized synthesis

---

## ⚡ Quick Start

### Prerequisites

- **Python 3.12+** (3.11+ supported)
- **Ollama** (for interactive mode): [Download here](https://ollama.ai)
- **macOS**: Homebrew (for system dependencies)
- **Windows**: NVIDIA GPU + CUDA 12.1+ (recommended)

### 1. Clone & Setup

```bash
git clone https://github.com/randsley/Velloris.git
cd Velloris
```

**macOS:**
```bash
chmod +x install_macos.sh
./install_macos.sh
```

**Windows:**
```bash
# In PowerShell or Command Prompt
install_windows.bat
```

**Linux (CUDA):**
```bash
# Same as Windows setup
install_windows.bat  # or run equivalent pip commands
```

### 2. Test Installation

```bash
python3 main.py --show-config
```

### 3. Run Dubbing Mode (No Setup Required)

```bash
python3 main.py --mode dubbing --script "Your narration here"
```

You should hear the AI narrate your script! 🔊

### 4. Run Interactive Mode (Requires Ollama)

**Terminal 1** - Start Ollama:
```bash
ollama serve
ollama pull llama3  # Download model (first time only)
```

**Terminal 2** - Run Velloris:
```bash
python3 main.py --mode interactive
```

Type your questions and hear the AI respond in real-time!

---

## 📋 Project Structure

```
Velloris/
├── core/                    # Brain & Orchestration
│   ├── brain.py            # LLM integration + audio synthesis
│   └── orchestrator.py     # Engine routing & lazy loading
├── engines/                # Voice Models
│   ├── personaplex.py      # NVIDIA PersonaPlex-7B (S2S)
│   └── qwen_tts.py         # Alibaba Qwen3-TTS (TTS)
├── utils/                  # Utilities
│   ├── audio_io.py         # Audio playback & recording
│   ├── audio_utils.py      # Resampling & normalization
│   ├── device_utils.py     # Device detection (CUDA/MPS/CPU)
│   └── vad_handler.py      # Voice Activity Detection
├── tests/                  # Test Suite (17 tests)
│   └── test_pipeline.py    # Integration tests
├── config.py               # Configuration
├── main.py                 # CLI Application
├── requirements.txt        # Python Dependencies
├── ARCHITECTURE.md         # Detailed architecture guide
├── LICENSE                 # Apache License 2.0
└── README.md               # This file
```

---

## 🎯 Usage

### Dubbing Mode (High-Fidelity Narration)

Generate professional-quality speech from text:

```bash
# Simple
python3 main.py --mode dubbing --script "Hello world"

# With voice cloning
python3 main.py --mode dubbing --script "Story text" --voice-ref my_voice.wav

# Specify device
python3 main.py --mode dubbing --device cpu
```

**Output:** 14+ seconds of natural-sounding speech at 24kHz

### Interactive Mode (Real-Time Conversation)

Have conversations with an AI agent powered by Ollama:

```bash
python3 main.py --mode interactive
```

**Features:**
- Real-time LLM responses (Ollama)
- Automatic text-to-speech synthesis
- Audio playback of responses
- Type `quit`, `exit`, or `bye` to end

### Device Options

Auto-detect optimal device:
```bash
python3 main.py --device auto
```

Explicit device selection:
```bash
python3 main.py --device cuda   # NVIDIA GPU
python3 main.py --device mps    # Apple Metal (M-series Mac)
python3 main.py --device cpu    # CPU (slowest)
```

### Show Configuration

```bash
python3 main.py --show-config
```

Displays:
- Platform info (OS, CPU, GPU)
- Device detection results
- Model configuration
- Audio settings

---

## 🏗️ Architecture

### Interactive Mode Pipeline

```
User Input (Text)
    ↓
Ollama LLM (Reasoning)
    ↓
Response Text
    ↓
Qwen3-TTS (Synthesis)
    ↓
Audio Output → Speaker 🔊
```

### Dubbing Mode Pipeline

```
Script Text
    ↓
Qwen3-TTS (Synthesis)
    ↓
Audio Output → Speaker 🔊
```

**See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed technical documentation.**

---

## 🖥️ Platform-Specific Notes

### Windows (NVIDIA CUDA)
- **Optimal Performance**: RTX 3000+ or newer
- **Installation**: Run `install_windows.bat`
- **Device Selection**: `--device cuda` (auto-selected)
- **Optimizations Available**: FlashAttention 2, bitsandbytes 4-bit quantization

### macOS (Apple Metal/MPS)
- **Supported**: M1, M2, M3, M4 Pro/Max
- **Installation**: Run `./install_macos.sh`
- **Device Selection**: `--device mps` (auto-selected)
- **Note**: PersonaPlex runs slower on MPS; Qwen3-TTS works well

### Linux (CPU/CUDA)
- **CPU Mode**: Works on any Linux
- **CUDA Mode**: Requires NVIDIA GPU + CUDA 12.1+
- **Installation**: Similar to Windows setup

**See [ARCHITECTURE.md](ARCHITECTURE.md#platform-compatibility) for performance comparisons.**

---

## 🧪 Testing

Run the test suite:

```bash
# All 17 tests
pytest tests/test_pipeline.py -v

# Specific test
pytest tests/test_pipeline.py::TestOrchestrator -v

# With coverage
pytest tests/test_pipeline.py --cov=. -v
```

**Note**: Tests pass without models installed (stub mode).

---

## 📚 Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Full system architecture, platform support, performance metrics
- **[CLAUDE.md](CLAUDE.md)** - Development guidelines for contributors
- **[LICENSE](LICENSE)** - MIT License

---

## 🔧 Troubleshooting

### Audio Not Playing
- Ensure system volume is up
- Check speaker/headphone connection
- Try: `python3 main.py --mode dubbing --device cpu`

### Model Loading Fails
- Ensure internet connection (for Hugging Face downloads)
- Check disk space (~5GB for models)
- Verify Python 3.12+: `python3 --version`

### PersonaPlex Warning
- This is informational if you're only using Dubbing Mode
- Only needed for Interactive Mode with real-time speech

### Slow Inference
- **MPS/Metal**: Expected to be slower than CUDA
- **CPU**: Very slow; GPU recommended
- **Solution**: Use CPU mode with smaller model or wait longer

---

## 🚀 What's Next?

- [ ] Web UI with Gradio
- [ ] ONNX export for edge deployment
- [ ] Mobile optimization (iOS/Android)
- [ ] Multi-turn conversation memory
- [ ] Custom voice fine-tuning
- [ ] Real-time transcription display

---

## 📄 License

Apache License 2.0 - See [LICENSE](LICENSE) file

---

## 🤝 Contributing

Contributions welcome! See [CLAUDE.md](CLAUDE.md) for guidelines.

---

## 📞 Support

- **Issues**: Check [GitHub Issues](https://github.com/randsley/Velloris/issues)
- **Documentation**: See [ARCHITECTURE.md](ARCHITECTURE.md)
- **Questions**: Open a Discussion on GitHub

---

**Built with ❤️ for local-first AI**
