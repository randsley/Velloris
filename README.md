# 🎙️ Velloris

**The Local-First, High-Fidelity Voice Agent Engine**

Velloris is a state-of-the-art framework for creating lifelike, interactive AI agents that run entirely on your local hardware. With three specialized modes, Velloris delivers the perfect voice AI solution for any use case—from ultra-low latency conversations to professional-quality content creation.

**Key Features:**
- ⚡ **Three-Mode Architecture**: Real-time S2S + high-fidelity dubbing + creative synthesis
- 🎯 **70-170ms Latency**: Full-duplex conversations with PersonaPlex-7B (18x faster than Gemini Live!)
- 🌐 **Cross-Platform**: Windows (NVIDIA CUDA) + macOS (Apple Metal/MPS) + Linux (CPU)
- 🚀 **Production Ready**: Optimized device detection, lazy loading, mode-based routing
- 🔒 **Privacy First**: 100% local processing, no cloud dependencies
- 🎭 **10 Languages**: Multilingual support via Qwen3-TTS
- 🧠 **Ollama Optional**: Not required for basic conversations

---

## 🔄 **IMPORTANT: Migration Notice**

**Velloris v2.0** introduces a new three-mode architecture with significantly improved performance!

**If you were using `--mode interactive`:**
- ✅ **Use `--mode realtime`** for faster, full-duplex conversations (no Ollama needed!)
- ✅ **Use `--mode creative`** for LLM-powered emotional content (similar to old behavior)

Old commands still work with deprecation warnings. See [MIGRATION.md](MIGRATION.md) for details.

---

## ⚡ Quick Start

### Prerequisites

- **Python 3.12+** (3.11+ supported)
- **For Real-Time Mode**: NVIDIA GPU (16GB+ VRAM) + CUDA 12.1+
- **For Creative Mode**: Ollama running ([Download here](https://ollama.ai))
- **For Dubbing Mode**: GPU recommended (6GB+ VRAM) or CPU
- **macOS**: Homebrew (for system dependencies)
- **Windows/Linux**: NVIDIA GPU recommended for best performance

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

### 3. Choose Your Mode

#### **Real-Time Conversation** (PersonaPlex S2S)
Ultra-low latency, full-duplex conversations:
```bash
python3 main.py --mode realtime --persona "You are a helpful tutor" --voice NATF2
```
- ⚡ **70-170ms latency**
- ✅ **Full-duplex** (can interrupt naturally)
- ❌ **No Ollama needed**
- 🎯 **Best for**: Interactive conversations, customer service

#### **High-Fidelity Dubbing** (Qwen3-TTS)
Professional narration for content creation:
```bash
python3 main.py --mode dubbing --script "Your narration here"
```
- 🎨 **Professional quality** (12kHz)
- 🌍 **10 languages** supported
- 🎭 **Voice cloning** available
- 🎯 **Best for**: Audiobooks, podcasts, video narration

#### **Creative Assistant** (Ollama + Qwen3-TTS)
Emotional storytelling with LLM reasoning:

**Terminal 1** - Start Ollama:
```bash
ollama serve
ollama pull llama3  # First time only
```

**Terminal 2** - Run Velloris:
```bash
python3 main.py --mode creative --emotion "Speak with excitement"
```
- 🧠 **LLM reasoning** (Ollama)
- 🎭 **Emotion control**
- 🌍 **Multilingual**
- 🎯 **Best for**: Storytelling, creative content

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

## 🎯 Usage Guide

### Mode Comparison

| Feature | Real-Time | Dubbing | Creative |
|---------|-----------|---------|----------|
| **Latency** | 70-170ms ⚡ | N/A | 1-3s |
| **Full-Duplex** | ✅ Yes | ❌ No | ❌ No |
| **Interruption** | ✅ 95% success | ❌ No | ❌ No |
| **Languages** | English | 10 languages | 10 languages |
| **Voice Options** | 16 preset | Unlimited | Unlimited |
| **Emotion Control** | Limited | ✅ Yes | ✅ Yes |
| **Ollama Required** | ❌ No | ❌ No | ✅ Yes |
| **GPU Required** | ✅ NVIDIA (16GB+) | Recommended | Recommended |
| **Best For** | Conversations | Narration | Creative content |

### Real-Time Mode Examples

Ultra-low latency conversations with PersonaPlex:

```bash
# Basic conversation
python3 main.py --mode realtime

# Custom persona
python3 main.py --mode realtime --persona "You are a friendly tutor"

# Different voice
python3 main.py --mode realtime --voice NATM1 --persona "You are a wise mentor"

# Available voices: NATF0-3 (natural female), NATM0-3 (natural male),
#                   VARF0-4 (varied female), VARM0-4 (varied male)
```

**Features:**
- ⚡ **70-170ms latency** (18x faster than Gemini Live)
- ✅ **Full-duplex** (natural interruptions)
- ✅ **16 voices** with persona control
- ✅ **No LLM needed** (PersonaPlex handles reasoning)

### Dubbing Mode Examples

Professional narration with Qwen3-TTS:

```bash
# Simple narration
python3 main.py --mode dubbing --script "Hello world"

# With voice cloning (3-5 second sample)
python3 main.py --mode dubbing --script "Story text" --voice-ref my_voice.wav

# Specify device
python3 main.py --mode dubbing --script "Your script" --device cpu
```

**Features:**
- 🎨 **Professional quality** (12kHz)
- 🌍 **10 languages** (Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian)
- 🎭 **Voice cloning** from 3-second samples
- 🎨 **Voice design** via natural language

### Creative Mode Examples

Emotional storytelling with Ollama + Qwen3-TTS:

```bash
# Start Ollama first
ollama serve  # In separate terminal

# Basic creative mode
python3 main.py --mode creative --script "Tell me a story about space"

# With emotion control
python3 main.py --mode creative --script "Describe a sunset" --emotion "Speak poetically"

# Different LLM model
python3 main.py --mode creative --llm-model mistral --emotion "Excited tone"
```

**Features:**
- 🧠 **LLM reasoning** (Ollama: llama3, mistral, mixtral, etc.)
- 🎭 **Emotion control** via natural language instructions
- 🌍 **Multilingual** support
- 🎨 **Creative flexibility**

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

### Real-Time Mode Pipeline (NEW!)

```
User Speech (24kHz)
    ↓
PersonaPlex-7B (end-to-end S2S)
  • Listen & Understand
  • Reason & Respond
  • Generate Speech
    ↓
Agent Speech (24kHz) → Speaker 🔊

Latency: 70-170ms ⚡
Full-Duplex: ✅ Yes
Ollama: ❌ Not needed
```

### Dubbing Mode Pipeline

```
Script Text
    ↓
Qwen3-TTS (High-Fidelity Synthesis)
  • 10 languages
  • Voice cloning
  • Emotion control
    ↓
Audio Output (12kHz) → Speaker 🔊

Quality: Professional
Ollama: ❌ Not needed
```

### Creative Mode Pipeline

```
User Text
    ↓
Ollama LLM (Reasoning/Creativity)
    ↓
Response Text
    ↓
Qwen3-TTS (Emotional Synthesis)
    ↓
Audio Output (12kHz) → Speaker 🔊

Flexibility: High
Ollama: ✅ Required
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
- **Research**: Exploring [MLX Stack](https://ml-explore.github.io/mlx/) (Apple's ML framework) for optimized M-series inference. Could provide 2-3x speedup vs PyTorch MPS.

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
- **[LICENSE](LICENSE)** - Apache License 2.0

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

- [ ] **MLX Stack Integration** (macOS): Research & implement MLX for 2-3x speedup on M-series Macs
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
