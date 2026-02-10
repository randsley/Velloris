# 🚀 Velloris Quick Start Guide

Get up and running with Velloris in **5 minutes**! This guide will help you install and test all three modes.

## ⚡ Prerequisites Checklist

Before you begin, make sure you have:

- [ ] **Python 3.12+** installed (`python3 --version`)
- [ ] **Git** installed
- [ ] **16GB+ RAM** recommended
- [ ] **GPU** (optional but recommended):
  - NVIDIA GPU with 16GB+ VRAM for realtime mode
  - 6GB+ VRAM for dubbing/creative modes
  - macOS M1/M2/M3/M4 works with MPS
- [ ] **Ollama** installed (only for creative mode) - [Download here](https://ollama.ai)

---

## 📦 Step 1: Installation (2 minutes)

### macOS

```bash
# Clone repository
git clone https://github.com/randsley/Velloris.git
cd Velloris

# Run installation script
chmod +x install_macos.sh
./install_macos.sh

# Activate environment
source venv_py312/bin/activate
```

### Windows

```bash
# Clone repository
git clone https://github.com/randsley/Velloris.git
cd Velloris

# Run installation script
install_windows.bat

# Activate environment
venv_py312\Scripts\activate
```

### Linux

```bash
# Clone repository
git clone https://github.com/randsley/Velloris.git
cd Velloris

# Install dependencies (similar to Windows)
python3 -m venv venv_py312
source venv_py312/bin/activate
pip install -r requirements.txt
```

---

## ✅ Step 2: Verify Installation (30 seconds)

```bash
python3 main.py --show-config
```

**Expected Output:**
```
=== Velloris Configuration ===

Platform:
  OS: Darwin (arm64)  # or Windows/Linux
  Python: 3.12.x
  CUDA Available: True/False
  MPS Available: True/False

Audio:
  Input SR: 16000 Hz
  Output SR: 24000 Hz
  Buffer: 2.0s

Models:
  Device: cuda/mps/cpu
  Dtype: bfloat16/float32

Application:
  Default Mode: realtime
  Available Modes: realtime, dubbing, creative
```

---

## 🎯 Step 3: Choose Your Mode

Velloris has **three modes**. Pick the one that fits your use case:

| Mode | Best For | Status | Latency Target |
|------|----------|--------|----------------|
| **Realtime** | Conversations, interactive chat | 🔧 Infrastructure Ready | 70-170ms (CUDA) |
| **Dubbing** | Narration, audiobooks, videos | ✅ Production Ready | N/A |
| **Creative** | Storytelling, emotional content | ✅ Production Ready | 1-3s |

---

## 🎙️ Mode 1: Realtime Conversation

**Status:** 🔧 **Infrastructure Ready** (99 tests passing) | **Target: Windows/Linux CUDA**

### Current State

**✅ What's Working:**
- Complete audio I/O pipeline (microphone/speaker)
- Voice Activity Detection (Silero VAD)
- Background transcription (MLX-Whisper)
- Interruption/barge-in capability
- 99 comprehensive tests validating infrastructure

**⚠️ What's Pending:**
- **Windows/Linux (CUDA):** Requires PersonaPlex-7B installation
- **macOS (Apple Silicon):** Requires MacEcho integration (future)

### Target Performance (CUDA Systems)

When PersonaPlex-7B is installed on Windows/Linux with NVIDIA GPU:

**Target Features:**
- ⚡ Ultra-low latency (70-170ms)
- ✅ Full-duplex (natural interruptions)
- 🎭 16 voice options
- ❌ No Ollama needed
- 🎯 Real-time speech-to-speech

### Try It (Infrastructure Mode):

```bash
# Test the complete audio infrastructure
python3 main.py --mode realtime --persona "You are a helpful assistant" --voice NATF2
```

**What happens (current):**
1. ✅ Audio I/O system initializes
2. ✅ You speak into the microphone (captured and transcribed)
3. ⚠️ S2S engine returns stub response (infrastructure validated)
4. ✅ Interruption handling works correctly
5. Press Ctrl+C to exit

**For production voice synthesis, use Creative or Dubbing modes below.**

### Setup for Production Use (CUDA Required)

**Requirements:**
- NVIDIA GPU (Ampere+: RTX 3000/4000, A100, H100)
- 16GB+ VRAM
- Windows or Linux
- CUDA 12.1+

**Installation (PersonaPlex-7B):**
```bash
# 1. Install system dependencies
# Ubuntu/Debian:
sudo apt install libopus-dev

# 2. Clone PersonaPlex repository
git clone https://github.com/NVIDIA/personaplex

# 3. Install PersonaPlex
pip install personaplex/moshi/.

# 4. Login to Hugging Face (accept model license)
huggingface-cli login

# 5. Run realtime mode
python3 main.py --mode realtime --persona "helpful assistant" --voice NATF2
```

**Target Voices (when PersonaPlex installed):**
- Female: `NATF0`, `NATF1`, `NATF2`, `NATF3`, `VARF0-4`
- Male: `NATM0`, `NATM1`, `NATM2`, `NATM3`, `VARM0-4`

**macOS Users:** Realtime infrastructure is validated. MacEcho integration planned for future release. Use Creative or Dubbing modes for production.

---

## 📖 Mode 2: High-Fidelity Dubbing

**Status:** ✅ **Production Ready** (User-verified quality)

**Best for:** Content creation, video narration, audiobooks, podcasts

**Features:**
- 🎨 Professional quality audio (MLX-Audio TTS)
- 🌍 10 languages supported
- 🎭 Voice cloning capability
- ❌ No Ollama needed
- ✅ Works on all platforms (CUDA, MPS, CPU)

### Try It:

```bash
python3 main.py --mode dubbing --script "Once upon a time in a digital landscape, AI models lived in harmony."
```

**What happens:**
1. Velloris loads Qwen3-TTS (one-time, ~15 seconds)
2. Generates high-quality narration
3. Plays audio through speakers
4. Audio saved to `output.wav` (if configured)

### With Voice Cloning:

```bash
python3 main.py --mode dubbing --script "Your narration here" --voice-ref voices/my_voice.wav
```

**Tip:** Provide a 3-5 second reference audio for best results

---

## 🎨 Mode 3: Creative Synthesis

**Status:** ✅ **Production Ready** (User-verified: "perfect audio")

**Best for:** Storytelling, creative writing, emotional content

**Features:**
- 🧠 LLM reasoning (Ollama)
- 🎭 Emotion control (natural language prompts)
- 🌍 Multilingual support
- 🎨 High-quality MLX-Audio TTS
- ✅ Requires Ollama running
- ✅ Works on all platforms (CUDA, MPS, CPU)

### Setup Ollama (One-Time):

```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Download model (first time only)
ollama pull llama3
```

### Try It:

```bash
# Terminal 2 (with Ollama running in Terminal 1):
python3 main.py --mode creative --script "Tell me a short story about a space explorer" --emotion "Speak with excitement"
```

**What happens:**
1. Velloris connects to Ollama
2. LLM generates creative response
3. Qwen3-TTS synthesizes with emotion
4. Plays audio through speakers

---

## 🔧 Quick Troubleshooting

### "Realtime mode returns silence"
**Explanation:** Realtime mode infrastructure is complete (99 tests passing), but S2S engines require installation:
- **Windows/Linux:** Install PersonaPlex-7B (see setup instructions above)
- **macOS:** MacEcho integration pending (use Creative or Dubbing modes)

**Current Options:**
```bash
# ✅ Production: Creative mode (LLM + TTS)
python3 main.py --mode creative --script "Tell me a story" --emotion "excited"

# ✅ Production: Dubbing mode (high-quality TTS)
python3 main.py --mode dubbing --script "Test narration"
```

### "Ollama not available"
**Solution:** Ollama is only required for creative mode. Make sure it's running:
```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Pull model and run
ollama pull llama3
python3 main.py --mode creative --script "Test"
```

### "No audio output"
**Solution:**
- Check system volume
- Verify speaker/headphone connection
- Try CPU mode: `python3 main.py --mode dubbing --script "Test" --device cpu`
- Verify with: `python3 main.py --show-config`

### "CUDA out of memory"
**Solution:**
- Close other GPU applications
- Use smaller LLM: `--llm-model llama3:8b` (creative mode)
- Try CPU mode: `--device cpu`
- Dubbing mode uses less VRAM than creative mode

---

## 📚 What's Next?

### Learn More

- **[README.md](README.md)** - Full documentation and usage guide
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Technical architecture and design
- **[MIGRATION.md](MIGRATION.md)** - Upgrading from v1.x
- **[FAQ.md](FAQ.md)** - Common questions answered
- **[EXAMPLES.md](EXAMPLES.md)** - More code examples

### Customize Your Setup

1. **Edit Configuration:**
   ```bash
   cp .env.example .env
   nano .env  # Edit default settings
   ```

2. **Change Default Mode:**
   ```bash
   # In .env file
   DEFAULT_MODE=realtime  # or dubbing, creative
   ```

3. **Customize Voice:**
   ```bash
   # In .env file
   REALTIME_VOICE=NATM1  # Male voice
   REALTIME_PERSONA="You are a friendly tutor"
   ```

### Run Tests

```bash
# All tests (99 total: 98 passing, 1 skipped)
pytest tests/ -v

# Integration tests only (17 tests)
pytest tests/test_pipeline.py -v

# Realtime infrastructure tests (40 tests)
pytest tests/test_realtime_callbacks.py tests/test_vad_interruption.py tests/test_realtime_e2e.py -v
```

### Explore Examples

```bash
# Test all modes
python3 main.py --mode realtime --device cpu
python3 main.py --mode dubbing --script "Welcome to Velloris" --device cpu
python3 main.py --mode creative --script "Hello" --device cpu  # Requires Ollama
```

---

## 🤝 Get Help

- **Issues**: [GitHub Issues](https://github.com/randsley/Velloris/issues)
- **Discussions**: [GitHub Discussions](https://github.com/randsley/Velloris/discussions)
- **Documentation**: [Full Docs](README.md)

---

## ⏱️ Quick Reference Card

```bash
# Show configuration
python3 main.py --show-config

# Realtime mode (fastest, no Ollama)
python3 main.py --mode realtime --persona "helpful assistant" --voice NATF2

# Dubbing mode (high quality, no Ollama)
python3 main.py --mode dubbing --script "Your text here"

# Creative mode (LLM reasoning, needs Ollama)
ollama serve  # Terminal 1
python3 main.py --mode creative --script "Tell a story" --emotion "excited"  # Terminal 2

# Device selection
python3 main.py --device auto    # Auto-detect (default)
python3 main.py --device cuda    # NVIDIA GPU
python3 main.py --device mps     # Apple Metal (M1/M2/M3/M4)
python3 main.py --device cpu     # CPU fallback
```

---

**🎉 Congratulations! You're ready to use Velloris!**

For more advanced usage, see the [full documentation](README.md).
