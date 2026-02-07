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

| Mode | Best For | Requires Ollama? | Latency |
|------|----------|-----------------|---------|
| **Realtime** | Conversations, interactive chat | ❌ No | 70-170ms |
| **Dubbing** | Narration, audiobooks, videos | ❌ No | N/A |
| **Creative** | Storytelling, emotional content | ✅ Yes | 1-3s |

---

## 🎙️ Mode 1: Realtime Conversation (Fastest)

**Best for:** Interactive voice conversations, customer service, live tutoring

**Features:**
- ⚡ Ultra-low latency (70-170ms)
- ✅ Full-duplex (natural interruptions)
- 🎭 16 voice options
- ❌ No Ollama needed

### Try It:

```bash
python3 main.py --mode realtime --persona "You are a helpful assistant" --voice NATF2
```

**What happens:**
1. Velloris loads PersonaPlex-7B (one-time, ~30 seconds)
2. You speak into the microphone
3. Agent responds in real-time
4. Press Ctrl+C to exit

**Available Voices:**
- Female: `NATF0`, `NATF1`, `NATF2`, `NATF3`, `VARF0-4`
- Male: `NATM0`, `NATM1`, `NATM2`, `NATM3`, `VARM0-4`

---

## 📖 Mode 2: High-Fidelity Dubbing

**Best for:** Content creation, video narration, audiobooks, podcasts

**Features:**
- 🎨 Professional quality audio
- 🌍 10 languages supported
- 🎭 Voice cloning capability
- ❌ No Ollama needed

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

## 🎨 Mode 3: Creative Synthesis (Most Flexible)

**Best for:** Storytelling, creative writing, emotional content

**Features:**
- 🧠 LLM reasoning (Ollama)
- 🎭 Emotion control
- 🌍 Multilingual
- ✅ Requires Ollama running

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

### "Ollama not available"
**Solution:** Make sure Ollama is running:
```bash
# Terminal 1
ollama serve

# Terminal 2
ollama pull llama3
python3 main.py --mode creative --script "Test"
```

### "PersonaPlex engine not available"
**Solution:** PersonaPlex requires NVIDIA GPU. Try dubbing mode instead:
```bash
python3 main.py --mode dubbing --script "Test narration"
```

### "No audio output"
**Solution:**
- Check system volume
- Verify speaker/headphone connection
- Try CPU mode: `python3 main.py --mode dubbing --script "Test" --device cpu`

### "CUDA out of memory"
**Solution:**
- Close other GPU applications
- Use smaller model: `--llm-model llama3:8b` (creative mode)
- Try CPU mode: `--device cpu`

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
pytest tests/test_pipeline.py -v
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
