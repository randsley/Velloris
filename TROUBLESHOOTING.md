# 🔧 Troubleshooting Guide

This guide helps you diagnose and fix common issues with Velloris v2.0.

---

## Table of Contents

- [Installation Issues](#installation-issues)
- [Ollama Issues](#ollama-issues)
- [GPU & CUDA Issues](#gpu--cuda-issues)
- [Audio Issues](#audio-issues)
- [Model Loading Issues](#model-loading-issues)
- [Performance Issues](#performance-issues)
- [Mode-Specific Issues](#mode-specific-issues)
- [Migration Issues](#migration-issues)
- [Advanced Debugging](#advanced-debugging)

---

## Installation Issues

### "Command not found: python3"

**Problem:** Python 3 not installed or not in PATH.

**Solution:**

```bash
# macOS (via Homebrew)
brew install python@3.12

# Ubuntu/Debian
sudo apt update
sudo apt install python3.12

# Windows
# Download from https://www.python.org/downloads/
# Make sure "Add Python to PATH" is checked during installation
```

Verify installation:
```bash
python3 --version  # Should show 3.12+
```

---

### "No module named 'torch'"

**Problem:** PyTorch not installed or virtual environment not activated.

**Solution:**

```bash
# Make sure you're in the Velloris directory
cd /path/to/Velloris

# Activate virtual environment
source venv_py312/bin/activate  # macOS/Linux
venv_py312\Scripts\activate     # Windows

# Reinstall requirements
pip install --upgrade pip
pip install -r requirements.txt
```

If still failing:
```bash
# Install PyTorch manually
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

### "pip install" fails with SSL errors

**Problem:** Network or certificate issues.

**Solution:**

```bash
# Try with --trusted-host flags
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt

# Or upgrade pip first
python3 -m pip install --upgrade pip

# Then retry
pip install -r requirements.txt
```

---

### Installation script fails on macOS

**Problem:** Permission denied or script not executable.

**Solution:**

```bash
# Make script executable
chmod +x install_macos.sh

# Run script
./install_macos.sh

# If still failing, run manually:
python3 -m venv venv_py312
source venv_py312/bin/activate
pip install -r requirements.txt
```

---

## Ollama Issues

### "Ollama not available" or "Connection refused"

**Problem:** Ollama not installed or not running.

**Diagnosis:**
```bash
# Check if Ollama is installed
ollama --version

# Check if Ollama is running
curl http://localhost:11434/api/version
```

**Solution:**

1. **Install Ollama** (if not installed):
   ```bash
   # macOS/Linux
   curl -fsSL https://ollama.ai/install.sh | sh

   # Windows: Download from https://ollama.ai
   ```

2. **Start Ollama**:
   ```bash
   # Terminal 1
   ollama serve
   ```

3. **Download a model** (if first time):
   ```bash
   # Terminal 2
   ollama pull llama3
   ```

4. **Run Velloris**:
   ```bash
   # Terminal 2 (with Ollama running in Terminal 1)
   python3 main.py --mode creative --script "Test"
   ```

---

### "Model not found" error

**Problem:** LLM model not downloaded by Ollama.

**Solution:**

```bash
# List available models
ollama list

# Download missing model
ollama pull llama3

# Or specify a different model
python3 main.py --mode creative --script "Test" --llm-model mistral
```

---

### Ollama is slow or unresponsive

**Problem:** System resources exhausted or large model.

**Diagnosis:**
```bash
# Check system resources
top  # macOS/Linux
# Look for "ollama" process

# Windows: Task Manager → Performance tab
```

**Solution:**

1. **Use a smaller model**:
   ```bash
   ollama pull llama3:8b  # Instead of default llama3:70b
   python3 main.py --mode creative --script "Test" --llm-model llama3:8b
   ```

2. **Close other applications** to free RAM

3. **Restart Ollama**:
   ```bash
   # Kill Ollama
   pkill ollama  # macOS/Linux
   # Windows: Task Manager → End Task

   # Restart
   ollama serve
   ```

---

### "Only creative mode requires Ollama"

**Problem:** Trying to use Ollama with realtime or dubbing modes.

**Solution:**

This is **by design**. Ollama is only needed for creative mode.

- **Realtime Mode**: Uses PersonaPlex-7B (no LLM needed)
- **Dubbing Mode**: Uses Qwen3-TTS (no LLM needed)
- **Creative Mode**: Uses Ollama + Qwen3-TTS

If you don't want to use Ollama, use realtime or dubbing mode instead.

---

## GPU & CUDA Issues

### "CUDA not available" or "torch.cuda.is_available() returns False"

**Problem:** CUDA not installed or not detected.

**Diagnosis:**
```bash
# Check CUDA installation
nvcc --version

# Check PyTorch CUDA
python3 -c "import torch; print(torch.cuda.is_available())"
python3 -c "import torch; print(torch.version.cuda)"
```

**Solution:**

1. **Install CUDA Toolkit** (if not installed):
   - Download from https://developer.nvidia.com/cuda-downloads
   - Install CUDA 11.8 or 12.1 (check PyTorch compatibility)

2. **Reinstall PyTorch with CUDA**:
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Verify CUDA**:
   ```bash
   python3 -c "import torch; print(torch.cuda.is_available())"
   # Should print: True
   ```

4. **If still failing, use CPU mode**:
   ```bash
   python3 main.py --mode dubbing --device cpu --script "Test"
   ```

---

### "CUDA out of memory" error

**Problem:** Insufficient VRAM for the model.

**Diagnosis:**
```bash
# Check VRAM usage
nvidia-smi

# Expected for Velloris:
# - Realtime mode: 16GB+ VRAM required
# - Dubbing mode: 4-6GB VRAM
# - Creative mode: 6-8GB VRAM
```

**Solution:**

1. **Close other GPU applications**:
   ```bash
   # Check what's using GPU
   nvidia-smi

   # Close browsers, games, other AI tools
   ```

2. **Use a smaller model** (creative mode only):
   ```bash
   python3 main.py --mode creative --script "Test" --llm-model llama3:8b
   ```

3. **Try CPU mode** (dubbing/creative only):
   ```bash
   python3 main.py --mode dubbing --device cpu --script "Test"
   ```

4. **Upgrade GPU** (for realtime mode, 16GB+ VRAM required)

---

### "PersonaPlex engine not available" error

**Problem:** PersonaPlex requires NVIDIA GPU with CUDA. Not available on:
- AMD GPUs
- Apple Silicon (M1/M2/M3/M4)
- CPUs

**Solution:**

**Option 1: Use Dubbing Mode** (no GPU required)
```bash
python3 main.py --mode dubbing --script "Your narration here"
```

**Option 2: Use Creative Mode** (works on CPU/MPS)
```bash
python3 main.py --mode creative --script "Your story here"
```

**Option 3: Get an NVIDIA GPU** (for realtime mode)
- RTX 3090 (24GB VRAM)
- RTX 4090 (24GB VRAM)
- A100 (40GB/80GB VRAM)

---

### "MPS not available" on Apple Silicon

**Problem:** Metal Performance Shaders not detected.

**Diagnosis:**
```bash
python3 -c "import torch; print(torch.backends.mps.is_available())"
# Should print: True on M1/M2/M3/M4
```

**Solution:**

1. **Update PyTorch**:
   ```bash
   pip install --upgrade torch torchvision torchaudio
   ```

2. **Verify macOS version** (MPS requires macOS 12.3+):
   ```bash
   sw_vers
   ```

3. **If still failing, use CPU**:
   ```bash
   python3 main.py --mode dubbing --device cpu --script "Test"
   ```

---

## Audio Issues

### No audio output / silent playback

**Problem:** Audio device not configured or system volume muted.

**Diagnosis:**
```bash
# List audio devices
python3 main.py --list-devices

# Check output file was created
ls -lh output.wav
file output.wav
```

**Solution:**

1. **Check system volume** and unmute speakers

2. **Verify audio device**:
   ```bash
   # macOS
   system_profiler SPAudioDataType

   # Linux
   aplay -l
   ```

3. **Try playing the output file manually**:
   ```bash
   # macOS
   afplay output.wav

   # Linux
   aplay output.wav

   # Windows
   start output.wav
   ```

4. **If file is silent, try CPU mode**:
   ```bash
   python3 main.py --mode dubbing --device cpu --script "Test audio"
   ```

---

### Audio is distorted, robotic, or low quality

**Problem:** Wrong mode, sample rate, or CPU overload.

**Diagnosis:**

Check which mode you're using:
- **Realtime Mode**: Optimized for latency, not quality
- **Dubbing Mode**: High-quality narration
- **Creative Mode**: High-quality with emotion

**Solution:**

1. **Use Dubbing Mode for narration**:
   ```bash
   python3 main.py --mode dubbing --script "High quality audio test"
   ```

2. **Check sample rate** in `.env`:
   ```bash
   OUTPUT_SAMPLE_RATE=24000  # Should be 24000, not 16000
   ```

3. **Close background applications** to reduce CPU load

4. **Try GPU mode** (faster processing):
   ```bash
   python3 main.py --mode dubbing --device cuda --script "Test"
   ```

---

### Audio playback is choppy or stuttering

**Problem:** System resources exhausted or disk I/O issues.

**Solution:**

1. **Close other applications** to free CPU/RAM

2. **Check disk space**:
   ```bash
   df -h  # macOS/Linux
   # Make sure you have 5GB+ free
   ```

3. **Try CPU mode** (if GPU is overloaded):
   ```bash
   python3 main.py --mode dubbing --device cpu --script "Test"
   ```

4. **Reduce buffer size** in `config.py`:
   ```python
   AUDIO_BUFFER_SIZE = 1.0  # Reduce from 2.0
   ```

---

### "No microphone detected" (realtime mode)

**Problem:** Microphone not connected or not permitted.

**Diagnosis:**
```bash
# List input devices
python3 main.py --list-devices

# macOS: Check permissions
# System Settings → Privacy & Security → Microphone
```

**Solution:**

1. **Connect a microphone**

2. **Grant microphone permission** (macOS):
   - System Settings → Privacy & Security → Microphone
   - Enable for Terminal/iTerm

3. **Test microphone**:
   ```bash
   # macOS
   rec test.wav  # Speak, then Ctrl+C
   play test.wav

   # Linux
   arecord -d 5 test.wav
   aplay test.wav
   ```

4. **Specify microphone device**:
   ```bash
   python3 main.py --mode realtime --input-device 2
   ```

---

## Model Loading Issues

### "Failed to load PersonaPlex model"

**Problem:** Insufficient VRAM, CUDA not available, or model weights not downloaded.

**Diagnosis:**
```bash
# Check VRAM
nvidia-smi

# Check CUDA
python3 -c "import torch; print(torch.cuda.is_available())"
```

**Solution:**

1. **Verify 16GB+ VRAM available**:
   ```bash
   nvidia-smi
   # Close other GPU applications if needed
   ```

2. **Model downloads automatically on first run**
   - Wait for download to complete (~10-20GB)
   - Check `~/.cache/huggingface/` for model files

3. **Use alternative modes**:
   ```bash
   python3 main.py --mode dubbing --script "Test"
   ```

---

### "Failed to load Qwen3-TTS model"

**Problem:** Model weights not downloaded or incompatible version.

**Solution:**

1. **Model downloads automatically on first run**
   - Wait for download to complete (~2-5GB)
   - Check `~/.cache/huggingface/` for model files

2. **Manually download** (if auto-download fails):
   ```python
   from transformers import AutoModel
   model = AutoModel.from_pretrained("Qwen/Qwen3-TTS")
   ```

3. **Check internet connection** (models download from HuggingFace)

---

### Models load very slowly (10+ minutes)

**Problem:** Slow internet connection or disk I/O.

**Solution:**

1. **First-time load is normal** (downloading 10-20GB)
   - Subsequent loads should be faster (<30 seconds)

2. **Check download progress**:
   ```bash
   # Monitor cache directory
   watch du -sh ~/.cache/huggingface/
   ```

3. **Use faster storage** (SSD instead of HDD)

4. **Pre-download models**:
   ```bash
   python3 main.py --mode dubbing --script "Test"  # Downloads Qwen3-TTS
   python3 main.py --mode realtime --persona "Test"  # Downloads PersonaPlex
   ```

---

## Performance Issues

### High latency in realtime mode (500ms+)

**Expected:** 70-170ms with NVIDIA GPU
**Problem:** Something is slowing down inference.

**Diagnosis:**
```bash
# Check GPU usage
nvidia-smi

# Check CPU usage
top  # macOS/Linux
```

**Solution:**

1. **Close other GPU applications**:
   ```bash
   nvidia-smi
   # Look for other processes using GPU
   ```

2. **Verify GPU mode**:
   ```bash
   python3 main.py --show-config
   # Check "Device: cuda" (not "cpu")
   ```

3. **Check thermal throttling**:
   ```bash
   nvidia-smi
   # Look at "Temp" column (should be <85°C)
   ```

4. **Upgrade GPU** (realtime mode needs high-end GPU)

---

### Creative mode is very slow (10+ seconds)

**Expected:** 1-3 seconds
**Problem:** Large Ollama model or CPU bottleneck.

**Solution:**

1. **Use a smaller model**:
   ```bash
   ollama pull llama3:8b
   python3 main.py --mode creative --script "Test" --llm-model llama3:8b
   ```

2. **Check Ollama is using GPU**:
   ```bash
   nvidia-smi
   # Look for "ollama" process using GPU
   ```

3. **Restart Ollama**:
   ```bash
   pkill ollama
   ollama serve
   ```

---

### Dubbing mode is slower than realtime

**Problem:** Wrong device or large script.

**Solution:**

1. **Use GPU mode**:
   ```bash
   python3 main.py --mode dubbing --device cuda --script "Test"
   ```

2. **Check GPU usage**:
   ```bash
   nvidia-smi
   ```

3. **Break up long scripts** (process in batches)

---

## Mode-Specific Issues

### Realtime mode: Frequent interruptions

**Problem:** Voice Activity Detection (VAD) too sensitive.

**Solution:**

1. **Adjust VAD threshold** in `config.py`:
   ```python
   VAD_THRESHOLD = 0.5  # Increase from 0.3 (less sensitive)
   ```

2. **Reduce background noise**:
   - Use headphones
   - Close windows
   - Move away from fans/AC

3. **Use a better microphone**

---

### Dubbing mode: Voice cloning not working

**Problem:** Poor reference audio or incorrect format.

**Diagnosis:**
```bash
# Check reference audio
file voices/my_voice.wav
# Should be: WAV, 16kHz+, mono or stereo
```

**Solution:**

1. **Use high-quality reference**:
   - 3-5 seconds of clear speech
   - Minimal background noise
   - Single speaker
   - 16kHz+ sample rate

2. **Convert to correct format**:
   ```bash
   ffmpeg -i input.mp3 -ar 24000 -ac 1 voices/my_voice.wav
   ```

3. **Try without voice cloning** first:
   ```bash
   python3 main.py --mode dubbing --script "Test"
   ```

---

### Creative mode: Boring/generic responses

**Problem:** LLM not generating creative content or emotion not applied.

**Solution:**

1. **Use better prompts**:
   ```bash
   python3 main.py --mode creative \
     --script "Tell a vivid, dramatic story about a space explorer discovering an ancient alien civilization" \
     --emotion "Speak with wonder and excitement"
   ```

2. **Try a different model**:
   ```bash
   ollama pull mistral
   python3 main.py --mode creative --script "Story" --llm-model mistral
   ```

3. **Adjust temperature** (code modification):
   ```python
   # In core/brain.py
   self.llm = Ollama(model=model_name, temperature=0.8)  # More creative
   ```

---

## Migration Issues

### "interactive mode deprecated" warning

**Problem:** Using old v1.x command syntax.

**Solution:**

Replace `--mode interactive` with `--mode realtime`:

```bash
# Old (v1.x)
python3 main.py --mode interactive

# New (v2.0)
python3 main.py --mode realtime
```

See [MIGRATION.md](MIGRATION.md) for full migration guide.

---

### v1.x scripts not working in v2.0

**Problem:** API changed in v2.0.

**Solution:**

Update imports and API calls:

```python
# Old (v1.x)
from core.brain import Brain
brain = Brain()
response = brain.process("Hello")

# New (v2.0)
from core.orchestrator import Orchestrator
orchestrator = Orchestrator(mode="realtime")
audio, sr = orchestrator.route_request(mode="realtime", audio_input=user_audio)
```

See [MIGRATION.md](MIGRATION.md) for detailed code migration.

---

### ".env configuration not working"

**Problem:** Using old v1.x `.env` format.

**Solution:**

1. **Backup old config**:
   ```bash
   cp .env .env.backup
   ```

2. **Copy new template**:
   ```bash
   cp .env.example .env
   ```

3. **Migrate settings**:
   ```bash
   # Old
   DEFAULT_MODE=interactive

   # New
   DEFAULT_MODE=realtime
   REALTIME_VOICE=NATF2
   REALTIME_PERSONA="You are a helpful assistant"
   ```

See `.env.example` for full v2.0 configuration options.

---

## Advanced Debugging

### Enable verbose logging

**Solution:**

```bash
# Set environment variable
export VELLORIS_DEBUG=1
python3 main.py --mode realtime

# Or in code
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

### Capture full error trace

**Solution:**

```bash
# Run with full traceback
python3 -u main.py --mode realtime 2>&1 | tee debug.log

# The debug.log file will contain full error details
```

---

### Test individual components

**Solution:**

```python
# Test PersonaPlex
from engines.personaplex import PersonaPlexEngine
engine = PersonaPlexEngine()
print(engine.check_availability())

# Test Qwen3-TTS
from engines.qwen3tts import Qwen3TTSEngine
engine = Qwen3TTSEngine()
audio, sr = engine.synthesize("Test audio")

# Test Ollama
from langchain_community.llms import Ollama
llm = Ollama(model="llama3")
response = llm.invoke("Hello")
print(response)
```

---

### Check system configuration

**Solution:**

```bash
# Velloris configuration
python3 main.py --show-config

# Python environment
python3 --version
pip list | grep torch

# GPU information
nvidia-smi

# macOS MPS
python3 -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"

# CUDA
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Ollama
curl http://localhost:11434/api/version
ollama list
```

---

### Clean reinstall

**Solution:**

```bash
# Remove virtual environment
rm -rf venv_py312

# Remove cached models (optional - will re-download)
rm -rf ~/.cache/huggingface/

# Reinstall
python3 -m venv venv_py312
source venv_py312/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Test installation
python3 main.py --show-config
```

---

## Still stuck?

If you've tried everything above and still have issues:

1. **Check GitHub Issues**: [Existing issues](https://github.com/randsley/Velloris/issues)
2. **Open a new issue**: [Report a bug](https://github.com/randsley/Velloris/issues/new)
3. **Ask the community**: [GitHub Discussions](https://github.com/randsley/Velloris/discussions)

When reporting issues, include:
- Velloris version (`git rev-parse HEAD`)
- Operating system and Python version
- GPU model (if applicable)
- Full error message and traceback
- Output of `python3 main.py --show-config`
- Steps to reproduce

---

## Quick Reference: Common Commands

```bash
# Check configuration
python3 main.py --show-config

# Test modes
python3 main.py --mode realtime --persona "Test" --voice NATF2
python3 main.py --mode dubbing --script "Test narration"
python3 main.py --mode creative --script "Test story"

# Device selection
python3 main.py --device auto    # Auto-detect (default)
python3 main.py --device cuda    # NVIDIA GPU
python3 main.py --device mps     # Apple Metal
python3 main.py --device cpu     # CPU fallback

# Debugging
export VELLORIS_DEBUG=1
python3 -u main.py --mode realtime 2>&1 | tee debug.log
```

---

For more help, see:
- [FAQ.md](FAQ.md) - Common questions
- [QUICKSTART.md](QUICKSTART.md) - Getting started
- [README.md](README.md) - Full documentation
- [GitHub Issues](https://github.com/randsley/Velloris/issues) - Report bugs
