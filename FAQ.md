# ❓ Frequently Asked Questions (FAQ)

## General Questions

### What is Velloris?

Velloris is an advanced AI voice interaction system that provides three distinct modes for different use cases:
- **Realtime Mode**: Ultra-low latency (70-170ms) conversational AI using NVIDIA PersonaPlex-7B
- **Dubbing Mode**: High-fidelity narration and audio production using Qwen3-TTS
- **Creative Mode**: LLM-powered storytelling with emotion control using Ollama + Qwen3-TTS

### What's new in v2.0?

v2.0 represents a complete architectural overhaul:
- **3 distinct modes** instead of one interactive mode
- **PersonaPlex-7B used correctly** for end-to-end Speech-to-Speech (was misused in v1.x)
- **Ollama made optional** (only needed for creative mode)
- **10-15x faster latency** in realtime mode (170ms vs 2000ms+)
- **Voice cloning** support in dubbing/creative modes
- **Better documentation** and migration guides

See [MIGRATION.md](MIGRATION.md) for detailed upgrade instructions.

---

## Mode Selection

### Which mode should I use?

Choose based on your use case:

| Use Case | Recommended Mode | Why? |
|----------|------------------|------|
| Interactive chat, customer service | **Realtime** | Ultra-low latency, natural interruptions |
| Video narration, audiobooks | **Dubbing** | High-quality audio, voice cloning |
| Creative writing, storytelling | **Creative** | LLM reasoning, emotion control |
| Live tutoring, virtual assistant | **Realtime** | Full-duplex, instant responses |
| Podcast production | **Dubbing** | Professional audio quality |
| Game dialogue, characters | **Creative** | Personality and emotion |

### Can I use realtime mode without a GPU?

No. Realtime mode requires:
- **NVIDIA GPU with 16GB+ VRAM** (PersonaPlex-7B is GPU-only)
- CUDA 11.8+

If you don't have a compatible GPU:
- Use **Dubbing Mode** for narration (works on CPU, MPS, or GPU)
- Use **Creative Mode** for LLM-powered content (works on CPU with Ollama)

### Do I need Ollama for all modes?

**No!** Ollama is only required for **Creative Mode**.

- ✅ **Realtime Mode**: No Ollama (PersonaPlex has built-in reasoning)
- ✅ **Dubbing Mode**: No Ollama (just TTS)
- ⚠️ **Creative Mode**: Requires Ollama running

This is a major change from v1.x where Ollama was always required.

---

## Installation & Setup

### What are the system requirements?

**Minimum:**
- Python 3.12+
- 16GB RAM
- 50GB disk space

**Recommended for Realtime Mode:**
- NVIDIA GPU with 16GB+ VRAM (RTX 3090, RTX 4090, A100, etc.)
- CUDA 11.8+
- 32GB RAM

**Recommended for Dubbing/Creative Modes:**
- NVIDIA/AMD GPU with 6GB+ VRAM, or Apple M1/M2/M3/M4
- 16GB RAM

### Can I run Velloris on Apple Silicon (M1/M2/M3/M4)?

**Yes, but with limitations:**

✅ **Dubbing Mode**: Full support via MPS (Metal Performance Shaders)
✅ **Creative Mode**: Full support via MPS + Ollama
❌ **Realtime Mode**: Not supported (PersonaPlex requires NVIDIA GPU)

```bash
# On Apple Silicon
python3 main.py --mode dubbing --device mps --script "Your narration here"
```

### How do I install Ollama?

Only needed for Creative Mode:

```bash
# macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Or download from https://ollama.ai

# Start Ollama
ollama serve

# Download a model (in another terminal)
ollama pull llama3
```

For Windows, download the installer from https://ollama.ai.

### Installation fails with "No module named 'torch'"

Make sure you're using Python 3.12+ and have activated the virtual environment:

```bash
# Check Python version
python3 --version  # Should be 3.12+

# Activate environment
source venv_py312/bin/activate  # macOS/Linux
venv_py312\Scripts\activate     # Windows

# Reinstall requirements
pip install -r requirements.txt
```

---

## Usage Questions

### How do I change the voice?

**Realtime Mode:**
```bash
python3 main.py --mode realtime --voice NATM1  # Male voice
python3 main.py --mode realtime --voice NATF2  # Female voice
```

**Available voices:**
- Female: `NATF0`, `NATF1`, `NATF2`, `NATF3`, `VARF0-4`
- Male: `NATM0`, `NATM1`, `NATM2`, `NATM3`, `VARM0-4`

**Dubbing/Creative Modes:**
Use voice cloning with a reference audio file:
```bash
python3 main.py --mode dubbing --script "Text" --voice-ref voices/my_voice.wav
```

### How do I customize the AI personality?

**Realtime Mode:**
```bash
python3 main.py --mode realtime --persona "You are a friendly tutor who explains concepts clearly"
```

**Creative Mode:**
Use the `--script` parameter to guide the LLM:
```bash
python3 main.py --mode creative --script "Tell a story as a wise old wizard" --emotion "Speak with gravitas"
```

### Can I use Velloris programmatically?

**Yes!** Import the orchestrator directly:

```python
from core.orchestrator import Orchestrator

# Initialize
orchestrator = Orchestrator(mode="dubbing")

# Generate speech
audio, sr = orchestrator.route_request(
    mode="dubbing",
    text="Your text here"
)

# Save or process audio
import soundfile as sf
sf.write("output.wav", audio, sr)
```

See [API.md](API.md) for full Python API reference.

### How do I save the generated audio?

Audio is automatically saved to `output.wav` in the current directory (configurable in `.env`).

To specify a custom output path:
```bash
# Set in .env file
OUTPUT_AUDIO_PATH=/path/to/my/output.wav

# Or set environment variable
OUTPUT_AUDIO_PATH=/path/to/output.wav python3 main.py --mode dubbing --script "Text"
```

### Can I process batch audio files?

Yes, using Python scripting:

```python
from core.orchestrator import Orchestrator
import soundfile as sf
import os

orchestrator = Orchestrator(mode="dubbing")

scripts = [
    "First narration segment",
    "Second narration segment",
    "Third narration segment"
]

for i, text in enumerate(scripts):
    audio, sr = orchestrator.route_request(mode="dubbing", text=text)
    sf.write(f"output_{i:03d}.wav", audio, sr)
    print(f"Generated segment {i+1}/{len(scripts)}")
```

---

## Troubleshooting

### "Ollama not available" error

**Solution:**

1. Make sure Ollama is installed: https://ollama.ai
2. Start Ollama in a separate terminal:
   ```bash
   ollama serve
   ```
3. Download a model:
   ```bash
   ollama pull llama3
   ```
4. Run Velloris in creative mode:
   ```bash
   python3 main.py --mode creative --script "Test"
   ```

**Important:** Ollama is only needed for Creative Mode. Use Realtime or Dubbing modes if you don't want to use Ollama.

### "PersonaPlex engine not available" error

**Cause:** PersonaPlex-7B requires an NVIDIA GPU with CUDA.

**Solutions:**
1. Use **Dubbing Mode** instead (works on CPU/MPS):
   ```bash
   python3 main.py --mode dubbing --script "Test narration"
   ```
2. Use **Creative Mode** (works on CPU with Ollama):
   ```bash
   python3 main.py --mode creative --script "Test story"
   ```
3. If you have an NVIDIA GPU, verify CUDA installation:
   ```bash
   python3 -c "import torch; print(torch.cuda.is_available())"
   ```

### "CUDA out of memory" error

**Solutions:**

1. **Close other GPU applications** (browsers, games, etc.)
2. **Use a smaller LLM model** (creative mode only):
   ```bash
   python3 main.py --mode creative --script "Test" --llm-model llama3:8b
   ```
3. **Try CPU mode** (dubbing/creative only):
   ```bash
   python3 main.py --mode dubbing --script "Test" --device cpu
   ```
4. **Upgrade GPU** (realtime mode requires 16GB+ VRAM)

### No audio output / silent playback

**Solutions:**

1. **Check system volume** and speaker connections
2. **Verify audio device** is set correctly:
   ```bash
   python3 main.py --mode dubbing --script "Test" --list-devices
   ```
3. **Try CPU mode** to rule out GPU issues:
   ```bash
   python3 main.py --mode dubbing --script "Test" --device cpu
   ```
4. **Check output file** was generated:
   ```bash
   ls -lh output.wav
   ```

### Audio sounds robotic or distorted

**Possible causes:**

1. **Using wrong mode**: Realtime mode is optimized for low latency, not quality
   - Solution: Use **Dubbing Mode** for high-quality narration
2. **Low sample rate**: Check `.env` configuration
   - Solution: Set `OUTPUT_SAMPLE_RATE=24000` (default)
3. **CPU overload**: System struggling with real-time processing
   - Solution: Close background applications, use GPU if available

### Slow performance / high latency

**Realtime Mode:**
- Expected: 70-170ms with NVIDIA GPU
- If slower, check GPU load: `nvidia-smi`
- Close other GPU applications

**Creative Mode:**
- Expected: 1-3 seconds (includes LLM reasoning)
- If slower, try a smaller Ollama model:
  ```bash
  ollama pull llama3:8b  # Smaller than default llama3
  ```

**Dubbing Mode:**
- Expected: Faster than real-time (e.g., 10 seconds audio in 2-3 seconds)
- If slower, check GPU usage or try `--device cpu`

### "Failed to load PersonaPlex model" error

**Causes:**
- Insufficient VRAM (need 16GB+)
- CUDA not installed
- Model weights not downloaded

**Solutions:**
1. Verify CUDA:
   ```bash
   python3 -c "import torch; print(torch.cuda.is_available())"
   ```
2. Check VRAM:
   ```bash
   nvidia-smi  # Should show 16GB+ available
   ```
3. Use alternative modes (Dubbing/Creative)

---

## Migration from v1.x

### What happened to "interactive mode"?

Interactive mode has been **deprecated** and replaced with three specialized modes:

**v1.x:**
```bash
python3 main.py --mode interactive  # Old way
```

**v2.0:**
```bash
python3 main.py --mode realtime     # New equivalent (faster!)
```

The old command still works but shows a deprecation warning. See [MIGRATION.md](MIGRATION.md).

### Do I need to change my scripts?

If you're using the CLI directly, minimal changes needed:
- Replace `--mode interactive` with `--mode realtime`
- Ollama is now optional (only for creative mode)

If you're importing Velloris as a library:
- Update to use new `Orchestrator.route_request()` API
- See [MIGRATION.md](MIGRATION.md) for code migration examples

### Will my v1.x configuration work?

Mostly yes, but you should update:

1. **Copy new template:**
   ```bash
   cp .env.example .env
   ```
2. **Migrate settings:**
   - `DEFAULT_MODE` → Use `"realtime"` instead of `"interactive"`
   - Add new `REALTIME_*`, `DUBBING_*`, `CREATIVE_*` settings
3. **Review changes:**
   - Ollama settings only needed if using creative mode
   - PersonaPlex voices now configurable per-mode

See [MIGRATION.md](MIGRATION.md) for detailed migration guide.

### Why is v2.0 so much faster?

**v1.x misused PersonaPlex!**

- **Old way**: PersonaPlex → Transcription → Ollama → Qwen3-TTS (2000ms+)
- **New way**: PersonaPlex end-to-end Speech-to-Speech (70-170ms)

PersonaPlex-7B includes built-in reasoning and doesn't need an external LLM for basic conversations. This was the key insight that led to the v2.0 refactor.

---

## Advanced Usage

### Can I fine-tune the models?

**PersonaPlex-7B**: Not officially supported by NVIDIA yet. Monitor their repo for updates.

**Qwen3-TTS**: Yes, Alibaba provides fine-tuning scripts. See [Qwen3-TTS repo](https://github.com/QwenLM/Qwen3-TTS).

**Ollama models**: Yes, use Ollama's Modelfile system:
```bash
ollama create my-custom-model -f Modelfile
```

### Can I use a different LLM provider?

**Yes!** In creative mode, you can replace Ollama with any LLM:

```python
from core.brain import Brain

# Custom LLM integration
brain = Brain(mode="creative")
brain.llm = YourCustomLLMClient()  # Implement compatible interface
```

Supported out-of-the-box:
- Ollama (default)
- OpenAI API (via langchain)
- Anthropic Claude (via langchain)
- Any langchain-compatible LLM

### How do I optimize for production?

1. **Use GPU inference** for best performance
2. **Pre-load models** on startup (set in `.env`)
3. **Monitor memory** usage with `nvidia-smi` or `htop`
4. **Use appropriate mode**:
   - Realtime for conversations
   - Dubbing for pre-scripted narration
   - Creative only when LLM reasoning is truly needed
5. **Cache common responses** (implement your own caching layer)

See [PERFORMANCE.md](PERFORMANCE.md) for detailed optimization guide.

### Can I run Velloris as a service?

**Yes!** You can wrap Velloris in a web API:

```python
from flask import Flask, request, send_file
from core.orchestrator import Orchestrator
import soundfile as sf
import io

app = Flask(__name__)
orchestrator = Orchestrator(mode="dubbing")

@app.route('/synthesize', methods=['POST'])
def synthesize():
    text = request.json['text']
    audio, sr = orchestrator.route_request(mode="dubbing", text=text)

    # Return audio as WAV
    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format='WAV')
    buffer.seek(0)
    return send_file(buffer, mimetype='audio/wav')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

Consider adding:
- Authentication
- Rate limiting
- Request queuing
- Caching layer

---

## Contributing & Support

### How can I contribute?

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development setup
- Code style guidelines
- Testing requirements
- Pull request process

### Where can I get help?

- **GitHub Issues**: [Report bugs or request features](https://github.com/randsley/Velloris/issues)
- **GitHub Discussions**: [Ask questions or share ideas](https://github.com/randsley/Velloris/discussions)
- **Documentation**: [Full docs](README.md)

### How do I report a bug?

1. Check [existing issues](https://github.com/randsley/Velloris/issues)
2. Verify you're using the latest version
3. Create a new issue with:
   - Velloris version
   - Operating system and Python version
   - GPU model (if applicable)
   - Full error message
   - Steps to reproduce

### Can I use Velloris commercially?

Check the LICENSE file in the repository. Note that:
- Velloris code may have one license
- PersonaPlex-7B has its own license (NVIDIA)
- Qwen3-TTS has its own license (Alibaba)
- Ollama models have individual licenses

**Always review licenses before commercial use.**

---

## Performance & Hardware

### What GPU do I need for realtime mode?

**Minimum:**
- NVIDIA GPU with 16GB VRAM
- Examples: RTX 3090, RTX 4090, A100

**Recommended:**
- RTX 4090 (24GB VRAM) - best consumer option
- A100 (40GB/80GB VRAM) - best datacenter option

**Not supported:**
- AMD GPUs (PersonaPlex requires CUDA)
- Apple Silicon M1/M2/M3/M4 (use dubbing/creative modes instead)
- CPUs (PersonaPlex is GPU-only)

### What's the expected latency?

**Realtime Mode:**
- GPU: 70-170ms (typical: ~100ms)
- 18x faster than Gemini Live
- Full-duplex (natural interruptions)

**Dubbing Mode:**
- Faster than real-time
- 10 seconds of audio in 2-3 seconds
- Quality over latency

**Creative Mode:**
- 1-3 seconds total
- Includes LLM reasoning time
- Depends on Ollama model size

See [PERFORMANCE.md](PERFORMANCE.md) for detailed benchmarks.

### Can I reduce VRAM usage?

**Realtime Mode:** No, PersonaPlex-7B requires 16GB minimum.

**Dubbing/Creative Modes:**
- Use CPU mode: `--device cpu`
- Use smaller Ollama model: `--llm-model llama3:8b`
- Reduce batch size (in code)

### Does Velloris support multi-GPU?

Not currently. PersonaPlex-7B uses a single GPU.

Future versions may support:
- Model parallelism for larger models
- Multiple inference instances across GPUs

---

## Miscellaneous

### What languages are supported?

**Qwen3-TTS** (Dubbing/Creative modes):
- English, Chinese, Japanese, Korean
- French, Spanish, Portuguese
- German, Arabic, Italian

**PersonaPlex-7B** (Realtime mode):
- Primarily English
- Check NVIDIA's documentation for other languages

### Can I use custom voice references?

**Yes!** For dubbing and creative modes:

```bash
python3 main.py --mode dubbing --script "Your text" --voice-ref voices/custom_voice.wav
```

**Best practices:**
- 3-5 seconds of clear audio
- Mono or stereo, 16kHz+ sample rate
- Minimal background noise
- Single speaker

### Is there a GUI?

Not currently. Velloris is a command-line and Python API tool.

Community contributions for GUIs are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

### Can I integrate with Discord/Telegram/WhatsApp?

**Yes!** Velloris can be integrated with any platform that supports audio I/O.

Example integrations:
- Discord bot with voice channels
- Telegram voice message bot
- WhatsApp audio responses
- Twilio phone calls

See [EXAMPLES.md](EXAMPLES.md) for integration examples.

### What's on the roadmap?

See [ROADMAP.md](ROADMAP.md) for planned features:
- Multi-speaker support
- Voice mixing and effects
- Streaming API
- Docker container
- Cloud deployment guides
- More language support

---

## Still have questions?

Check out:
- [QUICKSTART.md](QUICKSTART.md) - Get started in 5 minutes
- [README.md](README.md) - Full documentation
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Detailed debugging guide
- [GitHub Discussions](https://github.com/randsley/Velloris/discussions) - Ask the community

Or [open an issue](https://github.com/randsley/Velloris/issues) if you've found a bug!
