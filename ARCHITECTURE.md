# Velloris Architecture: PersonaPlex-7B + Qwen3-TTS

## Vision Statement

**"By orchestrating the real-time interaction of PersonaPlex-7B with the expressive Voice Design of Qwen3-TTS, Velloris achieves human-level conversation without the cloud."**

Velloris is a local-first dual-engine voice agent system that combines two state-of-the-art models:
- **PersonaPlex-7B** for real-time conversational understanding and synthesis
- **Qwen3-TTS** for expressive, high-fidelity voice generation

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Velloris Application                      │
│              (core/orchestrator.py routing)                  │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
   ┌─────────────┐         ┌──────────────┐
   │ Interactive │         │   Dubbing    │
   │   Mode      │         │    Mode      │
   └──────┬──────┘         └──────┬───────┘
          │                       │
          ▼                       ▼
   ┌──────────────────┐  ┌─────────────────┐
   │PersonaPlex-7B    │  │  Qwen3-TTS      │
   │(NVIDIA)          │  │(Alibaba)        │
   │                  │  │                 │
   │Audio In → Audio  │  │Text → Audio     │
   │Out (24kHz)       │  │Out (12kHz)      │
   │                  │  │                 │
   │ • Real-time S2S  │  │ • Voice Design  │
   │ • Full-duplex    │  │ • Voice Cloning │
   │ • Interruptions  │  │ • Emotion       │
   │ • 16 Voices      │  │ • Multilingual  │
   └──────────────────┘  └─────────────────┘
```

## Mode Selection

### Interactive Mode (PersonaPlex-7B)

**Best for:** Real-time voice conversations, customer service, voice assistants

**Input:** User audio + optional persona prompt
```python
audio = np.array([...], dtype=np.float32)  # 24kHz
persona = "A helpful AI assistant with a friendly tone"

result = orchestrator.route_request(
    text=persona,
    mode="interactive",
    audio_input=audio
)
```

**Process:**
1. User audio captured at 24kHz
2. PersonaPlex-7B understands and generates response simultaneously
3. Agent audio streamed back in real-time
4. Supports full-duplex (user can interrupt)

**Output:** Agent audio (24kHz, numpy array)

### Dubbing Mode (Qwen3-TTS)

**Best for:** Content creation, video dubbing, high-fidelity narration

**Input:** Script text
```python
script = "Once upon a time..."
ref_audio = "voices/reference.wav"  # Optional voice reference

result = orchestrator.route_request(
    text=script,
    mode="dubbing",
    ref_audio_path=ref_audio
)
```

**Process:**
1. Script divided into chunks
2. Qwen3-TTS generates expressive speech for each chunk
3. Optional voice cloning from reference audio
4. Output combined for final audio

**Output:** High-fidelity audio (12kHz, numpy array)

## Engine Specifications

### PersonaPlex-7B (Interactive)

**Model:** `nvidia/personaplex-7b-v1` (Hugging Face)

**Hardware Requirements:**
- NVIDIA GPU: Ampere or newer (A100, H100, etc.)
- OS: Linux (primary), macOS with limitations
- VRAM: 16GB+ recommended

**Installation:**
```bash
# Clone repository
git clone https://github.com/NVIDIA/personaplex

# Install system dependency
brew install opus  # macOS
apt install libopus-dev  # Ubuntu

# Install Python package
pip install personaplex/moshi/.

# Accept model license and login
huggingface-cli login
```

**Sample Rate:** 24kHz (native)

**Available Voices:**
- Natural Female: NATF0, NATF1, NATF2, NATF3
- Natural Male: NATM0, NATM1, NATM2, NATM3
- Varied Female: VARF0, VARF1, VARF2, VARF3, VARF4
- Varied Male: VARM0, VARM1, VARM2, VARM3, VARM4

**Features:**
- Full-duplex conversations (simultaneous listening/speaking)
- Barge-in support (user can interrupt)
- Voice conditioning (speaker characteristics)
- Persona control (role, background, scenario)
- Streaming architecture for low latency

### Qwen3-TTS (Dubbing)

**Model:** Official `qwen-tts>=0.1.0` (PyPI)

**Available Models:**
- 1.7B-CustomVoice (recommended, voice cloning)
- 1.7B-Base (baseline)
- 1.7B-VoiceDesign (natural language voice control)
- 0.6B-CustomVoice (lightweight, voice cloning)
- 0.6B-Base (lightweight, baseline)

**Installation:**
```bash
pip install qwen-tts>=0.1.0 soundfile>=0.12.0

# Optional: FlashAttention 2 for reduced GPU memory
pip install flash-attn --no-build-isolation
```

**Sample Rate:** 12kHz (native)

**Features:**
- Voice Design via natural language prompts
- Voice cloning from reference audio
- Multilingual support (English, Mandarin, etc.)
- Emotion and style control
- Multiple model sizes for efficiency

**Specifications:**
- Dtype support: float32, float16, bfloat16 (recommended for CUDA)
- Optional FlashAttention 2 reduces memory usage
- Auto device detection (CUDA/CPU/MPS)

## Data Flow

### Interactive Conversation Flow

```
User Audio (24kHz)
       ↓
[VAD Detection] → Speech Activity Detection
       ↓
PersonaPlex-7B
  ↓
  • Processes audio with persona prompt
  • Generates understanding + response
  ↓
Agent Audio (24kHz)
       ↓
[Audio Playback] → Speaker Output
       ↓
[Repeat] ← User Interrupted?
```

### Dubbing Generation Flow

```
Script Text
       ↓
[Text Chunking] → Split into synthesis chunks
       ↓
Qwen3-TTS Voice Design
  ↓
  • Generate audio for each chunk
  • Apply voice cloning (if ref_audio provided)
  • Apply emotion/style (if instruct provided)
  ↓
[Audio Combination] → Concatenate chunks
       ↓
Final Audio Output (12kHz)
       ↓
[File Save] → Save to disk
```

## Implementation Details

### Lazy Loading Strategy

Models are loaded only when first used:

```python
orchestrator = LocalVoiceOrchestrator()

# PersonaPlex-7B loaded on first interactive mode call
result = orchestrator.route_request(text, mode="interactive")

# Qwen3-TTS loaded on first dubbing mode call
result = orchestrator.route_request(text, mode="dubbing")

# Unload to free memory
orchestrator.unload_engines()
```

### Audio Pipeline

**Input Handling:**
- Microphone: sounddevice with VAD (Silero VAD)
- File: librosa with automatic resampling
- Buffer: 2-second chunks for optimal latency

**Processing:**
- Real-time callbacks for audio I/O
- Background worker threads for non-blocking transcription
- Queue-based communication between threads

**Output:**
- Speaker: sounddevice streaming
- File: soundfile for persistence

### Configuration

All settings centralized in `config.py`:

```python
AudioConfig:
  - Input sample rate: 16000 Hz (Whisper standard)
  - Output sample rate: 24000 Hz (PersonaPlex)
  - Buffer duration: 2.0 seconds
  - Chunk size: 512 frames

ModelConfig:
  - STT: Whisper base
  - LLM: llama3 (via Ollama)
  - Interactive: PersonaPlex-7B
  - Dubbing: Qwen3-TTS 1.7B-CustomVoice

VADConfig:
  - Threshold: 0.5
  - Min speech duration: 0.3s
  - Min silence duration: 0.3s
  - Barge-in: Enabled
```

## Cloud-Free Design

Velloris operates entirely locally with no cloud dependencies:

✅ **PersonaPlex-7B**: Local inference on NVIDIA GPU
✅ **Qwen3-TTS**: Local inference on NVIDIA GPU
✅ **Audio Processing**: Local with librosa/sounddevice
✅ **Configuration**: Local YAML/Python files
✅ **Voice Data**: Stored locally

⚠️ **One-time Setup:**
- Model downloads from Hugging Face (one-time)
- License acceptance (one-time)
- No telemetry or cloud connectivity required after setup

## Performance Characteristics

### PersonaPlex-7B (Interactive)

| Metric | Value | Notes |
|--------|-------|-------|
| Latency | <200ms | Full-duplex, streaming |
| VRAM | 12-16GB | With fp16 on A100 |
| Throughput | Real-time | 1x audio speed |
| Voices | 16 | Pre-trained options |

### Qwen3-TTS (Dubbing)

| Metric | Value | Notes |
|--------|-------|-------|
| Quality | High | Professional narration |
| Speed | 0.5-2x | Depends on text length |
| VRAM | 6-12GB | Model size dependent |
| Voices | 3+ | Voice cloning capable |

## Platform Compatibility

### Device Detection

Velloris uses intelligent device detection with priority: **CUDA → MPS → CPU**

```python
from utils.device_utils import get_optimal_device, get_optimal_dtype

# Auto-detect best device
device = get_optimal_device("auto")  # Returns: "cuda", "mps", or "cpu"
dtype = get_optimal_dtype(device)    # Returns: optimal torch.dtype
```

### Windows (NVIDIA CUDA)

**Optimal Setup:**
- NVIDIA GPU: Ampere or newer (RTX 3000/4000 series, A100, H100, etc.)
- CUDA 12.1+ with cuDNN
- 16GB+ VRAM (8GB minimum with quantization)
- Python 3.11+

**Installation:**
```bash
run install_windows.bat
```

**Performance Optimizations:**
- **FlashAttention 2**: 2-3x faster inference, reduced VRAM usage
- **bitsandbytes**: 60% VRAM reduction with 4-bit quantization
- **bfloat16 dtype**: Better numerical stability, faster on modern GPUs
- **Device**: `python main.py --device cuda`

**Models Performance:**
- PersonaPlex-7B: Fast (native support)
- Qwen3-TTS: Fast

### macOS (Apple Metal/MPS)

**Optimal Setup:**
- M1/M2/M3/M4 Pro/Max (any M-series chip)
- Unified memory: 16GB+ recommended
- macOS 12.3+ (required for MPS)
- Python 3.11+

**Installation:**
```bash
chmod +x install_macos.sh
./install_macos.sh
```

**Limitations:**
- PersonaPlex-7B: No native MPS optimization, emulated (slow)
- Qwen3-TTS: Works but slower than CUDA
- FlashAttention 2: Not available (CUDA-only)
- Dtype: float32 only (float16/bfloat16 unstable on MPS)

**Device:**
```bash
# Recommended (if stable)
python main.py --device mps

# Alternative if MPS is unstable
python main.py --device cpu
```

**Optional MLX Stack:**
```bash
pip install mlx mlx-audio mlx-lm
```
This provides future support for M-series optimized models.

### Performance Comparison

| Feature | CUDA (Windows) | MPS (macOS) | CPU |
|---------|---------------|------------|-----|
| **PersonaPlex-7B** | ⚡ Fast | 🐢 Slow | 🐌 Very Slow |
| **Qwen3-TTS** | ⚡ Fast | 🐢 Moderate | 🐌 Slow |
| **FlashAttention** | ✅ Yes | ❌ No | ❌ No |
| **bfloat16** | ✅ Yes | ❌ No | ❌ No |
| **4-bit Quant** | ✅ Yes | ❌ No | ❌ No |
| **Recommended** | ⭐ Production | 💻 Development | 📚 Fallback |

### Dtype Selection

- **CUDA**: Automatically selects `bfloat16` (if supported) or `float16`
- **MPS**: Forced to `float32` (MPS has limited reduced-precision support)
- **CPU**: `float32` (standard precision)

All dtypes are selected automatically based on device. Override with:
```bash
python main.py --device cuda  # Will auto-select optimal dtype
```

## Testing

All components tested in stub mode (no models required):

```bash
# Run test suite (17 tests)
pytest tests/test_pipeline.py -v

# Interactive mode test
python main.py --mode interactive --device cpu

# Dubbing mode test
python main.py --mode dubbing --script "Test text"
```

## Future Enhancements

1. **Multi-turn conversations**: Track conversation history in PersonaPlex
2. **Real-time transcription**: Extract text from PersonaPlex-7B output
3. **Emotion control**: PersonaPlex-7B emotion guidance from Qwen3-TTS
4. **Voice blending**: Mix PersonaPlex and Qwen3-TTS outputs
5. **Mobile optimization**: ONNX exports for edge devices
6. **Web interface**: Gradio UI for easy interaction

## Related Files

- `engines/personaplex.py` - PersonaPlex-7B integration
- `engines/qwen_tts.py` - Qwen3-TTS integration
- `core/orchestrator.py` - Mode-based routing
- `utils/audio_io.py` - Audio I/O with VAD
- `config.py` - Centralized configuration
- `main.py` - CLI application

## References

- [PersonaPlex Paper](https://arxiv.org/abs/2407.04952)
- [PersonaPlex GitHub](https://github.com/NVIDIA/personaplex)
- [Qwen3-TTS Models](https://huggingface.co/collections/Qwen/qwen3-tts)
- [Qwen3-TTS GitHub](https://github.com/QwenLM/Qwen3-TTS)
