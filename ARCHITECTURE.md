# Velloris Architecture: Three-Mode Voice AI System

## Vision Statement

**"By orchestrating the real-time interaction of PersonaPlex-7B with the expressive Voice Design of Qwen3-TTS, Velloris achieves human-level conversation without the cloud."**

Velloris is a local-first three-mode voice agent system with platform-specific engine selection:
- **Realtime Mode** (VERIFIED WORKING): PersonaPlex-7B S2S on CUDA, MacEcho on macOS
- **Dubbing Mode** (Production): Qwen3-TTS on CUDA/CPU, MLX-Audio on macOS
- **Creative Mode** (Production): Ollama LLM + TTS (platform-specific)

## Architecture Overview (v2.0)

```
+---------------------------------------------------------------+
|                    Velloris Application                        |
|              (core/orchestrator.py mode-based routing)         |
+----------------------------+----------------------------------+
                             |
        +--------------------+--------------------+
        |                    |                    |
        v                    v                    v
   +---------+          +---------+          +---------+
   |Realtime |          | Dubbing |          |Creative |
   |  Mode   |          |  Mode   |          |  Mode   |
   |  Prod   |          |  Prod   |          |  Prod   |
   +----+----+          +----+----+          +----+----+
        |                    |                    |
        v                    v                    v
+------------------+  +--------------+  +-----------------+
|S2S Engine        |  |TTS Engine    |  |Ollama + TTS     |
|(platform-select) |  |(platform-sel)|  |(platform-sel)   |
|                  |  |              |  |                  |
|macOS: MacEcho    |  |macOS:MLX-TTS |  |Text->LLM->Audio |
| (MLX, 16kHz)     |  |Other:Qwen3   |  | (24kHz)         |
|Other: PersonaPlex|  | -TTS (24kHz) |  |                  |
| (CUDA, 24kHz)    |  |              |  |*LLM Reasoning    |
|                  |  |*Voice Design |  |*Emotion Control  |
|*Full-duplex      |  |*10 languages |  |*Creative Output  |
|*18 voices (CUDA) |  |*Voice Clone  |  |*Requires Ollama  |
|*No LLM needed    |  |              |  |                  |
+------------------+  +--------------+  +-----------------+

Platform-Conditional Engine Selection (orchestrator.py):
  macOS:  MacEchoEngine (S2S) + MLXTTSEngine (TTS)
  Other:  PersonaPlexEngine (S2S) + Qwen3TTSEngine (TTS)
```

## Three-Mode Architecture

### Mode Comparison Table

| Feature | Realtime | Dubbing | Creative |
|---------|----------|---------|----------|
| **Status** | **VERIFIED WORKING** | **Production** | **Production** |
| **Engine (CUDA)** | PersonaPlex-7B | Qwen3-TTS | Ollama + Qwen3-TTS |
| **Engine (macOS)** | MacEcho (MLX) | MLX-Audio TTS | Ollama + MLX-Audio |
| **Latency** | 80-150ms (CUDA) | N/A | 1-3s |
| **Full-Duplex** | Infrastructure ready | No | No |
| **Interruption** | VAD + callbacks ready | No | No |
| **Languages** | English + accents | **10 languages** | **10 languages** |
| **Voice Options** | 18 preset voices | **Unlimited** | **Unlimited** |
| **Emotion Control** | Built-in | **Yes** | **Yes** |
| **Voice Cloning** | No | **Yes** | **Yes** |
| **Ollama Required** | **No** | **No** | Yes |
| **GPU Required** | NVIDIA (16GB+) | Optional | Optional |
| **VRAM Usage** | 16GB+ | 4-6GB | 4-6GB (TTS) |
| **Sample Rate** | 24kHz | 24kHz | 24kHz |
| **Best For** | Conversations | Narration | Creative content |

### Performance Benchmarks

| Mode | Status | Latency | User Feedback | Test Coverage |
|------|--------|---------|---------------|---------------|
| **Realtime** | VERIFIED WORKING | 80-150ms (RTX 3080) | Full S2S working | 99 tests (93 passing, 6 skipped) |
| **Creative** | Production | 1-3s | "Perfect audio" | User-verified |
| **Dubbing** | Production | N/A | High-quality | User-verified |

#### Realtime Mode Performance (PersonaPlex-7B on CUDA)

Verified on Windows/CUDA (RTX 3080, Feb 2026):

| Metric | Result | Notes |
|--------|--------|-------|
| Latency | 80-150ms | 100ms input -> 80ms output |
| Naturalness | 3.90/5.0 MOS | PersonaPlex evaluation |
| Full-Duplex | Infrastructure ready | Streaming architecture |
| Barge-in | VAD ready | Silero-based interruption |
| Voices | 18 | 4 natural F, 4 natural M, 5 varied F, 5 varied M |

#### Production Quality (Creative & Dubbing)

| Mode | Quality | Notes |
|------|---------|-------|
| **Creative** | High | Ollama LLM + platform TTS, emotion pass-through |
| **Dubbing** | High | Voice cloning, emotion control, 10 languages |

---

## Mode Selection Guide

### Realtime Mode (PersonaPlex-7B / MacEcho)

**Status:** VERIFIED WORKING | **Platforms:** Windows/Linux CUDA, macOS MPS

**Best for:** Interactive voice conversations, customer service, live tutoring, gaming NPCs

#### Usage

```bash
# Basic conversation
python main.py --mode realtime

# Custom persona and voice
python main.py --mode realtime --persona "You are a helpful tutor" --voice natural_female_2

# List available voices
python main.py --show-config
```

#### Pipeline (CUDA - PersonaPlex-7B)

```python
# Full S2S inference - no separate STT/LLM/TTS needed
result = orchestrator.route_request(
    mode="realtime",
    audio_input=audio_24khz,
    voice_prompt="NATF2",
    text_prompt="You are a helpful assistant"
)
# Returns: (agent_audio, 24000)
```

**Process:**
1. User audio captured at 24kHz
2. VAD detects speech (Silero)
3. PersonaPlex-7B processes audio end-to-end (encode -> LM -> decode)
4. Agent audio returned at 24kHz
5. Interruption/barge-in handling via VAD

#### Pipeline (macOS - MacEcho)

```
User Audio (16kHz) -> SenseVoice ASR -> Qwen LLM (MLX) -> CosyVoice TTS -> Agent Audio (16kHz)
```

**Output:** Agent audio (24kHz numpy array, resampled from engine native rate)

### Dubbing Mode (Qwen3-TTS / MLX-Audio)

**Status:** Production | **Best for:** Content creation, video dubbing, high-fidelity narration

```bash
# Simple narration
python main.py --mode dubbing --script "Once upon a time..."

# With voice cloning
python main.py --mode dubbing --script "Your script" --voice-ref my_voice.wav
```

```python
result = orchestrator.route_request(
    mode="dubbing",
    text="Once upon a time...",
    ref_audio_path="voices/reference.wav"
)
```

**Process:**
1. Script text received
2. TTS engine generates expressive speech
3. Optional voice cloning from reference audio (3-5 second sample)
4. Optional emotion/style via natural language instruction

**Output:** High-fidelity audio (24kHz numpy array)

### Creative Mode (Ollama + TTS)

**Status:** Production | **Best for:** Storytelling, emotional content, interactive prompts

**Requires:** Ollama running (`ollama serve`)

```bash
# Interactive emotional synthesis
python main.py --mode creative --emotion "Speak with excitement"

# Different LLM model
python main.py --mode creative --llm-model mistral --emotion "Excited tone"
```

**Process:**
1. User text -> Ollama LLM (reasoning/creativity)
2. LLM response -> TTS engine (emotional synthesis)
3. Emotion instruction passed through to TTS

**Output:** Emotionally expressive audio (24kHz numpy array)

---

## Engine Specifications

### PersonaPlex-7B (Realtime Mode - CUDA/Linux/Windows)

**Status:** VERIFIED WORKING | **File:** `engines/personaplex.py`

**Model:** `nvidia/personaplex-7b-v1` (Hugging Face)

**Architecture Components:**
1. **Mimi Audio Codec** - Encodes audio frames to discrete codes, decodes tokens to waveforms (24kHz, 1920 sample frames)
2. **Moshi LM** - Language model for code processing with streaming inference
3. **LMGen Manager** - Frame-by-frame processing with voice conditioning
4. **SentencePiece Tokenizer** - Text prompt tokenization with system tag wrapping

**Pipeline:**
```
User Audio (24kHz)
    |
[Mimi.encode()] -> Audio Codes
    |
[LMGen.step()] -> Output Tokens (frame-by-frame)
    |
[Mimi.decode()] -> Agent Audio
    |
[Normalize] -> Final Output (24kHz)
```

**Hardware Requirements:**
- NVIDIA GPU: Ampere or newer (RTX 3000/4000, A100, H100)
- OS: Windows or Linux (CUDA 12.1+ required)
- VRAM: 16GB+ recommended
- Triton (triton-windows on Windows) for torch.compile() optimization

**Installation:**
```bash
# 1. Clone PersonaPlex repository
git clone https://github.com/NVIDIA/personaplex

# 2. Install system dependency
# Ubuntu/Debian:
sudo apt install libopus-dev

# 3. Install Python package
pip install personaplex/moshi/.

# 4. Accept model license and login
huggingface-cli login

# 5. Run realtime mode
python3 main.py --mode realtime --persona "helpful assistant" --voice natural_female_2
```

**Sample Rate:** 24kHz (native)

**Available Voices (18 total):**
- Natural Female: NATF0, NATF1, NATF2, NATF3
- Natural Male: NATM0, NATM1, NATM2, NATM3
- Varied Female: VARF0, VARF1, VARF2, VARF3, VARF4
- Varied Male: VARM0, VARM1, VARM2, VARM3, VARM4

**Features:**
- Full-duplex conversations (simultaneous listening/speaking)
- Barge-in support (user can interrupt via VAD)
- Voice conditioning (speaker characteristics)
- Persona control (role, background, scenario)
- Streaming architecture for low latency (80-150ms verified)

### MacEcho (Realtime Mode - macOS)

**Status:** Production | **File:** `engines/macecho_s2s.py`

**Architecture:** Silero VAD -> SenseVoice ASR -> Qwen LLM (MLX) -> CosyVoice TTS

**Platform:** macOS with Apple Silicon (M1/M2/M3/M4)

**Sample Rate:** 16kHz native (resampled to 24kHz for output consistency)

**Features:**
- Apple Silicon optimized via MLX framework
- 50+ language support (SenseVoice)
- Voice quality at 16kHz native
- Emotion/prosody control via LLM prompting

### Qwen3-TTS (Dubbing & Creative - CUDA/Linux/Windows)

**Status:** Production | **File:** `engines/qwen_tts.py`

**Package:** `qwen-tts` (PyPI) - Official Alibaba implementation

**Model:** Qwen3-TTS-12Hz-1.7B-CustomVoice (recommended)

**Available Models:**
- 1.7B-CustomVoice (voice cloning, recommended)
- 1.7B-Base (baseline)
- 1.7B-VoiceDesign (voice design via natural language)
- 0.6B-CustomVoice (lightweight with voice cloning)
- 0.6B-Base (lightweight baseline)

**Sample Rate:** 24kHz output

**Features:**
- Voice cloning from reference audio (3-5 second samples)
- Voice design via natural language descriptions
- Multilingual support (10 languages: Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian)
- Emotion and style control
- Three generation modes: `generate_custom_voice()`, `generate_voice_design()`, `generate_voice_clone()`

**Dtype Selection:**
- CUDA: bfloat16 (recommended) or float16
- CPU: float32

### MLX-Audio TTS (Dubbing & Creative - macOS)

**Status:** Production | **File:** `engines/mlx_tts.py`

**Package:** `mlx-audio` - Native MLX framework implementation

**Model:** Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit (via mlx-audio)

**Sample Rate:** 12kHz native, resampled to 24kHz for output

**Features:**
- Apple Silicon optimized (M-series chips)
- Voice design via natural language prompts
- Voice cloning from reference audio
- Multilingual support (10 languages)
- Emotion and style control
- CLI subprocess for guaranteed quality

**Installation (macOS only):**
```bash
pip install mlx-audio soundfile
```

---

## Data Flow

### Real-Time Conversation Flow (PersonaPlex on CUDA)

```
User Audio (24kHz)
       |
[VAD Detection] -> Speech Activity Detection
       |
[Mimi Encode] -> Audio Codes
       |
[Moshi LM + LMGen] -> Output Tokens
       |
[Mimi Decode] -> Agent Audio
       |
Agent Audio (24kHz)
       |
[Audio Playback] -> Speaker Output
       |
[Repeat] <- User Interrupted? (VAD barge-in)
```

### Dubbing Generation Flow

```
Script Text
       |
[TTS Engine] (Qwen3-TTS or MLX-Audio)
  |
  * Generate audio
  * Apply voice cloning (if ref_audio provided)
  * Apply emotion/style (if instruct provided)
  |
Final Audio Output (24kHz)
       |
[File Save] -> output.wav
[Playback] -> Speaker Output
```

### Creative Mode Flow

```
User Text
       |
[Ollama LLM] -> Creative/Emotional Response Text
       |
[TTS Engine] (Qwen3-TTS or MLX-Audio)
  |
  * Synthesize with emotion control
  |
Final Audio Output (24kHz)
       |
[Playback] -> Speaker Output
```

## Implementation Details

### Platform-Conditional Engine Selection

The orchestrator automatically selects engines based on platform:

```python
# core/orchestrator.py
if sys.platform == "darwin":
    from engines.macecho_s2s import MacEchoEngine as S2SEngine
    from engines.mlx_tts import MLXTTSEngine as TTSEngine
else:
    from engines.personaplex import PersonaPlexEngine as S2SEngine
    from engines.qwen_tts import Qwen3TTSEngine as TTSEngine
```

### Lazy Loading Strategy

Models are loaded only when first used:

```python
orchestrator = LocalVoiceOrchestrator()

# S2S engine loaded on first realtime mode call
result = orchestrator.route_request(mode="realtime", audio_input=audio)

# TTS engine loaded on first dubbing mode call
result = orchestrator.route_request(mode="dubbing", text=script)

# Ollama loaded on first creative mode call
result = orchestrator.route_request(mode="creative", text=prompt, emotion="excited")

# Unload to free memory
orchestrator.unload_engines()
```

### Audio Pipeline

**Input Handling:**
- Microphone: sounddevice with VAD (Silero VAD)
- File: librosa with automatic resampling
- Buffer: 2-second chunks for optimal latency
- sounddevice callbacks are NOT async: queue-based thread communication

**Processing:**
- Real-time callbacks for audio I/O (regular functions, not async)
- Background worker threads for non-blocking transcription
- `queue.Queue()` for thread-safe communication between threads

**Output:**
- Speaker: sounddevice streaming, sox fallback for WSL2
- File: soundfile for WAV persistence

### WSL2 Support

Velloris supports WSL2 (Windows Subsystem for Linux) with:

- **Audio routing**: PulseAudio/WSLg pipes audio to Windows speakers
- **Ollama auto-detection**: Automatically finds Ollama running on Windows host via gateway IP
- **CUDA support**: NVIDIA GPU passthrough for CUDA acceleration
- **Configuration**: `~/.asoundrc` routes ALSA through PulseAudio

```bash
# WSL2 audio setup
sudo apt-get install -y pulseaudio-utils libasound2-plugins
echo -e "pcm.default pulse\nctl.default pulse" > ~/.asoundrc

# Ollama on Windows auto-detected - no manual config needed
# Set OLLAMA_HOST=0.0.0.0 on Windows side before running ollama serve
```

### Configuration

All settings centralized in `config.py`:

```python
AudioConfig:
  - Input sample rate: 16000 Hz (Whisper standard)
  - Output sample rate: 24000 Hz
  - Buffer duration: 2.0 seconds
  - Chunk size: 512 frames

ModelConfig:
  - STT: Whisper base
  - LLM: llama3 (via Ollama, auto-detects WSL2 host)
  - Real-Time: PersonaPlex-7B (CUDA) / MacEcho (macOS)
  - Dubbing: Qwen3-TTS 1.7B-CustomVoice (CUDA) / MLX-Audio (macOS)

VADConfig:
  - Threshold: 0.5
  - Min speech duration: 0.3s
  - Min silence duration: 0.3s
  - Barge-in: Enabled
```

## Cloud-Free Design

Velloris operates entirely locally with no cloud dependencies:

- **PersonaPlex-7B**: Local inference on NVIDIA GPU
- **Qwen3-TTS**: Local inference on GPU or CPU
- **MLX-Audio**: Local inference on Apple Silicon
- **MacEcho**: Local inference on Apple Silicon
- **Ollama**: Local LLM server
- **Audio Processing**: Local with sounddevice/sox/soundfile
- **Configuration**: Local Python files

**One-time Setup:**
- Model downloads from Hugging Face (one-time)
- License acceptance (one-time)
- No telemetry or cloud connectivity required after setup

## Platform Compatibility

### Device Detection

Velloris uses intelligent device detection with priority: **CUDA -> MPS -> CPU**

```python
from utils.device_utils import get_optimal_device, get_optimal_dtype

# Auto-detect best device
device = get_optimal_device("auto")  # Returns: "cuda", "mps", or "cpu"
dtype = get_optimal_dtype(device)    # Returns: optimal torch.dtype
```

### Windows / Linux (NVIDIA CUDA)

**Optimal Setup:**
- NVIDIA GPU: Ampere or newer (RTX 3000/4000 series, A100, H100)
- CUDA 12.1+ with cuDNN
- 16GB+ VRAM (8GB minimum with quantization)
- Python 3.12+

**Installation:**
```bash
# Windows
install_windows.bat

# Linux / WSL2
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-dev.txt
```

**Performance Optimizations:**
- **FlashAttention 2**: 2-3x faster inference, reduced VRAM usage
- **bitsandbytes**: 60% VRAM reduction with 4-bit quantization
- **bfloat16 dtype**: Better numerical stability, faster on modern GPUs

**Engines Used:**
- S2S: PersonaPlex-7B (full S2S, no LLM needed)
- TTS: Qwen3-TTS (high-fidelity, 10 languages)

### macOS (Apple Metal/MPS)

**Optimal Setup:**
- M1/M2/M3/M4 Pro/Max (any M-series chip)
- Unified memory: 16GB+ recommended
- macOS 12.3+ (required for MPS)
- Python 3.12+

**Installation:**
```bash
chmod +x install_macos.sh
./install_macos.sh
```

**Engines Used:**
- S2S: MacEcho (MLX-optimized, SenseVoice + Qwen + CosyVoice)
- TTS: MLX-Audio (native MLX, 0.6B-8bit model)

**Limitations:**
- FlashAttention 2: Not available (CUDA-only)
- Dtype: float32 only (float16/bfloat16 unstable on MPS)

### Linux / WSL2 (CPU/CUDA)

**CUDA Mode:** Same as Windows CUDA setup
**CPU Mode:** Works on any Linux (slower)

**WSL2 Specifics:**
- Audio: PulseAudio/WSLg routing to Windows
- Ollama: Auto-detected on Windows host
- CUDA: GPU passthrough supported
- System deps: `portaudio19-dev ffmpeg sox libasound2-plugins pulseaudio-utils`

### Performance Comparison

| Feature | CUDA (Windows/Linux) | MPS (macOS) | CPU |
|---------|---------------------|------------|-----|
| **PersonaPlex-7B** | Fast (80-150ms) | N/A (uses MacEcho) | Slow |
| **MacEcho** | N/A (uses PersonaPlex) | Fast (MLX) | N/A |
| **Qwen3-TTS** | Fast | N/A (uses MLX-Audio) | Slow |
| **MLX-Audio** | N/A (uses Qwen3-TTS) | Fast (native) | N/A |
| **FlashAttention** | Yes | No | No |
| **bfloat16** | Yes | No | No |
| **4-bit Quant** | Yes | No | No |
| **Recommended** | Production | Production | Fallback |

### Dtype Selection

- **CUDA**: Automatically selects `bfloat16` (if supported) or `float16`
- **MPS**: Forced to `float32` (MPS has limited reduced-precision support)
- **CPU**: `float32` (standard precision)

All dtypes are selected automatically based on device.

## Testing

All components tested in stub mode (no models required):

```bash
# Run all tests (99 total: 93 passing, 6 skipped)
pytest tests/ -v

# Integration tests (22 tests)
pytest tests/test_pipeline.py -v

# Critical path & platform tests (37 tests)
pytest tests/test_critical_paths.py -v

# Realtime audio callbacks (12 tests)
pytest tests/test_realtime_callbacks.py -v

# End-to-end realtime tests (15 tests)
pytest tests/test_realtime_e2e.py -v

# VAD & interruption tests (13 tests)
pytest tests/test_vad_interruption.py -v

# With coverage
pytest tests/ --cov=. -v
```

**Note:** 6 skipped tests are platform-specific (MLX/MacEcho macOS-only on Linux, PersonaPlex CUDA-only on macOS).

**Mode Testing:**
```bash
# Realtime mode (requires CUDA + PersonaPlex models)
python main.py --mode realtime --device cuda

# Dubbing mode (requires Qwen3-TTS or MLX-Audio)
python main.py --mode dubbing --script "Test text"

# Creative mode (requires Ollama running + TTS engine)
python main.py --mode creative --emotion "Speak with excitement"
```

## Future Enhancements

1. **Multi-turn conversations**: Track conversation history in PersonaPlex
2. **Real-time transcription**: Extract text from PersonaPlex-7B output
3. **Web interface**: Gradio UI for easy interaction
4. **ONNX export**: Edge deployment optimization
5. **Mobile optimization**: iOS/Android support
6. **Custom voice fine-tuning**: User-specific voice models

## Related Files

- `engines/personaplex.py` - PersonaPlex-7B S2S engine (CUDA)
- `engines/macecho_s2s.py` - MacEcho S2S engine (macOS)
- `engines/qwen_tts.py` - Qwen3-TTS engine (CUDA/CPU)
- `engines/mlx_tts.py` - MLX-Audio TTS engine (macOS)
- `core/orchestrator.py` - Platform-conditional mode routing
- `core/brain.py` - Ollama LLM integration with WSL2 auto-detection
- `utils/audio_io.py` - Audio I/O with VAD and sox fallback
- `utils/device_utils.py` - Cross-platform device detection
- `utils/vad_handler.py` - Voice Activity Detection and interruption
- `config.py` - Centralized configuration with WSL2 Ollama auto-detect
- `main.py` - CLI application (3 modes + voice converter)

## References

- [PersonaPlex Paper](https://arxiv.org/abs/2407.04952)
- [PersonaPlex GitHub](https://github.com/NVIDIA/personaplex)
- [Qwen3-TTS Models](https://huggingface.co/collections/Qwen/qwen3-tts)
- [Qwen3-TTS GitHub](https://github.com/QwenLM/Qwen3-TTS)
- [MLX-Audio GitHub](https://github.com/Blaizzy/mlx-audio)
- [MacEcho GitHub](https://github.com/realtime-ai/mac-echo)
