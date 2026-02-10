# Velloris Examples & Tutorials

Comprehensive Python examples demonstrating how to use Velloris in your own projects.

## Quick Index

| Example | Purpose | Difficulty | Time |
|---------|---------|------------|------|
| [realtime_conversation.py](#1-realtime-conversation) | Direct PersonaPlex-7B S2S usage | Beginner | 5 min |
| [text_to_speech_simple.py](#2-text-to-speech) | Basic TTS with Qwen3 | Beginner | 5 min |
| [orchestrator_api.py](#3-orchestrator-api) | Unified API for all modes | Intermediate | 10 min |
| [batch_dubbing.py](#4-batch-dubbing) | Process multiple scripts | Intermediate | 15 min |
| [voice_cloning_workflow.py](#5-voice-cloning) | Custom voice creation | Advanced | 20 min |

---

## 1. Real-Time Conversation

**File:** `realtime_conversation.py`

Direct usage of PersonaPlex-7B for end-to-end speech-to-speech conversations.

### Features
- Real-time S2S processing (100ms input → 80ms output)
- 18 pre-trained voices
- Custom persona/role definition
- No LLM required (PersonaPlex handles everything)

### Quick Start
```bash
python examples/realtime_conversation.py
python examples/realtime_conversation.py --voice natural_female_2
python examples/realtime_conversation.py --device cuda
```

### Key Code
```python
from engines.personaplex import PersonaPlexEngine

engine = PersonaPlexEngine(
    device="cuda",
    voice="natural_female_2",
    persona="You are a helpful tutor"
)

# Process user audio → agent response
agent_audio, sr = engine.generate_s2s_response(
    audio=user_audio,
    sr=24000,
    text_prompt="You are a helpful tutor"
)
```

### Available Voices
```
Natural Female: natural_female_0, natural_female_1, natural_female_2, natural_female_3
Natural Male:   natural_male_0, natural_male_1, natural_male_2, natural_male_3
Varied Female:  varied_female_0 through varied_female_4
Varied Male:    varied_male_0 through varied_male_4
```

### Performance
- **Latency:** 80-150ms per audio chunk (RTX 3080)
- **Sample rate:** 24kHz
- **Requirements:** NVIDIA GPU (16GB+ VRAM) recommended

---

## 2. Text-to-Speech

**File:** `text_to_speech_simple.py`

Simple, high-quality speech synthesis with Qwen3-TTS.

### Features
- 10 languages supported
- Emotion/style control
- Professional quality (12kHz)
- Optional audio playback

### Quick Start
```bash
python examples/text_to_speech_simple.py
python examples/text_to_speech_simple.py --text "Hello world"
python examples/text_to_speech_simple.py --language en --emotion "enthusiastic"
python examples/text_to_speech_simple.py --play
```

### Key Code
```python
from engines.qwen_tts import Qwen3TTSEngine

engine = Qwen3TTSEngine(device="auto")

# Synthesize speech
audio, sr = engine.generate_dubbing(
    text="Hello, world!",
    language="en",
    instruct="Speak with enthusiasm"
)
```

### Supported Languages
- **Code:** zh, en, ja, ko, de, fr, ru, pt, es, it
- **Names:** Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian

---

## 3. Orchestrator API

**File:** `orchestrator_api.py`

Unified interface for all three Velloris modes with automatic routing.

### Features
- Single API for all modes
- Automatic device detection
- Lazy loading (load models only when needed)
- Mode-based request routing

### Quick Start
```bash
python examples/orchestrator_api.py --mode realtime
python examples/orchestrator_api.py --mode dubbing
python examples/orchestrator_api.py --mode creative
python examples/orchestrator_api.py --mode all
```

### Key Code
```python
from core.orchestrator import LocalVoiceOrchestrator

orchestrator = LocalVoiceOrchestrator(device="auto")

# Real-time mode
result = orchestrator.route_request(
    mode="realtime",
    audio_input=audio,
    voice_prompt="natural_female_2",
    text_prompt="You are helpful"
)

# Dubbing mode
result = orchestrator.route_request(
    mode="dubbing",
    text="Your narration script",
    language="en"
)

# Creative mode (requires Ollama)
result = orchestrator.route_request(
    mode="creative",
    text="Tell me a story",
    emotion="Speak with wonder"
)
```

### Mode Comparison
| Feature | Realtime | Dubbing | Creative |
|---------|----------|---------|----------|
| **Latency** | 80-150ms ⚡ | N/A | 1-3s |
| **Requires GPU** | Yes | Optional | Optional |
| **Requires Ollama** | No | No | Yes |
| **Use Case** | Conversations | Narration | Creative content |

---

## 4. Batch Dubbing

**File:** `batch_dubbing.py`

Process multiple scripts efficiently for content creation workflows.

### Features
- Process batches of scripts
- Per-script emotion control
- Progress tracking
- Optional audio file saving

### Quick Start
```bash
python examples/batch_dubbing.py
python examples/batch_dubbing.py --language en --emotion "professional"
python examples/batch_dubbing.py --save-audio
```

### Key Code
```python
from engines.qwen_tts import Qwen3TTSEngine

engine = Qwen3TTSEngine(device="auto")

scripts = [
    {"title": "Intro", "text": "Welcome to...", "emotion": "enthusiastic"},
    {"title": "Content", "text": "Today we...", "emotion": "informative"},
    {"title": "Outro", "text": "Thanks for...", "emotion": "warm"},
]

for script in scripts:
    audio, sr = engine.generate_dubbing(
        text=script["text"],
        language="en",
        instruct=script["emotion"]
    )
    # Process or save audio
```

### Use Cases
- Audiobook narration
- Podcast episodes
- Video voice-overs
- Tutorial scripts

---

## 5. Voice Cloning

**File:** `voice_cloning_workflow.py`

Create custom narration voices from reference audio samples.

### Features
- Voice cloning from 3-5 second samples
- Custom voice personality preservation
- Batch synthesis with cloned voice
- Audio validation

### Quick Start
```bash
# With an existing voice sample
python examples/voice_cloning_workflow.py --voice-ref path/to/sample.wav

# Create demo voice (for testing)
python examples/voice_cloning_workflow.py --demo

# Customize script
python examples/voice_cloning_workflow.py --voice-ref sample.wav --script "Your narration"
```

### Key Code
```python
from engines.qwen_tts import Qwen3TTSEngine

engine = Qwen3TTSEngine(device="auto")

# Generate speech with cloned voice
audio, sr = engine.generate_dubbing(
    text="Narration in custom voice",
    ref_audio_path="path/to/voice_sample.wav",
    language="en"
)
```

### Voice Sample Requirements
| Requirement | Details |
|-------------|---------|
| **Duration** | 3-5 seconds optimal (1-10s acceptable) |
| **Content** | Clear speech, minimal background noise |
| **Sample Rate** | 16kHz or higher recommended |
| **Format** | WAV, MP3, or other common formats |
| **Quality** | Low background noise for best results |

### Tips for Best Results
1. **Record high-quality audio:** Use a good microphone in a quiet room
2. **Clear speech:** Speak naturally with proper articulation
3. **Consistent tone:** Maintain the voice/emotion you want to clone
4. **Multiple takes:** Try different voice samples to find the best clone
5. **Standard length:** 3-5 seconds is the sweet spot

---

## Running Examples

### Prerequisites
```bash
# Install required dependencies
pip install -r requirements.txt

# For voice cloning audio save:
pip install soundfile

# For audio playback:
pip install sounddevice
```

### Basic Workflow
```bash
# 1. Check which examples are available
ls examples/

# 2. Run an example
python examples/text_to_speech_simple.py

# 3. Check output
ls *.wav  # Audio files generated

# 4. Explore the code
cat examples/text_to_speech_simple.py
```

### Using in Your Code
```python
import sys
from pathlib import Path

# Add Velloris to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import engines
from engines.personaplex import PersonaPlexEngine
from engines.qwen_tts import Qwen3TTSEngine
from core.orchestrator import LocalVoiceOrchestrator

# Use in your project
orchestrator = LocalVoiceOrchestrator()
result = orchestrator.route_request(mode="dubbing", text="Hello")
```

---

## Common Patterns

### Pattern 1: Simple TTS
```python
from engines.qwen_tts import Qwen3TTSEngine

engine = Qwen3TTSEngine()
audio, sr = engine.generate_dubbing(text="Hello world")
```

### Pattern 2: Real-Time Conversation
```python
from engines.personaplex import PersonaPlexEngine

engine = PersonaPlexEngine(persona="You are helpful")
agent_audio, sr = engine.generate_s2s_response(user_audio)
```

### Pattern 3: Batch Processing
```python
from engines.qwen_tts import Qwen3TTSEngine

engine = Qwen3TTSEngine()
for script in scripts:
    audio, sr = engine.generate_dubbing(text=script)
    # Process audio
```

### Pattern 4: Mode Routing
```python
from core.orchestrator import LocalVoiceOrchestrator

orchestrator = LocalVoiceOrchestrator()
if realtime:
    result = orchestrator.route_request(mode="realtime", audio_input=audio)
else:
    result = orchestrator.route_request(mode="dubbing", text=text)
```

---

## Troubleshooting

### Model Loading Fails
```
[X] Failed to load PersonaPlex-7B model
```
**Solution:** Ensure models are downloaded
```bash
python download_models.py
```

### CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution:** Try CPU mode or use a smaller model
```bash
python examples/text_to_speech_simple.py --device cpu
```

### Ollama Not Available
```
[X] Creative mode failed (check if Ollama is running)
```
**Solution:** Start Ollama in another terminal
```bash
ollama serve
```

### Audio Not Playing
**Solution:** Install audio dependencies
```bash
pip install sounddevice
```

---

## Performance Tips

### Optimize for Speed
```python
# Use GPU for faster inference
engine = Qwen3TTSEngine(device="cuda")

# Batch similar languages together
# Process shorter texts first
```

### Optimize for Quality
```python
# Use longer voice samples for cloning
# Minimize background noise in reference audio
# Use full language codes (en, not eng)
```

### Optimize for Memory
```python
# Unload engines after use
orchestrator.unload_engines()

# Use lazy loading (automatic with orchestrator)
```

---

## Next Steps

After running these examples:

1. **Read the code:** Each example is well-commented
2. **Modify parameters:** Try different voices, languages, emotions
3. **Combine examples:** Mix modes for your use case
4. **Integrate:** Use as template for your own projects
5. **Check docs:** See [ARCHITECTURE.md](../ARCHITECTURE.md) for details

---

## Contributing

Found an issue or have an improvement?
- Check [GitHub Issues](https://github.com/randsley/Velloris/issues)
- Submit a pull request with your example
- Share feedback on [Discussions](https://github.com/randsley/Velloris/discussions)

---

**Happy coding! 🎙️**
