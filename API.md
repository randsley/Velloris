# 📡 Velloris Python API Reference

Complete Python API documentation for Velloris v2.0.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Classes](#core-classes)
- [Engines](#engines)
- [Configuration](#configuration)
- [Utilities](#utilities)
- [Examples](#examples)
- [Error Handling](#error-handling)

---

## Installation

```bash
# Clone repository
git clone https://github.com/randsley/Velloris.git
cd Velloris

# Install as package (development mode)
pip install -e .

# Or just add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/Velloris"
```

---

## Quick Start

```python
from core.orchestrator import Orchestrator
import soundfile as sf

# Initialize for dubbing mode
orchestrator = Orchestrator(mode="dubbing")

# Generate speech
audio, sample_rate = orchestrator.route_request(
    mode="dubbing",
    text="Hello from Velloris!"
)

# Save to file
sf.write("output.wav", audio, sample_rate)
```

---

## Core Classes

### Orchestrator

**Main entry point for all Velloris operations.**

```python
from core.orchestrator import Orchestrator
```

#### `__init__(mode: str = "realtime", device: str = "auto")`

Initialize the orchestrator.

**Parameters:**
- `mode` (str): Operating mode - `"realtime"`, `"dubbing"`, or `"creative"`
- `device` (str): Device to use - `"auto"`, `"cuda"`, `"mps"`, or `"cpu"`

**Returns:** Orchestrator instance

**Example:**
```python
# Auto-detect device
orchestrator = Orchestrator(mode="dubbing")

# Force CPU
orchestrator = Orchestrator(mode="dubbing", device="cpu")

# Force CUDA GPU
orchestrator = Orchestrator(mode="realtime", device="cuda")
```

---

#### `route_request(...) -> Optional[Tuple[np.ndarray, int]]`

Process a request and generate audio.

**Parameters:**
- `mode` (str): Operating mode - `"realtime"`, `"dubbing"`, or `"creative"`
- `text` (Optional[str]): Input text (dubbing/creative modes)
- `audio_input` (Optional[np.ndarray]): Input audio (realtime mode)
- `ref_audio_path` (Optional[str]): Reference audio for voice cloning
- `voice_prompt` (Optional[str]): Voice/persona description (realtime mode)
- `text_prompt` (Optional[str]): Text instructions (realtime mode)
- `emotion` (Optional[str]): Emotion description (creative mode)
- `**kwargs`: Additional mode-specific parameters

**Returns:**
- `Tuple[np.ndarray, int]`: (audio_array, sample_rate) or None on failure

**Example:**
```python
# Dubbing mode
audio, sr = orchestrator.route_request(
    mode="dubbing",
    text="Welcome to Velloris"
)

# Creative mode with emotion
audio, sr = orchestrator.route_request(
    mode="creative",
    text="Tell a scary story",
    emotion="Speak with suspense and fear"
)

# Realtime mode (requires audio input)
import numpy as np
user_audio = np.random.randn(16000)  # 1 second at 16kHz
audio, sr = orchestrator.route_request(
    mode="realtime",
    audio_input=user_audio,
    voice_prompt="You are a helpful assistant"
)
```

---

### Brain

**LLM reasoning component (creative mode only).**

```python
from core.brain import Brain
```

#### `__init__(mode: str = "creative", model_name: str = "llama3", ...)`

Initialize the brain with LLM capabilities.

**Parameters:**
- `mode` (str): Operating mode - only `"creative"` requires LLM
- `model_name` (str): Ollama model name (e.g., `"llama3"`, `"mistral"`)
- `tts_engine` (Optional): TTS engine instance (injected by orchestrator)
- `orchestrator` (Optional): Orchestrator instance (injected)

**Returns:** Brain instance

**Example:**
```python
# Basic initialization
brain = Brain(mode="creative", model_name="llama3")

# With custom TTS engine
from engines.qwen3tts import Qwen3TTSEngine
tts = Qwen3TTSEngine()
brain = Brain(mode="creative", tts_engine=tts)
```

---

#### `process(user_input: str, emotion: str = "") -> Tuple[np.ndarray, int]`

Process user input with LLM reasoning and generate speech.

**Parameters:**
- `user_input` (str): Text prompt for LLM
- `emotion` (str): Emotion description for TTS

**Returns:**
- `Tuple[np.ndarray, int]`: (audio_array, sample_rate)

**Raises:**
- `RuntimeError`: If Ollama is not available

**Example:**
```python
brain = Brain(mode="creative")

# Generate creative response
audio, sr = brain.process(
    user_input="Tell me a short story about a robot",
    emotion="Speak with wonder and curiosity"
)

import soundfile as sf
sf.write("story.wav", audio, sr)
```

---

## Engines

### PersonaPlexEngine

**NVIDIA PersonaPlex-7B engine for realtime Speech-to-Speech.**

```python
from engines.personaplex import PersonaPlexEngine
```

#### `__init__(device: str = "auto")`

Initialize PersonaPlex engine.

**Parameters:**
- `device` (str): Device to use - `"auto"`, `"cuda"` (MPS/CPU not supported)

**Returns:** PersonaPlexEngine instance

**Raises:**
- `RuntimeError`: If CUDA is not available

**Example:**
```python
engine = PersonaPlexEngine(device="cuda")
```

---

#### `check_availability() -> bool`

Check if PersonaPlex is available on this system.

**Returns:**
- `bool`: True if CUDA GPU available, False otherwise

**Example:**
```python
from engines.personaplex import PersonaPlexEngine

if PersonaPlexEngine.check_availability():
    engine = PersonaPlexEngine()
else:
    print("PersonaPlex requires NVIDIA GPU")
```

---

#### `generate_s2s_response(...) -> Optional[Tuple[np.ndarray, int]]`

**PRIMARY METHOD:** Generate end-to-end speech-to-speech response.

**Parameters:**
- `audio` (np.ndarray): Input audio array
- `sr` (int): Sample rate of input audio (default: 24000)
- `voice_prompt` (Optional[str]): Persona/voice description
- `text_prompt` (Optional[str]): Text instructions
- `streaming` (bool): Enable streaming mode (default: False)

**Returns:**
- `Tuple[np.ndarray, int]`: (response_audio, sample_rate) or None

**Example:**
```python
import numpy as np
import soundfile as sf

engine = PersonaPlexEngine()

# Record or load user audio
user_audio, sr = sf.read("user_input.wav")

# Generate response
response_audio, sr = engine.generate_s2s_response(
    audio=user_audio,
    sr=sr,
    voice_prompt="You are a helpful assistant"
)

# Save response
sf.write("agent_response.wav", response_audio, sr)
```

---

#### `transcribe_audio(audio: np.ndarray, sr: int = 24000) -> str` [DEPRECATED]

**⚠️ DEPRECATED:** Do not use. PersonaPlex should be used for end-to-end S2S.

**This method is only kept for backward compatibility and will be removed in v3.0.**

---

### Qwen3TTSEngine

**Alibaba Qwen3-TTS engine for high-quality text-to-speech.**

```python
from engines.qwen3tts import Qwen3TTSEngine
```

#### `__init__(device: str = "auto")`

Initialize Qwen3-TTS engine.

**Parameters:**
- `device` (str): Device to use - `"auto"`, `"cuda"`, `"mps"`, or `"cpu"`

**Returns:** Qwen3TTSEngine instance

**Example:**
```python
# Auto-detect device
engine = Qwen3TTSEngine()

# Force CPU
engine = Qwen3TTSEngine(device="cpu")
```

---

#### `synthesize(text: str, ref_audio_path: Optional[str] = None) -> Tuple[np.ndarray, int]`

Synthesize speech from text.

**Parameters:**
- `text` (str): Input text to synthesize
- `ref_audio_path` (Optional[str]): Reference audio for voice cloning

**Returns:**
- `Tuple[np.ndarray, int]`: (audio_array, sample_rate)

**Raises:**
- `ValueError`: If text is empty
- `RuntimeError`: If synthesis fails

**Example:**
```python
import soundfile as sf

engine = Qwen3TTSEngine()

# Basic synthesis
audio, sr = engine.synthesize("Hello from Velloris!")
sf.write("output.wav", audio, sr)

# With voice cloning
audio, sr = engine.synthesize(
    text="This will sound like the reference voice",
    ref_audio_path="voices/my_voice.wav"
)
sf.write("cloned.wav", audio, sr)
```

---

#### `synthesize_with_emotion(text: str, emotion: str, ref_audio_path: Optional[str] = None) -> Tuple[np.ndarray, int]`

Synthesize speech with emotion control.

**Parameters:**
- `text` (str): Input text
- `emotion` (str): Emotion description (e.g., "happy", "sad", "excited")
- `ref_audio_path` (Optional[str]): Reference audio for voice cloning

**Returns:**
- `Tuple[np.ndarray, int]`: (audio_array, sample_rate)

**Example:**
```python
engine = Qwen3TTSEngine()

audio, sr = engine.synthesize_with_emotion(
    text="This is amazing news!",
    emotion="Speak with excitement and joy"
)

import soundfile as sf
sf.write("excited.wav", audio, sr)
```

---

## Configuration

### Config

**Global configuration management.**

```python
from config import Config
```

#### Platform Configuration

```python
# Platform info
Config.platform.OS             # Operating system name
Config.platform.ARCHITECTURE   # CPU architecture
Config.platform.PYTHON_VERSION # Python version

# Hardware detection
Config.platform.CUDA_AVAILABLE # CUDA GPU available?
Config.platform.MPS_AVAILABLE  # Apple MPS available?
```

#### Model Configuration

```python
# Device selection
Config.model.DEVICE            # Selected device (cuda/mps/cpu)
Config.model.DTYPE             # Data type (bfloat16/float32)

# Model paths (auto-download from HuggingFace)
Config.model.PERSONAPLEX_MODEL # PersonaPlex-7B model ID
Config.model.QWEN3_TTS_MODEL   # Qwen3-TTS model ID
```

#### Audio Configuration

```python
# Sample rates
Config.audio.INPUT_SAMPLE_RATE   # 16000 Hz (input)
Config.audio.OUTPUT_SAMPLE_RATE  # 24000 Hz (output)

# Buffer settings
Config.audio.BUFFER_SIZE         # 2.0 seconds
Config.audio.CHUNK_SIZE          # 1024 samples
```

#### Application Configuration

```python
# Mode settings
Config.app.DEFAULT_MODE          # "realtime"
Config.app.MODES                 # ["realtime", "dubbing", "creative"]

# Realtime mode
Config.app.REALTIME_VOICE        # "NATF2" (default voice)
Config.app.REALTIME_PERSONA      # Default persona
Config.app.REALTIME_STREAMING    # True
Config.app.REALTIME_SAMPLE_RATE  # 24000 Hz

# Creative mode
Config.app.CREATIVE_LLM          # "llama3" (Ollama model)
Config.app.CREATIVE_DEFAULT_EMOTION  # ""
```

**Example:**
```python
from config import Config

# Check if CUDA available
if Config.platform.CUDA_AVAILABLE:
    print("CUDA GPU detected")
    print(f"Using device: {Config.model.DEVICE}")

# Get default mode
print(f"Default mode: {Config.app.DEFAULT_MODE}")

# Get available voices
print(f"Available voices: {Config.app.AVAILABLE_VOICES}")
```

---

## Utilities

### Audio Processing

**Common audio utilities.**

```python
import numpy as np
import soundfile as sf
from scipy import signal

# Load audio
audio, sr = sf.read("input.wav")

# Resample audio
target_sr = 24000
if sr != target_sr:
    audio = signal.resample(audio, int(len(audio) * target_sr / sr))

# Convert stereo to mono
if len(audio.shape) > 1:
    audio = audio.mean(axis=1)

# Normalize audio
audio = audio / np.max(np.abs(audio))

# Save audio
sf.write("output.wav", audio, target_sr)
```

---

### Device Detection

**Detect available hardware.**

```python
import torch

# Check CUDA
cuda_available = torch.cuda.is_available()
if cuda_available:
    device_name = torch.cuda.get_device_name(0)
    device_count = torch.cuda.device_count()
    print(f"CUDA: {device_name} (x{device_count})")

# Check MPS (Apple Silicon)
mps_available = torch.backends.mps.is_available()
if mps_available:
    print("MPS: Apple Silicon GPU available")

# Select device
if cuda_available:
    device = "cuda"
elif mps_available:
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")
```

---

## Examples

### Complete Workflow Example

```python
from core.orchestrator import Orchestrator
import soundfile as sf
import numpy as np

# 1. Initialize orchestrator for dubbing mode
orchestrator = Orchestrator(mode="dubbing")

# 2. Prepare input text
text = """
Welcome to Velloris, the advanced AI voice interaction system.
This is an example of high-quality narration using Qwen3-TTS.
"""

# 3. Generate speech
audio, sr = orchestrator.route_request(
    mode="dubbing",
    text=text
)

# 4. Save to file
sf.write("narration.wav", audio, sr)
print(f"Saved narration.wav ({len(audio)/sr:.2f} seconds)")
```

---

### Multi-Mode Application

```python
from core.orchestrator import Orchestrator
import soundfile as sf

class VoiceAssistant:
    def __init__(self):
        # Initialize orchestrators for different modes
        self.realtime_orch = Orchestrator(mode="realtime")
        self.dubbing_orch = Orchestrator(mode="dubbing")
        self.creative_orch = Orchestrator(mode="creative")

    def conversation(self, user_audio):
        """Realtime conversation."""
        audio, sr = self.realtime_orch.route_request(
            mode="realtime",
            audio_input=user_audio,
            voice_prompt="You are a helpful assistant"
        )
        return audio, sr

    def narrate(self, text):
        """High-quality narration."""
        audio, sr = self.dubbing_orch.route_request(
            mode="dubbing",
            text=text
        )
        return audio, sr

    def tell_story(self, prompt):
        """Creative storytelling."""
        audio, sr = self.creative_orch.route_request(
            mode="creative",
            text=prompt,
            emotion="Speak with wonder"
        )
        return audio, sr

# Usage
assistant = VoiceAssistant()

# Narration
audio, sr = assistant.narrate("Welcome to the podcast")
sf.write("intro.wav", audio, sr)

# Story
audio, sr = assistant.tell_story("Tell a story about AI")
sf.write("story.wav", audio, sr)
```

---

### Voice Cloning Example

```python
from core.orchestrator import Orchestrator
import soundfile as sf

# Prepare reference audio (3-5 seconds of clear speech)
ref_audio, ref_sr = sf.read("my_voice.wav")
if ref_sr != 24000:
    from scipy import signal
    ref_audio = signal.resample(ref_audio, int(len(ref_audio) * 24000 / ref_sr))
sf.write("voices/prepared_voice.wav", ref_audio, 24000)

# Initialize orchestrator
orchestrator = Orchestrator(mode="dubbing")

# Generate with voice cloning
texts = [
    "This is the first sentence in my voice.",
    "Here's the second sentence, also in my voice.",
    "And finally, the third sentence."
]

for i, text in enumerate(texts):
    audio, sr = orchestrator.route_request(
        mode="dubbing",
        text=text,
        ref_audio_path="voices/prepared_voice.wav"
    )
    sf.write(f"cloned_{i+1}.wav", audio, sr)

print("Voice cloning complete!")
```

---

### Batch Processing Example

```python
from core.orchestrator import Orchestrator
from concurrent.futures import ThreadPoolExecutor
import soundfile as sf
from tqdm import tqdm

# Load scripts
with open("scripts.txt") as f:
    scripts = [line.strip() for line in f if line.strip()]

def process_script(args):
    """Process a single script."""
    i, text = args
    orchestrator = Orchestrator(mode="dubbing")
    audio, sr = orchestrator.route_request(mode="dubbing", text=text)
    filename = f"output_{i:03d}.wav"
    sf.write(filename, audio, sr)
    return filename

# Process in parallel
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(tqdm(
        executor.map(process_script, enumerate(scripts)),
        total=len(scripts),
        desc="Processing"
    ))

print(f"Processed {len(results)} scripts")
```

---

## Error Handling

### Common Exceptions

```python
from core.orchestrator import Orchestrator

orchestrator = Orchestrator(mode="creative")

try:
    audio, sr = orchestrator.route_request(
        mode="creative",
        text="Tell a story"
    )
except RuntimeError as e:
    # Ollama not available
    print(f"Error: {e}")
    print("Make sure Ollama is running: ollama serve")

except ValueError as e:
    # Invalid parameters
    print(f"Invalid input: {e}")

except Exception as e:
    # Other errors
    print(f"Unexpected error: {e}")
```

---

### Graceful Degradation

```python
from core.orchestrator import Orchestrator

# Try realtime mode, fall back to dubbing if GPU not available
try:
    orchestrator = Orchestrator(mode="realtime")
    print("Using realtime mode (PersonaPlex)")
except RuntimeError:
    orchestrator = Orchestrator(mode="dubbing")
    print("Falling back to dubbing mode (Qwen3-TTS)")

# Process request
audio, sr = orchestrator.route_request(
    mode=orchestrator.mode,  # Use whatever mode we initialized
    text="Hello from Velloris"
)
```

---

## Type Hints

For better IDE support, use type hints:

```python
from typing import Optional, Tuple
import numpy as np
from core.orchestrator import Orchestrator

def process_text(
    orchestrator: Orchestrator,
    text: str,
    mode: str = "dubbing"
) -> Optional[Tuple[np.ndarray, int]]:
    """
    Process text and return audio.

    Args:
        orchestrator: Velloris orchestrator instance
        text: Input text to synthesize
        mode: Operating mode (dubbing, creative)

    Returns:
        Tuple of (audio_array, sample_rate) or None on error
    """
    try:
        audio, sr = orchestrator.route_request(mode=mode, text=text)
        return audio, sr
    except Exception as e:
        print(f"Error: {e}")
        return None
```

---

## Further Reading

- [EXAMPLES.md](EXAMPLES.md) - More code examples
- [QUICKSTART.md](QUICKSTART.md) - Getting started guide
- [README.md](README.md) - Full documentation
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Debugging guide

---

**Last updated:** February 2024
