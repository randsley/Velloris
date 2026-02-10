# PersonaPlex-7B S2S Usage Examples

## Basic Initialization

```python
from engines.personaplex import PersonaPlexEngine
import numpy as np

# Initialize the engine
engine = PersonaPlexEngine(
    device="cuda",  # auto-detects CUDA, MPS, or CPU
    voice="natural_female_2",
    persona="You are a helpful AI assistant"
)
```

## Real-Time Conversation Example

```python
# Record user audio (24kHz, numpy array)
user_audio = record_microphone(duration=2.0)  # Your recording function

# Generate S2S response
agent_audio, sr = engine.generate_s2s_response(
    audio=user_audio,
    sr=24000  # PersonaPlex expects 24kHz
)

# Play the response
play_audio(agent_audio, sr)
```

## Voice Selection Example

```python
# List all available voices
voices = PersonaPlexEngine.get_available_voices()
print(f"Available voices: {voices}")

# Create engine with different voices
voices_to_test = [
    "natural_female_0",
    "natural_male_1",
    "varied_female_2",
    "varied_male_3"
]

for voice in voices_to_test:
    engine = PersonaPlexEngine(device="cuda", voice=voice)
    # Use engine for S2S inference...
    engine.unload()  # Free memory
```

## Persona/Role Control Example

```python
# Different personas for different use cases
personas = {
    "teacher": "You are a patient and encouraging teacher. Explain concepts simply.",
    "tutor": "You are an expert tutor. Answer questions thoroughly and ask follow-up questions.",
    "friend": "You are a friendly and casual conversational partner. Be warm and engaging.",
    "therapist": "You are a supportive and empathetic counselor. Listen carefully and respond thoughtfully."
}

for role, persona_text in personas.items():
    engine = PersonaPlexEngine(
        device="cuda",
        voice="natural_female_2",
        persona=persona_text
    )

    # Process user audio with this persona
    response_audio, sr = engine.generate_s2s_response(user_audio)

    # Save response
    save_audio(f"response_{role}.wav", response_audio, sr)
    engine.unload()
```

## Batch Processing Example

```python
# Process multiple audio files with same voice/persona
audio_files = ["audio1.wav", "audio2.wav", "audio3.wav"]

engine = PersonaPlexEngine(
    device="cuda",
    voice="natural_male_0",
    persona="You are a professional customer service representative"
)

for audio_file in audio_files:
    # Load audio
    user_audio, sr = load_audio(audio_file)

    # Generate response
    agent_audio, output_sr = engine.generate_s2s_response(user_audio, sr=sr)

    # Save response
    output_file = audio_file.replace(".wav", "_response.wav")
    save_audio(output_file, agent_audio, output_sr)

engine.unload()
```

## Integration with Real-Time Audio I/O

```python
from utils.audio_io import IntegratedAudioController
from utils.vad_handler import InterruptionHandler

# Set up audio I/O with interruption detection
vad_handler = InterruptionHandler()
audio_controller = IntegratedAudioController(vad_handler)

# Initialize PersonaPlex
engine = PersonaPlexEngine(
    device="cuda",
    voice="natural_female_1",
    persona="Respond naturally to user input"
)

# Main conversation loop
while True:
    # Record user speech
    user_audio = audio_controller.record_audio(duration=2.0)

    # Check for silence (early termination)
    if not vad_handler.check_for_speech(user_audio):
        print("No speech detected, waiting for input...")
        continue

    # Generate response
    response_audio, sr = engine.generate_s2s_response(user_audio)

    # Play response
    audio_controller.queue_audio_output(response_audio)

engine.unload()
```

## Multi-Voice Conversation Example

```python
# Simulate multi-turn conversation with different voices

# User speaks
user_audio = record_audio()

# Agent responds (female voice)
agent = PersonaPlexEngine(
    device="cuda",
    voice="natural_female_2",
    persona="You are Alice"
)
alice_response, sr = agent.generate_s2s_response(user_audio)
play_audio(alice_response, sr)
agent.unload()

# Second agent responds (male voice)
agent2 = PersonaPlexEngine(
    device="cuda",
    voice="natural_male_1",
    persona="You are Bob"
)
bob_response, sr = agent2.generate_s2s_response(user_audio)
play_audio(bob_response, sr)
agent2.unload()
```

## Device-Specific Examples

```python
import torch

# Auto-detect device (CUDA -> MPS -> CPU)
engine_auto = PersonaPlexEngine(device="auto")
print(f"Using device: {engine_auto.device}")

# Force CPU mode (slower but works everywhere)
engine_cpu = PersonaPlexEngine(device="cpu")

# Force CUDA (fastest, requires NVIDIA GPU)
if torch.cuda.is_available():
    engine_cuda = PersonaPlexEngine(device="cuda")

# Force MPS (macOS Metal, good performance on Apple Silicon)
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    engine_mps = PersonaPlexEngine(device="mps")
```

## Error Handling Example

```python
import traceback

def safe_s2s_inference(user_audio, engine):
    """Safe wrapper for S2S inference with error handling"""
    try:
        if engine.model is None:
            print("Engine not loaded, initializing...")
            return None

        # Ensure audio is the correct format
        if user_audio.dtype != np.float32:
            user_audio = user_audio.astype(np.float32)

        # Generate response
        result = engine.generate_s2s_response(user_audio, sr=24000)

        if result is None:
            print("Inference failed, no output generated")
            return None

        agent_audio, sr = result
        return agent_audio, sr

    except Exception as e:
        print(f"S2S inference error: {e}")
        traceback.print_exc()
        return None

# Usage
engine = PersonaPlexEngine(device="cuda")
result = safe_s2s_inference(user_audio, engine)
if result:
    agent_audio, sr = result
    play_audio(agent_audio, sr)
```

## Model Memory Management

```python
# Load engine once, reuse multiple times
engine = PersonaPlexEngine(device="cuda")

# Process many audio samples
for i in range(100):
    user_audio = load_audio(f"audio_{i}.wav")
    response, sr = engine.generate_s2s_response(user_audio)
    save_audio(f"response_{i}.wav", response, sr)

# Unload when done to free VRAM
engine.unload()

# VRAM is now available for other tasks
```

## Advanced: Custom Prompts Per Request

```python
engine = PersonaPlexEngine(device="cuda", voice="natural_female_0")

# Override persona at inference time
custom_responses = engine.generate_s2s_response(
    audio=user_audio,
    sr=24000,
    voice_prompt="natural_male_2.pt",  # Override voice
    text_prompt="Respond as a pirate would"  # Override persona
)
```

## Integration with Orchestrator

```python
from core.orchestrator import LocalVoiceOrchestrator

# Use through the orchestrator for automatic mode selection
orchestrator = LocalVoiceOrchestrator(device="cuda")

# Real-time mode (PersonaPlex S2S)
result = orchestrator.route_request(
    mode="realtime",
    audio_input=user_audio,
    voice_prompt="natural_female_2.pt",
    text_prompt="You are a helpful assistant"
)

# Access the audio response
if result:
    agent_audio, sr = result
    play_audio(agent_audio, sr)

orchestrator.unload_engines()
```

## Notes

- **Sample Rate**: PersonaPlex expects 24kHz audio. Input at other rates is auto-resampled.
- **Audio Dtype**: Expects float32 numpy arrays. Conversion is automatic.
- **GPU Memory**: Model uses ~14GB VRAM on RTX 3080+. Use `engine.unload()` to free memory.
- **Latency**: Target 70-170ms per 1-second audio chunk on CUDA.
- **Voice Files**: Stored in `models/personaplex/voices/`. 18 voices available.
- **Error Handling**: All methods return `None` on error instead of raising exceptions.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of memory | Use CPU mode or process shorter audio chunks |
| Slow inference | Use CUDA device instead of CPU |
| No audio output | Check voice file exists in models/personaplex/voices/ |
| Microphone not working | Use alternative audio input method |
| Model loading fails | Ensure all model files downloaded (run download_models.py) |
