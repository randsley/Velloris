# 📚 Velloris Examples

This guide provides practical code examples and usage patterns for Velloris v2.0.

---

## Table of Contents

- [Command-Line Examples](#command-line-examples)
- [Python API Examples](#python-api-examples)
- [Integration Examples](#integration-examples)
- [Advanced Use Cases](#advanced-use-cases)
- [Batch Processing](#batch-processing)
- [Custom Workflows](#custom-workflows)

---

## Command-Line Examples

### Realtime Mode Examples

**Basic conversation:**
```bash
python3 main.py --mode realtime --persona "You are a helpful assistant" --voice NATF2
```

**Custom personality:**
```bash
python3 main.py --mode realtime \
  --persona "You are a friendly tutor who explains concepts clearly with examples" \
  --voice NATM1
```

**Different voices:**
```bash
# Female voices
python3 main.py --mode realtime --voice NATF0  # Natural Female 0
python3 main.py --mode realtime --voice NATF1  # Natural Female 1
python3 main.py --mode realtime --voice NATF2  # Natural Female 2 (default)
python3 main.py --mode realtime --voice NATF3  # Natural Female 3
python3 main.py --mode realtime --voice VARF0  # Variant Female 0

# Male voices
python3 main.py --mode realtime --voice NATM0  # Natural Male 0
python3 main.py --mode realtime --voice NATM1  # Natural Male 1
python3 main.py --mode realtime --voice NATM2  # Natural Male 2
python3 main.py --mode realtime --voice NATM3  # Natural Male 3
python3 main.py --mode realtime --voice VARM0  # Variant Male 0
```

**Specific use cases:**
```bash
# Customer service agent
python3 main.py --mode realtime \
  --persona "You are a professional customer service agent. Be polite, helpful, and concise." \
  --voice NATF2

# Technical support
python3 main.py --mode realtime \
  --persona "You are a technical support specialist. Provide clear, step-by-step solutions." \
  --voice NATM1

# Language tutor
python3 main.py --mode realtime \
  --persona "You are a patient language tutor. Correct mistakes gently and provide examples." \
  --voice NATF1
```

---

### Dubbing Mode Examples

**Basic narration:**
```bash
python3 main.py --mode dubbing --script "Welcome to Velloris, the advanced AI voice interaction system."
```

**Long-form content:**
```bash
python3 main.py --mode dubbing --script "$(cat narration.txt)"
```

**With voice cloning:**
```bash
python3 main.py --mode dubbing \
  --script "This narration will sound like my voice" \
  --voice-ref voices/my_voice.wav
```

**Audiobook production:**
```bash
python3 main.py --mode dubbing \
  --script "Chapter 1: The Beginning. It was a dark and stormy night..." \
  --voice-ref voices/narrator.wav \
  --output audiobook_chapter1.wav
```

**Video narration:**
```bash
python3 main.py --mode dubbing \
  --script "In this tutorial, we'll explore the three modes of Velloris and how to use them effectively." \
  --output tutorial_audio.wav
```

**Podcast intro:**
```bash
python3 main.py --mode dubbing \
  --script "Welcome to the AI Insights podcast. I'm your host, and today we're discussing the future of voice AI." \
  --voice-ref voices/podcast_host.wav
```

---

### Creative Mode Examples

**Basic storytelling:**
```bash
# Make sure Ollama is running first
ollama serve  # Terminal 1

# Terminal 2
python3 main.py --mode creative \
  --script "Tell me a short story about a robot learning to paint"
```

**With emotion control:**
```bash
python3 main.py --mode creative \
  --script "Describe a thrilling space battle between two starships" \
  --emotion "Speak with excitement and urgency"
```

**Different LLM models:**
```bash
# Using Llama 3 (default)
python3 main.py --mode creative \
  --script "Write a poem about autumn leaves" \
  --llm-model llama3

# Using Mistral
ollama pull mistral
python3 main.py --mode creative \
  --script "Write a poem about autumn leaves" \
  --llm-model mistral

# Using smaller model for faster responses
ollama pull llama3:8b
python3 main.py --mode creative \
  --script "Write a short bedtime story" \
  --llm-model llama3:8b
```

**Character dialogue:**
```bash
python3 main.py --mode creative \
  --script "You are a wise old wizard. Give advice to a young adventurer about facing their fears." \
  --emotion "Speak with wisdom and gravitas" \
  --voice-ref voices/wizard_voice.wav
```

**Interactive storytelling:**
```bash
python3 main.py --mode creative \
  --script "Continue this story: The spaceship's alarms blared as the crew realized they were not alone..." \
  --emotion "Build suspense and tension"
```

---

## Python API Examples

### Basic Usage

**Realtime conversation:**
```python
from core.orchestrator import Orchestrator
import numpy as np

# Initialize orchestrator
orchestrator = Orchestrator(mode="realtime")

# Simulate user audio input (replace with actual microphone audio)
user_audio = np.random.randn(16000)  # 1 second at 16kHz

# Generate response
agent_audio, sample_rate = orchestrator.route_request(
    mode="realtime",
    audio_input=user_audio,
    voice_prompt="You are a helpful assistant"
)

# Play or save audio
import soundfile as sf
sf.write("response.wav", agent_audio, sample_rate)
```

**Dubbing/narration:**
```python
from core.orchestrator import Orchestrator
import soundfile as sf

# Initialize orchestrator
orchestrator = Orchestrator(mode="dubbing")

# Generate narration
text = "Welcome to this audiobook. Today we explore the mysteries of the universe."
audio, sr = orchestrator.route_request(
    mode="dubbing",
    text=text
)

# Save to file
sf.write("narration.wav", audio, sr)
```

**Creative synthesis:**
```python
from core.orchestrator import Orchestrator
import soundfile as sf

# Initialize orchestrator (requires Ollama running)
orchestrator = Orchestrator(mode="creative")

# Generate creative content
audio, sr = orchestrator.route_request(
    mode="creative",
    text="Tell a story about a lonely robot",
    emotion="Speak with sadness and reflection"
)

# Save to file
sf.write("story.wav", audio, sr)
```

---

### Voice Cloning

**Using reference audio:**
```python
from core.orchestrator import Orchestrator
import soundfile as sf

orchestrator = Orchestrator(mode="dubbing")

# Generate with voice cloning
audio, sr = orchestrator.route_request(
    mode="dubbing",
    text="This will sound like the reference voice",
    ref_audio_path="voices/my_voice.wav"
)

sf.write("cloned_voice.wav", audio, sr)
```

**Preparing reference audio:**
```python
import soundfile as sf
import numpy as np

# Load and preprocess reference audio
audio, sr = sf.read("raw_voice.wav")

# Convert to mono if stereo
if len(audio.shape) > 1:
    audio = audio.mean(axis=1)

# Resample to 24kHz (if needed)
from scipy import signal
if sr != 24000:
    audio = signal.resample(audio, int(len(audio) * 24000 / sr))

# Save preprocessed reference
sf.write("voices/prepared_voice.wav", audio, 24000)
```

---

### Batch Processing

**Process multiple scripts:**
```python
from core.orchestrator import Orchestrator
import soundfile as sf

orchestrator = Orchestrator(mode="dubbing")

scripts = [
    "Chapter 1: The Beginning",
    "Chapter 2: The Journey",
    "Chapter 3: The Revelation",
    "Chapter 4: The End"
]

for i, text in enumerate(scripts):
    print(f"Processing chapter {i+1}/{len(scripts)}...")

    audio, sr = orchestrator.route_request(
        mode="dubbing",
        text=text,
        ref_audio_path="voices/narrator.wav"
    )

    sf.write(f"chapter_{i+1:02d}.wav", audio, sr)
    print(f"Saved chapter_{i+1:02d}.wav")

print("All chapters processed!")
```

**Process with progress bar:**
```python
from core.orchestrator import Orchestrator
import soundfile as sf
from tqdm import tqdm

orchestrator = Orchestrator(mode="dubbing")

scripts = ["Script 1", "Script 2", "Script 3", ...]  # Your scripts

for i, text in enumerate(tqdm(scripts, desc="Processing")):
    audio, sr = orchestrator.route_request(mode="dubbing", text=text)
    sf.write(f"output_{i:03d}.wav", audio, sr)
```

---

### Error Handling

**Robust processing with error handling:**
```python
from core.orchestrator import Orchestrator
import soundfile as sf
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

orchestrator = Orchestrator(mode="dubbing")

def process_script(text, output_path):
    """Process a script with error handling."""
    try:
        audio, sr = orchestrator.route_request(
            mode="dubbing",
            text=text
        )
        sf.write(output_path, audio, sr)
        logger.info(f"Successfully saved: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to process script: {e}")
        return False

# Use it
scripts = {
    "intro.wav": "Welcome to the podcast",
    "outro.wav": "Thanks for listening"
}

for output_path, text in scripts.items():
    process_script(text, output_path)
```

---

## Integration Examples

### Flask Web API

**Simple REST API:**
```python
from flask import Flask, request, send_file, jsonify
from core.orchestrator import Orchestrator
import soundfile as sf
import io
import numpy as np

app = Flask(__name__)
orchestrator = Orchestrator(mode="dubbing")

@app.route('/synthesize', methods=['POST'])
def synthesize():
    """Synthesize speech from text."""
    try:
        data = request.get_json()
        text = data.get('text')
        mode = data.get('mode', 'dubbing')

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # Generate audio
        audio, sr = orchestrator.route_request(mode=mode, text=text)

        # Convert to WAV bytes
        buffer = io.BytesIO()
        sf.write(buffer, audio, sr, format='WAV')
        buffer.seek(0)

        return send_file(buffer, mimetype='audio/wav')

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

**Client usage:**
```python
import requests

# Synthesize speech
response = requests.post('http://localhost:5000/synthesize', json={
    'text': 'Hello from Velloris API!',
    'mode': 'dubbing'
})

# Save audio
with open('output.wav', 'wb') as f:
    f.write(response.content)
```

---

### Discord Bot Integration

**Basic Discord voice bot:**
```python
import discord
from discord.ext import commands
from core.orchestrator import Orchestrator
import soundfile as sf
import asyncio

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

orchestrator = Orchestrator(mode="creative")

@bot.command()
async def speak(ctx, *, text: str):
    """Generate and play speech in voice channel."""
    if not ctx.author.voice:
        await ctx.send("You must be in a voice channel!")
        return

    # Generate audio
    audio, sr = orchestrator.route_request(mode="creative", text=text)

    # Save to file
    sf.write("temp_speech.wav", audio, sr)

    # Join voice channel and play
    voice_channel = ctx.author.voice.channel
    vc = await voice_channel.connect()
    vc.play(discord.FFmpegPCMAudio("temp_speech.wav"))

    # Wait for playback to finish
    while vc.is_playing():
        await asyncio.sleep(1)

    await vc.disconnect()

bot.run('YOUR_BOT_TOKEN')
```

---

### Telegram Bot Integration

**Text-to-speech bot:**
```python
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from core.orchestrator import Orchestrator
import soundfile as sf

orchestrator = Orchestrator(mode="dubbing")

async def start(update: Update, context):
    """Welcome message."""
    await update.message.reply_text(
        "Send me text and I'll convert it to speech!"
    )

async def synthesize(update: Update, context):
    """Convert text message to speech."""
    text = update.message.text

    # Generate audio
    audio, sr = orchestrator.route_request(mode="dubbing", text=text)

    # Save to file
    sf.write("output.wav", audio, sr)

    # Send audio file
    await update.message.reply_voice(voice=open("output.wav", "rb"))

def main():
    app = Application.builder().token("YOUR_BOT_TOKEN").build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, synthesize))

    app.run_polling()

if __name__ == '__main__':
    main()
```

---

### Command-Line Script

**Batch audiobook processor:**
```bash
#!/bin/bash
# batch_audiobook.sh

# Usage: ./batch_audiobook.sh input_text.txt output_directory narrator_voice.wav

INPUT_FILE=$1
OUTPUT_DIR=$2
VOICE_REF=$3

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Split text into chapters (assuming ## Chapter markers)
csplit --prefix="$OUTPUT_DIR/chapter_" "$INPUT_FILE" '/^## Chapter/' '{*}' --suppress-matched

# Process each chapter
for chapter in "$OUTPUT_DIR"/chapter_*; do
    chapter_num=$(basename "$chapter" | sed 's/chapter_//')

    echo "Processing chapter $chapter_num..."

    python3 main.py \
        --mode dubbing \
        --script "$(cat "$chapter")" \
        --voice-ref "$VOICE_REF" \
        --output "$OUTPUT_DIR/chapter_${chapter_num}.wav"

    rm "$chapter"  # Clean up text file
done

echo "Audiobook generation complete!"
```

---

## Advanced Use Cases

### Real-Time Conversation with Context

**Maintain conversation history:**
```python
from core.orchestrator import Orchestrator
import numpy as np
import soundfile as sf

class ConversationAgent:
    def __init__(self, persona="You are a helpful assistant"):
        self.orchestrator = Orchestrator(mode="realtime")
        self.persona = persona
        self.history = []

    def process_audio(self, user_audio: np.ndarray) -> tuple:
        """Process user audio and generate response."""
        # Build context from history
        context = self.persona + "\n\nConversation history:\n"
        for turn in self.history[-5:]:  # Last 5 turns
            context += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"

        # Generate response with context
        agent_audio, sr = self.orchestrator.route_request(
            mode="realtime",
            audio_input=user_audio,
            voice_prompt=context
        )

        # Update history (simplified - you'd need actual transcription)
        self.history.append({
            'user': '[audio]',
            'assistant': '[audio response]'
        })

        return agent_audio, sr

# Usage
agent = ConversationAgent(persona="You are a friendly tutor")

# Simulate conversation
user_audio = np.random.randn(16000)  # Replace with real audio
response_audio, sr = agent.process_audio(user_audio)
sf.write("response.wav", response_audio, sr)
```

---

### Custom Voice Mixing

**Mix narration with background music:**
```python
from core.orchestrator import Orchestrator
import soundfile as sf
import numpy as np

def mix_audio(narration, music, narration_volume=1.0, music_volume=0.3):
    """Mix narration with background music."""
    # Ensure same length
    if len(narration) > len(music):
        music = np.pad(music, (0, len(narration) - len(music)))
    else:
        music = music[:len(narration)]

    # Mix with volume control
    mixed = (narration * narration_volume) + (music * music_volume)

    # Normalize
    mixed = mixed / np.max(np.abs(mixed))

    return mixed

# Generate narration
orchestrator = Orchestrator(mode="dubbing")
narration, sr = orchestrator.route_request(
    mode="dubbing",
    text="In a world where AI and humans coexist..."
)

# Load background music
music, music_sr = sf.read("background_music.wav")

# Resample music if needed
if music_sr != sr:
    from scipy import signal
    music = signal.resample(music, int(len(music) * sr / music_sr))

# Mix
final_audio = mix_audio(narration, music)

# Save
sf.write("narration_with_music.wav", final_audio, sr)
```

---

### Dynamic Voice Selection

**Choose voice based on content:**
```python
from core.orchestrator import Orchestrator
import soundfile as sf

def get_appropriate_voice(text: str) -> str:
    """Select voice based on text content."""
    text_lower = text.lower()

    # Keyword-based selection
    if any(word in text_lower for word in ['excited', 'amazing', 'incredible']):
        return 'NATF3'  # Energetic female
    elif any(word in text_lower for word in ['serious', 'important', 'warning']):
        return 'NATM2'  # Authoritative male
    elif any(word in text_lower for word in ['gentle', 'calm', 'peaceful']):
        return 'NATF1'  # Soft female
    else:
        return 'NATF2'  # Default

orchestrator = Orchestrator(mode="realtime")

texts = [
    "This is an exciting announcement!",
    "Please take this serious warning carefully.",
    "Breathe gently and relax."
]

for i, text in enumerate(texts):
    voice = get_appropriate_voice(text)
    print(f"Using voice: {voice}")

    # Generate with selected voice
    # Note: Voice selection in realtime mode is via CLI flag
    # For programmatic use, you'd need to modify the engine directly
```

---

### Streaming Audio Processing

**Process audio in chunks:**
```python
from core.orchestrator import Orchestrator
import soundfile as sf
import numpy as np

def stream_audio_chunks(orchestrator, text, chunk_duration=5.0):
    """Generate and stream audio in chunks."""
    # Split text into sentences
    sentences = text.split('. ')

    for sentence in sentences:
        if not sentence.strip():
            continue

        # Generate audio for sentence
        audio, sr = orchestrator.route_request(
            mode="dubbing",
            text=sentence + '.'
        )

        # Yield audio chunk
        yield audio, sr

# Usage
orchestrator = Orchestrator(mode="dubbing")
long_text = "First sentence. Second sentence. Third sentence. Fourth sentence."

all_audio = []
for audio_chunk, sr in stream_audio_chunks(orchestrator, long_text):
    # Process chunk (e.g., play immediately, stream to network, etc.)
    all_audio.append(audio_chunk)
    print(f"Processed chunk of {len(audio_chunk)/sr:.2f} seconds")

# Concatenate all chunks
final_audio = np.concatenate(all_audio)
sf.write("streamed_output.wav", final_audio, sr)
```

---

## Custom Workflows

### Multi-Language Narration

**Generate narration in multiple languages:**
```python
from core.orchestrator import Orchestrator
import soundfile as sf

orchestrator = Orchestrator(mode="dubbing")

translations = {
    "en": "Welcome to Velloris, the advanced AI voice system.",
    "es": "Bienvenido a Velloris, el avanzado sistema de voz AI.",
    "fr": "Bienvenue à Velloris, le système vocal IA avancé.",
    "de": "Willkommen bei Velloris, dem fortschrittlichen KI-Sprachsystem.",
    "zh": "欢迎使用Velloris,先进的AI语音系统。"
}

for lang_code, text in translations.items():
    print(f"Generating {lang_code}...")

    audio, sr = orchestrator.route_request(
        mode="dubbing",
        text=text
    )

    sf.write(f"welcome_{lang_code}.wav", audio, sr)

print("Multi-language generation complete!")
```

---

### Interactive Story Generator

**Generate branching narrative:**
```python
from core.orchestrator import Orchestrator
import soundfile as sf

orchestrator = Orchestrator(mode="creative")

def generate_story_segment(prompt, emotion="neutral"):
    """Generate a story segment."""
    audio, sr = orchestrator.route_request(
        mode="creative",
        text=prompt,
        emotion=emotion
    )
    return audio, sr

# Story structure
story_tree = {
    "intro": {
        "prompt": "Begin a fantasy story about a young wizard",
        "emotion": "Speak with wonder",
        "next": ["choice_magic", "choice_sword"]
    },
    "choice_magic": {
        "prompt": "The wizard chooses to study ancient magic",
        "emotion": "Speak with curiosity",
        "next": ["ending_powerful"]
    },
    "choice_sword": {
        "prompt": "The wizard picks up a legendary sword",
        "emotion": "Speak with determination",
        "next": ["ending_warrior"]
    },
    "ending_powerful": {
        "prompt": "The wizard becomes the most powerful mage",
        "emotion": "Speak with triumph"
    },
    "ending_warrior": {
        "prompt": "The wizard becomes a legendary warrior",
        "emotion": "Speak with pride"
    }
}

# Generate one path through the story
path = ["intro", "choice_magic", "ending_powerful"]
all_audio = []

for segment_id in path:
    segment = story_tree[segment_id]
    print(f"Generating: {segment_id}")

    audio, sr = generate_story_segment(
        segment["prompt"],
        segment.get("emotion", "neutral")
    )

    all_audio.append(audio)
    sf.write(f"story_{segment_id}.wav", audio, sr)

# Combine full story
import numpy as np
full_story = np.concatenate(all_audio)
sf.write("full_story.wav", full_story, sr)
```

---

## More Examples

For more examples and use cases, see:
- [FAQ.md](FAQ.md) - Common questions and answers
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Debugging guides
- [QUICKSTART.md](QUICKSTART.md) - Getting started examples
- [README.md](README.md) - Full documentation
- [GitHub Discussions](https://github.com/randsley/Velloris/discussions) - Community examples

---

## Contributing Examples

Have a cool use case or integration? We'd love to see it!

1. Create a new example in this file or as a separate script
2. Test it thoroughly
3. Submit a pull request
4. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines

---

**Happy building with Velloris!** 🚀
