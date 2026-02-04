# 🎙️ Velloris
### *The Local-First, High-Fidelity Voice Agent Engine*
 
**Velloris** is a state-of-the-art framework for creating lifelike, interactive AI agents that run entirely on your local hardware (**M-Series Mac** or **Windows NVIDIA GPU**). By orchestrating the real-time interaction of **PersonaPlex-7B** with the expressive "Voice Design" of **Qwen3-TTS**, Velloris achieves human-level conversation without the cloud.

## 🛠️ Quick Start

1.  **Clone & Setup:**
    ```bash
    git clone https://github.randsley/Velloris.git
    cd Velloris
    chmod +x setup.sh # Make the setup script executable (for Mac/Linux)
    ./setup.sh         # Run the setup script
    # On Windows, you might need to run `.\setup.sh` in PowerShell
    # And manually activate venv: `.\venv\Scripts\activate`
    ```
    After running `setup.sh`, make sure your Python virtual environment is activated.

2.  **Configure Voice:**
    Place a 3-second `.wav` of your target voice in `voices/reference.wav`. This will be used for voice cloning.

3.  **Run:**
    ```bash
    python main.py --mode interactive
    ```
    (Note: `main.py` is not yet created, this is a placeholder instruction)

## 🏗️ Project Structure

```
Velloris/
├── .github/              # GitHub Actions workflows and issue templates
├── assets/               # Branding, logos, and architecture diagrams
├── core/                 # The "Brain" and Routing logic
│   ├── orchestrator.py   # Manages switching between S2S and TTS engines
│   └── brain.py          # Integrates with local LLMs (e.g., Ollama)
├── engines/              # Model-specific implementations
│   ├── personaplex.py    # Placeholder for PersonaPlex S2S logic
│   └── qwen_tts.py       # Qwen3-TTS for expressive cloning and dubbing
├── utils/                # Audio, VAD, and AEC helpers
│   ├── audio_io.py       # Handles audio input/output and AEC
│   └── vad_handler.py    # Silero VAD for real-time voice activity detection
├── voices/               # Storage for 3s voice cloning samples (e.g., reference.wav)
├── LICENSE               # Apache License 2.0
├── README.md             # This file
├── requirements.txt      # Python dependencies
└── setup.sh              # OS-aware setup script for dependencies
```