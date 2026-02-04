#!/bin/bash

echo "🚀 Starting Local Voice AI Setup for Velloris..."

# Create a fresh environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

source venv/bin/activate  # Activate for Mac/Linux
# For Windows, you'd typically run `venv\Scripts\activate` manually in your shell

echo "Installing core dependencies from requirements.txt..."
pip install -r requirements.txt

if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🍎 Detected MacOS (M-Series). Installing MLX-optimized stack..."
    pip install mlx mlx-audio mlx-lm
    # No specific sounddevice install needed here as it's in requirements.txt
    # Silero VAD is loaded via torch.hub.load, so no direct pip install
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "🪟 Detected Windows. Installing CUDA-optimized stack..."
    # Ensure PyTorch with CUDA is installed
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    # Install FlashAttention 2 for a 2-3x speed boost on Windows
    pip install flash-attn --no-build-isolation
    # For 4-bit quantization on Windows/NVIDIA
    pip install bitsandbytes
else
    echo "⚠️ OS not fully recognized. Proceeding with generic install (might require manual hardware-specific setup)."
fi

echo "✅ Velloris Environment Setup Ready! Remember to activate your venv."
echo "   On Mac/Linux: source venv/bin/activate"
echo "   On Windows: .\venv\Scripts\activate"
