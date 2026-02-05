#!/bin/bash

# Velloris Installation Script for macOS (Metal/MPS)
#
# This script installs Velloris with Apple Metal (MPS) support
# Works on M1/M2/M3 Macs with macOS 12.3+

set -e  # Exit on error

echo ""
echo "========================================"
echo "Velloris Installation - macOS (Metal)"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 not found"
    echo "Please install Python 3.11+ using Homebrew:"
    echo "  brew install python@3.12"
    exit 1
fi

echo "Step 1: Installing system dependencies via Homebrew..."
echo "This may ask for your password"
echo ""

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Install system dependencies
echo "Installing: opus, ffmpeg, portaudio"
brew install opus ffmpeg portaudio

echo ""
echo "Step 2: Installing PyTorch with Metal support..."
pip3 install torch torchvision torchaudio

echo ""
echo "Step 3: Installing core dependencies..."
pip3 install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies"
    exit 1
fi

echo ""
echo "Step 4: Installing optional Apple MLX stack (recommended)..."
echo "This provides optimized M-series support for future features"
echo ""

pip3 install mlx mlx-audio mlx-lm

if [ $? -ne 0 ]; then
    echo "Warning: MLX installation failed (optional)"
    echo "You can still use Velloris with standard PyTorch"
else
    echo "MLX stack installed successfully"
fi

echo ""
echo "========================================"
echo "Installation Complete!"
echo "========================================"
echo ""
echo "To run Velloris on Metal (MPS):"
echo "  python3 main.py --mode interactive --device mps"
echo "  python3 main.py --mode dubbing --script \"Your script\""
echo ""
echo "For CPU mode (if MPS is unstable):"
echo "  python3 main.py --mode interactive --device cpu"
echo ""
echo "For more information:"
echo "  python3 main.py --show-config"
echo ""
echo "Note: PersonaPlex-7B is optimized for NVIDIA GPUs."
echo "      On M-series Macs, Qwen3-TTS will work well,"
echo "      but PersonaPlex may be slow. Consider using CPU mode."
echo ""
