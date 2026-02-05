@echo off
REM Velloris Installation Script for Windows (CUDA)
REM
REM This script installs Velloris with NVIDIA CUDA support
REM Prerequisites: Python 3.11+ and NVIDIA GPU with CUDA 12.1+

echo.
echo ========================================
echo Velloris Installation - Windows (CUDA)
echo ========================================
echo.

REM Check Python version
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.11+ from python.org
    pause
    exit /b 1
)

echo Step 1: Installing PyTorch with CUDA support...
echo This may take a few minutes...
echo.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

if errorlevel 1 (
    echo Error: Failed to install PyTorch
    pause
    exit /b 1
)

echo.
echo Step 2: Installing core dependencies...
pip install -r requirements.txt

if errorlevel 1 (
    echo Error: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo Step 3: Installing optional performance packages...
echo.

REM Install FlashAttention 2
echo Installing FlashAttention 2 (optional, for faster inference)...
pip install psutil ninja packaging >nul 2>&1
pip install flash-attn --no-build-isolation

if errorlevel 1 (
    echo Warning: FlashAttention 2 installation failed (optional)
    echo You can skip this and still use Velloris
) else (
    echo FlashAttention 2 installed successfully
)

echo.
REM Install bitsandbytes
echo Installing bitsandbytes (optional, for 4-bit quantization)...
pip install bitsandbytes

if errorlevel 1 (
    echo Warning: bitsandbytes installation failed (optional)
) else (
    echo bitsandbytes installed successfully
)

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo To run Velloris:
echo   python main.py --mode interactive
echo   python main.py --mode dubbing --script "Your script here"
echo.
echo For more information:
echo   python main.py --show-config
echo.
pause
