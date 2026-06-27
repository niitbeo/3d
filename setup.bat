@echo off
echo ========================================================
echo Setting up 3D Generation Pipeline for Windows (GTX 1050ti)
echo ========================================================

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH. Please install Python 3.10 or 3.11.
    pause
    exit /b
)

:: Create Virtual Environment
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

:: Activate Virtual Environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

:: Install PyTorch with CUDA 11.8 or 12.1 (Update index-url if you have CUDA 12.1 installed)
echo Installing PyTorch with CUDA support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

:: Install Core Dependencies
echo Installing Core Dependencies...
pip install numpy opencv-python Pillow trimesh xatlas transformers accelerate xformers scipy imageio

:: Install SAM-2 for Background Removal
echo Installing SAM-2...
pip install git+https://github.com/facebookresearch/sam2.git

:: Install TRELLIS dependencies (if requirements.txt exists)
if exist "trellis\requirements.txt" (
    echo Installing TRELLIS dependencies...
    pip install -r trellis\requirements.txt
)

:: Install Sparse Convolution for TRELLIS (Required for CUDA backend)
echo Installing spconv (CUDA 11.8 version)...
pip install spconv-cu118

echo ========================================================
echo Setup Complete! 
echo Remember to run "venv\Scripts\activate.bat" before running any scripts.
echo Note: Your GTX 1050ti has 4GB VRAM. Expect slow generations and possible Memory (OOM) errors.
echo ========================================================
pause
