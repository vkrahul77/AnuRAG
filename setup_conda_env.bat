@echo off
REM AnuRAG Setup Script for Windows
REM Creates conda environment and installs dependencies

echo ============================================================
echo AnuRAG Environment Setup
echo ============================================================

REM Check if conda is available
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Conda is not installed or not in PATH
    echo Please install Anaconda or Miniconda first
    pause
    exit /b 1
)

REM Create conda environment
echo.
echo Creating conda environment 'anurag'...
call conda create -n anurag python=3.11 -y

REM Activate environment
echo.
echo Activating environment...
call conda activate anurag

REM Install PyTorch (CPU version for compatibility)
echo.
echo Installing PyTorch...
call pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

REM Install main requirements
echo.
echo Installing requirements...
call pip install -r requirements.txt

REM Install additional dependencies
echo.
echo Installing additional dependencies...
call pip install google-generativeai
call pip install nltk
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

echo.
echo ============================================================
echo Setup Complete!
echo ============================================================
echo.
echo Next steps:
echo 1. Create a .env file with your GOOGLE_API_KEY
echo 2. Run: conda activate anurag
echo 3. cd gemini\tools
echo 4. python main.py --help
echo.
pause
