@echo off
REM MuaLLM-Gemini Run Script for Windows

echo ============================================================
echo MuaLLM-Gemini - Multimodal AI for Analog Circuit Design
echo ============================================================

REM Activate conda environment
call conda activate muallm-gemini

REM Change to tools directory
cd /d "%~dp0gemini\tools"

REM Check for command line arguments
if "%1"=="" (
    echo.
    echo Usage:
    echo   run_muallm.bat process [path]  - Process PDF papers
    echo   run_muallm.bat build           - Build vector database
    echo   run_muallm.bat query "..."     - Ask a question
    echo   run_muallm.bat interactive     - Interactive mode
    echo.
    echo Running in interactive mode...
    python main.py --interactive
) else if "%1"=="process" (
    if "%2"=="" (
        echo Error: Please provide path to papers
        echo Example: run_muallm.bat process "C:\path\to\papers"
        exit /b 1
    )
    python main.py --process_papers "%2"
) else if "%1"=="build" (
    python main.py --build_db
) else if "%1"=="query" (
    python main.py --query "%~2"
) else if "%1"=="interactive" (
    python main.py --interactive
) else (
    echo Unknown command: %1
    echo Use: process, build, query, or interactive
)

pause
