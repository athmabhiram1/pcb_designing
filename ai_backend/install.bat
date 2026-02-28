@echo off
REM AI PCB Assistant - Backend Installer for Windows
REM Run this script to install all dependencies

echo ==========================================
echo    AI PCB Assistant - Backend Installer
echo ==========================================
echo.

REM Check for Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

echo [1/4] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat

echo [3/4] Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo [4/4] Installation complete!
echo.
echo ==========================================
echo To start the AI backend, run:
echo   start_backend.bat
echo ==========================================
echo.
pause
