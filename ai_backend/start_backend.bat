@echo off
REM AI PCB Assistant - Start Backend Server
REM Run this to start the local AI server

echo Starting AI PCB Backend...
echo.

REM Change to the directory containing this batch file
cd /d "%~dp0"

REM Activate virtual environment
if exist "venv\Scripts\activate.bat" (
	call venv\Scripts\activate.bat
) else if exist "..\.venv\Scripts\activate.bat" (
	call ..\.venv\Scripts\activate.bat
) else (
	echo WARNING: No virtual environment activate script found. Continuing with current Python.
)

REM Use qwen2.5-coder:7b for backend startup
set "OLLAMA_MODEL=qwen2.5-coder:7b"
echo Using OLLAMA_MODEL=%OLLAMA_MODEL%

REM Start server
python ai_server.py

pause
