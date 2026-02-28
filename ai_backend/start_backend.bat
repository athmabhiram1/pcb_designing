@echo off
REM AI PCB Assistant - Start Backend Server
REM Run this to start the local AI server

echo Starting AI PCB Backend...
echo.

REM Change to the directory containing this batch file
cd /d "%~dp0"

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Start server
python ai_server.py

pause
