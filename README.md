# KiCad AI PCB Plugin

A free, locally-run AI-powered PCB design assistant. Generates schematics, optimizes placement, and checks DFM in real-time.

## Architecture

- **`plugin/`**: KiCad Action Plugin (PCM installable)
- **`ai_backend/`**: FastAPI server with AI models (separate installer)
- **`models/`**: Pre-trained ONNX and GGUF weights

## Quick Start

```bash
# 1. Install the AI Backend
cd ai_backend
pip install -r requirements.txt

# 2. Start the server
python ai_server.py

# 3. Install plugin in KiCad
# Copy `plugin/` folder to:
# Windows: %APPDATA%\KiCad\9.0\scripting\plugins\
# Linux: ~/.local/share/kicad/9.0/scripting/plugins/
```

## Features

- ✅ Natural language to schematic
- ✅ AI-optimized component placement
- ✅ Real-time DFM checking
- ✅ Incremental simulation
