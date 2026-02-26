# AI PCB Assistant for KiCad

AI-powered PCB design assistant for KiCad with a local backend.

It supports:
- Natural language circuit generation
- Placement optimization
- DFM checks
- KiCad schematic export (`.kicad_sch`)

This repository contains both:
- A KiCad Action Plugin (`plugin/`)
- A FastAPI backend (`ai_backend/`)

---

## Repository Layout

- `plugin/` → KiCad Action Plugin UI + board integration
- `ai_backend/` → FastAPI server + generation/placement/DFM engines
- `models/` → local AI model assets (GGUF / ONNX)
- `frontend/` → optional dashboard app (Next.js)
- `templates/` (inside backend) → deterministic template-based circuit generation

---

## System Requirements

### Required
- KiCad 9.x
- Python 3.10+
- Windows 10/11 (primary tested path in this repo)

### Optional (for stronger AI generation)
- [Ollama](https://ollama.com/) + a coding model (recommended)
- `llama-cpp-python` + local GGUF model file
- ONNX Runtime model for RL placement (`placement_model.onnx`)

---

## How the AI Modes Work

The backend can run in multiple modes:

1. **Template-only mode (always available)**
	- Uses built-in templates such as 555 timer, regulator, LED resistor, op-amp buffer.
	- No external model process required.

2. **Ollama mode (recommended for LLM generation)**
	- Backend auto-detects Ollama at `http://localhost:11434`.
	- Uses the first matching model from preferred list (`deepseek-coder`, `codellama`, etc.).

3. **GGUF local mode (`llama-cpp-python`)**
	- If Ollama is unavailable, backend can fall back to a GGUF file in `models/`.
	- Default expected filename:
	  - `deepseek-coder-6.7b-instruct.Q5_K_M.gguf`

4. **RL placement model (optional)**
	- If ONNX model is present + ONNX runtime available, backend enables RL placement.
	- Otherwise, placement falls back to analytical/rule-based methods.

---

## Quick Start (Windows)

### 1) Install backend dependencies

From repo root:

```powershell
cd ai_backend
install.bat
```

This creates `ai_backend/venv` and installs requirements.

### 2) Start backend

```powershell
cd ai_backend
start_backend.bat
```

Server default:
- `http://127.0.0.1:8765`

Health check:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8765/health -Method GET
```

### 3) Install plugin in KiCad

Copy the entire `plugin/` folder to:

- Windows: `%APPDATA%\KiCad\9.0\scripting\plugins\`
- Linux: `~/.local/share/kicad/9.0/scripting/plugins/`

Then restart KiCad.

### 4) Open plugin in KiCad

- Open a board in KiCad PCB Editor.
- Launch **AI PCB Assistant Pro** from Action Plugins.
- Backend URL should be `http://localhost:8765` (default).

---

## Model Setup (Detailed)

## A) Ollama setup (recommended)

Install Ollama, then run:

```powershell
ollama serve
ollama pull deepseek-coder:6.7b
```

The backend will auto-detect available Ollama models.

Optional env vars:
- `OLLAMA_API_URL`
- `OLLAMA_TAGS_URL`
- `OLLAMA_MODEL`

## B) GGUF setup (`llama-cpp-python`)

Place GGUF model in `models/` (or set `MODELS_DIR`).

Default model filename expected by backend:
- `deepseek-coder-6.7b-instruct.Q5_K_M.gguf`

Optional env vars:
- `MODELS_DIR`
- `LLM_GGUF_MODEL`

## C) RL placement model (ONNX)

If you have an ONNX placement model, place it as:
- `ai_backend/models/placement_model.onnx`

If missing, backend still works using fallback placement methods.

---

## Running the Optional Frontend

The frontend is optional and not required for KiCad plugin flow.

```powershell
cd frontend
npm install
npm run dev
```

Frontend runs at:
- `http://localhost:3000`

It currently targets backend at:
- `http://127.0.0.1:8765`

---

## API Endpoints (Backend)

Base URL: `http://127.0.0.1:8765`

- `GET /health`
- `GET /templates`
- `POST /generate`
- `POST /placement/optimize`
- `POST /dfm/check`
- `GET /download/{filename}`
- `GET /circuit/{name}`

---

## Validate the Setup

After backend starts:

```powershell
cd ai_backend
python smoke_test.py
```

Expected outcome:
- Health endpoint responds
- Templates are listed
- Generate endpoint returns successful outputs
- Download endpoint returns generated files

---

## Common Troubleshooting

### 1) Port 8765 already in use

Start backend on different port:

```powershell
set PORT=8767
python ai_server.py
```

Then update plugin backend URL to match.

### 2) Plugin opens but actions fail

Check backend is running:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8765/health -Method GET
```

If not running, start `ai_backend/start_backend.bat`.

### 3) LLM not loaded

This is not fatal. Template mode still works.

To enable LLM generation:
- start Ollama + pull a model, or
- add GGUF model and ensure `llama-cpp-python` is installed.

### 4) RL placement not loaded

Also not fatal. Placement uses analytical/rule-based fallback.

To enable RL:
- install ONNX runtime and
- provide `placement_model.onnx` at expected path.

### 5) KiCad plugin not visible

- Verify plugin folder location is correct for your KiCad version
- Restart KiCad after copying files
- Check KiCad plugin manager/logs for import errors

---

## Development Notes

- Backend main file: `ai_backend/ai_server.py`
- Plugin main file: `plugin/plugin.py`
- KiCad exporter: `ai_backend/engines/kicad_exporter.py`
- LLM engine: `ai_backend/engines/llm_engine.py`
- Placement engine: `ai_backend/engines/placement_engine.py`

---

## Recommended First Run Path

1. Install and start backend (`install.bat` → `start_backend.bat`)
2. Verify `GET /health`
3. Install plugin into KiCad plugins directory
4. Open a board and launch plugin
5. Test:
	- Generate
	- DFM check
	- Placement optimize

This gives a fully local KiCad-native workflow.
