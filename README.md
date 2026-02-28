# AI PCB Assistant for KiCad

An AI-powered PCB design assistant that runs entirely on your local machine, integrated directly into KiCad as an Action Plugin.

**What it does:**
- Generate circuits from natural language descriptions
- Optimize component placement
- Run DFM (Design for Manufacturability) checks
- Export generated circuits as `.kicad_sch` schematic files

**AI is powered by [Ollama](https://ollama.com/) running locally — no cloud, no API keys.**

---

## Repository Structure

```
pcb/
 plugin/             # KiCad Action Plugin (install this into KiCad)
    __init__.py         # Package init — triggers plugin registration
    pcbnew_action.py    # KiCad 9 PCM entry point
    plugin.py           # Full UI + board integration logic
    metadata.json       # KiCad PCM package descriptor

 ai_backend/         # FastAPI server + AI engines
    ai_server.py        # Main FastAPI application
    circuit_schema.py   # Pydantic models / request schemas
    requirements.txt    # Python dependencies
    engines/
       llm_engine.py       # Ollama / GGUF LLM interface
       placement_engine.py # Placement optimization
       dfm_engine.py       # DFM rule checker
       schematic_engine.py # Circuit graph builder
       kicad_exporter.py   # .kicad_sch file generator
    templates/          # Built-in circuit templates
    output/             # Generated files (gitignored)

 models/             # AI model assets (see models/README.md)
 frontend/           # Optional Next.js dashboard (not required)
 build_pcm.ps1       # Script: build versioned PCM ZIP
 deploy_kicad_plugin.ps1  # Script: fast dev deploy to KiCad
 dist/               # Built PCM packages (gitignored)
```

---

## System Requirements

| Requirement | Version |
|-------------|---------|
| KiCad | 9.x |
| Python | 3.10 or newer |
| OS | Windows 10/11 (primary), Linux supported |
| Ollama | Latest — [ollama.com](https://ollama.com/) |

---

## Quick Start

### Step 1 — Install and start Ollama

Download Ollama from [ollama.com](https://ollama.com/) and install it. Then pull the coding model:

```powershell
ollama pull deepseek-coder:6.7b
```

Ollama runs as a background service automatically after install. Verify it is up:

```powershell
Invoke-RestMethod http://localhost:11434/api/tags
```

---

### Step 2 — Set up the backend

```powershell
cd ai_backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Start the backend server:

```powershell
python -m uvicorn ai_server:app --host 0.0.0.0 --port 8765
```

Verify it is healthy:

```powershell
Invoke-RestMethod http://127.0.0.1:8765/health
```

Expected response:

```json
{
  "status": "healthy",
  "version": "2.1.0",
  "llm_loaded": true,
  "templates_available": 5
}
```

---

### Step 3 — Install the KiCad plugin

Copy the entire `plugin/` folder into KiCad's scripting plugins directory and name it `ai_pcb_assistant`:

**Windows:**
```
%APPDATA%\KiCad\9.0\scripting\plugins\ai_pcb_assistant\
```

**Linux:**
```
~/.local/share/kicad/9.0/scripting/plugins/ai_pcb_assistant/
```

Then **fully restart KiCad**.

Or use the deploy script from repo root (Windows only):

```powershell
.\deploy_kicad_plugin.ps1
```

---

### Step 4 — Use the plugin

1. Open a PCB file (`.kicad_pcb`) in KiCad PCB Editor
2. Go to **Tools  External Plugins  AI PCB Assistant Pro**
3. Backend URL defaults to `http://localhost:8765`

| Action | Description |
|--------|-------------|
| Type a prompt  **Execute** | Generate a circuit or get AI placement advice |
| Toolbar **Optimize** | Run placement optimization on the current board |
| Toolbar **DFM Check** | Analyze board for manufacturing issues |
| Toolbar **Generate** | Open circuit generation dialog |

---

## AI Generation Modes

The backend selects the best available mode automatically:

### Mode 1 — Ollama (recommended, used by default)

Backend auto-detects Ollama at `http://localhost:11434` and picks the first available model:

```
deepseek-coder:6.7b   recommended
deepseek-coder:latest
codellama
llama3
mistral
```

Optional environment variable overrides:

```powershell
$env:OLLAMA_API_URL = "http://localhost:11434"   # default
$env:OLLAMA_MODEL   = "deepseek-coder:6.7b"      # force a specific model
```

### Mode 2 — Template-only (always available, no Ollama needed)

Built-in deterministic templates — works offline with zero AI:

| Template | Description |
|----------|-------------|
| 555 Timer | Astable / monostable NE555 |
| LED Resistor | LED with current limiting resistor |
| 3.3V Regulator | LDO from 5V or 12V |
| MOSFET Switch | N-channel low-side switch |
| Op-Amp Buffer | Unity gain buffer |

### Mode 3 — GGUF local model (optional fallback)

If Ollama is not installed, the backend can use a local GGUF file via `llama-cpp-python`:

```
models/deepseek-coder-6.7b-instruct.Q5_K_M.gguf
```

Download link and instructions: `models/README.md`

---

## Backend API Reference

Base URL: `http://127.0.0.1:8765`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Server health and capability status |
| `GET` | `/templates` | List available circuit templates |
| `POST` | `/generate` | Generate a circuit from a text prompt |
| `POST` | `/placement/optimize` | Optimize component placement |
| `POST` | `/dfm/check` | Run DFM analysis |
| `GET` | `/download/{filename}` | Download generated `.kicad_sch` |

### `/generate` request body

```json
{
  "prompt": "555 timer astable LED blinker at 1Hz",
  "priority": "quality"
}
```

`priority` options: `"quality"` | `"speed"` | `"template"`

---

## Development

### Rebuild the PCM ZIP (for distribution)

```powershell
.\build_pcm.ps1
```

Output: `dist/pcm/v{version}/ai-pcb-assistant-pcm-v{version}.zip`

Install this ZIP via KiCad  Plugin and Content Manager  **Install from File**.

### Fast dev deploy (direct copy, skip PCM)

```powershell
.\deploy_kicad_plugin.ps1
```

Restart KiCad after running this.

### Run smoke tests

```powershell
cd ai_backend
python smoke_test.py
```

---

## Frontend (Optional)

A Next.js dashboard is available but not required for the KiCad plugin workflow.

```powershell
cd frontend
npm install
npm run dev
```

Runs at `http://localhost:3000`, targets backend at `http://127.0.0.1:8765`.

---

## Troubleshooting

### Plugin not appearing in KiCad

- Folder must be named exactly `ai_pcb_assistant` inside the plugins directory
- Fully restart KiCad (not just "Refresh Plugins")
- Open the KiCad Scripting Console and run `import ai_pcb_assistant` to see any error

### Plugin window closes immediately

Re-run `.\deploy_kicad_plugin.ps1` then restart KiCad. The root cause (frame garbage-collection) is fixed in the current version.

### Backend not reachable

```powershell
Invoke-RestMethod http://127.0.0.1:8765/health
```

If unreachable, start the backend (see Step 2 above).

### LLM not loaded / generation falls back to templates

```powershell
ollama list          # check what models are available
ollama pull deepseek-coder:6.7b   # pull if missing
```

Backend logs will show `LLM engine: Ollama ready` when connected.

### Port 8765 already in use

```powershell
$env:PORT = "8767"
python ai_server.py
```

Update the URL in plugin Settings to `http://localhost:8767`.

### `onnxruntime not installed` warning

Non-critical — RL placement is just disabled. All other features work. To enable:

```powershell
pip install onnxruntime
```

---

## Branch Strategy

| Branch | Purpose |
|--------|---------|
| `main` | Stable releases |
| `testing` | Active development |
