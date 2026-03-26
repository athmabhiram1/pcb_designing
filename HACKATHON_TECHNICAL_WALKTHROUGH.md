# Hackathon Technical Walkthrough

## 1. What This Project Is

This project is a local AI assistant for PCB design that integrates with KiCad.

It has two main parts:

1. `plugin/`
   A KiCad action plugin with the UI and board integration logic.
2. `ai_backend/`
   A local FastAPI backend that handles AI generation, placement, DFM analysis, and KiCad schematic export.

The core idea is simple:

- The user works inside KiCad.
- The plugin reads the live board state.
- It sends requests to a local backend over `http://localhost:8765`.
- The backend returns generated circuits, placement suggestions, DFM issues, or schematic files.

This is a local-first architecture:

- no cloud dependency is required
- no API key is required for the default local workflow
- the system can use Ollama or a local GGUF model

## 2. The Real Architecture in This Repo

For tomorrow, the safest architecture summary is:

```text
KiCad UI Plugin
  -> extracts live PCB data
  -> sends HTTP requests to local FastAPI backend

FastAPI Backend
  -> loads templates
  -> loads optional LLM engine
  -> generates circuit JSON or SKiDL-based netlists
  -> sanitizes and validates circuit data
  -> auto-places components if needed
  -> runs DFM checks
  -> exports .kicad_sch files

KiCad Plugin
  -> imports returned components/nets into the current board
  -> applies placement results
  -> highlights DFM issues
```

Important talking point:

- This repo already implements a working MVP pipeline.
- Some longer planning documents mention future phases like RL training, simulation, and broader platform ideas.
- For the hackathon, present the implemented MVP first, and position RL or deeper automation as roadmap work.

## 3. Main Backend Responsibilities

### App startup

On backend startup in `ai_backend/ai_server.py`, the app:

- creates output and temp directories
- loads circuit templates from `ai_backend/templates/`
- loads an LLM engine if available
- tries to load an optional RL placement model
- exposes capabilities through `/health`

What this means architecturally:

- the backend is modular
- optional features degrade gracefully instead of crashing startup

### Core API endpoints

The main endpoints are:

- `GET /health`
- `GET /templates`
- `POST /generate`
- `POST /generate/stream`
- `POST /generate/schematic`
- `POST /placement/optimize`
- `POST /dfm/check`
- `POST /export/kicad`
- `GET /download/{filename}`

This is important for judges because it shows the system is not just a UI mockup. It has a clear service boundary and a reusable local API.

## 4. Circuit Generation Pipeline

The most important technical flow is `POST /generate`.

The implemented generation priority is:

1. SKiDL via LLM, if enabled and the LLM is available
2. strong template match
3. LLM JSON generation
4. weak template fallback

### Why this matters

This is a reliability strategy, not just an AI call.

- If the model can produce SKiDL, the system gets structured, executable circuit generation.
- If that fails, it can still generate validated JSON.
- If the AI path is unavailable or unreliable, templates provide deterministic fallback behavior.

That gives you a strong hackathon explanation:

"We did not design this as a single-shot LLM demo. We designed it as a resilient generation pipeline with multiple fallback layers."

### SKiDL path

The SKiDL flow uses:

- `services/generation_service.py`
- `services/skidl_runner.py`

The backend:

- asks the model for SKiDL Python
- extracts code from model output
- screens it against blocked imports and unsafe patterns
- runs it in a subprocess with time/resource limits
- parses the generated netlist back into circuit JSON

Security and safety details worth mentioning:

- blocked patterns include `os`, `sys`, `subprocess`, `socket`, `importlib`, `eval`, `exec`, and `open`
- Linux uses resource limits for CPU, file size, and open files
- Windows uses Job Object limits when available, otherwise timeout fallback
- SKiDL execution is isolated in temp run directories

### LLM engine options

The backend supports two model modes in `ai_backend/engines/llm_engine.py`:

- `LLMEngine`
  Local GGUF inference through `llama-cpp-python`
- `OllamaEngine`
  Local or remote Ollama-compatible HTTP generation

Why this is technically strong:

- the project is model-provider flexible
- the higher-level generation pipeline stays the same
- the system can run fully offline

### Template fallback

Templates live in `ai_backend/templates/`.

Current templates include:

- `555_timer`
- `3v3_regulator`
- `led_resistor`
- `mosfet_switch`
- `motor_driver_nmos`
- `opamp_buffer`

These templates are scored against the user prompt with keyword matching. This is not as glamorous as LLM output, but it is reliable and demo-friendly.

## 5. Data Validation and Normalization

This project spends real effort on cleaning AI output before using it.

### Schema layer

`ai_backend/circuit_schema.py` defines strict Pydantic models for:

- components
- pins
- connections
- nets
- design rules
- full circuit data

This gives the system:

- type validation
- normalized references and pin formats
- board-level integrity checks
- a stable internal representation for placement, DFM, and export

### Sanitization layer

`ai_backend/ai_server.py` also sanitizes raw generated data before it becomes `BoardData`.

It handles things like:

- duplicate component references
- malformed connection formats
- conflicting pins
- missing defaults
- circuit guardrails for known topologies

Examples already covered in tests:

- 555 timer wiring guardrails
- AMS1117 regulator decoupling checks

This is a strong point to emphasize:

"The AI output is not trusted blindly. It is normalized, validated, and checked before being imported into KiCad."

## 6. Placement Engine

Placement is implemented in `ai_backend/ai_server.py` through `PlacementOptimizer`.

Supported algorithms:

- force-directed placement
- simulated annealing
- connectivity-based grid fallback

Runtime behavior:

- if NumPy is available, force-directed placement is preferred
- otherwise the backend can use simulated annealing
- if needed, it can fall back to a simpler grid strategy

### Placement logic

The placement engine builds a connectivity graph from nets, then tries to reduce wire length while keeping placements reasonable.

The force-directed approach:

- treats net connections like springs
- adds repulsion to avoid collapsing components together
- iteratively stabilizes positions inside board boundaries

The annealing approach:

- perturbs component positions randomly
- keeps better solutions
- occasionally accepts worse solutions to escape local minima

This gives you a clean explanation:

"We start from circuit connectivity, convert that into a weighted graph, and then run a placement heuristic that minimizes interconnect cost while respecting the board area."

### RL status

The backend can try to load an RL placement model, but the current MVP should be presented as heuristic-first.

Safe phrasing:

- "The architecture is ready for RL-assisted placement later."
- "The current implemented MVP uses practical heuristics with optional RL hooks."

## 7. DFM Engine

DFM analysis is implemented by `AdvancedDFMEngine` in `ai_backend/ai_server.py`.

It performs checks for:

- component spacing
- board boundaries
- orientation sanity
- power integrity
- signal integrity
- thermal issues
- floating components
- estimated net lengths

The DFM engine uses:

- board geometry
- net/component relationships
- a simple spatial index for faster neighbor checks

Why this matters:

- the project is not only generating circuits
- it also evaluates whether the result is manufacturable and layout-aware

This is an important hackathon differentiation point:

"We are not stopping at generation. We added an engineering validation layer so the tool can surface real board issues."

## 8. KiCad Schematic Export

The project can export generated circuits to `.kicad_sch`.

This is handled by `ai_backend/engines/kicad_exporter.py`.

Notable technical choices:

- zero-dependency exporter
- pure Python S-expression generation
- explicit symbol and wire generation
- normalization for KiCad symbol naming edge cases

This is a strong implementation detail because it means the system does not just produce JSON. It can generate a native KiCad schematic artifact.

## 9. KiCad Plugin Responsibilities

The plugin side is split across:

- `plugin/plugin.py`
- `plugin/_legacy_impl.py`
- `plugin/transport/backend_client.py`
- `plugin/board/board_ops.py`

### What the plugin does

The plugin:

- checks backend health
- can auto-start the backend if needed
- opens a wxPython-based frame in KiCad
- extracts live board footprints and nets
- sends requests to the backend
- applies returned placement changes
- imports generated components and nets into the board
- displays DFM results and highlights issues

### Board extraction

The board extraction logic reads:

- references
- values
- footprints
- positions
- rotation
- layers
- pad/net relationships

The plugin sends normalized board data to the backend, including board dimensions.

### Importing generated results

When generation succeeds, the plugin:

- creates missing footprints when possible
- updates existing footprints
- assigns net connections to pads
- rebuilds connectivity
- refreshes the KiCad board

This is an especially good point for a live demo:

"We are not just printing text output. We are taking backend results and mutating the live KiCad board."

## 10. Streaming and UX

The backend also has `POST /generate/stream`, which emits progress events for:

- generation
- placement
- DFM

The orchestration is implemented in `ai_backend/services/generation_orchestrator.py`.

This is good to mention because it shows product thinking:

- not just backend correctness
- also responsiveness and explainability during long-running operations

## 11. Reliability and Testing

The repo includes backend tests under `ai_backend/tests/`.

The tests cover:

- generation-service helpers
- SSE generation flow
- orchestration stages
- template fallback behavior
- LLM engine parsing
- sanitizer behavior
- topology checks
- board module importability without KiCad
- exporter wiring behavior
- runtime config parsing

What you should say honestly:

- "The repo includes automated tests for the critical backend pipeline."
- "I could not execute the tests in this current shell environment because `pytest` and the Python runtime were not usable from the sandbox, so I am describing what is present in the codebase."

That keeps the presentation credible.

## 12. What Is Safe To Claim Tomorrow

You can safely claim:

- This is a KiCad-integrated local AI PCB assistant.
- It uses a plugin plus a local FastAPI backend.
- It supports AI-assisted circuit generation.
- It supports template-backed deterministic generation.
- It can sanitize and validate generated circuit data.
- It can auto-place components with heuristic optimization.
- It can run DFM analysis on board data.
- It can export native KiCad schematic files.
- It can import generated results back into KiCad.

## 13. What To Present As Roadmap, Not Current MVP

Present these as future-facing or optional:

- full production RL placement
- large-scale trained placement models
- full simulation workflow
- full autorouting workflow
- enterprise-scale database/platform features

If asked, say:

"The codebase already has the right modular boundaries for those features, but the hackathon MVP focuses on local generation, placement, DFM, and KiCad integration."

## 14. Best Demo Story

The best demo sequence is:

1. Open KiCad and show the plugin UI.
2. Show that the backend is local and health-checkable.
3. Give a prompt like "Generate a 555 timer LED blinker" or "Generate a 3.3V regulator with caps".
4. Explain the generation pipeline while it runs.
5. Show the imported result and mention the returned generation method.
6. Run placement optimization on the board.
7. Run DFM analysis and show issues or confirmations.
8. Mention that the backend can export a `.kicad_sch` file for downstream use.

## 15. One-Line Technical Summary

This project is a local KiCad plugin plus FastAPI backend that turns natural-language circuit requests into validated circuit data, applies placement heuristics, runs DFM analysis, and exports KiCad-native schematic outputs.
