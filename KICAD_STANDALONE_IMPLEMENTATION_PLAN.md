# KiCad Standalone Integration Plan (Plugin-First)

## Goal
Run everything from inside KiCad with no dependency on a browser UI, while keeping your existing Python plugin and FastAPI backend architecture.

---

## What you already have (verified)

- KiCad Action Plugin entry and registration flow in `plugin/plugin.py`.
- Local backend API at `http://127.0.0.1:8765` with:
  - `POST /generate`
  - `POST /placement/optimize`
  - `POST /dfm/check`
  - `GET /health`
  - `GET /download/{filename}`
- KiCad `.kicad_sch` exporter in backend (`ai_backend/engines/kicad_exporter.py`).

This is already the correct baseline for a KiCad-only product.

---

## WebSocket vs Better Option

## Recommendation
Use **local HTTP over loopback** (`127.0.0.1`) as the primary transport, not WebSocket.

## Why
- Your plugin is request/response driven (generate, optimize, DFM), which maps naturally to HTTP.
- HTTP is simpler to debug, retry, and version.
- KiCad Action Plugins are synchronous UI-driven workflows; streaming is not required for MVP.
- You already implemented HTTP and backend endpoints, so this minimizes risk and rework.

## When to add WebSocket
Only add WebSocket later for optional long-running progress streaming (e.g., LLM step-by-step updates).

---

## Recommended Architecture (Standalone)

1. **KiCad Plugin = UI + Orchestrator**
   - Collect board/schematic context from KiCad API.
   - Start/check backend process if needed.
   - Call backend endpoints over local HTTP.
   - Apply placement/DFM results in editor.

2. **Local Backend Worker**
   - Runs only on localhost.
   - Contains LLM/template generation, placement optimization, DFM logic, KiCad export.

3. **Packaged Runtime**
   - Bundle backend dependencies for Windows distribution (no manual Python setup for end users).

---

## Phase Plan

### Phase 1 — Stabilize plugin-backend contract (now)
- Ensure health-check accepts degraded-but-running backend states.
- Ensure async callback routing is request-type based (not string matching payload).
- Ensure DFM response parser supports list response shape.
- Ensure placement parser supports backend response without `success` wrapper.
- Remove duplicate plugin registration to avoid double-load behavior.

### Phase 2 — Backend lifecycle managed by plugin
- Implement backend launcher in plugin:
  - Start backend if `/health` fails.
  - Wait/retry with timeout + user feedback.
  - Save backend PID/state in plugin config directory.
- Add “Stop Backend” and “Restart Backend” plugin actions.
- Add robust connection error banners with one-click retry.

### Phase 3 — KiCad-only UX flow
- From plugin UI:
  - Generate schematic
  - Save/export `.kicad_sch`
  - Offer “Open output folder” and “Import/Open in KiCad” action.
- Placement + DFM should work directly on current board with undo-safe commits.

### Phase 4 — Distribution and install
- Produce plugin package for KiCad Plugin & Content Manager (PCM).
- Bundle backend as:
  - Option A (recommended): PyInstaller executable + local assets.
  - Option B: managed Python venv installer scripts.
- Provide one-click installer for Windows users.

### Phase 5 — Hardening and observability
- Structured logs for plugin and backend with correlation ID per request.
- Retry policy and timeout tuning for long prompts.
- Add graceful fallback to template mode if LLM unavailable.

### Phase 6 — Optional advanced integration
- Evaluate KiCad IPC API path for deeper editor integration (future).
- Add WebSocket only for progress streaming if needed.

---

## Test Plan (KiCad-native)

1. Plugin launches with KiCad open board.
2. Backend auto-detect and auto-start path works.
3. `Generate` returns valid output and downloadable `.kicad_sch`.
4. `Optimize` updates footprint positions and supports undo.
5. `DFM` highlights components and reports violations.
6. Kill backend manually, then verify plugin auto-recovery flow.
7. Fresh machine install test (no preinstalled Python) for standalone target.

---

## Immediate Next Build Tasks

1. Implement backend process launcher in `plugin/plugin.py` (replace placeholder installer function).
2. Add health-check retry/backoff before opening main frame.
3. Add UI action for opening generated schematic/output directory.
4. Add packaging script for plugin + backend runtime.

---

## Success Criteria

- User installs once, opens KiCad, launches plugin, and completes Generate/Optimize/DFM without opening any external app.
- Plugin remains functional if LLM is unavailable (template fallback).
- No manual terminal commands required for normal users.
