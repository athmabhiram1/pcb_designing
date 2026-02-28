# AI PCB Backend — Postman Testing Guide

**Backend version:** 2.0.0  
**Base URL:** `http://127.0.0.1:8765`  
**Total endpoints:** 7  

---

## Table of Contents

1. [Prerequisites & Server Startup](#1-prerequisites--server-startup)
2. [Postman Collection Setup](#2-postman-collection-setup)
3. [GET /health](#3-get-health)
4. [GET /templates](#4-get-templates)
5. [POST /generate](#5-post-generate)
6. [POST /analyze/dfm](#6-post-analyzedfm)
7. [POST /dfm/check](#7-post-dfmcheck)
8. [POST /placement/optimize](#8-post-placementoptimize)
9. [POST /export/kicad](#9-post-exportkicad)
10. [Environment Variables Reference](#10-environment-variables-reference)
11. [Expected Status Codes & Error Reference](#11-expected-status-codes--error-reference)
12. [Full Test Sequence Checklist](#12-full-test-sequence-checklist)

---

## 1. Prerequisites & Server Startup

### 1.1 Install Python dependencies

Open a terminal in the `ai_backend` folder and run:

```powershell
cd "c:\Users\athma\OneDrive\Desktop\my projects\pcb\ai_backend"
pip install -r requirements.txt
```

### 1.2 (Optional) Start Ollama for LLM generation

In a **separate terminal**, keep it running:

```powershell
ollama serve
ollama pull deepseek-coder:6.7b
```

> **If Ollama is not running:** The server still works. `/generate` will fall back to templates instead of LLM output. All other endpoints are unaffected.

### 1.3 Start the backend server

```powershell
cd "c:\Users\athma\OneDrive\Desktop\my projects\pcb\ai_backend"
python ai_server.py
```

**Healthy startup output looks like this:**

```
INFO:     Started server process [XXXXX]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8765 (Press CTRL+C to quit)
```

**If port 8765 is blocked (WinError 10048)**, use a free port:

```powershell
$env:PORT=8767; python ai_server.py
```

Then update `base_url` in Postman to `http://127.0.0.1:8767`.

### 1.4 Install Postman

Download from [https://www.postman.com/downloads](https://www.postman.com/downloads) if not already installed.

---

## 2. Postman Collection Setup

### 2.1 Create the collection

1. Open Postman
2. Click **Collections** (left sidebar) → **+** (New Collection)
3. Name it: `AI PCB Backend v2.0`
4. Click **Create**

### 2.2 Create an Environment

1. Click the **Environments** icon (left sidebar, looks like an eye) → **+**
2. Name it: `PCB Local`
3. Add the following variable:

| Variable | Initial Value | Current Value |
|---|---|---|
| `base_url` | `http://127.0.0.1:8765` | `http://127.0.0.1:8765` |

4. Click **Save**
5. In the top-right corner of Postman, select **PCB Local** from the environment dropdown

### 2.3 How to add a request to the collection

For each endpoint below:
1. In your collection, click the **...** menu → **Add request**
2. Set the method and URL as described
3. Go to the **Body** tab, select **raw**, and set type to **JSON**
4. Paste the request body
5. Click **Save**

---

## 3. GET /health

**Purpose:** Verify the server is alive and check which capabilities are loaded.

### Request

| Field | Value |
|---|---|
| Method | `GET` |
| URL | `{{base_url}}/health` |
| Headers | none required |
| Body | none |

### Steps

1. Create a new request in the collection, name it `Health Check`
2. Set method to `GET`
3. Set URL to `{{base_url}}/health`
4. Click **Send**

### Expected Response — HTTP 200

```json
{
  "status": "degraded",
  "version": "2.0.0",
  "uptime_seconds": 4.1,
  "models_loaded": false,
  "llm_loaded": false,
  "placement_engine_loaded": false,
  "templates_available": 5,
  "capabilities": [
    "basic_dfm",
    "netlist_analysis"
  ]
}
```

### What each field means

| Field | Pass Condition | Notes |
|---|---|---|
| `status` | `"healthy"` or `"degraded"` | `"degraded"` is normal without Ollama |
| `templates_available` | Must be `> 0` | Means template JSON files loaded correctly |
| `capabilities` | Contains `"basic_dfm"` | Minimum required for DFM tests |
| `llm_loaded` | `true` only if Ollama is running | Optional |

### If you see HTTP 000 / connection refused

The server is not running. Go back to [Step 1.3](#13-start-the-backend-server).

---

## 4. GET /templates

**Purpose:** List all built-in circuit templates.

### Request

| Field | Value |
|---|---|
| Method | `GET` |
| URL | `{{base_url}}/templates` |
| Headers | none required |
| Body | none |

### Steps

1. Create a new request, name it `List Templates`
2. Set method to `GET`
3. Set URL to `{{base_url}}/templates`
4. Click **Send**

### Expected Response — HTTP 200

```json
[
  {
    "name": "led_resistor",
    "description": "Simple LED with current-limiting resistor",
    "components": 2,
    "nets": 2
  },
  {
    "name": "555_timer",
    "description": "555 timer astable oscillator",
    "components": 6,
    "nets": 5
  },
  {
    "name": "3v3_regulator",
    "description": "AMS1117-3.3 LDO voltage regulator",
    "components": 4,
    "nets": 3
  },
  {
    "name": "mosfet_switch",
    "description": "NMOS low-side switch",
    "components": 3,
    "nets": 3
  },
  {
    "name": "opamp_buffer",
    "description": "Op-amp unity-gain voltage buffer",
    "components": 3,
    "nets": 4
  }
]
```

> The template names listed here are what `/generate` will return in `"template_used"`.

### Pass condition

The array is non-empty. At least the 5 default templates above should be present.

---

## 5. POST /generate

**Purpose:** Generate a circuit from a plain English description. The server matches keywords to templates, or uses Ollama LLM if available.

### Request

| Field | Value |
|---|---|
| Method | `POST` |
| URL | `{{base_url}}/generate` |
| Content-Type | `application/json` |

### Body

```json
{
  "prompt": "555 timer LED blinker circuit",
  "priority": "quality"
}
```

### Steps

1. Create a new request, name it `Generate Circuit`
2. Set method to `POST`, URL to `{{base_url}}/generate`
3. Click the **Body** tab → select **raw** → set type dropdown to **JSON**
4. Paste the body above
5. Click **Send**

### Expected Response — HTTP 200 (template fallback)

```json
{
  "success": true,
  "circuit_data": {
    "components": [
      {"ref": "U1", "value": "NE555", "footprint": "DIP-8", "x": 50.0, "y": 40.0, ...},
      {"ref": "R1", "value": "10k",   "footprint": "0402",  "x": 20.0, "y": 20.0, ...},
      {"ref": "R2", "value": "100k",  "footprint": "0402",  "x": 30.0, "y": 20.0, ...},
      {"ref": "C1", "value": "10uF",  "footprint": "0805",  "x": 70.0, "y": 20.0, ...},
      {"ref": "C2", "value": "10nF",  "footprint": "0402",  "x": 80.0, "y": 30.0, ...},
      {"ref": "LED1", "value": "RED", "footprint": "0603",  "x": 85.0, "y": 55.0, ...}
    ],
    "connections": [...],
    "board_width": 100.0,
    "board_height": 80.0
  },
  "template_used": "555_timer",
  "generation_time_ms": 38.4,
  "warnings": [],
  "request_id": "a1b2c3d4"
}
```

### Test variations — try different prompts

| Prompt | Expected `template_used` |
|---|---|
| `"simple LED with 330 ohm resistor"` | `led_resistor` |
| `"3.3v LDO power supply AMS1117"` | `3v3_regulator` or `3v3_regulator_ldo` |
| `"MOSFET low side switch"` | `mosfet_switch` |
| `"op-amp voltage buffer unity gain"` | `opamp_buffer` |
| `"555 timer blinker"` | `555_timer` |

### Pass conditions

| Check | Pass Condition |
|---|---|
| HTTP status | `200` |
| `success` field | `true` |
| `circuit_data.components` | Non-empty array |
| `template_used` | Matches expected template from table above |

### If `success` is `false`

- Check `"error"` field in the response
- `"No matching template and LLM unavailable"` — prompt has no matching keywords AND Ollama is off. Try a prompt from the table above.
- `"Validation failed: ..."` — a template file has bad data. Run `python smoke_test.py` (see below).

---

## 6. POST /analyze/dfm

**Purpose:** Run Design for Manufacturability checks on a complete board. Checks component spacing, edge clearances, decoupling, thermal, and more.

### Request

| Field | Value |
|---|---|
| Method | `POST` |
| URL | `{{base_url}}/analyze/dfm` |
| Content-Type | `application/json` |

### Body — Board with intentional spacing violation

R1 and R2 are only **0.1 mm apart** (minimum is 0.5 mm), which should trigger a `DFM-SPC-001` error.

```json
{
  "components": [
    {
      "ref": "R1",
      "value": "10k",
      "footprint": "0402",
      "x": 10.0,
      "y": 10.0,
      "rotation": 0,
      "layer": "top"
    },
    {
      "ref": "R2",
      "value": "1k",
      "footprint": "0402",
      "x": 10.1,
      "y": 10.0,
      "rotation": 0,
      "layer": "top"
    },
    {
      "ref": "U1",
      "value": "NE555",
      "footprint": "DIP-8",
      "x": 50.0,
      "y": 40.0,
      "rotation": 0,
      "layer": "top"
    },
    {
      "ref": "C1",
      "value": "100nF",
      "footprint": "0402",
      "x": 50.5,
      "y": 40.0,
      "rotation": 0,
      "layer": "top"
    },
    {
      "ref": "LED1",
      "value": "RED",
      "footprint": "0603",
      "x": 80.0,
      "y": 60.0,
      "rotation": 0,
      "layer": "top"
    }
  ],
  "connections": [
    {
      "net": "VCC",
      "pins": [
        {"ref": "R1", "pin": "1"},
        {"ref": "U1", "pin": "8"}
      ],
      "properties": {
        "net_type": "power",
        "voltage": 5.0
      }
    },
    {
      "net": "GND",
      "pins": [
        {"ref": "R2", "pin": "2"},
        {"ref": "U1", "pin": "1"},
        {"ref": "LED1", "pin": "K"}
      ],
      "properties": {
        "net_type": "ground"
      }
    },
    {
      "net": "OUT",
      "pins": [
        {"ref": "U1", "pin": "3"},
        {"ref": "LED1", "pin": "A"}
      ],
      "properties": {
        "net_type": "signal"
      }
    },
    {
      "net": "BYPASS",
      "pins": [
        {"ref": "U1", "pin": "5"},
        {"ref": "C1", "pin": "1"}
      ],
      "properties": {
        "net_type": "signal"
      }
    }
  ],
  "board_width": 100.0,
  "board_height": 80.0,
  "design_rules": {
    "min_spacing_mm": 0.5
  }
}
```

### Steps

1. Create a new request, name it `DFM Analysis`
2. Set method to `POST`, URL to `{{base_url}}/analyze/dfm`
3. Click **Body** → **raw** → **JSON**
4. Paste the body above
5. Click **Send**

### Expected Response — HTTP 200

```json
[
  {
    "rule_id": "DFM-SPC-001",
    "type": "spacing",
    "severity": "error",
    "message": "R1 and R2 are 0.1mm apart — minimum spacing is 0.5mm",
    "components": ["R1", "R2"],
    "nets": [],
    "location": {"x": 10.05, "y": 10.0},
    "suggested_fix": "Increase spacing between R1 and R2 to at least 0.5mm",
    "estimated_cost_impact": "low"
  }
]
```

> You may also see `DFM-THM` or `DFM-PWR` violations depending on the engine version.

### Test a clean board (no violations expected)

Use this body where all components are well-spaced:

```json
{
  "components": [
    {"ref": "R1",   "value": "10k",   "footprint": "0402", "x": 10.0, "y": 10.0, "rotation": 0, "layer": "top"},
    {"ref": "R2",   "value": "1k",    "footprint": "0402", "x": 30.0, "y": 10.0, "rotation": 0, "layer": "top"},
    {"ref": "LED1", "value": "RED",   "footprint": "0603", "x": 50.0, "y": 20.0, "rotation": 0, "layer": "top"}
  ],
  "connections": [
    {
      "net": "VCC",
      "pins": [{"ref": "R1", "pin": "1"}, {"ref": "LED1", "pin": "A"}]
    },
    {
      "net": "GND",
      "pins": [{"ref": "R2", "pin": "2"}, {"ref": "LED1", "pin": "K"}]
    },
    {
      "net": "MID",
      "pins": [{"ref": "R1", "pin": "2"}, {"ref": "R2", "pin": "1"}]
    }
  ],
  "board_width": 100.0,
  "board_height": 80.0
}
```

Expected: empty array `[]` or only low-severity `"info"` items.

### Pass conditions

| Check | Pass Condition |
|---|---|
| HTTP status | `200` |
| Response type | JSON array (can be empty) |
| Spacing violation body | At least 1 violation with `"rule_id": "DFM-SPC-001"` |
| Clean board body | Empty array or only `"severity": "info"` items |

---

## 7. POST /dfm/check

**Purpose:** Plugin compatibility alias — identical to `/analyze/dfm`. This is the endpoint called by the KiCad plugin (`plugin.py`).

### Request

| Field | Value |
|---|---|
| Method | `POST` |
| URL | `{{base_url}}/dfm/check` |
| Content-Type | `application/json` |

### Steps

1. Duplicate the **DFM Analysis** request from the previous step (right-click → Duplicate)
2. Rename it to `DFM Check (Plugin Alias)`
3. Change only the URL to `{{base_url}}/dfm/check`
4. Body stays identical
5. Click **Send**

### Expected Response

**Exactly the same** as `/analyze/dfm`. If the two responses differ, that is a bug.

### Pass condition

Response body is byte-for-byte equal to the `/analyze/dfm` response with the same input body.

---

## 8. POST /placement/optimize

**Purpose:** Automatically calculate non-overlapping x/y positions for components. Useful when all components start at `(0, 0)`.

### Request

| Field | Value |
|---|---|
| Method | `POST` |
| URL | `{{base_url}}/placement/optimize?algorithm=force_directed` |
| Content-Type | `application/json` |

> The `algorithm` query parameter is optional. Accepted values: `auto`, `force_directed`, `annealing`, `grid`. Default is `auto`.

### Body — All components stacked at origin

```json
{
  "components": [
    {
      "ref": "U1",
      "value": "NE555",
      "footprint": "DIP-8",
      "x": 0.0,
      "y": 0.0,
      "rotation": 0,
      "layer": "top"
    },
    {
      "ref": "R1",
      "value": "10k",
      "footprint": "0402",
      "x": 0.0,
      "y": 0.0,
      "rotation": 0,
      "layer": "top"
    },
    {
      "ref": "R2",
      "value": "100k",
      "footprint": "0402",
      "x": 0.0,
      "y": 0.0,
      "rotation": 0,
      "layer": "top"
    },
    {
      "ref": "C1",
      "value": "10uF",
      "footprint": "0805",
      "x": 0.0,
      "y": 0.0,
      "rotation": 0,
      "layer": "top"
    },
    {
      "ref": "LED1",
      "value": "RED",
      "footprint": "0603",
      "x": 0.0,
      "y": 0.0,
      "rotation": 0,
      "layer": "top"
    }
  ],
  "connections": [
    {
      "net": "VCC",
      "pins": [
        {"ref": "U1", "pin": "8"},
        {"ref": "R1", "pin": "1"}
      ],
      "properties": {"net_type": "power", "voltage": 5.0}
    },
    {
      "net": "GND",
      "pins": [
        {"ref": "U1", "pin": "1"},
        {"ref": "C1", "pin": "2"},
        {"ref": "LED1", "pin": "K"}
      ],
      "properties": {"net_type": "ground"}
    },
    {
      "net": "DISCHARGE",
      "pins": [
        {"ref": "U1", "pin": "7"},
        {"ref": "R2", "pin": "1"}
      ],
      "properties": {"net_type": "signal"}
    },
    {
      "net": "THRESHOLD",
      "pins": [
        {"ref": "U1", "pin": "2"},
        {"ref": "C1", "pin": "1"}
      ],
      "properties": {"net_type": "signal"}
    },
    {
      "net": "OUT",
      "pins": [
        {"ref": "U1", "pin": "3"},
        {"ref": "LED1", "pin": "A"}
      ],
      "properties": {"net_type": "signal"}
    }
  ],
  "board_width": 100.0,
  "board_height": 80.0
}
```

### Steps

1. Create a new request, name it `Placement Optimize`
2. Set method to `POST`
3. URL: `{{base_url}}/placement/optimize?algorithm=force_directed`
4. **Body** → **raw** → **JSON** → paste the body above
5. Click **Send**

### Alternative algorithm tests

| Query string | What to check |
|---|---|
| `?algorithm=grid` | All positions are on a regular grid |
| `?algorithm=force_directed` | Connected components are placed closer together |
| `?algorithm=annealing` | Best overall wirelength but slower |
| `?algorithm=auto` | Same as `force_directed` (no RL model loaded) |

### Expected Response — HTTP 200

```json
{
  "positions": {
    "U1":   {"x": 42.5, "y": 35.0},
    "R1":   {"x": 15.0, "y": 15.0},
    "R2":   {"x": 25.0, "y": 15.0},
    "C1":   {"x": 60.0, "y": 55.0},
    "LED1": {"x": 75.0, "y": 55.0}
  },
  "algorithm": "force_directed",
  "metrics": {
    "wirelength_mm": 148.3,
    "density_uniformity": 0.74,
    "power_integrity_score": 62.0,
    "thermal_score": 80.0,
    "iterations": 200,
    "convergence_delta": 0.001
  },
  "time_ms": 12.6
}
```

> Exact coordinates will differ on each run (physics simulation). What matters is that coordinates are **not all zero** and components are spread across the board area.

### Pass conditions

| Check | Pass Condition |
|---|---|
| HTTP status | `200` |
| `positions` | All component refs present with non-zero x/y |
| `algorithm` | Matches the query param you sent |
| `time_ms` | Any positive number |

---

## 9. POST /export/kicad

**Purpose:** Export a circuit definition to a `.kicad_sch` schematic file for use in KiCad EDA.

### Request

| Field | Value |
|---|---|
| Method | `POST` |
| URL | `{{base_url}}/export/kicad` |
| Content-Type | `application/json` |

### Body

> Note: This endpoint uses the legacy `CircuitData` format (not `BoardData`). Pins are strings like `"R1.1"` not objects.

```json
{
  "name": "LED Indicator Circuit",
  "components": [
    {
      "ref": "R1",
      "value": "330R",
      "footprint": "Resistor_SMD:R_0402_1005Metric",
      "x": 100.0,
      "y": 100.0
    },
    {
      "ref": "LED1",
      "value": "LED_RED",
      "footprint": "LED_SMD:LED_0603_1608Metric",
      "x": 150.0,
      "y": 100.0
    }
  ],
  "connections": [
    {
      "net": "VCC",
      "pins": ["R1.1"]
    },
    {
      "net": "SIG",
      "pins": ["R1.2", "LED1.A"]
    },
    {
      "net": "GND",
      "pins": ["LED1.K"]
    }
  ]
}
```

### Steps

1. Create a new request, name it `Export KiCad`
2. Set method to `POST`, URL to `{{base_url}}/export/kicad`
3. **Body** → **raw** → **JSON** → paste body above
4. Click the **dropdown arrow** next to **Send** → click **Send and Download**
5. When prompted, save the file as `circuit.kicad_sch`
6. Open the file in a text editor

### Expected Response

- HTTP status: `200`
- Content-Type: `application/x-kicad-schematic`
- File content starts with `(kicad_sch`

```
(kicad_sch
  (version 20230121)
  (generator ai_pcb_assistant)
  ...
)
```

### If you get HTTP 501

The KiCad exporter module is not available. Check that `engines/kicad_exporter.py` exists.

### Pass conditions

| Check | Pass Condition |
|---|---|
| HTTP status | `200` |
| Content-Type header | `application/x-kicad-schematic` |
| File content | Starts with `(kicad_sch` |

---

## 10. Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `base_url` | `http://127.0.0.1:8765` | Backend server base URL |

To change the port (e.g., if you started the server on 8767):

1. Go to **Environments** → **PCB Local**
2. Change `base_url` to `http://127.0.0.1:8767`
3. Click **Save**

All requests immediately pick up the new value.

---

## 11. Expected Status Codes & Error Reference

| Status Code | Meaning | Common Cause |
|---|---|---|
| `200 OK` | Success | Normal response |
| `422 Unprocessable Entity` | Validation failed | Bad request body — check Pydantic model rules below |
| `500 Internal Server Error` | Server crashed | Check terminal logs |
| `501 Not Implemented` | Module missing | `engines/kicad_exporter.py` not found |
| `000 / No response` | Server not running | Run `python ai_server.py` |

### Common 422 causes and fixes

| Error message | Cause | Fix |
|---|---|---|
| `ref: string does not match pattern` | Component ref like `r1` (lowercase) | Use uppercase: `R1` |
| `ref: string does not match pattern` | Ref like `RELAY1` (4+ letter prefix) | Use `RLY1` — max 3 letters before digit |
| `pins: List should have at least 2 items` | Net with only 1 pin | Add at least 2 pins to every net |
| `Net 'X' references unknown component 'Y'` | Pin ref not in components list | Add the component or remove the pin |
| `Duplicate pin in net` | Same pin listed twice in one net | Remove the duplicate |
| `value: min_length = 1` | Empty value string | Provide a value like `"10k"` |

---

## 12. Full Test Sequence Checklist

Run these in order. Each test depends on the server running from Step 1.

| # | Request Name | Method | URL | Pass Condition |
|---|---|---|---|---|
| 1 | Health Check | `GET` | `/health` | `200`, `templates_available > 0` |
| 2 | List Templates | `GET` | `/templates` | `200`, non-empty array |
| 3 | Generate — 555 Timer | `POST` | `/generate` | `success: true`, `template_used: "555_timer"` |
| 4 | Generate — LED | `POST` | `/generate` | `success: true`, `template_used: "led_resistor"` |
| 5 | Generate — 3v3 Regulator | `POST` | `/generate` | `success: true`, `template_used` contains `"3v3"` |
| 6 | DFM — Spacing Violation | `POST` | `/analyze/dfm` | `200`, contains `DFM-SPC-001` violation |
| 7 | DFM — Clean Board | `POST` | `/analyze/dfm` | `200`, empty array or only `"info"` |
| 8 | DFM Check (Plugin Alias) | `POST` | `/dfm/check` | Identical result to test #6 |
| 9 | Placement — Force Directed | `POST` | `/placement/optimize?algorithm=force_directed` | `200`, all positions non-zero |
| 10 | Placement — Grid | `POST` | `/placement/optimize?algorithm=grid` | `200`, positions on a grid pattern |
| 11 | Export KiCad | `POST` | `/export/kicad` | File download, starts with `(kicad_sch` |

---

## Appendix: Smoke Test (Terminal Alternative)

If you want to verify without Postman, run the built-in smoke test:

```powershell
cd "c:\Users\athma\OneDrive\Desktop\my projects\pcb\ai_backend"
python smoke_test.py
```

Expected output:

```
[PASS] /health
[PASS] /templates
[PASS] /generate
[PASS] /analyze/dfm
[PASS] /dfm/check
[PASS] /placement/optimize
All smoke tests passed.
```

---

*Document generated for AI PCB Backend v2.0.0 — February 2026*
