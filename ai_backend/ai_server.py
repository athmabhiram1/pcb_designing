"""
AI PCB Assistant – FastAPI Backend v3.0
Handles LLM inference, netlist-aware placement optimisation, and advanced DFM checking.

Fixes vs v2.1:
  ── CRITICAL ────────────────────────────────────────────────────────────────
  - [BUG] Template preference block ran unconditionally, silently overwriting
    a valid SKiDL result.  Now only activates when circuit_data is still None.
  - [BUG] Diode polarization check had an always-False predicate
    `ref.startswith("D") and not ref.startswith("D")` → diodes never flagged.
  - [BUG] `_execute_skidl_code` created TEMP_DIR subdirs that were never
    deleted → unbounded disk leak.  Background task now cleans up each run dir.
  - [BUG] `AppState.get_capabilities()` never advertised `skidl_generation`
    even when SKiDL was installed and the LLM was loaded.
  - [BUG] `circuit_schema` / `kicad_exporter` imported silently inside every
    request; missing modules produced confusing per-request warnings instead of
    a clear startup log.

  ── SECURITY ─────────────────────────────────────────────────────────────────
  - `_screen_skidl_code` extended to catch `__import__`, `importlib`, and
    attribute-access bypasses (`os.system`, `sys.exit` …).
  - OS-level resource limits (CPU, file-size, open-files) applied to the SKiDL
    subprocess on Linux via `preexec_fn`; Windows falls back to job-object
    timeout only.
  - `subprocess` import moved to module level (was hidden inside a function,
    obscuring the dependency and preventing static analysis).

  ── LOGIC / QUALITY ──────────────────────────────────────────────────────────
  - Generation priority is now explicit and documented:
      1. SKiDL via LLM (if enabled and LLM loaded)
      2. Strong template match (score ≥ 100) as a *fast path*, only when
         SKiDL produced nothing  ← was overwriting SKiDL result before
      3. LLM JSON generation
      4. Weak template fallback
  - `_enrich_component_properties` diode detection fixed and simplified.
  - `_enrich_net_properties` POWER_PATTERNS / GROUND_PATTERNS promoted to
    module-level constants (was recreated on every call).
  - `HealthResponse` now includes `skidl_available` field.
  - `GenerateResponse` now includes `generation_method` field so callers can
    tell which path was taken (skidl / template / llm_json / none).
  - SA acceptance criterion fixed: was only updating `best` on improvement,
    but not accepting worse solutions probabilistically when delta ≥ 0.
    Added proper Metropolis acceptance.
  - `_union_find_groups` uses iterative path compression to avoid Python
    recursion-limit crashes on large boards.
  - Temp-dir cleanup uses `shutil.rmtree` scheduled as a background task.
  - All `import copy` / `import subprocess` moved to top-level.
  - Module-level `_SKIDL`, `_CIRCUIT_SCHEMA`, `_KICAD_EXPORTER` flags set at
    import time; startup logs their status clearly.
"""

from __future__ import annotations

import asyncio
import copy
import importlib
import importlib.util
import json
import logging
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import xml.etree.ElementTree as ET
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Set, Tuple

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from services.generation_service import (
    SkidlRetryConfig,
    default_kicad_symbol_dir as _default_kicad_symbol_dir,
    extract_python_code_block as _extract_python_code_block,
    generate_via_skidl_retry,
    score_template as _score_template,
    screen_skidl_code as _screen_skidl_code,
)
from services.generation_orchestrator import orchestrate
from services.runtime_config import parse_cors_settings

try:
    import resource as _resource  # POSIX only
except ImportError:
    _resource = None  # type: ignore[assignment]

# ── Optional heavy dependencies ───────────────────────────────────────────────

try:
    import numpy as np
    _NP = True
except ImportError:
    np = None  # type: ignore[assignment]
    _NP = False

try:
    import networkx as nx
    _NX = True
except ImportError:
    nx = None  # type: ignore[assignment]
    _NX = False

try:
    import aiofiles
    _AIOFILES = True
except ImportError:
    aiofiles = None  # type: ignore[assignment]
    _AIOFILES = False

_SKIDL = importlib.util.find_spec("skidl") is not None

# KiCad export modules – probed once at import time
_CIRCUIT_SCHEMA_MOD:  Any = None
_KICAD_EXPORTER_MOD:  Any = None
try:
    _CIRCUIT_SCHEMA_MOD  = importlib.import_module("circuit_schema")
    _KICAD_EXPORTER_MOD  = importlib.import_module("engines.kicad_exporter")
    _KICAD_EXPORT_AVAIL  = True
except ImportError:
    _KICAD_EXPORT_AVAIL  = False

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

_BASE_DIR     = Path(__file__).parent
TEMPLATES_DIR = _BASE_DIR / "templates"
OUTPUT_DIR    = _BASE_DIR / "output"
TEMP_DIR      = Path(tempfile.gettempdir()) / "ai_pcb"

# DFM thresholds
DFM_MIN_SPACING_MM         = 0.5
DFM_EDGE_CLEARANCE_MM      = 1.0
DFM_MAX_COMPONENT_HMM      = 25.0
DFM_DECOUPLING_MAX_DIST_MM = 10.0

# Spatial index
SPATIAL_GRID_SIZE_MM = 5.0

# Placement
SA_INITIAL_TEMP  = 100.0
SA_COOLING_RATE  = 0.98
SA_ITERATIONS    = 500

# SKiDL execution
SKIDL_EXEC_TIMEOUT = 40   # seconds per attempt
SKIDL_MAX_RETRIES  = 3

SKIDL_SYSTEM_PROMPT = (
    "You are a circuit design expert. Generate valid SKiDL Python code only.\n"
    "Rules:\n"
    "- Use only real KiCad library parts "
    "(Device, Timer, Regulator_Linear, Amplifier_Operational, "
    "Transistor_FET, Connector, Power)\n"
    "- Every component must define value and footprint where applicable\n"
    "- Include power and ground nets explicitly\n"
    "- Call ERC() then generate_netlist(file_='circuit.net', tool=KICAD9)\n"
    "- Output executable Python only – no markdown, no prose"
)

# ── Net / component enrichment constants ─────────────────────────────────────

_POWER_NET_MAP: Dict[str, Dict[str, Any]] = {
    "VCC":    {"net_type": "power", "voltage": 5.0},
    "VDD":    {"net_type": "power", "voltage": 3.3},
    "3V3":    {"net_type": "power", "voltage": 3.3},
    "3.3V":   {"net_type": "power", "voltage": 3.3},
    "5V":     {"net_type": "power", "voltage": 5.0},
    "1V8":    {"net_type": "power", "voltage": 1.8},
    "1.8V":   {"net_type": "power", "voltage": 1.8},
    "12V":    {"net_type": "power", "voltage": 12.0},
    "24V":    {"net_type": "power", "voltage": 24.0},
    "VPWR":   {"net_type": "power", "voltage": 5.0},
    "VSUP":   {"net_type": "power", "voltage": 5.0},
    "AVCC":   {"net_type": "power", "voltage": 3.3},
    "DVCC":   {"net_type": "power", "voltage": 3.3},
    "VCCIO":  {"net_type": "power", "voltage": 3.3},
    "VCCINT": {"net_type": "power", "voltage": 1.8},
}

_GROUND_NET_PATTERNS = frozenset([
    "GND", "VSS", "AGND", "DGND", "PGND", "SGND", "VEE", "VSSA", "VSSD",
])

_GROUND_PROPS: Dict[str, Any] = {
    "net_type": "ground", "voltage": None, "current_max": None,
    "frequency": None, "impedance_ohms": None, "length_mm": None,
    "is_critical": False,
}

# ── Template keyword registry ─────────────────────────────────────────────────
# Tuple: (keyword_list, template_stem, score_weight)
# STRONG_TEMPLATE_THRESHOLD: score at which we prefer a template over
# waiting for SKiDL (used only when circuit_data is still None).

STRONG_TEMPLATE_THRESHOLD = 100

TEMPLATE_KEYWORDS: List[Tuple[List[str], str, int]] = [
    (["motor driver", "dc motor", "pwm motor", "flyback diode"], "motor_driver_nmos", 130),
    (["555 timer", "ne555", "astable 555", "555 oscillator"],    "555_timer",          110),
    (["555", "timer", "blink", "astable", "multivibrator"],      "555_timer",           80),
    (["3.3v regulator", "3v3 ldo", "ams1117-3.3"],               "3v3_regulator",      110),
    (["3.3v", "3v3", "ams1117", "ldo", "voltage regulator"],     "3v3_regulator",       80),
    (["led", "diode", "indicator", "resistor led"],               "led_resistor",        80),
    (["opamp buffer", "unity gain buffer", "voltage follower"],   "opamp_buffer",       100),
    (["opamp", "op-amp", "operational amplifier", "gain"],        "opamp_buffer",        70),
    (["mosfet switch", "high side switch", "low side switch"],    "mosfet_switch",      110),
    (["mosfet", "nmos", "pmos", "transistor switch"],             "mosfet_switch",       80),
]

_POWER_SYMBOL_RE = re.compile(r'^#(PWR|FLG|GND|VCC|VDD)\d*$', re.IGNORECASE)

# ── Pydantic models ───────────────────────────────────────────────────────────

class PinRef(BaseModel):
    model_config = ConfigDict(frozen=True)
    ref: str = Field(..., min_length=1, max_length=32)
    pin: str = Field(..., min_length=1, max_length=16)

    def __str__(self) -> str:
        return f"{self.ref}.{self.pin}"

    @property
    def component_prefix(self) -> str:
        m = re.match(r'^#?([A-Z]+)', self.ref)
        return m.group(1) if m else "U"


class NetProperties(BaseModel):
    net_type: Literal["power", "ground", "signal", "clock", "differential", "analog"] = "signal"
    voltage:        Optional[float] = None
    current_max:    Optional[float] = None
    frequency:      Optional[float] = None
    impedance_ohms: Optional[float] = None
    length_mm:      Optional[float] = None
    is_critical:    bool = False


class BoardConnection(BaseModel):
    model_config = ConfigDict(frozen=True)
    net:        str            = Field(..., min_length=1, max_length=100)
    pins:       List[PinRef]   = Field(..., min_length=2)
    properties: NetProperties  = Field(default_factory=NetProperties)

    @field_validator("pins")
    @classmethod
    def validate_unique_pins(cls, v: List[PinRef]) -> List[PinRef]:
        seen: Set[str] = set()
        for pin in v:
            key = str(pin)
            if key in seen:
                raise ValueError(f"Duplicate pin: {key}")
            seen.add(key)
        return v

    @property
    def components(self) -> Set[str]:
        return {p.ref for p in self.pins}

    def has_component(self, ref: str) -> bool:
        return any(p.ref == ref for p in self.pins)


class ComponentData(BaseModel):
    """
    model_config frozen=False: PlacementOptimizer mutates x/y in-place after
    auto-placement without reconstructing the whole object.
    """
    model_config = ConfigDict(frozen=False)

    ref:       str   = Field(..., min_length=1, max_length=32)
    value:     str   = Field(..., min_length=1)
    footprint: str   = Field(default="")
    x:         float = Field(default=0.0, ge=-1000, le=1000)
    y:         float = Field(default=0.0, ge=-1000, le=1000)
    rotation:  float = Field(default=0.0, ge=-360, le=360)
    layer:     str   = Field(default="top")
    height_mm:            Optional[float] = Field(default=None, ge=0, le=50)
    power_dissipation_mw: Optional[float] = Field(default=None, ge=0)
    is_polarized:         bool            = False

    @model_validator(mode="before")
    @classmethod
    def flatten_position(cls, values: Any) -> Any:
        if isinstance(values, dict):
            if "position" in values and "x" not in values:
                pos = values["position"]
                if isinstance(pos, dict):
                    values = {**values, "x": pos.get("x", 0.0), "y": pos.get("y", 0.0)}
            values.setdefault("x", 0.0)
            values.setdefault("y", 0.0)
        return values

    @field_validator("layer")
    @classmethod
    def normalize_layer(cls, v: str) -> str:
        return {"F.Cu": "top", "B.Cu": "bottom"}.get(v, v if v in ("top", "bottom") else "top")

    @property
    def prefix(self) -> str:
        m = re.match(r'^#?([A-Z]+)', self.ref)
        return m.group(1) if m else "U"

    @property
    def is_ic(self) -> bool:
        return self.prefix in {"U", "IC"}

    @property
    def is_passive(self) -> bool:
        return self.prefix in {"R", "C", "L", "F"}

    @property
    def is_connector(self) -> bool:
        return self.prefix in {"J", "P", "CONN"}

    @property
    def is_power_symbol(self) -> bool:
        return bool(_POWER_SYMBOL_RE.match(self.ref))


class BoardData(BaseModel):
    components:   List[ComponentData]   = Field(default_factory=list)
    connections:  List[BoardConnection] = Field(default_factory=list)
    board_width:  float = Field(default=100.0, gt=0, le=1000)
    board_height: float = Field(default=80.0,  gt=0, le=1000)
    design_rules: Dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def soft_validate_references(self) -> "BoardData":
        comp_refs = {c.ref for c in self.components}
        for conn in self.connections:
            for pin in conn.pins:
                if pin.ref not in comp_refs and not _POWER_SYMBOL_RE.match(pin.ref):
                    logger.warning(
                        "Net '%s' references unknown component '%s' – skipped",
                        conn.net, pin.ref,
                    )
        return self

    def get_component(self, ref: str) -> Optional[ComponentData]:
        for c in self.components:
            if c.ref == ref:
                return c
        return None

    def get_nets_for_component(self, ref: str) -> List[BoardConnection]:
        return [c for c in self.connections if c.has_component(ref)]

    def build_graph(self) -> Optional[Any]:
        if not _NX:
            return None
        G = nx.Graph()
        for comp in self.components:
            G.add_node(comp.ref, data=comp)
        for conn in self.connections:
            for i, p1 in enumerate(conn.pins):
                for p2 in conn.pins[i + 1:]:
                    G.add_edge(p1.ref, p2.ref, net=conn.net)
        return G


class DFMViolation(BaseModel):
    rule_id:   str = Field(..., pattern=r'^DFM-[A-Z]{2,4}-\d{3}$')
    type:      str
    severity:  Literal["info", "warning", "error", "critical"]
    message:   str
    components: List[str]             = Field(default_factory=list)
    nets:       List[str]             = Field(default_factory=list)
    location:   Optional[Dict[str, float]] = None
    suggested_fix:         Optional[str]                       = None
    estimated_cost_impact: Optional[Literal["low","medium","high"]] = None


class HealthResponse(BaseModel):
    status:                  str
    version:                 str
    uptime_seconds:          float
    models_loaded:           bool
    llm_loaded:              bool  = False
    placement_engine_loaded: bool  = False
    skidl_available:         bool  = False   # NEW
    kicad_export_available:  bool  = False   # NEW
    templates_available:     int   = 0
    llm_provider:            Optional[str] = None
    llm_model:               Optional[str] = None
    capabilities:            List[str] = Field(default_factory=list)


class GenerateRequest(BaseModel):
    prompt:      str  = Field(..., min_length=1, max_length=5000)
    constraints: Optional[Dict[str, Any]] = None
    priority:    Literal["speed", "quality", "compact"] = "quality"


class GenerateResponse(BaseModel):
    success:            bool
    circuit_data:       Optional[Dict[str, Any]] = None
    template_used:      Optional[str]  = None
    generation_method:  Optional[str]  = None   # "skidl" | "template" | "llm_json" | None
    generation_time_ms: float          = 0.0
    warnings:           List[str]      = Field(default_factory=list)
    error:              Optional[str]  = None
    request_id:         str            = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    output_file:        Optional[str]  = None
    download_url:       Optional[str]  = None


class SchematicRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=5000)


class SchematicResponse(BaseModel):
    success:         bool
    component_count: int          = 0
    output_file:     Optional[str] = None
    download_url:    Optional[str] = None
    error:           Optional[str] = None


# ── Application State ─────────────────────────────────────────────────────────

class AppState:
    def __init__(self) -> None:
        self.llm:            Any             = None
        self.rl_model:       Any             = None
        self.template_cache: Dict[str, Dict] = {}
        self.start_time:     float           = time.time()

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.start_time

    @property
    def models_loaded(self) -> bool:
        return self.llm is not None or self.rl_model is not None

    def get_capabilities(self) -> List[str]:
        caps = ["basic_dfm", "netlist_analysis", "template_matching"]
        if self.llm:
            caps.append("llm_generation")
            if _SKIDL:
                caps.append("skidl_generation")   # FIXED: was never added
        if self.rl_model:
            caps.append("rl_placement")
        if _NX:
            caps.append("graph_analysis")
        if _NP:
            caps.append("force_directed_placement")
        if _KICAD_EXPORT_AVAIL:
            caps.append("kicad_sch_export")
        return caps


_state = AppState()


# ── Spatial Index ─────────────────────────────────────────────────────────────

class SpatialIndex:
    def __init__(self, cell_size: float = SPATIAL_GRID_SIZE_MM) -> None:
        self.cell_size = cell_size
        self.grid:            Dict[Tuple[int,int], List[ComponentData]] = defaultdict(list)
        self.component_cells: Dict[str, Tuple[int,int]]                = {}

    def _cell(self, x: float, y: float) -> Tuple[int, int]:
        return int(x / self.cell_size), int(y / self.cell_size)

    def insert(self, comp: ComponentData) -> None:
        cell = self._cell(comp.x, comp.y)
        self.grid[cell].append(comp)
        self.component_cells[comp.ref] = cell

    def query_neighbors(
        self, comp: ComponentData, radius: float
    ) -> List[Tuple[ComponentData, float]]:
        cell_r = int(radius / self.cell_size) + 1
        cx, cy = self.component_cells.get(comp.ref, self._cell(comp.x, comp.y))
        result: List[Tuple[ComponentData, float]] = []
        for dx in range(-cell_r, cell_r + 1):
            for dy in range(-cell_r, cell_r + 1):
                for other in self.grid.get((cx+dx, cy+dy), []):
                    if other.ref == comp.ref:
                        continue
                    d = math.hypot(comp.x - other.x, comp.y - other.y)
                    if d <= radius:
                        result.append((other, d))
        return result


# ── DFM Engine ────────────────────────────────────────────────────────────────

class AdvancedDFMEngine:
    POWER_NETS: Set[str] = {
        "VCC","VDD","3V3","3.3V","5V","1V8","1.8V","12V","24V",
        "VPWR","VSUP","AVCC","DVCC","VCCIO","VCCINT",
    }
    GROUND_NETS: Set[str] = {
        "GND","VSS","AGND","DGND","PGND","SGND","VEE","VSSA","VSSD",
    }
    STANDARD_ANGLES: Set[float] = {0.0, 90.0, 180.0, 270.0}
    ANGLE_TOL = 0.5

    def __init__(self, board: BoardData) -> None:
        self.board      = board
        self.violations: List[DFMViolation] = []
        self.spatial    = SpatialIndex()
        for c in board.components:
            if not c.is_power_symbol:
                self.spatial.insert(c)
        self.net_pins:       Dict[str, List[PinRef]] = {}
        self.component_nets: Dict[str, List[str]]    = defaultdict(list)
        for conn in board.connections:
            self.net_pins[conn.net] = list(conn.pins)
            for pin in conn.pins:
                self.component_nets[pin.ref].append(conn.net)

    def analyze(self) -> List[DFMViolation]:
        self._check_component_spacing()
        self._check_board_boundaries()
        self._check_orientation()
        self._check_power_integrity()
        self._check_signal_integrity()
        self._check_thermal()
        self._check_floating_components()
        self._check_net_lengths()
        SEV_ORDER = {"critical":0,"error":1,"warning":2,"info":3}
        return sorted(self.violations, key=lambda v: SEV_ORDER.get(v.severity, 4))

    # ── Spacing ───────────────────────────────────────────────────────────────

    def _check_component_spacing(self) -> None:
        checked: Set[Tuple[str,str]] = set()
        for comp in self.board.components:
            if comp.is_power_symbol:
                continue
            for other, dist in self.spatial.query_neighbors(comp, DFM_MIN_SPACING_MM * 3):
                pair: Tuple[str,str] = tuple(sorted((comp.ref, other.ref)))  # type: ignore[assignment]
                if pair in checked or self._are_connected(comp.ref, other.ref):
                    continue
                checked.add(pair)
                req = self._required_spacing(comp, other)
                if dist < req:
                    self.violations.append(DFMViolation(
                        rule_id="DFM-SPC-001", type="component_spacing",
                        severity="error" if dist < req * 0.5 else "warning",
                        message=f"{comp.ref}↔{other.ref}: {dist:.2f} mm < {req:.1f} mm required",
                        components=[comp.ref, other.ref],
                        location={"x": (comp.x+other.x)/2, "y": (comp.y+other.y)/2},
                        suggested_fix=f"Increase separation by {req-dist:.1f} mm",
                        estimated_cost_impact="high" if dist < req * 0.5 else "medium",
                    ))

    def _required_spacing(self, c1: ComponentData, c2: ComponentData) -> float:
        base = DFM_MIN_SPACING_MM
        for c in (c1, c2):
            if c.power_dissipation_mw and c.power_dissipation_mw > 1000: base += 1.0
            if c.height_mm and c.height_mm > 10:                          base += 2.0
            if c.is_connector:                                             base += 1.5
        return base

    # ── Boundary ──────────────────────────────────────────────────────────────

    def _check_board_boundaries(self) -> None:
        for comp in self.board.components:
            if comp.is_power_symbol:
                continue
            bb = self._bounding_box(comp)
            for violated, edge, overflow in [
                (bb["x"] < DFM_EDGE_CLEARANCE_MM,
                 "left",  DFM_EDGE_CLEARANCE_MM - bb["x"]),
                (bb["x"]+bb["w"] > self.board.board_width - DFM_EDGE_CLEARANCE_MM,
                 "right", bb["x"]+bb["w"] - (self.board.board_width - DFM_EDGE_CLEARANCE_MM)),
                (bb["y"] < DFM_EDGE_CLEARANCE_MM,
                 "bottom", DFM_EDGE_CLEARANCE_MM - bb["y"]),
                (bb["y"]+bb["h"] > self.board.board_height - DFM_EDGE_CLEARANCE_MM,
                 "top",   bb["y"]+bb["h"] - (self.board.board_height - DFM_EDGE_CLEARANCE_MM)),
            ]:
                if violated:
                    self.violations.append(DFMViolation(
                        rule_id="DFM-BND-001", type="board_boundary", severity="error",
                        message=f"{comp.ref} violates {edge} edge by {overflow:.1f} mm",
                        components=[comp.ref], location={"x": comp.x, "y": comp.y},
                        suggested_fix=f"Move {comp.ref} inward ≥{overflow+0.5:.1f} mm",
                    ))

    # ── Orientation ───────────────────────────────────────────────────────────

    def _check_orientation(self) -> None:
        for comp in self.board.components:
            if not comp.is_polarized or comp.is_power_symbol:
                continue
            norm = comp.rotation % 360
            ok = any(
                abs(norm - a) < self.ANGLE_TOL or abs(norm - a - 360) < self.ANGLE_TOL
                for a in self.STANDARD_ANGLES
            )
            if not ok:
                self.violations.append(DFMViolation(
                    rule_id="DFM-ORI-001", type="orientation", severity="warning",
                    message=f"{comp.ref} polarised at non-standard {comp.rotation:.1f}°",
                    components=[comp.ref], location={"x": comp.x, "y": comp.y},
                    suggested_fix=f"Snap {comp.ref} to 0°, 90°, 180°, or 270°",
                ))

    # ── Power integrity ───────────────────────────────────────────────────────

    def _check_power_integrity(self) -> None:
        for ic in self.board.components:
            if not ic.is_ic:
                continue
            ic_nets     = set(self.component_nets.get(ic.ref, []))
            power_nets  = ic_nets & self.POWER_NETS
            ground_nets = ic_nets & self.GROUND_NETS
            if not power_nets:
                self.violations.append(DFMViolation(
                    rule_id="DFM-PWR-001", type="power_connection", severity="error",
                    message=f"{ic.ref} has no power supply net",
                    components=[ic.ref], location={"x": ic.x, "y": ic.y},
                ))
                continue
            if not ground_nets:
                self.violations.append(DFMViolation(
                    rule_id="DFM-PWR-002", type="ground_connection", severity="error",
                    message=f"{ic.ref} has no ground net",
                    components=[ic.ref], location={"x": ic.x, "y": ic.y},
                ))
                continue
            for pwr_net in power_nets:
                self._check_decoupling(ic, pwr_net, ground_nets)

    def _check_decoupling(
        self, ic: ComponentData, pwr_net: str, gnd_nets: Set[str]
    ) -> None:
        candidates: List[Tuple[ComponentData, float]] = [
            (c, math.hypot(c.x - ic.x, c.y - ic.y))
            for c in self.board.components
            if c.ref.startswith("C")
            and pwr_net in set(self.component_nets.get(c.ref, []))
            and set(self.component_nets.get(c.ref, [])) & gnd_nets
        ]
        if not candidates:
            self.violations.append(DFMViolation(
                rule_id="DFM-PWR-003", type="missing_decoupling", severity="warning",
                message=f"{ic.ref}: no decoupling cap on net {pwr_net}",
                components=[ic.ref], nets=[pwr_net], location={"x": ic.x, "y": ic.y},
                suggested_fix=f"Add 100 nF ceramic cap within {DFM_DECOUPLING_MAX_DIST_MM} mm",
            ))
            return
        closest_cap, closest_dist = min(candidates, key=lambda t: t[1])
        if closest_dist > DFM_DECOUPLING_MAX_DIST_MM:
            self.violations.append(DFMViolation(
                rule_id="DFM-PWR-004", type="decoupling_distance", severity="warning",
                message=(
                    f"{closest_cap.ref} for {ic.ref} is {closest_dist:.1f} mm "
                    f"away (limit {DFM_DECOUPLING_MAX_DIST_MM} mm)"
                ),
                components=[ic.ref, closest_cap.ref], nets=[pwr_net],
                location={"x": ic.x, "y": ic.y},
                suggested_fix=f"Move {closest_cap.ref} closer to {ic.ref}",
            ))
        if ic.power_dissipation_mw and ic.power_dissipation_mw > 500:
            bulk = [c for c, _ in candidates
                    if c.value and any(u in c.value.upper() for u in ("UF","µF","MF"))]
            if not bulk:
                self.violations.append(DFMViolation(
                    rule_id="DFM-PWR-005", type="missing_bulk_capacitance", severity="info",
                    message=f"High-power {ic.ref} may need bulk capacitance (>1 µF)",
                    components=[ic.ref], nets=[pwr_net],
                    suggested_fix="Add 10 µF electrolytic/tantalum nearby",
                ))

    # ── Signal integrity ──────────────────────────────────────────────────────

    def _check_signal_integrity(self) -> None:
        for conn in self.board.connections:
            props = conn.properties
            if props.frequency and props.frequency > 1e6 and props.net_type == "clock":
                if len(conn.pins) > 3:
                    self.violations.append(DFMViolation(
                        rule_id="DFM-SI-001", type="clock_fanout", severity="warning",
                        message=f"Clock '{conn.net}' has {len(conn.pins)} loads",
                        nets=[conn.net],
                        suggested_fix="Add a clock buffer",
                    ))
            if props.net_type == "differential" and len(conn.pins) > 4:
                self.violations.append(DFMViolation(
                    rule_id="DFM-SI-002", type="differential_pair_stub", severity="warning",
                    message=f"Diff net '{conn.net}' has {len(conn.pins)} pins – stubs degrade SI",
                    nets=[conn.net],
                    suggested_fix="Route as matched-length pairs",
                ))

    # ── Thermal ───────────────────────────────────────────────────────────────

    def _check_thermal(self) -> None:
        for comp in self.board.components:
            if not (comp.power_dissipation_mw and comp.power_dissipation_mw > 500):
                continue
            if len(self.spatial.query_neighbors(comp, 5.0)) < 4:
                self.violations.append(DFMViolation(
                    rule_id="DFM-THM-001", type="thermal_management", severity="warning",
                    message=(
                        f"{comp.ref} dissipates {comp.power_dissipation_mw:.0f} mW "
                        "– insufficient copper area nearby"
                    ),
                    components=[comp.ref], location={"x": comp.x, "y": comp.y},
                    suggested_fix="Add thermal vias, copper pour, or heatsink",
                ))

    # ── Floating components ───────────────────────────────────────────────────

    def _check_floating_components(self) -> None:
        connected = {p.ref for conn in self.board.connections for p in conn.pins}
        for comp in self.board.components:
            if comp.is_ic and comp.ref not in connected and not comp.is_power_symbol:
                self.violations.append(DFMViolation(
                    rule_id="DFM-CNN-001", type="floating_component", severity="error",
                    message=f"{comp.ref} ({comp.value}) has no connections",
                    components=[comp.ref], location={"x": comp.x, "y": comp.y},
                    suggested_fix="Connect all pins or remove the component",
                ))

    # ── Net length ────────────────────────────────────────────────────────────

    def _check_net_lengths(self) -> None:
        for conn in self.board.connections:
            if len(conn.pins) < 2 or not conn.properties.length_mm:
                continue
            max_dist = 0.0
            for i, p1 in enumerate(conn.pins):
                c1 = self.board.get_component(p1.ref)
                if not c1:
                    continue
                for p2 in conn.pins[i+1:]:
                    c2 = self.board.get_component(p2.ref)
                    if not c2:
                        continue
                    max_dist = max(max_dist, abs(c1.x-c2.x) + abs(c1.y-c2.y))
            if max_dist > conn.properties.length_mm * 1.5:
                self.violations.append(DFMViolation(
                    rule_id="DFM-LEN-001", type="excessive_trace_length", severity="warning",
                    message=(
                        f"Net '{conn.net}' estimated {max_dist:.1f} mm "
                        f"> target {conn.properties.length_mm:.1f} mm"
                    ),
                    nets=[conn.net],
                    suggested_fix="Move components closer or add termination",
                ))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _are_connected(self, r1: str, r2: str) -> bool:
        return bool(
            set(self.component_nets.get(r1, []))
            & set(self.component_nets.get(r2, []))
        )

    def _bounding_box(self, comp: ComponentData) -> Dict[str, float]:
        sizes = {"R":1.6,"C":1.6,"L":2.0,"D":2.0,"U":5.0,
                 "Q":3.0,"J":10.0,"SW":6.0,"TP":2.0}
        s = sizes.get(comp.prefix, 5.0)
        if comp.rotation % 180 != 0:
            s *= 1.4
        return {"x": comp.x - s/2, "y": comp.y - s/2, "w": s, "h": s}


# ── Placement Optimiser ───────────────────────────────────────────────────────

class PlacementOptimizer:
    def __init__(self, board: BoardData) -> None:
        self.board = board
        self.graph = board.build_graph()
        self.adj: Dict[str, Dict[str, float]] = defaultdict(dict)
        for conn in board.connections:
            refs = [p.ref for p in conn.pins if not _POWER_SYMBOL_RE.match(p.ref)]
            w = 1.0 / max(len(refs), 1)
            for i, r1 in enumerate(refs):
                for r2 in refs[i+1:]:
                    self.adj[r1][r2] = self.adj[r1].get(r2, 0) + w
                    self.adj[r2][r1] = self.adj[r2].get(r1, 0) + w

    def optimize(self, algorithm: str = "force_directed") -> Dict[str, Any]:
        if algorithm == "force_directed" and _NP:
            return self._force_directed()
        if algorithm == "annealing":
            return self._simulated_annealing()
        return self._grid_placement()

    def _wire_length(self, positions: Dict[str, Tuple[float,float]]) -> float:
        total = 0.0
        for r1, nbrs in self.adj.items():
            if r1 not in positions:
                continue
            for r2, w in nbrs.items():
                if r2 not in positions:
                    continue
                total += w * math.hypot(
                    positions[r1][0]-positions[r2][0],
                    positions[r1][1]-positions[r2][1],
                )
        return total / 2

    def _force_directed(self) -> Dict[str, Any]:
        w, h = self.board.board_width, self.board.board_height
        pos: Dict[str, Any] = {
            c.ref: np.array([
                c.x if c.x else random.uniform(10, w-10),
                c.y if c.y else random.uniform(10, h-10),
            ])
            for c in self.board.components if not c.is_power_symbol
        }
        spring_k, repulse_k = 0.08, 150.0
        for iteration in range(120):
            forces  = {r: np.zeros(2) for r in pos}
            damping = max(0.05, 0.9 - iteration / 150)
            for r1, nbrs in self.adj.items():
                if r1 not in pos:
                    continue
                for r2, wt in nbrs.items():
                    if r2 not in pos:
                        continue
                    diff  = pos[r2] - pos[r1]
                    dist  = float(np.linalg.norm(diff)) or 1e-6
                    ideal = 10.0 * (1 - wt)
                    f     = diff / dist * spring_k * (dist - ideal)
                    forces[r1] += f
                    forces[r2] -= f
            refs_list = list(pos.keys())
            for i, r1 in enumerate(refs_list):
                for r2 in refs_list[i+1:]:
                    diff = pos[r2] - pos[r1]
                    dist = float(np.linalg.norm(diff)) or 1e-6
                    if dist < 30:
                        f = -diff / dist * repulse_k / dist
                        forces[r1] += f
                        forces[r2] -= f
            for ref in pos:
                pos[ref] += forces[ref] * damping
                pos[ref][0] = float(np.clip(pos[ref][0], 5, w-5))
                pos[ref][1] = float(np.clip(pos[ref][1], 5, h-5))
        return {
            "positions": {
                ref: {"x": float(p[0]), "y": float(p[1]), "rotation": 0.0}
                for ref, p in pos.items()
            },
            "algorithm": "force_directed", "iterations": 120,
        }

    def _simulated_annealing(self) -> Dict[str, Any]:
        w, h = self.board.board_width, self.board.board_height
        cur: Dict[str, Tuple[float,float]] = {
            c.ref: (
                c.x if c.x else random.uniform(5, w-5),
                c.y if c.y else random.uniform(5, h-5),
            )
            for c in self.board.components if not c.is_power_symbol
        }
        best      = dict(cur)
        best_cost = self._wire_length(best)
        temp      = SA_INITIAL_TEMP

        for _ in range(SA_ITERATIONS):
            ref  = random.choice(list(cur.keys()))
            ox, oy = cur[ref]
            step = temp * 0.3
            nx   = max(5, min(w-5, ox + random.uniform(-step, step)))
            ny   = max(5, min(h-5, oy + random.uniform(-step, step)))
            cur[ref] = (nx, ny)

            cost  = self._wire_length(cur)
            delta = cost - best_cost

            # FIXED: proper Metropolis acceptance — accept improvements always,
            # accept worse solutions with probability exp(-delta/T)
            if delta < 0:
                if cost < best_cost:
                    best      = dict(cur)
                    best_cost = cost
            elif random.random() < math.exp(-delta / max(temp, 1e-6)):
                pass   # accept worse solution — don't revert
            else:
                cur[ref] = (ox, oy)   # revert

            temp *= SA_COOLING_RATE

        return {
            "positions": {
                ref: {"x": x, "y": y, "rotation": 0.0}
                for ref, (x, y) in best.items()
            },
            "algorithm": "simulated_annealing",
            "iterations": SA_ITERATIONS,
            "final_cost": best_cost,
        }

    def _grid_placement(self) -> Dict[str, Any]:
        groups     = self._union_find_groups()
        positions: Dict[str, Dict[str,float]] = {}
        grid_step, margin, cols = 10.0, 10.0, 5
        global_col = 0
        for group in groups:
            for i, ref in enumerate(group):
                positions[ref] = {
                    "x": margin + (global_col + i % cols) * grid_step,
                    "y": margin + (i // cols) * grid_step,
                    "rotation": 0.0,
                }
            global_col += max(len(group) % cols or cols, 1) + 1
        return {"positions": positions, "algorithm": "connectivity_grid", "iterations": 1}

    def _union_find_groups(self) -> List[List[str]]:
        parent = {c.ref: c.ref for c in self.board.components if not c.is_power_symbol}

        def find(x: str) -> str:
            # FIXED: iterative path compression avoids RecursionError on large boards
            root = x
            while parent[root] != root:
                root = parent[root]
            while parent[x] != root:
                parent[x], x = root, parent[x]
            return root

        for conn in self.board.connections:
            refs = [p.ref for p in conn.pins if not _POWER_SYMBOL_RE.match(p.ref)]
            for i in range(1, len(refs)):
                rx, ry = find(refs[0]), find(refs[i])
                if rx != ry:
                    parent[rx] = ry

        groups: Dict[str, List[str]] = defaultdict(list)
        for ref in parent:
            groups[find(ref)].append(ref)
        return sorted(groups.values(), key=len, reverse=True)


# ── Helper functions ──────────────────────────────────────────────────────────

def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):   return value
    if isinstance(value, str):    return value.strip().lower() in {"1","true","yes","on"}
    if value is None:             return default
    return bool(value)


def _make_preexec_fn() -> Optional[Callable[[], None]]:
    """Return a preexec_fn that applies OS resource limits (Linux only)."""
    if os.name == "nt" or _resource is None:
        return None
    def _set_limits() -> None:
        try:
            _resource.setrlimit(_resource.RLIMIT_CPU,   (10, 10))
            _resource.setrlimit(_resource.RLIMIT_FSIZE, (5*1024*1024, 5*1024*1024))
            _resource.setrlimit(_resource.RLIMIT_NOFILE,(20, 20))
        except Exception:
            pass
    return _set_limits


def _normalise_connections(circuit_data: Dict[str, Any]) -> Dict[str, Any]:
    data      = copy.deepcopy(circuit_data)
    comp_refs = {c["ref"] for c in data.get("components", []) if "ref" in c}
    out: List[Dict[str, Any]] = []
    for conn in data.get("connections", []):
        new_pins: List[Dict[str,str]] = []
        for p in conn.get("pins", []):
            if isinstance(p, str) and "." in p:
                ref, pin = p.split(".", 1)
                new_pins.append({"ref": ref, "pin": pin})
            elif isinstance(p, dict):
                new_pins.append(p)
        new_pins = [
            p for p in new_pins
            if p.get("ref","") in comp_refs
            or _POWER_SYMBOL_RE.match(p.get("ref",""))
        ]
        if len(new_pins) >= 2:
            out.append({**conn, "pins": new_pins})
    data["connections"] = out
    return data


def _ref_prefix(ref: str) -> str:
    m = re.match(r"^([A-Za-z]+)", ref or "")
    return (m.group(1).upper() if m else "U")


def _ensure_unique_component_refs(circuit_data: Dict[str, Any]) -> List[str]:
    """Rename duplicate component refs and rewrite connection refs accordingly."""
    warnings: List[str] = []
    comps = circuit_data.get("components", [])
    if not isinstance(comps, list):
        return warnings

    used: Set[str] = set()
    by_prefix_max: Dict[str, int] = defaultdict(int)
    ref_map: Dict[str, str] = {}

    for comp in comps:
        if not isinstance(comp, dict):
            continue
        old_ref = str(comp.get("ref", "")).strip() or "U1"
        prefix = _ref_prefix(old_ref)
        m = re.match(r"^[A-Za-z]+(\d+)$", old_ref)
        if m:
            by_prefix_max[prefix] = max(by_prefix_max[prefix], int(m.group(1)))

        new_ref = old_ref
        if new_ref in used:
            by_prefix_max[prefix] += 1
            new_ref = f"{prefix}{by_prefix_max[prefix]}"
            warnings.append(f"Renamed duplicate reference {old_ref} -> {new_ref}")
        used.add(new_ref)
        ref_map.setdefault(old_ref, new_ref)
        comp["ref"] = new_ref
        if not str(comp.get("value", "")).strip():
            comp["value"] = new_ref

    for conn in circuit_data.get("connections", []):
        if not isinstance(conn, dict):
            continue
        pins = conn.get("pins", [])
        for pin in pins:
            if isinstance(pin, dict):
                ref = str(pin.get("ref", "")).strip()
                if ref in ref_map:
                    pin["ref"] = ref_map[ref]
            elif isinstance(pin, str) and "." in pin:
                ref, num = pin.split(".", 1)
                if ref in ref_map:
                    idx = pins.index(pin)
                    pins[idx] = f"{ref_map[ref]}.{num}"

    return warnings


def _remove_multi_net_pin_conflicts(circuit_data: Dict[str, Any]) -> List[str]:
    """Ensure each component pin belongs to only one net; keep first occurrence."""
    warnings: List[str] = []
    seen: Set[str] = set()

    for conn in circuit_data.get("connections", []):
        if not isinstance(conn, dict):
            continue
        net = str(conn.get("net", "")).strip() or "N"
        kept: List[Dict[str, str]] = []
        for pin in conn.get("pins", []):
            if not isinstance(pin, dict):
                continue
            ref = str(pin.get("ref", "")).strip()
            num = str(pin.get("pin", "")).strip()
            if not ref or not num:
                continue
            key = f"{ref}.{num}"
            if key in seen:
                warnings.append(f"Dropped conflicting pin {key} from net {net}")
                continue
            seen.add(key)
            kept.append({"ref": ref, "pin": num})
        conn["pins"] = kept

    return warnings


def _ensure_555_core_wiring(circuit_data: Dict[str, Any]) -> List[str]:
    """Apply minimal electrical guardrails for NE555-style components."""
    warnings: List[str] = []
    comps = [c for c in circuit_data.get("components", []) if isinstance(c, dict)]
    if not comps:
        return warnings

    conns = [c for c in circuit_data.get("connections", []) if isinstance(c, dict)]
    if not conns:
        conns = []
        circuit_data["connections"] = conns

    def _find_or_make_net(net_name: str) -> Dict[str, Any]:
        for c in conns:
            if str(c.get("net", "")).upper() == net_name.upper():
                return c
        c = {"net": net_name, "pins": []}
        conns.append(c)
        return c

    def _add_pin(net: Dict[str, Any], ref: str, pin_no: str) -> None:
        pins = net.setdefault("pins", [])
        if not any(p.get("ref") == ref and p.get("pin") == pin_no for p in pins if isinstance(p, dict)):
            pins.append({"ref": ref, "pin": pin_no})

    def _remove_pin(ref: str, pin_no: str) -> None:
        for net in conns:
            pins = net.get("pins", [])
            net["pins"] = [
                p for p in pins
                if not (isinstance(p, dict) and p.get("ref") == ref and p.get("pin") == pin_no)
            ]

    for comp in comps:
        part = str(comp.get("part", "")).upper()
        value = str(comp.get("value", "")).upper()
        if "555" not in part and "555" not in value:
            continue

        ref = str(comp.get("ref", "")).strip()
        if not ref:
            continue

        vcc = _find_or_make_net("VCC")
        gnd = _find_or_make_net("GND")
        _add_pin(vcc, ref, "8")
        _add_pin(vcc, ref, "4")
        _add_pin(gnd, ref, "1")

        # Pins 2 and 6 must share the same threshold/trigger node in astable mode.
        _remove_pin(ref, "2")
        _remove_pin(ref, "6")
        trig_thr = _find_or_make_net("TRIG_THR")
        _add_pin(trig_thr, ref, "2")
        _add_pin(trig_thr, ref, "6")
        warnings.append(f"Applied 555 core wiring guardrails for {ref}")

    # Drop nets that are no longer valid after rewrites.
    circuit_data["connections"] = [
        c for c in conns if len([p for p in c.get("pins", []) if isinstance(p, dict)]) >= 2
    ]
    return warnings


def _sanitize_circuit_data(circuit_data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Repair common generation defects before schema/build validation."""
    data = copy.deepcopy(circuit_data)
    warnings: List[str] = []
    warnings.extend(_ensure_unique_component_refs(data))
    norm = _normalise_connections(data)
    warnings.extend(_remove_multi_net_pin_conflicts(norm))
    warnings.extend(_ensure_555_core_wiring(norm))
    _enrich_net_properties(norm)
    _enrich_component_properties(norm)
    return norm, warnings


def _to_kicad_copper_layer(layer: Any) -> str:
    s = str(layer or "").strip()
    if not s:
        return "F.Cu"

    lut = {
        "top": "F.Cu",
        "bottom": "B.Cu",
        "f.cu": "F.Cu",
        "b.cu": "B.Cu",
        "front": "F.Cu",
        "back": "B.Cu",
    }
    return lut.get(s.lower(), "F.Cu")


def _normalize_for_kicad_schema(circuit_data: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy normalized for circuit_schema / KiCad exporter validation."""
    data = copy.deepcopy(circuit_data)

    board_w = float(data.get("board_width", 100.0) or 100.0)
    board_h = float(data.get("board_height", 80.0) or 80.0)

    # circuit_schema expects centered coordinates (-W/2..W/2, -H/2..H/2), while
    # generation commonly uses top-left coordinates (0..W, 0..H).
    comps = [c for c in data.get("components", []) if isinstance(c, dict)]
    looks_like_top_left = bool(comps) and all(
        0.0 <= float(c.get("x", 0.0)) <= board_w and 0.0 <= float(c.get("y", 0.0)) <= board_h
        for c in comps
    )

    # Build per-component pin sets from net connections (handles both "REF.PIN"
    # strings and {ref, pin} dict style).
    pins_by_ref: Dict[str, Set[str]] = defaultdict(set)
    for conn in data.get("connections", []):
        if not isinstance(conn, dict):
            continue
        for p in conn.get("pins", []):
            ref = ""
            pin = ""
            if isinstance(p, str) and "." in p:
                ref, pin = p.split(".", 1)
            elif isinstance(p, dict):
                ref = str(p.get("ref", "")).strip()
                pin = str(p.get("pin", "")).strip()
            if ref and pin:
                pins_by_ref[ref].add(pin)

    for comp in data.get("components", []):
        if isinstance(comp, dict):
            comp["layer"] = _to_kicad_copper_layer(comp.get("layer"))

            if looks_like_top_left:
                comp["x"] = float(comp.get("x", 0.0)) - (board_w / 2.0)
                comp["y"] = float(comp.get("y", 0.0)) - (board_h / 2.0)

            ref = str(comp.get("ref", "")).strip()
            value = str(comp.get("value", "")).strip()
            part = str(comp.get("part", "")).strip()
            footprint = str(comp.get("footprint", "")).upper()

            if not value:
                comp["value"] = ref or "N/A"

            # Best-effort defaults so schema validation can resolve symbols.
            if not comp.get("lib"):
                if "1117" in value.upper() or "1117" in part.upper() or "1117" in footprint:
                    comp["lib"] = "Regulator_Linear"
                else:
                    comp["lib"] = "Device"

            if not comp.get("part"):
                if "1117" in value.upper() or "1117" in footprint:
                    comp["part"] = "AMS1117-3.3"
                elif ref.upper().startswith("R"):
                    comp["part"] = "R"
                elif ref.upper().startswith("C"):
                    comp["part"] = "C"
                elif ref.upper().startswith("D"):
                    comp["part"] = "D"
                else:
                    comp["part"] = "U"

            # If pins are missing, infer pin list from connection usage.
            if ref and (not isinstance(comp.get("pins"), list) or len(comp.get("pins", [])) == 0):
                pin_nums = sorted(pins_by_ref.get(ref, set()), key=lambda x: (not x.isdigit(), x))
                if pin_nums:
                    comp["pins"] = [{"number": p, "name": p} for p in pin_nums]

    return data


def _is_ground_net_name(net_name: str) -> bool:
    u = (net_name or "").upper()
    return any(token in u for token in _GROUND_NET_PATTERNS)


def _is_power_net_name(net_name: str) -> bool:
    u = (net_name or "").upper()
    if not u:
        return False
    if u in _POWER_NET_MAP or u.startswith("+"):
        return True
    return any(token in u for token in _POWER_NET_MAP)


def _detect_circuit_quality_issues(circuit_data: Dict[str, Any]) -> List[str]:
    """Return high-signal topology issues that often produce invalid schematics."""
    issues: List[str] = []

    comps = [c for c in circuit_data.get("components", []) if isinstance(c, dict)]
    conns = [c for c in circuit_data.get("connections", []) if isinstance(c, dict)]
    if not comps:
        return ["No components present"]

    pin_nets: Dict[str, Dict[str, str]] = defaultdict(dict)
    comp_refs: Set[str] = set()
    for comp in comps:
        ref = str(comp.get("ref", "")).strip()
        if ref:
            comp_refs.add(ref)

    for conn in conns:
        net = str(conn.get("net", "")).strip()
        for p in conn.get("pins", []):
            if not isinstance(p, dict):
                continue
            ref = str(p.get("ref", "")).strip()
            pin = str(p.get("pin", "")).strip()
            if ref and pin and ref in comp_refs:
                pin_nets[ref][pin] = net

    # Generic connectivity sanity checks.
    for comp in comps:
        ref = str(comp.get("ref", "")).strip()
        if not ref or _POWER_SYMBOL_RE.match(ref):
            continue
        connected_pin_count = len(pin_nets.get(ref, {}))
        if connected_pin_count == 0:
            issues.append(f"{ref} is floating (no connected pins)")
        elif ref.upper().startswith("U") and connected_pin_count < 2:
            issues.append(f"{ref} has too few connections for an IC-like part")

    def _caps_between(net_a: str, net_b: str) -> int:
        count = 0
        for comp in comps:
            ref = str(comp.get("ref", "")).strip().upper()
            if not ref.startswith("C"):
                continue
            nets = set(pin_nets.get(comp.get("ref", ""), {}).values())
            if net_a in nets and net_b in nets:
                count += 1
        return count

    # Topology checks for common generated circuits.
    for comp in comps:
        ref = str(comp.get("ref", "")).strip()
        part = str(comp.get("part", "")).upper()
        value = str(comp.get("value", "")).upper()
        nets = pin_nets.get(ref, {})

        if "555" in part or "555" in value:
            n1 = nets.get("1", "")
            n2 = nets.get("2", "")
            n3 = nets.get("3", "")
            n4 = nets.get("4", "")
            n6 = nets.get("6", "")
            n8 = nets.get("8", "")

            if not n1 or not _is_ground_net_name(n1):
                issues.append(f"{ref} pin 1 (GND) is not tied to a ground net")
            if not n8 or not _is_power_net_name(n8):
                issues.append(f"{ref} pin 8 (VCC) is not tied to a power net")
            if not n4 or n4 != n8:
                issues.append(f"{ref} pin 4 (RESET) should be tied to VCC")
            if not n2 or not n6 or n2 != n6:
                issues.append(f"{ref} pins 2 and 6 must share the same timing net")
            if not n3:
                issues.append(f"{ref} pin 3 (OUT) is unconnected")

        if "1117" in part or "1117" in value:
            n1 = nets.get("1", "")
            n2 = nets.get("2", "")
            n3 = nets.get("3", "")

            if not n1 or not _is_ground_net_name(n1):
                issues.append(f"{ref} pin 1 must connect to GND")
            if not n2 or not n3:
                issues.append(f"{ref} requires both IN(pin3) and OUT(pin2) nets")
            elif n2 == n3:
                issues.append(f"{ref} IN and OUT nets are shorted together")
            else:
                if _caps_between(n3, n1) == 0:
                    issues.append(f"{ref} missing input decoupling capacitor between IN and GND")
                if _caps_between(n2, n1) == 0:
                    issues.append(f"{ref} missing output decoupling capacitor between OUT and GND")

    return issues


def _enrich_net_properties(circuit_data: Dict[str, Any]) -> None:
    for conn in circuit_data.get("connections", []):
        net_name = conn.get("net", "")
        props    = conn.get("properties", {})
        if props and props.get("net_type", "signal") != "signal":
            continue
        upper = net_name.upper()
        for pattern, power_props in _POWER_NET_MAP.items():
            if pattern in upper:
                conn["properties"] = {
                    "net_type": "power", "voltage": power_props["voltage"],
                    "current_max": None, "frequency": None,
                    "impedance_ohms": None, "length_mm": None, "is_critical": False,
                }
                break
        else:
            if any(p in upper for p in _GROUND_NET_PATTERNS):
                conn["properties"] = dict(_GROUND_PROPS)


def _enrich_component_properties(circuit_data: Dict[str, Any]) -> None:
    for comp in circuit_data.get("components", []):
        part  = comp.get("part",  "").upper()
        lib   = comp.get("lib",   "").upper()
        value = comp.get("value", "").lower()
        ref   = comp.get("ref",   "")

        # LED
        if (part == "LED" or "LED" in lib
                or (ref.upper().startswith("LED"))
                or "led" in value):
            comp["is_polarized"] = True

        # Electrolytic / polarized capacitor
        if part in ("C", "CP") or ref.upper().startswith("C"):
            if part == "CP" or "polarized" in comp.get("description", "").lower():
                comp["is_polarized"] = True
            elif "uf" in value or "µf" in value:
                val_str = value.replace("uf","").replace("µf","").strip()
                try:
                    if float(val_str) >= 10.0:
                        comp["is_polarized"] = True
                except ValueError:
                    pass

        # FIXED: previous condition was always-False (D…and not D…)
        # Diodes (all polarized) — match D1, D2 … but not U1, R1 etc.
        if part in ("D", "DIODE") or "diode" in comp.get("description", "").lower():
            comp["is_polarized"] = True
        elif re.match(r'^D\d+$', ref):   # ref like D1, D12
            comp["is_polarized"] = True


def _parse_kicad_netlist_to_circuit_data(
    netlist_path: Path, description: str
) -> Dict[str, Any]:
    tree = ET.parse(str(netlist_path))
    root = tree.getroot()
    comp_map: Dict[str, Dict[str, Any]] = {}

    for comp in root.findall(".//components/comp"):
        ref = (comp.attrib.get("ref") or "").strip()
        if not ref:
            continue
        libsource = comp.find("libsource")
        comp_map[ref] = {
            "ref":       ref,
            "value":    (comp.findtext("value") or ref).strip(),
            "footprint":(comp.findtext("footprint") or "").strip(),
            "lib":      (libsource.attrib.get("lib")  if libsource is not None else "") or "Device",
            "part":     (libsource.attrib.get("part") if libsource is not None else "") or "R",
            "x": 0.0, "y": 0.0, "rotation": 0.0, "layer": "top",
            "pins": set(),
        }

    connections: List[Dict[str, Any]] = []
    for net in root.findall(".//nets/net"):
        net_name = (net.attrib.get("name") or f"N{net.attrib.get('code','')}").strip()
        pin_refs: List[Dict[str,str]] = []
        for node in net.findall("node"):
            ref = (node.attrib.get("ref") or "").strip()
            pin = (node.attrib.get("pin") or "").strip()
            if not ref or not pin:
                continue
            pin_refs.append({"ref": ref, "pin": pin})
            if ref not in comp_map:
                comp_map[ref] = {
                    "ref": ref, "value": ref, "footprint": "",
                    "lib": "Device", "part": "R",
                    "x": 0.0, "y": 0.0, "rotation": 0.0, "layer": "top",
                    "pins": set(),
                }
            comp_map[ref]["pins"].add(pin)
        if len(pin_refs) >= 2:
            connections.append({"net": net_name, "pins": pin_refs})

    def _pin_key(s: str) -> Tuple[int, str]:
        return (0, str(int(s))) if s.isdigit() else (1, s)

    components = []
    for ref, comp in comp_map.items():
        comp["pins"] = [{"number": p, "name": p}
                        for p in sorted(comp["pins"], key=_pin_key)]
        components.append(comp)

    return {
        "description": description,
        "components":  components,
        "connections": connections,
        "board_width": 100.0,
        "board_height": 80.0,
        "design_rules": {},
    }


# ── SKiDL execution ───────────────────────────────────────────────────────────

def _execute_skidl_code(
    code: str, request_id: str, prompt: str
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """Run generated SKiDL code in a sandboxed subprocess."""
    warnings: List[str] = []
    code = _extract_python_code_block(code)

    blocked = _screen_skidl_code(code)
    if blocked:
        return None, [f"SKiDL code rejected: {blocked}"]

    run_dir = TEMP_DIR / f"skidl_{request_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    script  = run_dir / "circuit.py"

    suffix = (
        "\n\nERC()\ngenerate_netlist(file_='circuit.net', tool=KICAD9)\n"
        if "generate_netlist" not in code else ""
    )
    script.write_text(code + suffix, encoding="utf-8")

    env = os.environ.copy()
    sym_dir = _default_kicad_symbol_dir()
    if sym_dir:
        env.setdefault("KICAD_SYMBOL_DIR", sym_dir)
        # SKiDL checks versioned env vars too, so mirror the detected path.
        env.setdefault("KICAD6_SYMBOL_DIR", sym_dir)
        env.setdefault("KICAD7_SYMBOL_DIR", sym_dir)
        env.setdefault("KICAD8_SYMBOL_DIR", sym_dir)
        env.setdefault("KICAD9_SYMBOL_DIR", sym_dir)
    else:
        return None, [
            "SKiDL skipped: KiCad symbol directory not found. "
            "Set KICAD_SYMBOL_DIR (or KICAD9_SYMBOL_DIR) to your KiCad symbols path."
        ]

    cf = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    try:
        run_kwargs: Dict[str, Any] = {
            "cwd": str(run_dir),
            "capture_output": True,
            "text": True,
            "timeout": SKIDL_EXEC_TIMEOUT,
            "env": env,
            "creationflags": cf,
        }
        preexec = _make_preexec_fn()
        if preexec is not None:
            run_kwargs["preexec_fn"] = preexec

        completed = subprocess.run(
            [sys.executable, "-I", "-B", str(script)],
            **run_kwargs,
        )
    except subprocess.TimeoutExpired:
        return None, ["SKiDL execution timed out"]
    except Exception as exc:
        return None, [f"SKiDL execution exception: {exc}"]

    if completed.returncode != 0:
        err = (completed.stderr or completed.stdout or "unknown error").strip()
        return None, [f"SKiDL failed: {err[:400]}"]

    net_files = list(run_dir.glob("*.net"))
    if not net_files:
        return None, ["SKiDL produced no .net file"]

    try:
        cdata = _parse_kicad_netlist_to_circuit_data(net_files[0], prompt)
        warnings.append("Used SKiDL generation pipeline")
        return cdata, warnings
    except Exception as exc:
        return None, [f"Netlist parse failed: {exc}"]


async def _generate_via_skidl(
    prompt: str, request_id: str
) -> Tuple[Optional[Dict[str, Any]], List[str], Optional[str]]:
    """LLM → SKiDL code → execute → netlist → circuit_data."""
    if _state.llm is None:
        return None, ["SKiDL skipped: LLM not loaded"], None
    try:
        return await generate_via_skidl_retry(
            prompt=prompt,
            request_id=request_id,
            config=SkidlRetryConfig(
                system_prompt=SKIDL_SYSTEM_PROMPT,
                max_retries=SKIDL_MAX_RETRIES,
            ),
            llm_generate=lambda payload, max_tokens, temperature: _state.llm.generate_async(
                payload,
                max_tokens=max_tokens,
                temperature=temperature,
            ),
            execute_skidl_code=_execute_skidl_code,
            symbol_dir_resolver=_default_kicad_symbol_dir,
        )
    except Exception as exc:
        return None, [f"LLM code generation failed: {exc}"], None


async def _cleanup_run_dir(path: Path) -> None:
    """Background task: remove temporary SKiDL run directory."""
    await asyncio.to_thread(shutil.rmtree, str(path), True)


def _safe_filename(desc: str, fallback: str) -> str:
    ascii_only = desc.encode("ascii", errors="ignore").decode("ascii")
    safe = re.sub(r"[^\w\s\-]", "", ascii_only).strip()
    safe = re.sub(r"\s+", "_", safe)
    safe = re.sub(r"_+", "_", safe).strip("_")
    return (safe[:80] or fallback)


def _normalize_kicad_schematic_text(content: str) -> str:
    """Normalize known KiCad syntax incompatibilities in generated .kicad_sch text.

    Guardrail for old exporter output: KiCad expects junction coordinates as
    `(junction (at x y) ...)`, not `(junction (at x y angle) ...)`.
    """
    return re.sub(
        r"\(junction\s+\(at\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+-?\d+(?:\.\d+)?\)",
        r"(junction (at \1 \2)",
        content,
    )


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting AI PCB Assistant v3.0…")
    logger.info("SKiDL available: %s", _SKIDL)
    logger.info("KiCad export available: %s", _KICAD_EXPORT_AVAIL)
    if not _KICAD_EXPORT_AVAIL:
        logger.warning(
            "circuit_schema / engines.kicad_exporter not found – "
            ".kicad_sch export will be disabled"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    if TEMPLATES_DIR.exists():
        for path in sorted(TEMPLATES_DIR.glob("*.json")):
            try:
                _state.template_cache[path.stem] = json.loads(
                    path.read_text(encoding="utf-8")
                )
            except Exception as exc:
                logger.warning("Template load failed %s: %s", path.name, exc)
    logger.info("Templates loaded: %d", len(_state.template_cache))

    try:
        from engines.llm_engine import load_llm  # type: ignore
        _state.llm = load_llm()
        logger.info(
            "LLM loaded: provider=%s model=%s",
            type(_state.llm).__name__ if _state.llm is not None else "None",
            getattr(_state.llm, "model", None),
        )
    except ImportError:
        logger.warning("engines.llm_engine not found – LLM disabled.")
    except Exception as exc:
        logger.warning("LLM load failed: %s", exc)

    try:
        from engines.placement_engine import load_placement_model  # type: ignore
        _state.rl_model = load_placement_model()
        logger.info("RL placement engine loaded.")
    except ImportError:
        logger.warning("engines.placement_engine not found – RL disabled.")
    except Exception as exc:
        logger.warning("RL placement load failed: %s", exc)

    logger.info("Capabilities: %s", _state.get_capabilities())
    yield
    logger.info("Shutting down.")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI PCB Assistant Backend",
    description="AI backend for KiCad PCB design with SKiDL netlist generation",
    version="3.0.0",
    lifespan=lifespan,
)
app.add_middleware(GZipMiddleware, minimum_size=1000)
cors_settings = parse_cors_settings(
    os.environ.get("CORS_ORIGINS", "http://127.0.0.1:3000,http://localhost:3000"),
    os.environ.get("CORS_ALLOW_CREDENTIALS", "true"),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_settings.allow_origins,
    allow_credentials=cors_settings.allow_credentials,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_demo_dir = _BASE_DIR / "demo"
if _demo_dir.exists():
    app.mount("/demo", StaticFiles(directory=str(_demo_dir), html=True), name="demo")


@app.middleware("http")
async def add_timing_header(request: Request, call_next: Any):
    t0 = time.perf_counter()
    resp = await call_next(request)
    resp.headers["X-Process-Time-Ms"] = f"{(time.perf_counter()-t0)*1000:.1f}"
    return resp


@app.exception_handler(Exception)
async def global_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled %s %s", request.method, request.url)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health_check() -> HealthResponse:
    llm_provider = type(_state.llm).__name__ if _state.llm is not None else None
    llm_model = getattr(_state.llm, "model", None) if _state.llm is not None else None
    return HealthResponse(
        status="healthy" if _state.models_loaded else "degraded",
        version=app.version,
        uptime_seconds=round(_state.uptime_seconds, 1),
        models_loaded=_state.models_loaded,
        llm_loaded=_state.llm is not None,
        placement_engine_loaded=_state.rl_model is not None,
        skidl_available=_SKIDL,
        kicad_export_available=_KICAD_EXPORT_AVAIL,
        templates_available=len(_state.template_cache),
        llm_provider=llm_provider,
        llm_model=llm_model,
        capabilities=_state.get_capabilities(),
    )


@app.post("/generate", response_model=GenerateResponse, tags=["generation"])
async def generate_circuit(
    request: GenerateRequest, background_tasks: BackgroundTasks
) -> GenerateResponse:
    """
    Generation priority (each stage only runs if the previous produced nothing):
      0. SKiDL via LLM           (if use_skidl=true AND LLM loaded)
      1. Strong template match    (score ≥ STRONG_TEMPLATE_THRESHOLD)
      2. LLM JSON generation
      3. Weak template fallback   (any score > 0)
    """
    t0         = time.perf_counter()
    request_id = str(uuid.uuid4())[:8]
    warnings:  List[str]       = []

    prompt_lower          = request.prompt.lower()
    best_name, best_score = _score_template(prompt_lower, TEMPLATE_KEYWORDS)

    circuit_data:      Optional[Dict[str, Any]] = None
    template_used:     Optional[str]            = None
    generation_method: Optional[str]            = None
    llm_failed                                  = False

    strict_llm = _coerce_bool(
        (request.constraints or {}).get("strict_llm"), default=True
    )
    prefer_model = _coerce_bool(
        (request.constraints or {}).get("prefer_model"), default=True
    )
    use_skidl = _coerce_bool(
        (request.constraints or {}).get("use_skidl"), default=False
    )
    allow_template = _coerce_bool(
        (request.constraints or {}).get("allow_template"), default=False
    )

    # ── 0. SKiDL-first ───────────────────────────────────────────────────────
    if use_skidl and _state.llm and _SKIDL:
        skidl_data, skidl_w, skidl_code = await _generate_via_skidl(request.prompt, request_id)
        warnings.extend(skidl_w)

        # Always schedule cleanup of attempt directories, even if SKiDL fails.
        for i in range(SKIDL_MAX_RETRIES):
            run_dir = TEMP_DIR / f"skidl_{request_id}_{i}"
            if run_dir.exists():
                background_tasks.add_task(_cleanup_run_dir, run_dir)

        if skidl_data:
            circuit_data      = skidl_data
            template_used     = "skidl_llm"
            generation_method = "skidl"

    # ── 1. Strong template match (ONLY when SKiDL produced nothing) ──────────
    # FIXED: previously ran unconditionally and overwrote a valid SKiDL result.
    can_use_template_fast_path = (
        circuit_data is None
        and allow_template
        and not strict_llm
        and (not prefer_model or _state.llm is None)
    )
    if can_use_template_fast_path:
        if best_name and best_name in _state.template_cache and best_score >= STRONG_TEMPLATE_THRESHOLD:
            circuit_data      = _state.template_cache[best_name]
            template_used     = best_name
            generation_method = "template"
            warnings.append(f"Strong template match: {best_name} (score={best_score})")

    # ── 2. LLM JSON generation ────────────────────────────────────────────────
    if circuit_data is None and _state.llm:
        try:
            circuit_data = await _state.llm.generate_circuit_json(request.prompt) or None
            if circuit_data:
                generation_method = "llm_json"
            else:
                llm_failed = True
                warnings.append("LLM returned no circuit data")
        except Exception as exc:
            llm_failed = True
            details = str(exc).strip() or repr(exc)
            warnings.append(f"LLM JSON generation failed ({type(exc).__name__}): {details}")

    # ── 3. Weak template fallback ─────────────────────────────────────────────
    can_use_template_fallback = (
        circuit_data is None
        and allow_template
        and not strict_llm
        and (not prefer_model or _state.llm is None or llm_failed)
    )
    if can_use_template_fallback:
        if best_name and best_name in _state.template_cache and best_score > 0:
            circuit_data      = _state.template_cache[best_name]
            template_used     = best_name
            generation_method = "template"
            if _state.llm:
                warnings.append(f"Template fallback: {best_name} (score={best_score})")

    if circuit_data is None:
        return GenerateResponse(
            success=False,
            error=(
                "Could not generate a circuit from the AI model. "
                "Ensure the model backend is reachable (for Ollama cloud set "
                "LLM_BACKEND=ollama, OLLAMA_BASE_URL, and OLLAMA_MODEL). "
                "Template fallback is disabled by default; set constraints.allow_template=true to enable it."
            ),
            warnings=warnings,
            request_id=request_id,
            generation_method=generation_method,
        )

    # ── Schema validation ─────────────────────────────────────────────────────
    if _KICAD_EXPORT_AVAIL:
        try:
            schema_src = _normalize_for_kicad_schema(circuit_data)
            _CIRCUIT_SCHEMA_MOD.CircuitData(**schema_src)
        except Exception as exc:
            warnings.append(f"Circuit schema validation failed: {exc}")
            # Keep processing through internal sanitizer/BoardData path.
            # KiCad export step will be attempted later and can fail independently.

    # ── Sanitize, normalise, enrich, validate ─────────────────────────────────
    try:
        norm, sanitize_warnings = _sanitize_circuit_data(circuit_data)
        warnings.extend(sanitize_warnings)

        quality_issues = _detect_circuit_quality_issues(norm)
        if quality_issues:
            for issue in quality_issues[:10]:
                warnings.append(f"[TOPOLOGY] {issue}")

            can_fallback_to_template = (
                allow_template
                and
                (not strict_llm)
                and (not prefer_model or _state.llm is None or llm_failed or generation_method == "llm_json")
                and generation_method != "template"
                and best_name is not None
                and best_name in _state.template_cache
                and best_score > 0
            )

            if can_fallback_to_template:
                template_used = best_name
                generation_method = "template"
                warnings.append(
                    f"Switched to template '{best_name}' after topology checks failed"
                )
                norm, fallback_warnings = _sanitize_circuit_data(
                    _state.template_cache[best_name]
                )
                warnings.extend(fallback_warnings)

        board = BoardData(**norm)
    except Exception as exc:
        return GenerateResponse(
            success=False,
            error=f"Board schema validation failed: {exc}",
            warnings=warnings,
            request_id=request_id,
            generation_method=generation_method,
        )

    # ── Auto-place if all at origin ───────────────────────────────────────────
    if all(c.x == 0.0 and c.y == 0.0 for c in board.components if not c.is_power_symbol):
        algo = (
            "force_directed" if _NP
            else "annealing"  if request.priority == "quality"
            else "grid"
        )
        placement = PlacementOptimizer(board).optimize(algo)
        for ref, pos in placement["positions"].items():
            comp = board.get_component(ref)
            if comp:
                comp.x, comp.y = pos["x"], pos["y"]

    # ── DFM ───────────────────────────────────────────────────────────────────
    for v in AdvancedDFMEngine(board).analyze():
        if v.severity in ("error", "critical"):
            warnings.append(f"[DFM {v.rule_id}] {v.message}")

    # ── Persist JSON ──────────────────────────────────────────────────────────
    board_dict  = board.model_dump()
    json_path   = OUTPUT_DIR / f"circuit_{request_id}.json"

    async def _save_json() -> None:
        txt = json.dumps(board_dict, indent=2)
        if _AIOFILES:
            async with aiofiles.open(json_path, "w") as f:
                await f.write(txt)
        else:
            json_path.write_text(txt)

    background_tasks.add_task(_save_json)

    # ── KiCad schematic export ────────────────────────────────────────────────
    sch_filename: Optional[str] = None
    download_url: Optional[str] = None

    if _KICAD_EXPORT_AVAIL:
        try:
            raw_src  = (_state.template_cache.get(template_used, {})
                        if template_used and template_used != "skidl_llm"
                        else circuit_data)
            schema_src = _normalize_for_kicad_schema(raw_src)
            schema   = _CIRCUIT_SCHEMA_MOD.CircuitData(**schema_src)
            content  = _KICAD_EXPORTER_MOD.export_to_kicad_sch(schema)
            content  = _normalize_kicad_schematic_text(content)
            desc     = raw_src.get("description", template_used or request_id)
            safe     = _safe_filename(desc, request_id)
            sch_path = OUTPUT_DIR / f"{safe}.kicad_sch"
            sch_path.write_text(content, encoding="utf-8")
            sch_filename = sch_path.name
            download_url = f"/download/{sch_filename}"
            logger.info("Schematic saved: %s", sch_path)
        except Exception as exc:
            warnings.append(f"KiCad export failed: {exc}")
            logger.warning("KiCad export failed: %s", exc)

    return GenerateResponse(
        success=True,
        circuit_data=board_dict,
        template_used=template_used,
        generation_method=generation_method,
        generation_time_ms=round((time.perf_counter()-t0)*1000, 1),
        warnings=warnings,
        request_id=request_id,
        output_file=sch_filename,
        download_url=download_url,
    )


@app.post("/generate/stream", tags=["generation"])
async def generate_circuit_stream(
    request: GenerateRequest, background_tasks: BackgroundTasks
) -> StreamingResponse:
    """SSE endpoint that streams generation progress, then the final result."""
    event_queue: asyncio.Queue = asyncio.Queue()

    async def _run_orchestration() -> None:
        try:
            result = await orchestrate(
                request=request,
                llm=_state.llm,
                template_cache=_state.template_cache,
                output_dir=OUTPUT_DIR,
                bg_add_task=background_tasks.add_task,
                queue_callback=event_queue,
                circuit_schema=_CIRCUIT_SCHEMA_MOD,
                kicad_exporter=_KICAD_EXPORTER_MOD,
                aiofiles_mod=aiofiles if _AIOFILES else None,
            )

            if result.success:
                await event_queue.put(("result", GenerateResponse(
                    success=True,
                    circuit_data=result.board_dict,
                    template_used=result.template_used,
                    generation_method=result.generation_method,
                    generation_time_ms=result.generation_time_ms,
                    warnings=result.warnings,
                    request_id=result.request_id,
                    output_file=result.sch_filename,
                    download_url=result.download_url,
                ).model_dump()))
            else:
                await event_queue.put(("error", {"error": result.error}))
        except Exception as exc:
            await event_queue.put(("error", {"error": str(exc)}))
        finally:
            # None sentinel tells the stream generator to close.
            await event_queue.put(None)

    async def _event_stream():
        runner = asyncio.create_task(_run_orchestration())
        try:
            while True:
                item = await event_queue.get()
                # ── Sentinel: orchestration finished ──────────────────
                if item is None:
                    break
                # ── Status dict from orchestrate() ────────────────────
                if isinstance(item, dict):
                    payload = json.dumps(item, ensure_ascii=False)
                    yield f"event: status\ndata: {payload}\n\n"
                    continue
                # ── Tagged tuple: ("result"|"error", payload_dict) ────
                name, payload = item
                yield f"event: {name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
        finally:
            if not runner.done():
                runner.cancel()

    return StreamingResponse(_event_stream(), media_type="text/event-stream")


@app.post("/generate/schematic", response_model=SchematicResponse, tags=["generation"])
async def generate_schematic(request: SchematicRequest) -> SchematicResponse:
    result = await generate_circuit(GenerateRequest(prompt=request.prompt), BackgroundTasks())
    return SchematicResponse(
        success=result.success,
        component_count=(
            len(result.circuit_data.get("components", [])) if result.circuit_data else 0
        ),
        output_file=result.output_file,
        download_url=result.download_url,
        error=result.error,
    )


@app.post("/analyze/dfm", response_model=List[DFMViolation], tags=["dfm"])
async def analyze_dfm(board: BoardData) -> List[DFMViolation]:
    return AdvancedDFMEngine(board).analyze()


@app.post("/dfm/check", response_model=List[DFMViolation], tags=["dfm"])
async def dfm_check_compat(board: BoardData) -> List[DFMViolation]:
    """Alias for /analyze/dfm — keeps v2 plugin builds working."""
    return await analyze_dfm(board)


@app.post("/placement/optimize", tags=["placement"])
async def optimize_placement(board: BoardData, algorithm: str = "auto") -> Dict[str, Any]:
    t0 = time.perf_counter()
    if algorithm == "auto":
        algorithm = ("rl" if _state.rl_model
                     else "force_directed" if _NP
                     else "annealing")

    if algorithm == "rl" and _state.rl_model:
        try:
            from engines.placement_engine import optimize_with_rl  # type: ignore
            result: Dict[str, Any] = optimize_with_rl(_state.rl_model, board.model_dump())
            result.update({"algorithm": "rl", "time_ms": (time.perf_counter()-t0)*1000})
            return result
        except Exception as exc:
            logger.warning("RL placement failed, falling back: %s", exc)
            algorithm = "force_directed" if _NP else "annealing"

    result = PlacementOptimizer(board).optimize(algorithm)
    result["time_ms"] = round((time.perf_counter()-t0)*1000, 1)
    return result


@app.post("/export/kicad", tags=["export"])
async def export_kicad(circuit: dict) -> StreamingResponse:
    if not _KICAD_EXPORT_AVAIL:
        raise HTTPException(status_code=501, detail="KiCad exporter not available")
    try:
        data    = _CIRCUIT_SCHEMA_MOD.CircuitData(**circuit)
        content = _KICAD_EXPORTER_MOD.export_to_kicad_sch(data)
        content = _normalize_kicad_schematic_text(content)
        return StreamingResponse(
            iter([content]),
            media_type="application/x-kicad-schematic",
            headers={"Content-Disposition": "attachment; filename=circuit.kicad_sch"},
        )
    except Exception as exc:
        logger.error("KiCad export: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/download/{filename}", tags=["export"])
async def download_file(filename: str):
    safe = re.sub(r"[^a-zA-Z0-9_.\-]", "", filename)
    if not (safe.endswith(".kicad_sch") or safe.endswith(".json")):
        raise HTTPException(status_code=400, detail="Only .kicad_sch or .json allowed")
    if "/" in safe or "\\" in safe:
        raise HTTPException(status_code=400, detail="Invalid filename")
    fp = OUTPUT_DIR / safe
    if not fp.exists():
        raise HTTPException(status_code=404, detail=f"Not found: {safe}")

    if safe.endswith(".kicad_sch"):
        # Backward-compatibility guard for older exported files on disk.
        # This ensures plugin downloads remain loadable even if the file was
        # generated before syntax fixes were deployed.
        content = fp.read_text(encoding="utf-8")
        normalized = _normalize_kicad_schematic_text(content)
        if normalized != content:
            fp.write_text(normalized, encoding="utf-8")
        return StreamingResponse(
            iter([normalized]),
            media_type="application/x-kicad-schematic",
            headers={"Content-Disposition": f'attachment; filename={safe}'},
        )

    return FileResponse(path=str(fp), filename=safe, media_type="application/octet-stream")


@app.get("/circuit/{name}", tags=["templates"])
async def get_circuit_template(name: str) -> Dict[str, Any]:
    safe = re.sub(r"[^a-zA-Z0-9_\-]", "", name)
    data = _state.template_cache.get(safe)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Template '{safe}' not found")
    return data


@app.get("/templates", tags=["templates"])
async def list_templates() -> List[Dict[str, Any]]:
    return [
        {
            "name":        name,
            "description": data.get("description", ""),
            "components":  len(data.get("components", [])),
            "nets":        len(data.get("connections", [])),
            "category":    data.get("metadata", {}).get("category", ""),
        }
        for name, data in sorted(_state.template_cache.items())
    ]


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "ai_server:app",
        host=os.environ.get("HOST",    "127.0.0.1"),
        port=int(os.environ.get("PORT", "8765")),
        reload=os.environ.get("RELOAD", "false").lower() == "true",
        workers=int(os.environ.get("WORKERS", "1")),
        log_level=os.environ.get("LOG_LEVEL", "info"),
    )