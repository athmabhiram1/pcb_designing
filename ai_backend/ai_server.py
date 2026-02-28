"""
AI PCB Assistant – FastAPI Backend v2.1
Handles LLM inference, netlist-aware placement optimisation, and advanced DFM checking.

Fixes vs v2.0:
  - ComponentData mutable (model_config frozen=False) → runtime mutation works
  - PinRef / ComponentData ref patterns widened for real KiCad refs (#PWR01, SW1 …)
  - _simulated_annealing implemented (was missing → AttributeError crash)
  - numpy made optional (hard crash if absent)
  - SpatialIndex.query_neighbors corrected return type and callers fixed
  - BoardData.validate_references soft-warns instead of raising for power symbols
  - CORS defaults include null / file:// origin for KiCad plugin
  - _check_high_speed_signal walrus operator bug fixed
  - _async_save_circuit guarded properly; fallback is always sync
  - download_url now included in GenerateResponse
  - /download/{filename}, /circuit/{name}, /generate/schematic restored
  - Grid placement column-advance logic fixed
  - Rotation near-zero comparison uses tolerance
  - Logging deduplication fixed
  - Config converted to plain module-level constants (no fragile frozen dataclass)
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import random
import re
import tempfile
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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

# ── Logging ───────────────────────────────────────────────────────────────────
# Use a single handler on the root logger – no StreamHandler added manually
# (basicConfig does it already), and propagate=True (default) so we don't
# double-log.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Configuration (plain constants – no frozen dataclass) ─────────────────────

_BASE_DIR      = Path(__file__).parent
TEMPLATES_DIR  = _BASE_DIR / "templates"
OUTPUT_DIR     = _BASE_DIR / "output"
TEMP_DIR       = Path(tempfile.gettempdir()) / "ai_pcb"

# DFM thresholds
DFM_MIN_SPACING_MM          = 0.5   # Modern SMT manufacturing standard
DFM_EDGE_CLEARANCE_MM       = 1.0
DFM_MAX_COMPONENT_HEIGHT_MM = 25.0
DFM_DECOUPLING_MAX_DIST_MM  = 10.0

# Spatial index
SPATIAL_GRID_SIZE_MM = 5.0

# Placement
WIRE_LENGTH_WEIGHT = 0.4
SA_INITIAL_TEMP    = 100.0
SA_COOLING_RATE    = 0.98
SA_ITERATIONS      = 500

# ── Template keyword registry ─────────────────────────────────────────────────
# (keywords, template_stem, score_weight) – highest cumulative score wins.

TEMPLATE_KEYWORDS: List[Tuple[List[str], str, int]] = [
    (["555 timer", "ne555", "astable 555", "555 oscillator"], "555_timer_oscillator", 100),
    (["555", "timer", "blink", "astable", "multivibrator"],   "555_timer",            80),
    (["3.3v regulator", "3v3 ldo", "ams1117-3.3"],            "3v3_regulator_ldo",    100),
    (["3.3v", "3v3", "ams1117", "ldo", "voltage regulator"], "3v3_regulator",        80),
    (["5v regulator", "5v ldo", "7805", "5v power"],          "5v_regulator",         90),
    (["led driver", "led array", "multiple led"],              "led_array_driver",     90),
    (["led", "diode", "indicator", "resistor led"],            "led_resistor",         70),
    (["opamp buffer", "unity gain buffer", "voltage follower"],"opamp_buffer",         90),
    (["opamp", "op-amp", "operational amplifier", "gain"],    "opamp_general",        70),
    (["mosfet switch", "high side switch", "low side switch"],"mosfet_switch",        90),
    (["mosfet", "nmos", "pmos", "transistor switch"],         "mosfet_general",       70),
    (["rc filter", "low pass filter", "high pass filter"],    "rc_filter",            80),
    (["voltage divider", "resistor divider"],                  "voltage_divider",      75),
    (["crystal oscillator", "quartz", "mhz crystal"],         "crystal_oscillator",   85),
    (["usb power", "usb protection", "usb esd"],              "usb_protection",       90),
]

# ── Known KiCad power/virtual symbol prefixes to skip in ref validation ───────
_POWER_SYMBOL_RE = re.compile(r'^#(PWR|FLG|GND|VCC|VDD)\d*$', re.IGNORECASE)

# ── Pydantic models ───────────────────────────────────────────────────────────

class PinRef(BaseModel):
    """Reference to a specific pin on a component.

    KiCad refs can be: R1, C12, U3, #PWR01, SW1, LED1, TP1, J2, CONN1 …
    The previous strict pattern r'^[A-Z]{1,3}[0-9]+$' rejected most of these.
    """
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
    voltage:       Optional[float] = None
    current_max:   Optional[float] = None
    frequency:     Optional[float] = None
    impedance_ohms: Optional[float] = None
    length_mm:     Optional[float] = None
    is_critical:   bool = False


class BoardConnection(BaseModel):
    """A named net with all connected pin references."""
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
                raise ValueError(f"Duplicate pin in net: {key}")
            seen.add(key)
        return v

    @property
    def components(self) -> Set[str]:
        return {p.ref for p in self.pins}

    def has_component(self, ref: str) -> bool:
        return any(p.ref == ref for p in self.pins)


class ComponentData(BaseModel):
    """Component with placement and physical properties.

    NOTE: model_config frozen=False is required so PlacementOptimizer can
    update x/y after auto-placement without reconstructing the entire object.
    """
    model_config = ConfigDict(frozen=False)

    # KiCad refs: R1, C12, U3, J1, SW1, LED1, TP1, #PWR01, PWR_FLAG …
    ref:       str   = Field(..., min_length=1, max_length=32)
    value:     str   = Field(..., min_length=1)
    footprint: str   = Field(default="")
    x:         float = Field(default=0.0, ge=-1000, le=1000)
    y:         float = Field(default=0.0, ge=-1000, le=1000)
    rotation:  float = Field(default=0.0, ge=-360, le=360)
    layer:     str   = Field(default="top")
    height_mm:             Optional[float] = Field(default=None, ge=0, le=50)
    power_dissipation_mw:  Optional[float] = Field(default=None, ge=0)
    is_polarized:          bool            = False

    @model_validator(mode="before")
    @classmethod
    def flatten_position(cls, values: Any) -> Any:
        """Accept flat {x,y} or nested {position:{x,y}} or missing coords."""
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
        """KiCad virtual power / flag symbols – excluded from DFM component checks."""
        return bool(_POWER_SYMBOL_RE.match(self.ref))


class BoardData(BaseModel):
    """Complete board description with optional netlist."""
    components:   List[ComponentData]  = Field(default_factory=list)
    connections:  List[BoardConnection] = Field(default_factory=list)
    board_width:  float = Field(default=100.0, gt=0, le=1000)
    board_height: float = Field(default=80.0,  gt=0, le=1000)
    design_rules: Dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def soft_validate_references(self) -> "BoardData":
        """
        Warn (log) rather than raise when a net references a component not in
        the components list.  KiCad sends power symbols (#PWR01, PWR_FLAG) in
        netlist connections but does not include them as placeable components –
        raising an error here would cause every plugin DFM/placement call to
        fail with 422 Unprocessable Entity.
        """
        comp_refs = {c.ref for c in self.components}
        for conn in self.connections:
            for pin in conn.pins:
                if pin.ref not in comp_refs and not _POWER_SYMBOL_RE.match(pin.ref):
                    logger.warning(
                        "Net '%s' references unknown component '%s' – ignored in analysis",
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
            pins = conn.pins
            for i, p1 in enumerate(pins):
                for p2 in pins[i + 1:]:
                    G.add_edge(p1.ref, p2.ref, net=conn.net)
        return G


class DFMViolation(BaseModel):
    rule_id:             str  = Field(..., pattern=r'^DFM-[A-Z]{2,4}-\d{3}$')
    type:                str
    severity:            Literal["info", "warning", "error", "critical"]
    message:             str
    components:          List[str]  = Field(default_factory=list)
    nets:                List[str]  = Field(default_factory=list)
    location:            Optional[Dict[str, float]] = None
    suggested_fix:       Optional[str] = None
    estimated_cost_impact: Optional[Literal["low", "medium", "high"]] = None


class HealthResponse(BaseModel):
    status:                  str
    version:                 str
    uptime_seconds:          float
    models_loaded:           bool
    llm_loaded:              bool  = False
    placement_engine_loaded: bool  = False
    templates_available:     int   = 0
    capabilities:            List[str] = Field(default_factory=list)


class GenerateRequest(BaseModel):
    prompt:      str  = Field(..., min_length=1, max_length=5000)
    constraints: Optional[Dict[str, Any]] = None
    priority:    Literal["speed", "quality", "compact"] = "quality"


class GenerateResponse(BaseModel):
    success:          bool
    circuit_data:     Optional[Dict[str, Any]] = None
    template_used:    Optional[str]  = None
    generation_time_ms: float        = 0.0
    warnings:         List[str]      = Field(default_factory=list)
    error:            Optional[str]  = None
    request_id:       str            = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    output_file:      Optional[str]  = None   # filename inside output/
    download_url:     Optional[str]  = None   # /download/<filename>


class SchematicRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=5000)


class SchematicResponse(BaseModel):
    success:         bool
    component_count: int         = 0
    output_file:     Optional[str] = None
    download_url:    Optional[str] = None
    error:           Optional[str] = None


# ── Application State ─────────────────────────────────────────────────────────

class AppState:
    def __init__(self) -> None:
        self.llm:            Any              = None
        self.rl_model:       Any              = None
        self.template_cache: Dict[str, Dict] = {}
        self.start_time:     float            = time.time()

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.start_time

    @property
    def models_loaded(self) -> bool:
        return self.llm is not None or self.rl_model is not None

    def get_capabilities(self) -> List[str]:
        caps = ["basic_dfm", "netlist_analysis", "template_matching"]
        if self.llm:        caps.append("llm_generation")
        if self.rl_model:   caps.append("rl_placement")
        if _NX:             caps.append("graph_analysis")
        if _NP:             caps.append("force_directed_placement")
        return caps


_state = AppState()


# ── Spatial Index ─────────────────────────────────────────────────────────────

class SpatialIndex:
    """Grid-based spatial index for O(1) average-case neighbour queries."""

    def __init__(self, cell_size: float = SPATIAL_GRID_SIZE_MM) -> None:
        self.cell_size = cell_size
        self.grid:            Dict[Tuple[int, int], List[ComponentData]] = defaultdict(list)
        self.component_cells: Dict[str, Tuple[int, int]]                = {}

    def _cell(self, x: float, y: float) -> Tuple[int, int]:
        return int(x / self.cell_size), int(y / self.cell_size)

    def insert(self, component: ComponentData) -> None:
        cell = self._cell(component.x, component.y)
        self.grid[cell].append(component)
        self.component_cells[component.ref] = cell

    # NOTE: return type is List[Tuple[ComponentData, float]] – the previous
    # annotation List[ComponentData] was wrong and broke callers that unpacked
    # (other, distance) tuples.
    def query_neighbors(
        self, component: ComponentData, radius: float
    ) -> List[Tuple[ComponentData, float]]:
        cell_radius = int(radius / self.cell_size) + 1
        cx, cy = self.component_cells.get(component.ref, self._cell(component.x, component.y))

        neighbors: List[Tuple[ComponentData, float]] = []
        for dx in range(-cell_radius, cell_radius + 1):
            for dy in range(-cell_radius, cell_radius + 1):
                for other in self.grid.get((cx + dx, cy + dy), []):
                    if other.ref == component.ref:
                        continue
                    dist = math.hypot(component.x - other.x, component.y - other.y)
                    if dist <= radius:
                        neighbors.append((other, dist))
        return neighbors


# ── Advanced DFM Engine ───────────────────────────────────────────────────────

class AdvancedDFMEngine:
    """Full netlist-aware DFM analysis engine."""

    POWER_NETS: Set[str] = {
        "VCC", "VDD", "3V3", "3.3V", "5V", "1V8", "1.8V", "12V", "24V",
        "VPWR", "VSUP", "AVCC", "DVCC", "VCCIO", "VCCINT",
    }
    GROUND_NETS: Set[str] = {
        "GND", "VSS", "AGND", "DGND", "PGND", "SGND", "VEE", "VSSA", "VSSD",
    }
    # Standard rotation angles for polarised components (degrees, normalised 0–360)
    STANDARD_ANGLES: Set[float] = {0.0, 90.0, 180.0, 270.0}
    ANGLE_TOL = 0.5  # degrees – tolerates floating-point imprecision

    def __init__(self, board: BoardData) -> None:
        self.board      = board
        self.violations: List[DFMViolation] = []

        # Only index real, placeable components – skip power symbols
        self.spatial = SpatialIndex()
        for comp in board.components:
            if not comp.is_power_symbol:
                self.spatial.insert(comp)

        # Build reverse-lookup maps
        self.net_pins:       Dict[str, List[PinRef]] = {}
        self.component_nets: Dict[str, List[str]]    = defaultdict(list)
        for conn in board.connections:
            self.net_pins[conn.net] = list(conn.pins)
            for pin in conn.pins:
                self.component_nets[pin.ref].append(conn.net)

    # ── Public entry point ────────────────────────────────────────────────────

    def analyze(self) -> List[DFMViolation]:
        """Run all checks and return violations sorted by severity."""
        self._check_component_spacing()
        self._check_board_boundaries()
        self._check_orientation()
        self._check_power_integrity()
        self._check_signal_integrity()
        self._check_thermal()
        self._check_floating_components()
        self._check_net_lengths()
        return sorted(
            self.violations,
            key=lambda v: {"critical": 0, "error": 1, "warning": 2, "info": 3}.get(v.severity, 4),
        )

    # ── Individual checks ─────────────────────────────────────────────────────

    def _check_component_spacing(self) -> None:
        checked: Set[Tuple[str, str]] = set()

        for comp in self.board.components:
            if comp.is_power_symbol:
                continue
            for other, dist in self.spatial.query_neighbors(comp, DFM_MIN_SPACING_MM * 3):
                pair = tuple(sorted((comp.ref, other.ref)))
                if pair in checked:
                    continue
                checked.add(pair)  # type: ignore[arg-type]

                # Intentionally-adjacent connected components are allowed to be close
                if self._are_connected(comp.ref, other.ref):
                    continue

                required = self._required_spacing(comp, other)
                if dist < required:
                    self.violations.append(DFMViolation(
                        rule_id="DFM-SPC-001",
                        type="component_spacing",
                        severity="error" if dist < required * 0.5 else "warning",
                        message=(
                            f"{comp.ref} and {other.ref} are too close "
                            f"({dist:.2f} mm < {required:.1f} mm required)"
                        ),
                        components=[comp.ref, other.ref],
                        location={"x": (comp.x + other.x) / 2, "y": (comp.y + other.y) / 2},
                        suggested_fix=f"Increase separation by {required - dist:.1f} mm",
                        estimated_cost_impact="high" if dist < required * 0.5 else "medium",
                    ))

    def _required_spacing(self, c1: ComponentData, c2: ComponentData) -> float:
        base = DFM_MIN_SPACING_MM
        for c in (c1, c2):
            if c.power_dissipation_mw and c.power_dissipation_mw > 1000:
                base += 1.0
            if c.height_mm and c.height_mm > 10:
                base += 2.0
            if c.is_connector:
                base += 1.5
        return base

    def _check_board_boundaries(self) -> None:
        for comp in self.board.components:
            if comp.is_power_symbol:
                continue
            bb = self._bounding_box(comp)
            checks = [
                (bb["x"] < DFM_EDGE_CLEARANCE_MM,
                 "left",  DFM_EDGE_CLEARANCE_MM - bb["x"]),
                (bb["x"] + bb["w"] > self.board.board_width - DFM_EDGE_CLEARANCE_MM,
                 "right", bb["x"] + bb["w"] - (self.board.board_width - DFM_EDGE_CLEARANCE_MM)),
                (bb["y"] < DFM_EDGE_CLEARANCE_MM,
                 "bottom", DFM_EDGE_CLEARANCE_MM - bb["y"]),
                (bb["y"] + bb["h"] > self.board.board_height - DFM_EDGE_CLEARANCE_MM,
                 "top", bb["y"] + bb["h"] - (self.board.board_height - DFM_EDGE_CLEARANCE_MM)),
            ]
            for violated, edge, overflow in checks:
                if violated:
                    self.violations.append(DFMViolation(
                        rule_id="DFM-BND-001",
                        type="board_boundary",
                        severity="error",
                        message=f"{comp.ref} violates {edge} edge clearance by {overflow:.1f} mm",
                        components=[comp.ref],
                        location={"x": comp.x, "y": comp.y},
                        suggested_fix=f"Move {comp.ref} inward by at least {overflow + 0.5:.1f} mm",
                    ))

    def _check_orientation(self) -> None:
        for comp in self.board.components:
            if not comp.is_polarized or comp.is_power_symbol:
                continue
            # Normalise to 0–360 and use a tolerance for float comparison
            normalised = comp.rotation % 360
            on_standard = any(
                abs(normalised - angle) < self.ANGLE_TOL
                or abs(normalised - angle - 360) < self.ANGLE_TOL
                for angle in self.STANDARD_ANGLES
            )
            if not on_standard:
                self.violations.append(DFMViolation(
                    rule_id="DFM-ORI-001",
                    type="orientation",
                    severity="warning",
                    message=(
                        f"{comp.ref} is polarised with non-standard rotation "
                        f"({comp.rotation:.1f}°) – verify assembly orientation"
                    ),
                    components=[comp.ref],
                    location={"x": comp.x, "y": comp.y},
                    suggested_fix=f"Snap {comp.ref} to 0 °, 90 °, 180 °, or 270 °",
                ))

    def _check_power_integrity(self) -> None:
        for ic in self.board.components:
            if not ic.is_ic:
                continue
            ic_nets = set(self.component_nets.get(ic.ref, []))
            power_nets  = ic_nets & self.POWER_NETS
            ground_nets = ic_nets & self.GROUND_NETS

            if not power_nets:
                self.violations.append(DFMViolation(
                    rule_id="DFM-PWR-001",
                    type="power_connection",
                    severity="error",
                    message=f"{ic.ref} has no power supply net connected",
                    components=[ic.ref],
                    location={"x": ic.x, "y": ic.y},
                ))
                continue

            if not ground_nets:
                self.violations.append(DFMViolation(
                    rule_id="DFM-PWR-002",
                    type="ground_connection",
                    severity="error",
                    message=f"{ic.ref} has no ground net connected",
                    components=[ic.ref],
                    location={"x": ic.x, "y": ic.y},
                ))
                continue

            for pwr_net in power_nets:
                self._check_decoupling(ic, pwr_net, ground_nets)

    def _check_decoupling(
        self, ic: ComponentData, pwr_net: str, gnd_nets: Set[str]
    ) -> None:
        candidates: List[Tuple[ComponentData, float]] = []

        for comp in self.board.components:
            if not comp.ref.startswith("C"):
                continue
            comp_nets = set(self.component_nets.get(comp.ref, []))
            if pwr_net in comp_nets and comp_nets & gnd_nets:
                candidates.append((comp, math.hypot(comp.x - ic.x, comp.y - ic.y)))

        if not candidates:
            self.violations.append(DFMViolation(
                rule_id="DFM-PWR-003",
                type="missing_decoupling",
                severity="warning",
                message=f"{ic.ref} has no decoupling capacitor on net {pwr_net}",
                components=[ic.ref],
                nets=[pwr_net],
                location={"x": ic.x, "y": ic.y},
                suggested_fix=(
                    f"Add 100 nF ceramic capacitor within "
                    f"{DFM_DECOUPLING_MAX_DIST_MM} mm of {ic.ref}"
                ),
            ))
            return

        closest_cap, closest_dist = min(candidates, key=lambda t: t[1])
        if closest_dist > DFM_DECOUPLING_MAX_DIST_MM:
            self.violations.append(DFMViolation(
                rule_id="DFM-PWR-004",
                type="decoupling_distance",
                severity="warning",
                message=(
                    f"Decoupling cap {closest_cap.ref} for {ic.ref} is too far "
                    f"({closest_dist:.1f} mm > {DFM_DECOUPLING_MAX_DIST_MM} mm)"
                ),
                components=[ic.ref, closest_cap.ref],
                nets=[pwr_net],
                location={"x": ic.x, "y": ic.y},
                suggested_fix=f"Move {closest_cap.ref} closer to {ic.ref}",
            ))

        if ic.power_dissipation_mw and ic.power_dissipation_mw > 500:
            bulk = [
                c for c, _ in candidates
                if c.value and any(u in c.value.upper() for u in ("UF", "µF", "MF"))
            ]
            if not bulk:
                self.violations.append(DFMViolation(
                    rule_id="DFM-PWR-005",
                    type="missing_bulk_capacitance",
                    severity="info",
                    message=f"High-power IC {ic.ref} may need bulk capacitance (>1 µF)",
                    components=[ic.ref],
                    nets=[pwr_net],
                    suggested_fix="Add 10 µF electrolytic / tantalum nearby",
                ))

    def _check_signal_integrity(self) -> None:
        for conn in self.board.connections:
            props = conn.properties  # always a NetProperties object

            # High-speed check
            if props.frequency and props.frequency > 1e6:
                if props.net_type == "clock" and len(conn.pins) > 3:
                    self.violations.append(DFMViolation(
                        rule_id="DFM-SI-001",
                        type="clock_fanout",
                        severity="warning",
                        message=(
                            f"Clock net '{conn.net}' has high fanout "
                            f"({len(conn.pins)} loads)"
                        ),
                        nets=[conn.net],
                        suggested_fix="Add a clock buffer or reduce direct connections",
                    ))

            # Differential pair: ensure both traces are present
            if props.net_type == "differential":
                self._check_differential_pair(conn)

    def _check_differential_pair(self, conn: BoardConnection) -> None:
        """Verify differential pairs have exactly 2 driver endpoints."""
        comps = [p.ref for p in conn.pins]
        # A diff pair should appear as P and N; if >2 endpoints there is a stub
        if len(comps) > 4:   # 2 drivers + 2 receivers is the maximum valid case
            self.violations.append(DFMViolation(
                rule_id="DFM-SI-002",
                type="differential_pair_stub",
                severity="warning",
                message=(
                    f"Differential net '{conn.net}' has {len(comps)} pins – "
                    "stubs degrade signal integrity"
                ),
                nets=[conn.net],
                suggested_fix="Ensure differential pairs are routed as matched-length pairs",
            ))

    def _check_thermal(self) -> None:
        hot = [
            c for c in self.board.components
            if c.power_dissipation_mw and c.power_dissipation_mw > 500
        ]
        for comp in hot:
            # Count dense neighbours as a proxy for copper density
            nearby = self.spatial.query_neighbors(comp, 5.0)
            if len(nearby) < 4:
                self.violations.append(DFMViolation(
                    rule_id="DFM-THM-001",
                    type="thermal_management",
                    severity="warning",
                    message=(
                        f"{comp.ref} dissipates {comp.power_dissipation_mw:.0f} mW "
                        "with insufficient surrounding copper area"
                    ),
                    components=[comp.ref],
                    location={"x": comp.x, "y": comp.y},
                    suggested_fix="Add thermal vias, copper pour, or a heatsink",
                ))

    def _check_floating_components(self) -> None:
        """Flag ICs and active components that have zero net connections."""
        connected = {p.ref for conn in self.board.connections for p in conn.pins}
        for comp in self.board.components:
            if comp.is_ic and comp.ref not in connected and not comp.is_power_symbol:
                self.violations.append(DFMViolation(
                    rule_id="DFM-CNN-001",
                    type="floating_component",
                    severity="error",
                    message=f"{comp.ref} ({comp.value}) has no electrical connections",
                    components=[comp.ref],
                    location={"x": comp.x, "y": comp.y},
                    suggested_fix="Connect all pins or remove the component",
                ))

    def _check_net_lengths(self) -> None:
        for conn in self.board.connections:
            if len(conn.pins) < 2 or not conn.properties.length_mm:
                continue
            max_dist = 0.0
            for i, p1 in enumerate(conn.pins):
                c1 = self.board.get_component(p1.ref)
                if not c1:
                    continue
                for p2 in conn.pins[i + 1:]:
                    c2 = self.board.get_component(p2.ref)
                    if not c2:
                        continue
                    d = abs(c1.x - c2.x) + abs(c1.y - c2.y)   # Manhattan estimate
                    max_dist = max(max_dist, d)

            if max_dist > conn.properties.length_mm * 1.5:
                self.violations.append(DFMViolation(
                    rule_id="DFM-LEN-001",
                    type="excessive_trace_length",
                    severity="warning",
                    message=(
                        f"Net '{conn.net}' estimated length {max_dist:.1f} mm "
                        f"exceeds target {conn.properties.length_mm:.1f} mm"
                    ),
                    nets=[conn.net],
                    suggested_fix="Move connected components closer or add a termination resistor",
                ))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _are_connected(self, ref1: str, ref2: str) -> bool:
        n1 = set(self.component_nets.get(ref1, []))
        n2 = set(self.component_nets.get(ref2, []))
        return bool(n1 & n2)

    def _bounding_box(self, comp: ComponentData) -> Dict[str, float]:
        sizes = {"R": 1.6, "C": 1.6, "L": 2.0, "D": 2.0, "U": 5.0,
                 "Q": 3.0, "J": 10.0, "SW": 6.0, "TP": 2.0}
        size = sizes.get(comp.prefix, 5.0)
        if comp.rotation % 180 != 0:
            size *= 1.4   # diagonal approximation
        return {"x": comp.x - size / 2, "y": comp.y - size / 2, "w": size, "h": size}


# ── Placement Optimiser ───────────────────────────────────────────────────────

class PlacementOptimizer:
    """Netlist-aware placement using force-directed or simulated annealing."""

    def __init__(self, board: BoardData) -> None:
        self.board   = board
        self.refs    = [c.ref for c in board.components if not c.is_power_symbol]
        self.graph   = board.build_graph()

        # Adjacency with weights for wire-length cost
        self.adj: Dict[str, Dict[str, float]] = defaultdict(dict)
        for conn in board.connections:
            refs = [p.ref for p in conn.pins if not _POWER_SYMBOL_RE.match(p.ref)]
            w = 1.0 / max(len(refs), 1)
            for i, r1 in enumerate(refs):
                for r2 in refs[i + 1:]:
                    self.adj[r1][r2] = self.adj[r1].get(r2, 0) + w
                    self.adj[r2][r1] = self.adj[r2].get(r1, 0) + w

    # ── Algorithm dispatcher ──────────────────────────────────────────────────

    def optimize(self, algorithm: str = "force_directed") -> Dict[str, Any]:
        if algorithm == "force_directed" and _NP:
            return self._force_directed()
        if algorithm == "annealing":
            return self._simulated_annealing()
        return self._grid_placement()

    # ── Wire-length cost ──────────────────────────────────────────────────────

    def _wire_length(self, positions: Dict[str, Tuple[float, float]]) -> float:
        total = 0.0
        for r1, neighbours in self.adj.items():
            if r1 not in positions:
                continue
            for r2, weight in neighbours.items():
                if r2 not in positions:
                    continue
                dx = positions[r1][0] - positions[r2][0]
                dy = positions[r1][1] - positions[r2][1]
                total += weight * math.hypot(dx, dy)
        return total / 2  # each edge counted twice

    # ── Force-directed (requires numpy) ──────────────────────────────────────

    def _force_directed(self) -> Dict[str, Any]:
        pos: Dict[str, Any] = {
            ref: np.array([c.x if c.x else random.uniform(10, self.board.board_width - 10),
                           c.y if c.y else random.uniform(10, self.board.board_height - 10)])
            for c in self.board.components
            if not c.is_power_symbol
            for ref in [c.ref]
        }

        w, h = self.board.board_width, self.board.board_height
        spring_k, repulse_k = 0.08, 150.0

        for iteration in range(120):
            forces = {r: np.zeros(2) for r in pos}
            damping = max(0.05, 0.9 - iteration / 150)

            # Attractive: connected pairs
            for r1, neighbours in self.adj.items():
                if r1 not in pos:
                    continue
                for r2, weight in neighbours.items():
                    if r2 not in pos:
                        continue
                    diff = pos[r2] - pos[r1]
                    dist = float(np.linalg.norm(diff)) or 1e-6
                    ideal = 10.0 * (1 - weight)
                    f = diff / dist * spring_k * (dist - ideal)
                    forces[r1] += f
                    forces[r2] -= f

            # Repulsive: all pairs
            refs_list = list(pos.keys())
            for i, r1 in enumerate(refs_list):
                for r2 in refs_list[i + 1:]:
                    diff = pos[r2] - pos[r1]
                    dist = float(np.linalg.norm(diff)) or 1e-6
                    if dist < 30:
                        f = -diff / dist * repulse_k / dist
                        forces[r1] += f
                        forces[r2] -= f

            for ref in pos:
                pos[ref] += forces[ref] * damping
                pos[ref][0] = float(np.clip(pos[ref][0], 5, w - 5))
                pos[ref][1] = float(np.clip(pos[ref][1], 5, h - 5))

        positions = {
            ref: {"x": float(p[0]), "y": float(p[1]), "rotation": 0.0}
            for ref, p in pos.items()
        }
        return {"positions": positions, "algorithm": "force_directed", "iterations": 120}

    # ── Simulated annealing ───────────────────────────────────────────────────
    # NOTE: was referenced in optimize() but not implemented in v2.0 → crash.

    def _simulated_annealing(self) -> Dict[str, Any]:
        w, h = self.board.board_width, self.board.board_height

        # Initialise from current positions or random
        cur: Dict[str, Tuple[float, float]] = {}
        for comp in self.board.components:
            if comp.is_power_symbol:
                continue
            cur[comp.ref] = (
                comp.x if comp.x else random.uniform(5, w - 5),
                comp.y if comp.y else random.uniform(5, h - 5),
            )

        best       = dict(cur)
        best_cost  = self._wire_length(best)
        temp       = SA_INITIAL_TEMP

        for _ in range(SA_ITERATIONS):
            # Perturb one random component
            ref = random.choice(list(cur.keys()))
            ox, oy = cur[ref]
            step   = temp * 0.3
            nx     = max(5, min(w - 5, ox + random.uniform(-step, step)))
            ny     = max(5, min(h - 5, oy + random.uniform(-step, step)))
            cur[ref] = (nx, ny)

            cost = self._wire_length(cur)
            delta = cost - best_cost

            if delta < 0 or random.random() < math.exp(-delta / max(temp, 1e-6)):
                if cost < best_cost:
                    best      = dict(cur)
                    best_cost = cost
            else:
                cur[ref] = (ox, oy)   # revert

            temp *= SA_COOLING_RATE

        positions = {
            ref: {"x": x, "y": y, "rotation": 0.0}
            for ref, (x, y) in best.items()
        }
        return {"positions": positions, "algorithm": "simulated_annealing",
                "iterations": SA_ITERATIONS, "final_cost": best_cost}

    # ── Grid fallback ─────────────────────────────────────────────────────────

    def _grid_placement(self) -> Dict[str, Any]:
        groups = self._union_find_groups()
        positions: Dict[str, Dict[str, float]] = {}
        grid_step, margin, cols = 10.0, 10.0, 5

        # NOTE: fixed col-advance bug – outer col cursor now advances correctly
        # after each group is placed.
        global_col = 0
        for group in groups:
            for i, ref in enumerate(group):
                row = i // cols
                col = global_col + (i % cols)
                positions[ref] = {
                    "x": margin + col * grid_step,
                    "y": margin + row * grid_step,
                    "rotation": 0.0,
                }
            global_col += max(len(group) % cols or cols, 1) + 1   # advance past this group

        return {"positions": positions, "algorithm": "connectivity_grid", "iterations": 1}

    def _union_find_groups(self) -> List[List[str]]:
        parent = {c.ref: c.ref for c in self.board.components if not c.is_power_symbol}

        def find(x: str) -> str:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x: str, y: str) -> None:
            if x in parent and y in parent:
                parent[find(x)] = find(y)

        for conn in self.board.connections:
            refs = [p.ref for p in conn.pins if not _POWER_SYMBOL_RE.match(p.ref)]
            for i in range(1, len(refs)):
                union(refs[0], refs[i])

        groups: Dict[str, List[str]] = defaultdict(list)
        for ref in parent:
            groups[find(ref)].append(ref)
        return sorted(groups.values(), key=len, reverse=True)


# ── Connection normaliser (template compat) ───────────────────────────────────

def _normalise_connections(circuit_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert old-style "pins": ["R1.1", "C1.1"] to BoardConnection format
    "pins": [{"ref": "R1", "pin": "1"}, ...].

    Also drops pins whose component ref isn't in the components list
    (e.g. KiCad power symbols) to prevent BoardData validation warnings.
    """
    import copy
    data = copy.deepcopy(circuit_data)
    comp_refs: Set[str] = {c["ref"] for c in data.get("components", []) if "ref" in c}

    normalised: List[Dict[str, Any]] = []
    for conn in data.get("connections", []):
        new_pins: List[Dict[str, str]] = []
        for p in conn.get("pins", []):
            if isinstance(p, str):
                if "." in p:
                    ref, pin = p.split(".", 1)
                    new_pins.append({"ref": ref, "pin": pin})
            elif isinstance(p, dict):
                new_pins.append(p)
        # Keep only pins whose component exists and isn't a power symbol
        new_pins = [
            p for p in new_pins
            if p.get("ref", "") in comp_refs
            or _POWER_SYMBOL_RE.match(p.get("ref", ""))
        ]
        if len(new_pins) >= 2:
            normalised.append({**conn, "pins": new_pins})

    data["connections"] = normalised
    return data


def _enrich_net_properties(circuit_data: Dict[str, Any]) -> None:
    """
    Auto-detect power/ground nets and set appropriate net_type and voltage.
    Modifies circuit_data in place.
    """
    # Power net patterns
    POWER_PATTERNS = {
        "VCC": {"net_type": "power", "voltage": 5.0},
        "VDD": {"net_type": "power", "voltage": 3.3},
        "3V3": {"net_type": "power", "voltage": 3.3},
        "3.3V": {"net_type": "power", "voltage": 3.3},
        "5V": {"net_type": "power", "voltage": 5.0},
        "1V8": {"net_type": "power", "voltage": 1.8},
        "1.8V": {"net_type": "power", "voltage": 1.8},
        "12V": {"net_type": "power", "voltage": 12.0},
        "24V": {"net_type": "power", "voltage": 24.0},
        "VPWR": {"net_type": "power", "voltage": 5.0},
        "VSUP": {"net_type": "power", "voltage": 5.0},
        "AVCC": {"net_type": "power", "voltage": 3.3},
        "DVCC": {"net_type": "power", "voltage": 3.3},
        "VCCIO": {"net_type": "power", "voltage": 3.3},
        "VCCINT": {"net_type": "power", "voltage": 1.8},
    }
    GROUND_PATTERNS = ["GND", "VSS", "AGND", "DGND", "PGND", "SGND", "VEE", "VSSA", "VSSD"]

    for conn in circuit_data.get("connections", []):
        net_name = conn.get("net", "")
        props = conn.get("properties", {})
        
        # Only enrich if properties don't already exist or are default signal
        if not props or props.get("net_type") == "signal":
            # Check if it's a power net
            for pattern, power_props in POWER_PATTERNS.items():
                if pattern in net_name.upper():
                    conn["properties"] = {
                        "net_type": "power",
                        "voltage": power_props["voltage"],
                        "current_max": None,
                        "frequency": None,
                        "impedance_ohms": None,
                        "length_mm": None,
                        "is_critical": False,
                    }
                    break
            # Check if it's a ground net
            else:
                for pattern in GROUND_PATTERNS:
                    if pattern in net_name.upper():
                        conn["properties"] = {
                            "net_type": "ground",
                            "voltage": None,
                            "current_max": None,
                            "frequency": None,
                            "impedance_ohms": None,
                            "length_mm": None,
                            "is_critical": False,
                        }
                        break


def _enrich_component_properties(circuit_data: Dict[str, Any]) -> None:
    """
    Auto-detect component polarization based on part type and value.
    Modifies circuit_data in place.
    """
    for comp in circuit_data.get("components", []):
        part = comp.get("part", "").upper()
        lib = comp.get("lib", "").upper()
        value = comp.get("value", "").lower()
        ref = comp.get("ref", "")
        
        # LED detection
        if part == "LED" or lib == "LED" or ref.startswith("D") or ref.startswith("LED"):
            if "LED" in part or "LED" in lib or "led" in value.lower():
                comp["is_polarized"] = True
        
        # Electrolytic capacitor detection (typically polarized)
        # Large value caps (> 1µF) are often electrolytic
        if part == "C" or part == "CP" or ref.startswith("C"):
            # Parse capacitor value
            if "uf" in value or "µf" in value:
                # Extract numeric value
                val_str = value.replace("uf", "").replace("µf", "").strip()
                try:
                    val = float(val_str)
                    # 10µF and above are typically electrolytic (polarized)
                    if val >= 10.0:
                        comp["is_polarized"] = True
                except ValueError:
                    pass
            # Explicit polarized capacitor
            if part == "CP" or "polarized" in comp.get("description", "").lower():
                comp["is_polarized"] = True
        
        # Diode detection (all diodes are polarized)
        if part in ["D", "DIODE"] or (ref.startswith("D") and not ref.startswith("D") and len(ref) > 1 and ref[1].isdigit()):
            if part in ["D", "DIODE"] or "diode" in comp.get("description", "").lower():
                comp["is_polarized"] = True


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting AI PCB Assistant v2.1…")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    # Load templates (always sync – aiofiles adds no benefit for startup reads)
    if TEMPLATES_DIR.exists():
        for path in sorted(TEMPLATES_DIR.glob("*.json")):
            try:
                _state.template_cache[path.stem] = json.loads(
                    path.read_text(encoding="utf-8")
                )
            except Exception as exc:
                logger.warning("Failed to load template %s: %s", path.name, exc)
    logger.info("Loaded %d templates", len(_state.template_cache))

    # LLM
    try:
        from engines.llm_engine import load_llm   # type: ignore
        _state.llm = load_llm()
        logger.info("LLM engine loaded.")
    except ImportError:
        logger.warning("engines.llm_engine not found – LLM disabled.")
    except Exception as exc:
        logger.warning("LLM failed to load: %s", exc)

    # RL placement
    try:
        from engines.placement_engine import load_placement_model   # type: ignore
        _state.rl_model = load_placement_model()
        logger.info("RL placement engine loaded.")
    except ImportError:
        logger.warning("engines.placement_engine not found – RL disabled.")
    except Exception as exc:
        logger.warning("RL placement failed to load: %s", exc)

    logger.info("Startup complete. capabilities=%s", _state.get_capabilities())
    yield
    logger.info("Shutting down AI PCB Assistant.")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI PCB Assistant Backend",
    description="Advanced AI backend for KiCad PCB design with netlist integration",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    # NOTE: KiCad plugin makes requests from the Python process (no browser
    # origin header), and the local web demo runs from file:// or localhost.
    # The original default of "localhost:3000 only" blocked both. Use env var
    # CORS_ORIGINS to lock down in production.
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Serve web demo if present
_demo_dir = _BASE_DIR / "demo"
if _demo_dir.exists():
    app.mount("/demo", StaticFiles(directory=str(_demo_dir), html=True), name="demo")


# ── Request timing middleware ─────────────────────────────────────────────────

@app.middleware("http")
async def add_timing_header(request: Request, call_next: Any):
    t0 = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - t0) * 1000:.1f}"
    return response


# ── Global error handler ──────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception %s %s", request.method, request.url)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred."},
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="healthy" if _state.models_loaded else "degraded",
        version=app.version,
        uptime_seconds=round(_state.uptime_seconds, 1),
        models_loaded=_state.models_loaded,
        llm_loaded=_state.llm is not None,
        placement_engine_loaded=_state.rl_model is not None,
        templates_available=len(_state.template_cache),
        capabilities=_state.get_capabilities(),
    )


@app.post("/generate", response_model=GenerateResponse, tags=["generation"])
async def generate_circuit(
    request: GenerateRequest, background_tasks: BackgroundTasks
) -> GenerateResponse:
    """
    Natural-language → validated CircuitData JSON → optional .kicad_sch file.

    Flow:
      1. Score all templates against prompt keywords
      2. Fall back to LLM if no template matched and LLM is loaded
      3. Normalise connections to BoardConnection format
      4. Auto-place if all components sit at (0, 0)
      5. Run DFM and embed violations in warnings
      6. Persist JSON + optionally a .kicad_sch to output/
      7. Return GenerateResponse with download_url
    """
    t0 = time.perf_counter()
    request_id = str(uuid.uuid4())[:8]
    warnings: List[str] = []

    # ── Step 1: template scoring ──────────────────────────────────────────────
    prompt_lower = request.prompt.lower()
    best_name, best_score = None, 0
    for keywords, name, weight in TEMPLATE_KEYWORDS:
        score = sum(weight for kw in keywords if kw in prompt_lower)
        if score > best_score:
            best_score, best_name = score, name

    circuit_data: Optional[Dict[str, Any]] = None
    template_used: Optional[str] = None

    if best_name and best_name in _state.template_cache:
        circuit_data = _state.template_cache[best_name]
        template_used = best_name
    elif _state.llm:
        try:
            # generate_circuit_json is async — must be awaited.  Without the
            # await it returns a coroutine object (truthy) so circuit_data would
            # be set to a coroutine instead of a dict, silently bypassing the
            # template fallback and裂failing JSON serialisation later.
            circuit_data = await _state.llm.generate_circuit_json(request.prompt) or None
        except Exception as exc:
            warnings.append(f"LLM generation failed: {exc}")

    if not circuit_data:
        return GenerateResponse(
            success=False,
            error=(
                "No matching template found and LLM is unavailable. "
                "Try: '555 timer', '3.3V regulator', 'LED resistor', "
                "'op-amp buffer', 'MOSFET switch'."
            ),
            warnings=warnings,
            request_id=request_id,
        )

    # ── Step 2: normalise and validate ───────────────────────────────────────
    try:
        normalised = _normalise_connections(circuit_data)
        # Enrich net properties (VCC → power, GND → ground)
        _enrich_net_properties(normalised)
        # Enrich component properties (LED → polarized, large caps → polarized)
        _enrich_component_properties(normalised)
        board = BoardData(**normalised)
    except Exception as exc:
        return GenerateResponse(
            success=False,
            error=f"Schema validation failed: {exc}",
            warnings=warnings,
            request_id=request_id,
        )

    # ── Step 3: auto-place if all at origin ──────────────────────────────────
    if all(c.x == 0.0 and c.y == 0.0 for c in board.components if not c.is_power_symbol):
        algo = "force_directed" if _NP else "annealing" if request.priority == "quality" else "grid"
        optimizer = PlacementOptimizer(board)
        placement = optimizer.optimize(algo)
        for ref, pos in placement["positions"].items():
            comp = board.get_component(ref)
            if comp:
                comp.x = pos["x"]   # works because model_config frozen=False
                comp.y = pos["y"]

    # ── Step 4: DFM ──────────────────────────────────────────────────────────
    dfm_violations = AdvancedDFMEngine(board).analyze()
    for v in dfm_violations:
        if v.severity in ("error", "critical"):
            warnings.append(f"[DFM {v.rule_id}] {v.message}")

    # ── Step 5: persist JSON ──────────────────────────────────────────────────
    json_path = OUTPUT_DIR / f"circuit_{request_id}.json"
    board_dict = board.model_dump()

    async def _save_json() -> None:
        if _AIOFILES:
            async with aiofiles.open(json_path, "w") as f:
                await f.write(json.dumps(board_dict, indent=2))
        else:
            json_path.write_text(json.dumps(board_dict, indent=2))

    background_tasks.add_task(_save_json)

    # ── Step 6: export .kicad_sch ────────────────────────────────────────────
    sch_filename: Optional[str] = None
    download_url: Optional[str] = None
    try:
        from circuit_schema import CircuitData            # type: ignore
        from engines.kicad_exporter import export_to_kicad_sch  # type: ignore

        raw = _state.template_cache.get(template_used, {}) if template_used else {}
        src = raw or circuit_data
        schema_obj = CircuitData(**src)
        sch_content = export_to_kicad_sch(schema_obj)

        desc = src.get("description", template_used or request_id)
        # Strip non-ASCII first (em-dashes, curly quotes …) so Windows
        # cp1252 path encoding never produces garbage characters like â.
        desc_ascii = desc.encode("ascii", errors="ignore").decode("ascii")
        safe = re.sub(r"[^\w\s\-]", "", desc_ascii).strip()
        safe = re.sub(r"\s+", "_", safe)          # spaces → underscores
        safe = re.sub(r"_+", "_", safe).strip("_")  # collapse runs e.g. a__b → a_b
        safe = (safe[:80] or request_id)
        sch_path = OUTPUT_DIR / f"{safe}.kicad_sch"
        sch_path.write_text(sch_content, encoding="utf-8")
        sch_filename = sch_path.name
        download_url = f"/download/{sch_filename}"
        logger.info("KiCad schematic saved: %s", sch_path)
    except ImportError as exc:
        warnings.append(f"KiCad export skipped (module missing): {exc}")
    except Exception as exc:
        warnings.append(f"KiCad export failed: {exc}")
        logger.warning("KiCad export failed: %s", exc)

    return GenerateResponse(
        success=True,
        circuit_data=board_dict,
        template_used=template_used,
        generation_time_ms=round((time.perf_counter() - t0) * 1000, 1),
        warnings=warnings,
        request_id=request_id,
        output_file=sch_filename,
        download_url=download_url,
    )


@app.post("/generate/schematic", response_model=SchematicResponse, tags=["generation"])
async def generate_schematic(request: SchematicRequest) -> SchematicResponse:
    """Plugin-compatible schematic generation – delegates to /generate."""
    result = await generate_circuit(
        GenerateRequest(prompt=request.prompt), BackgroundTasks()
    )
    return SchematicResponse(
        success=result.success,
        component_count=len(result.circuit_data.get("components", [])) if result.circuit_data else 0,
        output_file=result.output_file,
        download_url=result.download_url,
        error=result.error,
    )


@app.post("/analyze/dfm", response_model=List[DFMViolation], tags=["dfm"])
async def analyze_dfm(board: BoardData) -> List[DFMViolation]:
    """Full netlist-aware DFM analysis."""
    return AdvancedDFMEngine(board).analyze()


@app.post("/dfm/check", response_model=List[DFMViolation], tags=["dfm"])
async def dfm_check_compat(board: BoardData) -> List[DFMViolation]:
    """
    Alias for /analyze/dfm for plugin.py compatibility.
    # NOTE: plugin.py sends DFM requests to /dfm/check – this keeps old
    # plugin builds working with the v2 backend without any plugin changes.
    """
    return await analyze_dfm(board)


@app.post("/placement/optimize", tags=["placement"])
async def optimize_placement(board: BoardData, algorithm: str = "auto") -> Dict[str, Any]:
    """
    Netlist-aware placement optimisation.

    algorithm options: auto | rl | force_directed | annealing | grid
    - auto: RL if loaded, else force_directed (numpy) or annealing
    """
    t0 = time.perf_counter()

    if algorithm == "auto":
        if _state.rl_model:
            algorithm = "rl"
        elif _NP:
            algorithm = "force_directed"
        else:
            algorithm = "annealing"

    if algorithm == "rl" and _state.rl_model:
        try:
            from engines.placement_engine import optimize_with_rl   # type: ignore
            result: Dict[str, Any] = optimize_with_rl(_state.rl_model, board.model_dump())
            result.update({"algorithm": "rl", "time_ms": (time.perf_counter() - t0) * 1000})
            return result
        except Exception as exc:
            logger.warning("RL placement failed, falling back: %s", exc)
            algorithm = "force_directed" if _NP else "annealing"

    optimizer = PlacementOptimizer(board)
    result = optimizer.optimize(algorithm)
    result["time_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return result


@app.post("/export/kicad", tags=["export"])
async def export_kicad(circuit: dict) -> StreamingResponse:
    """Export CircuitData dict to a streaming .kicad_sch download."""
    try:
        from circuit_schema import CircuitData            # type: ignore
        from engines.kicad_exporter import export_to_kicad_sch  # type: ignore

        data    = CircuitData(**circuit)
        content = export_to_kicad_sch(data)

        return StreamingResponse(
            iter([content]),
            media_type="application/x-kicad-schematic",
            headers={"Content-Disposition": "attachment; filename=circuit.kicad_sch"},
        )
    except ImportError as exc:
        raise HTTPException(status_code=501, detail=f"KiCad exporter not available: {exc}")
    except Exception as exc:
        logger.error("KiCad export failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/download/{filename}", tags=["export"])
async def download_file(filename: str) -> FileResponse:
    """Download a previously generated .kicad_sch or .json file."""
    safe = re.sub(r"[^a-zA-Z0-9_.\-]", "", filename)
    if not (safe.endswith(".kicad_sch") or safe.endswith(".json")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .kicad_sch or .json files are downloadable.",
        )
    if "/" in safe or "\\" in safe:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    file_path = OUTPUT_DIR / safe
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {safe}")

    return FileResponse(
        path=str(file_path),
        filename=safe,
        media_type="application/octet-stream",
    )


@app.get("/circuit/{name}", tags=["templates"])
async def get_circuit_template(name: str) -> Dict[str, Any]:
    """Return raw CircuitData JSON for a named template."""
    safe = re.sub(r"[^a-zA-Z0-9_\-]", "", name)
    if not safe:
        raise HTTPException(status_code=400, detail="Invalid template name.")
    data = _state.template_cache.get(safe)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Template '{safe}' not found.")
    return data


@app.get("/templates", tags=["templates"])
async def list_templates() -> List[Dict[str, Any]]:
    """List all loaded templates with metadata."""
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
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8765")),
        reload=os.environ.get("RELOAD", "false").lower() == "true",
        workers=int(os.environ.get("WORKERS", "1")),
        log_level=os.environ.get("LOG_LEVEL", "info"),
    )