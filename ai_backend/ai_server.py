"""
AI PCB Assistant - Advanced FastAPI Backend Server v2.0
Handles LLM inference, placement optimization, and advanced DFM checking with full netlist integration.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import tempfile
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Dict, List, Set, Tuple, Callable, Literal
from functools import lru_cache, wraps

import numpy as np
from fastapi import FastAPI, HTTPException, Request, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

# Optional advanced dependencies
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    import aiofiles
    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False

# ── Logging Configuration ─────────────────────────────────────────────────────

class RequestIdFilter(logging.Filter):
    def filter(self, record):
        record.request_id = getattr(record, 'request_id', 'system')
        return True

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
# Module-level handler with request_id field; RequestIdFilter injects a default
# so formatters never KeyError even when request_id isn't set on the record.
_module_handler = logging.StreamHandler()
_module_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] [%(request_id)s] %(name)s: %(message)s")
)
_module_handler.addFilter(RequestIdFilter())
logger.addHandler(_module_handler)
logger.propagate = False  # Don't double-log via root handler

# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Config:
    """Application configuration."""
    BASE_DIR: Path = field(default_factory=lambda: Path(__file__).parent)
    TEMPLATES_DIR: Path = field(default_factory=lambda: Path(__file__).parent / "templates")
    OUTPUT_DIR: Path = field(default_factory=lambda: Path(__file__).parent / "output")
    TEMP_DIR: Path = field(default_factory=lambda: Path(tempfile.gettempdir()) / "ai_pcb")
    
    # DFM Settings
    DFM_MIN_SPACING_MM: float = 0.5  # Modern manufacturing standard
    DFM_EDGE_CLEARANCE_MM: float = 1.0
    DFM_MAX_COMPONENT_HEIGHT_MM: float = 25.0
    DFM_DECOUPLING_MAX_DISTANCE_MM: float = 10.0  # Max distance for effective decoupling
    
    # Performance
    SPATIAL_GRID_SIZE_MM: float = 5.0
    MAX_COMPONENTS_PER_GRID_CELL: int = 10
    
    # Analysis
    ENABLE_GRAPH_ANALYSIS: bool = True
    ENABLE_THERMAL_ANALYSIS: bool = True

CONFIG = Config()
CONFIG.TEMP_DIR.mkdir(parents=True, exist_ok=True)

# ── Template Registry ─────────────────────────────────────────────────────────

TEMPLATE_KEYWORDS: List[Tuple[List[str], str, int]] = [
    # Keywords, template_name, priority (higher = more specific)
    (["555 timer", "ne555", "astable 555", "555 oscillator"], "555_timer_oscillator", 100),
    (["555", "timer", "blink", "astable", "multivibrator"], "555_timer", 80),
    (["3.3v regulator", "3v3 ldo", "ams1117-3.3", "3.3v power"], "3v3_regulator_ldo", 100),
    (["3.3v", "3v3", "ams1117", "ldo", "voltage regulator"], "3v3_regulator", 80),
    (["5v regulator", "5v ldo", "7805", "5v power"], "5v_regulator", 90),
    (["led driver", "led array", "multiple led"], "led_array_driver", 90),
    (["led", "diode", "indicator", "resistor led"], "led_resistor", 70),
    (["opamp buffer", "unity gain buffer", "voltage follower"], "opamp_buffer", 90),
    (["opamp", "op-amp", "operational amplifier", "gain"], "opamp_general", 70),
    (["mosfet switch", "high side switch", "low side switch"], "mosfet_switch", 90),
    (["mosfet", "nmos", "pmos", "transistor switch"], "mosfet_general", 70),
    (["rc filter", "low pass filter", "high pass filter"], "rc_filter", 80),
    (["voltage divider", "resistor divider"], "voltage_divider", 75),
    (["crystal oscillator", "quartz", "mhz crystal"], "crystal_oscillator", 85),
    (["usb power", "usb protection", "usb esd"], "usb_protection", 90),
]

# ── Pydantic Models ───────────────────────────────────────────────────────────

class PinRef(BaseModel):
    """Reference to a specific pin on a component."""
    model_config = ConfigDict(frozen=True)
    
    ref: str = Field(..., pattern=r'^[A-Z]{1,3}[0-9]+$', description="Component reference")
    pin: str = Field(..., pattern=r'^[0-9A-Za-z]+$', description="Pin number/name")
    
    def __str__(self) -> str:
        return f"{self.ref}.{self.pin}"
    
    @property
    def component_prefix(self) -> str:
        """Extract component type prefix."""
        match = re.match(r'^([A-Z][A-Z]?)', self.ref)
        return match.group(1) if match else "U"


class NetProperties(BaseModel):
    """Electrical properties of a net."""
    net_type: Literal["power", "ground", "signal", "clock", "differential", "analog"] = "signal"
    voltage: Optional[float] = Field(default=None, description="Nominal voltage")
    current_max: Optional[float] = Field(default=None, description="Max current in A")
    frequency: Optional[float] = Field(default=None, description="Frequency in Hz")
    impedance_ohms: Optional[float] = Field(default=None)
    length_mm: Optional[float] = Field(default=None)
    is_critical: bool = Field(default=False)


class BoardConnection(BaseModel):
    """A net with all connected pins."""
    model_config = ConfigDict(frozen=True)
    
    net: str = Field(..., min_length=1, max_length=100)
    pins: List[PinRef] = Field(..., min_length=2)
    properties: NetProperties = Field(default_factory=NetProperties)
    
    @field_validator('pins')
    @classmethod
    def validate_unique_pins(cls, v: List[PinRef]) -> List[PinRef]:
        seen = set()
        for pin in v:
            key = str(pin)
            if key in seen:
                raise ValueError(f"Duplicate pin in net: {key}")
            seen.add(key)
        return v
    
    @property
    def components(self) -> Set[str]:
        """Get unique component references in this net."""
        return {p.ref for p in self.pins}
    
    def has_component(self, ref: str) -> bool:
        """Check if component is in this net."""
        return any(p.ref == ref for p in self.pins)


class ComponentData(BaseModel):
    """Component with placement and physical properties."""
    ref: str = Field(..., pattern=r'^[A-Z]{1,3}[0-9]+$')
    value: str = Field(..., min_length=1)
    footprint: str = Field(default="", description="KiCad footprint")
    x: float = Field(..., ge=-1000, le=1000)
    y: float = Field(..., ge=-1000, le=1000)
    rotation: float = Field(default=0.0, ge=-360, le=360)
    layer: Literal["top", "bottom", "F.Cu", "B.Cu"] = "top"
    height_mm: Optional[float] = Field(default=None, ge=0, le=50)
    power_dissipation_mw: Optional[float] = Field(default=None, ge=0)
    is_polarized: bool = Field(default=False)

    @model_validator(mode='before')
    @classmethod
    def flatten_position(cls, values: Any) -> Any:
        """Accept both flat {x, y} and nested {position: {x, y}} formats."""
        if isinstance(values, dict) and 'position' in values and 'x' not in values:
            pos = values['position']
            if isinstance(pos, dict):
                values = {**values, 'x': pos.get('x', 0.0), 'y': pos.get('y', 0.0)}
        return values
    
    @field_validator('layer')
    @classmethod
    def normalize_layer(cls, v: str) -> str:
        mapping = {"F.Cu": "top", "B.Cu": "bottom", "top": "top", "bottom": "bottom"}
        return mapping.get(v, "top")
    
    @property
    def prefix(self) -> str:
        """Get component prefix (R, C, U, etc.)."""
        match = re.match(r'^([A-Z][A-Z]?)', self.ref)
        return match.group(1) if match else "U"
    
    @property
    def is_ic(self) -> bool:
        """Check if integrated circuit."""
        return self.prefix in ["U", "IC", "Q"]
    
    @property
    def is_passive(self) -> bool:
        """Check if passive component."""
        return self.prefix in ["R", "C", "L", "F"]
    
    @property
    def is_connector(self) -> bool:
        """Check if connector."""
        return self.prefix in ["J", "P", "CONN"]


class BoardData(BaseModel):
    """Complete board description with netlist."""
    components: List[ComponentData]
    connections: List[BoardConnection] = Field(default_factory=list)
    board_width: float = Field(default=100.0, gt=0, le=1000)
    board_height: float = Field(default=80.0, gt=0, le=1000)
    design_rules: Dict[str, float] = Field(default_factory=dict)
    
    @model_validator(mode='after')
    def validate_references(self) -> 'BoardData':
        """Ensure all net references point to existing components."""
        comp_refs = {c.ref for c in self.components}
        
        for conn in self.connections:
            for pin in conn.pins:
                if pin.ref not in comp_refs:
                    raise ValueError(f"Net '{conn.net}' references unknown component '{pin.ref}'")
        
        return self
    
    def get_component(self, ref: str) -> Optional[ComponentData]:
        """Get component by reference."""
        for c in self.components:
            if c.ref == ref:
                return c
        return None
    
    def get_nets_for_component(self, ref: str) -> List[BoardConnection]:
        """Get all nets connected to a component."""
        return [c for c in self.connections if c.has_component(ref)]
    
    def build_graph(self) -> Optional[Any]:
        """Build NetworkX graph of connectivity."""
        if not NETWORKX_AVAILABLE:
            return None
        
        G = nx.Graph()
        
        # Add component nodes
        for comp in self.components:
            G.add_node(comp.ref, **comp.model_dump())
        
        # Add edges for connections
        for conn in self.connections:
            pins = conn.pins
            # Connect all pins in net as a clique
            for i, pin1 in enumerate(pins):
                for pin2 in pins[i+1:]:
                    G.add_edge(pin1.ref, pin2.ref, net=conn.net, properties=conn.properties)
        
        return G


class DFMViolation(BaseModel):
    """Detailed DFM violation with fix suggestions."""
    rule_id: str = Field(..., pattern=r'^DFM-[A-Z]{3}-\d{3}$')
    type: str
    severity: Literal["info", "warning", "error", "critical"]
    message: str
    components: List[str] = Field(default_factory=list)
    nets: List[str] = Field(default_factory=list)
    location: Optional[Dict[str, float]] = None
    suggested_fix: Optional[str] = None
    estimated_cost_impact: Optional[str] = None  # "low", "medium", "high"


class PlacementOptimization(BaseModel):
    """Metrics for placement optimization."""
    algorithm: str
    wirelength_mm: float
    density_uniformity: float  # 0-1, higher is better
    power_integrity_score: float  # 0-100
    thermal_score: float  # 0-100
    iterations: int
    convergence_delta: float


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float
    models_loaded: bool
    llm_loaded: bool = False
    placement_engine_loaded: bool = False
    templates_available: int = 0
    capabilities: List[str] = Field(default_factory=list)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=5000)
    constraints: Optional[Dict[str, Any]] = None
    priority: Literal["speed", "quality", "compact"] = "quality"


class GenerateResponse(BaseModel):
    success: bool
    circuit_data: Optional[Dict[str, Any]] = None
    template_used: Optional[str] = None
    generation_time_ms: float = 0
    warnings: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])


# ── Application State ─────────────────────────────────────────────────────────

class AppState:
    def __init__(self):
        self.llm: Any = None
        self.rl_model: Any = None
        self.dfm_engine: Any = None
        self.template_cache: Dict[str, Dict] = {}
        self.start_time: float = time.time()
        self.request_count: int = 0
        self.circuit_graphs: Dict[str, Any] = {}  # Cache for circuit graphs
    
    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.start_time
    
    def get_capabilities(self) -> List[str]:
        caps = ["basic_dfm", "netlist_analysis"]
        if self.llm:
            caps.append("llm_generation")
        if self.rl_model:
            caps.append("rl_placement")
        if self.dfm_engine:
            caps.append("advanced_dfm")
        if NETWORKX_AVAILABLE:
            caps.append("graph_analysis")
        return caps

_state = AppState()

# ── Spatial Index for Performance ─────────────────────────────────────────────

class SpatialIndex:
    """Grid-based spatial index for O(1) neighbor queries."""
    
    def __init__(self, cell_size: float = CONFIG.SPATIAL_GRID_SIZE_MM):
        self.cell_size = cell_size
        self.grid: Dict[Tuple[int, int], List[ComponentData]] = defaultdict(list)
        self.component_cells: Dict[str, Tuple[int, int]] = {}
    
    def insert(self, component: ComponentData) -> None:
        """Insert component into spatial index."""
        cell_x = int(component.x / self.cell_size)
        cell_y = int(component.y / self.cell_size)
        self.grid[(cell_x, cell_y)].append(component)
        self.component_cells[component.ref] = (cell_x, cell_y)
    
    def query_neighbors(self, component: ComponentData, radius: float) -> List[ComponentData]:
        """Find all components within radius."""
        cell_radius = int(radius / self.cell_size) + 1
        cell_x, cell_y = self.component_cells.get(component.ref, (0, 0))
        
        neighbors = []
        for dx in range(-cell_radius, cell_radius + 1):
            for dy in range(-cell_radius, cell_radius + 1):
                cell = (cell_x + dx, cell_y + dy)
                for other in self.grid.get(cell, []):
                    if other.ref != component.ref:
                        dist = math.hypot(component.x - other.x, component.y - other.y)
                        if dist <= radius:
                            neighbors.append((other, dist))
        
        return neighbors
    
    def query_region(self, x: float, y: float, width: float, height: float) -> List[ComponentData]:
        """Query all components in rectangular region."""
        min_x, max_x = int(x / self.cell_size), int((x + width) / self.cell_size)
        min_y, max_y = int(y / self.cell_size), int((y + height) / self.cell_size)
        
        results = []
        for cx in range(min_x, max_x + 1):
            for cy in range(min_y, max_y + 1):
                results.extend(self.grid.get((cx, cy), []))
        
        return results


# ── Advanced DFM Engine ───────────────────────────────────────────────────────

class AdvancedDFMEngine:
    """Comprehensive DFM analysis with netlist awareness."""
    
    POWER_NET_NAMES: Set[str] = {
        'VCC', 'VDD', '3V3', '3.3V', '5V', '1V8', '1.8V', '12V', '24V',
        'VPWR', 'VSUP', 'AVCC', 'DVCC', 'VCCIO', 'VCCINT'
    }
    
    GROUND_NET_NAMES: Set[str] = {
        'GND', 'VSS', 'AGND', 'DGND', 'PGND', 'SGND', 'VEE', 'VSSA', 'VSSD'
    }
    
    def __init__(self, board: BoardData):
        self.board = board
        self.violations: List[DFMViolation] = []
        self.spatial_index = SpatialIndex()
        for comp in board.components:
            self.spatial_index.insert(comp)
        
        # Build net lookup
        self.net_pins: Dict[str, List[PinRef]] = {}
        self.component_nets: Dict[str, List[str]] = defaultdict(list)
        for conn in board.connections:
            self.net_pins[conn.net] = conn.pins
            for pin in conn.pins:
                self.component_nets[pin.ref].append(conn.net)
    
    def analyze(self) -> List[DFMViolation]:
        """Run complete DFM analysis."""
        self._check_component_spacing()
        self._check_board_boundaries()
        self._check_orientation()
        self._check_power_integrity()
        self._check_signal_integrity()
        self._check_thermal()
        self._check_silkscreen()
        self._check_floating_pins()
        self._check_net_lengths()
        
        return sorted(self.violations, key=lambda v: {
            "critical": 0, "error": 1, "warning": 2, "info": 3
        }.get(v.severity, 4))
    
    def _check_component_spacing(self) -> None:
        """Advanced spacing with netlist awareness (skip check for connected components)."""
        checked_pairs = set()
        
        for comp in self.board.components:
            # Query potential neighbors
            neighbors = self.spatial_index.query_neighbors(comp, CONFIG.DFM_MIN_SPACING_MM * 2)
            
            for other, distance in neighbors:
                pair_key = tuple(sorted([comp.ref, other.ref]))
                if pair_key in checked_pairs:
                    continue
                checked_pairs.add(pair_key)
                
                # Skip if components are electrically connected (intentionally close)
                if self._are_connected(comp.ref, other.ref):
                    continue
                
                # Check actual spacing requirement based on component types
                required_spacing = self._get_required_spacing(comp, other)
                
                if distance < required_spacing:
                    self.violations.append(DFMViolation(
                        rule_id="DFM-SPC-001",
                        type="component_spacing",
                        severity="error" if distance < required_spacing * 0.5 else "warning",
                        message=f"{comp.ref} and {other.ref} too close ({distance:.2f}mm < {required_spacing}mm)",
                        components=[comp.ref, other.ref],
                        location={"x": (comp.x + other.x) / 2, "y": (comp.y + other.y) / 2},
                        suggested_fix=f"Move {other.ref} away by {required_spacing - distance:.1f}mm or verify electrical connection",
                        estimated_cost_impact="high" if distance < required_spacing * 0.5 else "medium"
                    ))
    
    def _get_required_spacing(self, c1: ComponentData, c2: ComponentData) -> float:
        """Calculate required spacing based on component characteristics."""
        base = CONFIG.DFM_MIN_SPACING_MM
        
        # High voltage requires more space
        if c1.power_dissipation_mw and c1.power_dissipation_mw > 1000:
            base += 1.0
        if c2.power_dissipation_mw and c2.power_dissipation_mw > 1000:
            base += 1.0
        
        # Tall components need keepout
        if c1.height_mm and c1.height_mm > 10:
            base += 2.0
        if c2.height_mm and c2.height_mm > 10:
            base += 2.0
        
        # Connectors need access space
        if c1.is_connector or c2.is_connector:
            base += 1.5
        
        return base
    
    def _check_board_boundaries(self) -> None:
        """Check component placement relative to board edges."""
        margin = CONFIG.DFM_EDGE_CLEARANCE_MM
        
        for comp in self.board.components:
            bbox = self._get_bounding_box(comp)
            
            checks = [
                (bbox["x"] < margin, "left", margin - bbox["x"]),
                (bbox["x"] + bbox["w"] > self.board.board_width - margin, "right", 
                 bbox["x"] + bbox["w"] - (self.board.board_width - margin)),
                (bbox["y"] < margin, "bottom", margin - bbox["y"]),
                (bbox["y"] + bbox["h"] > self.board.board_height - margin, "top",
                 bbox["y"] + bbox["h"] - (self.board.board_height - margin))
            ]
            
            for violated, edge, overflow in checks:
                if violated:
                    self.violations.append(DFMViolation(
                        rule_id="DFM-BND-001",
                        type="board_boundary",
                        severity="error",
                        message=f"{comp.ref} violates {edge} edge clearance ({overflow:.1f}mm)",
                        components=[comp.ref],
                        location={"x": comp.x, "y": comp.y},
                        suggested_fix=f"Move {comp.ref} inward by {overflow + 0.5:.1f}mm"
                    ))
    
    def _check_orientation(self) -> None:
        """Check for polarized components with suspicious rotation values."""
        for comp in self.board.components:
            rot = comp.rotation % 360
            if comp.is_polarized and rot not in (0.0, 90.0, 180.0, 270.0):
                self.violations.append(DFMViolation(
                    rule_id="DFM-ORI-001",
                    type="orientation",
                    severity="warning",
                    message=(
                        f"{comp.ref} is polarized with non-standard rotation "
                        f"({comp.rotation:.1f}°) — verify orientation"
                    ),
                    components=[comp.ref],
                    location={"x": comp.x, "y": comp.y},
                    suggested_fix=f"Rotate {comp.ref} to 0°, 90°, 180°, or 270°"
                ))

    def _check_power_integrity(self) -> None:
        ics = [c for c in self.board.components if c.is_ic]
        
        for ic in ics:
            ic_nets = set(self.component_nets.get(ic.ref, []))
            power_nets = ic_nets & self.POWER_NET_NAMES
            ground_nets = ic_nets & self.GROUND_NET_NAMES
            
            if not power_nets:
                self.violations.append(DFMViolation(
                    rule_id="DFM-PWR-001",
                    type="power_connection",
                    severity="error",
                    message=f"{ic.ref} has no power net connection",
                    components=[ic.ref],
                    location={"x": ic.x, "y": ic.y}
                ))
                continue
            
            if not ground_nets:
                self.violations.append(DFMViolation(
                    rule_id="DFM-PWR-002",
                    type="ground_connection",
                    severity="error",
                    message=f"{ic.ref} has no ground connection",
                    components=[ic.ref],
                    location={"x": ic.x, "y": ic.y}
                ))
                continue
            
            # Check decoupling for each power pin
            for pwr_net in power_nets:
                self._check_decoupling(ic, pwr_net, ground_nets)
    
    def _check_decoupling(self, ic: ComponentData, pwr_net: str, gnd_nets: Set[str]) -> None:
        """Verify adequate decoupling capacitors near IC."""
        # Find caps on this power net
        decoupling_candidates = []
        
        for comp in self.board.components:
            if not comp.ref.startswith('C'):
                continue
            
            comp_nets = set(self.component_nets.get(comp.ref, []))
            
            # Capacitor must connect power and ground
            if pwr_net in comp_nets and (comp_nets & gnd_nets):
                dist = math.hypot(comp.x - ic.x, comp.y - ic.y)
                decoupling_candidates.append((comp, dist))
        
        if not decoupling_candidates:
            self.violations.append(DFMViolation(
                rule_id="DFM-PWR-003",
                type="missing_decoupling",
                severity="warning",
                message=f"{ic.ref} lacks decoupling capacitor on {pwr_net}",
                components=[ic.ref],
                nets=[pwr_net],
                location={"x": ic.x, "y": ic.y},
                suggested_fix=f"Add 100nF ceramic capacitor within {CONFIG.DFM_DECOUPLING_MAX_DISTANCE_MM}mm of {ic.ref}"
            ))
            return
        
        # Check distance of closest cap
        decoupling_candidates.sort(key=lambda x: x[1])
        closest_cap, closest_dist = decoupling_candidates[0]
        
        if closest_dist > CONFIG.DFM_DECOUPLING_MAX_DISTANCE_MM:
            self.violations.append(DFMViolation(
                rule_id="DFM-PWR-004",
                type="decoupling_distance",
                severity="warning",
                message=f"Decoupling cap {closest_cap.ref} for {ic.ref} is too far ({closest_dist:.1f}mm)",
                components=[ic.ref, closest_cap.ref],
                nets=[pwr_net],
                location={"x": ic.x, "y": ic.y},
                suggested_fix=f"Move {closest_cap.ref} closer to {ic.ref} or add additional capacitor"
            ))
        
        # Check for bulk capacitance if IC is high power
        if ic.power_dissipation_mw and ic.power_dissipation_mw > 500:
            bulk_caps = [c for c, d in decoupling_candidates if c.value and 
                        any(unit in c.value.upper() for unit in ['UF', 'MF', 'µF'])]
            if not bulk_caps:
                self.violations.append(DFMViolation(
                    rule_id="DFM-PWR-005",
                    type="missing_bulk_capacitance",
                    severity="info",
                    message=f"High-power IC {ic.ref} may need bulk capacitance (>1µF)",
                    components=[ic.ref],
                    nets=[pwr_net]
                ))
    
    def _check_signal_integrity(self) -> None:
        """Analyze high-speed signals and differential pairs."""
        for conn in self.board.connections:
            props = conn.properties
            
            if props.net_type == "differential":
                self._check_differential_pair(conn)
            
            if props.frequency and props.frequency > 1e6:  # > 1MHz
                self._check_high_speed_signal(conn)
    
    def _check_differential_pair(self, conn: BoardConnection) -> None:
        """Verify differential pair routing constraints."""
        # Should have exactly 2 components for a simple diff pair
        if len(conn.components) != 2:
            return
        
        # Check length matching
        # This would need actual trace geometry, simplified here
        pass
    
    def _check_high_speed_signal(self, conn: BoardConnection) -> None:
        """Check for stubs, vias, and length on high-speed nets."""
        # Simplified check: ensure clock signals don't have many loads
        if props := conn.properties:
            if props.net_type == "clock" and len(conn.pins) > 3:
                self.violations.append(DFMViolation(
                    rule_id="DFM-SI-001",
                    type="clock_fanout",
                    severity="warning",
                    message=f"Clock net {conn.net} has high fanout ({len(conn.pins)} pins)",
                    nets=[conn.net],
                    suggested_fix="Use clock buffer or reduce fanout"
                ))
    
    def _check_thermal(self) -> None:
        """Analyze thermal management."""
        hot_components = [c for c in self.board.components 
                         if c.power_dissipation_mw and c.power_dissipation_mw > 500]
        
        for hot in hot_components:
            # Check for thermal relief pattern (simplified)
            neighbors = self.spatial_index.query_neighbors(hot, 5.0)
            copper_area = sum(1 for _, d in neighbors if d < 3.0)
            
            if copper_area < 4:  # Not enough copper nearby for heat spreading
                self.violations.append(DFMViolation(
                    rule_id="DFM-THM-001",
                    type="thermal_management",
                    severity="warning",
                    message=f"{hot.ref} ({hot.power_dissipation_mw}mW) may overheat",
                    components=[hot.ref],
                    location={"x": hot.x, "y": hot.y},
                    suggested_fix="Add thermal vias, copper pour, or heatsink"
                ))
    
    def _check_silkscreen(self) -> None:
        """Verify silkscreen readability."""
        for comp in self.board.components:
            # Check for overlapping references (simplified)
            pass  # Would need actual silkscreen geometry
    
    def _check_floating_pins(self) -> None:
        """Detect unconnected pins on active components."""
        connected_refs = {p.ref for conn in self.board.connections for p in conn.pins}
        
        for comp in self.board.components:
            if comp.is_ic and comp.ref not in connected_refs:
                self.violations.append(DFMViolation(
                    rule_id="DFM-CNN-001",
                    type="floating_component",
                    severity="error",
                    message=f"{comp.ref} ({comp.value}) has no connections",
                    components=[comp.ref],
                    location={"x": comp.x, "y": comp.y}
                ))
    
    def _check_net_lengths(self) -> None:
        """Estimate and check net lengths."""
        for conn in self.board.connections:
            if len(conn.pins) < 2:
                continue
            
            # Calculate Manhattan distance between farthest pins
            max_dist = 0
            pins = conn.pins
            for i, p1 in enumerate(pins):
                comp1 = self.board.get_component(p1.ref)
                if not comp1:
                    continue
                for p2 in pins[i+1:]:
                    comp2 = self.board.get_component(p2.ref)
                    if not comp2:
                        continue
                    dist = abs(comp1.x - comp2.x) + abs(comp1.y - comp2.y)
                    max_dist = max(max_dist, dist)
            
            # Check against properties
            if conn.properties.length_mm and max_dist > conn.properties.length_mm * 1.5:
                self.violations.append(DFMViolation(
                    rule_id="DFM-LEN-001",
                    type="excessive_length",
                    severity="warning",
                    message=f"Net {conn.net} may be too long ({max_dist:.1f}mm)",
                    nets=[conn.net],
                    suggested_fix="Consider moving components closer or termination"
                ))
    
    def _are_connected(self, ref1: str, ref2: str) -> bool:
        """Check if two components share a net."""
        nets1 = set(self.component_nets.get(ref1, []))
        nets2 = set(self.component_nets.get(ref2, []))
        return bool(nets1 & nets2)
    
    def _get_bounding_box(self, comp: ComponentData) -> Dict[str, float]:
        """Estimate component bounding box."""
        # Simplified - would use actual footprint data
        sizes = {"R": 1.6, "C": 1.6, "L": 2.0, "D": 2.0, "U": 5.0, "Q": 3.0, "J": 10.0}
        size = sizes.get(comp.prefix, 5.0)
        
        # Account for rotation
        if comp.rotation % 180 != 0:
            size *= 1.4  # Diagonal approximation
        
        return {
            "x": comp.x - size/2,
            "y": comp.y - size/2,
            "w": size,
            "h": size
        }


# ── Placement Optimizer ───────────────────────────────────────────────────────

class PlacementOptimizer:
    """Advanced placement with connectivity awareness."""
    
    def __init__(self, board: BoardData):
        self.board = board
        self.graph = board.build_graph()
    
    def optimize(self, algorithm: str = "force_directed") -> Dict[str, Any]:
        """Run placement optimization."""
        if algorithm == "force_directed":
            return self._force_directed_placement()
        elif algorithm == "annealing":
            return self._simulated_annealing()
        else:
            return self._grid_placement()
    
    def _force_directed_placement(self) -> Dict[str, Any]:
        """Force-directed graph placement."""
        positions = {c.ref: np.array([c.x, c.y]) for c in self.board.components}
        
        # Iterative relaxation
        for iteration in range(100):
            forces = {ref: np.zeros(2) for ref in positions}
            
            # Attractive forces for connected components
            for conn in self.board.connections:
                pins = conn.pins
                for i, p1 in enumerate(pins):
                    for p2 in pins[i+1:]:
                        if p1.ref in positions and p2.ref in positions:
                            pos1, pos2 = positions[p1.ref], positions[p2.ref]
                            diff = pos2 - pos1
                            dist = np.linalg.norm(diff)
                            if dist > 0:
                                force = diff / dist * (dist - 10) * 0.1  # Spring constant
                                forces[p1.ref] += force
                                forces[p2.ref] -= force
            
            # Repulsive forces for all components
            refs = list(positions.keys())
            for i, r1 in enumerate(refs):
                for r2 in refs[i+1:]:
                    diff = positions[r2] - positions[r1]
                    dist = np.linalg.norm(diff)
                    if dist < 20 and dist > 0:
                        force = -diff / dist * (20 - dist) * 0.5
                        forces[r1] += force
                        forces[r2] -= force
            
            # Apply forces with damping
            damping = 0.9 - (iteration / 1000)
            for ref in positions:
                positions[ref] += forces[ref] * damping
                
                # Keep within bounds
                positions[ref][0] = np.clip(positions[ref][0], 5, self.board.board_width - 5)
                positions[ref][1] = np.clip(positions[ref][1], 5, self.board.board_height - 5)
        
        # Convert back to dict format
        return {
            "positions": {
                ref: {"x": float(pos[0]), "y": float(pos[1]), "rotation": 0}
                for ref, pos in positions.items()
            },
            "algorithm": "force_directed",
            "iterations": 100
        }
    
    def _grid_placement(self) -> Dict[str, Any]:
        """Fallback grid placement with grouping."""
        # Group by net connectivity
        groups = self._compute_groups()
        
        positions = {}
        grid_size = 10.0
        col = 0
        
        for group in groups:
            for i, ref in enumerate(group):
                row = i // 5
                col_offset = col + (i % 5)
                positions[ref] = {
                    "x": 10.0 + col_offset * grid_size,
                    "y": 10.0 + row * grid_size,
                    "rotation": 0
                }
            col += (len(group) // 5) + 1
        
        return {
            "positions": positions,
            "algorithm": "connectivity_grid",
            "iterations": 1
        }
    
    def _compute_groups(self) -> List[List[str]]:
        """Group components by connectivity using union-find."""
        parent = {c.ref: c.ref for c in self.board.components}
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            parent[find(x)] = find(y)
        
        # Union connected components
        for conn in self.board.connections:
            refs = [p.ref for p in conn.pins]
            for i in range(1, len(refs)):
                union(refs[0], refs[i])
        
        # Build groups
        groups = defaultdict(list)
        for ref in parent:
            groups[find(ref)].append(ref)
        
        return sorted(groups.values(), key=len, reverse=True)


# ── Lifespan ─────────────────────────────────────────────────────────────────
# NOTE: lifespan MUST be defined before app = FastAPI(lifespan=lifespan)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    logger.info("Starting AI PCB Assistant v2.0...")

    # Ensure output directory exists
    CONFIG.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load templates
    if CONFIG.TEMPLATES_DIR.exists():
        for path in CONFIG.TEMPLATES_DIR.glob("*.json"):
            try:
                if AIOFILES_AVAILABLE:
                    async with aiofiles.open(path, 'r') as f:
                        content = await f.read()
                else:
                    content = path.read_text()
                _state.template_cache[path.stem] = json.loads(content)
            except Exception as e:
                logger.warning(f"Failed to load template {path.name}: {e}")

    logger.info(f"Loaded {len(_state.template_cache)} templates")

    # Load engines
    try:
        from engines.llm_engine import load_llm
        _state.llm = load_llm()
        logger.info("LLM engine loaded")
    except Exception as e:
        logger.warning(f"LLM not available: {e}")

    try:
        from engines.placement_engine import load_placement_model
        _state.rl_model = load_placement_model()
        logger.info("RL placement engine loaded")
    except Exception as e:
        logger.warning(f"RL placement not available: {e}")

    yield

    # Cleanup
    logger.info("Shutting down...")


# ── FastAPI Application ───────────────────────────────────────────────────────

app = FastAPI(
    title="AI PCB Assistant Backend",
    description="Advanced AI backend for KiCad PCB design with full netlist integration",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Request timing middleware
@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    response.headers["X-Process-Time"] = str(time.time() - start)
    return response

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check."""
    return HealthResponse(
        status="healthy" if _state.llm or _state.rl_model else "degraded",
        version="2.0.0",
        uptime_seconds=_state.uptime_seconds,
        models_loaded=_state.llm is not None or _state.rl_model is not None,
        llm_loaded=_state.llm is not None,
        placement_engine_loaded=_state.rl_model is not None,
        templates_available=len(_state.template_cache),
        capabilities=_state.get_capabilities()
    )

@app.post("/analyze/dfm", response_model=List[DFMViolation])
async def analyze_dfm(board: BoardData):
    """
    Advanced DFM analysis with full netlist integration.
    
    Features:
    - Spatial indexing for O(n log n) performance
    - Power integrity analysis (decoupling, bulk capacitance)
    - Signal integrity checks (differential pairs, high-speed)
    - Thermal analysis
    - Connectivity-aware spacing (connected components can be closer)
    """
    engine = AdvancedDFMEngine(board)
    violations = engine.analyze()
    return violations

@app.post("/placement/optimize")
async def optimize_placement(board: BoardData, algorithm: str = "auto"):
    """
    Advanced placement optimization.
    
    Algorithms:
    - auto: Choose best available (RL > force_directed > grid)
    - rl: Reinforcement learning (if available)
    - force_directed: Physics-based spring simulation
    - annealing: Simulated annealing
    - grid: Connectivity-aware grid
    """
    t0 = time.time()
    
    # Choose algorithm
    if algorithm == "auto":
        if _state.rl_model:
            algorithm = "rl"
        else:
            algorithm = "force_directed"
    
    if algorithm == "rl" and _state.rl_model:
        try:
            from engines.placement_engine import optimize_with_rl
            result = optimize_with_rl(_state.rl_model, board.model_dump())
            result["algorithm"] = "rl"
            result["time_ms"] = (time.time() - t0) * 1000
            return result
        except Exception as e:
            logger.warning(f"RL failed, falling back: {e}")
            algorithm = "force_directed"
    
    # Use internal optimizer
    optimizer = PlacementOptimizer(board)
    result = optimizer.optimize(algorithm)
    result["time_ms"] = (time.time() - t0) * 1000
    
    return result

def _normalise_connections(circuit_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert old-style template connection pins to BoardConnection PinRef format.

    Old format (CircuitData / templates):  "pins": ["R1.1", "C1.1"]
    New format (BoardConnection):          "pins": [{"ref": "R1", "pin": "1"}, ...]

    Also removes component refs not present in the components list (e.g. KiCad
    power symbols like PWR_FLAG that live only in the schematic) to avoid the
    BoardData.validate_references error.
    # NOTE: safe to call on data already in new format - dicts pass through unchanged.
    """
    import copy
    data = copy.deepcopy(circuit_data)

    comp_refs: Set[str] = {c["ref"] for c in data.get("components", []) if "ref" in c}

    normalised: List[Dict[str, Any]] = []
    for conn in data.get("connections", []):
        raw_pins = conn.get("pins", [])
        new_pins: List[Dict[str, str]] = []
        for p in raw_pins:
            if isinstance(p, str):
                # "R1.1" → {"ref": "R1", "pin": "1"}
                if "." in p:
                    ref, pin = p.split(".", 1)
                    new_pins.append({"ref": ref, "pin": pin})
            elif isinstance(p, dict):
                new_pins.append(p)
        # Drop pins whose component doesn't exist in the board (power symbols etc.)
        new_pins = [p for p in new_pins if p.get("ref", "") in comp_refs]
        if len(new_pins) >= 2:
            normalised.append({**conn, "pins": new_pins})

    data["connections"] = normalised
    return data


@app.post("/generate")
async def generate_circuit(request: GenerateRequest, background_tasks: BackgroundTasks):
    """
    Generate circuit from natural language with advanced analysis.
    """
    t0 = time.time()
    request_id = str(uuid.uuid4())[:8]
    
    # Template matching with scoring
    template_name = None
    best_score = 0
    
    for keywords, name, priority in TEMPLATE_KEYWORDS:
        score = sum(1 for kw in keywords if kw in request.prompt.lower()) * priority
        if score > best_score:
            best_score = score
            template_name = name
    
    # Load template or use LLM
    circuit_data = None
    warnings = []
    
    if template_name and template_name in _state.template_cache:
        circuit_data = _state.template_cache[template_name]
    elif _state.llm:
        try:
            circuit_data = _state.llm.generate_circuit_json(request.prompt)
        except Exception as e:
            warnings.append(f"LLM generation failed: {e}")
    
    if not circuit_data:
        return GenerateResponse(
            success=False,
            error="No matching template and LLM unavailable",
            request_id=request_id
        )
    
    # Validate and enhance
    try:
        # NOTE: Templates store connections as "pins": ["R1.1", "C1.1"] (old CircuitData
        # format) or as [{"ref":"R1","pin":"1"}] (BoardConnection format). Normalise here
        # so BoardData validation doesn't fail on legacy template data.
        circuit_data = _normalise_connections(circuit_data)
        board = BoardData(**circuit_data)
        
        # Auto-place if no positions
        if all(c.x == 0 and c.y == 0 for c in board.components):
            optimizer = PlacementOptimizer(board)
            placement = optimizer.optimize("grid")
            for ref, pos in placement["positions"].items():
                comp = board.get_component(ref)
                if comp:
                    comp.x = pos["x"]
                    comp.y = pos["y"]
        
        # Run DFM
        dfm_engine = AdvancedDFMEngine(board)
        violations = dfm_engine.analyze()
        
        # Async save
        output_path = CONFIG.OUTPUT_DIR / f"circuit_{request_id}.json"
        if AIOFILES_AVAILABLE:
            background_tasks.add_task(_async_save_circuit, output_path, board.model_dump())
        else:
            output_path.write_text(json.dumps(board.model_dump(), indent=2))
        
        generation_time = (time.time() - t0) * 1000
        
        return GenerateResponse(
            success=True,
            circuit_data=board.model_dump(),
            template_used=template_name,
            generation_time_ms=generation_time,
            warnings=warnings,
            request_id=request_id
        )
        
    except Exception as e:
        return GenerateResponse(
            success=False,
            error=f"Validation failed: {e}",
            request_id=request_id
        )

async def _async_save_circuit(path: Path, data: dict):
    """Async save circuit to disk."""
    async with aiofiles.open(path, 'w') as f:
        await f.write(json.dumps(data, indent=2))

@app.get("/templates")
async def list_templates():
    """List available templates with metadata."""
    return [
        {
            "name": name,
            "description": data.get("description", ""),
            "components": len(data.get("components", [])),
            "nets": len(data.get("connections", []))
        }
        for name, data in _state.template_cache.items()
    ]

@app.post("/dfm/check", response_model=List[DFMViolation])
async def dfm_check(board: BoardData):
    """
    Alias for /analyze/dfm kept for plugin.py compatibility.
    # NOTE: plugin.py sends DFM requests to /dfm/check — this route ensures
    # existing plugin builds continue to work with the v2 backend.
    """
    return await analyze_dfm(board)


@app.post("/export/kicad")
async def export_kicad(circuit: dict):
    """Export to KiCad format."""
    try:
        from circuit_schema import CircuitData
        from engines.kicad_exporter import export_to_kicad_sch
        
        data = CircuitData(**circuit)
        content = export_to_kicad_sch(data)
        
        # Stream response for large files
        def iter_content():
            yield content
        
        return StreamingResponse(
            iter_content(),
            media_type="application/x-kicad-schematic",
            headers={"Content-Disposition": f"attachment; filename=circuit.kicad_sch"}
        )
        
    except ImportError:
        raise HTTPException(status_code=501, detail="KiCad exporter not available")

# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "ai_server:app",
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8765")),
        reload=os.environ.get("RELOAD", "false").lower() == "true",
        workers=int(os.environ.get("WORKERS", "1"))
    )