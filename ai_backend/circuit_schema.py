"""
Circuit JSON Schema v2.0 — Production-Ready Circuit Data Structure

Strictly typed, validated, and extensible schema for PCB circuit representation.
Supports KiCad integration, DFM analysis, and version migration.
"""

from __future__ import annotations

import re
from typing import Optional, List, Dict, Literal, Union, Any, ClassVar
from enum import Enum
from datetime import datetime
from pydantic import (
    BaseModel, 
    Field, 
    field_validator, 
    model_validator, 
    ConfigDict,
    ValidationInfo
)


# ─── Enums and Constants ─────────────────────────────────────────────────────

class NetType(str, Enum):
    """Electrical net types for analysis and routing priority."""
    POWER = "power"           # VCC, VDD, 3V3, 5V, etc.
    GROUND = "ground"         # GND, AGND, DGND, etc.
    SIGNAL = "signal"         # General signals
    CLOCK = "clock"           # High-speed clock signals
    DIFFERENTIAL = "diff"     # Differential pairs (USB, HDMI, etc.)
    ANALOG = "analog"         # Sensitive analog signals
    HIGH_VOLTAGE = "hv"       # Safety-critical high voltage


class ComponentType(str, Enum):
    """Component categories for BOM grouping and placement rules."""
    RESISTOR = "resistor"
    CAPACITOR = "capacitor"
    INDUCTOR = "inductor"
    DIODE = "diode"
    LED = "led"
    TRANSISTOR = "transistor"
    IC = "ic"
    CONNECTOR = "connector"
    MECHANICAL = "mechanical"
    POWER = "power"


class Severity(str, Enum):
    """DFM violation severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class RotationPreset(float, Enum):
    """Standard KiCad rotation angles."""
    DEG_0 = 0.0
    DEG_90 = 90.0
    DEG_180 = 180.0
    DEG_270 = 270.0


# ─── Geometric Types ───────────────────────────────────────────────────────────

class Point2D(BaseModel):
    """2D point with PCB coordinate validation."""
    model_config = ConfigDict(frozen=True)
    
    x: float = Field(..., ge=-1000.0, le=1000.0, description="X coordinate in mm")
    y: float = Field(..., ge=-1000.0, le=1000.0, description="Y coordinate in mm")
    
    def distance_to(self, other: Point2D) -> float:
        """Calculate Euclidean distance to another point."""
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5
    
    def __add__(self, other: Point2D) -> Point2D:
        return Point2D(x=self.x + other.x, y=self.y + other.y)


class BoundingBox(BaseModel):
    """Rectangular area on PCB."""
    model_config = ConfigDict(frozen=True)
    
    x: float = Field(..., description="Left edge")
    y: float = Field(..., description="Bottom edge")
    width: float = Field(..., gt=0, le=1000, description="Width in mm")
    height: float = Field(..., gt=0, le=1000, description="Height in mm")
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    @property
    def center(self) -> Point2D:
        return Point2D(x=self.x + self.width/2, y=self.y + self.height/2)
    
    def contains(self, point: Point2D) -> bool:
        """Check if point is inside bounding box."""
        return (self.x <= point.x <= self.x + self.width and 
                self.y <= point.y <= self.y + self.height)


# ─── Pin and Component Models ──────────────────────────────────────────────────

class Pin(BaseModel):
    """
    A single pin on a component with electrical characteristics.
    Supports both through-hole and SMD pad definitions.
    """
    model_config = ConfigDict(frozen=True)
    
    number: str = Field(
        ..., 
        min_length=1, 
        max_length=10,
        description="Pin number (e.g. '1', '2', 'A1', ' ThermalPad')"
    )
    name: str = Field(
        default="",
        max_length=50,
        description="Pin function (e.g. 'VCC', 'GND', 'OUT', 'NC')"
    )
    electrical_type: Literal[
        "input", "output", "bidirectional", "tri_state",
        "passive", "power_in", "power_out", "open_collector",
        "open_emitter", "unspecified", "no_connect"
    ] = Field(default="unspecified", description="KiCad electrical type")
    
    # Geometry
    position: Point2D = Field(default_factory=lambda: Point2D(x=0, y=0))
    rotation: float = Field(default=0.0, ge=-360, le=360)
    
    # Physical properties
    shape: Literal["circle", "rect", "oval", "roundrect"] = Field(default="circle")
    size_mm: float = Field(default=1.0, gt=0, le=10, description="Pad diameter/size in mm")
    drill_mm: Optional[float] = Field(default=None, gt=0, le=10, description="Drill hole for THT")
    
    # Electrical properties
    net: Optional[str] = Field(default=None, description="Connected net name")
    
    @field_validator('number')
    @classmethod
    def validate_pin_number(cls, v: str) -> str:
        """Validate pin number format."""
        if not re.match(r'^[0-9A-Za-z]+$', v):
            raise ValueError(f"Invalid pin number: {v}. Use alphanumeric only.")
        return v
    
    @field_validator('name')
    @classmethod
    def validate_pin_name(cls, v: str) -> str:
        """Standardize common pin names."""
        v_upper = v.upper()
        # Normalize power pin names
        if v_upper in ['VCC', 'VDD', 'VSUP', 'V+']:
            return 'VCC'
        if v_upper in ['GND', 'VSS', 'VEE', 'V-', 'AGND', 'DGND']:
            return 'GND'
        return v
    
    @property
    def is_power(self) -> bool:
        """Check if this is a power pin."""
        return self.name.upper() in ['VCC', 'VDD', '3V3', '5V', 'GND', 'VSS']


class Component(BaseModel):
    """
    A single component with full KiCad symbol and placement information.
    Supports hierarchical design with sub-components.
    """
    model_config = ConfigDict(extra='allow')  # Allow extra fields for extensions
    
    # Identity
    ref: str = Field(
        ...,
        pattern=r'^[A-Z]{1,3}[0-9]+$',
        description="Reference designator (e.g. 'R1', 'U1', 'C10', 'LED3')"
    )
    uuid: Optional[str] = Field(
        default=None,
        pattern=r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
        description="UUID v4 for tracking"
    )
    
    # Library info
    lib: str = Field(
        default="Device",
        min_length=1,
        max_length=100,
        description="KiCad symbol library"
    )
    part: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Symbol name in library"
    )
    
    # Value and specs
    value: str = Field(
        ...,
        max_length=100,
        description="Component value (e.g. '10k', '100nF', 'NE555D')"
    )
    tolerance: Optional[str] = Field(default=None, description="e.g. '1%', '5%', 'J'")
    voltage_rating: Optional[str] = Field(default=None, description="e.g. '50V', '25V'")
    power_rating: Optional[str] = Field(default=None, description="e.g. '1/4W', '0.125W'")
    
    # Package
    footprint: str = Field(
        default="",
        max_length=200,
        description="KiCad footprint 'Library:Footprint'"
    )
    footprint_type: Optional[ComponentType] = Field(default=None)
    
    # Metadata
    description: str = Field(default="", max_length=500)
    keywords: List[str] = Field(default_factory=list)
    datasheet_url: Optional[str] = Field(default=None, max_length=500)
    manufacturer: Optional[str] = Field(default=None, max_length=100)
    mpn: Optional[str] = Field(default=None, max_length=100, description="Manufacturer Part Number")
    
    # Placement
    position: Point2D = Field(default_factory=lambda: Point2D(x=0, y=0))
    # x / y are flat aliases kept in sync with position so the KiCad exporter
    # (which reads and mutates comp.x / comp.y directly) works without changes.
    x: float = Field(default=0.0, ge=-1000.0, le=1000.0)
    y: float = Field(default=0.0, ge=-1000.0, le=1000.0)
    rotation: float = Field(default=0.0, ge=-360, le=360)
    layer: Literal["F.Cu", "B.Cu"] = Field(default="F.Cu", description="Top or bottom layer")
    locked: bool = Field(default=False, description="Prevent auto-placement")
    
    # Pins
    pins: List[Pin] = Field(default_factory=list)
    
    # DFM
    height_mm: Optional[float] = Field(default=None, gt=0, le=50, description="Component height for 3D checking")
    keepout_mm: Optional[float] = Field(default=None, gt=0, le=20, description="Exclusion zone around component")
    
    # Schema version for migrations
    schema_version: str = Field(default="2.0", pattern=r'^\d+\.\d+$')
    
    _ref_prefixes: ClassVar[set[str]] = {
        'R', 'C', 'L', 'D', 'LED', 'Q', 'U', 'J', 'P', 
        'F', 'FB', 'SW', 'Y', 'X', 'TP', 'MH', 'H'
    }
    
    @field_validator('ref')
    @classmethod
    def validate_reference(cls, v: str) -> str:
        """Ensure reference follows EIA standard."""
        match = re.match(r'^([A-Z]{1,3})([0-9]+)$', v)
        if not match:
            raise ValueError(f"Invalid reference '{v}'. Format: Letter(s) + Number (e.g., R1, U23, LED1)")
        
        prefix, number = match.groups()
        if prefix not in cls._ref_prefixes:
            raise ValueError(f"Unknown reference prefix '{prefix}'. Valid: {cls._ref_prefixes}")
        
        if len(number) > 4:
            raise ValueError(f"Reference number too long: {number}")
        
        return v
    
    @field_validator('footprint')
    @classmethod
    def validate_footprint(cls, v: str) -> str:
        """Validate KiCad footprint format."""
        if not v:
            return v
        
        # Allow legacy formats but warn about modern format
        if ':' not in v and v:
            # Legacy format without library - accept but it's not best practice
            return v
        
        if ':' in v:
            lib, fp = v.split(':', 1)
            if not lib or not fp:
                raise ValueError(f"Invalid footprint '{v}'. Use 'Library:Footprint' format")
        
        return v
    
    @field_validator('value')
    @classmethod
    def validate_value_format(cls, v: str, info: ValidationInfo) -> str:
        """Normalize common value formats."""
        # Get part type from context if available
        part = info.data.get('part', '').upper()
        ref = info.data.get('ref', '').upper()
        
        # Normalize resistance
        if part in ['R', 'RESISTOR'] or ref.startswith('R'):
            v = cls._normalize_resistance(v)
        # Normalize capacitance
        elif part in ['C', 'CAPACITOR'] or ref.startswith('C'):
            v = cls._normalize_capacitance(v)
        
        return v
    
    @staticmethod
    def _normalize_resistance(v: str) -> str:
        """Normalize resistance values (10k -> 10kΩ, 1k0 -> 1.0k)."""
        v = v.strip().replace(' ', '').replace('Ω', '').replace('Ohm', '').replace('ohms', '')
        
        # Handle multiplier suffixes
        multipliers = {'R': 1, 'E': 1, 'K': 1e3, 'M': 1e6, 'G': 1e9}
        
        for suffix, mult in multipliers.items():
            if suffix in v.upper():
                try:
                    num = v.upper().replace(suffix, '')
                    if mult == 1:
                        return f"{num}Ω"
                    else:
                        return f"{num}{suffix.lower()}Ω"
                except ValueError:
                    pass
        
        return f"{v}Ω" if v else v
    
    @staticmethod
    def _normalize_capacitance(v: str) -> str:
        """Normalize capacitance values."""
        v = v.strip().replace(' ', '').replace('F', '')
        
        multipliers = {'P': 1e-12, 'N': 1e-9, 'U': 1e-6, 'M': 1e-3}
        
        for suffix, mult in multipliers.items():
            if suffix in v.upper():
                num = v.upper().replace(suffix, '')
                try:
                    val = float(num) * mult
                    if val >= 1e-6:
                        return f"{val*1e6:.1f}µF"
                    elif val >= 1e-9:
                        return f"{val*1e9:.1f}nF"
                    else:
                        return f"{val*1e12:.1f}pF"
                except ValueError:
                    pass
        
        return f"{v}F" if v else v

    @model_validator(mode='after')
    def sync_xy_from_position(self) -> 'Component':
        """Keep flat x/y fields in sync with the nested position object.

        Priority rules (at construction time):
        - If x/y were explicitly supplied (non-zero) but position is still at
          default (0,0), promote x/y into position.
        - If position was explicitly supplied (non-zero) but x/y are at default
          (0.0), copy position into x/y.
        - If both are supplied, x/y take priority (they're the simpler API).
        """
        pos_non_default = self.position.x != 0.0 or self.position.y != 0.0
        xy_non_default  = self.x != 0.0 or self.y != 0.0

        if xy_non_default:
            # x/y have a real value — make position match
            self.position = Point2D(x=self.x, y=self.y)
        elif pos_non_default:
            # only position has a real value — copy out to x/y
            self.x = self.position.x
            self.y = self.position.y
        return self

    def __setattr__(self, name: str, value: Any) -> None:
        """Keep flat x/y and nested position in sync on every mutation.

        The KiCad exporter layout code mutates comp.x / comp.y directly
        (e.g. ic.x = sheet_cx).  Without this override, position.x stays
        stale, so get_bounding_box() and any code that reads position after
        layout would return wrong coordinates.
        """
        super().__setattr__(name, value)
        if name == 'x':
            super().__setattr__('position', Point2D(x=float(value), y=self.y))
        elif name == 'y':
            super().__setattr__('position', Point2D(x=self.x, y=float(value)))
        elif name == 'position' and isinstance(value, Point2D):
            super().__setattr__('x', value.x)
            super().__setattr__('y', value.y)

    @property
    def ref_prefix(self) -> str:
        """Extract reference prefix."""
        match = re.match(r'^([A-Z]{1,3})', self.ref)
        return match.group(1) if match else 'U'
    
    @property
    def ref_number(self) -> int:
        """Extract reference number."""
        match = re.search(r'(\d+)$', self.ref)
        return int(match.group(1)) if match else 0
    
    @property
    def is_smd(self) -> bool:
        """Check if SMD based on footprint."""
        if not self.footprint:
            return False
        smd_indicators = ['SMD', '0805', '0603', '0402', '1206', '1210', 'QFN', 'QFP', 'BGA']
        return any(ind in self.footprint for ind in smd_indicators)
    
    def get_pin(self, number: str) -> Optional[Pin]:
        """Get pin by number."""
        for pin in self.pins:
            if pin.number == number:
                return pin
        return None
    
    def get_bounding_box(self, padding_mm: float = 0.0) -> BoundingBox:
        """Estimate component bounding box."""
        # Default sizes by type
        defaults = {
            'R': (1.6, 0.8), 'C': (1.6, 0.8), 'L': (2.0, 1.2),
            'D': (2.0, 1.2), 'LED': (2.0, 1.2),
            'U': (5.0, 5.0), 'Q': (3.0, 3.0), 'J': (10.0, 5.0)
        }
        
        w, h = defaults.get(self.ref_prefix, (5.0, 5.0))
        
        # Rotate dimensions if needed
        if self.rotation % 180 != 0:
            w, h = h, w
        
        # Use flat x/y (always current) not position (may lag after kicad
        # exporter or placement engine mutates x/y directly).
        return BoundingBox(
            x=self.x - w/2 - padding_mm,
            y=self.y - h/2 - padding_mm,
            width=w + 2*padding_mm,
            height=h + 2*padding_mm
        )


# ─── Net and Connection Models ─────────────────────────────────────────────────

class Net(BaseModel):
    """
    Electrical net with properties for routing and analysis.
    """
    model_config = ConfigDict(frozen=True)
    
    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Net name"
    )
    type: NetType = Field(default=NetType.SIGNAL)
    class_: Literal["default", "power", "signal", "bus"] = Field(default="default", alias="class")
    
    # Electrical properties
    voltage: Optional[float] = Field(default=None, description="Nominal voltage (V)")
    current_max: Optional[float] = Field(default=None, description="Maximum current (A)")
    frequency: Optional[float] = Field(default=None, description="Signal frequency (Hz)")
    
    # Physical properties
    length_estimated_mm: Optional[float] = Field(default=None, gt=0)
    trace_width_mm: Optional[float] = Field(default=None, gt=0, le=10)
    impedance_ohms: Optional[float] = Field(default=None, gt=0)
    
    # Routing constraints
    layer_constraint: Optional[Literal["F.Cu", "B.Cu", "any", "paired"]] = Field(default=None)
    length_match_group: Optional[str] = Field(default=None)
    
    @field_validator('name')
    @classmethod
    def validate_net_name(cls, v: str) -> str:
        """Validate and normalize net names."""
        # Remove illegal characters
        v = re.sub(r'[^a-zA-Z0-9_\-/+]', '_', v)
        
        # Standardize power nets
        v_upper = v.upper()
        if v_upper in ['VCC', 'VDD', 'VSUP', 'VPLUS', 'VPWR']:
            return 'VCC'
        if v_upper in ['GND', 'VSS', 'VEE', 'VMINUS', 'AGND', 'DGND', 'PGND', 'SGND']:
            return 'GND'
        if re.match(r'^\d+\.\d+V$|^\d+V$', v_upper):
            return v_upper
        
        return v
    
    @property
    def is_power(self) -> bool:
        return self.type in [NetType.POWER, NetType.GROUND] or self.name in ['VCC', 'GND', '3V3', '5V']
    
    @property
    def is_critical(self) -> bool:
        """Nets requiring special attention."""
        return self.type in [NetType.CLOCK, NetType.HIGH_VOLTAGE, NetType.DIFFERENTIAL]


class Connection(BaseModel):
    """
    A net connection linking multiple pins.
    Replaces simple string-based connections with structured data.
    """
    model_config = ConfigDict(frozen=True)
    
    net: str = Field(..., description="Net name")
    pins: List[str] = Field(
        ...,
        min_length=2,
        description="Connected pins as 'REF.PIN' (e.g. ['R1.1', 'U1.8'])"
    )
    
    @field_validator('pins')
    @classmethod
    def validate_pins(cls, pins: List[str]) -> List[str]:
        """Validate pin reference format."""
        pattern = re.compile(r'^[A-Z][A-Z]?[0-9]+\.[0-9A-Za-z]+$')
        invalid = [p for p in pins if not pattern.match(p)]
        if invalid:
            raise ValueError(f"Invalid pin format(s): {invalid}. Expected 'REF.PIN'")
        return pins
    
    @field_validator('pins')
    @classmethod
    def validate_unique_pins(cls, pins: List[str]) -> List[str]:
        """Ensure no duplicate pins in same net."""
        if len(pins) != len(set(pins)):
            raise ValueError("Duplicate pins in connection")
        return pins
    
    def get_components(self) -> set[str]:
        """Extract unique component references."""
        return {pin.split('.')[0] for pin in self.pins}


# ─── Design Rules ─────────────────────────────────────────────────────────────

class DesignRules(BaseModel):
    """
    PCB design constraints for DFM checking.
    """
    model_config = ConfigDict(frozen=True)
    
    # Clearances
    trace_clearance_mm: float = Field(default=0.2, gt=0)
    via_clearance_mm: float = Field(default=0.2, gt=0)
    component_clearance_mm: float = Field(default=0.5, gt=0)
    board_edge_clearance_mm: float = Field(default=0.5, gt=0)
    
    # Trace specs
    min_trace_width_mm: float = Field(default=0.15, gt=0)
    max_trace_width_mm: float = Field(default=10.0, gt=0)
    min_via_size_mm: float = Field(default=0.3, gt=0)
    
    # Manufacturing
    min_drill_size_mm: float = Field(default=0.2, gt=0)
    solder_mask_sliver_mm: float = Field(default=0.1, gt=0)
    silkscreen_width_mm: float = Field(default=0.15, gt=0)
    
    # Advanced
    max_copper_weight_oz: float = Field(default=2.0, gt=0)
    impedance_controlled: bool = Field(default=False)
    blind_buried_vias: bool = Field(default=False)
    microvias: bool = Field(default=False)
    
    @property
    def is_high_density(self) -> bool:
        """Check if rules indicate high-density design."""
        return self.min_trace_width_mm < 0.15 or self.component_clearance_mm < 0.3


# ─── Top-Level Circuit ─────────────────────────────────────────────────────────

class CircuitMetadata(BaseModel):
    """Structured metadata for circuit tracking."""
    title: str = Field(default="Untitled Circuit")
    author: Optional[str] = Field(default=None)
    organization: Optional[str] = Field(default=None)
    created: datetime = Field(default_factory=datetime.utcnow)
    modified: datetime = Field(default_factory=datetime.utcnow)
    revision: str = Field(default="1.0", pattern=r'^\d+\.\d+(\.\d+)?$')
    tags: List[str] = Field(default_factory=list)
    category: Literal[
        "analog", "digital", "power", "mixed", "rf",
        "indicator", "switching", "timing", "amplifier", "filter", "oscillator",
        "other"
    ] = Field(default="mixed")
    license: Optional[str] = Field(default=None)
    source_url: Optional[str] = Field(default=None)


class CircuitData(BaseModel):
    """
    Complete validated circuit description.
    Version 2.0 with full DFM and placement support.
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "description": "555 Timer Astable Oscillator",
                "components": [],
                "connections": [],
                "board_width": 80.0,
                "board_height": 60.0,
                "schema_version": "2.0"
            }
        }
    )
    
    # Core identity
    description: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Human-readable circuit description"
    )
    schema_version: str = Field(default="2.0", pattern=r'^\d+\.\d+$')
    circuit_id: Optional[str] = Field(
        default=None,
        description="Unique circuit identifier"
    )
    
    # Geometry
    board_width: float = Field(default=80.0, gt=0, le=1000, description="PCB width in mm")
    board_height: float = Field(default=60.0, gt=0, le=1000, description="PCB height in mm")
    board_layers: int = Field(default=2, ge=1, le=32, description="Number of copper layers")
    board_thickness_mm: float = Field(default=1.6, gt=0, le=10)
    
    # Components and connectivity
    components: List[Component] = Field(..., max_length=10000)
    connections: List[Connection] = Field(..., max_length=50000)
    nets: List[Net] = Field(default_factory=list)
    
    # Design data
    design_rules: DesignRules = Field(default_factory=DesignRules)
    metadata: CircuitMetadata = Field(default_factory=CircuitMetadata)
    
    # Placement state
    placement_locked: bool = Field(default=False, description="Prevent auto-placement")
    placement_score: Optional[float] = Field(default=None, ge=0, le=100)
    
    # DFM state
    dfm_checked: bool = Field(default=False)
    dfm_score: float = Field(default=0.0, ge=0, le=100)
    dfm_violations: List[str] = Field(default_factory=list)
    
    # Extensions
    custom_data: Dict[str, Any] = Field(default_factory=dict, description="Application-specific extensions")
    
    @model_validator(mode='after')
    def validate_circuit_integrity(self) -> 'CircuitData':
        """Cross-field validation for circuit consistency."""
        # Check for duplicate references
        refs = [c.ref for c in self.components]
        if len(refs) != len(set(refs)):
            from collections import Counter
            dups = [ref for ref, count in Counter(refs).items() if count > 1]
            raise ValueError(f"Duplicate component references: {dups}")
        
        # Validate all pins in connections exist
        comp_dict = {c.ref: c for c in self.components}
        
        for conn in self.connections:
            for pin_ref in conn.pins:
                ref, pin_num = pin_ref.split('.', 1)
                if ref not in comp_dict:
                    raise ValueError(f"Connection references unknown component: {ref}")
                comp = comp_dict[ref]
                if not comp.get_pin(pin_num):
                    raise ValueError(f"Component {ref} has no pin {pin_num}")
        
        # Validate board size vs components
        for comp in self.components:
            if abs(comp.position.x) > self.board_width / 2 + 10:
                raise ValueError(f"Component {comp.ref} X position outside board")
            if abs(comp.position.y) > self.board_height / 2 + 10:
                raise ValueError(f"Component {comp.ref} Y position outside board")
        
        return self
    
    @field_validator('components')
    @classmethod
    def sort_components(cls, v: List[Component]) -> List[Component]:
        """Sort components by reference for consistent ordering."""
        return sorted(v, key=lambda c: (c.ref_prefix, c.ref_number))
    
    def get_net(self, name: str) -> Optional[Net]:
        """Get net by name."""
        for net in self.nets:
            if net.name == name:
                return net
        return None
    
    def get_component(self, ref: str) -> Optional[Component]:
        """Get component by reference."""
        for comp in self.components:
            if comp.ref == ref:
                return comp
        return None
    
    def get_components_by_type(self, prefix: str) -> List[Component]:
        """Get all components with given reference prefix."""
        return [c for c in self.components if c.ref.startswith(prefix)]
    
    def get_bounding_box(self) -> BoundingBox:
        """Get circuit bounding box."""
        return BoundingBox(
            x=-self.board_width / 2,
            y=-self.board_height / 2,
            width=self.board_width,
            height=self.board_height
        )
    
    def estimate_complexity(self) -> Dict[str, Any]:
        """Calculate circuit complexity metrics."""
        return {
            "component_count": len(self.components),
            "unique_types": len(set(c.part for c in self.components)),
            "net_count": len(self.connections),
            "smd_ratio": sum(1 for c in self.components if c.is_smd) / max(len(self.components), 1),
            "power_nets": sum(1 for n in self.nets if n.is_power),
            "critical_nets": sum(1 for n in self.nets if n.is_critical),
            "density": len(self.components) / (self.board_width * self.board_height / 100),  # per cm²
        }


# ─── API Models ────────────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    """Request to generate circuit from description."""
    model_config = ConfigDict(str_max_length=5000)
    
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Natural language circuit description"
    )
    constraints: Optional[Dict[str, Any]] = Field(default=None)
    preferred_template: Optional[str] = Field(default=None)
    options: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('prompt')
    @classmethod
    def sanitize_prompt(cls, v: str) -> str:
        """Basic prompt injection protection."""
        # Remove potential code injection patterns
        dangerous = ['__import__', 'eval(', 'exec(', 'os.system', 'subprocess', 'import os']
        v_lower = v.lower()
        for d in dangerous:
            if d in v_lower:
                raise ValueError(f"Prompt contains disallowed pattern: {d}")
        return v.strip()


class GenerateResponse(BaseModel):
    """Response from circuit generation."""
    success: bool
    circuit: Optional[CircuitData] = None
    template_used: Optional[str] = None
    llm_used: bool = Field(default=False)
    generation_time_ms: float = Field(default=0.0)
    warnings: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    request_id: str = Field(default="")


class PlacementRequest(BaseModel):
    """Request to optimize component placement."""
    circuit: CircuitData
    constraints: Optional[Dict[str, Any]] = Field(default=None)
    algorithm: Literal["rl", "annealing", "force_directed", "manual"] = Field(default="rl")


class PlacementResponse(BaseModel):
    """Response from placement optimization."""
    success: bool
    circuit: Optional[CircuitData] = None
    algorithm: str = Field(default="unknown")
    improvement_metrics: Dict[str, float] = Field(default_factory=dict)
    iterations: int = Field(default=0)
    error: Optional[str] = None


class DFMCheckRequest(BaseModel):
    """Request DFM analysis."""
    circuit: CircuitData
    manufacturer: Literal["jlcpcb", "pcbway", "oshpark", "eurocircuits", "custom"] = Field(default="jlcpcb")
    service_level: Literal["standard", "advanced", "prototype"] = Field(default="standard")


class DFMViolation(BaseModel):
    """Single DFM violation with fix suggestion."""
    rule_id: str = Field(..., pattern=r'^DFM-\d{3}$')
    type: str = Field(..., max_length=50)
    severity: Severity = Field(default=Severity.WARNING)
    message: str = Field(..., max_length=500)
    location: Optional[Point2D] = None
    components_involved: List[str] = Field(default_factory=list)
    suggested_fix: Optional[str] = Field(default=None, max_length=500)
    auto_fixable: bool = Field(default=False)


class DFMResponse(BaseModel):
    """DFM check results."""
    success: bool
    manufacturer: str = Field(default="jlcpcb")
    violations: List[DFMViolation] = Field(default_factory=list)
    score: float = Field(default=0.0, ge=0, le=100)
    passed: bool = Field(default=False)
    checked_rules: List[str] = Field(default_factory=list)
    estimated_cost: Optional[float] = Field(default=None, description="Estimated PCB cost USD")
    error: Optional[str] = None


class ExportRequest(BaseModel):
    """Request to export circuit."""
    circuit: CircuitData
    format: Literal["kicad_sch", "kicad_pcb", "json", "csv_bom", "gerber"] = Field(default="kicad_sch")
    options: Dict[str, Any] = Field(default_factory=dict)


class ExportResponse(BaseModel):
    """Export result."""
    success: bool
    filename: str = Field(default="")
    format: str = Field(default="kicad_sch")
    file_size_bytes: Optional[int] = Field(default=None)
    download_url: Optional[str] = Field(default=None)
    expires_at: Optional[datetime] = Field(default=None)
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """System health status."""
    status: Literal["healthy", "degraded", "unhealthy"] = Field(default="healthy")
    version: str = Field(default="2.0.0")
    schema_version: str = Field(default="2.0")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    uptime_seconds: float = Field(default=0.0)
    
    # Capabilities
    llm_available: bool = Field(default=False)
    placement_engine_available: bool = Field(default=False)
    drc_engine_available: bool = Field(default=False)
    
    # Resources
    templates_loaded: int = Field(default=0)
    circuits_generated: int = Field(default=0)
    memory_usage_mb: Optional[float] = Field(default=None)
    
    # Latency indicators
    avg_generation_time_ms: Optional[float] = Field(default=None)
    last_error: Optional[str] = Field(default=None)


# ─── Utility Functions ─────────────────────────────────────────────────────────

def migrate_v1_to_v2(circuit_v1: Dict[str, Any]) -> CircuitData:
    """
    Migrate version 1.0 circuit data to 2.0 schema.
    Handles field renames and structure changes.
    """
    # Deep copy to avoid mutation
    import copy
    data = copy.deepcopy(circuit_v1)
    
    # Handle old 'connections' format (list of strings vs structured)
    if 'connections' in data and data['connections']:
        first = data['connections'][0]
        if isinstance(first, str):
            # Old format: ["R1.1-U1.3", ...] -> new format
            new_conns = []
            for conn_str in data['connections']:
                if '-' in conn_str:
                    pins = conn_str.split('-')
                    # Generate net name
                    net_name = f"NET{len(new_conns)+1}"
                    new_conns.append({
                        "net": net_name,
                        "pins": pins
                    })
            data['connections'] = new_conns
    
    # Add default metadata if missing
    if 'metadata' not in data or isinstance(data.get('metadata'), dict):
        data['metadata'] = CircuitMetadata(
            **(data.get('metadata') or {})
        ).model_dump()
    
    # Ensure schema version
    data['schema_version'] = "2.0"
    
    return CircuitData(**data)


def validate_kicad_footprint(footprint: str) -> bool:
    """Validate KiCad footprint string format."""
    if not footprint:
        return True  # Empty is valid (unassigned)
    
    # Modern format: Library:Footprint
    if ':' in footprint:
        parts = footprint.split(':', 1)
        return len(parts) == 2 and all(parts)
    
    # Legacy format: just Footprint (accept but warn)
    return len(footprint) > 0 and not footprint.startswith(':')