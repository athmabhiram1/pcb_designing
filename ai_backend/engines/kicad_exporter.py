"""
KiCad Schematic Exporter — Zero-dependency S-expression writer.

Converts CircuitData (Pydantic) → valid .kicad_sch file (KiCad 8.0+ compatible).
No external libraries required. Pure Python string generation.
"""
import uuid
import math
import logging
import os
from typing import Optional
from dataclasses import dataclass, field

# Import from parent package
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from circuit_schema import CircuitData, Component, Connection, Pin

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

KICAD_VERSION = 20231120
GENERATOR = "kicad_copilot"
GENERATOR_VERSION = "1.0"
DEFAULT_PAPER = "A4"

# Schematic grid: KiCad uses 1.27mm (50mil) grid for schematics
GRID = 2.54  # mm — standard KiCad schematic grid
PIN_LENGTH = 2.54  # mm

# Default symbol spacing when no coordinates provided
DEFAULT_X_START = 50.8
DEFAULT_Y_START = 50.8
DEFAULT_X_SPACING = 30.48
DEFAULT_Y_SPACING = 25.4
MAX_COLS = 5

# Smart layout spacing constants
SCH_MARGIN        = 25.4   # mm from sheet edge
IC_X_SPACING      = 50.8   # horizontal gap between ICs
IC_Y_SPACING      = 45.72  # vertical gap between IC rows
PASSIVE_X_SPACING = 15.24  # gap between passives in the same row
PASSIVE_Y_SPACING = 20.32  # vertical gap between passive rows
BYPASS_OFFSET_Y   = 20.32  # how far bypass caps sit above/below IC centre

# Text sizes
REF_FONT_SIZE = 1.27
VALUE_FONT_SIZE = 1.27
LABEL_FONT_SIZE = 1.27


# =============================================================================
# S-Expression Builder
# =============================================================================

def _uuid() -> str:
    """Generate a KiCad-compatible UUID."""
    return str(uuid.uuid4())


def _quote(s: str) -> str:
    """Quote a string for S-expression output."""
    if not s or " " in s or ":" in s or '"' in s or any(c in s for c in "(){}[]"):
        return f'"{s}"'
    return f'"{s}"'


def _xy(x: float, y: float) -> str:
    """Format an (xy x y) point."""
    return f"(xy {x:.4f} {y:.4f})"


def _at(x: float, y: float, angle: float = 0) -> str:
    """Format an (at x y angle) position. KiCad 9.0 requires the angle always."""
    return f"(at {x:.4f} {y:.4f} {angle:.0f})"


def _effects(size: float = 1.27, hide: bool = False, justify: str = "") -> str:
    """Format text effects block."""
    parts = [f"(font (size {size} {size}))"]
    if justify:
        parts.append(f"(justify {justify})")
    if hide:
        parts.append("hide")
    return f"(effects {' '.join(parts)})"


def _stroke(width: float = 0, stype: str = "default") -> str:
    """Format a stroke definition."""
    return f"(stroke (width {width}) (type {stype}))"


def _property(name: str, value: str, at_x: float, at_y: float,
              angle: float = 0, font_size: float = 1.27,
              hide: bool = False, prop_id: Optional[int] = None) -> str:
    """Format a property block."""
    parts = [f'(property {_quote(name)} {_quote(value)}']
    if prop_id is not None:
        parts[0] = f'(property {_quote(name)} {_quote(value)}'
    parts.append(f"  {_at(at_x, at_y, angle)}")
    parts.append(f"  {_effects(font_size, hide)}")
    parts.append(")")
    return "\n".join(parts)


# =============================================================================
# Symbol Library Definitions
# =============================================================================

# Pre-built lib_symbol templates for common KiCad parts.
# Each entry defines the graphical body + pins for one symbol type.

def _lib_symbol_resistor(lib_id: str = "Device:R") -> str:
    """Generate lib_symbol entry for a resistor."""
    uid = "R"
    return f"""    (symbol {_quote(lib_id)}
      (pin_numbers hide)
      (pin_names (offset 0) hide)
      (in_bom yes) (on_board yes)
      (property "Reference" "R" {_at(2.032, 0, 90)} {_effects(REF_FONT_SIZE)})
      (property "Value" "R" {_at(-1.778, 0, 90)} {_effects(VALUE_FONT_SIZE)})
      (property "Footprint" "" {_at(0, 0, 0)} {_effects(REF_FONT_SIZE, hide=True)})
      (property "Datasheet" "~" {_at(0, 0, 0)} {_effects(REF_FONT_SIZE, hide=True)})
      (symbol "{uid}_0_1"
        (rectangle (start -1.016 -2.54) (end 1.016 2.54)
          {_stroke(0.254)}
          (fill (type none))
        )
      )
      (symbol "{uid}_1_1"
        (pin passive line {_at(0, 3.81, 270)} (length {PIN_LENGTH})
          (name "~" {_effects(REF_FONT_SIZE)})
          (number "1" {_effects(REF_FONT_SIZE)})
        )
        (pin passive line {_at(0, -3.81, 90)} (length {PIN_LENGTH})
          (name "~" {_effects(REF_FONT_SIZE)})
          (number "2" {_effects(REF_FONT_SIZE)})
        )
      )
    )"""


def _lib_symbol_capacitor(lib_id: str = "Device:C") -> str:
    """Generate lib_symbol entry for a capacitor."""
    uid = "C"
    return f"""    (symbol {_quote(lib_id)}
      (pin_numbers hide)
      (pin_names (offset 0.254) hide)
      (in_bom yes) (on_board yes)
      (property "Reference" "C" {_at(1.524, 0, 90)} {_effects(REF_FONT_SIZE)})
      (property "Value" "C" {_at(-1.524, 0, 90)} {_effects(VALUE_FONT_SIZE)})
      (property "Footprint" "" {_at(0, 0, 0)} {_effects(REF_FONT_SIZE, hide=True)})
      (property "Datasheet" "~" {_at(0, 0, 0)} {_effects(REF_FONT_SIZE, hide=True)})
      (symbol "{uid}_0_1"
        (polyline
          (pts {_xy(-2.032, -0.762)} {_xy(2.032, -0.762)})
          {_stroke(0.508)}
          (fill (type none))
        )
        (polyline
          (pts {_xy(-2.032, 0.762)} {_xy(2.032, 0.762)})
          {_stroke(0.508)}
          (fill (type none))
        )
      )
      (symbol "{uid}_1_1"
        (pin passive line {_at(0, 3.81, 270)} (length 2.794)
          (name "~" {_effects(REF_FONT_SIZE)})
          (number "1" {_effects(REF_FONT_SIZE)})
        )
        (pin passive line {_at(0, -3.81, 90)} (length 2.794)
          (name "~" {_effects(REF_FONT_SIZE)})
          (number "2" {_effects(REF_FONT_SIZE)})
        )
      )
    )"""


def _lib_symbol_led(lib_id: str = "Device:LED") -> str:
    """Generate lib_symbol entry for an LED."""
    uid = "LED"
    return f"""    (symbol {_quote(lib_id)}
      (pin_numbers hide)
      (pin_names (offset 1.016) hide)
      (in_bom yes) (on_board yes)
      (property "Reference" "D" {_at(1.524, 1.27, 0)} {_effects(REF_FONT_SIZE)})
      (property "Value" "LED" {_at(1.524, -1.27, 0)} {_effects(VALUE_FONT_SIZE)})
      (property "Footprint" "" {_at(0, 0, 0)} {_effects(REF_FONT_SIZE, hide=True)})
      (property "Datasheet" "~" {_at(0, 0, 0)} {_effects(REF_FONT_SIZE, hide=True)})
      (symbol "{uid}_0_1"
        (polyline
          (pts {_xy(-1.27, -1.27)} {_xy(-1.27, 1.27)})
          {_stroke(0.254)}
          (fill (type none))
        )
        (polyline
          (pts {_xy(-1.27, 0)} {_xy(1.27, 0)})
          {_stroke(0)}
          (fill (type none))
        )
        (polyline
          (pts {_xy(1.27, -1.27)} {_xy(1.27, 1.27)} {_xy(-1.27, 0)} {_xy(1.27, -1.27)})
          {_stroke(0.254)}
          (fill (type none))
        )
        (polyline
          (pts {_xy(-3.048, -0.762)} {_xy(-4.572, -2.286)} {_xy(-3.81, -2.286)} {_xy(-4.572, -2.286)} {_xy(-4.572, -1.524)})
          {_stroke(0.254)}
          (fill (type none))
        )
        (polyline
          (pts {_xy(-1.778, -0.762)} {_xy(-3.302, -2.286)} {_xy(-2.54, -2.286)} {_xy(-3.302, -2.286)} {_xy(-3.302, -1.524)})
          {_stroke(0.254)}
          (fill (type none))
        )
      )
      (symbol "{uid}_1_1"
        (pin passive line {_at(-3.81, 0, 0)} (length {PIN_LENGTH})
          (name "A" {_effects(REF_FONT_SIZE)})
          (number "1" {_effects(REF_FONT_SIZE)})
        )
        (pin passive line {_at(3.81, 0, 180)} (length {PIN_LENGTH})
          (name "K" {_effects(REF_FONT_SIZE)})
          (number "2" {_effects(REF_FONT_SIZE)})
        )
      )
    )"""


def _lib_symbol_ne555(lib_id: str = "Timer:NE555") -> str:
    """Generate lib_symbol entry for NE555 timer IC."""
    uid = "NE555"
    return f"""    (symbol {_quote(lib_id)}
      (pin_names (offset 1.016))
      (in_bom yes) (on_board yes)
      (property "Reference" "U" {_at(8.89, 8.89, 0)} {_effects(REF_FONT_SIZE)})
      (property "Value" "NE555" {_at(8.89, -8.89, 0)} {_effects(VALUE_FONT_SIZE)})
      (property "Footprint" "" {_at(0, 0, 0)} {_effects(REF_FONT_SIZE, hide=True)})
      (property "Datasheet" "http://www.ti.com/lit/ds/symlink/ne555.pdf" {_at(0, 0, 0)} {_effects(REF_FONT_SIZE, hide=True)})
      (symbol "{uid}_0_1"
        (rectangle (start -7.62 7.62) (end 7.62 -7.62)
          {_stroke(0.254)}
          (fill (type background))
        )
      )
      (symbol "{uid}_1_1"
        (pin power_in line {_at(0, -10.16, 90)} (length {PIN_LENGTH})
          (name "GND" {_effects(REF_FONT_SIZE)})
          (number "1" {_effects(REF_FONT_SIZE)})
        )
        (pin input line {_at(-10.16, 2.54, 0)} (length {PIN_LENGTH})
          (name "TR" {_effects(REF_FONT_SIZE)})
          (number "2" {_effects(REF_FONT_SIZE)})
        )
        (pin output line {_at(10.16, 0, 180)} (length {PIN_LENGTH})
          (name "Q" {_effects(REF_FONT_SIZE)})
          (number "3" {_effects(REF_FONT_SIZE)})
        )
        (pin input inverted {_at(10.16, 5.08, 180)} (length {PIN_LENGTH})
          (name "R" {_effects(REF_FONT_SIZE)})
          (number "4" {_effects(REF_FONT_SIZE)})
        )
        (pin input line {_at(-10.16, -2.54, 0)} (length {PIN_LENGTH})
          (name "CV" {_effects(REF_FONT_SIZE)})
          (number "5" {_effects(REF_FONT_SIZE)})
        )
        (pin input line {_at(-10.16, 5.08, 0)} (length {PIN_LENGTH})
          (name "THR" {_effects(REF_FONT_SIZE)})
          (number "6" {_effects(REF_FONT_SIZE)})
        )
        (pin open_collector line {_at(10.16, -5.08, 180)} (length {PIN_LENGTH})
          (name "DIS" {_effects(REF_FONT_SIZE)})
          (number "7" {_effects(REF_FONT_SIZE)})
        )
        (pin power_in line {_at(0, 10.16, 270)} (length {PIN_LENGTH})
          (name "VCC" {_effects(REF_FONT_SIZE)})
          (number "8" {_effects(REF_FONT_SIZE)})
        )
      )
    )"""


def _lib_symbol_inductor(lib_id: str = "Device:L") -> str:
    """Generate lib_symbol entry for an inductor."""
    uid = "L"
    return f"""    (symbol {_quote(lib_id)}
      (pin_numbers hide)
      (pin_names (offset 1.016) hide)
      (in_bom yes) (on_board yes)
      (property "Reference" "L" {_at(1.27, 0, 90)} {_effects(REF_FONT_SIZE)})
      (property "Value" "L" {_at(-1.016, 0, 90)} {_effects(VALUE_FONT_SIZE)})
      (property "Footprint" "" {_at(0, 0, 0)} {_effects(REF_FONT_SIZE, hide=True)})
      (property "Datasheet" "~" {_at(0, 0, 0)} {_effects(REF_FONT_SIZE, hide=True)})
      (symbol "{uid}_0_1"
        (arc (start 0 -2.54) (mid 0.6323 -1.905) (end 0 -1.27)
          {_stroke(0)} (fill (type none)))
        (arc (start 0 -1.27) (mid 0.6323 -0.635) (end 0 0)
          {_stroke(0)} (fill (type none)))
        (arc (start 0 0) (mid 0.6323 0.635) (end 0 1.27)
          {_stroke(0)} (fill (type none)))
        (arc (start 0 1.27) (mid 0.6323 1.905) (end 0 2.54)
          {_stroke(0)} (fill (type none)))
      )
      (symbol "{uid}_1_1"
        (pin passive line {_at(0, 3.81, 270)} (length 1.27)
          (name "~" {_effects(REF_FONT_SIZE)})
          (number "1" {_effects(REF_FONT_SIZE)})
        )
        (pin passive line {_at(0, -3.81, 90)} (length 1.27)
          (name "~" {_effects(REF_FONT_SIZE)})
          (number "2" {_effects(REF_FONT_SIZE)})
        )
      )
    )"""


def _lib_symbol_diode(lib_id: str = "Device:D") -> str:
    """Generate lib_symbol entry for a diode."""
    uid = "D"
    return f"""    (symbol {_quote(lib_id)}
      (pin_numbers hide)
      (pin_names (offset 1.016) hide)
      (in_bom yes) (on_board yes)
      (property "Reference" "D" {_at(0, 2.54, 0)} {_effects(REF_FONT_SIZE)})
      (property "Value" "D" {_at(0, -2.54, 0)} {_effects(VALUE_FONT_SIZE)})
      (property "Footprint" "" {_at(0, 0, 0)} {_effects(REF_FONT_SIZE, hide=True)})
      (property "Datasheet" "~" {_at(0, 0, 0)} {_effects(REF_FONT_SIZE, hide=True)})
      (symbol "{uid}_0_1"
        (polyline
          (pts {_xy(-1.27, 1.27)} {_xy(-1.27, -1.27)})
          {_stroke(0.254)}
          (fill (type none))
        )
        (polyline
          (pts {_xy(1.27, 0)} {_xy(-1.27, 0)})
          {_stroke(0)}
          (fill (type none))
        )
        (polyline
          (pts {_xy(1.27, 1.27)} {_xy(1.27, -1.27)} {_xy(-1.27, 0)} {_xy(1.27, 1.27)})
          {_stroke(0.254)}
          (fill (type none))
        )
      )
      (symbol "{uid}_1_1"
        (pin passive line {_at(-3.81, 0, 0)} (length {PIN_LENGTH})
          (name "A" {_effects(REF_FONT_SIZE)})
          (number "1" {_effects(REF_FONT_SIZE)})
        )
        (pin passive line {_at(3.81, 0, 180)} (length {PIN_LENGTH})
          (name "K" {_effects(REF_FONT_SIZE)})
          (number "2" {_effects(REF_FONT_SIZE)})
        )
      )
    )"""


def _lib_symbol_regulator(lib_id: str = "Regulator_Linear:AMS1117-3.3") -> str:
    """Generate lib_symbol entry for a 3-pin linear voltage regulator (AMS1117 style)."""
    uid = "AMS1117"
    return f"""    (symbol {_quote(lib_id)}
      (pin_names (offset 1.016))
      (in_bom yes) (on_board yes)
      (property "Reference" "U" {_at(0, 7.62, 0)} {_effects(REF_FONT_SIZE)})
      (property "Value" "AMS1117-3.3" {_at(0, -7.62, 0)} {_effects(VALUE_FONT_SIZE)})
      (property "Footprint" "" {_at(0, 0, 0)} {_effects(REF_FONT_SIZE, hide=True)})
      (property "Datasheet" "~" {_at(0, 0, 0)} {_effects(REF_FONT_SIZE, hide=True)})
      (symbol "{uid}_0_1"
        (rectangle (start -5.08 5.08) (end 5.08 -5.08)
          {_stroke(0.254)}
          (fill (type background))
        )
      )
      (symbol "{uid}_1_1"
        (pin power_in line {_at(-10.16, 0, 0)} (length {PIN_LENGTH})
          (name "IN" {_effects(REF_FONT_SIZE)})
          (number "3" {_effects(REF_FONT_SIZE)})
        )
        (pin power_out line {_at(10.16, 0, 180)} (length {PIN_LENGTH})
          (name "OUT" {_effects(REF_FONT_SIZE)})
          (number "2" {_effects(REF_FONT_SIZE)})
        )
        (pin power_in line {_at(0, -10.16, 90)} (length {PIN_LENGTH})
          (name "GND" {_effects(REF_FONT_SIZE)})
          (number "1" {_effects(REF_FONT_SIZE)})
        )
      )
    )"""


def _lib_symbol_opamp(lib_id: str = "Amplifier_Operational:LM358") -> str:
    """Generate lib_symbol for a single-section op-amp (LM358 / LM741 style)."""
    uid = "LM358"
    return f"""    (symbol {_quote(lib_id)}
      (pin_names (offset 1.016))
      (in_bom yes) (on_board yes)
      (property "Reference" "U" {_at(0, 7.62, 0)} {_effects(REF_FONT_SIZE)})
      (property "Value" "LM358" {_at(0, -7.62, 0)} {_effects(VALUE_FONT_SIZE)})
      (property "Footprint" "" {_at(0, 0, 0)} {_effects(REF_FONT_SIZE, hide=True)})
      (property "Datasheet" "~" {_at(0, 0, 0)} {_effects(REF_FONT_SIZE, hide=True)})
      (symbol "{uid}_0_1"
        (polyline
          (pts {_xy(-5.08, 5.08)} {_xy(-5.08, -5.08)} {_xy(5.08, 0)} {_xy(-5.08, 5.08)})
          {_stroke(0.254)}
          (fill (type background))
        )
      )
      (symbol "{uid}_1_1"
        (pin input line {_at(-7.62, 2.54, 0)} (length {PIN_LENGTH})
          (name "IN-" {_effects(REF_FONT_SIZE)})
          (number "2" {_effects(REF_FONT_SIZE)})
        )
        (pin input line {_at(-7.62, -2.54, 0)} (length {PIN_LENGTH})
          (name "IN+" {_effects(REF_FONT_SIZE)})
          (number "3" {_effects(REF_FONT_SIZE)})
        )
        (pin power_in line {_at(0, -7.62, 90)} (length {PIN_LENGTH})
          (name "GND" {_effects(REF_FONT_SIZE)})
          (number "4" {_effects(REF_FONT_SIZE)})
        )
        (pin output line {_at(7.62, 0, 180)} (length {PIN_LENGTH})
          (name "OUT" {_effects(REF_FONT_SIZE)})
          (number "1" {_effects(REF_FONT_SIZE)})
        )
        (pin power_in line {_at(0, 7.62, 270)} (length {PIN_LENGTH})
          (name "VCC" {_effects(REF_FONT_SIZE)})
          (number "8" {_effects(REF_FONT_SIZE)})
        )
      )
    )"""


def _lib_symbol_nmos(lib_id: str = "Device:Q_NMOS_GSD") -> str:
    """Generate lib_symbol for an N-channel MOSFET (G/S/D pinout, e.g. 2N7002)."""
    uid = "NMOS"
    return f"""    (symbol {_quote(lib_id)}
      (pin_names (offset 1.016))
      (in_bom yes) (on_board yes)
      (property "Reference" "Q" {_at(5.08, 1.905, 0)} {_effects(REF_FONT_SIZE)})
      (property "Value" "Q_NMOS_GSD" {_at(5.08, -1.905, 0)} {_effects(VALUE_FONT_SIZE)})
      (property "Footprint" "" {_at(0, 0, 0)} {_effects(REF_FONT_SIZE, hide=True)})
      (property "Datasheet" "~" {_at(0, 0, 0)} {_effects(REF_FONT_SIZE, hide=True)})
      (symbol "{uid}_0_1"
        (polyline
          (pts {_xy(0, -2.54)} {_xy(0, 2.54)})
          {_stroke(0.508)}
          (fill (type none))
        )
        (polyline
          (pts {_xy(0, -1.27)} {_xy(2.54, -1.27)} {_xy(2.54, -2.54)})
          {_stroke(0.254)}
          (fill (type none))
        )
        (polyline
          (pts {_xy(0, 1.27)} {_xy(2.54, 1.27)} {_xy(2.54, 2.54)})
          {_stroke(0.254)}
          (fill (type none))
        )
        (polyline
          (pts {_xy(0, 0)} {_xy(2.54, 0)})
          {_stroke(0.254)}
          (fill (type none))
        )
        (polyline
          (pts {_xy(2.54, 0)} {_xy(1.778, 0.508)} {_xy(1.778, -0.508)} {_xy(2.54, 0)})
          {_stroke(0)}
          (fill (type outline))
        )
      )
      (symbol "{uid}_1_1"
        (pin input line {_at(-2.54, 0, 0)} (length {PIN_LENGTH})
          (name "G" {_effects(REF_FONT_SIZE)})
          (number "2" {_effects(REF_FONT_SIZE)})
        )
        (pin passive line {_at(2.54, -5.08, 90)} (length {PIN_LENGTH})
          (name "S" {_effects(REF_FONT_SIZE)})
          (number "1" {_effects(REF_FONT_SIZE)})
        )
        (pin passive line {_at(2.54, 5.08, 270)} (length {PIN_LENGTH})
          (name "D" {_effects(REF_FONT_SIZE)})
          (number "3" {_effects(REF_FONT_SIZE)})
        )
      )
    )"""


def _lib_symbol_power(symbol_name: str, pin_name: str, pin_number: str = "1",
                      direction: int = 0) -> str:
    """Generate lib_symbol entry for a power symbol (VCC, GND, etc.)."""
    lib_id = f"power:{symbol_name}"
    uid = symbol_name
    return f"""    (symbol {_quote(lib_id)}
      (power)
      (pin_numbers hide)
      (pin_names (offset 0) hide)
      (in_bom yes) (on_board yes)
      (property "Reference" "#PWR" {_at(0, 2.54, 0)} {_effects(REF_FONT_SIZE, hide=True)})
      (property "Value" "{symbol_name}" {_at(0, 3.81, 0)} {_effects(VALUE_FONT_SIZE)})
      (property "Footprint" "" {_at(0, 0, 0)} {_effects(REF_FONT_SIZE, hide=True)})
      (property "Datasheet" "" {_at(0, 0, 0)} {_effects(REF_FONT_SIZE, hide=True)})
      (symbol "{uid}_0_1"
        (pin power_in line {_at(0, 0, direction)} (length 0)
          (name "{pin_name}" {_effects(REF_FONT_SIZE)})
          (number "{pin_number}" {_effects(REF_FONT_SIZE)})
        )
      )
    )"""


def _lib_symbol_generic(lib_id: str, ref_prefix: str, part_name: str,
                        pins: list[Pin]) -> str:
    """Generate a generic rectangular lib_symbol for unknown components."""
    uid = part_name.replace(" ", "_").replace("-", "_")
    n_pins = len(pins)
    # Divide pins: left side & right side
    left_count = (n_pins + 1) // 2
    right_count = n_pins - left_count

    # Box dimensions
    box_half_w = 7.62
    box_half_h = max(left_count, right_count, 1) * 2.54 + 2.54

    lines = []
    lines.append(f'    (symbol {_quote(lib_id)}')
    lines.append(f'      (pin_names (offset 1.016))')
    lines.append(f'      (in_bom yes) (on_board yes)')
    lines.append(f'      (property "Reference" "{ref_prefix}" {_at(box_half_w + 1.27, box_half_h, 0)} {_effects(REF_FONT_SIZE)})')
    lines.append(f'      (property "Value" "{part_name}" {_at(box_half_w + 1.27, -box_half_h, 0)} {_effects(VALUE_FONT_SIZE)})')
    lines.append(f'      (property "Footprint" "" {_at(0, 0, 0)} {_effects(REF_FONT_SIZE, hide=True)})')
    lines.append(f'      (property "Datasheet" "~" {_at(0, 0, 0)} {_effects(REF_FONT_SIZE, hide=True)})')

    # Body rectangle
    lines.append(f'      (symbol "{uid}_0_1"')
    lines.append(f'        (rectangle (start -{box_half_w} {box_half_h}) (end {box_half_w} -{box_half_h})')
    lines.append(f'          {_stroke(0.254)}')
    lines.append(f'          (fill (type background))')
    lines.append(f'        )')
    lines.append(f'      )')

    # Pins
    lines.append(f'      (symbol "{uid}_1_1"')
    left_pins = pins[:left_count]
    right_pins = pins[left_count:]

    for i, pin in enumerate(left_pins):
        py = box_half_h - 2.54 - i * 2.54
        pname = pin.name if pin.name else f"~"
        lines.append(f'        (pin passive line {_at(-box_half_w - PIN_LENGTH, py, 0)} (length {PIN_LENGTH})')
        lines.append(f'          (name "{pname}" {_effects(REF_FONT_SIZE)})')
        lines.append(f'          (number "{pin.number}" {_effects(REF_FONT_SIZE)})')
        lines.append(f'        )')

    for i, pin in enumerate(right_pins):
        py = box_half_h - 2.54 - i * 2.54
        pname = pin.name if pin.name else f"~"
        lines.append(f'        (pin passive line {_at(box_half_w + PIN_LENGTH, py, 180)} (length {PIN_LENGTH})')
        lines.append(f'          (name "{pname}" {_effects(REF_FONT_SIZE)})')
        lines.append(f'          (number "{pin.number}" {_effects(REF_FONT_SIZE)})')
        lines.append(f'        )')

    lines.append(f'      )')
    lines.append(f'    )')
    return "\n".join(lines)


# Registry mapping lib:part → generator function
LIB_SYMBOL_GENERATORS = {
    "Device:R": _lib_symbol_resistor,
    "Device:C": _lib_symbol_capacitor,
    "Device:LED": _lib_symbol_led,
    "Device:L": _lib_symbol_inductor,
    "Device:D": _lib_symbol_diode,
    "Timer:NE555": _lib_symbol_ne555,
    "Regulator_Linear:AMS1117-3.3": _lib_symbol_regulator,
    "Regulator_Linear:LM7805": lambda lib_id: _lib_symbol_regulator(lib_id),
    "Amplifier_Operational:LM358": _lib_symbol_opamp,
    "Device:Q_NMOS_GSD": _lib_symbol_nmos,
}


# =============================================================================
# Pin Position Lookup for Known Symbols
# =============================================================================

# Map (lib:part, pin_number) → (dx, dy) offset from component center
# These match the pin positions in the lib_symbol definitions above.

KNOWN_PIN_OFFSETS: dict[str, dict[str, tuple[float, float]]] = {
    "Device:R": {
        "1": (0, 3.81),
        "2": (0, -3.81),
    },
    "Device:C": {
        "1": (0, 3.81),
        "2": (0, -3.81),
    },
    "Device:L": {
        "1": (0, 3.81),
        "2": (0, -3.81),
    },
    "Device:LED": {
        "1": (-3.81, 0),   # Anode
        "2": (3.81, 0),    # Cathode
    },
    "Device:D": {
        "1": (-3.81, 0),   # Anode
        "2": (3.81, 0),    # Cathode
    },
    "Timer:NE555": {
        "1": (0, -10.16),     # GND
        "2": (-10.16, 2.54),  # TRIG
        "3": (10.16, 0),      # OUT
        "4": (10.16, 5.08),   # RESET
        "5": (-10.16, -2.54), # CV
        "6": (-10.16, 5.08),  # THR
        "7": (10.16, -5.08),  # DIS
        "8": (0, 10.16),      # VCC
    },
    "Regulator_Linear:AMS1117-3.3": {
        "1": (0, -10.16),   # GND
        "2": (10.16, 0),    # OUTPUT
        "3": (-10.16, 0),   # INPUT
    },
    "Regulator_Linear:LM7805": {
        "1": (-10.16, 0),   # INPUT
        "2": (0, -10.16),   # GND
        "3": (10.16, 0),    # OUTPUT
    },
    "Amplifier_Operational:LM358": {
        "1": (7.62, 0),     # OUT
        "2": (-7.62, 2.54), # IN-
        "3": (-7.62, -2.54),# IN+
        "4": (0, -7.62),    # GND
        "8": (0, 7.62),     # VCC
    },
    "Device:Q_NMOS_GSD": {
        "1": (2.54, -5.08), # S (Source)
        "2": (-2.54, 0),    # G (Gate)
        "3": (2.54, 5.08),  # D (Drain)
    },
}


# =============================================================================
# Main Exporter Class
# =============================================================================

class KiCadSchematicWriter:
    """
    Converts a CircuitData object into a valid .kicad_sch S-expression string.

    Usage:
        writer = KiCadSchematicWriter()
        sch_content = writer.export(circuit_data)
        with open("output.kicad_sch", "w") as f:
            f.write(sch_content)
    """

    def __init__(self):
        self._pwr_counter = 0

    def export(self, circuit: CircuitData) -> str:
        """
        Full pipeline: CircuitData → .kicad_sch string.

        Args:
            circuit: The CircuitData Pydantic model to export.

        Returns:
            A complete .kicad_sch file as a string.
        """
        self._pwr_counter = 0

        # Auto-place components if no positions are set
        self._auto_place(circuit)

        # Build component lookup: ref → Component
        comp_map = {c.ref: c for c in circuit.components}

        # Collect unique lib:part IDs
        unique_parts = {}
        for c in circuit.components:
            lib_id = f"{c.lib}:{c.part}"
            if lib_id not in unique_parts:
                unique_parts[lib_id] = c

        # Detect power nets
        power_nets = self._detect_power_nets(circuit.connections)

        # Assemble sections
        header = self._build_header()
        lib_symbols = self._build_lib_symbols(unique_parts, power_nets)
        symbols = self._build_symbols(circuit.components)
        power_syms = self._build_power_symbols(circuit.connections, comp_map, power_nets)
        wires = self._build_wires(circuit.connections, comp_map, power_nets)
        labels = self._build_labels(circuit.connections, comp_map, power_nets)
        footer = self._build_footer()

        # Assemble full file
        sections = [
            header,
            "",
            lib_symbols,
            "",
            *symbols,
            "",
            *power_syms,
            "",
            *wires,
            "",
            *labels,
            "",
            footer,
            ")",  # close kicad_sch
        ]

        return "\n".join(s for s in sections if s is not None) + "\n"

    # -------------------------------------------------------------------------
    # Header & Footer
    # -------------------------------------------------------------------------

    def _build_header(self) -> str:
        root_uuid = _uuid()
        return f"""(kicad_sch
  (version {KICAD_VERSION})
  (generator {_quote(GENERATOR)})
  (generator_version {_quote(GENERATOR_VERSION)})
  (uuid {_quote(root_uuid)})
  (paper {_quote(DEFAULT_PAPER)})"""

    def _build_footer(self) -> str:
        return f"""  (sheet_instances
    (path "/" (page "1"))
  )"""

    # -------------------------------------------------------------------------
    # Auto-placement  (smart topology-aware layout)
    # -------------------------------------------------------------------------

    @staticmethod
    def _comp_class(ref: str) -> str:
        """Return coarse component class from reference designator."""
        prefix = "".join(c for c in ref if c.isalpha()).upper()
        if prefix in ("U", "IC"):
            return "ic"
        if prefix in ("Q", "M", "T"):
            return "transistor"
        if prefix in ("J", "P", "CN", "X", "SW", "BTN"):
            return "connector"
        if prefix in ("R",):
            return "resistor"
        if prefix in ("C",):
            return "capacitor"
        if prefix in ("L",):
            return "inductor"
        if prefix in ("D", "LED", "Z"):
            return "diode"
        return "other"

    @staticmethod
    def _ic_size(comp) -> tuple:
        """Return (half_w, half_h) bounding estimate in mm for a symbol."""
        n_pins = len(comp.pins) if comp.pins else 2
        half_h = max(7.62, (n_pins / 2) * 2.54 + 2.54)
        half_w = 7.62
        return half_w, half_h

    def _auto_place(self, circuit: CircuitData):
        """
        Smart topology-aware placement for KiCad schematic symbols.

        Layout strategy
        ───────────────
        1. Connectors  – left column (inputs) and right column (outputs).
        2. ICs          – centre column(s), one per row, evenly spaced.
        3. Transistors  – placed to the right of the IC they connect to.
        4. Passives     – resistors/inductors in the signal path placed
                          horizontally between the components they join;
                          bypass/decoupling capacitors placed below their IC.
        5. Remaining    – simple grid at the bottom.
        """
        needs_placement = all(c.x == 0.0 and c.y == 0.0 for c in circuit.components)
        if not needs_placement:
            return

        comps = circuit.components
        if not comps:
            return

        # ── 1. Classify ────────────────────────────────────────────────────
        ics          = [c for c in comps if self._comp_class(c.ref) == "ic"]
        transistors  = [c for c in comps if self._comp_class(c.ref) == "transistor"]
        resistors    = [c for c in comps if self._comp_class(c.ref) == "resistor"]
        capacitors   = [c for c in comps if self._comp_class(c.ref) == "capacitor"]
        inductors    = [c for c in comps if self._comp_class(c.ref) == "inductor"]
        diodes       = [c for c in comps if self._comp_class(c.ref) == "diode"]
        connectors   = [c for c in comps if self._comp_class(c.ref) == "connector"]
        others       = [c for c in comps if self._comp_class(c.ref) == "other"]

        # Build net → component mapping to detect bypass caps
        net_to_comps: dict[str, list[str]] = {}
        if circuit.connections:
            for conn in circuit.connections:
                for pin_ref in (conn.pins or []):
                    cref = pin_ref.split(".")[0]
                    net_to_comps.setdefault(conn.net, []).append(cref)

        def shares_net_with_ic(cap_ref: str) -> str | None:
            """Return the IC ref that this capacitor connects to, if any."""
            ic_refs = {ic.ref for ic in ics}
            for net, crefs in net_to_comps.items():
                if cap_ref in crefs:
                    for cr in crefs:
                        if cr in ic_refs:
                            return cr
            return None

        bypass_caps:   dict[str, list] = {ic.ref: [] for ic in ics}
        signal_caps:   list = []
        for cap in capacitors:
            ic_ref = shares_net_with_ic(cap.ref)
            if ic_ref:
                bypass_caps[ic_ref].append(cap)
            else:
                signal_caps.append(cap)

        # ── 2. Place ICs ───────────────────────────────────────────────────
        # Sheet centre X; stack ICs vertically with generous spacing
        sheet_cx = DEFAULT_X_START + 76.2          # ~126 mm from left
        ic_y_cursor = DEFAULT_Y_START + 30.0

        ic_positions: dict[str, tuple[float, float]] = {}
        for ic in ics:
            _, half_h = self._ic_size(ic)
            ic.x = sheet_cx
            ic.y = ic_y_cursor
            ic_positions[ic.ref] = (ic.x, ic.y)
            # Place bypass caps for this IC directly below it
            bcaps = bypass_caps.get(ic.ref, [])
            for j, bcap in enumerate(bcaps):
                bcap.x = ic.x - PASSIVE_X_SPACING * (len(bcaps) - 1) / 2 + j * PASSIVE_X_SPACING
                bcap.y = ic.y + half_h + BYPASS_OFFSET_Y
            ic_y_cursor += IC_Y_SPACING + (len(bcaps) > 0) * (BYPASS_OFFSET_Y + 10.16)

        # ── 3. Place transistors ───────────────────────────────────────────
        trans_x = sheet_cx + IC_X_SPACING
        trans_y = DEFAULT_Y_START + 30.0
        for tr in transistors:
            tr.x = trans_x
            tr.y = trans_y
            trans_y += IC_Y_SPACING

        # ── 4. Place connectors ───────────────────────────────────────────
        # Input connectors on the left, output on the right
        left_x  = DEFAULT_X_START
        right_x = sheet_cx + IC_X_SPACING * 2 + 20.32
        left_y  = DEFAULT_Y_START + 20.32
        right_y = DEFAULT_Y_START + 20.32
        for idx, con in enumerate(connectors):
            if idx < len(connectors) // 2 + len(connectors) % 2:
                con.x = left_x
                con.y = left_y
                left_y += IC_Y_SPACING
            else:
                con.x = right_x
                con.y = right_y
                right_y += IC_Y_SPACING

        # ── 5. Place signal-path passives (R, L, signal C, D) ─────────────
        # Lay them out in rows below the ICs
        passive_list = resistors + inductors + signal_caps + diodes + others
        if passive_list:
            # Start below the IC block or bypass caps
            if ics:
                bottom_y = max(c.y for c in ics) + IC_Y_SPACING * 0.5 + 20.32
                if any(bypass_caps.values()):
                    bottom_y += BYPASS_OFFSET_Y + 15.24
            else:
                bottom_y = DEFAULT_Y_START + 40.64

            row_y    = bottom_y
            x_cursor = DEFAULT_X_START
            cols_per_row = max(3, min(8, len(passive_list)))
            for idx, comp in enumerate(passive_list):
                col = idx % cols_per_row
                row = idx // cols_per_row
                comp.x = DEFAULT_X_START + col * PASSIVE_X_SPACING * 2
                comp.y = bottom_y + row * PASSIVE_Y_SPACING

        # ── 6. Fallback for the completely-passive circuit case ────────────
        # If there are NO ics/transistors/connectors, use a clean 2-row grid
        placed_refs = (
            {c.ref for c in ics}
            | {c.ref for c in transistors}
            | {c.ref for c in connectors}
            | {c.ref for caps in bypass_caps.values() for c in caps}
            | {c.ref for c in resistors + inductors + signal_caps + diodes + others}
        )
        unplaced = [c for c in comps if c.ref not in placed_refs or (c.x == 0 and c.y == 0)]
        if unplaced:
            # Simple centred grid
            cols = max(3, min(6, len(unplaced)))
            sx = DEFAULT_X_START
            sy = DEFAULT_Y_START + 20.32
            for idx, comp in enumerate(unplaced):
                col = idx % cols
                row = idx // cols
                comp.x = sx + col * PASSIVE_X_SPACING * 2
                comp.y = sy + row * PASSIVE_Y_SPACING

        # Pure-passive fallback (no ICs at all) — horizontal chain layout
        if not ics and not transistors and not connectors:
            bypass_descs = ("bypass", "decoupling", "decouple", "filter", "power supply")
            bypass_c = [c for c in capacitors
                        if any(kw in (c.description or "").lower() for kw in bypass_descs)]
            signal_caps = [c for c in capacitors if c not in bypass_c]

            # Signal-path chain: R, L, signal-caps, diodes, others in a row
            chain = resistors + inductors + signal_caps + diodes + others
            CHAIN_SPACING = 30.48   # 1.2 inch between component centres
            chain_y = DEFAULT_Y_START + 40.64
            chain_start_x = DEFAULT_X_START + 25.4

            for idx, comp in enumerate(chain):
                comp.x = chain_start_x + idx * CHAIN_SPACING
                comp.y = chain_y
                # Rotate Rs, Cs, Ls to horizontal so pins face left/right
                cls = self._comp_class(comp.ref)
                if cls in ("resistor", "capacitor", "inductor"):
                    comp.rotation = 90
                else:
                    comp.rotation = 0

            # Bypass caps go below the chain, one per vertical slot
            bypass_y = chain_y + 30.48
            for idx, bcap in enumerate(bypass_c):
                bcap.x = chain_start_x + idx * CHAIN_SPACING
                bcap.y = bypass_y
                bcap.rotation = 0  # vertical: pin1 top (VCC), pin2 bottom (GND)

    # -------------------------------------------------------------------------
    # Power Net Detection
    # -------------------------------------------------------------------------

    def _detect_power_nets(self, connections: list[Connection]) -> set[str]:
        """Identify power nets (VCC, GND, +5V, +3V3, etc.)."""
        power_names = {"VCC", "GND", "VSS", "VDD", "VBUS", "V+", "V-",
                       "+5V", "+3V3", "+3.3V", "+12V", "+24V", "AVCC", "AGND",
                       "DVCC", "DGND"}
        power_nets = set()
        for conn in connections:
            name_upper = conn.net.upper()
            if name_upper in power_names or name_upper.startswith("+") or name_upper.startswith("-"):
                power_nets.add(conn.net)
        return power_nets

    # -------------------------------------------------------------------------
    # lib_symbols Section
    # -------------------------------------------------------------------------

    def _build_lib_symbols(self, unique_parts: dict[str, Component],
                           power_nets: set[str]) -> str:
        """Build the (lib_symbols ...) block."""
        entries = []

        for lib_id, comp in unique_parts.items():
            if lib_id in LIB_SYMBOL_GENERATORS:
                entries.append(LIB_SYMBOL_GENERATORS[lib_id](lib_id))
            else:
                # Generic rectangular symbol
                ref_prefix = "".join(c for c in comp.ref if c.isalpha())
                entries.append(_lib_symbol_generic(lib_id, ref_prefix, comp.part, comp.pins))

        # Power symbols
        for net in sorted(power_nets):
            if net.upper() in ("GND", "VSS", "AGND", "DGND"):
                entries.append(_lib_symbol_power(net, net, "1", 90))
            else:
                entries.append(_lib_symbol_power(net, net, "1", 270))

        return "  (lib_symbols\n" + "\n".join(entries) + "\n  )"

    # -------------------------------------------------------------------------
    # Symbol Instances (Component Placement)
    # -------------------------------------------------------------------------

    def _build_symbols(self, components: list[Component]) -> list[str]:
        """Build symbol placement blocks for all components."""
        symbols = []
        for comp in components:
            lib_id = f"{comp.lib}:{comp.part}"
            sym_uuid = _uuid()
            rot = comp.rotation

            # For horizontal components (rotation=90/270) place text above/below
            if rot in (90, 270):
                ref_x, ref_y = comp.x, comp.y - 3.81
                val_x, val_y = comp.x, comp.y + 3.81
            else:
                ref_x, ref_y = comp.x + 2.54, comp.y
                val_x, val_y = comp.x + 2.54, comp.y + 2.54

            lines = []
            lines.append(f'  (symbol (lib_id {_quote(lib_id)}) {_at(comp.x, comp.y, rot)} (unit 1)')
            lines.append(f'    (in_bom yes) (on_board yes) (dnp no)')
            lines.append(f'    (uuid {_quote(sym_uuid)})')

            # Properties: Reference
            lines.append(f'    (property "Reference" {_quote(comp.ref)} {_at(ref_x, ref_y, 0)}')
            lines.append(f'      {_effects(REF_FONT_SIZE)})')

            # Properties: Value
            lines.append(f'    (property "Value" {_quote(comp.value)} {_at(val_x, val_y, 0)}')
            lines.append(f'      {_effects(VALUE_FONT_SIZE)})')

            # Properties: Footprint
            if comp.footprint:
                lines.append(f'    (property "Footprint" {_quote(comp.footprint)} {_at(comp.x, comp.y, 0)}')
                lines.append(f'      {_effects(REF_FONT_SIZE, hide=True)})')

            # Properties: Datasheet
            lines.append(f'    (property "Datasheet" "~" {_at(comp.x, comp.y, 0)}')
            lines.append(f'      {_effects(REF_FONT_SIZE, hide=True)})')

            # Properties: Description
            if comp.description:
                lines.append(f'    (property "Description" {_quote(comp.description)} {_at(comp.x, comp.y, 0)}')
                lines.append(f'      {_effects(REF_FONT_SIZE, hide=True)})')

            # Pins (with UUIDs)
            if comp.pins:
                for pin in comp.pins:
                    lines.append(f'    (pin {_quote(pin.number)} (uuid {_quote(_uuid())}))')
            else:
                # Fallback: two pins for passives
                lines.append(f'    (pin "1" (uuid {_quote(_uuid())}))')
                lines.append(f'    (pin "2" (uuid {_quote(_uuid())}))')

            lines.append(f'  )')
            symbols.append("\n".join(lines))

        return symbols

    # -------------------------------------------------------------------------
    # Power Symbols
    # -------------------------------------------------------------------------

    def _build_power_symbols(self, connections: list[Connection],
                             comp_map: dict[str, Component],
                             power_nets: set[str]) -> list[str]:
        """Place one power symbol at EVERY pin of each power net.

        IMPORTANT: KiCad power symbols connect at their OWN ORIGIN (0,0).
        The symbol must therefore be placed at EXACTLY the pin coordinate—
        no offset. The graphical flag/bar is drawn by the symbol definition
        itself relative to the origin.
        """
        symbols = []
        placed_positions: set[tuple] = set()  # deduplicate exact overlaps

        for conn in connections:
            if conn.net not in power_nets:
                continue
            if not conn.pins:
                continue

            lib_id = f"power:{conn.net}"

            for pin_ref in conn.pins:
                px, py = self._resolve_pin_position(pin_ref, comp_map)
                if px is None:
                    continue

                # Snap to grid to deduplicate near-identical positions
                pos_key = (round(px, 1), round(py, 1))
                if pos_key in placed_positions:
                    continue
                placed_positions.add(pos_key)

                self._pwr_counter += 1
                sym_uuid = _uuid()

                # Place AT the pin position — the flag graphic is part of the
                # symbol definition and appears above/below automatically.
                lines = []
                lines.append(f'  (symbol (lib_id {_quote(lib_id)}) {_at(px, py, 0)} (unit 1)')
                lines.append(f'    (in_bom yes) (on_board yes) (dnp no)')
                lines.append(f'    (uuid {_quote(sym_uuid)})')
                lines.append(f'    (property "Reference" {_quote(f"#PWR0{self._pwr_counter:02d}")} {_at(px, py, 0)}')
                lines.append(f'      {_effects(REF_FONT_SIZE, hide=True)})')
                lines.append(f'    (property "Value" {_quote(conn.net)} {_at(px, py, 0)}')
                lines.append(f'      {_effects(VALUE_FONT_SIZE)})')
                lines.append(f'    (pin "1" (uuid {_quote(_uuid())}))')
                lines.append(f'  )')
                symbols.append("\n".join(lines))

        return symbols

    # -------------------------------------------------------------------------
    # Wire Routing
    # -------------------------------------------------------------------------

    def _build_wires(self, connections: list[Connection],
                     comp_map: dict[str, Component],
                     power_nets: set[str] | None = None) -> list[str]:
        """Build wire segments connecting pins within each net.
        Power nets are skipped because each pin gets its own power symbol.
        """
        wires = []
        power_nets = power_nets or set()

        for conn in connections:
            # Power nets use per-pin power symbols — no explicit wires needed
            if conn.net in power_nets:
                continue

            # Resolve all pin positions for this net
            pin_positions = []
            for pin_ref in conn.pins:
                px, py = self._resolve_pin_position(pin_ref, comp_map)
                if px is not None:
                    pin_positions.append((px, py))

            if len(pin_positions) < 2:
                continue

            # Connect pins sequentially with Manhattan routing
            for i in range(len(pin_positions) - 1):
                x1, y1 = pin_positions[i]
                x2, y2 = pin_positions[i + 1]

                if abs(x1 - x2) < 0.01 or abs(y1 - y2) < 0.01:
                    # Direct horizontal or vertical wire
                    wires.append(self._wire_segment(x1, y1, x2, y2))
                else:
                    # Manhattan routing: horizontal then vertical
                    mid_x = x2
                    mid_y = y1
                    wires.append(self._wire_segment(x1, y1, mid_x, mid_y))
                    wires.append(self._wire_segment(mid_x, mid_y, x2, y2))

        return wires

    def _wire_segment(self, x1: float, y1: float, x2: float, y2: float) -> str:
        """Generate a single wire segment."""
        return f"""  (wire (pts {_xy(x1, y1)} {_xy(x2, y2)})
    {_stroke(0)}
    (uuid {_quote(_uuid())})
  )"""

    # -------------------------------------------------------------------------
    # Net Labels
    # -------------------------------------------------------------------------

    def _build_labels(self, connections: list[Connection],
                      comp_map: dict[str, Component],
                      power_nets: set[str]) -> list[str]:
        """Place net labels only for multi-fan-out signal nets (3+ pins).

        Two-pin nets are fully described by their wire — no label needed.
        Labels at pin tips overlap component bodies; placing one label per
        net at a midpoint of the first wire segment is cleaner.
        """
        labels = []

        for conn in connections:
            # Skip power nets (they get power symbols instead)
            if conn.net in power_nets:
                continue
            # Skip two-pin nets — the wire is sufficient
            if not conn.pins or len(conn.pins) < 3:
                continue

            # Place label at midpoint of the wire between pin[0] and pin[1]
            px0, py0 = self._resolve_pin_position(conn.pins[0], comp_map)
            px1, py1 = self._resolve_pin_position(conn.pins[1], comp_map)
            if px0 is None or px1 is None:
                continue

            # Midpoint, snapped to grid (2.54mm)
            mid_x = round((px0 + px1) / 2 / 2.54) * 2.54
            mid_y = round((py0 + py1) / 2 / 2.54) * 2.54 - 2.54  # offset up slightly

            label_uuid = _uuid()
            labels.append(f"""  (label {_quote(conn.net)} {_at(mid_x, mid_y, 0)}
    {_effects(LABEL_FONT_SIZE)}
    (uuid {_quote(label_uuid)})
  )""")

        return labels

    # -------------------------------------------------------------------------
    # Pin Position Resolution
    # -------------------------------------------------------------------------

    def _resolve_pin_position(self, pin_ref: str,
                              comp_map: dict[str, Component]) -> tuple[Optional[float], Optional[float]]:
        """
        Resolve a pin reference (e.g. 'R1.1') to absolute (x, y) coordinates.

        Args:
            pin_ref: Pin reference in 'REF.PIN' format.
            comp_map: Component lookup by reference designator.

        Returns:
            (x, y) tuple or (None, None) if unresolvable.
        """
        parts = pin_ref.split(".")
        if len(parts) != 2:
            logger.warning(f"Invalid pin reference: {pin_ref}")
            return None, None

        ref, pin_num = parts
        comp = comp_map.get(ref)
        if comp is None:
            logger.warning(f"Component not found: {ref}")
            return None, None

        lib_id = f"{comp.lib}:{comp.part}"
        rot_rad = math.radians(comp.rotation)

        # Try known pin offsets first
        if lib_id in KNOWN_PIN_OFFSETS:
            offsets = KNOWN_PIN_OFFSETS[lib_id]
            if pin_num in offsets:
                dx, dy = offsets[pin_num]
                # Apply rotation
                rx = dx * math.cos(rot_rad) - dy * math.sin(rot_rad)
                ry = dx * math.sin(rot_rad) + dy * math.cos(rot_rad)
                return comp.x + rx, comp.y + ry

        # Try pin offsets from component data
        for pin in comp.pins:
            if pin.number == pin_num:
                dx, dy = pin.x, pin.y
                if dx == 0 and dy == 0:
                    # Assign default offsets for 2-pin passives
                    if pin_num == "1":
                        dx, dy = 0, 3.81
                    elif pin_num == "2":
                        dx, dy = 0, -3.81
                rx = dx * math.cos(rot_rad) - dy * math.sin(rot_rad)
                ry = dx * math.sin(rot_rad) + dy * math.cos(rot_rad)
                return comp.x + rx, comp.y + ry

        # Fallback: center of component
        logger.warning(f"Pin {pin_num} not found on {ref}, using component center")
        return comp.x, comp.y


# =============================================================================
# Public API
# =============================================================================

def export_to_kicad_sch(circuit: CircuitData) -> str:
    """
    Export a CircuitData object to a .kicad_sch string.

    Args:
        circuit: The circuit data to export.

    Returns:
        Complete .kicad_sch file content as a string.
    """
    writer = KiCadSchematicWriter()
    return writer.export(circuit)


def save_kicad_sch(circuit: CircuitData, filepath: str) -> str:
    """
    Export and save a CircuitData object to a .kicad_sch file.

    Args:
        circuit: The circuit data to export.
        filepath: Output file path.

    Returns:
        The filepath of the saved file.
    """
    content = export_to_kicad_sch(circuit)
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info(f"Saved schematic to {filepath}")
    return filepath


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    """Quick smoke test with the 555 timer template."""
    import json

    template_path = os.path.join(os.path.dirname(__file__), "..", "templates", "555_timer.json")

    if os.path.exists(template_path):
        with open(template_path, "r") as f:
            data = json.load(f)
        circuit = CircuitData(**data)
        output_path = os.path.join(os.path.dirname(__file__), "..", "output", "555_timer.kicad_sch")
        save_kicad_sch(circuit, output_path)
        print(f"✓ Exported to: {output_path}")

        # Print first 50 lines for inspection
        content = export_to_kicad_sch(circuit)
        for i, line in enumerate(content.split("\n")[:50]):
            print(f"  {i+1:3d} | {line}")
    else:
        print(f"Template not found: {template_path}")
