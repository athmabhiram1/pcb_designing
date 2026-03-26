"""
KiCad Schematic Exporter — Zero-dependency S-expression writer.

Converts CircuitData (Pydantic) → valid .kicad_sch file (KiCad 8.0+ compatible).
No external libraries required. Pure Python string generation.

KiCad lib_symbols sub-unit naming rule (v1.2):
  In a .kicad_sch lib_symbols section the PARENT uses the full qualified name
  but sub-unit names must use ONLY the part name (strip the "Library:" prefix).
  e.g. parent: (symbol "Device:C" ...)
       sub-units MUST be: (symbol "C_0_1" ...)   ← NO "Device:" prefix
                          (symbol "C_1_1" ...)
  Using the full lib_id (e.g. "Device:C_0_1") causes the KiCad error:
    "Invalid symbol unit name prefix Device:C_0_1 ... line N, offset 16"

  All _lib_symbol_* functions use `{lib_id.split(":")[-1]}_0_1` to produce
  the correct part-name-only sub-unit prefix.

  Affected functions (all fixed):
    _lib_symbol_resistor, _lib_symbol_capacitor, _lib_symbol_led,
    _lib_symbol_ne555, _lib_symbol_inductor, _lib_symbol_diode,
    _lib_symbol_regulator, _lib_symbol_opamp, _lib_symbol_nmos,
    _lib_symbol_power, _lib_symbol_generic
"""
import uuid
import math
import logging
import os
import re
from typing import Optional
from dataclasses import dataclass, field

# Import from parent package — use relative import when running as part of
# the ai_backend package (normal case) and fall back to path manipulation
# only when executed directly as a standalone script.
try:
    from ..circuit_schema import CircuitData, Component, Connection, Pin
except ImportError:
    import sys as _sys
    import os as _os
    _parent = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
    if _parent not in _sys.path:
        _sys.path.insert(0, _parent)
    from circuit_schema import CircuitData, Component, Connection, Pin

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

KICAD_VERSION = 20231120
GENERATOR = "kicad_copilot"
GENERATOR_VERSION = "1.3"   # v1.3: D_Zener/D_Schottky symbols, junction generation, sys.path fix
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
#
# CRITICAL RULE: sub-symbol names MUST use only the part name (after the ":")
#   as their prefix — NOT the full lib_id with the library namespace.
#   Correct:   (symbol "R_0_1" ...)
#   Wrong:     (symbol "Device:R_0_1" ...)  ← causes KiCad "Invalid symbol unit
#                                               name prefix" error on load

def _lib_symbol_resistor(lib_id: str = "Device:R") -> str:
    """Generate lib_symbol entry for a resistor."""
    return f"""    (symbol {_quote(lib_id)}
      (pin_numbers hide)
      (pin_names (offset 0) hide)
      (in_bom yes) (on_board yes)
      (property "Reference" "R" {_at(2.032, 0, 90)} {_effects(REF_FONT_SIZE)})
      (property "Value" "R" {_at(-1.778, 0, 90)} {_effects(VALUE_FONT_SIZE)})
      (property "Footprint" "" {_at(0, 0, 0)} {_effects(REF_FONT_SIZE, hide=True)})
      (property "Datasheet" "~" {_at(0, 0, 0)} {_effects(REF_FONT_SIZE, hide=True)})
      (symbol "{lib_id.split(":")[-1]}_0_1"
        (rectangle (start -1.016 -2.54) (end 1.016 2.54)
          {_stroke(0.254)}
          (fill (type none))
        )
      )
      (symbol "{lib_id.split(":")[-1]}_1_1"
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
    return f"""    (symbol {_quote(lib_id)}
      (pin_numbers hide)
      (pin_names (offset 0.254) hide)
      (in_bom yes) (on_board yes)
      (property "Reference" "C" {_at(1.524, 0, 90)} {_effects(REF_FONT_SIZE)})
      (property "Value" "C" {_at(-1.524, 0, 90)} {_effects(VALUE_FONT_SIZE)})
      (property "Footprint" "" {_at(0, 0, 0)} {_effects(REF_FONT_SIZE, hide=True)})
      (property "Datasheet" "~" {_at(0, 0, 0)} {_effects(REF_FONT_SIZE, hide=True)})
      (symbol "{lib_id.split(":")[-1]}_0_1"
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
      (symbol "{lib_id.split(":")[-1]}_1_1"
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
    return f"""    (symbol {_quote(lib_id)}
      (pin_numbers hide)
      (pin_names (offset 1.016) hide)
      (in_bom yes) (on_board yes)
      (property "Reference" "D" {_at(1.524, 1.27, 0)} {_effects(REF_FONT_SIZE)})
      (property "Value" "LED" {_at(1.524, -1.27, 0)} {_effects(VALUE_FONT_SIZE)})
      (property "Footprint" "" {_at(0, 0, 0)} {_effects(REF_FONT_SIZE, hide=True)})
      (property "Datasheet" "~" {_at(0, 0, 0)} {_effects(REF_FONT_SIZE, hide=True)})
      (symbol "{lib_id.split(":")[-1]}_0_1"
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
      (symbol "{lib_id.split(":")[-1]}_1_1"
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
    return f"""    (symbol {_quote(lib_id)}
      (pin_names (offset 1.016))
      (in_bom yes) (on_board yes)
      (property "Reference" "U" {_at(8.89, 8.89, 0)} {_effects(REF_FONT_SIZE)})
      (property "Value" "NE555" {_at(8.89, -8.89, 0)} {_effects(VALUE_FONT_SIZE)})
      (property "Footprint" "" {_at(0, 0, 0)} {_effects(REF_FONT_SIZE, hide=True)})
      (property "Datasheet" "http://www.ti.com/lit/ds/symlink/ne555.pdf" {_at(0, 0, 0)} {_effects(REF_FONT_SIZE, hide=True)})
      (symbol "{lib_id.split(":")[-1]}_0_1"
        (rectangle (start -7.62 7.62) (end 7.62 -7.62)
          {_stroke(0.254)}
          (fill (type background))
        )
      )
      (symbol "{lib_id.split(":")[-1]}_1_1"
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
    return f"""    (symbol {_quote(lib_id)}
      (pin_numbers hide)
      (pin_names (offset 1.016) hide)
      (in_bom yes) (on_board yes)
      (property "Reference" "L" {_at(1.27, 0, 90)} {_effects(REF_FONT_SIZE)})
      (property "Value" "L" {_at(-1.016, 0, 90)} {_effects(VALUE_FONT_SIZE)})
      (property "Footprint" "" {_at(0, 0, 0)} {_effects(REF_FONT_SIZE, hide=True)})
      (property "Datasheet" "~" {_at(0, 0, 0)} {_effects(REF_FONT_SIZE, hide=True)})
      (symbol "{lib_id.split(":")[-1]}_0_1"
        (arc (start 0 -2.54) (mid 0.6323 -1.905) (end 0 -1.27)
          {_stroke(0)} (fill (type none)))
        (arc (start 0 -1.27) (mid 0.6323 -0.635) (end 0 0)
          {_stroke(0)} (fill (type none)))
        (arc (start 0 0) (mid 0.6323 0.635) (end 0 1.27)
          {_stroke(0)} (fill (type none)))
        (arc (start 0 1.27) (mid 0.6323 1.905) (end 0 2.54)
          {_stroke(0)} (fill (type none)))
      )
      (symbol "{lib_id.split(":")[-1]}_1_1"
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
    return f"""    (symbol {_quote(lib_id)}
      (pin_numbers hide)
      (pin_names (offset 1.016) hide)
      (in_bom yes) (on_board yes)
      (property "Reference" "D" {_at(0, 2.54, 0)} {_effects(REF_FONT_SIZE)})
      (property "Value" "D" {_at(0, -2.54, 0)} {_effects(VALUE_FONT_SIZE)})
      (property "Footprint" "" {_at(0, 0, 0)} {_effects(REF_FONT_SIZE, hide=True)})
      (property "Datasheet" "~" {_at(0, 0, 0)} {_effects(REF_FONT_SIZE, hide=True)})
      (symbol "{lib_id.split(":")[-1]}_0_1"
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
      (symbol "{lib_id.split(":")[-1]}_1_1"
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
    part_value = lib_id.split(":")[-1]   # e.g. "AMS1117-3.3"
    return f"""    (symbol {_quote(lib_id)}
      (pin_names (offset 1.016))
      (in_bom yes) (on_board yes)
      (property "Reference" "U" {_at(0, 7.62, 0)} {_effects(REF_FONT_SIZE)})
      (property "Value" "{part_value}" {_at(0, -7.62, 0)} {_effects(VALUE_FONT_SIZE)})
      (property "Footprint" "" {_at(0, 0, 0)} {_effects(REF_FONT_SIZE, hide=True)})
      (property "Datasheet" "~" {_at(0, 0, 0)} {_effects(REF_FONT_SIZE, hide=True)})
      (symbol "{lib_id.split(":")[-1]}_0_1"
        (rectangle (start -5.08 5.08) (end 5.08 -5.08)
          {_stroke(0.254)}
          (fill (type background))
        )
      )
      (symbol "{lib_id.split(":")[-1]}_1_1"
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
    return f"""    (symbol {_quote(lib_id)}
      (pin_names (offset 1.016))
      (in_bom yes) (on_board yes)
      (property "Reference" "U" {_at(0, 7.62, 0)} {_effects(REF_FONT_SIZE)})
      (property "Value" "LM358" {_at(0, -7.62, 0)} {_effects(VALUE_FONT_SIZE)})
      (property "Footprint" "" {_at(0, 0, 0)} {_effects(REF_FONT_SIZE, hide=True)})
      (property "Datasheet" "~" {_at(0, 0, 0)} {_effects(REF_FONT_SIZE, hide=True)})
      (symbol "{lib_id.split(":")[-1]}_0_1"
        (polyline
          (pts {_xy(-5.08, 5.08)} {_xy(-5.08, -5.08)} {_xy(5.08, 0)} {_xy(-5.08, 5.08)})
          {_stroke(0.254)}
          (fill (type background))
        )
      )
      (symbol "{lib_id.split(":")[-1]}_1_1"
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
    return f"""    (symbol {_quote(lib_id)}
      (pin_names (offset 1.016))
      (in_bom yes) (on_board yes)
      (property "Reference" "Q" {_at(5.08, 1.905, 0)} {_effects(REF_FONT_SIZE)})
      (property "Value" "Q_NMOS_GSD" {_at(5.08, -1.905, 0)} {_effects(VALUE_FONT_SIZE)})
      (property "Footprint" "" {_at(0, 0, 0)} {_effects(REF_FONT_SIZE, hide=True)})
      (property "Datasheet" "~" {_at(0, 0, 0)} {_effects(REF_FONT_SIZE, hide=True)})
      (symbol "{lib_id.split(":")[-1]}_0_1"
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
      (symbol "{lib_id.split(":")[-1]}_1_1"
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
    direction = int(direction)   # guard: ensure it is never accidentally a set literal
    lib_id = f"power:{symbol_name}"
    # NOTE: sub-symbol name uses the full lib_id  e.g. "power:GND_0_1"
    return f"""    (symbol {_quote(lib_id)}
      (power)
      (pin_numbers hide)
      (pin_names (offset 0) hide)
      (in_bom yes) (on_board yes)
      (property "Reference" "#PWR" {_at(0, 2.54, 0)} {_effects(REF_FONT_SIZE, hide=True)})
      (property "Value" "{symbol_name}" {_at(0, 3.81, 0)} {_effects(VALUE_FONT_SIZE)})
      (property "Footprint" "" {_at(0, 0, 0)} {_effects(REF_FONT_SIZE, hide=True)})
      (property "Datasheet" "" {_at(0, 0, 0)} {_effects(REF_FONT_SIZE, hide=True)})
      (symbol "{lib_id.split(":")[-1]}_0_1"
        (pin power_in line {_at(0, 0, direction)} (length 0)
          (name "{pin_name}" {_effects(REF_FONT_SIZE)})
          (number "{pin_number}" {_effects(REF_FONT_SIZE)})
        )
      )
    )"""


def _lib_symbol_generic(lib_id: str, ref_prefix: str, part_name: str,
                        pins: list) -> str:
    """Generate a generic rectangular lib_symbol for unknown components."""
    # Safe internal uid used ONLY for display value — NOT for sub-symbol naming
    n_pins = len(pins)
    left_count = (n_pins + 1) // 2
    right_count = n_pins - left_count

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

    # Body — sub-symbol name MUST use full lib_id as prefix
    lines.append(f'      (symbol "{lib_id.split(":")[-1]}_0_1"')
    lines.append(f'        (rectangle (start -{box_half_w} {box_half_h}) (end {box_half_w} -{box_half_h})')
    lines.append(f'          {_stroke(0.254)}')
    lines.append(f'          (fill (type background))')
    lines.append(f'        )')
    lines.append(f'      )')

    # Pins — sub-symbol name MUST use full lib_id as prefix
    lines.append(f'      (symbol "{lib_id.split(":")[-1]}_1_1"')
    left_pins  = pins[:left_count]
    right_pins = pins[left_count:]

    for i, pin in enumerate(left_pins):
        py    = box_half_h - 2.54 - i * 2.54
        pname = pin.name if pin.name else "~"
        lines.append(f'        (pin passive line {_at(-box_half_w - PIN_LENGTH, py, 0)} (length {PIN_LENGTH})')
        lines.append(f'          (name "{pname}" {_effects(REF_FONT_SIZE)})')
        lines.append(f'          (number "{pin.number}" {_effects(REF_FONT_SIZE)})')
        lines.append(f'        )')

    for i, pin in enumerate(right_pins):
        py    = box_half_h - 2.54 - i * 2.54
        pname = pin.name if pin.name else "~"
        lines.append(f'        (pin passive line {_at(box_half_w + PIN_LENGTH, py, 180)} (length {PIN_LENGTH})')
        lines.append(f'          (name "{pname}" {_effects(REF_FONT_SIZE)})')
        lines.append(f'          (number "{pin.number}" {_effects(REF_FONT_SIZE)})')
        lines.append(f'        )')

    lines.append(f'      )')
    lines.append(f'    )')
    return "\n".join(lines)


# Registry mapping lib:part → generator function
LIB_SYMBOL_GENERATORS = {
    "Device:R":                     _lib_symbol_resistor,
    "Device:C":                     _lib_symbol_capacitor,
    "Device:LED":                   _lib_symbol_led,
    "Device:L":                     _lib_symbol_inductor,
    "Device:D":                     _lib_symbol_diode,
    "Device:D_Zener":               lambda lib_id: _lib_symbol_diode(lib_id),
    "Device:D_Schottky":            lambda lib_id: _lib_symbol_diode(lib_id),
    "Timer:NE555":                  _lib_symbol_ne555,
    "Regulator_Linear:AMS1117-3.3": _lib_symbol_regulator,
    "Regulator_Linear:LM7805":      lambda lib_id: _lib_symbol_regulator(lib_id),
    "Amplifier_Operational:LM358":  _lib_symbol_opamp,
    "Device:Q_NMOS_GSD":            _lib_symbol_nmos,
    "Device:Q_NMOS_DGS":            _lib_symbol_nmos,
}


# =============================================================================
# Pin Position Lookup for Known Symbols
# =============================================================================

KNOWN_PIN_OFFSETS: dict = {
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
        "1": (7.62, 0),      # OUT
        "2": (-7.62, 2.54),  # IN-
        "3": (-7.62, -2.54), # IN+
        "4": (0, -7.62),     # GND
        "8": (0, 7.62),      # VCC
    },
    "Device:Q_NMOS_GSD": {
        "1": (2.54, -5.08),  # S (Source)
        "2": (-2.54, 0),     # G (Gate)
        "3": (2.54, 5.08),   # D (Drain)
    },
    "Device:Q_NMOS_DGS": {
        "1": (2.54, 5.08),   # D (Drain)
        "2": (-2.54, 0),     # G (Gate)
        "3": (2.54, -5.08),  # S (Source)
    },
    "Device:D_Zener": {
        "1": (-3.81, 0),   # Anode
        "2": (3.81, 0),    # Cathode
    },
    "Device:D_Schottky": {
        "1": (-3.81, 0),   # Anode
        "2": (3.81, 0),    # Cathode
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
        """Full pipeline: CircuitData → .kicad_sch string."""
        self._pwr_counter = 0

        self._normalize_input_coordinates(circuit)
        self._auto_place(circuit)

        comp_map = {c.ref: c for c in circuit.components}

        unique_parts = {}
        for c in circuit.components:
            lib_id = f"{c.lib}:{c.part}"
            if lib_id not in unique_parts:
                unique_parts[lib_id] = c

        power_nets = self._detect_power_nets(circuit.connections)

        header       = self._build_header()
        lib_symbols  = self._build_lib_symbols(unique_parts, power_nets)
        symbols      = self._build_symbols(circuit.components)
        power_syms   = self._build_power_symbols(circuit.connections, comp_map, power_nets)
        wires, junctions = self._build_wires(circuit.connections, comp_map, power_nets)
        labels       = self._build_labels(circuit.connections, comp_map, power_nets)
        footer       = self._build_footer()

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
            *junctions,
            "",
            *labels,
            "",
            footer,
            ")",
        ]

        return "\n".join(s for s in sections if s is not None) + "\n"

    def _normalize_input_coordinates(self, circuit: CircuitData) -> None:
        """Map board-style coordinates into a readable schematic-page area.

        Many generated circuits provide component positions in board space
        (0..board_width, 0..board_height). KiCad schematic sheets are not board
        coordinates, so those values render cramped in the top-left corner.
        """
        if not circuit.components:
            return

        # Leave pure auto-placement cases untouched.
        if all(c.x == 0.0 and c.y == 0.0 for c in circuit.components):
            return

        xs = [c.x for c in circuit.components]
        ys = [c.y for c in circuit.components]
        bw = float(getattr(circuit, "board_width", 0.0) or 0.0)
        bh = float(getattr(circuit, "board_height", 0.0) or 0.0)

        looks_like_board_space = (
            bw > 0.0 and bh > 0.0
            and min(xs) >= -1.0 and min(ys) >= -1.0
            and max(xs) <= bw + 1.0 and max(ys) <= bh + 1.0
        )

        looks_like_centered_space = (
            bw > 0.0 and bh > 0.0
            and min(xs) >= -(bw / 2.0) - 2.0 and max(xs) <= (bw / 2.0) + 2.0
            and min(ys) >= -(bh / 2.0) - 2.0 and max(ys) <= (bh / 2.0) + 2.0
        )

        if looks_like_board_space or looks_like_centered_space:
            if looks_like_centered_space and not looks_like_board_space:
                src_points = [(c.x + bw / 2.0, c.y + bh / 2.0) for c in circuit.components]
            else:
                src_points = [(c.x, c.y) for c in circuit.components]

            src_xs = [p[0] for p in src_points]
            src_ys = [p[1] for p in src_points]
            min_x, max_x = min(src_xs), max(src_xs)
            min_y, max_y = min(src_ys), max(src_ys)
            span_x = max(0.1, max_x - min_x)
            span_y = max(0.1, max_y - min_y)

            # Fit and center into a comfortable A4 schematic working area.
            target_cx = 140.0
            target_cy = 95.0
            src_cx = (min_x + max_x) / 2.0
            src_cy = (min_y + max_y) / 2.0

            # Keep the cluster readable: shrink large layouts and also enlarge
            # very small ones so symbols/labels do not overlap visually.
            max_target_w = 170.0
            max_target_h = 110.0
            min_target_w = 120.0
            min_target_h = 75.0

            fit_scale = min(max_target_w / span_x, max_target_h / span_y)
            fill_scale = min(min_target_w / span_x, min_target_h / span_y)
            scale = max(fill_scale, min(fit_scale, 3.0))

            for comp, (sx, sy) in zip(circuit.components, src_points):
                tx = target_cx + (sx - src_cx) * scale
                ty = target_cy + (sy - src_cy) * scale
                # Keep a safe margin from sheet borders.
                comp.x = max(25.0, min(255.0, tx))
                comp.y = max(20.0, min(175.0, ty))

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
    # Auto-placement
    # -------------------------------------------------------------------------

    @staticmethod
    def _comp_class(ref: str) -> str:
        prefix = "".join(c for c in ref if c.isalpha()).upper()
        if prefix in ("U", "IC"):           return "ic"
        if prefix in ("Q", "M", "T"):       return "transistor"
        if prefix in ("J", "P", "CN", "X", "SW", "BTN"): return "connector"
        if prefix in ("R",):                return "resistor"
        if prefix in ("C",):                return "capacitor"
        if prefix in ("L",):                return "inductor"
        if prefix in ("D", "LED", "Z"):     return "diode"
        return "other"

    @staticmethod
    def _ic_size(comp) -> tuple:
        n_pins = len(comp.pins) if comp.pins else 2
        half_h = max(7.62, (n_pins / 2) * 2.54 + 2.54)
        half_w = 7.62
        return half_w, half_h

    def _auto_place(self, circuit: CircuitData):
        needs_placement = all(c.x == 0.0 and c.y == 0.0 for c in circuit.components)
        if not needs_placement:
            return

        comps = circuit.components
        if not comps:
            return

        ics         = [c for c in comps if self._comp_class(c.ref) == "ic"]
        transistors = [c for c in comps if self._comp_class(c.ref) == "transistor"]
        resistors   = [c for c in comps if self._comp_class(c.ref) == "resistor"]
        capacitors  = [c for c in comps if self._comp_class(c.ref) == "capacitor"]
        inductors   = [c for c in comps if self._comp_class(c.ref) == "inductor"]
        diodes      = [c for c in comps if self._comp_class(c.ref) == "diode"]
        connectors  = [c for c in comps if self._comp_class(c.ref) == "connector"]
        others      = [c for c in comps if self._comp_class(c.ref) == "other"]

        net_to_comps: dict = {}
        if circuit.connections:
            for conn in circuit.connections:
                for pin_ref in (conn.pins or []):
                    cref = pin_ref.split(".")[0]
                    net_to_comps.setdefault(conn.net, []).append(cref)

        def shares_net_with_ic(cap_ref: str) -> Optional[str]:
            ic_refs = {ic.ref for ic in ics}
            for net, crefs in net_to_comps.items():
                if cap_ref in crefs:
                    for cr in crefs:
                        if cr in ic_refs:
                            return cr
            return None

        bypass_caps: dict = {ic.ref: [] for ic in ics}
        signal_caps: list = []
        for cap in capacitors:
            ic_ref = shares_net_with_ic(cap.ref)
            if ic_ref:
                bypass_caps[ic_ref].append(cap)
            else:
                signal_caps.append(cap)

        sheet_cx    = DEFAULT_X_START + 76.2
        ic_y_cursor = DEFAULT_Y_START + 30.0

        ic_positions: dict = {}
        for ic in ics:
            _, half_h = self._ic_size(ic)
            ic.x = sheet_cx
            ic.y = ic_y_cursor
            ic_positions[ic.ref] = (ic.x, ic.y)
            bcaps = bypass_caps.get(ic.ref, [])
            for j, bcap in enumerate(bcaps):
                bcap.x = ic.x - PASSIVE_X_SPACING * (len(bcaps) - 1) / 2 + j * PASSIVE_X_SPACING
                bcap.y = ic.y + half_h + BYPASS_OFFSET_Y
            ic_y_cursor += IC_Y_SPACING + (len(bcaps) > 0) * (BYPASS_OFFSET_Y + 10.16)

        trans_x = sheet_cx + IC_X_SPACING
        trans_y = DEFAULT_Y_START + 30.0
        for tr in transistors:
            tr.x = trans_x
            tr.y = trans_y
            trans_y += IC_Y_SPACING

        left_x  = DEFAULT_X_START
        right_x = sheet_cx + IC_X_SPACING * 2 + 20.32
        left_y  = DEFAULT_Y_START + 20.32
        right_y = DEFAULT_Y_START + 20.32
        for idx, con in enumerate(connectors):
            if idx < len(connectors) // 2 + len(connectors) % 2:
                con.x = left_x; con.y = left_y; left_y += IC_Y_SPACING
            else:
                con.x = right_x; con.y = right_y; right_y += IC_Y_SPACING

        passive_list = resistors + inductors + signal_caps + diodes + others
        if passive_list:
            if ics:
                bottom_y = max(c.y for c in ics) + IC_Y_SPACING * 0.5 + 20.32
                if any(bypass_caps.values()):
                    bottom_y += BYPASS_OFFSET_Y + 15.24
            else:
                bottom_y = DEFAULT_Y_START + 40.64
            cols_per_row = max(3, min(8, len(passive_list)))
            for idx, comp in enumerate(passive_list):
                col = idx % cols_per_row
                row = idx // cols_per_row
                comp.x = DEFAULT_X_START + col * PASSIVE_X_SPACING * 2
                comp.y = bottom_y + row * PASSIVE_Y_SPACING

        placed_refs = (
            {c.ref for c in ics}
            | {c.ref for c in transistors}
            | {c.ref for c in connectors}
            | {c.ref for caps in bypass_caps.values() for c in caps}
            | {c.ref for c in resistors + inductors + signal_caps + diodes + others}
        )
        unplaced = [c for c in comps if c.ref not in placed_refs or (c.x == 0 and c.y == 0)]
        if unplaced:
            cols = max(3, min(6, len(unplaced)))
            for idx, comp in enumerate(unplaced):
                col = idx % cols
                row = idx // cols
                comp.x = DEFAULT_X_START + col * PASSIVE_X_SPACING * 2
                comp.y = DEFAULT_Y_START + 20.32 + row * PASSIVE_Y_SPACING

        if not ics and not transistors and not connectors:
            bypass_descs = ("bypass", "decoupling", "decouple", "filter", "power supply")
            bypass_c = [c for c in capacitors
                        if any(kw in (getattr(c, "description", "") or "").lower()
                               for kw in bypass_descs)]
            signal_caps_fb = [c for c in capacitors if c not in bypass_c]
            chain = resistors + inductors + signal_caps_fb + diodes + others
            CHAIN_SPACING = 30.48
            chain_y = DEFAULT_Y_START + 40.64
            chain_start_x = DEFAULT_X_START + 25.4
            for idx, comp in enumerate(chain):
                comp.x = chain_start_x + idx * CHAIN_SPACING
                comp.y = chain_y
                cls = self._comp_class(comp.ref)
                comp.rotation = 90 if cls in ("resistor", "capacitor", "inductor") else 0
            bypass_y = chain_y + 30.48
            for idx, bcap in enumerate(bypass_c):
                bcap.x = chain_start_x + idx * CHAIN_SPACING
                bcap.y = bypass_y
                bcap.rotation = 0

    # -------------------------------------------------------------------------
    # Power Net Detection
    # -------------------------------------------------------------------------

    def _detect_power_nets(self, connections: list) -> set:
        power_names = {"VCC", "GND", "VSS", "VDD", "VBUS", "V+", "V-",
                       "+5V", "+3V3", "+3.3V", "+12V", "+24V", "AVCC", "AGND",
                       "DVCC", "DGND"}
        power_nets = set()
        for conn in connections:
            name_upper = conn.net.upper()
            if (name_upper in power_names
                    or name_upper.startswith("+")
                    or name_upper.startswith("-")):
                power_nets.add(conn.net)
        return power_nets

    # -------------------------------------------------------------------------
    # lib_symbols Section
    # -------------------------------------------------------------------------

    def _build_lib_symbols(self, unique_parts: dict, power_nets: set) -> str:
        entries = []

        for lib_id, comp in unique_parts.items():
            if lib_id in LIB_SYMBOL_GENERATORS:
                entries.append(LIB_SYMBOL_GENERATORS[lib_id](lib_id))
            else:
                ref_prefix = "".join(c for c in comp.ref if c.isalpha())
                entries.append(_lib_symbol_generic(lib_id, ref_prefix, comp.part, comp.pins))

        for net in sorted(power_nets):
            if net.upper() in ("GND", "VSS", "AGND", "DGND"):
                entries.append(_lib_symbol_power(net, net, "1", 90))
            else:
                entries.append(_lib_symbol_power(net, net, "1", 270))

        return "  (lib_symbols\n" + "\n".join(entries) + "\n  )"

    # -------------------------------------------------------------------------
    # Symbol Instances
    # -------------------------------------------------------------------------

    def _build_symbols(self, components: list) -> list:
        symbols = []
        for comp in components:
            lib_id   = f"{comp.lib}:{comp.part}"
            sym_uuid = _uuid()
            rot      = comp.rotation

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
            lines.append(f'    (property "Reference" {_quote(comp.ref)} {_at(ref_x, ref_y, 0)}')
            lines.append(f'      {_effects(REF_FONT_SIZE)})')
            lines.append(f'    (property "Value" {_quote(comp.value)} {_at(val_x, val_y, 0)}')
            lines.append(f'      {_effects(VALUE_FONT_SIZE)})')
            if comp.footprint:
                lines.append(f'    (property "Footprint" {_quote(comp.footprint)} {_at(comp.x, comp.y, 0)}')
                lines.append(f'      {_effects(REF_FONT_SIZE, hide=True)})')
            lines.append(f'    (property "Datasheet" "~" {_at(comp.x, comp.y, 0)}')
            lines.append(f'      {_effects(REF_FONT_SIZE, hide=True)})')
            if comp.description:
                lines.append(f'    (property "Description" {_quote(comp.description)} {_at(comp.x, comp.y, 0)}')
                lines.append(f'      {_effects(REF_FONT_SIZE, hide=True)})')
            if comp.pins:
                for pin in comp.pins:
                    lines.append(f'    (pin {_quote(pin.number)} (uuid {_quote(_uuid())}))')
            else:
                lines.append(f'    (pin "1" (uuid {_quote(_uuid())}))')
                lines.append(f'    (pin "2" (uuid {_quote(_uuid())}))')
            lines.append(f'  )')
            symbols.append("\n".join(lines))

        return symbols

    # -------------------------------------------------------------------------
    # Power Symbols
    # -------------------------------------------------------------------------

    def _build_power_symbols(self, connections: list, comp_map: dict,
                             power_nets: set) -> list:
        symbols = []
        placed_positions: set = set()

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

                pos_key = (round(px, 1), round(py, 1))
                if pos_key in placed_positions:
                    continue
                placed_positions.add(pos_key)

                self._pwr_counter += 1
                sym_uuid = _uuid()

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

    def _build_wires(self, connections: list, comp_map: dict,
                     power_nets: Optional[set] = None) -> tuple:
        """Build wire segments and junction dots.

        Returns (wires: list[str], junctions: list[str]).
        """
        wires = []
        junctions = []
        junction_pts: set = set()   # de-dup on rounded coords
        power_nets = power_nets or set()

        for conn in connections:
            pin_positions = []
            for pin_ref in conn.pins:
                px, py = self._resolve_pin_position(pin_ref, comp_map)
                if px is not None:
                    pin_positions.append((px, py))

            if len(pin_positions) < 2:
                continue

            # For power nets, create explicit star routing from the first pin.
            if conn.net in power_nets:
                ax, ay = pin_positions[0]
                for x2, y2 in pin_positions[1:]:
                    if abs(ax - x2) < 0.01 or abs(ay - y2) < 0.01:
                        wires.append(self._wire_segment(ax, ay, x2, y2))
                    else:
                        wires.append(self._wire_segment(ax, ay, x2, ay))
                        wires.append(self._wire_segment(x2, ay, x2, y2))
                # Junction at hub if 3+ pins converge
                if len(pin_positions) >= 3:
                    jkey = (round(ax, 2), round(ay, 2))
                    if jkey not in junction_pts:
                        junction_pts.add(jkey)
                        junctions.append(self._junction(ax, ay))
                continue

            for i in range(len(pin_positions) - 1):
                x1, y1 = pin_positions[i]
                x2, y2 = pin_positions[i + 1]

                if abs(x1 - x2) < 0.01 or abs(y1 - y2) < 0.01:
                    wires.append(self._wire_segment(x1, y1, x2, y2))
                else:
                    mid_x, mid_y = x2, y1
                    wires.append(self._wire_segment(x1, y1, mid_x, mid_y))
                    wires.append(self._wire_segment(mid_x, mid_y, x2, y2))
                    # Junction at L-bend mid-point if chained
                    if i > 0:
                        jkey = (round(x1, 2), round(y1, 2))
                        if jkey not in junction_pts:
                            junction_pts.add(jkey)
                            junctions.append(self._junction(x1, y1))

        return wires, junctions

    def _wire_segment(self, x1: float, y1: float, x2: float, y2: float) -> str:
        return f"""  (wire (pts {_xy(x1, y1)} {_xy(x2, y2)})
    {_stroke(0)}
    (uuid {_quote(_uuid())})
  )"""

    @staticmethod
    def _junction(x: float, y: float) -> str:
        """Generate a junction dot at a wire T-intersection."""
        return f"""  (junction (at {x:.4f} {y:.4f}) (diameter 0) (color 0 0 0 0)
    (uuid {_quote(_uuid())})
  )"""

    # -------------------------------------------------------------------------
    # Net Labels
    # -------------------------------------------------------------------------

    def _build_labels(self, connections: list, comp_map: dict,
                      power_nets: set) -> list:
        labels = []

        for conn in connections:
            if conn.net in power_nets:
                continue
            if not conn.pins or len(conn.pins) < 3:
                continue

            px0, py0 = self._resolve_pin_position(conn.pins[0], comp_map)
            px1, py1 = self._resolve_pin_position(conn.pins[1], comp_map)
            if px0 is None or px1 is None:
                continue

            mid_x = round((px0 + px1) / 2 / 2.54) * 2.54
            mid_y = round((py0 + py1) / 2 / 2.54) * 2.54 - 2.54

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
                              comp_map: dict) -> tuple:
        parts = pin_ref.split(".")
        if len(parts) != 2:
            logger.warning(f"Invalid pin reference: {pin_ref}")
            return None, None

        ref, pin_num = parts
        comp = comp_map.get(ref)
        if comp is None:
            logger.warning(f"Component not found: {ref}")
            return None, None

        lib_id  = f"{comp.lib}:{comp.part}"
        rot_rad = math.radians(comp.rotation)

        if lib_id in KNOWN_PIN_OFFSETS:
            offsets = KNOWN_PIN_OFFSETS[lib_id]
            if pin_num in offsets:
                dx, dy = offsets[pin_num]
                rx = dx * math.cos(rot_rad) - dy * math.sin(rot_rad)
                ry = dx * math.sin(rot_rad) + dy * math.cos(rot_rad)
                return comp.x + rx, comp.y + ry

        for pin in comp.pins:
            if pin.number == pin_num:
                dx = getattr(pin, 'x', 0) or getattr(getattr(pin, 'position', None), 'x', 0)
                dy = getattr(pin, 'y', 0) or getattr(getattr(pin, 'position', None), 'y', 0)
                if dx == 0 and dy == 0:
                    dx, dy = (0, 3.81) if pin_num == "1" else (0, -3.81)
                rx = dx * math.cos(rot_rad) - dy * math.sin(rot_rad)
                ry = dx * math.sin(rot_rad) + dy * math.cos(rot_rad)
                return comp.x + rx, comp.y + ry

        logger.warning(f"Pin {pin_num} not found on {ref}, using component center")
        return comp.x, comp.y


# =============================================================================
# Public API
# =============================================================================

def export_to_kicad_sch(circuit: CircuitData) -> str:
    """Export a CircuitData object to a .kicad_sch string."""
    writer = KiCadSchematicWriter()
    content = writer.export(circuit)
    return _normalize_lib_symbol_unit_names(content)


def _normalize_lib_symbol_unit_names(content: str) -> str:
    """Strip library namespace prefixes from lib_symbols sub-unit names.

    KiCad accepts ``C_0_1`` but rejects ``Device:C_0_1`` with
    "Invalid symbol unit name prefix".
    """
    fixed = re.sub(
        r'(\(symbol\s+")([^"\n]*:)([^"\n]+_[0-9]+_[0-9]+")',
        r'\1\3',
        content,
    )
    if fixed != content:
        logger.warning("Normalized lib symbol unit names by stripping namespace prefixes")
    return fixed


def save_kicad_sch(circuit: CircuitData, filepath: str) -> str:
    """Export and save a CircuitData object to a .kicad_sch file."""
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
    import json

    template_path = os.path.join(os.path.dirname(__file__), "..", "templates", "555_timer.json")

    if os.path.exists(template_path):
        with open(template_path, "r") as f:
            data = json.load(f)
        circuit = CircuitData(**data)
        output_path = os.path.join(os.path.dirname(__file__), "..", "output", "555_timer.kicad_sch")
        save_kicad_sch(circuit, output_path)
        print(f"✓ Exported to: {output_path}")

        content = export_to_kicad_sch(circuit)
        for i, line in enumerate(content.split("\n")[:50]):
            print(f"  {i+1:3d} | {line}")
    else:
        print(f"Template not found: {template_path}")