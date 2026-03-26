"""
DFM Engine v2.0 — Design for Manufacturing Checks
Validates PCB designs against manufacturer-specific rules.

v2.0 improvements:
  - Trace width/spacing validation
  - Via/drill size and aspect ratio checks
  - Annular ring width validation
  - Silkscreen width checks
  - Footprint-aware component sizing for overlap detection
  - Severity escalation (info → warning → error → critical)
  - Expanded manufacturer rules (pcbgogo added)
  - Improved cost estimator with quantity breaks and finish options
"""
import logging
import math
from typing import Any, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ── Component size lookup ─────────────────────────────────────────────────────
# Approximate physical dimensions (width, height) in mm keyed by reference
# prefix.  Used for overlap and spacing checks when real footprint data is
# unavailable.

COMPONENT_SIZES: dict[str, tuple[float, float]] = {
    "R":    (2.0, 1.2),    # 0805 resistor
    "C":    (2.0, 1.2),    # 0805 capacitor
    "L":    (3.0, 3.0),    # inductor
    "D":    (2.5, 1.5),    # diode
    "LED":  (2.0, 1.2),    # 0805 LED
    "Q":    (3.0, 3.5),    # SOT-23 transistor
    "U":    (8.0, 8.0),    # IC (QFP/SOIC average)
    "IC":   (8.0, 8.0),
    "J":    (10.0, 5.0),   # connector
    "P":    (10.0, 5.0),
    "SW":   (6.0, 6.0),    # switch
    "F":    (3.2, 1.6),    # fuse
    "Y":    (5.0, 2.0),    # crystal
    "X":    (5.0, 2.0),
    "FB":   (2.0, 1.2),    # ferrite bead
}

DEFAULT_COMP_SIZE = (3.0, 3.0)


def _get_comp_size(ref: str) -> tuple[float, float]:
    """Return (width, height) in mm for a component reference."""
    prefix = "".join(c for c in ref if c.isalpha()).upper()
    return COMPONENT_SIZES.get(prefix, DEFAULT_COMP_SIZE)


@dataclass
class ManufacturerRules:
    """Manufacturing rules for a specific PCB manufacturer."""
    name: str
    min_trace_width: float      # mm
    min_trace_spacing: float    # mm
    min_hole_size: float        # mm
    max_aspect_ratio: float     # board_thickness / hole_diameter
    min_annular_ring: float     # mm
    min_silkscreen_width: float # mm
    min_component_spacing: float # mm
    # Extended rules
    board_thickness: float = 1.6       # mm (default FR4)
    min_soldermask_dam: float = 0.1    # mm
    min_copper_to_edge: float = 0.3    # mm
    max_board_size: tuple[float, float] = (500.0, 500.0)  # mm


# Pre-defined manufacturer rules
MANUFACTURER_RULES = {
    "jlcpcb": ManufacturerRules(
        name="JLCPCB",
        min_trace_width=0.127,
        min_trace_spacing=0.127,
        min_hole_size=0.3,
        max_aspect_ratio=10,
        min_annular_ring=0.15,
        min_silkscreen_width=0.15,
        min_component_spacing=0.2,
        min_copper_to_edge=0.3,
    ),
    "pcbway": ManufacturerRules(
        name="PCBWay",
        min_trace_width=0.1,
        min_trace_spacing=0.1,
        min_hole_size=0.2,
        max_aspect_ratio=12,
        min_annular_ring=0.127,
        min_silkscreen_width=0.12,
        min_component_spacing=0.15,
        min_copper_to_edge=0.25,
    ),
    "oshpark": ManufacturerRules(
        name="OSH Park",
        min_trace_width=0.152,
        min_trace_spacing=0.152,
        min_hole_size=0.254,
        max_aspect_ratio=8,
        min_annular_ring=0.178,
        min_silkscreen_width=0.15,
        min_component_spacing=0.25,
        min_copper_to_edge=0.381,
    ),
    "pcbgogo": ManufacturerRules(
        name="PCBgogo",
        min_trace_width=0.1,
        min_trace_spacing=0.1,
        min_hole_size=0.2,
        max_aspect_ratio=10,
        min_annular_ring=0.127,
        min_silkscreen_width=0.15,
        min_component_spacing=0.15,
        min_copper_to_edge=0.25,
    ),
}


# ── Severity helpers ──────────────────────────────────────────────────────────

def _severity_for_ratio(ratio: float) -> str:
    """Return severity based on how far a value exceeds the limit.

    ratio = actual / required.  Lower is worse.
    """
    if ratio < 0.5:
        return "critical"
    if ratio < 0.8:
        return "error"
    if ratio < 1.0:
        return "warning"
    return "info"


# ── Main DFM check entry point ───────────────────────────────────────────────

def check_dfm_rules(board_data: dict, manufacturer: str = "jlcpcb") -> list[dict]:
    """
    Check Design for Manufacturing rules.

    Args:
        board_data: Dict with components, traces, vias, board dimensions
        manufacturer: Manufacturer name to check rules against

    Returns:
        List of violation dicts with type, message, location, severity
    """
    rules = MANUFACTURER_RULES.get(manufacturer, MANUFACTURER_RULES["jlcpcb"])
    violations: list[dict] = []

    violations.extend(_check_component_spacing(board_data, rules))
    violations.extend(_check_board_boundary(board_data, rules))
    violations.extend(_check_overlapping_components(board_data))
    violations.extend(_check_trace_width(board_data, rules))
    violations.extend(_check_trace_spacing(board_data, rules))
    violations.extend(_check_via_drill(board_data, rules))
    violations.extend(_check_annular_ring(board_data, rules))
    violations.extend(_check_silkscreen(board_data, rules))
    violations.extend(_check_board_size(board_data, rules))

    # Sort by severity for readability: critical > error > warning > info
    severity_order = {"critical": 0, "error": 1, "warning": 2, "info": 3}
    violations.sort(key=lambda v: severity_order.get(v.get("severity", "info"), 3))

    return violations


# ── Individual checks ─────────────────────────────────────────────────────────

def _check_component_spacing(board_data: dict, rules: ManufacturerRules) -> list[dict]:
    """Check minimum spacing between components using footprint-aware sizing."""
    violations = []
    components = board_data.get("components", [])
    min_spacing = rules.min_component_spacing

    for i, comp1 in enumerate(components):
        w1, h1 = _get_comp_size(comp1.get("ref", ""))
        for comp2 in components[i + 1:]:
            w2, h2 = _get_comp_size(comp2.get("ref", ""))

            dx = abs(comp1.get("x", 0) - comp2.get("x", 0))
            dy = abs(comp1.get("y", 0) - comp2.get("y", 0))

            # Edge-to-edge clearance (subtract half-widths from center distance)
            clearance_x = dx - (w1 + w2) / 2
            clearance_y = dy - (h1 + h2) / 2
            clearance = max(clearance_x, clearance_y)

            if clearance < min_spacing:
                ratio = max(0.01, clearance / min_spacing) if min_spacing > 0 else 1.0
                violations.append({
                    "type": "component_spacing",
                    "message": (
                        f"{comp1.get('ref', '?')} and {comp2.get('ref', '?')} are too close "
                        f"({clearance:.2f}mm clearance < {min_spacing}mm minimum)"
                    ),
                    "location": {"x": comp1.get("x", 0), "y": comp1.get("y", 0)},
                    "severity": _severity_for_ratio(ratio),
                    "refs": [comp1.get("ref", ""), comp2.get("ref", "")],
                })

    return violations


def _check_board_boundary(board_data: dict, rules: ManufacturerRules) -> list[dict]:
    """Check that components are within board boundaries with edge clearance."""
    violations = []
    components = board_data.get("components", [])
    board_width = board_data.get("board_width", 100.0)
    board_height = board_data.get("board_height", 80.0)
    margin = max(1.0, rules.min_copper_to_edge)

    for comp in components:
        ref = comp.get("ref", "?")
        x = comp.get("x", 0)
        y = comp.get("y", 0)
        w, h = _get_comp_size(ref)
        half_w, half_h = w / 2, h / 2

        # Check all four edges with component footprint extent
        if x - half_w < margin:
            violations.append({
                "type": "boundary_violation",
                "message": f"{ref} too close to left edge ({x - half_w:.2f}mm, min {margin}mm)",
                "location": {"x": x, "y": y},
                "severity": "error",
                "ref": ref,
            })
        if x + half_w > board_width - margin:
            violations.append({
                "type": "boundary_violation",
                "message": f"{ref} too close to right edge ({board_width - x - half_w:.2f}mm, min {margin}mm)",
                "location": {"x": x, "y": y},
                "severity": "error",
                "ref": ref,
            })
        if y - half_h < margin:
            violations.append({
                "type": "boundary_violation",
                "message": f"{ref} too close to bottom edge ({y - half_h:.2f}mm, min {margin}mm)",
                "location": {"x": x, "y": y},
                "severity": "error",
                "ref": ref,
            })
        if y + half_h > board_height - margin:
            violations.append({
                "type": "boundary_violation",
                "message": f"{ref} too close to top edge ({board_height - y - half_h:.2f}mm, min {margin}mm)",
                "location": {"x": x, "y": y},
                "severity": "error",
                "ref": ref,
            })

    return violations


def _check_overlapping_components(board_data: dict) -> list[dict]:
    """Check for overlapping components using footprint-aware bounding boxes."""
    violations = []
    components = board_data.get("components", [])

    for i, comp1 in enumerate(components):
        ref1 = comp1.get("ref", "?")
        x1, y1 = comp1.get("x", 0), comp1.get("y", 0)
        w1, h1 = _get_comp_size(ref1)

        for comp2 in components[i + 1:]:
            ref2 = comp2.get("ref", "?")
            x2, y2 = comp2.get("x", 0), comp2.get("y", 0)
            w2, h2 = _get_comp_size(ref2)

            # AABB overlap check
            overlap_x = (w1 + w2) / 2 - abs(x1 - x2)
            overlap_y = (h1 + h2) / 2 - abs(y1 - y2)

            if overlap_x > 0 and overlap_y > 0:
                overlap_area = overlap_x * overlap_y
                min_area = min(w1 * h1, w2 * h2)
                overlap_pct = (overlap_area / min_area * 100) if min_area > 0 else 100

                violations.append({
                    "type": "overlap",
                    "message": f"{ref1} and {ref2} overlap ({overlap_pct:.0f}% of smaller component)",
                    "location": {"x": x1, "y": y1},
                    "severity": "critical" if overlap_pct > 50 else "error",
                    "refs": [ref1, ref2],
                })

    return violations


def _check_trace_width(board_data: dict, rules: ManufacturerRules) -> list[dict]:
    """Check trace widths against manufacturer minimums."""
    violations = []
    traces = board_data.get("traces", [])
    min_width = rules.min_trace_width

    for trace in traces:
        width = trace.get("width", 0.254)  # default 10mil
        net = trace.get("net", "unknown")

        if width < min_width:
            ratio = width / min_width if min_width > 0 else 1.0
            violations.append({
                "type": "trace_width",
                "message": (
                    f"Trace on net '{net}' is {width:.3f}mm wide "
                    f"(minimum {min_width:.3f}mm for {rules.name})"
                ),
                "location": trace.get("start", {}),
                "severity": _severity_for_ratio(ratio),
                "net": net,
            })

    return violations


def _check_trace_spacing(board_data: dict, rules: ManufacturerRules) -> list[dict]:
    """Check trace-to-trace spacing against manufacturer minimums."""
    violations = []
    traces = board_data.get("traces", [])
    min_spacing = rules.min_trace_spacing

    # Simple pairwise check for traces on the same layer
    for i, t1 in enumerate(traces):
        for t2 in traces[i + 1:]:
            if t1.get("layer", "F.Cu") != t2.get("layer", "F.Cu"):
                continue

            # Approximate: use start-point distance minus half-widths
            start1 = t1.get("start", {})
            start2 = t2.get("start", {})
            dx = abs(start1.get("x", 0) - start2.get("x", 0))
            dy = abs(start1.get("y", 0) - start2.get("y", 0))
            dist = math.hypot(dx, dy)
            clearance = dist - (t1.get("width", 0.254) + t2.get("width", 0.254)) / 2

            if 0 < clearance < min_spacing:
                violations.append({
                    "type": "trace_spacing",
                    "message": (
                        f"Traces '{t1.get('net', '?')}' and '{t2.get('net', '?')}' "
                        f"are {clearance:.3f}mm apart (minimum {min_spacing:.3f}mm)"
                    ),
                    "location": start1,
                    "severity": _severity_for_ratio(clearance / min_spacing),
                })

    return violations


def _check_via_drill(board_data: dict, rules: ManufacturerRules) -> list[dict]:
    """Check via drill sizes and aspect ratios."""
    violations = []
    vias = board_data.get("vias", [])
    min_hole = rules.min_hole_size
    max_ar = rules.max_aspect_ratio
    thickness = rules.board_thickness

    for via in vias:
        drill = via.get("drill", 0.3)
        net = via.get("net", "unknown")

        if drill < min_hole:
            violations.append({
                "type": "via_drill",
                "message": (
                    f"Via on net '{net}' has {drill:.3f}mm drill "
                    f"(minimum {min_hole:.3f}mm for {rules.name})"
                ),
                "location": {"x": via.get("x", 0), "y": via.get("y", 0)},
                "severity": _severity_for_ratio(drill / min_hole),
                "net": net,
            })

        # Aspect ratio check
        if drill > 0:
            ar = thickness / drill
            if ar > max_ar:
                violations.append({
                    "type": "via_aspect_ratio",
                    "message": (
                        f"Via on net '{net}' has aspect ratio {ar:.1f}:1 "
                        f"(max {max_ar:.0f}:1 for {rules.name})"
                    ),
                    "location": {"x": via.get("x", 0), "y": via.get("y", 0)},
                    "severity": "error" if ar > max_ar * 1.2 else "warning",
                    "net": net,
                })

    return violations


def _check_annular_ring(board_data: dict, rules: ManufacturerRules) -> list[dict]:
    """Check annular ring width on vias and through-hole pads."""
    violations = []
    vias = board_data.get("vias", [])
    min_ring = rules.min_annular_ring

    for via in vias:
        drill = via.get("drill", 0.3)
        pad_diameter = via.get("pad_diameter", drill + 0.3)  # default 0.15mm ring
        ring_width = (pad_diameter - drill) / 2

        if ring_width < min_ring:
            violations.append({
                "type": "annular_ring",
                "message": (
                    f"Via annular ring is {ring_width:.3f}mm "
                    f"(minimum {min_ring:.3f}mm for {rules.name})"
                ),
                "location": {"x": via.get("x", 0), "y": via.get("y", 0)},
                "severity": _severity_for_ratio(ring_width / min_ring),
            })

    return violations


def _check_silkscreen(board_data: dict, rules: ManufacturerRules) -> list[dict]:
    """Check silkscreen line widths."""
    violations = []
    silkscreen = board_data.get("silkscreen", [])
    min_width = rules.min_silkscreen_width

    for item in silkscreen:
        width = item.get("width", 0.15)
        if width < min_width:
            violations.append({
                "type": "silkscreen_width",
                "message": (
                    f"Silkscreen element is {width:.3f}mm wide "
                    f"(minimum {min_width:.3f}mm for {rules.name})"
                ),
                "location": item.get("location", {}),
                "severity": "warning",
            })

    return violations


def _check_board_size(board_data: dict, rules: ManufacturerRules) -> list[dict]:
    """Check board dimensions against manufacturer maximums."""
    violations = []
    bw = board_data.get("board_width", 100.0)
    bh = board_data.get("board_height", 80.0)
    max_w, max_h = rules.max_board_size

    if bw > max_w or bh > max_h:
        violations.append({
            "type": "board_size",
            "message": (
                f"Board {bw:.0f}×{bh:.0f}mm exceeds {rules.name} maximum "
                f"{max_w:.0f}×{max_h:.0f}mm"
            ),
            "location": {"x": 0, "y": 0},
            "severity": "error",
        })

    # Warn on very small boards
    if bw < 10 or bh < 10:
        violations.append({
            "type": "board_size",
            "message": f"Board {bw:.0f}×{bh:.0f}mm is unusually small — verify dimensions",
            "location": {"x": 0, "y": 0},
            "severity": "warning",
        })

    return violations


# ── Cost Estimator ────────────────────────────────────────────────────────────

def estimate_manufacturing_cost(board_data: dict, manufacturer: str = "jlcpcb") -> dict:
    """
    Estimate manufacturing cost based on board parameters.

    Returns:
        Dict with unit_cost, quantity pricing tiers, lead_time, and options
    """
    board_width = board_data.get("board_width", 100.0)
    board_height = board_data.get("board_height", 80.0)
    layer_count = board_data.get("layer_count", 2)
    component_count = len(board_data.get("components", []))
    finish = board_data.get("surface_finish", "HASL")

    # Board area in cm²
    area_cm2 = (board_width * board_height) / 100

    # Base pricing by layer count (approximate JLCPCB rates for 5 pcs)
    base_prices = {1: 1.5, 2: 2.0, 4: 5.0, 6: 15.0, 8: 25.0, 10: 40.0}
    base_cost = base_prices.get(layer_count, 30.0 + (layer_count - 8) * 10)

    # Area surcharge (boards > 100cm² cost more)
    if area_cm2 > 100:
        base_cost += (area_cm2 - 100) * 0.025
    elif area_cm2 < 25:
        # Small boards often have minimums
        base_cost = max(base_cost, 2.0)

    # Surface finish surcharge
    finish_surcharge = {
        "HASL": 0.0,
        "HASL_LEAD_FREE": 0.5,
        "ENIG": 3.0,
        "OSP": 0.5,
        "IMMERSION_SILVER": 2.0,
        "IMMERSION_TIN": 1.5,
    }
    base_cost += finish_surcharge.get(finish.upper().replace(" ", "_"), 0.0)

    # Copper weight surcharge
    copper_oz = board_data.get("copper_weight_oz", 1.0)
    if copper_oz > 1.0:
        base_cost += (copper_oz - 1.0) * 2.0

    return {
        "unit_cost": round(base_cost, 2),
        "quantity_5": round(base_cost * 5, 2),
        "quantity_10": round(base_cost * 10 * 0.9, 2),     # 10% discount
        "quantity_50": round(base_cost * 50 * 0.7, 2),     # 30% discount
        "quantity_100": round(base_cost * 100 * 0.55, 2),  # 45% discount
        "quantity_500": round(base_cost * 500 * 0.4, 2),   # 60% discount
        "lead_time_days": 3 if layer_count <= 2 else 5 if layer_count <= 4 else 10,
        "express_lead_time_days": 1 if layer_count <= 2 else 3,
        "currency": "USD",
        "notes": (
            f"{layer_count}L, {area_cm2:.1f}cm², {finish}, "
            f"{copper_oz}oz Cu, {component_count} components"
        ),
    }
