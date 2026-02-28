"""
DFM Engine - Design for Manufacturing Checks
Validates PCB designs against manufacturer-specific rules.
"""
import logging
from typing import Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ManufacturerRules:
    """Manufacturing rules for a specific PCB manufacturer."""
    name: str
    min_trace_width: float  # mm
    min_trace_spacing: float  # mm
    min_hole_size: float  # mm
    max_aspect_ratio: float  # depth/diameter
    min_annular_ring: float  # mm
    min_silkscreen_width: float  # mm
    min_component_spacing: float  # mm


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
    ),
}


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
    violations = []

    # Check component spacing
    violations.extend(_check_component_spacing(board_data, rules))

    # Check board boundary
    violations.extend(_check_board_boundary(board_data))

    # Check for overlapping components
    violations.extend(_check_overlapping_components(board_data))

    return violations


def _check_component_spacing(board_data: dict, rules: ManufacturerRules) -> list[dict]:
    """Check minimum spacing between components."""
    violations = []
    components = board_data.get("components", [])
    min_spacing = rules.min_component_spacing

    for i, comp1 in enumerate(components):
        for comp2 in components[i + 1:]:
            dx = abs(comp1["x"] - comp2["x"])
            dy = abs(comp1["y"] - comp2["y"])
            distance = (dx ** 2 + dy ** 2) ** 0.5

            if distance < min_spacing:
                violations.append({
                    "type": "component_spacing",
                    "message": f"{comp1['ref']} and {comp2['ref']} are too close "
                               f"({distance:.2f}mm < {min_spacing}mm)",
                    "location": {"x": comp1["x"], "y": comp1["y"]},
                    "severity": "warning",
                })

    return violations


def _check_board_boundary(board_data: dict) -> list[dict]:
    """Check that components are within board boundaries."""
    violations = []
    components = board_data.get("components", [])
    board_width = board_data.get("board_width", 100.0)
    board_height = board_data.get("board_height", 80.0)
    margin = 1.0  # mm edge margin

    for comp in components:
        x, y = comp["x"], comp["y"]

        if x < margin:
            violations.append({
                "type": "boundary_violation",
                "message": f"{comp['ref']} is too close to left edge ({x:.2f}mm)",
                "location": {"x": x, "y": y},
                "severity": "error",
            })
        elif x > board_width - margin:
            violations.append({
                "type": "boundary_violation",
                "message": f"{comp['ref']} is too close to right edge",
                "location": {"x": x, "y": y},
                "severity": "error",
            })

        if y < margin:
            violations.append({
                "type": "boundary_violation",
                "message": f"{comp['ref']} is too close to bottom edge",
                "location": {"x": x, "y": y},
                "severity": "error",
            })
        elif y > board_height - margin:
            violations.append({
                "type": "boundary_violation",
                "message": f"{comp['ref']} is too close to top edge",
                "location": {"x": x, "y": y},
                "severity": "error",
            })

    return violations


def _check_overlapping_components(board_data: dict) -> list[dict]:
    """Check for overlapping components (simple bounding box check)."""
    violations = []
    components = board_data.get("components", [])

    # Approximate component sizes (could be enhanced with real footprint data)
    default_size = 2.0  # mm

    for i, comp1 in enumerate(components):
        for comp2 in components[i + 1:]:
            dx = abs(comp1["x"] - comp2["x"])
            dy = abs(comp1["y"] - comp2["y"])

            # Simple overlap check
            if dx < default_size and dy < default_size:
                violations.append({
                    "type": "overlap",
                    "message": f"{comp1['ref']} and {comp2['ref']} appear to overlap",
                    "location": {"x": comp1["x"], "y": comp1["y"]},
                    "severity": "error",
                })

    return violations


def estimate_manufacturing_cost(board_data: dict, manufacturer: str = "jlcpcb") -> dict:
    """
    Estimate manufacturing cost based on board parameters.
    
    Returns:
        Dict with unit_cost, quantity pricing, lead_time
    """
    board_width = board_data.get("board_width", 100.0)
    board_height = board_data.get("board_height", 80.0)
    layer_count = board_data.get("layer_count", 2)

    # Calculate board area in cm²
    area_cm2 = (board_width * board_height) / 100

    # Base pricing (approximate JLCPCB rates)
    base_prices = {2: 2.0, 4: 5.0, 6: 15.0, 8: 25.0}
    base_cost = base_prices.get(layer_count, 30.0)

    # Area surcharge
    if area_cm2 > 100:
        base_cost += (area_cm2 - 100) * 0.02

    return {
        "unit_cost": round(base_cost, 2),
        "quantity_5": round(base_cost * 5, 2),
        "quantity_10": round(base_cost * 10 * 0.9, 2),  # 10% discount
        "quantity_50": round(base_cost * 50 * 0.7, 2),  # 30% discount
        "lead_time_days": 5 if layer_count <= 4 else 10,
        "currency": "USD",
    }
