"""
Schematic Engine - Converts natural language to KiCad schematics.
Uses LLM JSON circuit generation → CircuitData schema → KiCad exporter.
"""
import os
import logging
import re
from pathlib import Path
from typing import Optional, Any

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent.parent / "output"


def generate_skidl_from_prompt(llm: Any, prompt: str) -> dict:
    """
    Generate a KiCad schematic from a natural language description.
    Uses the LLM JSON circuit pipeline (not SKiDL).

    Returns:
        dict with success, component_count, output_file, error
    """
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Step 1: Generate circuit JSON from LLM
        circuit_dict = llm.generate_circuit_json(prompt)
        if not circuit_dict:
            return {"success": False, "error": "LLM returned empty circuit data", "component_count": 0}

        # Step 2: Validate against CircuitData schema
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from circuit_schema import CircuitData
        circuit = CircuitData(**circuit_dict)

        # Step 3: Export to .kicad_sch
        from engines.kicad_exporter import export_to_kicad_sch
        slug = re.sub(r"[^a-zA-Z0-9_-]", "_", circuit.description[:40]).strip("_") or "circuit"
        output_path = OUTPUT_DIR / f"{slug}.kicad_sch"
        sch_content = export_to_kicad_sch(circuit)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(sch_content)

        logger.info(f"Schematic saved: {output_path}")
        return {
            "success": True,
            "component_count": len(circuit.components),
            "output_file": str(output_path),
            "error": None,
        }

    except Exception as e:
        logger.error(f"Schematic generation failed: {e}")
        return {"success": False, "error": str(e), "component_count": 0}


# ── Legacy helpers kept for import compatibility ──────────────────────────────

def _clean_generated_code(code: str) -> str:
    """No-op legacy stub."""
    return code


def _validate_skidl_code(code: str) -> bool:
    """No-op legacy stub — always returns True."""
    return True


def _execute_skidl(code: str) -> Optional[str]:
    """No-op legacy stub."""
    return None


# --- Component Library Mapping ---

COMPONENT_ALIASES = {
    "resistor": ("Device", "R"),
    "capacitor": ("Device", "C"),
    "inductor": ("Device", "L"),
    "led": ("Device", "LED"),
    "diode": ("Device", "D"),
    "transistor": ("Device", "Q"),
    "mosfet": ("Device", "Q_NMOS_GSD"),
    "opamp": ("Amplifier_Operational", "LM358"),
    "regulator": ("Regulator_Linear", "LM7805_TO220"),
    "mcu": ("MCU_Microchip_ATmega", "ATmega328P-AU"),
    "usb": ("Connector", "USB_C_Receptacle_USB2.0"),
}


def get_component_info(component_name: str) -> tuple[str, str]:
    """Get library and part name for a component."""
    name_lower = component_name.lower()
    for alias, (lib, part) in COMPONENT_ALIASES.items():
        if alias in name_lower:
            return lib, part
    return "Device", "R"  # Default fallback
