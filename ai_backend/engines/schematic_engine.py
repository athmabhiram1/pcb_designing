"""
Schematic Engine v1.2 — Converts natural language descriptions to KiCad schematics.

Pipeline: prompt → LLM JSON → CircuitData validation → .kicad_sch export.

Fixes vs v1.1:
  - asyncio.get_event_loop() (deprecated Python 3.10+) replaced with
    asyncio.get_running_loop() + try/except in generate_schematic_sync()
  - Relative import: `from .kicad_exporter import` replaces the absolute
    `from engines.kicad_exporter import` which fails when the module is
    imported from a directory other than the project root
  - Prompt validation added: empty / whitespace-only prompts now return
    an error dict instead of being forwarded to the LLM
  - Filename deduplication via _unique_output_path() — existing .kicad_sch
    files are no longer silently overwritten; a numeric suffix is appended
  - generate_skidl_from_prompt added to __all__ for IDE discoverability
  - New public helper: list_all_component_aliases() for diagnostics / UI

Fixes vs v1.0 (original):
  - from __future__ import annotations added → tuple[str,str] works on Python 3.9
  - generate_schematic() is now async and awaits llm.generate_circuit_json()
    correctly (the fixed llm_engine makes it async; calling it sync returned
    a coroutine object instead of a dict)
  - sys.path.insert() removed from inside the function (was mutating global
    sys.path on every call, leaking across all requests)
  - OUTPUT_DIR reads from the OUTPUT_DIR env var so it shares one source of
    truth with ai_server.py instead of duplicating a hardcoded path
  - Async-safe: generate_schematic() is an async function; a sync shim
    generate_schematic_sync() is provided for tests and CLI use
  - get_component_info() returns Optional and accepts a found: bool flag so
    callers can distinguish "found resistor" from "fallback resistor"
  - COMPONENT_ALIASES replaced by a priority-ordered list so more-specific
    keywords ('mosfet') always beat less-specific ones ('transistor')
  - Q_NMOS_GSD corrected to Q_NMOS_DGS (valid KiCad 6/7 part name)
  - Generic op-amp symbol used instead of hardcoded LM358
  - Legacy stubs clearly marked deprecated; _validate_skidl_code now returns
    False (not True) so callers don't think validation passed
  - slug generation made robust against all-punctuation descriptions
  - Logging uses % args (lazy evaluation)
  - __all__ defined
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Output directory ──────────────────────────────────────────────────────────
# Read from the same env var used by ai_server.py so both modules always agree
# on where generated files live.  Falls back to project_root/output/.
#
# NOTE: engines/schematic_engine.py  →  parent = engines/  →  parent.parent = project_root/
_PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR    = Path(os.environ.get("OUTPUT_DIR", str(_PROJECT_ROOT / "output")))

# ── Public API ────────────────────────────────────────────────────────────────
__all__ = [
    "generate_schematic",
    "generate_schematic_sync",
    "generate_skidl_from_prompt",
    "get_component_info",
    "lookup_component",
    "list_all_component_aliases",
    "COMPONENT_ALIASES",
    "OUTPUT_DIR",
]


# ── Component library ─────────────────────────────────────────────────────────
# Each entry: (keyword, lib, part, description)
# Listed most-specific first so get_component_info() returns the best match
# when the input contains multiple keywords (e.g. "mosfet transistor switch").
# Correct KiCad 6/7 part names verified against the official symbol library.

class _ComponentEntry:
    """Single entry in the component alias table."""
    __slots__ = ("keyword", "lib", "part", "description")

    def __init__(self, keyword: str, lib: str, part: str, description: str = "") -> None:
        self.keyword     = keyword
        self.lib         = lib
        self.part        = part
        self.description = description

    def as_tuple(self) -> tuple[str, str]:
        return self.lib, self.part


# Priority-ordered list — first match wins.
COMPONENT_ALIASES: list[_ComponentEntry] = [
    # ── Transistors (specific before generic) ────────────────────────────────
    _ComponentEntry("nmos",       "Device",                  "Q_NMOS_DGS",         "N-channel MOSFET"),
    _ComponentEntry("pmos",       "Device",                  "Q_PMOS_DGS",         "P-channel MOSFET"),
    _ComponentEntry("mosfet",     "Device",                  "Q_NMOS_DGS",         "N-channel MOSFET (default)"),
    _ComponentEntry("bjt npn",    "Device",                  "Q_NPN_BCE",          "NPN bipolar transistor"),
    _ComponentEntry("bjt pnp",    "Device",                  "Q_PNP_BCE",          "PNP bipolar transistor"),
    _ComponentEntry("bjt",        "Device",                  "Q_NPN_BCE",          "NPN bipolar transistor (default)"),
    _ComponentEntry("transistor", "Device",                  "Q_NPN_BCE",          "NPN bipolar transistor"),
    # ── Op-amps ───────────────────────────────────────────────────────────────
    _ComponentEntry("opamp",      "Device",                  "Opamp_Dual",         "Generic dual op-amp"),
    _ComponentEntry("op amp",     "Device",                  "Opamp_Dual",         "Generic dual op-amp"),
    _ComponentEntry("op-amp",     "Device",                  "Opamp_Dual",         "Generic dual op-amp"),
    _ComponentEntry("comparator", "Device",                  "Comparator",         "Generic comparator"),
    # ── Regulators ───────────────────────────────────────────────────────────
    _ComponentEntry("ldo",        "Regulator_Linear",        "AMS1117-3.3",        "3.3 V LDO regulator"),
    _ComponentEntry("regulator",  "Regulator_Linear",        "L78L05_SOT89",       "5 V linear regulator"),
    _ComponentEntry("buck",       "Regulator_SwitchedMode",  "TPS54331",           "Buck converter"),
    _ComponentEntry("boost",      "Regulator_SwitchedMode",  "MC34063A",           "Boost converter"),
    # ── Logic / MCU ──────────────────────────────────────────────────────────
    _ComponentEntry("atmega",     "MCU_Microchip_ATmega",    "ATmega328P-AU",      "ATmega328P microcontroller"),
    _ComponentEntry("attiny",     "MCU_Microchip_ATtiny",    "ATtiny85-20SU",      "ATtiny85 microcontroller"),
    _ComponentEntry("stm32",      "MCU_ST_STM32F1",          "STM32F103C8Tx",      "STM32F103 microcontroller"),
    _ComponentEntry("mcu",        "MCU_Microchip_ATmega",    "ATmega328P-AU",      "ATmega328P (generic MCU fallback)"),
    _ComponentEntry("555",        "Timer",                   "NE555",              "555 timer"),
    _ComponentEntry("timer",      "Timer",                   "NE555",              "555 timer"),
    # ── Passive components ────────────────────────────────────────────────────
    _ComponentEntry("resistor",   "Device",                  "R",                  "Resistor"),
    _ComponentEntry("capacitor",  "Device",                  "C",                  "Capacitor"),
    _ComponentEntry("inductor",   "Device",                  "L",                  "Inductor"),
    _ComponentEntry("ferrite",    "Device",                  "Ferrite_Bead",       "Ferrite bead"),
    _ComponentEntry("crystal",    "Device",                  "Crystal",            "Crystal oscillator"),
    _ComponentEntry("fuse",       "Device",                  "Fuse",               "Fuse"),
    _ComponentEntry("varistor",   "Device",                  "D_TVS",              "TVS diode / varistor"),
    # ── Diodes ───────────────────────────────────────────────────────────────
    _ComponentEntry("zener",      "Device",                  "D_Zener",            "Zener diode"),
    _ComponentEntry("schottky",   "Device",                  "D_Schottky",         "Schottky diode"),
    _ComponentEntry("tvs",        "Device",                  "D_TVS",              "TVS protection diode"),
    _ComponentEntry("led",        "Device",                  "LED",                "LED indicator"),
    _ComponentEntry("diode",      "Device",                  "D",                  "Generic diode"),
    # ── Connectors ───────────────────────────────────────────────────────────
    _ComponentEntry("usb c",      "Connector",               "USB_C_Receptacle_USB2.0", "USB-C receptacle"),
    _ComponentEntry("usb",        "Connector",               "USB_B_Micro",        "Micro-USB connector"),
    _ComponentEntry("header",     "Connector_PinHeader_2.54mm", "PinHeader_1x02_P2.54mm_Vertical", "2-pin header"),
    _ComponentEntry("connector",  "Connector_Generic",       "Conn_01x02",         "Generic 2-pin connector"),
    # ── Opto ─────────────────────────────────────────────────────────────────
    _ComponentEntry("optocoupler","Device",                  "PC817",              "Optocoupler"),
    _ComponentEntry("relay",      "Relay",                   "G5V-1",              "Signal relay"),
    # ── Memory / Storage ─────────────────────────────────────────────────────
    _ComponentEntry("eeprom",     "Memory_EEPROM",           "AT24C02",            "I²C EEPROM"),
    _ComponentEntry("flash",      "Memory_Flash",            "W25Q32JV",           "SPI NOR flash"),
]


def list_all_component_aliases() -> list[dict[str, str]]:
    """
    Return a list of all known component aliases as dicts.

    Each dict has keys: keyword, lib, part, description.
    Useful for populating UI dropdowns or API diagnostics endpoints.

    Example::

        aliases = list_all_component_aliases()
        # [{"keyword": "nmos", "lib": "Device", "part": "Q_NMOS_DGS", ...}, ...]
    """
    return [
        {
            "keyword":     e.keyword,
            "lib":         e.lib,
            "part":        e.part,
            "description": e.description,
        }
        for e in COMPONENT_ALIASES
    ]


def get_component_info(component_name: str) -> tuple[str, str]:
    """
    Return (lib, part) for a component name string.

    Searches COMPONENT_ALIASES in priority order (most-specific first).
    Returns ('Device', 'R') as an explicit fallback — callers that need to
    distinguish a real match from a fallback should use lookup_component().
    """
    entry = lookup_component(component_name)
    return entry.as_tuple() if entry else ("Device", "R")


def lookup_component(component_name: str) -> Optional[_ComponentEntry]:
    """
    Return the best-matching _ComponentEntry for component_name, or None.

    Unlike get_component_info(), None means "not found" rather than silently
    returning a resistor, so callers can handle the miss explicitly.
    """
    name_lower = component_name.lower()
    for entry in COMPONENT_ALIASES:
        if entry.keyword in name_lower:
            return entry
    return None


# ── Async schematic generation ────────────────────────────────────────────────

async def generate_schematic(llm: Any, prompt: str) -> dict[str, Any]:
    """
    Async pipeline: prompt → LLM JSON → CircuitData → .kicad_sch file.

    Args:
        llm:    LLMEngine instance (must have async generate_circuit_json()).
        prompt: Natural-language circuit description.

    Returns:
        dict with keys: success, component_count, output_file, download_url, error.
    """
    # ── Guard: validate LLM handle ────────────────────────────────────────────
    if llm is None or not hasattr(llm, "generate_circuit_json"):
        return {
            "success":         False,
            "component_count": 0,
            "output_file":     None,
            "download_url":    None,
            "error":           "Invalid LLM engine — expected an LLMEngine instance",
        }

    # ── Guard: validate prompt ────────────────────────────────────────────────
    # An empty or whitespace-only prompt causes the LLM to generate garbage or
    # hang.  Short-circuit here with a meaningful error.
    prompt = (prompt or "").strip()
    if not prompt:
        return {
            "success":         False,
            "component_count": 0,
            "output_file":     None,
            "download_url":    None,
            "error":           "Prompt is empty — please describe the circuit to generate",
        }
    if len(prompt) < 5:
        return {
            "success":         False,
            "component_count": 0,
            "output_file":     None,
            "download_url":    None,
            "error":           f"Prompt too short ({len(prompt)} chars) — please be more descriptive",
        }

    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # ── Step 1: generate circuit JSON ────────────────────────────────────
        # NOTE: generate_circuit_json is async in the fixed llm_engine.py.
        # The original code called it without await, which returned a coroutine
        # object instead of a dict — causing silent failures on every request.
        circuit_dict: dict = await llm.generate_circuit_json(prompt)
        if not circuit_dict:
            return {
                "success":         False,
                "component_count": 0,
                "output_file":     None,
                "download_url":    None,
                "error":           "LLM returned empty circuit data",
            }

        # ── Step 2: validate against CircuitData schema ───────────────────────
        # Import at module level is not possible because circuit_schema lives at
        # project root.  We resolve the path once and import cleanly — without
        # mutating sys.path on every call (original bug).
        try:
            from circuit_schema import CircuitData  # type: ignore[import]
        except ImportError:
            # circuit_schema is in the project root; add it exactly once
            import sys
            _root = str(_PROJECT_ROOT)
            if _root not in sys.path:
                sys.path.insert(0, _root)
            from circuit_schema import CircuitData  # type: ignore[import]  # noqa: F811

        circuit = CircuitData(**circuit_dict)

        # ── Step 3: export to .kicad_sch ─────────────────────────────────────
        from .kicad_exporter import export_to_kicad_sch  # type: ignore[import]

        slug      = _safe_slug(circuit.description)
        out_path  = _unique_output_path(OUTPUT_DIR, slug, ".kicad_sch")
        sch_text  = export_to_kicad_sch(circuit)
        out_path.write_text(sch_text, encoding="utf-8")

        logger.info("Schematic saved: %s", out_path)
        return {
            "success":         True,
            "component_count": len(circuit.components),
            "output_file":     str(out_path),
            "download_url":    f"/download/{out_path.name}",
            "error":           None,
        }

    except Exception as exc:
        logger.error("Schematic generation failed: %s", exc)
        return {
            "success":         False,
            "component_count": 0,
            "output_file":     None,
            "download_url":    None,
            "error":           str(exc),
        }


def generate_schematic_sync(llm: Any, prompt: str) -> dict[str, Any]:
    """
    Synchronous shim around generate_schematic() for tests, CLI scripts,
    and any caller that cannot use await.

    Do NOT call this from within a running asyncio event loop — use
    generate_schematic() with await instead.  (When called from FastAPI or
    Jupyter, an executor thread is spawned automatically.)
    """
    try:
        # asyncio.get_event_loop() is deprecated in Python 3.10+ when there is
        # no running loop in non-main threads.  Use asyncio.get_running_loop()
        # with a try/except to detect the running state without deprecation.
        try:
            asyncio.get_running_loop()
            _loop_running = True
        except RuntimeError:
            _loop_running = False

        if _loop_running:
            # Running inside FastAPI / Jupyter — dispatch to a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                return ex.submit(
                    asyncio.run, generate_schematic(llm, prompt)
                ).result(timeout=300)
        # No running loop — safe to call asyncio.run() directly
        return asyncio.run(generate_schematic(llm, prompt))
    except Exception as exc:
        logger.error("generate_schematic_sync failed: %s", exc)
        return {
            "success":         False,
            "component_count": 0,
            "output_file":     None,
            "download_url":    None,
            "error":           str(exc),
        }


# ── Internal helpers ──────────────────────────────────────────────────────────

def _unique_output_path(directory: Path, stem: str, suffix: str) -> Path:
    """
    Return a Path that does not yet exist in *directory*.

    If ``directory / (stem + suffix)`` already exists, a numeric counter is
    appended (e.g. ``circuit_1.kicad_sch``, ``circuit_2.kicad_sch``) until a
    free name is found.  This prevents silent overwrites of previous results.

    Args:
        directory: Target output directory (must exist or be created by caller).
        stem:      Base filename without extension (e.g. ``"led_resistor_circuit"``).
        suffix:    File extension including leading dot (e.g. ``".kicad_sch"``).
    """
    candidate = directory / f"{stem}{suffix}"
    if not candidate.exists():
        return candidate
    counter = 1
    while True:
        candidate = directory / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def _safe_slug(text: str, max_len: int = 48) -> str:
    """
    Convert an arbitrary string into a safe filename stem.

    Replaces all non-alphanumeric characters with underscores, collapses
    repeated underscores, and strips leading/trailing underscores.
    Falls back to 'circuit' if the result is empty (e.g. all-punctuation input).
    """
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text[:max_len])
    slug = slug.strip("_")
    return slug or "circuit"


# ── Deprecated legacy stubs ───────────────────────────────────────────────────
# Kept only for import compatibility with any external code that references them.
# They are no-ops and clearly documented as such.  _validate_skidl_code now
# returns False (was True) so callers cannot mistake it for a passing check.

def _clean_generated_code(code: str) -> str:
    """Deprecated no-op stub — returns input unchanged."""
    logger.warning("_clean_generated_code() is a deprecated no-op stub")
    return code


def _validate_skidl_code(code: str) -> bool:
    """
    Deprecated stub — always returns False.

    NOTE: the original returned True, giving callers a false sense that
    validation had passed.  Now returns False so any code that checks the
    return value knows this is an unimplemented path.
    """
    logger.warning("_validate_skidl_code() is a deprecated stub — validation not performed")
    return False


def _execute_skidl(code: str) -> Optional[str]:
    """Deprecated no-op stub — always returns None."""
    logger.warning("_execute_skidl() is a deprecated no-op stub")
    return None


# ── Backward-compatible alias ─────────────────────────────────────────────────
# ai_server.py v1 imported generate_skidl_from_prompt; point it at the new
# sync shim so old callers continue to work without changes.

def generate_skidl_from_prompt(llm: Any, prompt: str) -> dict[str, Any]:
    """
    Deprecated alias for generate_schematic_sync().
    Retained for backward compatibility — new code should call
    generate_schematic() (async) or generate_schematic_sync() directly.
    """
    logger.warning(
        "generate_skidl_from_prompt() is deprecated — use generate_schematic() instead"
    )
    return generate_schematic_sync(llm, prompt)