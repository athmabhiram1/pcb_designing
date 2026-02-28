"""
LLM Engine — Local Large Language Model Inference for Circuit Generation.

Supports Ollama API (priority) and llama-cpp-python (fallback).
Optimised for structured JSON circuit output using the CircuitData schema.

Fixes vs original:
  - 'from __future__ import annotations' added → list[str]|None, set[str] work on Python 3.9+
  - MODELS_DIR path corrected (was three levels up, now project-root/models)
  - All Ollama HTTP calls run in a thread via asyncio.to_thread so the FastAPI
    event loop is never blocked
  - _fix_bypass_caps no longer mutates its input — works on a deep copy
  - generate_circuit_json wrapped in a top-level try/except so errors surface
    as clean return values, not uncaught exceptions into ai_server.py
  - OLLAMA_MODEL env var handled safely (stripped, empty string ignored)
  - _validate_circuit_structure checks every component, not just the first
  - _try_fix_json handles /* */ block comments and unquoted keys
  - _extract_json finds the outermost complete object reliably
  - Pin cross-validation: connection pins are checked against component pin lists
  - llama_cpp call runs in a thread and respects a configurable timeout
  - Temperature schedule and retry count are configurable via env vars
  - FEW_SHOT_EXAMPLE validated at import time so typos fail loudly
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
# engines/llm_engine.py lives inside the 'engines' package.
# Project layout:  project_root/
#                      engines/llm_engine.py   ← __file__
#                      models/                 ← MODELS_DIR
#
# Original used dirname(dirname(__file__)) + "../models" → one level too high.

_ENGINES_DIR  = Path(__file__).parent          # .../engines/
_PROJECT_ROOT = _ENGINES_DIR.parent            # .../project_root/
MODELS_DIR    = Path(os.environ.get("MODELS_DIR", str(_PROJECT_ROOT / "models")))

DEFAULT_GGUF_MODEL = os.environ.get(
    "LLM_GGUF_MODEL", "deepseek-coder-6.7b-instruct.Q5_K_M.gguf"
)

# ── Ollama configuration ──────────────────────────────────────────────────────
OLLAMA_API_URL  = os.environ.get("OLLAMA_API_URL",  "http://localhost:11434/api/generate")
OLLAMA_TAGS_URL = os.environ.get("OLLAMA_TAGS_URL", "http://localhost:11434/api/tags")
OLLAMA_TIMEOUT  = int(os.environ.get("OLLAMA_TIMEOUT_S", "120"))

# Preferred models tried in order; env-var override goes first (if set)
_env_model = os.environ.get("OLLAMA_MODEL", "").strip()
OLLAMA_MODEL_CANDIDATES: list[str] = (
    [_env_model] if _env_model else []
) + [
    "deepseek-coder:6.7b",
    "deepseek-coder:latest",
    "deepseek-coder-v2:latest",
    "codellama:7b",
    "codellama:latest",
    "llama3:latest",
    "mistral:latest",
]

# ── Generation parameters ─────────────────────────────────────────────────────
LLM_MAX_TOKENS    = int(os.environ.get("LLM_MAX_TOKENS", "3072"))
LLM_RETRIES       = int(os.environ.get("LLM_RETRIES", "2"))
LLM_BASE_TEMP     = float(os.environ.get("LLM_BASE_TEMP", "0.1"))
LLM_TEMP_STEP     = float(os.environ.get("LLM_TEMP_STEP", "0.15"))
LLAMA_CPP_TIMEOUT = int(os.environ.get("LLAMA_CPP_TIMEOUT_S", "180"))

# ── Thread pool for blocking inference calls ──────────────────────────────────
# Both Ollama HTTP and llama_cpp are synchronous; we run them in a thread so
# the FastAPI event loop stays responsive.
_THREAD_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="llm_worker")

# ── Known power and ground net names ─────────────────────────────────────────
_POWER_NETS: frozenset[str] = frozenset({
    "VCC", "VDD", "3V3", "3.3V", "5V", "1V8", "1.8V", "12V", "24V",
    "+5V", "+3V3", "+3.3V", "+12V", "VPWR", "VSUP",
    "AVCC", "DVCC", "VCCIO", "VCCINT",
})
_GROUND_NETS: frozenset[str] = frozenset({
    "GND", "VSS", "AGND", "DGND", "PGND", "SGND", "VEE", "VSSA", "VSSD",
})
_ALL_POWER_NETS: frozenset[str] = _POWER_NETS | _GROUND_NETS
# Pre-computed uppercase set — used in inner loops where rebuilding every
# iteration would be O(n) per call.  Defined here so module import validates it.
_ALL_POWER_NETS_UPPER: frozenset[str] = frozenset(n.upper() for n in _ALL_POWER_NETS)

_BYPASS_KEYWORDS = ("bypass", "decoupling", "decouple", "filter", "power supply")

# ── System prompt ─────────────────────────────────────────────────────────────
CIRCUIT_SYSTEM_PROMPT = """\
You are an expert electronic circuit designer. Your ONLY job is to output a single valid JSON object describing a circuit.

## STRICT RULES
1. Output ONLY a JSON object. No markdown, no explanations, no code fences.
2. Every component MUST have: ref, lib, part, value, footprint, pins.
3. Every connection MUST reference pins that exist on the components listed.
4. Pin format in connections: "REF.PIN_NUMBER" — e.g. "R1.1", "U1.8".
5. Use real KiCad library names: Device, Timer, Connector, Regulator_Linear, etc.
6. Use real KiCad footprints:
     Resistors    → Resistor_SMD:R_0805_2012Metric
     Capacitors   → Capacitor_SMD:C_0402_1005Metric
     ICs (DIP)    → Package_DIP:DIP-8_W7.62mm
     ICs (SOIC)   → Package_SO:SOIC-8_3.9x4.9mm_P1.27mm
     LEDs         → LED_SMD:LED_0805_2012Metric
     Connectors   → Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical
7. Include power nets VCC and GND connecting to the correct pins.
8. Bypass / decoupling capacitors MUST connect ONLY to power rails.
   Pin 1 → VCC net. Pin 2 → GND net. NEVER in the signal path.
9. Do NOT include the same pin ("REF.PIN") in more than one net.
10. Keep the description field concise (≤ 120 chars).

## OUTPUT SCHEMA
{
  "description": "<one-line circuit description>",
  "components": [
    {
      "ref":         "<designator>",
      "lib":         "<KiCad library>",
      "part":        "<part name>",
      "value":       "<value>",
      "footprint":   "<KiCad footprint>",
      "description": "<component purpose>",
      "pins": [
        {"number": "<pin#>", "name": "<function>"}
      ]
    }
  ],
  "connections": [
    {"net": "<net name>", "pins": ["REF.PIN", "REF.PIN"]}
  ]
}\
"""

# ── Few-shot example ──────────────────────────────────────────────────────────
# Validated at import time so a typo fails loudly rather than silently
# sending a broken example to the LLM.
_FEW_SHOT_RAW = """\
USER: Simple LED circuit with current-limiting resistor and bypass cap on VCC

ASSISTANT: {
  "description": "5 V LED indicator: 330 Ω limits current to ~13 mA, 100 nF bypass cap on supply",
  "components": [
    {
      "ref": "R1",
      "lib": "Device",
      "part": "R",
      "value": "330",
      "footprint": "Resistor_SMD:R_0805_2012Metric",
      "description": "Current-limiting resistor for LED",
      "pins": [{"number": "1", "name": "1"}, {"number": "2", "name": "2"}]
    },
    {
      "ref": "D1",
      "lib": "Device",
      "part": "LED",
      "value": "Red",
      "footprint": "LED_SMD:LED_0805_2012Metric",
      "description": "Red indicator LED",
      "pins": [{"number": "1", "name": "A"}, {"number": "2", "name": "K"}]
    },
    {
      "ref": "C1",
      "lib": "Device",
      "part": "C",
      "value": "100nF",
      "footprint": "Capacitor_SMD:C_0402_1005Metric",
      "description": "Power supply bypass capacitor — VCC to GND only, NOT in signal path",
      "pins": [{"number": "1", "name": "+"}, {"number": "2", "name": "-"}]
    }
  ],
  "connections": [
    {"net": "VCC",       "pins": ["R1.1", "C1.1"]},
    {"net": "LED_ANODE", "pins": ["R1.2", "D1.1"]},
    {"net": "GND",       "pins": ["D1.2", "C1.2"]}
  ]
}"""

# Extract and validate the example JSON at module load
def _validate_few_shot() -> str:
    """Parse the embedded JSON in FEW_SHOT_EXAMPLE and raise on error."""
    match = re.search(r'\{.*\}', _FEW_SHOT_RAW, re.DOTALL)
    if not match:
        raise ValueError("FEW_SHOT_EXAMPLE contains no JSON object — fix the constant")
    try:
        json.loads(match.group())
    except json.JSONDecodeError as exc:
        raise ValueError(f"FEW_SHOT_EXAMPLE JSON is invalid: {exc}") from exc
    return _FEW_SHOT_RAW

FEW_SHOT_EXAMPLE: str = _validate_few_shot()


# ── LLMEngine ────────────────────────────────────────────────────────────────

class LLMEngine:
    """
    Local LLM wrapper for circuit JSON generation.

    Backends (tried in order):
      1. Ollama  — HTTP API, model auto-selected from available list
      2. llama_cpp — GGUF file loaded directly

    All blocking inference calls are dispatched to a thread-pool executor so
    the FastAPI async event loop is never stalled.
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.backend:       Optional[str] = None   # 'ollama' | 'llama_cpp'
        self.model:         object        = None
        self.ollama_model:  str           = ""
        self.model_path:    Path          = (
            Path(model_path) if model_path else MODELS_DIR / DEFAULT_GGUF_MODEL
        )

    # ── Loading ───────────────────────────────────────────────────────────────

    def load(self) -> bool:
        """Detect and initialise the best available backend. Returns True on success."""
        model = self._detect_ollama_model()
        if model:
            self.ollama_model = model
            self.backend = "ollama"
            logger.info("LLM backend: Ollama — model=%s", self.ollama_model)
            return True
        return self._load_llama_cpp()

    def _detect_ollama_model(self) -> Optional[str]:
        """
        Query Ollama for available models and return the best candidate.
        Returns None if Ollama is not reachable or has no models.
        """
        try:
            resp = requests.get(OLLAMA_TAGS_URL, timeout=3)
            resp.raise_for_status()
            available: list[str] = [m["name"] for m in resp.json().get("models", [])]
            logger.info("Ollama models available: %s", available)

            for candidate in OLLAMA_MODEL_CANDIDATES:
                if not candidate:          # skip empty strings
                    continue
                if candidate in available:
                    return candidate
                prefix = candidate.split(":")[0].lower()
                for avail in available:
                    if avail.lower().startswith(prefix):
                        return avail

            # Nothing preferred — fall back to whatever Ollama has
            if available:
                logger.warning("No preferred Ollama model; using first available: %s", available[0])
                return available[0]

        except requests.exceptions.ConnectionError:
            logger.info("Ollama not reachable at %s", OLLAMA_TAGS_URL)
        except Exception as exc:
            logger.warning("Ollama detection failed: %s", exc)
        return None

    def _load_llama_cpp(self) -> bool:
        try:
            from llama_cpp import Llama   # type: ignore[import]
        except ImportError:
            logger.warning("llama-cpp-python not installed — GGUF backend unavailable")
            return False

        if not self.model_path.exists():
            logger.warning("GGUF model not found: %s", self.model_path)
            return False

        logger.info("Loading GGUF model: %s", self.model_path)
        try:
            self.model = Llama(
                model_path=str(self.model_path),
                n_ctx=4096,
                n_threads=int(os.environ.get("LLAMA_THREADS", "8")),
                n_gpu_layers=int(os.environ.get("LLAMA_GPU_LAYERS", "0")),
                verbose=False,
            )
            self.backend = "llama_cpp"
            logger.info("GGUF LLM loaded successfully")
            return True
        except Exception as exc:
            logger.error("Failed to load GGUF model: %s", exc)
            return False

    # ── Synchronous raw generation (runs in thread) ───────────────────────────

    def _generate_sync(
        self,
        prompt:      str,
        max_tokens:  int  = LLM_MAX_TOKENS,
        temperature: float = LLM_BASE_TEMP,
        stop:        Optional[list[str]] = None,
    ) -> str:
        """
        Blocking inference.  Do NOT call from an async context directly;
        use generate_async() instead.
        """
        if self.backend == "ollama":
            return self._generate_ollama(prompt, max_tokens, temperature, stop)
        if self.backend == "llama_cpp":
            return self._generate_llama_cpp(prompt, max_tokens, temperature, stop)
        raise RuntimeError("No LLM backend loaded — call load() first")

    def _generate_ollama(
        self,
        prompt:      str,
        max_tokens:  int,
        temperature: float,
        stop:        Optional[list[str]],
    ) -> str:
        payload = {
            "model":  self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "stop": stop or ["</s>", "[/INST]", "USER:", "## YOUR TASK"],
            },
        }
        try:
            resp = requests.post(OLLAMA_API_URL, json=payload, timeout=OLLAMA_TIMEOUT)
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except requests.exceptions.Timeout:
            logger.error("Ollama request timed out after %d s", OLLAMA_TIMEOUT)
            return ""
        except Exception as exc:
            logger.error("Ollama generation failed: %s", exc)
            return ""

    def _generate_llama_cpp(
        self,
        prompt:      str,
        max_tokens:  int,
        temperature: float,
        stop:        Optional[list[str]],
    ) -> str:
        if self.model is None:
            raise RuntimeError("llama_cpp model not loaded")
        # NOTE: run inside a ThreadPoolExecutor future so we can enforce a
        # timeout — the llama_cpp call itself has no built-in timeout.
        future = _THREAD_POOL.submit(
            self.model,          # type: ignore[operator]
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop or ["</s>", "[/INST]"],
            echo=False,
        )
        try:
            output = future.result(timeout=LLAMA_CPP_TIMEOUT)
            return output["choices"][0]["text"].strip()
        except FuturesTimeoutError:
            logger.error("llama_cpp inference timed out after %d s", LLAMA_CPP_TIMEOUT)
            future.cancel()
            return ""
        except Exception as exc:
            logger.error("llama_cpp inference failed: %s", exc)
            return ""

    # ── Async entry-point ─────────────────────────────────────────────────────

    async def generate_async(
        self,
        prompt:      str,
        max_tokens:  int   = LLM_MAX_TOKENS,
        temperature: float = LLM_BASE_TEMP,
        stop:        Optional[list[str]] = None,
    ) -> str:
        """
        Non-blocking wrapper around _generate_sync.
        Runs the blocking call in the shared thread-pool so the FastAPI event
        loop is never stalled during Ollama HTTP or llama_cpp inference.
        """
        # get_running_loop() is the correct call inside an async function.
        # get_event_loop() is deprecated in Python 3.10+ when a loop is running
        # and will raise DeprecationWarning on 3.12+.
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _THREAD_POOL,
            lambda: self._generate_sync(prompt, max_tokens, temperature, stop),
        )

    # ── Circuit JSON generation ───────────────────────────────────────────────

    async def generate_circuit_json(
        self,
        user_prompt: str,
        retries:     int = LLM_RETRIES,
    ) -> dict:
        """
        Generate a validated CircuitData JSON dict from natural language.

        Retries up to `retries` times with increasing temperature.
        Returns an empty dict {} on total failure — never raises.
        """
        full_prompt = _build_circuit_prompt(user_prompt)

        for attempt in range(retries + 1):
            temp = LLM_BASE_TEMP + attempt * LLM_TEMP_STEP
            logger.info(
                "Circuit generation attempt %d/%d (temp=%.2f)",
                attempt + 1, retries + 1, temp,
            )
            try:
                t0  = time.perf_counter()
                raw = await self.generate_async(full_prompt, temperature=temp)
                elapsed = time.perf_counter() - t0
                logger.info("LLM responded in %.1f s (%d chars)", elapsed, len(raw))

                parsed = _extract_json(raw)
                if not parsed:
                    logger.warning("Attempt %d: no JSON found in response", attempt + 1)
                    continue

                errors = _validate_circuit_structure(parsed)
                if errors:
                    logger.warning("Attempt %d: validation errors: %s", attempt + 1, errors)
                    continue

                parsed = _fix_bypass_caps(parsed)
                parsed = _cross_validate_pins(parsed)

                logger.info(
                    "Valid circuit: %d components, %d nets",
                    len(parsed.get("components", [])),
                    len(parsed.get("connections", [])),
                )
                return parsed

            except Exception as exc:
                logger.error("Attempt %d raised exception: %s", attempt + 1, exc)

        logger.error("All %d circuit generation attempts failed", retries + 1)
        return {}

    # Sync shim for callers that haven't been migrated to async yet
    def generate_circuit_json_sync(self, user_prompt: str, retries: int = LLM_RETRIES) -> dict:
        """
        Synchronous wrapper around generate_circuit_json.
        Use only from non-async contexts (e.g. tests, CLI scripts).

        asyncio.run() creates a fresh event loop each call — safe and clean on
        Python 3.7+. If a loop is already running (Jupyter / async test runner)
        we spawn a worker thread that gets its own loop via asyncio.run().
        """
        try:
            asyncio.get_running_loop()
            # A loop is already running — can't call asyncio.run() here.
            # Submit to a worker thread that will create its own fresh loop.
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(
                    asyncio.run,
                    self.generate_circuit_json(user_prompt, retries),
                )
                return future.result(timeout=LLAMA_CPP_TIMEOUT + OLLAMA_TIMEOUT)
        except RuntimeError:
            # No running loop — safe to use asyncio.run() directly.
            pass
        try:
            return asyncio.run(self.generate_circuit_json(user_prompt, retries))
        except Exception as exc:
            logger.error("generate_circuit_json_sync failed: %s", exc)
            return {}


# ── Module-level helpers (pure functions, no self) ────────────────────────────

def _build_circuit_prompt(user_prompt: str) -> str:
    return (
        f"{CIRCUIT_SYSTEM_PROMPT}\n\n"
        f"## EXAMPLE\n{FEW_SHOT_EXAMPLE}\n\n"
        f"## YOUR TASK\nUSER: {user_prompt}\n\nASSISTANT: "
    )
def _extract_json(text: str) -> dict:
    """
    Extract the outermost complete JSON object from (potentially noisy) LLM output.

    Strategy:
      1. Strip markdown code fences.
      2. Walk characters tracking brace depth to find the full outer {…}.
      3. Try json.loads; on failure, try _repair_json.
      4. Fall back to parsing the entire stripped string.
    """
    if not text:
        return {}

    # Strip markdown fences (```json, ```, etc.)
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()

    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                candidate = text[start : i + 1]
                result = _try_parse(candidate)
                if result:
                    return result
                # Keep scanning — the first match might be a prefix fragment

    # Last resort: parse the whole text
    return _try_parse(text)


def _try_parse(text: str) -> dict:
    """Try json.loads, then _repair_json; return {} on failure."""
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return _repair_json(text)


def _repair_json(text: str) -> dict:
    """
    Attempt to fix common JSON errors produced by language models:
      - Trailing commas before } or ]
      - Single-line  // comments
      - Block         /* ... */ comments
      - Unquoted object keys  (key: value → "key": value)
      - Single-quoted strings ('value' → "value") — simple cases
    """
    # Block comments
    fixed = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    # Single-line comments
    fixed = re.sub(r"//[^\n]*", "", fixed)
    # Trailing commas
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
    # Unquoted keys: word: → "word":
    fixed = re.sub(r'(?<=[{,\s])([A-Za-z_][A-Za-z0-9_]*)\s*:', r'"\1":', fixed)
    # Single-quoted strings → double-quoted (naive — no nested single quotes)
    fixed = re.sub(r"'([^']*)'", r'"\1"', fixed)
    try:
        obj = json.loads(fixed)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return {}


def _validate_circuit_structure(data: dict) -> list[str]:
    """
    Return a list of structural error strings (empty = valid).

    Checks every component, not just the first one.
    """
    errors: list[str] = []

    if not isinstance(data, dict):
        errors.append("root is not a dict")
        return errors

    components = data.get("components")
    connections = data.get("connections")

    if not isinstance(components, list) or len(components) == 0:
        errors.append("'components' missing or empty")
    else:
        for idx, comp in enumerate(components):
            if not isinstance(comp, dict):
                errors.append(f"components[{idx}] is not a dict")
                continue
            for field in ("ref", "part", "pins"):
                if field not in comp:
                    errors.append(f"components[{idx}] missing field '{field}'")
            if "pins" in comp and not isinstance(comp["pins"], list):
                errors.append(f"components[{idx}].pins is not a list")

    if not isinstance(connections, list) or len(connections) == 0:
        errors.append("'connections' missing or empty")
    else:
        for idx, conn in enumerate(connections):
            if not isinstance(conn, dict):
                errors.append(f"connections[{idx}] is not a dict")
                continue
            if "net" not in conn:
                errors.append(f"connections[{idx}] missing 'net'")
            pins = conn.get("pins")
            if not isinstance(pins, list) or len(pins) < 2:
                errors.append(
                    f"connections[{idx}] net='{conn.get('net','?')}' "
                    "must have >= 2 pins"
                )

    return errors


def _cross_validate_pins(data: dict) -> dict:
    """
    Check that every pin string "REF.PIN" in connections references a real
    component ref and a pin number defined in that component's pins list.

    Removes invalid pin strings (with a warning) rather than crashing.
    Does NOT mutate the input — works on a deep copy.
    """
    data = copy.deepcopy(data)

    # Build ref → set of valid pin numbers
    valid_pins: dict[str, set[str]] = {}
    for comp in data.get("components", []):
        ref = comp.get("ref", "")
        pins = {str(p.get("number", "")) for p in comp.get("pins", []) if isinstance(p, dict)}
        valid_pins[ref] = pins

    clean_connections: list[dict] = []
    for conn in data.get("connections", []):
        clean: list[str] = []
        for pin_str in conn.get("pins", []):
            if not isinstance(pin_str, str) or "." not in pin_str:
                logger.warning("Dropping invalid pin reference: %r", pin_str)
                continue
            ref, pin = pin_str.split(".", 1)
            if ref not in valid_pins:
                logger.warning(
                    "Net '%s': component '%s' not in components list — dropping pin %s",
                    conn.get("net"), ref, pin_str,
                )
                continue
            if pin not in valid_pins[ref]:
                logger.warning(
                    "Net '%s': pin %s not defined on component '%s' (valid: %s) — dropping",
                    conn.get("net"), pin_str, ref, sorted(valid_pins[ref]),
                )
                continue
            clean.append(pin_str)

        if len(clean) >= 2:
            clean_connections.append({**conn, "pins": clean})
        elif clean:
            logger.warning(
                "Net '%s' reduced to 1 pin after validation — removed from netlist",
                conn.get("net"),
            )

    data["connections"] = clean_connections
    return data


def _fix_bypass_caps(data: dict) -> dict:
    """
    Post-processing: ensure bypass / decoupling capacitors connect ONLY to
    power rails (VCC + GND), never in the signal path.

    NOTE: operates on a deep copy — the caller's dict is never mutated.
    This prevents the template_cache in AppState from being corrupted when
    the same template is reused across multiple /generate requests.
    """
    data        = copy.deepcopy(data)
    components  = data.get("components", [])
    connections = data.get("connections", [])  # this is now a fresh list

    # Identify bypass caps
    bypass_refs: set[str] = set()
    for comp in components:
        if (comp.get("part", "").upper() == "C"
                and any(kw in (comp.get("description") or "").lower()
                        for kw in _BYPASS_KEYWORDS)):
            bypass_refs.add(comp["ref"])

    if not bypass_refs:
        return data

    # Determine which VCC / GND net names are actually used
    vcc_net = "VCC"
    gnd_net = "GND"
    for conn in connections:
        net_up = conn.get("net", "").upper().replace(" ", "")
        if net_up in {n.upper() for n in _POWER_NETS}:
            vcc_net = conn["net"]
        elif net_up in {n.upper() for n in _GROUND_NETS}:
            gnd_net = conn["net"]

    changed = False

    for ref in bypass_refs:
        pin1, pin2 = f"{ref}.1", f"{ref}.2"

        # Remove these pins from any non-power net
        for conn in connections:
            if conn.get("net", "").upper() in _ALL_POWER_NETS_UPPER:
                continue
            before = len(conn.get("pins", []))
            conn["pins"] = [p for p in conn.get("pins", []) if p not in (pin1, pin2)]
            if len(conn["pins"]) != before:
                logger.info(
                    "Bypass cap fix: removed %s from signal net '%s'",
                    ref, conn["net"],
                )
                changed = True

        # Ensure pin 1 is in VCC net
        vcc_conn = next((c for c in connections if c.get("net") == vcc_net), None)
        if vcc_conn is None:
            connections.append({"net": vcc_net, "pins": [pin1]})
            changed = True
        elif pin1 not in vcc_conn.get("pins", []):
            vcc_conn.setdefault("pins", []).append(pin1)
            changed = True

        # Ensure pin 2 is in GND net
        gnd_conn = next((c for c in connections if c.get("net") == gnd_net), None)
        if gnd_conn is None:
            connections.append({"net": gnd_net, "pins": [pin2]})
            changed = True
        elif pin2 not in gnd_conn.get("pins", []):
            gnd_conn.setdefault("pins", []).append(pin2)
            changed = True

    if changed:
        logger.info("Bypass cap post-processing applied to: %s", bypass_refs)
        # Remove nets left with 0 or 1 pins after stripping bypass pins
        data["connections"] = [
            c for c in connections
            if len(c.get("pins", [])) >= 2
            or c.get("net", "").upper() in _ALL_POWER_NETS_UPPER
        ]

    return data


# ── Factory ───────────────────────────────────────────────────────────────────

def load_llm() -> Optional[LLMEngine]:
    """Load and return an LLMEngine, or None if no backend is available."""
    engine = LLMEngine()
    return engine if engine.load() else None