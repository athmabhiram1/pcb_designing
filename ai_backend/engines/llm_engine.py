"""
LLM Engine — Local Large Language Model Inference for Circuit Generation.

Supports Ollama API (priority) and llama-cpp-python (fallback).
Optimized for structured JSON circuit output using the CircuitData schema.
"""
import os
import json
import re
import logging
import time
import requests
from typing import Optional, Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "models")
DEFAULT_GGUF_MODEL = "deepseek-coder-6.7b-instruct.Q5_K_M.gguf"

# Ollama model candidates — tried in order until one is found
OLLAMA_MODEL_CANDIDATES = [
    os.environ.get("OLLAMA_MODEL", ""),
    "deepseek-coder:6.7b",
    "deepseek-coder:latest",
    "deepseek-coder-v2:latest",
    "codellama:7b",
    "codellama:latest",
    "llama3:latest",
    "mistral:latest",
]
OLLAMA_API_URL = os.environ.get("OLLAMA_API_URL", "http://localhost:11434/api/generate")
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"

# ---------------------------------------------------------------------------
# System prompt — the core of reliable circuit JSON generation
# ---------------------------------------------------------------------------
CIRCUIT_SYSTEM_PROMPT = """You are an expert electronic circuit designer. Your ONLY job is to output a single valid JSON object describing a circuit.

## STRICT RULES
1. Output ONLY a JSON object. No markdown, no explanations, no code fences.
2. Every component MUST have: ref, lib, part, value, footprint, pins.
3. Every connection MUST reference pins that exist on the components.
4. Pin format in connections: "REF.PIN_NUMBER" (e.g. "R1.1", "U1.8").
5. Use real KiCad library names: Device, Timer, Connector, Regulator_Linear, etc.
6. Use real footprints: Resistor_SMD:R_0805_2012Metric, Capacitor_SMD:C_0805_2012Metric, etc.
7. Include power nets VCC and GND connecting to appropriate pins.
8. Bypass/decoupling capacitors MUST connect ONLY to power rails. Pin 1 goes to VCC (or the supply rail) and Pin 2 goes to GND. NEVER include a bypass cap in the signal path or between two signal nodes.
9. Double-check every connection: a bypass cap described as 'decoupling' or 'bypass' must appear in the VCC net pins list AND the GND net pins list — nowhere else.

## JSON SCHEMA
{
  "description": "<circuit description>",
  "components": [
    {
      "ref": "<designator>",
      "lib": "<KiCad library>",
      "part": "<part name>",
      "value": "<value>",
      "footprint": "<KiCad footprint>",
      "description": "<purpose>",
      "pins": [
        {"number": "<pin#>", "name": "<function>"}
      ]
    }
  ],
  "connections": [
    {"net": "<net name>", "pins": ["REF.PIN", "REF.PIN"]}
  ]
}"""

# ---------------------------------------------------------------------------
# Few-shot example — one complete circuit for the LLM to learn from
# ---------------------------------------------------------------------------
FEW_SHOT_EXAMPLE = """\
USER: Simple LED circuit with current limiting resistor and bypass cap

ASSISTANT: {
  "description": "Simple LED indicator circuit: 330 ohm resistor limits current, red LED, 100nF bypass cap on power rail",
  "components": [
    {
      "ref": "R1",
      "lib": "Device",
      "part": "R",
      "value": "330",
      "footprint": "Resistor_SMD:R_0805_2012Metric",
      "description": "Current limiting resistor for LED (330 ohm limits to ~13mA at 5V)",
      "pins": [
        {"number": "1", "name": "1"},
        {"number": "2", "name": "2"}
      ]
    },
    {
      "ref": "D1",
      "lib": "Device",
      "part": "LED",
      "value": "Red",
      "footprint": "LED_SMD:LED_0805_2012Metric",
      "description": "Red indicator LED",
      "pins": [
        {"number": "1", "name": "A"},
        {"number": "2", "name": "K"}
      ]
    },
    {
      "ref": "C1",
      "lib": "Device",
      "part": "C",
      "value": "100nF",
      "footprint": "Capacitor_SMD:C_0402_1005Metric",
      "description": "Power supply bypass capacitor (VCC to GND, NOT in signal path)",
      "pins": [
        {"number": "1", "name": "+"},
        {"number": "2", "name": "-"}
      ]
    }
  ],
  "connections": [
    {"net": "VCC", "pins": ["R1.1", "C1.1"]},
    {"net": "LED_ANODE", "pins": ["R1.2", "D1.1"]},
    {"net": "GND", "pins": ["D1.2", "C1.2"]}
  ]
}"""


class LLMEngine:
    """Wrapper for local LLM inference (Ollama or GGUF)."""

    def __init__(self, model_path: Optional[str] = None):
        self.backend: Optional[str] = None  # 'ollama' or 'llama_cpp'
        self.model = None
        self.ollama_model: str = ""
        self.model_path = model_path or os.path.join(MODELS_DIR, DEFAULT_GGUF_MODEL)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> bool:
        """Load the LLM provider. Returns True on success."""
        model = self._detect_ollama_model()
        if model:
            self.ollama_model = model
            logger.info(f"Connected to Ollama — model: {self.ollama_model}")
            self.backend = "ollama"
            return True
        return self._load_llama_cpp()

    def _detect_ollama_model(self) -> Optional[str]:
        """Check Ollama is running and find the best available model."""
        try:
            r = requests.get(OLLAMA_TAGS_URL, timeout=3)
            if r.status_code != 200:
                return None
            available = [m["name"] for m in r.json().get("models", [])]
            logger.info(f"Ollama models available: {available}")
            # Try candidates in order
            for candidate in OLLAMA_MODEL_CANDIDATES:
                if not candidate:
                    continue
                # Exact match
                if candidate in available:
                    return candidate
                # Prefix match (e.g. "deepseek-coder:6.7b" matches "deepseek-coder:6.7b-instruct-q4_k_m")
                prefix = candidate.split(":")[0].lower()
                for avail in available:
                    if avail.lower().startswith(prefix):
                        return avail
            # If anything at all is available, use the first one
            if available:
                logger.warning(f"No preferred model found, using first available: {available[0]}")
                return available[0]
        except requests.exceptions.ConnectionError:
            pass
        except Exception as e:
            logger.warning(f"Ollama detection failed: {e}")
        return None

    def _load_llama_cpp(self) -> bool:
        try:
            from llama_cpp import Llama

            if not os.path.exists(self.model_path):
                logger.warning(f"GGUF model not found: {self.model_path}")
                return False

            logger.info(f"Loading GGUF from {self.model_path}…")
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=4096,
                n_threads=8,
                n_gpu_layers=0,
                verbose=False,
            )
            self.backend = "llama_cpp"
            logger.info("GGUF LLM loaded successfully")
            return True

        except ImportError:
            logger.warning("llama-cpp-python not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to load GGUF LLM: {e}")
            return False

    # ------------------------------------------------------------------
    # Raw generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.1,
        stop: list[str] | None = None,
    ) -> str:
        """Generate raw text from prompt."""
        if self.backend == "ollama":
            return self._generate_ollama(prompt, max_tokens, temperature, stop)
        elif self.backend == "llama_cpp":
            return self._generate_llama_cpp(prompt, max_tokens, temperature, stop)
        raise RuntimeError("No LLM backend loaded — call load() first")

    def _generate_ollama(
        self, prompt: str, max_tokens: int, temperature: float, stop: list[str] | None
    ) -> str:
        try:
            payload = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                    "stop": stop or ["</s>", "[/INST]"],
                },
            }
            r = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
            r.raise_for_status()
            return r.json().get("response", "").strip()
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return ""

    def _generate_llama_cpp(
        self, prompt: str, max_tokens: int, temperature: float, stop: list[str] | None
    ) -> str:
        if self.model is None:
            raise RuntimeError("Model not loaded")
        output = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop or ["</s>", "[/INST]"],
            echo=False,
        )
        return output["choices"][0]["text"].strip()

    # ------------------------------------------------------------------
    # Circuit JSON generation (the key method)
    # ------------------------------------------------------------------

    def generate_circuit_json(self, user_prompt: str, retries: int = 2) -> dict:
        """
        Generate a structured circuit JSON from a user's natural language prompt.

        Uses a circuit-focused system prompt and few-shot example to maximise
        the chance of getting valid, parseable JSON on the first try.  On
        failure it retries with slightly higher temperature.

        Returns:
            dict — parsed circuit data, or empty dict {} on total failure.
        """
        full_prompt = self._build_circuit_prompt(user_prompt)

        for attempt in range(retries + 1):
            temp = 0.1 + (attempt * 0.15)  # 0.1 → 0.25 → 0.4
            logger.info(
                f"Circuit generation attempt {attempt + 1}/{retries + 1} "
                f"(temp={temp:.2f})"
            )

            t0 = time.time()
            raw = self.generate(full_prompt, max_tokens=3072, temperature=temp)
            elapsed = time.time() - t0
            logger.info(f"LLM responded in {elapsed:.1f}s ({len(raw)} chars)")

            parsed = self._extract_json(raw)
            if parsed and self._validate_circuit_structure(parsed):
                parsed = self._fix_bypass_caps(parsed)
                logger.info(
                    f"Valid circuit JSON: {len(parsed.get('components', []))} "
                    f"components, {len(parsed.get('connections', []))} nets"
                )
                return parsed

            logger.warning(f"Attempt {attempt + 1} produced invalid JSON, retrying…")

        logger.error("All circuit generation attempts failed")
        return {}

    def _build_circuit_prompt(self, user_prompt: str) -> str:
        """Assemble the full prompt with system instructions + few-shot + user request."""
        return (
            f"{CIRCUIT_SYSTEM_PROMPT}\n\n"
            f"## EXAMPLE\n{FEW_SHOT_EXAMPLE}\n\n"
            f"## YOUR TASK\nUSER: {user_prompt}\n\nASSISTANT: "
        )

    # ------------------------------------------------------------------
    # JSON extraction & validation
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_json(text: str) -> dict:
        """Extract a JSON object from potentially noisy LLM output."""
        if not text:
            return {}

        # Strip markdown code fences
        text = re.sub(r"```(?:json)?", "", text).strip()

        # Strategy 1: find the outermost { … }
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
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        # Try fixing common LLM mistakes
                        return LLMEngine._try_fix_json(candidate)

        # Strategy 2: direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        return {}

    @staticmethod
    def _try_fix_json(text: str) -> dict:
        """Attempt to fix common JSON errors from LLMs."""
        # Remove trailing commas before } or ]
        fixed = re.sub(r",\s*([}\]])", r"\1", text)
        # Remove single-line comments
        fixed = re.sub(r"//.*$", "", fixed, flags=re.MULTILINE)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _validate_circuit_structure(data: dict) -> bool:
        """Quick structural validation — does this look like a circuit?"""
        if not isinstance(data, dict):
            return False
        components = data.get("components")
        connections = data.get("connections")
        if not isinstance(components, list) or len(components) == 0:
            return False
        if not isinstance(connections, list) or len(connections) == 0:
            return False
        # Check at least the first component has ref + part
        first = components[0]
        if not isinstance(first, dict):
            return False
        if "ref" not in first or "part" not in first:
            return False
        return True

    @staticmethod
    def _fix_bypass_caps(data: dict) -> dict:
        """
        Post-processing: ensure bypass/decoupling capacitors connect only to
        VCC and GND nets, never in the signal path.

        If a cap is described as bypass/decoupling/filter but one of its pins
        appears in a non-power net, this method moves that pin into the correct
        power net (VCC for pin 1, GND for pin 2).
        """
        bypass_keywords = ("bypass", "decoupling", "decouple", "filter", "power supply")
        power_names = {"GND", "VCC", "VSS", "VDD", "+5V", "+3V3", "+3.3V",
                       "+12V", "AVCC", "AGND", "DVCC", "DGND"}

        components = data.get("components", [])
        connections = data.get("connections", [])

        # Identify bypass caps by description
        bypass_refs: set[str] = set()
        for comp in components:
            desc = (comp.get("description") or "").lower()
            part = (comp.get("part") or "").upper()
            if part == "C" and any(kw in desc for kw in bypass_keywords):
                bypass_refs.add(comp["ref"])

        if not bypass_refs:
            return data

        # Find what VCC/GND net names are used in the circuit
        vcc_net = "VCC"
        gnd_net = "GND"
        for conn in connections:
            net_upper = conn.get("net", "").upper()
            if net_upper in ("VCC", "VDD", "+5V", "+3V3", "+3.3V", "+12V", "AVCC"):
                vcc_net = conn["net"]
            elif net_upper in ("GND", "VSS", "AGND", "DGND"):
                gnd_net = conn["net"]

        # Re-wire bypass caps: pin 1 → VCC net, pin 2 → GND net
        changed = False
        for ref in bypass_refs:
            pin1_ref = f"{ref}.1"
            pin2_ref = f"{ref}.2"

            # Remove these pins from any non-power net
            for conn in connections:
                net_upper = conn.get("net", "").upper()
                is_power = net_upper in {v.upper() for v in power_names}
                if not is_power and "pins" in conn:
                    orig_len = len(conn["pins"])
                    conn["pins"] = [p for p in conn["pins"]
                                    if p not in (pin1_ref, pin2_ref)]
                    if len(conn["pins"]) != orig_len:
                        changed = True
                        logger.info(f"Removed {ref} from signal net '{conn['net']}' (bypass cap fix)")

            # Ensure pin 1 is in VCC net
            vcc_conn = next((c for c in connections if c.get("net") == vcc_net), None)
            if vcc_conn is None:
                connections.append({"net": vcc_net, "pins": [pin1_ref]})
                changed = True
            elif pin1_ref not in vcc_conn.get("pins", []):
                vcc_conn.setdefault("pins", []).append(pin1_ref)
                changed = True

            # Ensure pin 2 is in GND net
            gnd_conn = next((c for c in connections if c.get("net") == gnd_net), None)
            if gnd_conn is None:
                connections.append({"net": gnd_net, "pins": [pin2_ref]})
                changed = True
            elif pin2_ref not in gnd_conn.get("pins", []):
                gnd_conn.setdefault("pins", []).append(pin2_ref)
                changed = True

        if changed:
            logger.info(f"Bypass cap post-processing applied to: {bypass_refs}")
            # Remove any now-empty non-power connections
            connections[:] = [c for c in connections
                               if c.get("pins") or c.get("net", "").upper() in
                               {v.upper() for v in power_names}]
            data["connections"] = connections

        return data


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def load_llm() -> Optional[LLMEngine]:
    """Factory function to load the LLM engine."""
    engine = LLMEngine()
    if engine.load():
        return engine
    return None
