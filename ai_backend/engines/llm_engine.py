from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


_CIRCUIT_JSON_SYSTEM_PROMPT = (
    "You are an expert PCB circuit generator. Return ONLY a valid JSON object.\n"
    "Schema (follow EXACTLY):\n"
    "{\n"
    '  "description": "<human readable circuit name>",\n'
    '  "components": [\n'
    "    {\n"
    '      "ref": "R1",\n'
    '      "lib": "Device",\n'
    '      "part": "R",\n'
    '      "value": "10k",\n'
    '      "footprint": "Resistor_SMD:R_0805_2012Metric",\n'
    '      "description": "Pull-up resistor",\n'
    '      "pins": [{"number": "1", "name": "1"}, {"number": "2", "name": "2"}],\n'
    '      "x": 10.0, "y": 10.0, "rotation": 0.0, "layer": "top"\n'
    "    }\n"
    "  ],\n"
    '  "connections": [\n'
    "    {\n"
    '      "net": "VOUT",\n'
    '      "pins": ["U1.3", "C2.1"]\n'
    "    }\n"
    "  ],\n"
    '  "board_width": 100.0,\n'
    '  "board_height": 80.0\n'
    "}\n\n"
    "RULES:\n"
    "- Every component MUST have: ref, value, footprint.\n"
    "- Include 'lib' (KiCad library name) and 'part' (symbol name) when known.\n"
    "- Include 'pins' array with {number, name} for each pin.\n"
    "- Common footprints: Resistor_SMD:R_0805_2012Metric, Capacitor_SMD:C_0805_2012Metric, "
    "LED_SMD:LED_0805_2012Metric, Package_TO_SOT_SMD:SOT-23, Package_DIP:DIP-8_W7.62mm.\n"
    "- Each connection.pins entry is a string 'REF.PIN_NUMBER' (e.g. 'R1.1', 'U1.3').\n"
    "- Power nets should be named: VCC, +3V3, +5V, GND.\n"
    "- NO markdown fences, NO comments, NO extra keys."
)

_MAX_RETRIES = 2
_RETRY_BACKOFF_BASE = 1.5  # seconds
_TRANSIENT_ERRORS = (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout)


def _default_model_dir() -> Path:
    engines_dir = Path(__file__).resolve().parent
    return (engines_dir.parent.parent / "models" / "deepseek-coder-6.7B-instruct-GGUF").resolve()


def _strip_markdown_fences(text: str) -> str:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    return (match.group(1) if match else text).strip()


class _LLMStub:
    """Returned when llama-cpp-python is unavailable in the runtime environment."""

    def __init__(self, reason: str) -> None:
        self._reason = reason

    async def generate_async(self, prompt: str, max_tokens: int, temperature: float) -> str:
        raise NotImplementedError(
            "llama-cpp-python is not installed; install it to enable local GGUF inference. "
            f"Reason: {self._reason}"
        )

    async def generate_circuit_json(self, prompt: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError(
            "llama-cpp-python is not installed; install it to enable local GGUF inference. "
            f"Reason: {self._reason}"
        )


class LLMEngine:
    """DeepSeek GGUF wrapper backed by llama_cpp.Llama."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        *,
        n_ctx: int = 4096,
        n_gpu_layers: Optional[int] = None,
        llama_cls: Optional[Any] = None,
    ) -> None:
        self.model_path = Path(model_path) if model_path else Path(
            os.environ.get("LOCAL_LLM_PATH", str(_default_model_dir()))
        )
        self.n_ctx = n_ctx
        self.n_gpu_layers = (
            n_gpu_layers if n_gpu_layers is not None else int(os.environ.get("LLM_GPU_LAYERS", "0"))
        )

        if llama_cls is None:
            from llama_cpp import Llama as llama_cls  # type: ignore

        resolved_path = self._resolve_model_file(self.model_path)
        self._llama = llama_cls(
            model_path=str(resolved_path),
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            verbose=False,
        )

    @staticmethod
    def _resolve_model_file(path: Path) -> Path:
        if path.is_file():
            return path
        if path.is_dir():
            candidates = sorted(path.glob("*.gguf"))
            if candidates:
                return candidates[0]
        raise FileNotFoundError(f"GGUF model not found at {path}")

    async def generate_async(self, prompt: str, max_tokens: int, temperature: float) -> str:
        def _run() -> str:
            out = self._llama(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                echo=False,
            )
            choices = out.get("choices", []) if isinstance(out, dict) else []
            if not choices:
                return ""
            return str(choices[0].get("text", "")).strip()

        return await asyncio.to_thread(_run)

    async def generate_circuit_json(self, prompt: str) -> Optional[Dict[str, Any]]:
        full_prompt = f"{_CIRCUIT_JSON_SYSTEM_PROMPT}\n\nUser request: {prompt}\n\nJSON:"
        raw = await self.generate_async(full_prompt, max_tokens=2400, temperature=0.2)
        parsed = _try_parse_json_dict(raw)
        if parsed is not None and _validate_circuit_json(parsed):
            return parsed

        repair_prompt = (
            f"{_CIRCUIT_JSON_SYSTEM_PROMPT}\n\n"
            "The previous output was invalid JSON. Return corrected JSON only.\n\n"
            f"User request: {prompt}\n\nJSON:"
        )
        repaired = await self.generate_async(repair_prompt, max_tokens=2400, temperature=0.1)
        result = _try_parse_json_dict(repaired)
        if result is not None and _validate_circuit_json(result):
            return result
        logger.warning("LLM (local) circuit JSON failed validation after repair")
        return result  # return best-effort even if validation fails


class OllamaEngine:
    """Ollama HTTP client backend, compatible with local or remote cloud endpoints."""

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_s: Optional[float] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip("/")
        self.timeout_s = timeout_s if timeout_s is not None else float(os.environ.get("OLLAMA_TIMEOUT_S", "300"))
        self.api_key = api_key or os.environ.get("OLLAMA_API_KEY", "").strip() or None
        self.model = model or os.environ.get("OLLAMA_MODEL", "").strip() or self._resolve_default_model()
        if not self.model:
            raise RuntimeError(
                "No Ollama model configured or discovered. "
                "Set OLLAMA_MODEL or ensure /api/tags returns installed models."
            )

    def _resolve_default_model(self) -> str:
        headers: Dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"{self.base_url}/api/tags", headers=headers)
                resp.raise_for_status()
                payload = resp.json() or {}
                models = payload.get("models") or []
                if models:
                    model_names: list[str] = []
                    for item in models:
                        if isinstance(item, dict):
                            name = str(item.get("name", "")).strip()
                        else:
                            name = str(item).strip()
                        if name:
                            model_names.append(name)

                    preferred = os.environ.get("OLLAMA_PREFERRED_MODEL", "qwen2.5-coder:7b").strip()
                    if preferred in model_names:
                        return preferred

                    return model_names[0] if model_names else ""
        except Exception:
            pass
        return ""

    async def generate_async(self, prompt: str, max_tokens: int, temperature: float) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }
        headers: Dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        timeout = httpx.Timeout(self.timeout_s, connect=10.0)
        last_exc: Optional[Exception] = None

        for attempt in range(_MAX_RETRIES + 1):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.post(
                        f"{self.base_url}/api/generate", json=payload, headers=headers
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    return str(data.get("response", "")).strip()
            except _TRANSIENT_ERRORS as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    wait = _RETRY_BACKOFF_BASE * (2 ** attempt)
                    logger.warning(
                        "Ollama request failed (attempt %d/%d): %s — retrying in %.1fs",
                        attempt + 1, _MAX_RETRIES + 1, exc, wait,
                    )
                    await asyncio.sleep(wait)
                    continue
                raise RuntimeError(
                    f"Ollama request failed after {_MAX_RETRIES + 1} attempts for model "
                    f"'{self.model}': {exc}"
                ) from exc

        raise RuntimeError(f"Ollama generate failed: {last_exc}")  # unreachable safeguard

    async def generate_circuit_json(self, prompt: str) -> Optional[Dict[str, Any]]:
        full_prompt = f"{_CIRCUIT_JSON_SYSTEM_PROMPT}\n\nUser request: {prompt}\n\nJSON:"
        raw = await self.generate_async(full_prompt, max_tokens=2400, temperature=0.2)
        parsed = _try_parse_json_dict(raw)
        if parsed is not None and _validate_circuit_json(parsed):
            return parsed

        repair_prompt = (
            f"{_CIRCUIT_JSON_SYSTEM_PROMPT}\n\n"
            "The previous output was invalid JSON. Return corrected JSON only.\n\n"
            f"User request: {prompt}\n\nJSON:"
        )
        repaired = await self.generate_async(repair_prompt, max_tokens=2400, temperature=0.1)
        result = _try_parse_json_dict(repaired)
        if result is not None and _validate_circuit_json(result):
            return result
        logger.warning("Ollama circuit JSON failed validation after repair")
        return result


def _validate_circuit_json(data: Dict[str, Any]) -> bool:
    """Check that parsed JSON has the minimum required structure."""
    if not isinstance(data.get("components"), list):
        logger.debug("Validation: missing or invalid 'components'")
        return False
    if not isinstance(data.get("connections"), list):
        logger.debug("Validation: missing or invalid 'connections'")
        return False
    # Must have at least one component
    if len(data["components"]) == 0:
        logger.debug("Validation: empty components list")
        return False
    # Each component needs at minimum a ref and value
    for comp in data["components"]:
        if not isinstance(comp, dict):
            return False
        if not comp.get("ref"):
            logger.debug("Validation: component missing 'ref'")
            return False
    return True


def _try_parse_json_dict(text: str) -> Optional[Dict[str, Any]]:
    cleaned = _strip_markdown_fences(text)

    # 1) Direct parse (fast path)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # 2) Recover first valid JSON object from mixed text output.
    decoder = json.JSONDecoder()
    for i, ch in enumerate(cleaned):
        if ch != "{":
            continue
        try:
            obj, _end = decoder.raw_decode(cleaned[i:])
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue

    return None


def _looks_like_ollama_available(base_url: str) -> bool:
    try:
        req = urllib.request.Request(f"{base_url.rstrip('/')}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


def load_llm() -> Any:
    """Factory called by ai_server.py lifespan; returns engine or stub."""
    backend = os.environ.get("LLM_BACKEND", "").strip().lower()
    if backend == "ollama" or os.environ.get("OLLAMA_MODEL") or os.environ.get("OLLAMA_BASE_URL"):
        try:
            return OllamaEngine()
        except Exception:
            return None

    try:
        return LLMEngine()
    except ImportError as exc:
        try:
            base_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
            if _looks_like_ollama_available(base_url):
                return OllamaEngine(base_url=base_url)
        except Exception:
            pass
        return _LLMStub(str(exc))
    except Exception:
        # Keep server alive; generation path will fallback if model is not available.
        return None
