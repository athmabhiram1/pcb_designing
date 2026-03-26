from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple


DEFAULT_SKIDL_BLOCKED: List[re.Pattern[str]] = [
    re.compile(r"\bimport\s+(os|sys|subprocess|socket|shutil|pathlib|ctypes|importlib)\b"),
    re.compile(r"\bfrom\s+(os|sys|subprocess|socket|shutil|pathlib|ctypes|importlib)\s+import\b"),
    re.compile(r"\b(__import__|eval|exec|open|compile|input|breakpoint)\s*\("),
    re.compile(r"\b(os|sys|subprocess|socket|shutil)\s*\."),
    re.compile(r"importlib\s*\."),
]


@dataclass(frozen=True)
class SkidlRetryConfig:
    system_prompt: str
    max_retries: int = 3


def extract_python_code_block(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    code = (match.group(1) if match else text).strip()

    # Remove common tokenizer artifacts from some local models.
    code = re.sub(r"<\|[^\n>]{1,120}\|>", "", code)
    code = re.sub(r"<｜[^\n>]{1,120}｜>", "", code)
    code = code.replace("<s>", "").replace("</s>", "")

    code = "".join(ch for ch in code if (ch.isprintable() or ch in "\n\t"))
    return code.strip()


def screen_skidl_code(code: str, blocked_patterns: List[re.Pattern[str]] = DEFAULT_SKIDL_BLOCKED) -> Optional[str]:
    if not code.strip():
        return "empty code"
    if len(code) > 30_000:
        return "code too large (> 30 000 chars)"
    for pat in blocked_patterns:
        if pat.search(code):
            return f"blocked pattern: {pat.pattern}"
    if "from skidl import" not in code and "import skidl" not in code:
        return "missing SKiDL import"
    return None


def default_kicad_symbol_dir() -> Optional[str]:
    for env_key in ("KICAD_SYMBOL_DIR", "KICAD9_SYMBOL_DIR", "KICAD8_SYMBOL_DIR", "KICAD7_SYMBOL_DIR", "KICAD6_SYMBOL_DIR"):
        val = os.environ.get(env_key)
        if val and os.path.isdir(val):
            return val

    if os.name == "nt":
        pf = os.environ.get("ProgramFiles", r"C:\Program Files")
        candidates = [os.path.join(pf, "KiCad", "share", "kicad", "symbols")]
        candidates.extend(
            os.path.join(pf, "KiCad", version, "share", "kicad", "symbols")
            for version in ("9.0", "8.0", "7.0")
        )
    else:
        candidates = ["/usr/share/kicad/symbols", "/usr/local/share/kicad/symbols"]

    return next((path for path in candidates if os.path.isdir(path)), None)


def score_template(prompt_lower: str, template_keywords: List[Tuple[List[str], str, int]]) -> Tuple[Optional[str], int]:
    best_name: Optional[str] = None
    best_score = 0
    for keywords, name, weight in template_keywords:
        score = sum(weight for kw in keywords if kw in prompt_lower)
        if score > best_score:
            best_score, best_name = score, name
    return best_name, best_score


async def generate_via_skidl_retry(
    prompt: str,
    request_id: str,
    config: SkidlRetryConfig,
    llm_generate: Callable[[str, int, float], Awaitable[str]],
    execute_skidl_code: Callable[[str, str, str], Tuple[Optional[Dict[str, Any]], List[str]]],
    symbol_dir_resolver: Callable[[], Optional[str]],
) -> Tuple[Optional[Dict[str, Any]], List[str], Optional[str]]:
    """LLM -> SKiDL code -> execute -> circuit_data with bounded retries."""
    if symbol_dir_resolver() is None:
        return None, [
            "SKiDL skipped: KiCad symbol directory not found. "
            "Set KICAD_SYMBOL_DIR (or KICAD9_SYMBOL_DIR)."
        ], None

    warnings: List[str] = []
    error_context = ""
    last_code: Optional[str] = None

    for attempt in range(config.max_retries):
        payload = f"Generate SKiDL for:\n{prompt}\n"
        if error_context:
            payload += f"\nFix these errors from the last attempt:\n{error_context}\n"

        raw = await llm_generate(
            f"{config.system_prompt}\n\n{payload}",
            1800,
            0.2 + 0.05 * attempt,
        )
        code = extract_python_code_block(raw)
        last_code = code

        circuit_data, run_warnings = await asyncio.to_thread(
            execute_skidl_code,
            code,
            f"{request_id}_{attempt}",
            prompt,
        )
        if circuit_data:
            if attempt > 0:
                warnings.append(f"SKiDL succeeded on attempt {attempt + 1}")
            warnings.extend(run_warnings)
            return circuit_data, warnings, code

        joined = "; ".join(run_warnings) or "unknown failure"
        warnings.append(f"SKiDL attempt {attempt + 1}: {joined[:220]}")

        if "symbol directory not found" in joined.lower() or "code rejected" in joined.lower():
            break
        error_context = joined

    warnings.append("SKiDL retries exhausted – falling back")
    return None, warnings, last_code
