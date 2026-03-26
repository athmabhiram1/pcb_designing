from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


@dataclass
class OrchestratorResult:
    board: Optional[Any] = None
    board_dict: Optional[Dict[str, Any]] = None
    template_used: Optional[str] = None
    generation_method: Optional[str] = None
    generation_time_ms: float = 0.0
    warnings: List[str] = field(default_factory=list)
    dfm_errors: List[Any] = field(default_factory=list)
    sch_filename: Optional[str] = None
    download_url: Optional[str] = None
    error: Optional[str] = None
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    @property
    def success(self) -> bool:
        return self.board is not None


async def _emit(queue: Optional[asyncio.Queue], stage: str, msg: str) -> None:
    if queue is None:
        return
    await queue.put({"stage": stage, "msg": msg})


async def orchestrate(
    request: Any,
    llm: Any,
    template_cache: Dict[str, Dict],
    output_dir: Path,
    bg_add_task: Callable,
    queue_callback: Optional[asyncio.Queue] = None,
    circuit_schema: Any = None,
    kicad_exporter: Any = None,
    aiofiles_mod: Any = None,
) -> OrchestratorResult:
    """Run generation/placement/dfm pipeline and emit SSE stage events."""
    # Import from ai_server lazily to avoid import-cycle issues during module load.
    from ai_server import (  # pylint: disable=import-outside-toplevel
        STRONG_TEMPLATE_THRESHOLD,
        AdvancedDFMEngine,
        BoardData,
        PlacementOptimizer,
        TEMPLATE_KEYWORDS,
        _NP,
        _cleanup_run_dir,
        _coerce_bool,
        _generate_via_skidl,
        _normalize_kicad_schematic_text,
        _safe_filename,
        _sanitize_circuit_data,
        _score_template,
    )

    t0 = time.perf_counter()
    request_id = str(uuid.uuid4())[:8]
    warnings: List[str] = []

    prompt_lower = request.prompt.lower()
    best_name, best_score = _score_template(prompt_lower, TEMPLATE_KEYWORDS)

    circuit_data: Optional[Dict[str, Any]] = None
    template_used: Optional[str] = None
    generation_method: Optional[str] = None
    llm_failed = False

    strict_llm = _coerce_bool((request.constraints or {}).get("strict_llm"), default=True)
    prefer_model = _coerce_bool((request.constraints or {}).get("prefer_model"), default=True)
    use_skidl = _coerce_bool((request.constraints or {}).get("use_skidl"), default=False)
    allow_template = _coerce_bool((request.constraints or {}).get("allow_template"), default=False)

    await _emit(queue_callback, "generation", "Generating circuit design…")

    if use_skidl and llm:
        skidl_data, skidl_w, _skidl_code = await _generate_via_skidl(request.prompt, request_id)
        warnings.extend(skidl_w)

        # Keep cleanup behavior consistent with /generate route.
        from ai_server import SKIDL_MAX_RETRIES, TEMP_DIR  # pylint: disable=import-outside-toplevel
        for i in range(SKIDL_MAX_RETRIES):
            run_dir = TEMP_DIR / f"skidl_{request_id}_{i}"
            if run_dir.exists():
                bg_add_task(_cleanup_run_dir, run_dir)

        if skidl_data:
            circuit_data = skidl_data
            template_used = "skidl_llm"
            generation_method = "skidl"

    can_use_template_fast_path = (
        circuit_data is None
        and allow_template
        and not strict_llm
        and (not prefer_model or llm is None)
    )
    if can_use_template_fast_path:
        if best_name and best_name in template_cache and best_score >= STRONG_TEMPLATE_THRESHOLD:
            circuit_data = template_cache[best_name]
            template_used = best_name
            generation_method = "template"
            warnings.append(f"Strong template match: {best_name} (score={best_score})")

    if circuit_data is None and llm:
        try:
            circuit_data = await llm.generate_circuit_json(request.prompt) or None
            if circuit_data:
                generation_method = "llm_json"
            else:
                llm_failed = True
                warnings.append("LLM returned no circuit data")
        except Exception as exc:  # pragma: no cover - defensive
            llm_failed = True
            details = str(exc).strip() or repr(exc)
            warnings.append(f"LLM JSON generation failed ({type(exc).__name__}): {details}")

    can_use_template_fallback = (
        circuit_data is None
        and allow_template
        and not strict_llm
        and (not prefer_model or llm is None or llm_failed)
    )
    if can_use_template_fallback:
        if best_name and best_name in template_cache and best_score > 0:
            circuit_data = template_cache[best_name]
            template_used = best_name
            generation_method = "template"
            if llm:
                warnings.append(f"Template fallback: {best_name} (score={best_score})")

    if circuit_data is None:
        return OrchestratorResult(
            warnings=warnings,
            error=(
                "Could not generate a circuit from the AI model. "
                "Ensure model backend connectivity. "
                "Template fallback is disabled by default; set constraints.allow_template=true to enable it."
            ),
            generation_method=generation_method,
            request_id=request_id,
            generation_time_ms=round((time.perf_counter() - t0) * 1000, 1),
        )

    if circuit_schema is not None:
        try:
            circuit_schema.CircuitData(**circuit_data)
        except Exception as exc:
            warnings.append(f"Circuit schema validation failed: {exc}")
            # Continue through sanitizer/BoardData path; export can fail independently.

    norm, sanitize_warnings = _sanitize_circuit_data(circuit_data)
    warnings.extend(sanitize_warnings)
    board = BoardData(**norm)

    await _emit(queue_callback, "placement", "Optimising component placement…")
    if all(c.x == 0.0 and c.y == 0.0 for c in board.components if not c.is_power_symbol):
        algo = (
            "force_directed" if _NP
            else "annealing" if request.priority == "quality"
            else "grid"
        )
        placement = PlacementOptimizer(board).optimize(algo)
        for ref, pos in placement["positions"].items():
            comp = board.get_component(ref)
            if comp:
                comp.x, comp.y = pos["x"], pos["y"]

    await _emit(queue_callback, "dfm", "Running DFM analysis…")
    dfm_errors = AdvancedDFMEngine(board).analyze()
    for violation in dfm_errors:
        if getattr(violation, "severity", "") in ("error", "critical"):
            warnings.append(f"[DFM {violation.rule_id}] {violation.message}")

    board_dict = board.model_dump()
    json_path = output_dir / f"circuit_{request_id}.json"

    async def _save_json() -> None:
        txt = json.dumps(board_dict, indent=2)
        if aiofiles_mod is not None:
            async with aiofiles_mod.open(json_path, "w") as f:
                await f.write(txt)
        else:
            json_path.write_text(txt)

    bg_add_task(_save_json)

    sch_filename: Optional[str] = None
    download_url: Optional[str] = None

    if circuit_schema is not None and kicad_exporter is not None:
        raw_src = (
            template_cache.get(template_used, {})
            if template_used and template_used != "skidl_llm"
            else circuit_data
        )
        schema = circuit_schema.CircuitData(**raw_src)
        content = kicad_exporter.export_to_kicad_sch(schema)
        content = _normalize_kicad_schematic_text(content)
        desc = raw_src.get("description", template_used or request_id)
        safe = _safe_filename(desc, request_id)
        sch_path = output_dir / f"{safe}.kicad_sch"
        sch_filename = sch_path.name
        download_url = f"/download/{sch_filename}"

        async def _save_sch() -> None:
            if aiofiles_mod is not None:
                async with aiofiles_mod.open(sch_path, "w") as f:
                    await f.write(content)
            else:
                sch_path.write_text(content, encoding="utf-8")

        bg_add_task(_save_sch)

    return OrchestratorResult(
        board=board,
        board_dict=board_dict,
        template_used=template_used,
        generation_method=generation_method,
        generation_time_ms=round((time.perf_counter() - t0) * 1000, 1),
        warnings=warnings,
        dfm_errors=dfm_errors,
        sch_filename=sch_filename,
        download_url=download_url,
        request_id=request_id,
    )
