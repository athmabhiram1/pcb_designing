from __future__ import annotations

import asyncio

from services.generation_service import (
    SkidlRetryConfig,
    extract_python_code_block,
    generate_via_skidl_retry,
    score_template,
    screen_skidl_code,
)


def test_extract_python_code_block_sanitizes_tokens() -> None:
    raw = "```python\n<s>from skidl import *\nprint('ok')\n</s>\n<|assistant|>\n```"
    cleaned = extract_python_code_block(raw)
    assert "<s>" not in cleaned
    assert "<|assistant|>" not in cleaned
    assert "from skidl import *" in cleaned


def test_screen_skidl_code_blocks_os_imports() -> None:
    blocked = screen_skidl_code("from skidl import *\nimport os\n")
    assert blocked is not None


def test_score_template_prefers_highest_score() -> None:
    template_keywords = [
        (["motor driver", "dc motor"], "motor_driver", 100),
        (["led"], "led_resistor", 20),
    ]
    name, score = score_template("need a dc motor driver", template_keywords)
    assert name == "motor_driver"
    assert score >= 100


def test_generate_via_skidl_retry_succeeds_after_retry() -> None:
    calls = {"n": 0}

    async def fake_llm_generate(payload: str, max_tokens: int, temperature: float) -> str:
        return "from skidl import *\nR1 = Part('Device', 'R', value='1k')"

    def fake_exec(code: str, run_id: str, prompt: str):
        calls["n"] += 1
        if calls["n"] == 1:
            return None, ["SKiDL failed: missing footprint"]
        return {"components": [], "connections": []}, ["Used SKiDL generation pipeline"]

    result, warnings, last_code = asyncio.run(
        generate_via_skidl_retry(
            prompt="simple resistor",
            request_id="abc123",
            config=SkidlRetryConfig(system_prompt="SYSTEM", max_retries=3),
            llm_generate=fake_llm_generate,
            execute_skidl_code=fake_exec,
            symbol_dir_resolver=lambda: "/tmp",
        )
    )

    assert result is not None
    assert calls["n"] == 2
    assert any("succeeded on attempt 2" in w.lower() for w in warnings)
    assert last_code is not None


def test_generate_via_skidl_retry_short_circuits_without_symbol_dir() -> None:
    async def fake_llm_generate(payload: str, max_tokens: int, temperature: float) -> str:
        raise AssertionError("LLM should not be called without symbol dir")

    def fake_exec(code: str, run_id: str, prompt: str):
        raise AssertionError("Executor should not run without symbol dir")

    result, warnings, last_code = asyncio.run(
        generate_via_skidl_retry(
            prompt="555 timer",
            request_id="abc123",
            config=SkidlRetryConfig(system_prompt="SYSTEM", max_retries=3),
            llm_generate=fake_llm_generate,
            execute_skidl_code=fake_exec,
            symbol_dir_resolver=lambda: None,
        )
    )

    assert result is None
    assert last_code is None
    assert any("symbol directory not found" in w.lower() for w in warnings)
