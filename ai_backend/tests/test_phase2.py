from __future__ import annotations

import asyncio
import builtins
import ctypes
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from ai_server import _state, app
from ai_server import _detect_circuit_quality_issues
from ai_server import _sanitize_circuit_data
from engines import llm_engine
from services.generation_orchestrator import orchestrate
from services import skidl_runner


def test_apply_windows_job_limits_returns_false_on_kernel32_error(monkeypatch) -> None:
    class BrokenWindll:
        @property
        def kernel32(self):
            raise RuntimeError("kernel32 unavailable")

    monkeypatch.setattr(skidl_runner.os, "name", "nt", raising=False)
    monkeypatch.setattr(ctypes, "windll", BrokenWindll(), raising=False)

    fake_proc = SimpleNamespace(_handle=1, pid=1234)
    assert skidl_runner._apply_windows_job_limits(fake_proc) is False


def test_run_returns_skidlresult_when_windows_job_limits_fail(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(skidl_runner.os, "name", "nt", raising=False)
    monkeypatch.setattr(skidl_runner, "_apply_windows_job_limits", lambda proc: False)
    monkeypatch.setattr(skidl_runner, "default_kicad_symbol_dir", lambda: str(tmp_path))
    monkeypatch.setattr(skidl_runner, "screen_skidl_code", lambda code: None)
    monkeypatch.setattr(skidl_runner, "extract_python_code_block", lambda text: text)

    code = (
        "from pathlib import Path\n"
        "Path('circuit.net').write_text('<export><components></components><nets></nets></export>', encoding='utf-8')\n"
        "# generate_netlist\n"
    )

    result = skidl_runner.run(
        code=code,
        request_id="phase2",
        prompt="test",
        timeout_seconds=5,
        temp_dir=tmp_path,
        parse_netlist=lambda netlist_path, description: {
            "description": description,
            "components": [],
            "connections": [],
            "board_width": 100.0,
            "board_height": 80.0,
            "design_rules": {},
        },
    )

    assert isinstance(result, skidl_runner.SkidlResult)
    assert result.success is True
    assert any("JobObject limits unavailable" in warning for warning in result.warnings)


# ── Minimal template used by SSE and generation tests ─────────────────────────

_MINIMAL_555 = {
    "description": "555 timer astable LED blinker",
    "components": [
        {
            "ref": "R1",
            "value": "1k",
            "footprint": "Resistor_SMD:R_0805_2012Metric",
            "x": 10,
            "y": 10,
            "rotation": 0,
            "layer": "top",
        },
        {
            "ref": "D1",
            "value": "LED",
            "footprint": "LED_SMD:LED_0805_2012Metric",
            "x": 20,
            "y": 10,
            "rotation": 0,
            "layer": "top",
        },
    ],
    "connections": [
        {
            "net": "N1",
            "pins": [{"ref": "R1", "pin": "1"}, {"ref": "D1", "pin": "1"}],
        }
    ],
    "board_width": 100,
    "board_height": 80,
}


class TestSSEEndpoint:
    """Tests for POST /generate/stream that genuinely exercise the SSE route."""

    @staticmethod
    def _frames(raw: str) -> list[str]:
        """Split raw SSE text into individual frames (separated by blank lines)."""
        return [frame.strip() for frame in raw.split("\n\n") if frame.strip()]

    def test_sse_emits_status_then_result(self) -> None:
        _state.template_cache["555_timer"] = dict(_MINIMAL_555)

        with TestClient(app) as client:
            _state.llm = None
            response = client.post(
                "/generate/stream",
                json={
                    "prompt": "555 timer",
                    "constraints": {"allow_template": True, "strict_llm": False},
                },
            )

        assert response.status_code == 200
        body = response.text
        frames = self._frames(body)

        # Must contain at least one status frame
        assert any("event: status" in f for f in frames), (
            f"No 'event: status' frame found.\nBody:\n{body}"
        )

        # Must contain exactly one result frame as the last meaningful event
        result_frames = [f for f in frames if f.startswith("event: result")]
        assert len(result_frames) >= 1, (
            f"No 'event: result' frame found.\nBody:\n{body}"
        )

        # Parse the result payload and verify success
        last_result = result_frames[-1]
        data_line = next(
            line for line in last_result.splitlines() if line.startswith("data: ")
        )
        payload = json.loads(data_line[len("data: "):])
        assert isinstance(payload, dict)
        assert payload.get("success") is True


class TestOrchestratorPipeline:
    def test_orchestrator_emits_all_three_stages(self, tmp_path) -> None:
        async def _run() -> tuple[Any, list[Any]]:
            q: asyncio.Queue = asyncio.Queue()
            request = SimpleNamespace(
                prompt="555 timer",
                constraints={"allow_template": True, "strict_llm": False},
                priority="quality",
            )
            result = await orchestrate(
                request=request,
                llm=None,
                template_cache={"555_timer": dict(_MINIMAL_555)},
                output_dir=tmp_path,
                bg_add_task=lambda *_args, **_kwargs: None,
                queue_callback=q,
                circuit_schema=None,
                kicad_exporter=None,
                aiofiles_mod=None,
            )
            items: list[Any] = []
            while not q.empty():
                items.append(await q.get())
            return result, items

        result, items = asyncio.run(_run())
        assert result.success is True

        stages = [item.get("stage") for item in items if isinstance(item, dict)]
        assert "generation" in stages
        assert "placement" in stages
        assert "dfm" in stages

    def test_orchestrator_result_has_board_on_success(self, tmp_path) -> None:
        async def _run() -> Any:
            request = SimpleNamespace(
                prompt="555 timer",
                constraints={"allow_template": True, "strict_llm": False},
                priority="quality",
            )
            return await orchestrate(
                request=request,
                llm=None,
                template_cache={"555_timer": dict(_MINIMAL_555)},
                output_dir=tmp_path,
                bg_add_task=lambda *_args, **_kwargs: None,
                queue_callback=None,
                circuit_schema=None,
                kicad_exporter=None,
                aiofiles_mod=None,
            )

        result = asyncio.run(_run())
        assert result.success is True
        assert result.board is not None

    def test_sse_emits_error_on_empty_prompt(self) -> None:
        with TestClient(app) as client:
            response = client.post("/generate/stream", json={"prompt": ""})

        # FastAPI should reject empty prompt (min_length=1) with 422
        # OR the handler may catch it and stream an error event
        if response.status_code == 422:
            return

        assert "event: error" in response.text


class TestGenerationPolicy:
    def test_prefer_model_true_avoids_template_when_llm_available(self, monkeypatch) -> None:
        class StubLLM:
            async def generate_circuit_json(self, _prompt: str):
                return {
                    "description": "llm circuit",
                    "components": [
                        {
                            "ref": "R1",
                            "lib": "Device",
                            "part": "R",
                            "value": "1k",
                            "footprint": "Resistor_SMD:R_0805_2012Metric",
                            "description": "resistor",
                            "pins": [{"number": "1", "name": "1"}, {"number": "2", "name": "2"}],
                        },
                        {
                            "ref": "R2",
                            "lib": "Device",
                            "part": "R",
                            "value": "2k",
                            "footprint": "Resistor_SMD:R_0805_2012Metric",
                            "description": "resistor",
                            "pins": [{"number": "1", "name": "1"}, {"number": "2", "name": "2"}],
                        },
                    ],
                    "connections": [
                        {"net": "N1", "pins": ["R1.1", "R2.1"]}
                    ],
                    "board_width": 100,
                    "board_height": 80,
                }

        _state.template_cache["555_timer"] = dict(_MINIMAL_555)

        with TestClient(app) as client:
            _state.llm = StubLLM()
            resp = client.post("/generate", json={"prompt": "555 timer", "priority": "quality"})

        assert resp.status_code == 200
        payload = resp.json()
        assert payload.get("success") is True
        assert payload.get("generation_method") == "llm_json"

    def test_prefer_model_true_without_llm_returns_error_by_default(self) -> None:
        _state.template_cache["555_timer"] = dict(_MINIMAL_555)

        with TestClient(app) as client:
            _state.llm = None
            resp = client.post("/generate", json={"prompt": "555 timer", "priority": "quality"})

        assert resp.status_code == 200
        payload = resp.json()
        assert payload.get("success") is False

    def test_template_fallback_can_be_enabled_explicitly(self) -> None:
        _state.template_cache["555_timer"] = dict(_MINIMAL_555)

        with TestClient(app) as client:
            _state.llm = None
            resp = client.post(
                "/generate",
                json={
                    "prompt": "555 timer",
                    "priority": "quality",
                    "constraints": {"allow_template": True, "strict_llm": False},
                },
            )

        assert resp.status_code == 200
        payload = resp.json()
        assert payload.get("success") is True
        assert payload.get("generation_method") == "template"


class TestLLMEngine:
    def test_load_llm_returns_stub_when_llama_cpp_absent(self, monkeypatch) -> None:
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "llama_cpp":
                raise ImportError("forced import error")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        monkeypatch.setattr(llm_engine, "_looks_like_ollama_available", lambda _url: False)
        monkeypatch.delenv("LLM_BACKEND", raising=False)
        monkeypatch.delenv("OLLAMA_MODEL", raising=False)
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)

        stub = llm_engine.load_llm()
        assert stub is not None

        with pytest.raises(NotImplementedError):
            asyncio.run(stub.generate_async("x", 100, 0.2))

    def test_generate_circuit_json_strips_fences_and_parses(self, tmp_path) -> None:
        model_file = tmp_path / "mock.gguf"
        model_file.write_text("mock", encoding="utf-8")

        class MockLlama:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def __call__(self, prompt, max_tokens, temperature, echo=False):
                return {
                    "choices": [
                        {
                            "text": "```json\n{\"description\":\"test\",\"components\":[],\"connections\":[],\"board_width\":100,\"board_height\":80}\n```"
                        }
                    ]
                }

        llm = llm_engine.LLMEngine(model_path=str(model_file), llama_cls=MockLlama)
        result = asyncio.run(llm.generate_circuit_json("make a circuit"))
        assert isinstance(result, dict)
        assert "description" in result

    def test_generate_circuit_json_returns_none_on_invalid_json(self, tmp_path) -> None:
        model_file = tmp_path / "mock.gguf"
        model_file.write_text("mock", encoding="utf-8")

        class MockLlama:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def __call__(self, prompt, max_tokens, temperature, echo=False):
                return {"choices": [{"text": "not valid json"}]}

        llm = llm_engine.LLMEngine(model_path=str(model_file), llama_cls=MockLlama)
        result = asyncio.run(llm.generate_circuit_json("make a circuit"))
        assert result is None


class TestCircuitSanitizer:
    def test_sanitize_fixes_duplicate_refs_and_pin_conflicts(self) -> None:
        raw = {
            "description": "bad circuit",
            "components": [
                {"ref": "R1", "value": "1k", "footprint": "", "x": 0, "y": 0, "rotation": 0, "layer": "top"},
                {"ref": "R1", "value": "2k", "footprint": "", "x": 0, "y": 0, "rotation": 0, "layer": "top"},
            ],
            "connections": [
                {"net": "N1", "pins": ["R1.1", "R1.2"]},
                {"net": "N2", "pins": ["R1.1", "R1.2"]},
            ],
        }
        norm, warnings = _sanitize_circuit_data(raw)

        refs = [c["ref"] for c in norm["components"]]
        assert len(set(refs)) == len(refs)
        assert any("Renamed duplicate reference" in w for w in warnings)
        assert any("Dropped conflicting pin" in w for w in warnings)

    def test_sanitize_applies_555_guardrails(self) -> None:
        raw = {
            "description": "broken 555",
            "components": [
                {"ref": "U1", "part": "NE555", "value": "NE555", "footprint": "", "x": 0, "y": 0, "rotation": 0, "layer": "top"},
                {"ref": "R1", "value": "10k", "footprint": "", "x": 0, "y": 0, "rotation": 0, "layer": "top"},
            ],
            "connections": [
                {"net": "GND", "pins": ["U1.4", "U1.1"]},
                {"net": "A", "pins": ["U1.2", "R1.1"]},
                {"net": "B", "pins": ["U1.6", "R1.1"]},
            ],
        }
        norm, warnings = _sanitize_circuit_data(raw)

        pins_by_net = {
            c["net"]: {(p["ref"], p["pin"]) for p in c["pins"]}
            for c in norm["connections"]
        }
        assert ("U1", "4") in pins_by_net.get("VCC", set())
        assert ("U1", "8") in pins_by_net.get("VCC", set())
        assert ("U1", "1") in pins_by_net.get("GND", set())
        assert ("U1", "2") in pins_by_net.get("TRIG_THR", set())
        assert ("U1", "6") in pins_by_net.get("TRIG_THR", set())
        assert any("Applied 555 core wiring guardrails" in w for w in warnings)

    def test_topology_check_detects_broken_555(self) -> None:
        raw = {
            "description": "bad 555 topology",
            "components": [
                {"ref": "U1", "part": "NE555", "value": "NE555", "footprint": "", "x": 0, "y": 0, "rotation": 0, "layer": "top"},
                {"ref": "R1", "part": "R", "value": "10k", "footprint": "", "x": 0, "y": 0, "rotation": 0, "layer": "top"},
            ],
            "connections": [
                {"net": "GND", "pins": [{"ref": "U1", "pin": "1"}]},
                {"net": "N_A", "pins": [{"ref": "U1", "pin": "2"}, {"ref": "R1", "pin": "1"}]},
                {"net": "N_B", "pins": [{"ref": "U1", "pin": "6"}, {"ref": "R1", "pin": "2"}]},
            ],
        }
        issues = _detect_circuit_quality_issues(raw)
        assert any("pin 8" in i for i in issues)
        assert any("pin 4" in i for i in issues)
        assert any("pins 2 and 6" in i for i in issues)

    def test_topology_check_detects_missing_ams1117_caps(self) -> None:
        raw = {
            "description": "bad ldo topology",
            "components": [
                {"ref": "U1", "part": "AMS1117-3.3", "value": "AMS1117-3.3", "footprint": "", "x": 0, "y": 0, "rotation": 0, "layer": "top"},
                {"ref": "R1", "part": "R", "value": "1k", "footprint": "", "x": 0, "y": 0, "rotation": 0, "layer": "top"},
            ],
            "connections": [
                {"net": "GND", "pins": [{"ref": "U1", "pin": "1"}, {"ref": "R1", "pin": "1"}]},
                {"net": "VIN", "pins": [{"ref": "U1", "pin": "3"}, {"ref": "R1", "pin": "2"}]},
                {"net": "+3V3", "pins": [{"ref": "U1", "pin": "2"}, {"ref": "R1", "pin": "2"}]},
            ],
        }
        issues = _detect_circuit_quality_issues(raw)
        assert any("input decoupling capacitor" in i for i in issues)
        assert any("output decoupling capacitor" in i for i in issues)


class TestBoardOpsImport:
    """Verify board_ops can be imported in CI without pcbnew."""

    def test_board_ops_imports_without_pcbnew(self) -> None:
        # The module should import cleanly even without pcbnew
        from plugin.board.board_ops import BoardReader

        # Instantiation should succeed (no pcbnew usage in __init__)
        reader = BoardReader(None)

        # Calling a method that uses pcbnew should raise RuntimeError
        with pytest.raises(RuntimeError, match="pcbnew is only available inside KiCad"):
            reader.read_components()


class TestKiCadExporterWiring:
    def test_export_includes_explicit_power_wires_for_3v3_template(self) -> None:
        from circuit_schema import CircuitData
        from engines.kicad_exporter import export_to_kicad_sch

        template_path = (
            Path(__file__).resolve().parents[1]
            / "templates"
            / "3v3_regulator.json"
        )
        data = json.loads(template_path.read_text(encoding="utf-8"))
        sch = export_to_kicad_sch(CircuitData(**data))

        # VIN net already had 3 wire segments; explicit power wiring should add more.
        wire_count = sch.count("(wire (pts")
        assert wire_count >= 8
