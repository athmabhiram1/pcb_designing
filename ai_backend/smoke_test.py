"""
Smoke test for critical backend endpoints.
Run after starting backend server.
"""

import json
import os
import sys
import time
import urllib.request

BASE = os.environ.get("BACKEND_BASE_URL", "http://127.0.0.1:8765").rstrip("/")
PASS = "OK"
FAIL = "FAIL"

GENERATE_CASES = [
    ("555 timer astable LED blinker", "555_timer"),
    ("3.3V LDO regulator from 5V input", "3v3_regulator"),
    ("simple LED with resistor from 5V", "led_resistor"),
    ("op-amp unity gain voltage buffer", "opamp_buffer"),
    ("NMOS low-side MOSFET switch", "mosfet_switch"),
]


def _get(path: str) -> dict:
    with urllib.request.urlopen(f"{BASE}{path}", timeout=10) as response:
        return json.loads(response.read())


def _post(path: str, body: dict, timeout: int = 90) -> dict:
    request = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def check(label: str, ok: bool, detail: str = "") -> bool:
    status = PASS if ok else FAIL
    line = f"  [{status}] {label}"
    if detail:
        line += f" -> {detail}"
    print(line)
    return ok


def main() -> None:
    print("\n=== AI PCB Backend Smoke Test ===\n")
    all_passed = True

    try:
        health = _get("/health")
        ok = health.get("status") in {"ok", "healthy", "degraded"}
        all_passed &= check(
            "/health",
            ok,
            f"llm={health.get('llm_loaded')} templates={health.get('templates_available')}",
        )
    except Exception as exc:
        all_passed &= check("/health", False, str(exc))

    try:
        templates = _get("/templates")
        ok = isinstance(templates, list) and len(templates) >= 5
        all_passed &= check("/templates", ok, f"count={len(templates)}")
    except Exception as exc:
        all_passed &= check("/templates", False, str(exc))

    last_filename = None
    for prompt, expected_template in GENERATE_CASES:
        try:
            start = time.time()
            result = _post("/generate", {"prompt": prompt})
            elapsed = time.time() - start
            ok = result.get("success") is True
            all_passed &= check(
                f"/generate [{expected_template}]",
                ok,
                (
                    f"method={result.get('generation_method')} "
                    f"template={result.get('template_used')} "
                    f"comps={result.get('component_count')} "
                    f"nets={result.get('net_count')} "
                    f"{elapsed:.1f}s"
                ),
            )
            if ok:
                last_filename = (result.get("download_url") or "").split("/")[-1]
        except Exception as exc:
            all_passed &= check(f"/generate [{expected_template}]", False, str(exc))

    if last_filename:
        try:
            with urllib.request.urlopen(f"{BASE}/download/{last_filename}", timeout=10) as response:
                size = len(response.read())
            all_passed &= check(f"/download/{last_filename[:30]}...", size > 100, f"{size} bytes")
        except Exception as exc:
            all_passed &= check("/download", False, str(exc))

    print()
    if all_passed:
        print("[OK] Smoke tests passed")
        sys.exit(0)

    print("[FAIL] Smoke tests failed")
    sys.exit(1)


if __name__ == "__main__":
    main()
