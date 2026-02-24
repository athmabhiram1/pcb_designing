"""
Smoke test — hit every critical endpoint and verify responses.
Run AFTER starting the backend:  python ai_server.py

Usage:
    python smoke_test.py
"""
import sys
import json
import time
import os
import urllib.request
import urllib.error

BASE = "http://127.0.0.1:8765"
PASS = "✓"
FAIL = "✗"

# (prompt, expected_template_name)
GENERATE_CASES = [
    ("555 timer astable LED blinker",        "555_timer"),
    ("3.3V LDO regulator from 5V input",     "3v3_regulator"),
    ("simple LED with resistor from 5V",     "led_resistor"),
    ("op-amp unity gain voltage buffer",     "opamp_buffer"),
    ("NMOS low-side MOSFET switch",          "mosfet_switch"),
]


def _get(path: str) -> dict:
    with urllib.request.urlopen(f"{BASE}{path}", timeout=10) as r:
        return json.loads(r.read())


def _post(path: str, body: dict, timeout: int = 90) -> dict:
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def check(label: str, ok: bool, detail: str = "") -> bool:
    sym = PASS if ok else FAIL
    msg = f"  {sym}  {label}"
    if detail:
        msg += f"  →  {detail}"
    print(msg)
    return ok


def main():
    print("\n=== AI PCB Backend Smoke Test ===\n")
    all_passed = True

    # ── /health ───────────────────────────────────────────────────────────────
    try:
        h = _get("/health")
        ok = h.get("status") == "ok"
        all_passed &= check("/health", ok,
            f"llm={h.get('llm_loaded')}  templates={h.get('templates_available')}")
    except Exception as e:
        all_passed &= check("/health", False, str(e))

    # ── /templates ────────────────────────────────────────────────────────────
    try:
        ts = _get("/templates")
        ok = isinstance(ts, list) and len(ts) >= 5
        names = [t["name"] for t in ts]
        all_passed &= check(f"/templates ({len(ts)} found)", ok, str(names))
    except Exception as e:
        all_passed &= check("/templates", False, str(e))

    # ── /generate for each template ───────────────────────────────────────────
    print()
    last_filename = None
    for prompt, expected_template in GENERATE_CASES:
        try:
            t0 = time.time()
            r = _post("/generate", {"prompt": prompt})
            elapsed = time.time() - t0
            ok = r.get("success") is True
            tpl_ok = r.get("template_used") == expected_template
            bom_ok = len(r.get("bom", [])) > 0
            all_passed &= check(
                f"/generate [{expected_template}]", ok and tpl_ok,
                f"comps={r.get('component_count')}  nets={r.get('net_count')}  "
                f"bom_lines={len(r.get('bom',[]))}  "
                f"template={r.get('template_used')}  {elapsed:.1f}s"
            )
            if ok:
                fp = r.get("output_file", "")
                file_ok = os.path.isfile(fp) and os.path.getsize(fp) > 100
                all_passed &= check(f"  .kicad_sch file", file_ok, fp)
                last_filename = r.get("download_url", "").split("/")[-1]
        except Exception as e:
            all_passed &= check(f"/generate [{expected_template}]", False, str(e))

    # ── /circuit/{name} ───────────────────────────────────────────────────────
    print()
    try:
        data = _get("/circuit/555_timer")
        ok = "components" in data and "connections" in data
        all_passed &= check("/circuit/555_timer", ok,
            f"comps={len(data.get('components',[]))}  nets={len(data.get('connections',[]))}")
    except Exception as e:
        all_passed &= check("/circuit/555_timer", False, str(e))

    # ── /download ─────────────────────────────────────────────────────────────
    if last_filename:
        try:
            req = urllib.request.Request(f"{BASE}/download/{last_filename}")
            with urllib.request.urlopen(req, timeout=10) as resp:
                size = len(resp.read())
            all_passed &= check(f"/download/{last_filename[:30]}…", size > 100, f"{size} bytes")
        except Exception as e:
            all_passed &= check("/download", False, str(e))

    # ── summary ───────────────────────────────────────────────────────────────
    print()
    if all_passed:
        print(f"  {PASS}  All checks passed — backend is ready.\n")
        sys.exit(0)
    else:
        print(f"  {FAIL}  Some checks failed — see above.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()

import sys
import json
import time
import urllib.request
import urllib.error

BASE = "http://127.0.0.1:8765"
PASS = "✓"
FAIL = "✗"


def _get(path: str) -> dict:
    with urllib.request.urlopen(f"{BASE}{path}", timeout=10) as r:
        return json.loads(r.read())


def _post(path: str, body: dict, timeout: int = 90) -> dict:
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def check(label: str, ok: bool, detail: str = ""):
    sym = PASS if ok else FAIL
    msg = f"  {sym}  {label}"
    if detail:
        msg += f"  →  {detail}"
    print(msg)
    return ok


def main():
    print("\n=== AI PCB Backend Smoke Test ===\n")
    all_passed = True

    # ── /health ───────────────────────────────────────────────────────────────
    try:
        h = _get("/health")
        ok = h.get("status") == "ok"
        all_passed &= check("/health", ok,
            f"llm_loaded={h.get('llm_loaded')}  templates={h.get('templates_available')}")
    except Exception as e:
        all_passed &= check("/health", False, str(e))

    # ── /templates ────────────────────────────────────────────────────────────
    try:
        ts = _get("/templates")
        ok = isinstance(ts, list) and len(ts) >= 1
        names = [t["name"] for t in ts]
        all_passed &= check("/templates", ok, f"found: {names}")
    except Exception as e:
        all_passed &= check("/templates", False, str(e))

    # ── /generate — 555 timer ─────────────────────────────────────────────────
    try:
        t0 = time.time()
        r = _post("/generate", {"prompt": "555 timer astable LED blinker"})
        elapsed = time.time() - t0
        ok = r.get("success") is True
        all_passed &= check(
            "/generate 555 timer", ok,
            f"components={r.get('component_count')}  nets={r.get('net_count')}  "
            f"template={r.get('template_used')}  time={elapsed:.1f}s  "
            f"file={r.get('download_url')}"
        )
        if ok:
            import os
            fp = r.get("output_file", "")
            file_ok = os.path.isfile(fp) and os.path.getsize(fp) > 100
            all_passed &= check("  .kicad_sch file exists", file_ok, fp)
    except Exception as e:
        all_passed &= check("/generate 555 timer", False, str(e))

    # ── /generate — 3.3V regulator ────────────────────────────────────────────
    try:
        t0 = time.time()
        r = _post("/generate", {"prompt": "3.3V LDO regulator from 5V input"})
        elapsed = time.time() - t0
        ok = r.get("success") is True
        all_passed &= check(
            "/generate 3v3 regulator", ok,
            f"components={r.get('component_count')}  nets={r.get('net_count')}  "
            f"template={r.get('template_used')}  time={elapsed:.1f}s  "
            f"file={r.get('download_url')}"
        )
        if ok:
            import os
            fp = r.get("output_file", "")
            file_ok = os.path.isfile(fp) and os.path.getsize(fp) > 100
            all_passed &= check("  .kicad_sch file exists", file_ok, fp)
    except Exception as e:
        all_passed &= check("/generate 3v3 regulator", False, str(e))

    # ── /download test ────────────────────────────────────────────────────────
    try:
        # Re-generate to get a known filename
        r = _post("/generate", {"prompt": "555 timer"})
        if r.get("success"):
            url_path = r["download_url"]
            fname = url_path.split("/")[-1]
            req = urllib.request.Request(f"{BASE}/download/{fname}")
            with urllib.request.urlopen(req, timeout=10) as resp:
                size = len(resp.read())
            ok = size > 100
            all_passed &= check(f"/download/{fname}", ok, f"{size} bytes")
    except Exception as e:
        all_passed &= check("/download", False, str(e))

    # ── summary ───────────────────────────────────────────────────────────────
    print()
    if all_passed:
        print(f"  {PASS}  All checks passed — backend is ready.\n")
        sys.exit(0)
    else:
        print(f"  {FAIL}  Some checks failed — see above.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
