from __future__ import annotations

import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import gettempdir
from typing import Any, Callable, Dict, List, Optional

from services.generation_service import (
    default_kicad_symbol_dir,
    extract_python_code_block,
    screen_skidl_code,
)

try:
    import resource as _resource  # POSIX only
except ImportError:
    _resource = None  # type: ignore[assignment]


SKIDL_EXEC_TIMEOUT = 40
DEFAULT_TEMP_DIR = Path(gettempdir()) / "ai_pcb"


@dataclass
class SkidlResult:
    success: bool
    circuit_data: Optional[Dict[str, Any]] = None
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None


def _make_preexec_fn() -> Optional[Callable[[], None]]:
    """Linux/POSIX resource limits path; unchanged for non-Windows execution."""
    if os.name == "nt" or _resource is None:
        return None

    def _set_limits() -> None:
        try:
            _resource.setrlimit(_resource.RLIMIT_CPU, (10, 10))
            _resource.setrlimit(_resource.RLIMIT_FSIZE, (5 * 1024 * 1024, 5 * 1024 * 1024))
            _resource.setrlimit(_resource.RLIMIT_NOFILE, (20, 20))
        except Exception:
            pass

    return _set_limits


def _default_parse_netlist(netlist_path: Path, description: str) -> Dict[str, Any]:
    tree = ET.parse(str(netlist_path))
    root = tree.getroot()
    comp_map: Dict[str, Dict[str, Any]] = {}

    for comp in root.findall(".//components/comp"):
        ref = (comp.attrib.get("ref") or "").strip()
        if not ref:
            continue
        libsource = comp.find("libsource")
        comp_map[ref] = {
            "ref": ref,
            "value": (comp.findtext("value") or ref).strip(),
            "footprint": (comp.findtext("footprint") or "").strip(),
            "lib": (libsource.attrib.get("lib") if libsource is not None else "") or "Device",
            "part": (libsource.attrib.get("part") if libsource is not None else "") or "R",
            "x": 0.0,
            "y": 0.0,
            "rotation": 0.0,
            "layer": "top",
            "pins": [],
        }

    connections: List[Dict[str, Any]] = []
    for net in root.findall(".//nets/net"):
        net_name = (net.attrib.get("name") or f"N{net.attrib.get('code', '')}").strip()
        pin_refs: List[Dict[str, str]] = []
        for node in net.findall("node"):
            ref = (node.attrib.get("ref") or "").strip()
            pin = (node.attrib.get("pin") or "").strip()
            if ref and pin:
                pin_refs.append({"ref": ref, "pin": pin})
        if len(pin_refs) >= 2:
            connections.append({"net": net_name, "pins": pin_refs})

    return {
        "description": description,
        "components": list(comp_map.values()),
        "connections": connections,
        "board_width": 100.0,
        "board_height": 80.0,
        "design_rules": {},
    }


def _apply_windows_job_limits(proc: subprocess.Popen[Any]) -> bool:
    """Apply Windows Job Object limits. Returns True on success, False on fallback."""
    if os.name != "nt":
        return False

    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32

        JOB_OBJECT_LIMIT_PROCESS_TIME = 0x00000002
        JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x00000008
        JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
        JobObjectExtendedLimitInformation = 9

        class LARGE_INTEGER(ctypes.Structure):
            _fields_ = [("QuadPart", ctypes.c_longlong)]

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", LARGE_INTEGER),
                ("PerJobUserTimeLimit", LARGE_INTEGER),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            wintypes.INT,
            wintypes.LPVOID,
            wintypes.DWORD,
        ]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL

        hjob = kernel32.CreateJobObjectW(None, None)
        if not hjob:
            return False

        limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        limits.BasicLimitInformation.LimitFlags = (
            JOB_OBJECT_LIMIT_PROCESS_TIME
            | JOB_OBJECT_LIMIT_PROCESS_MEMORY
            | JOB_OBJECT_LIMIT_ACTIVE_PROCESS
        )
        limits.BasicLimitInformation.ActiveProcessLimit = 1
        limits.BasicLimitInformation.PerProcessUserTimeLimit.QuadPart = 10 * 10_000_000
        limits.ProcessMemoryLimit = 256 * 1024 * 1024

        ok = kernel32.SetInformationJobObject(
            hjob,
            JobObjectExtendedLimitInformation,
            ctypes.byref(limits),
            ctypes.sizeof(limits),
        )
        if not ok:
            kernel32.CloseHandle(hjob)
            return False

        process_handle = getattr(proc, "_handle", None)
        if not process_handle:
            kernel32.CloseHandle(hjob)
            return False

        ok = kernel32.AssignProcessToJobObject(hjob, process_handle)
        if not ok:
            kernel32.CloseHandle(hjob)
            return False

        # Keep job object alive for the process lifetime.
        setattr(proc, "_job_handle", hjob)
        return True
    except Exception:
        return False


def run(
    code: str,
    request_id: str,
    prompt: str,
    timeout_seconds: int = SKIDL_EXEC_TIMEOUT,
    temp_dir: Path = DEFAULT_TEMP_DIR,
    parse_netlist: Callable[[Path, str], Dict[str, Any]] = _default_parse_netlist,
) -> SkidlResult:
    """Execute generated SKiDL code with sandbox limits and parse resulting netlist."""
    warnings: List[str] = []

    extracted = extract_python_code_block(code)
    blocked = screen_skidl_code(extracted)
    if blocked:
        return SkidlResult(success=False, warnings=[f"SKiDL code rejected: {blocked}"], error=blocked)

    run_dir = temp_dir / f"skidl_{request_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    script = run_dir / "circuit.py"

    suffix = "\n\nERC()\ngenerate_netlist(file_='circuit.net', tool=KICAD9)\n"
    payload = extracted if "generate_netlist" in extracted else (extracted + suffix)
    script.write_text(payload, encoding="utf-8")

    env = os.environ.copy()
    symbol_dir = default_kicad_symbol_dir()
    if symbol_dir:
        env.setdefault("KICAD_SYMBOL_DIR", symbol_dir)
        env.setdefault("KICAD6_SYMBOL_DIR", symbol_dir)
        env.setdefault("KICAD7_SYMBOL_DIR", symbol_dir)
        env.setdefault("KICAD8_SYMBOL_DIR", symbol_dir)
        env.setdefault("KICAD9_SYMBOL_DIR", symbol_dir)
    else:
        return SkidlResult(
            success=False,
            warnings=[
                "SKiDL skipped: KiCad symbol directory not found. "
                "Set KICAD_SYMBOL_DIR (or KICAD9_SYMBOL_DIR)."
            ],
            error="symbol directory not found",
        )

    create_no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    popen_kwargs: Dict[str, Any] = {
        "cwd": str(run_dir),
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "env": env,
        "creationflags": create_no_window,
    }

    preexec_fn = _make_preexec_fn()
    if preexec_fn is not None:
        popen_kwargs["preexec_fn"] = preexec_fn

    try:
        proc = subprocess.Popen([sys.executable, "-I", "-B", str(script)], **popen_kwargs)
    except Exception as exc:
        return SkidlResult(success=False, error=f"SKiDL execution exception: {exc}")

    if os.name == "nt":
        applied = _apply_windows_job_limits(proc)
        if not applied:
            warnings.append("Windows JobObject limits unavailable; using timeout fallback")

    try:
        stdout, stderr = proc.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        return SkidlResult(success=False, warnings=warnings, error="SKiDL execution timed out")

    if proc.returncode != 0:
        err = (stderr or stdout or "unknown error").strip()
        return SkidlResult(success=False, warnings=warnings, error=f"SKiDL failed: {err[:400]}")

    net_files = list(run_dir.glob("*.net"))
    if not net_files:
        return SkidlResult(success=False, warnings=warnings, error="SKiDL produced no .net file")

    try:
        circuit_data = parse_netlist(net_files[0], prompt)
    except Exception as exc:
        return SkidlResult(success=False, warnings=warnings, error=f"Netlist parse failed: {exc}")

    warnings.append("Used SKiDL generation pipeline")
    return SkidlResult(success=True, circuit_data=circuit_data, warnings=warnings)
