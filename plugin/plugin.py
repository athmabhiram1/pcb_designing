"""Thin plugin entrypoint.

Runtime implementation lives in modular packages:
- transport.backend_client
- board.board_ops
- ui.main_window
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import pcbnew
import wx

from ._legacy_impl import CONFIG
from .board.board_ops import BoardImporter, BoardReader, BoardWriter
from .transport.backend_client import AsyncHTTPWorker, BackendClient
from .ui.main_window import AIPlacementPlugin as _LegacyAIPlacementPlugin
from .ui.main_window import AIPCBFrame


def _pid_file_path() -> Path:
    path = Path(os.path.expanduser("~")) / ".ai_pcb_assistant"
    path.mkdir(parents=True, exist_ok=True)
    return path / "backend.pid"


def _find_ai_backend_dir() -> Optional[Path]:
    here = Path(__file__).resolve().parent
    for cur in [here] + list(here.parents):
        direct = cur / "ai_server.py"
        if direct.exists():
            return cur
        nested = cur / "ai_backend" / "ai_server.py"
        if nested.exists():
            return nested.parent
    return None


def _backend_manual_start_message(backend_dir: Optional[Path]) -> str:
    location = str(backend_dir) if backend_dir else "<not found>"
    return (
        "Backend is unreachable and auto-start failed.\n\n"
        "Manual start command:\n"
        "python -m uvicorn ai_server:app --host 127.0.0.1 --port 8765\n\n"
        f"Run it from:\n{location}"
    )


def _check_health_sync(client: BackendClient) -> bool:
    method = getattr(client, "check_health_sync", None)
    if callable(method):
        try:
            return bool(method())
        except Exception:
            return False
    try:
        return bool(client.health())
    except Exception:
        return False


class AIPlacementPlugin(_LegacyAIPlacementPlugin):
    def _prompt_backend_stop(self) -> None:
        pid_file = _pid_file_path()
        if not pid_file.exists():
            return

        pid: Optional[int] = None
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
        except Exception:
            pid = None

        if pid is None:
            try:
                pid_file.unlink()
            except Exception:
                pass
            return

        dlg = wx.MessageDialog(
            None,
            f"Stop the backend server when closing? (PID: {pid})",
            "AI PCB Assistant",
            wx.YES_NO | wx.ICON_QUESTION,
        )
        try:
            if dlg.ShowModal() == wx.ID_YES:
                try:
                    if os.name == "nt":
                        subprocess.run(["taskkill", "/PID", str(pid), "/F"], check=False)
                    else:
                        os.kill(pid, signal.SIGTERM)
                except Exception:
                    pass
        finally:
            dlg.Destroy()
            try:
                pid_file.unlink()
            except Exception:
                pass

    def _on_frame_close(self, event: wx.CloseEvent) -> None:
        self._prompt_backend_stop()
        event.Skip()

    def Run(self) -> None:  # noqa: N802 - KiCad API method name
        board = pcbnew.GetBoard()
        if board is None:
            wx.MessageBox("No board is open.", "Error", wx.OK | wx.ICON_ERROR)
            return

        existing: Optional[AIPCBFrame] = getattr(self, "_frame", None)
        if existing is not None:
            try:
                if existing.IsShown():
                    existing.board = board
                    existing._extract_board_data()
                    existing.Raise()
                    return
            except Exception:
                pass
            self._frame = None

        client = BackendClient(base_url=CONFIG.backend_url)
        healthy = _check_health_sync(client)

        if not healthy:
            backend_dir = _find_ai_backend_dir()
            if backend_dir is None:
                wx.MessageDialog(
                    None,
                    _backend_manual_start_message(None),
                    "Backend Not Reachable",
                    wx.OK | wx.ICON_WARNING,
                ).ShowModal()
                return

            try:
                proc = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "uvicorn",
                        "ai_server:app",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        "8765",
                    ],
                    cwd=str(backend_dir),
                )
                _pid_file_path().write_text(str(proc.pid), encoding="utf-8")
            except Exception:
                wx.MessageDialog(
                    None,
                    _backend_manual_start_message(backend_dir),
                    "Backend Not Reachable",
                    wx.OK | wx.ICON_WARNING,
                ).ShowModal()
                return

            for _ in range(10):
                time.sleep(1)
                if _check_health_sync(client):
                    healthy = True
                    break

            if not healthy:
                wx.MessageDialog(
                    None,
                    _backend_manual_start_message(backend_dir),
                    "Backend Not Reachable",
                    wx.OK | wx.ICON_WARNING,
                ).ShowModal()
                return

        self._frame = AIPCBFrame(None, board)
        self._frame.Bind(wx.EVT_CLOSE, self._on_frame_close)
        self._frame.Show()
        self._frame.Raise()


__all__ = [
    "AIPlacementPlugin",
    "AIPCBFrame",
    "BackendClient",
    "AsyncHTTPWorker",
    "BoardReader",
    "BoardWriter",
    "BoardImporter",
]
