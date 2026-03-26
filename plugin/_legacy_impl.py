"""
AI PCB Assistant â€” Advanced KiCad Action Plugin v3.0
Professional-grade PCB design automation with real-time AI assistance.

Key improvements over v2.0:
  â€¢ Tabbed UI: Assistant / Components / Nets / DFM / History
  â€¢ Per-request cancel tokens + exponential-backoff retry
  â€¢ Thermal heat-map overlay on canvas
  â€¢ Component search/filter bar with column sort
  â€¢ Net statistics table (length, type, pin-count, criticality)
  â€¢ Detailed DFM panel â€” clickable violations â†’ board highlight
  â€¢ Session history log with export
  â€¢ Progress gauge for long operations
  â€¢ Constraint enable/disable toggle in list
  â€¢ Export DFM / placement report to CSV
  â€¢ KiCad commit-based undo (SaveBoard fallback)
  â€¢ Robust footprint candidate fallback chain
  â€¢ Multi-level status bar (op | board stats | backend status)
  â€¢ All magic strings replaced with named constants
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import queue
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin

import pcbnew
import wx

try:
    import wx.lib.scrolledpanel as _sp
    _ScrolledPanel = _sp.ScrolledPanel
except Exception:
    _ScrolledPanel = wx.ScrolledWindow

try:
    from wx.lib.floatcanvas import NavCanvas as _NC
    HAS_FLOATCANVAS = True
except Exception:
    _NC = None
    HAS_FLOATCANVAS = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# â”€â”€ Constants â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

PLUGIN_VERSION       = "3.2"
DEFAULT_BACKEND_URL  = os.environ.get("AI_PCB_BACKEND_URL", "http://localhost:8765")
TIMER_MS             = 100
MAX_DFM_SHOWN        = 50
MAX_HISTORY          = 100
RETRY_ATTEMPTS       = 3
RETRY_BASE_S         = 0.5
ASSISTANT_TASKS      = ["Generate Circuit", "Optimize Placement", "Run DFM Check"]
ASSISTANT_PRIORITIES = ["quality", "balanced", "speed"]

# Component prefix â†’ colour (R,G,B)
PREFIX_COLOURS: Dict[str, Tuple[int,int,int]] = {
    "U":  (60,  60,  200),
    "R":  (100, 160, 220),
    "C":  (200, 200,  60),
    "L":  (220, 140,  40),
    "J":  (60,  160,  80),
    "P":  (60,  160,  80),
    "D":  (200,  80, 200),
    "Q":  (80,  200, 200),
    "M":  (80,  200, 200),
    "Y":  (200, 200, 200),
}

# â”€â”€ Configuration â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@dataclass
class PluginConfig:
    backend_url:       str   = DEFAULT_BACKEND_URL
    request_timeout:   int   = 120
    show_ratsnest:     bool  = True
    show_thermal:      bool  = True
    thermal_aware:     bool  = True
    placement_step_mm: float = 1.0
    auto_refresh:      bool  = True

    config_dir: str = field(default_factory=lambda:
        os.path.join(os.path.expanduser("~"), ".ai_pcb_assistant"))

    def __post_init__(self):
        os.makedirs(self.config_dir, exist_ok=True)

    def save(self):
        path = os.path.join(self.config_dir, "config.json")
        safe = {k: v for k, v in self.__dict__.items() if k != "config_dir"}
        with open(path, "w") as f:
            json.dump(safe, f, indent=2)

    @classmethod
    def load(cls) -> "PluginConfig":
        path = os.path.join(os.path.expanduser("~"), ".ai_pcb_assistant", "config.json")
        if os.path.exists(path):
            try:
                with open(path) as f:
                    d = json.load(f)
                valid = {k: v for k, v in d.items()
                         if k in cls.__dataclass_fields__}   # type: ignore[attr-defined]
                return cls(**valid)
            except Exception:
                pass
        return cls()

CONFIG = PluginConfig.load()


# â”€â”€ Data Structures â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class NetType(Enum):
    SIGNAL       = auto()
    POWER        = auto()
    GROUND       = auto()
    CLOCK        = auto()
    DIFFERENTIAL = auto()
    HIGH_SPEED   = auto()
    ANALOG       = auto()

    @property
    def wx_colour(self) -> wx.Colour:
        return wx.Colour(*{
            NetType.POWER:        (220,  60,  60),
            NetType.GROUND:       ( 60, 180,  60),
            NetType.CLOCK:        (180,  60, 220),
            NetType.DIFFERENTIAL: (220, 140,  40),
            NetType.HIGH_SPEED:   ( 40, 160, 220),
            NetType.ANALOG:       (220, 200,  40),
            NetType.SIGNAL:       (130, 130, 130),
        }.get(self, (130, 130, 130)))

    @property
    def fc_colour(self) -> str:
        return {
            NetType.POWER:        "red",
            NetType.GROUND:       "green",
            NetType.CLOCK:        "purple",
            NetType.DIFFERENTIAL: "orange",
            NetType.HIGH_SPEED:   "cyan",
            NetType.ANALOG:       "yellow",
            NetType.SIGNAL:       "lightgray",
        }.get(self, "lightgray")


@dataclass
class NetInfo:
    name:        str
    code:        int
    net_type:    NetType
    pins:        List[Dict[str, str]] = field(default_factory=list)
    length_mm:   float = 0.0
    is_critical: bool  = False

    @property
    def component_count(self) -> int:
        return len({p["ref"] for p in self.pins})


@dataclass
class ComponentInfo:
    ref:               str
    value:             str
    footprint:         str
    x:                 float
    y:                 float
    rotation:          float
    layer:             str
    width:             float = 0.0
    height:            float = 0.0
    pins:              List[Dict] = field(default_factory=list)
    power_dissipation: float = 0.0
    is_fixed:          bool  = False
    cluster_id:        Optional[str] = None

    @property
    def prefix(self) -> str:
        i = 0
        while i < len(self.ref) and self.ref[i].isalpha():
            i += 1
        return self.ref[:i].upper() or "?"

    @property
    def wx_colour(self) -> wx.Colour:
        r, g, b = PREFIX_COLOURS.get(self.prefix[:1], (120, 120, 120))
        return wx.Colour(r, g, b)

    @property
    def fc_colour(self) -> str:
        return {
            "U": "blue",  "R": "lightblue", "C": "yellow",
            "L": "orange","J": "green",      "P": "green",
            "D": "violet","Q": "cyan",       "M": "cyan",
        }.get(self.prefix[:1], "gray")


@dataclass
class Constraint:
    ctype:   str           # "fixed" | "spacing" | "region" | "alignment"
    refs:    List[str]
    params:  Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    def display(self) -> str:
        st   = "âœ“" if self.enabled else "âœ—"
        refs = ", ".join(self.refs)
        suf  = (f" (min={self.params['min_mm']} mm)"
                if "min_mm" in self.params else "")
        return f"[{st}] {self.ctype}: {refs}{suf}"


@dataclass
class DFMViolation:
    severity:   str
    vtype:      str
    message:    str
    components: List[str] = field(default_factory=list)

    @property
    def icon(self) -> str:
        return {"critical":"âœ–","error":"âœ–","warning":"âš ","info":"â„¹"}.get(
            self.severity.lower(), "â€¢")

    @property
    def wx_colour(self) -> wx.Colour:
        return {
            "critical": wx.Colour(239, 68,  68),
            "error":    wx.Colour(239, 68,  68),
            "warning":  wx.Colour(245,158,  11),
            "info":     wx.Colour( 56,189, 248),
        }.get(self.severity.lower(), wx.Colour(100,116,139))


@dataclass
class HistoryEntry:
    prompt:    str
    result:    str
    req_type:  str
    timestamp: float = field(default_factory=time.time)

    def header(self) -> str:
        t = time.strftime("%H:%M:%S", time.localtime(self.timestamp))
        return f"[{t}] [{self.req_type.upper()}] {self.prompt[:70]}"


# â”€â”€ Cancel Token â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class CancelToken:
    def __init__(self):
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled


# â”€â”€ Async HTTP Client with Retry â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class AsyncHTTPClient:
    """Non-blocking HTTP client with per-request cancel tokens and retry."""

    def __init__(self):
        self._q: queue.Queue = queue.Queue()
        self._results: Dict[str, Tuple[str, Any]] = {}
        self._lock = threading.Lock()
        t = threading.Thread(target=self._worker, daemon=True)
        t.start()

    def _worker(self):
        while True:
            item = self._q.get()
            if item is None:
                break
            req_id, url, data, callback, token = item
            result = self._do_request(url, data, token)
            with self._lock:
                self._results[req_id] = result
            if callback:
                wx.CallAfter(callback, req_id)

    def _do_request(self, url: str, data: Optional[Dict],
                    token: CancelToken) -> Tuple[str, Any]:
        last_err = "Unknown"
        for attempt in range(RETRY_ATTEMPTS):
            if token.is_cancelled:
                return ("cancelled", None)
            if attempt > 0:
                time.sleep(RETRY_BASE_S * (2 ** (attempt - 1)))
            try:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(data).encode() if data else None,
                    headers={"Content-Type": "application/json"},
                    method="POST" if data else "GET",
                )
                with urllib.request.urlopen(req, timeout=CONFIG.request_timeout) as resp:
                    return ("success", json.loads(resp.read().decode()))
            except urllib.error.HTTPError as e:
                try:
                    body = e.read().decode("utf-8", errors="replace")
                    parsed = json.loads(body)
                    detail = parsed.get("detail", parsed)
                    last_err = f"HTTP {e.code}: {detail}"
                except Exception:
                    last_err = f"HTTP {e.code}: {e.reason}"
                # 4xx â†’ no retry
                if 400 <= e.code < 500:
                    break
            except Exception as exc:
                last_err = str(exc)
        return ("error", last_err)

    def request(self, url: str, data: Optional[Dict] = None,
                callback: Optional[Callable] = None,
                token: Optional[CancelToken] = None) -> str:
        req_id = f"{time.monotonic():.9f}"
        self._q.put((req_id, url, data, callback, token or CancelToken()))
        return req_id

    def get_result(self, req_id: str) -> Tuple[str, Any]:
        with self._lock:
            return self._results.pop(req_id, ("pending", None))

    def close(self):
        self._q.put(None)


HTTP_CLIENT = AsyncHTTPClient()


# â”€â”€ Placement Preview Canvas â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

if HAS_FLOATCANVAS:
    _CanvasBase = _NC.NavCanvas  # type: ignore[union-attr]
else:
    _CanvasBase = wx.ScrolledWindow


class PlacementPreviewCanvas(_CanvasBase):
    """Interactive canvas with ratsnest and thermal overlay."""

    def __init__(self, parent):
        if HAS_FLOATCANVAS:
            super().__init__(parent, size=(500, 500))
        else:
            super().__init__(parent, size=(500, 500),
                             style=wx.HSCROLL | wx.VSCROLL)
            self.SetScrollRate(5, 5)
            self.SetBackgroundColour(wx.Colour(24, 36, 54))
            self.Bind(wx.EVT_PAINT, self._on_paint)

        self.board_width:   float = 100.0
        self.board_height:  float =  80.0
        self.components:    Dict[str, ComponentInfo] = {}
        self.nets:          List[NetInfo] = []
        self.selected_refs: Set[str] = set()
        self.show_ratsnest: bool = True
        self.show_thermal:  bool = True
        # Thermal map: ref â†’ heat scalar 0..1
        self.thermal_map:   Dict[str, float] = {}
        self.Bind(wx.EVT_SIZE, self._on_size)

    # â”€â”€ Public API â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def set_board_dimensions(self, w: float, h: float):
        self.board_width, self.board_height = w, h
        self._redraw()

    def update_components(self, comps: List[ComponentInfo],
                          nets: Optional[List[NetInfo]] = None):
        self.components = {c.ref: c for c in comps}
        if nets is not None:
            self.nets = nets
        self._rebuild_thermal()
        self._redraw()

    def highlight_refs(self, refs: Set[str]):
        self.selected_refs = refs
        self._redraw()

    def clear_highlights(self):
        self.selected_refs.clear()
        self._redraw()

    # â”€â”€ Internal â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _rebuild_thermal(self):
        """Recompute thermal map from power_dissipation values."""
        if not self.components:
            self.thermal_map = {}
            return
        maxp = max((c.power_dissipation for c in self.components.values()), default=0.0)
        if maxp <= 0:
            self.thermal_map = {}
            return
        self.thermal_map = {
            ref: min(1.0, c.power_dissipation / maxp)
            for ref, c in self.components.items()
        }

    def _redraw(self):
        if HAS_FLOATCANVAS:
            self._draw_fc()
        else:
            self.Refresh()

    def _guide_step_mm(self) -> float:
        span = max(self.board_width, self.board_height, 1.0)
        for step in (5.0, 10.0, 20.0, 25.0, 50.0, 100.0):
            if span / step <= 24:
                return step
        return 100.0

    # â”€â”€ FloatCanvas rendering â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _draw_fc(self):
        if not HAS_FLOATCANVAS:
            return
        self.Canvas.ClearAll()
        bw = self.board_width
        bh = self.board_height
        sc = 4.0

        def to_fc(x, y):
            return ((x - bw / 2) * sc, -(y - bh / 2) * sc)

        # Board outline and neutral background; avoid dense crosshatch mesh.
        w2, h2 = bw * sc, bh * sc
        self.Canvas.AddRectangle((-w2/2, -h2/2), (w2, h2),
            LineColor="#334155", LineWidth=2, FillColor="#f8fafc",
            FillStyle="Solid")

        step_mm = self._guide_step_mm()
        x = 0.0
        while x <= bw:
            sx1, sy1 = to_fc(x, 0.0)
            sx2, sy2 = to_fc(x, bh)
            self.Canvas.AddLine([(sx1, sy1), (sx2, sy2)],
                LineColor="#e2e8f0", LineWidth=1)
            x += step_mm
        y = 0.0
        while y <= bh:
            sx1, sy1 = to_fc(0.0, y)
            sx2, sy2 = to_fc(bw, y)
            self.Canvas.AddLine([(sx1, sy1), (sx2, sy2)],
                LineColor="#e2e8f0", LineWidth=1)
            y += step_mm

        # Ratsnest
        if self.show_ratsnest:
            for net in self.nets:
                positions = [to_fc(self.components[p["ref"]].x,
                                   self.components[p["ref"]].y)
                             for p in net.pins if p["ref"] in self.components]
                for i in range(len(positions) - 1):
                    self.Canvas.AddLine([positions[i], positions[i+1]],
                        LineColor=net.net_type.fc_colour,
                        LineWidth=1, LineStyle="Dot")

        # Components
        for ref, comp in self.components.items():
            cx, cy = to_fc(comp.x, comp.y)
            cw = max(2.0, comp.width * sc * 0.8)
            ch = max(2.0, comp.height * sc * 0.8)
            fill = "red" if ref in self.selected_refs else comp.fc_colour
            if self.show_thermal and ref in self.thermal_map:
                heat = self.thermal_map[ref]
                r = int(60 + 180 * heat)
                g = int(160 * (1 - heat))
                b = 40
                fill = wx.Colour(r, g, b).GetAsString(wx.C2S_HTML_SYNTAX)
            self.Canvas.AddRectangle((cx - cw/2, cy - ch/2), (cw, ch),
                LineColor="black", LineWidth=1, FillColor=fill)
            self.Canvas.AddText(ref, (cx, cy), Size=7, Color="white", Position="cc")

        if not self.components:
            self.Canvas.AddText(
                "No components on board yet.\nGenerate a circuit or import footprints.",
                (0, 0),
                Size=12,
                Color="#64748b",
                Position="cc",
            )

        self.Canvas.AddText(
            f"Board {bw:.1f} mm x {bh:.1f} mm   Grid {step_mm:.0f} mm",
            (-w2 / 2 + 8, h2 / 2 - 12),
            Size=8,
            Color="#475569",
            Position="tl",
        )

        self.Canvas.ZoomToBB()
        self.Canvas.Draw()

    # â”€â”€ DC fallback rendering â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _on_paint(self, event):
        dc = wx.PaintDC(self)
        self.DoPrepareDC(dc)
        cw, ch = self.GetClientSize()
        dc.SetBackground(wx.Brush(wx.Colour(243, 244, 246)))
        dc.Clear()
        if cw <= 0 or ch <= 0:
            return

        margin = 16
        pw = max(1, cw - 2 * margin)
        ph = max(1, ch - 2 * margin)

        # Board edge and clean neutral fill.
        dc.SetPen(wx.Pen(wx.Colour(51, 65, 85), 2))
        dc.SetBrush(wx.Brush(wx.Colour(248, 250, 252)))
        dc.DrawRectangle(margin, margin, pw, ph)

        bw = self.board_width  or 100.0
        bh = self.board_height or  80.0

        def to_px(x_mm: float, y_mm: float) -> Tuple[int, int]:
            return (int(margin + (x_mm / bw) * pw),
                    int(margin + (y_mm / bh) * ph))

        step_mm = self._guide_step_mm()
        dc.SetPen(wx.Pen(wx.Colour(226, 232, 240), 1))
        x = 0.0
        while x <= bw:
            x0, y0 = to_px(x, 0.0)
            x1, y1 = to_px(x, bh)
            dc.DrawLine(x0, y0, x1, y1)
            x += step_mm
        y = 0.0
        while y <= bh:
            x0, y0 = to_px(0.0, y)
            x1, y1 = to_px(bw, y)
            dc.DrawLine(x0, y0, x1, y1)
            y += step_mm

        # Ratsnest
        if self.show_ratsnest:
            for net in self.nets:
                pts = [to_px(self.components[p["ref"]].x,
                             self.components[p["ref"]].y)
                       for p in net.pins if p["ref"] in self.components]
                if len(pts) >= 2:
                    dc.SetPen(wx.Pen(net.net_type.wx_colour, 1, wx.PENSTYLE_DOT))
                    for i in range(len(pts) - 1):
                        dc.DrawLine(*pts[i], *pts[i+1])

        # Components
        for ref, comp in self.components.items():
            px, py = to_px(comp.x, comp.y)
            rw = max(5, int((comp.width  / bw) * pw * 0.8))
            rh = max(5, int((comp.height / bh) * ph * 0.8))

            if ref in self.selected_refs:
                colour = wx.Colour(239, 68, 68)
            elif self.show_thermal and ref in self.thermal_map:
                heat = self.thermal_map[ref]
                colour = wx.Colour(int(60 + 180*heat), int(160*(1-heat)), 40)
            else:
                colour = comp.wx_colour

            dc.SetPen(wx.Pen(wx.BLACK, 1))
            dc.SetBrush(wx.Brush(colour))
            dc.DrawRoundedRectangle(px - rw//2, py - rh//2, rw, rh, 2)
            dc.SetTextForeground(wx.Colour(15, 23, 42))
            dc.SetFont(wx.Font(6, wx.FONTFAMILY_DEFAULT,
                               wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
            dc.DrawText(ref, px - rw//2 + 1, py - rh//2 + 1)

        if not self.components:
            dc.SetTextForeground(wx.Colour(100, 116, 139))
            dc.SetFont(wx.Font(10, wx.FONTFAMILY_DEFAULT,
                               wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
            dc.DrawText(
                "No components on board yet. Generate a circuit or import footprints.",
                margin + 12,
                margin + 12,
            )

    def _on_size(self, event):
        if HAS_FLOATCANVAS:
            try:
                self.Canvas.ZoomToBB()
            except Exception:
                pass
        else:
            self.Refresh()
        event.Skip()


class _NullCanvas:
    """No-op canvas used for minimal UI mode."""

    show_ratsnest: bool = False
    show_thermal: bool = False

    def set_board_dimensions(self, _w: float, _h: float):
        return

    def update_components(self, _comps: List[ComponentInfo], _nets: Optional[List[NetInfo]] = None):
        return

    def highlight_refs(self, _refs: Set[str]):
        return

    def clear_highlights(self):
        return

    def _redraw(self):
        return


# â”€â”€ Net Classifier â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_NET_RULES: List[Tuple[NetType, List[str]]] = [
    (NetType.POWER,       ["VCC","VDD","3V3","5V","1V8","1V2","3V","12V","PWR","AVDD","DVDD"]),
    (NetType.GROUND,      ["GND","VSS","AGND","DGND","PGND","SGND"]),
    (NetType.CLOCK,       ["CLK","CLOCK","OSC","XTAL","SCK","SCLK"]),
    (NetType.DIFFERENTIAL,["_DP","_DN","_P","_N","USB_D","CAN_","LVDS","DIFF"]),
    (NetType.HIGH_SPEED,  ["USB","HDMI","ETH","LVDS","PCIe","DDR","MIPI"]),
    (NetType.ANALOG,      ["ADC","DAC","ANA","VREF","SENSOR","AIN","AOUT"]),
]

def classify_net(name: str) -> NetType:
    u = name.upper()
    for net_type, patterns in _NET_RULES:
        if any(p in u for p in patterns):
            return net_type
    return NetType.SIGNAL


# â”€â”€ KiCad API Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _get_footprint_name(fp: Any) -> str:
    try:
        fpid = fp.GetFPID()
        for attr in ("GetLibItemName", "GetUniStringLibItemName"):
            if hasattr(fpid, attr):
                return str(getattr(fpid, attr)())
    except Exception:
        pass
    return ""


def _get_orientation_degrees(fp: Any) -> float:
    for method in ("GetOrientation",):
        try:
            val = getattr(fp, method)()
            if hasattr(val, "AsDegrees"):
                return val.AsDegrees()
            return float(val) / 10.0
        except Exception:
            pass
    try:
        return float(fp.GetOrientationDegrees())
    except Exception:
        return 0.0


def _check_backend_url(url: str, timeout: int = 3) -> bool:
    try:
        req = urllib.request.Request(f"{url}/health", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode()).get("status") in (
                "ok", "healthy", "degraded")
    except Exception:
        return False


def _iter_board_footprints(board: Any) -> List[Any]:
    """Return footprints for KiCad 6/7/8/9 API variants."""
    seen: Set[int] = set()
    out: List[Any] = []

    for method in ("GetFootprints", "GetModules", "Footprints", "Modules"):
        try:
            if not hasattr(board, method):
                continue
            raw = getattr(board, method)
            items = raw() if callable(raw) else raw
            for fp in list(items):
                key = id(fp)
                if key in seen:
                    continue
                seen.add(key)
                out.append(fp)
        except Exception:
            continue

    # Last-resort fallback: derive owning footprints from pads.
    if not out:
        for pad_method in ("GetPads", "Pads"):
            try:
                if not hasattr(board, pad_method):
                    continue
                raw = getattr(board, pad_method)
                pads = raw() if callable(raw) else raw
                for pad in list(pads):
                    try:
                        fp = pad.GetParent()
                    except Exception:
                        continue
                    key = id(fp)
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(fp)
            except Exception:
                continue

    if out:
        return out
    return []


def _iter_board_nets(board: Any) -> List[Tuple[str, Any]]:
    """Return (net_name, net_obj) pairs across KiCad API variants."""
    out: List[Tuple[str, Any]] = []
    seen: Set[int] = set()

    # Variant 1: board.GetNetInfo().NetsByName()
    try:
        net_info = board.GetNetInfo() if hasattr(board, "GetNetInfo") else None
        nets_by_name = net_info.NetsByName() if net_info and hasattr(net_info, "NetsByName") else None
        if nets_by_name is not None:
            if hasattr(nets_by_name, "items"):
                items = list(nets_by_name.items())
            else:
                items = []
                for key in list(nets_by_name):
                    try:
                        items.append((key, nets_by_name[key]))
                    except Exception:
                        continue
            for name, net in items:
                key = id(net)
                if key in seen:
                    continue
                seen.add(key)
                out.append((str(name), net))
    except Exception:
        pass

    # Variant 2: board.GetNetsByName()
    try:
        if hasattr(board, "GetNetsByName"):
            nets_by_name = board.GetNetsByName()
            items = list(nets_by_name.items()) if hasattr(nets_by_name, "items") else []
            for name, net in items:
                key = id(net)
                if key in seen:
                    continue
                seen.add(key)
                out.append((str(name), net))
    except Exception:
        pass

    return out


def _build_connections_from_board(board: Any) -> List[Dict[str, Any]]:
    """Build backend-ready connections from board nets with multiple fallbacks."""
    connections: List[Dict[str, Any]] = []

    try:
        if hasattr(board, "BuildConnectivity"):
            board.BuildConnectivity()
    except Exception:
        pass

    # Primary: enumerate KiCad net objects.
    for net_name, net in _iter_board_nets(board):
        try:
            if hasattr(net, "GetNetCode") and int(net.GetNetCode()) == 0:
                continue
            pins: List[Dict[str, str]] = []
            pads = list(net.GetPads()) if hasattr(net, "GetPads") else []
            for pad in pads:
                try:
                    pins.append(
                        {
                            "ref": pad.GetParent().GetReference(),
                            "pin": pad.GetNumber(),
                        }
                    )
                except Exception:
                    continue
            if len(pins) >= 2:
                connections.append(
                    {
                        "net": str(net_name),
                        "net_type": classify_net(str(net_name)).name.lower(),
                        "pins": pins,
                        "is_critical": False,
                    }
                )
        except Exception:
            continue

    if connections:
        return connections

    # Fallback: derive nets by grouping pads.
    by_name: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for fp in _iter_board_footprints(board):
        try:
            pads = list(fp.GetPads()) if hasattr(fp, "GetPads") else []
        except Exception:
            pads = []
        for pad in pads:
            try:
                code = int(pad.GetNetCode())
                name = str(pad.GetNetname()).strip() if hasattr(pad, "GetNetname") else ""
                if (code == 0 and not name) or name.lower().startswith("unconnected"):
                    name = f"Net-{code}"
                if code == 0 and name == "Net-0":
                    continue
                by_name[name].append(
                    {
                        "ref": pad.GetParent().GetReference(),
                        "pin": pad.GetNumber(),
                    }
                )
            except Exception:
                continue

    for name, pins in by_name.items():
        if len(pins) >= 2:
            connections.append(
                {
                    "net": name,
                    "net_type": classify_net(name).name.lower(),
                    "pins": pins,
                    "is_critical": False,
                }
            )

    return connections


# â”€â”€ Footprint Loader â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_FOOTPRINT_FALLBACKS: Dict[str, List[str]] = {
    "R": ["Resistor_SMD:R_0805_2012Metric",
          "Resistor_THT:R_Axial_DIN0207_L6.3mm_D2.5mm_P7.62mm_Horizontal"],
    "C": ["Capacitor_SMD:C_0805_2012Metric",
          "Capacitor_THT:C_Disc_D5.0mm_W2.5mm_P5.00mm"],
    "L": ["Inductor_SMD:L_0805_2012Metric",
          "Inductor_THT:L_Axial_L9.0mm_D3.5mm_P15.24mm_Horizontal"],
    "D": ["LED_SMD:LED_0805_2012Metric", "LED_THT:LED_D5.0mm",
          "Diode_SMD:D_SOD-123"],
    "Q": ["Package_TO_SOT_SMD:SOT-23", "Package_TO_SOT_THT:TO-92_Inline"],
    "M": ["Package_TO_SOT_SMD:SOT-23", "Package_TO_SOT_THT:TO-92_Inline"],
    "J": ["Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical"],
    "P": ["Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical"],
    "U": ["Package_DIP:DIP-8_W7.62mm", "Package_SO:SOIC-8_3.9x4.9mm_P1.27mm"],
}


def load_footprint_by_id(fp_id: str) -> Tuple[Optional[Any], str]:
    if not fp_id or ":" not in fp_id:
        return None, "invalid id (expected Library:Footprint)"
    lib, name = (s.strip() for s in fp_id.split(":", 1))
    if not lib or not name:
        return None, "empty lib or name"
    errors: List[str] = []

    # 1) Try by library nickname from fp-lib-table.
    try:
        fp = pcbnew.FootprintLoad(lib, name)
        if fp:
            return fp, ""
        errors.append("nickname lookup returned None")
    except Exception as exc:
        errors.append(str(exc))

    # 2) Try absolute .pretty paths via KiCad footprint root environment variables.
    env_keys = [
        "KICAD9_FOOTPRINT_DIR",
        "KICAD8_FOOTPRINT_DIR",
        "KICAD7_FOOTPRINT_DIR",
        "KICAD6_FOOTPRINT_DIR",
        "KICAD_FOOTPRINT_DIR",
    ]
    roots = [os.environ.get(k, "").strip() for k in env_keys]
    roots = [r for r in roots if r]

    # 3) Fallback to common KiCad install locations when env vars are not set.
    if not roots and os.name == "nt":
        pf = os.environ.get("ProgramFiles", r"C:\Program Files")
        roots.extend([
            os.path.join(pf, "KiCad", "9.0", "share", "kicad", "footprints"),
            os.path.join(pf, "KiCad", "8.0", "share", "kicad", "footprints"),
            os.path.join(pf, "KiCad", "7.0", "share", "kicad", "footprints"),
        ])

    # Deduplicate and keep existing folders only.
    uniq_roots: List[str] = []
    for root in roots:
        if root and root not in uniq_roots and os.path.isdir(root):
            uniq_roots.append(root)
    roots = uniq_roots

    for root in roots:
        pretty_dir = os.path.join(root, f"{lib}.pretty")
        if not os.path.isdir(pretty_dir):
            continue
        try:
            fp = pcbnew.FootprintLoad(pretty_dir, name)
            if fp:
                return fp, ""
            errors.append(f"{pretty_dir}: returned None")
        except Exception as exc:
            errors.append(f"{pretty_dir}: {exc}")

    if not errors:
        return None, "footprint not found"
    return None, "; ".join(errors[:3])


def load_footprint_for_component(comp: Dict[str, Any]) -> Tuple[Optional[Any], str]:
    primary = str(comp.get("footprint", "")).strip()
    ref     = str(comp.get("ref",       "")).strip().upper()
    prefix  = "".join(c for c in ref if c.isalpha())[:1]

    candidates: List[str] = []
    if primary:
        candidates.append(primary)
    candidates.extend(_FOOTPRINT_FALLBACKS.get(prefix, []))

    seen: Set[str] = set()
    failures: List[str] = []
    for cid in candidates:
        if not cid or cid in seen:
            continue
        seen.add(cid)
        fp, err = load_footprint_by_id(cid)
        if fp is not None:
            return fp, ""
        failures.append(f"{cid}: {err}")
    if not failures:
        return None, "no footprint candidates"
    return None, "; ".join(failures[:2])


# â”€â”€ Board Net Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def find_board_net(board: Any, net_name: str) -> Optional[Any]:
    try:
        n = board.FindNet(net_name)
        if n:
            return n
    except Exception:
        pass
    try:
        for name, net in board.GetNetInfo().NetsByName().items():
            if str(name) == net_name:
                return net
    except Exception:
        pass
    return None


def ensure_board_net(board: Any, net_name: str) -> Any:
    existing = find_board_net(board, net_name)
    if existing:
        return existing
    net = pcbnew.NETINFO_ITEM(board, net_name)
    board.Add(net)
    return net


def _normalized_pin_token(pin: str) -> str:
    return re.sub(r"[^A-Z0-9+]", "", str(pin).upper())


def _pin_alias_candidates(pin: str) -> List[str]:
    token = _normalized_pin_token(pin)
    if not token:
        return []

    # Exact numeric pin identifiers should always win first.
    if token.isdigit():
        return [token]

    # Heuristics for common named pins emitted by LLM JSON paths.
    aliases: Dict[str, List[str]] = {
        "ANODE": ["1"],
        "A": ["1"],
        "CATHODE": ["2"],
        "K": ["2"],
        "GATE": ["1", "2"],
        "G": ["1", "2"],
        "SOURCE": ["2", "1"],
        "S": ["2", "1"],
        "DRAIN": ["3"],
        "D": ["3"],
        "IN": ["1", "3"],
        "VIN": ["1", "3"],
        "OUT": ["2"],
        "VOUT": ["2"],
        "PLUS": ["1"],
        "MINUS": ["2"],
    }
    out = aliases.get(token, []).copy()

    # Allow labels like PIN1 / PAD2.
    m = re.search(r"(\d+)$", token)
    if m:
        out.insert(0, m.group(1))

    # De-duplicate while preserving priority.
    dedup: List[str] = []
    for n in out:
        if n and n not in dedup:
            dedup.append(n)
    return dedup


def _resolve_pad_for_pin(fp: Any, pin: str) -> Optional[Any]:
    pin_raw = str(pin).strip()
    if not pin_raw:
        return None

    try:
        pad = fp.FindPadByNumber(pin_raw)
        if pad:
            return pad
    except Exception:
        pass

    for candidate in _pin_alias_candidates(pin_raw):
        try:
            pad = fp.FindPadByNumber(candidate)
            if pad:
                return pad
        except Exception:
            continue

    return None


# â”€â”€ Backend Setup Dialog â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class BackendSetupDialog(wx.Dialog):
    def __init__(self, parent):
        super().__init__(parent, title="AI Backend Setup", size=(520, 320))
        panel = wx.Panel(self)
        vs    = wx.BoxSizer(wx.VERTICAL)

        hdr = wx.StaticText(panel, label="âš¡  AI Backend Not Detected")
        hdr.SetFont(wx.Font(14, wx.FONTFAMILY_DEFAULT,
                            wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        vs.Add(hdr, 0, wx.ALL | wx.ALIGN_CENTER, 12)

        vs.Add(wx.StaticText(panel, label="Backend URL:"), 0, wx.LEFT, 12)
        self.txt_url = wx.TextCtrl(panel, value=CONFIG.backend_url)
        vs.Add(self.txt_url, 0, wx.EXPAND | wx.ALL, 6)

        self.rb_local  = wx.RadioButton(panel, label="Start local backend", style=wx.RB_GROUP)
        self.rb_remote = wx.RadioButton(panel, label="Connect to remote URL above")
        vs.Add(self.rb_local,  0, wx.LEFT | wx.TOP, 10)
        vs.Add(self.rb_remote, 0, wx.LEFT | wx.BOTTOM, 10)

        self.chk_install = wx.CheckBox(panel, label="Auto-start backend script if found")
        self.chk_install.SetValue(True)
        vs.Add(self.chk_install, 0, wx.ALL, 8)

        btns = wx.BoxSizer(wx.HORIZONTAL)
        btn_ok  = wx.Button(panel, wx.ID_OK, "Connect")
        btn_can = wx.Button(panel, wx.ID_CANCEL)
        btns.Add(btn_ok,  0, wx.ALL, 5)
        btns.Add(btn_can, 0, wx.ALL, 5)
        vs.Add(btns, 0, wx.ALIGN_CENTER | wx.ALL, 8)

        panel.SetSizer(vs)
        btn_ok.Bind(wx.EVT_BUTTON, self._on_setup)

    def _on_setup(self, event):
        url = self.txt_url.GetValue().strip() or CONFIG.backend_url
        CONFIG.backend_url = url
        CONFIG.save()

        if self.chk_install.GetValue() and self.rb_local.GetValue():
            self._try_start_backend()

        if _check_backend_url(url):
            self.EndModal(wx.ID_OK)
            return

        wx.MessageBox(
            f"Backend not reachable at {url}.\n"
            "Start it manually and click Connect again.",
            "Not Reachable", wx.OK | wx.ICON_WARNING)
        event.Skip(False)

    def _try_start_backend(self):
        scripts = [
            os.path.join(os.path.dirname(__file__), "..", "ai_backend", "start_backend.bat"),
            os.path.join(os.getcwd(), "ai_backend", "start_backend.bat"),
            os.path.join(os.path.dirname(__file__), "..", "ai_backend", "start.sh"),
        ]
        script = next((os.path.abspath(p) for p in scripts if os.path.exists(p)), None)
        if not script:
            wx.MessageBox(
                "Backend script not found.\nStart it manually:\n  cd ai_backend && start_backend.bat",
                "Backend Setup", wx.OK | wx.ICON_INFORMATION)
            return
        try:
            if script.endswith(".bat"):
                subprocess.Popen(["cmd", "/c", script], cwd=os.path.dirname(script))
            else:
                subprocess.Popen(["bash", script], cwd=os.path.dirname(script))
            wx.MessageBox(f"Starting backend:\n{script}\n\nWait a moment, then click Connect.",
                          "Backend Setup", wx.OK | wx.ICON_INFORMATION)
        except Exception as exc:
            wx.MessageBox(f"Failed to start backend:\n{exc}",
                          "Backend Setup Error", wx.OK | wx.ICON_ERROR)


# â”€â”€ Settings Dialog â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class SettingsDialog(wx.Dialog):
    def __init__(self, parent):
        super().__init__(parent, title="AI PCB Assistant â€” Settings", size=(480, 340))
        panel = wx.Panel(self)
        vs    = wx.BoxSizer(wx.VERTICAL)

        fgs = wx.FlexGridSizer(rows=0, cols=2, vgap=8, hgap=12)
        fgs.AddGrowableCol(1)

        def row(label, ctrl):
            fgs.Add(wx.StaticText(panel, label=label),
                    0, wx.ALIGN_CENTER_VERTICAL)
            fgs.Add(ctrl, 0, wx.EXPAND)

        self.txt_url     = wx.TextCtrl(panel, value=CONFIG.backend_url)
        self.txt_timeout = wx.TextCtrl(panel, value=str(CONFIG.request_timeout))
        self.chk_thermal = wx.CheckBox(panel, label="Enabled")
        self.chk_thermal.SetValue(CONFIG.thermal_aware)
        self.chk_ratsnest= wx.CheckBox(panel, label="Enabled")
        self.chk_ratsnest.SetValue(CONFIG.show_ratsnest)
        self.chk_heatmap = wx.CheckBox(panel, label="Enabled")
        self.chk_heatmap.SetValue(CONFIG.show_thermal)
        self.txt_step    = wx.TextCtrl(panel, value=str(CONFIG.placement_step_mm))

        row("Backend URL:",           self.txt_url)
        row("Timeout (s):",           self.txt_timeout)
        row("Placement step (mm):",   self.txt_step)
        row("Thermal-aware:",         self.chk_thermal)
        row("Show ratsnest:",         self.chk_ratsnest)
        row("Show heat-map:",         self.chk_heatmap)

        vs.Add(fgs, 1, wx.EXPAND | wx.ALL, 14)
        vs.Add(self.CreateButtonSizer(wx.OK | wx.CANCEL), 0,
               wx.EXPAND | wx.ALL, 8)
        panel.SetSizer(vs)

    def apply(self):
        CONFIG.backend_url = self.txt_url.GetValue().strip() or CONFIG.backend_url
        try:
            CONFIG.request_timeout = max(5, int(self.txt_timeout.GetValue()))
        except ValueError:
            pass
        try:
            CONFIG.placement_step_mm = max(0.1, float(self.txt_step.GetValue()))
        except ValueError:
            pass
        CONFIG.thermal_aware  = self.chk_thermal.GetValue()
        CONFIG.show_ratsnest  = self.chk_ratsnest.GetValue()
        CONFIG.show_thermal   = self.chk_heatmap.GetValue()
        CONFIG.save()


# â”€â”€ Main Frame â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class AIPCBFrame(wx.Frame):
    """Main application window â€” tabbed, multi-panel, async."""

    # Menu / toolbar IDs
    ID_EXIT      = wx.ID_EXIT
    ID_SETTINGS  = 1001
    ID_REFRESH   = 1002
    ID_EXPORT    = 1003
    ID_OPTIMIZE  = 101
    ID_DFM       = 102
    ID_GENERATE  = 103
    ID_CANCEL    = 104
    ID_RATSNEST  = 201
    ID_THERMAL   = 202

    def __init__(self, parent, board: Any):
        super().__init__(parent, title=f"AI PCB Assistant  v{PLUGIN_VERSION}",
                         size=(1300, 860))
        self.board       = board
        self.components: List[ComponentInfo] = []
        self.nets:       List[NetInfo]       = []
        self.constraints:List[Constraint]    = []
        self.dfm_violations: List[DFMViolation] = []
        self.history:    List[HistoryEntry]  = []

        self._pending:      Dict[str, CancelToken] = {}   # req_id â†’ token
        self._req_types:    Dict[str, str]          = {}   # req_id â†’ type
        self._active_token: Optional[CancelToken]   = None
        self._last_board_diag: str = ""

        self._build_ui()
        self._build_menu()
        self._build_toolbar()
        self._build_statusbar()
        self._extract_board_data()
        self._start_timer()
        self.Bind(wx.EVT_CLOSE, self._on_close)

    # â”€â”€ UI Construction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _build_ui(self):
        root = wx.Panel(self)
        layout = wx.BoxSizer(wx.VERTICAL)
        left = self._build_left(root)
        layout.Add(left, 1, wx.EXPAND)
        root.SetSizer(layout)

        # Minimal demo UI: hide the right preview board area.
        self.canvas = _NullCanvas()

    def _build_left(self, parent) -> wx.Panel:
        panel = wx.Panel(parent)
        vs    = wx.BoxSizer(wx.VERTICAL)

        self.notebook = wx.Notebook(panel)
        self.notebook.AddPage(self._build_assistant_tab(self.notebook),   "Assistant")
        self.notebook.AddPage(self._build_components_tab(self.notebook),  "Components")
        self.notebook.AddPage(self._build_nets_tab(self.notebook),        "Nets")
        self.notebook.AddPage(self._build_dfm_tab(self.notebook),         "DFM")
        self.notebook.AddPage(self._build_history_tab(self.notebook),     "History")
        self.notebook.AddPage(self._build_constraints_tab(self.notebook), "Constraints")

        vs.Add(self.notebook, 1, wx.EXPAND | wx.ALL, 4)
        panel.SetSizer(vs)
        return panel

    def _build_assistant_tab(self, parent) -> wx.Panel:
        panel = wx.Panel(parent)
        vs    = wx.BoxSizer(wx.VERTICAL)

        intro = wx.StaticText(panel, label="AI Assistant")
        intro.SetFont(wx.Font(11, wx.FONTFAMILY_DEFAULT,
                              wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        vs.Add(intro, 0, wx.LEFT | wx.TOP, 8)

        hint = wx.StaticText(
            panel,
            label="Generate uses prompt text. Optimize/DFM run on current board data.")
        vs.Add(hint, 0, wx.LEFT | wx.TOP, 8)

        ctrl_row = wx.FlexGridSizer(rows=2, cols=2, vgap=6, hgap=8)
        ctrl_row.AddGrowableCol(1, 1)
        ctrl_row.Add(wx.StaticText(panel, label="Task:"),
                     0, wx.ALIGN_CENTER_VERTICAL)
        self.task_choice = wx.Choice(panel, choices=ASSISTANT_TASKS)
        self.task_choice.SetSelection(0)
        ctrl_row.Add(self.task_choice, 1, wx.EXPAND)

        self.priority_label = wx.StaticText(panel, label="Priority:")
        ctrl_row.Add(self.priority_label, 0, wx.ALIGN_CENTER_VERTICAL)
        self.priority_choice = wx.Choice(panel, choices=ASSISTANT_PRIORITIES)
        self.priority_choice.SetSelection(0)
        ctrl_row.Add(self.priority_choice, 1, wx.EXPAND)
        vs.Add(ctrl_row, 0, wx.EXPAND | wx.ALL, 8)

        self.prompt_label = wx.StaticText(panel, label="Describe what you want to do:")
        vs.Add(self.prompt_label, 0, wx.LEFT | wx.TOP, 4)
        self.prompt_ctrl = wx.TextCtrl(
            panel, style=wx.TE_MULTILINE, size=(-1, 110))
        self.prompt_ctrl.SetHint(
            "Examples:\n"
            "- Optimize for minimal EMI\n"
            "- Place decoupling caps near each IC\n"
            "- Generate a 555 timer LED blinker\n"
            "- Check DFM for JLCPCB rules")
        self.prompt_ctrl.Bind(wx.EVT_KEY_DOWN, self._on_prompt_key_down)
        vs.Add(self.prompt_ctrl, 0, wx.EXPAND | wx.ALL, 6)

        self.quick_box = wx.StaticBoxSizer(wx.VERTICAL, panel, "Quick prompts")
        quick_row = wx.WrapSizer(wx.HORIZONTAL)
        self._quick_prompts = [
            "Generate a 3.3V regulator with input/output caps",
            "Generate a 555 timer astable LED blinker",
            "Optimize placement to reduce critical net length",
            "Run DFM check for manufacturability issues",
        ]
        self._quick_prompt_buttons: List[wx.Button] = []
        for text in self._quick_prompts:
            btn = wx.Button(panel, label=text)
            btn.Bind(wx.EVT_BUTTON, lambda _e, t=text: self._on_quick_prompt(t))
            quick_row.Add(btn, 0, wx.ALL, 3)
            self._quick_prompt_buttons.append(btn)
        self.quick_box.Add(quick_row, 1, wx.EXPAND | wx.ALL, 4)
        vs.Add(self.quick_box, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 6)

        hs = wx.BoxSizer(wx.HORIZONTAL)
        self.btn_clear_result = wx.Button(panel, label="Clear")
        self.btn_execute = wx.Button(panel, label="Run")
        self.btn_cancel  = wx.Button(panel, label="Cancel")
        self.btn_cancel.Disable()
        hs.Add(self.btn_clear_result, 0, wx.ALL, 4)
        hs.Add(self.btn_execute, 0, wx.ALL, 4)
        hs.Add(self.btn_cancel,  0, wx.ALL, 4)
        vs.Add(hs, 0, wx.ALIGN_RIGHT)

        self.gauge = wx.Gauge(panel, range=100, size=(-1, 6))
        vs.Add(self.gauge, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 6)

        self.response_label = wx.StaticText(panel, label="Response")
        self.response_label.SetFont(wx.Font(10, wx.FONTFAMILY_DEFAULT,
                                            wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        vs.Add(self.response_label, 0, wx.LEFT, 8)

        self.result_ctrl = wx.TextCtrl(
            panel, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_RICH2)
        vs.Add(self.result_ctrl, 1, wx.EXPAND | wx.ALL, 6)

        panel.SetSizer(vs)
        self.btn_execute.Bind(wx.EVT_BUTTON, self._on_execute_prompt)
        self.btn_cancel.Bind(wx.EVT_BUTTON,  self._on_cancel_request)
        self.btn_clear_result.Bind(wx.EVT_BUTTON, self._on_clear_result)
        self.task_choice.Bind(wx.EVT_CHOICE, self._on_task_changed)
        self._on_task_changed(None)
        return panel

    def _build_components_tab(self, parent) -> wx.Panel:
        panel = wx.Panel(parent)
        vs    = wx.BoxSizer(wx.VERTICAL)

        # Search bar
        hs = wx.BoxSizer(wx.HORIZONTAL)
        hs.Add(wx.StaticText(panel, label="Filter:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 4)
        self.comp_search = wx.SearchCtrl(panel, size=(200, -1))
        self.comp_search.ShowCancelButton(True)
        hs.Add(self.comp_search, 1)
        vs.Add(hs, 0, wx.EXPAND | wx.ALL, 6)

        self.comp_list = wx.ListCtrl(panel, style=wx.LC_REPORT | wx.LC_SINGLE_SEL)
        for col, (label, w) in enumerate([
            ("Ref", 60), ("Value", 90), ("Footprint", 130),
            ("X", 55), ("Y", 55), ("Layer", 50), ("Fixed", 46)
        ]):
            self.comp_list.AppendColumn(label, width=w)
        vs.Add(self.comp_list, 1, wx.EXPAND | wx.ALL, 4)

        hs2 = wx.BoxSizer(wx.HORIZONTAL)
        btn_fix = wx.Button(panel, label="Toggle Fixed")
        btn_fix.Bind(wx.EVT_BUTTON, self._on_toggle_fixed)
        btn_sel = wx.Button(panel, label="Select on Board")
        btn_sel.Bind(wx.EVT_BUTTON, self._on_select_on_board)
        hs2.Add(btn_fix, 0, wx.ALL, 4)
        hs2.Add(btn_sel, 0, wx.ALL, 4)
        vs.Add(hs2, 0)

        panel.SetSizer(vs)
        self.comp_list.Bind(wx.EVT_LIST_ITEM_SELECTED, self._on_component_selected)
        self.comp_list.Bind(wx.EVT_LIST_COL_CLICK,     self._on_comp_col_click)
        self.comp_search.Bind(wx.EVT_SEARCHCTRL_SEARCH_BTN, self._on_comp_filter)
        self.comp_search.Bind(wx.EVT_SEARCHCTRL_CANCEL_BTN, self._on_comp_filter_clear)
        self.comp_search.Bind(wx.EVT_TEXT, self._on_comp_filter)
        return panel

    def _build_nets_tab(self, parent) -> wx.Panel:
        panel = wx.Panel(parent)
        vs    = wx.BoxSizer(wx.VERTICAL)

        self.net_list = wx.ListCtrl(panel, style=wx.LC_REPORT | wx.LC_SINGLE_SEL)
        for label, w in [("Net", 130), ("Type", 80), ("Pins", 45),
                          ("Comps", 50), ("Length mm", 80), ("Critical", 55)]:
            self.net_list.AppendColumn(label, width=w)
        vs.Add(self.net_list, 1, wx.EXPAND | wx.ALL, 4)

        btn_mark = wx.Button(panel, label="Mark as Critical")
        btn_mark.Bind(wx.EVT_BUTTON, self._on_mark_net_critical)
        vs.Add(btn_mark, 0, wx.ALL, 4)

        panel.SetSizer(vs)
        self.net_list.Bind(wx.EVT_LIST_ITEM_SELECTED, self._on_net_selected)
        return panel

    def _build_dfm_tab(self, parent) -> wx.Panel:
        panel = wx.Panel(parent)
        vs    = wx.BoxSizer(wx.VERTICAL)

        self.dfm_summary = wx.StaticText(panel, label="No DFM check run yet.")
        self.dfm_summary.SetFont(wx.Font(10, wx.FONTFAMILY_DEFAULT,
                                         wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        vs.Add(self.dfm_summary, 0, wx.ALL, 8)

        self.dfm_list = wx.ListCtrl(panel, style=wx.LC_REPORT | wx.LC_SINGLE_SEL)
        for label, w in [("", 22), ("Severity", 68), ("Type", 100), ("Message", 260)]:
            self.dfm_list.AppendColumn(label, width=w)
        vs.Add(self.dfm_list, 1, wx.EXPAND | wx.ALL, 4)

        hs = wx.BoxSizer(wx.HORIZONTAL)
        btn_export = wx.Button(panel, label="Export CSV")
        btn_export.Bind(wx.EVT_BUTTON, self._on_export_dfm)
        btn_highlight = wx.Button(panel, label="Highlight on Board")
        btn_highlight.Bind(wx.EVT_BUTTON, self._on_highlight_dfm)
        hs.Add(btn_export,    0, wx.ALL, 4)
        hs.Add(btn_highlight, 0, wx.ALL, 4)
        vs.Add(hs, 0)

        panel.SetSizer(vs)
        self.dfm_list.Bind(wx.EVT_LIST_ITEM_SELECTED, self._on_dfm_item_selected)
        return panel

    def _build_history_tab(self, parent) -> wx.Panel:
        panel = wx.Panel(parent)
        vs    = wx.BoxSizer(wx.VERTICAL)

        self.history_list = wx.ListBox(panel)
        vs.Add(self.history_list, 0, wx.EXPAND | wx.ALL, 4)

        self.history_detail = wx.TextCtrl(
            panel, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_RICH2)
        vs.Add(self.history_detail, 1, wx.EXPAND | wx.ALL, 4)

        btn_clear = wx.Button(panel, label="Clear History")
        btn_clear.Bind(wx.EVT_BUTTON, lambda _: self._clear_history())
        vs.Add(btn_clear, 0, wx.ALL, 4)

        panel.SetSizer(vs)
        self.history_list.Bind(wx.EVT_LISTBOX, self._on_history_selected)
        return panel

    def _build_constraints_tab(self, parent) -> wx.Panel:
        panel = wx.Panel(parent)
        vs    = wx.BoxSizer(wx.VERTICAL)

        self.constraint_list = wx.ListBox(panel)
        vs.Add(self.constraint_list, 1, wx.EXPAND | wx.ALL, 4)

        hs = wx.BoxSizer(wx.HORIZONTAL)
        btn_add = wx.Button(panel, label="Add")
        btn_tog = wx.Button(panel, label="Enable/Disable")
        btn_del = wx.Button(panel, label="Delete")
        btn_add.Bind(wx.EVT_BUTTON, self._on_add_constraint)
        btn_tog.Bind(wx.EVT_BUTTON, self._on_toggle_constraint)
        btn_del.Bind(wx.EVT_BUTTON, self._on_delete_constraint)
        for b in (btn_add, btn_tog, btn_del):
            hs.Add(b, 0, wx.ALL, 4)
        vs.Add(hs, 0)

        panel.SetSizer(vs)
        return panel

    def _build_right(self, parent) -> wx.Panel:
        panel = wx.Panel(parent)
        vs    = wx.BoxSizer(wx.VERTICAL)

        # View toolbar
        tb = wx.ToolBar(panel, style=wx.TB_HORIZONTAL | wx.TB_TEXT)
        tb.AddCheckTool(self.ID_RATSNEST, "Ratsnest",
                        wx.NullBitmap, wx.NullBitmap, shortHelp="Toggle ratsnest")
        tb.AddCheckTool(self.ID_THERMAL,  "Heat-map",
                        wx.NullBitmap, wx.NullBitmap, shortHelp="Toggle thermal heat-map")
        tb.Realize()
        try:
            tb.ToggleTool(self.ID_RATSNEST, CONFIG.show_ratsnest)
            tb.ToggleTool(self.ID_THERMAL,  CONFIG.show_thermal)
        except Exception:
            pass
        self._view_tb = tb
        vs.Add(tb, 0, wx.EXPAND)

        self.canvas = PlacementPreviewCanvas(panel)
        vs.Add(self.canvas, 1, wx.EXPAND)

        panel.SetSizer(vs)
        tb.Bind(wx.EVT_TOOL, self._on_toggle_ratsnest, id=self.ID_RATSNEST)
        tb.Bind(wx.EVT_TOOL, self._on_toggle_thermal,  id=self.ID_THERMAL)
        return panel

    def _build_menu(self):
        mb   = wx.MenuBar()
        file = wx.Menu()
        file.Append(self.ID_EXPORT, "&Export Report\tCtrl+E")
        file.AppendSeparator()
        file.Append(self.ID_EXIT, "E&xit\tCtrl+Q")
        mb.Append(file, "&File")

        tools = wx.Menu()
        tools.Append(self.ID_SETTINGS, "&Settings\tCtrl+,")
        tools.Append(self.ID_REFRESH,  "&Refresh Board\tF5")
        mb.Append(tools, "&Tools")

        self.SetMenuBar(mb)
        self.Bind(wx.EVT_MENU, self._on_exit,     id=self.ID_EXIT)
        self.Bind(wx.EVT_MENU, self._on_settings, id=self.ID_SETTINGS)
        self.Bind(wx.EVT_MENU, self._on_refresh,  id=self.ID_REFRESH)
        self.Bind(wx.EVT_MENU, self._on_export,   id=self.ID_EXPORT)

    def _build_toolbar(self):
        tb = self.CreateToolBar(wx.TB_HORIZONTAL | wx.TB_TEXT)
        bitmaps = {
            self.ID_OPTIMIZE: wx.ART_EXECUTABLE_FILE,
            self.ID_DFM:      wx.ART_TICK_MARK,
            self.ID_GENERATE: wx.ART_NEW,
            self.ID_REFRESH:  wx.ART_REDO,
        }
        labels = {
            self.ID_OPTIMIZE: "Optimize",
            self.ID_DFM:      "DFM Check",
            self.ID_GENERATE: "Generate",
            self.ID_REFRESH:  "Refresh",
        }
        for tid, art in bitmaps.items():
            bmp = wx.ArtProvider.GetBitmap(art, wx.ART_TOOLBAR, (16, 16))
            tb.AddTool(tid, labels[tid], bmp, labels[tid])
        tb.Realize()
        self.Bind(wx.EVT_TOOL, self._on_optimize_tool, id=self.ID_OPTIMIZE)
        self.Bind(wx.EVT_TOOL, self._on_dfm_tool,      id=self.ID_DFM)
        self.Bind(wx.EVT_TOOL, self._on_generate_tool, id=self.ID_GENERATE)
        self.Bind(wx.EVT_TOOL, self._on_refresh,       id=self.ID_REFRESH)

    def _build_statusbar(self):
        sb = self.CreateStatusBar(3)
        sb.SetStatusWidths([-1, 200, 140])
        self.SetStatusText("Ready", 0)
        self.SetStatusText("No board", 1)
        self.SetStatusText("Backend: unknown", 2)
        self._update_backend_status()

    # â”€â”€ Timer / Async â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _start_timer(self):
        self._timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self._on_timer, self._timer)
        self._timer.Start(TIMER_MS)
        self._pulse_dir = 1

    def _on_timer(self, _event):
        completed = []
        try:
            for req_id in list(self._pending):
                status, result = HTTP_CLIENT.get_result(req_id)
                if status != "pending":
                    completed.append((req_id, status, result))

            for req_id, status, result in completed:
                self._pending.pop(req_id, None)
                self._handle_result(req_id, status, result)
        except Exception as exc:
            self._append_result(f"[ERROR] Timer processing failed: {exc}")
            self._set_status("Request processing error")

        if self._pending:
            v = self.gauge.GetValue()
            v += self._pulse_dir * 3
            if v >= 100 or v <= 0:
                self._pulse_dir = -self._pulse_dir
            self.gauge.SetValue(max(0, min(100, v)))
        else:
            self.gauge.SetValue(0)
            self.btn_execute.Enable()
            self.btn_cancel.Disable()

    def _handle_result(self, req_id: str, status: str, result: Any):
        req_type = self._req_types.pop(req_id, "")
        if status == "cancelled":
            self._set_status("Request cancelled.")
            return
        if status == "error":
            self._set_status(f"Error: {result}")
            self._append_result(f"[ERROR]\n{result}")
            self._add_history(self._last_prompt, f"ERROR: {result}", req_type)
            return

        if req_type == "optimize":
            self._apply_placement_result(result)
        elif req_type == "dfm":
            self._apply_dfm_result(result)
        elif req_type == "generate":
            self._apply_generate_result(result)

    def _queue_request(self, url: str, data: Optional[Dict],
                       req_type: str) -> str:
        token  = CancelToken()
        req_id = HTTP_CLIENT.request(url, data, self._on_req_complete, token)
        self._pending[req_id]   = token
        self._req_types[req_id] = req_type
        self._active_token      = token
        self._append_result(f"[Run] {req_type} request queued (id={req_id[-8:]})")
        self.btn_execute.Disable()
        self.btn_cancel.Enable()
        return req_id

    def _on_req_complete(self, _req_id: str):
        pass   # picked up by timer

    # â”€â”€ Board Data Extraction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _extract_board_data(self):
        self._sync_active_board()
        self.components.clear()
        self.nets.clear()

        try:
            fps = _iter_board_footprints(self.board)
        except Exception as exc:
            logger.error("GetFootprints/GetModules: %s", exc)
            fps = []

        for fp in fps:
            try:
                pos = fp.GetPosition()
                try:
                    bbox = fp.GetBoundingBox(False, False)
                except TypeError:
                    try:
                        bbox = fp.GetBoundingBox(False)
                    except TypeError:
                        bbox = fp.GetBoundingBox()

                pins = []
                for pad in fp.GetPads():
                    try:
                        pp = pad.GetPosition()
                        pins.append({"number": pad.GetNumber(),
                                     "net":    pad.GetNetCode(),
                                     "x":      pcbnew.ToMM(pp.x),
                                     "y":      pcbnew.ToMM(pp.y)})
                    except Exception:
                        pass

                self.components.append(ComponentInfo(
                    ref       = fp.GetReference(),
                    value     = fp.GetValue(),
                    footprint = _get_footprint_name(fp),
                    x         = pcbnew.ToMM(pos.x),
                    y         = pcbnew.ToMM(pos.y),
                    rotation  = _get_orientation_degrees(fp),
                    layer     = "top" if fp.GetLayer() == pcbnew.F_Cu else "bottom",
                    width     = pcbnew.ToMM(bbox.GetWidth()),
                    height    = pcbnew.ToMM(bbox.GetHeight()),
                    pins      = pins,
                    is_fixed  = fp.IsLocked(),
                ))
            except Exception as exc:
                logger.warning("Skipping footprint: %s", exc)

        try:
            for conn in _build_connections_from_board(self.board):
                self.nets.append(
                    NetInfo(
                        name=conn["net"],
                        code=0,
                        net_type=classify_net(conn["net"]),
                        pins=conn["pins"],
                    )
                )
        except Exception as exc:
            logger.error("Net extraction: %s", exc)

        self._refresh_comp_list()
        self._refresh_net_list()
        self._update_canvas()
        self.SetStatusText(
            f"{len(self.components)} components Â· {len(self.nets)} nets", 1)

    # â”€â”€ UI Refresh Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _refresh_comp_list(self, filt: str = ""):
        self.comp_list.DeleteAllItems()
        filt = filt.lower()
        for i, c in enumerate(self.components):
            if filt and filt not in c.ref.lower() and filt not in c.value.lower():
                continue
            self.comp_list.InsertItem(i, c.ref)
            self.comp_list.SetItem(i, 1, c.value)
            self.comp_list.SetItem(i, 2, c.footprint)
            self.comp_list.SetItem(i, 3, f"{c.x:.1f}")
            self.comp_list.SetItem(i, 4, f"{c.y:.1f}")
            self.comp_list.SetItem(i, 5, c.layer)
            self.comp_list.SetItem(i, 6, "ðŸ”’" if c.is_fixed else "")

    def _refresh_net_list(self):
        self.net_list.DeleteAllItems()
        for i, n in enumerate(self.nets):
            self.net_list.InsertItem(i, n.name)
            self.net_list.SetItem(i, 1, n.net_type.name.capitalize())
            self.net_list.SetItem(i, 2, str(len(n.pins)))
            self.net_list.SetItem(i, 3, str(n.component_count))
            self.net_list.SetItem(i, 4, f"{n.length_mm:.1f}")
            self.net_list.SetItem(i, 5, "âš‘" if n.is_critical else "")

    def _refresh_dfm_list(self):
        self.dfm_list.DeleteAllItems()
        total = len(self.dfm_violations)
        errs  = sum(1 for v in self.dfm_violations
                    if v.severity.lower() in ("critical","error"))
        warns = sum(1 for v in self.dfm_violations
                    if v.severity.lower() == "warning")
        self.dfm_summary.SetLabel(
            f"{total} issues Â· {errs} errors Â· {warns} warnings"
            if total else "âœ…  No DFM violations")

        for i, v in enumerate(self.dfm_violations[:MAX_DFM_SHOWN]):
            self.dfm_list.InsertItem(i, v.icon)
            self.dfm_list.SetItem(i, 1, v.severity.upper())
            self.dfm_list.SetItem(i, 2, v.vtype)
            self.dfm_list.SetItem(i, 3, v.message)
            self.dfm_list.SetItemTextColour(i, v.wx_colour)

    def _refresh_constraint_list(self):
        self.constraint_list.Clear()
        for c in self.constraints:
            self.constraint_list.Append(c.display())

    def _update_canvas(self):
        bbox = self.board.GetBoardEdgesBoundingBox()
        bw   = pcbnew.ToMM(bbox.GetWidth())
        bh   = pcbnew.ToMM(bbox.GetHeight())
        if bw <= 0 or bh <= 0:
            xs = [c.x for c in self.components] or [100.0]
            ys = [c.y for c in self.components] or [ 80.0]
            bw = max(40.0, max(xs) * 1.2)
            bh = max(40.0, max(ys) * 1.2)
        self.canvas.set_board_dimensions(bw, bh)
        self.canvas.show_ratsnest = CONFIG.show_ratsnest
        self.canvas.show_thermal  = CONFIG.show_thermal
        self.canvas.update_components(self.components, self.nets)

    def _update_backend_status(self):
        ok = _check_backend_url(CONFIG.backend_url, timeout=2)
        self.SetStatusText("Backend: âœ… online" if ok else "Backend: âš  offline", 2)

    # â”€â”€ Board Data â†’ Dict â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _board_dict(self) -> Dict[str, Any]:
        # Always build request payload from LIVE board state (not UI cache).
        self._sync_active_board(force=True)

        try:
            if hasattr(self.board, "BuildConnectivity"):
                self.board.BuildConnectivity()
        except Exception:
            pass

        live_components: List[Dict[str, Any]] = []
        min_x = float("inf")
        min_y = float("inf")
        max_x = float("-inf")
        max_y = float("-inf")

        for fp in _iter_board_footprints(self.board):
            try:
                pos = fp.GetPosition()
                x_mm = pcbnew.ToMM(pos.x)
                y_mm = pcbnew.ToMM(pos.y)
                min_x = min(min_x, x_mm)
                min_y = min(min_y, y_mm)
                max_x = max(max_x, x_mm)
                max_y = max(max_y, y_mm)

                live_components.append(
                    {
                        "ref": fp.GetReference(),
                        "value": fp.GetValue(),
                        "footprint": _get_footprint_name(fp),
                        "x": x_mm,
                        "y": y_mm,
                        "rotation": _get_orientation_degrees(fp),
                        "layer": "top" if fp.GetLayer() == pcbnew.F_Cu else "bottom",
                        "fixed": bool(fp.IsLocked()),
                    }
                )
            except Exception:
                continue

        if not live_components:
            return {
                "components": [],
                "connections": [],
                "constraints": [
                    {"type": c.ctype, "refs": c.refs, "params": c.params}
                    for c in self.constraints if c.enabled
                ],
                "board_width": 100.0,
                "board_height": 80.0,
            }

        # Normalize to local origin so backend DFM edge checks don't use huge global coords.
        margin_mm = 10.0
        origin_x = min_x - margin_mm
        origin_y = min_y - margin_mm
        for comp in live_components:
            comp["x"] = float(comp["x"] - origin_x)
            comp["y"] = float(comp["y"] - origin_y)

        # Prefer true board edge dimensions; fallback to footprint extents when no Edge.Cuts.
        try:
            bbox = self.board.GetBoardEdgesBoundingBox()
            bw = float(pcbnew.ToMM(bbox.GetWidth()))
            bh = float(pcbnew.ToMM(bbox.GetHeight()))
        except Exception:
            bw = 0.0
            bh = 0.0

        if bw <= 1.0 or bh <= 1.0:
            span_w = max_x - min_x
            span_h = max_y - min_y
            bw = max(40.0, span_w + 2 * margin_mm)
            bh = max(40.0, span_h + 2 * margin_mm)

        live_connections: List[Dict[str, Any]] = _build_connections_from_board(self.board)

        pad_total = 0
        pad_with_net = 0
        for fp in _iter_board_footprints(self.board):
            try:
                pads = list(fp.GetPads()) if hasattr(fp, "GetPads") else []
            except Exception:
                pads = []
            for pad in pads:
                pad_total += 1
                try:
                    code = int(pad.GetNetCode())
                except Exception:
                    code = 0
                name = ""
                try:
                    name = str(pad.GetNetname()).strip()
                except Exception:
                    pass
                if code != 0 or (name and not name.lower().startswith("unconnected")):
                    pad_with_net += 1

        if not live_connections and self.nets:
            live_connections = [
                {
                    "net": n.name,
                    "net_type": n.net_type.name.lower(),
                    "pins": n.pins,
                    "is_critical": n.is_critical,
                }
                for n in self.nets
            ]

        self._last_board_diag = (
            f"live_components={len(live_components)}, pads={pad_total}, "
            f"pads_with_net={pad_with_net}, connections={len(live_connections)}"
        )

        return {
            "components": live_components,
            "connections": live_connections,
            "constraints": [
                {"type": c.ctype, "refs": c.refs, "params": c.params}
                for c in self.constraints if c.enabled
            ],
            "board_width":  bw,
            "board_height": bh,
        }

    # â”€â”€ Action Handlers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _on_execute_prompt(self, _event):
        try:
            task = ASSISTANT_TASKS[self.task_choice.GetSelection()]
            prompt = self.prompt_ctrl.GetValue().strip()
            self._set_status(f"Running: {task}")

            if task == "Generate Circuit":
                if not prompt:
                    wx.MessageBox("Enter a prompt for circuit generation.", "Assistant", wx.OK | wx.ICON_INFORMATION)
                    return
                self._last_prompt = prompt
                priority = ASSISTANT_PRIORITIES[self.priority_choice.GetSelection()]
                self._set_status("Generating circuit…")
                self._queue_request(
                    f"{CONFIG.backend_url}/generate",
                    {
                        "prompt": prompt,
                        "priority": priority,
                        "constraints": {
                            "prefer_model": True,
                            "strict_llm": True,
                            "use_skidl": True,
                        },
                    },
                    "generate")
                return

            board_payload = self._board_dict()
            if not board_payload.get("components"):
                wx.MessageBox(
                    "No PCB components found on the active board.\n"
                    "Click Refresh once, then run again.",
                    "Assistant",
                    wx.OK | wx.ICON_INFORMATION,
                )
                return

            if task == "Optimize Placement":
                self._last_prompt = "optimize placement"
                self._set_status("Optimizing placement…")
                board_payload["thermal_aware"] = CONFIG.thermal_aware
                net_count = len(board_payload.get("connections", []))
                if net_count == 0:
                    self._append_result(
                        "[Info] Optimize input has 0 connected nets. "
                        "Wirelength will be 0 mm until at least one net connects 2+ pads.\n"
                        f"[Diag] {self._last_board_diag}"
                    )
                self._queue_request(
                    f"{CONFIG.backend_url}/placement/optimize?algorithm=auto",
                    board_payload,
                    "optimize")
                return

            self._last_prompt = "dfm check"
            self._set_status("Running DFM check…")
            self._queue_request(
                f"{CONFIG.backend_url}/dfm/check",
                board_payload,
                "dfm")
        except Exception as exc:
            self._append_result(f"[ERROR] Run failed: {exc}")
            self._set_status("Run failed")

    def _on_task_changed(self, _event):
        """Adjust prompt UI based on selected assistant task."""
        task = ASSISTANT_TASKS[self.task_choice.GetSelection()]
        needs_prompt = task == "Generate Circuit"

        self.prompt_ctrl.Enable(needs_prompt)
        self.priority_choice.Enable(needs_prompt)
        self.priority_label.Enable(needs_prompt)
        if needs_prompt:
            self.prompt_label.SetLabel("Describe what you want to do:")
            self.prompt_ctrl.SetHint(
                "Examples:\n"
                "- Generate a 3.3V regulator with input/output caps\n"
                "- Generate a 555 timer LED blinker"
            )
            self.quick_box.GetStaticBox().SetLabel("Quick prompts")
            for btn in self._quick_prompt_buttons:
                btn.Enable(True)
        else:
            self.prompt_label.SetLabel("Prompt (not needed for this task):")
            self.prompt_ctrl.SetHint(
                "This task uses current PCB data directly."
            )
            self.prompt_ctrl.SetValue("")
            self.quick_box.GetStaticBox().SetLabel("Quick actions")
            for btn in self._quick_prompt_buttons:
                btn.Enable(False)

    def _on_prompt_key_down(self, event):
        key = event.GetKeyCode()
        if key in (wx.WXK_RETURN, wx.WXK_NUMPAD_ENTER) and event.ControlDown():
            self._on_execute_prompt(None)
            return
        event.Skip()

    def _on_quick_prompt(self, text: str):
        self.prompt_ctrl.SetValue(text)
        lower = text.lower()
        if "optimiz" in lower:
            self.task_choice.SetSelection(1)
        elif "dfm" in lower or "manufactur" in lower:
            self.task_choice.SetSelection(2)
        else:
            self.task_choice.SetSelection(0)
        self.prompt_ctrl.SetFocus()

    def _on_clear_result(self, _event):
        self.result_ctrl.Clear()
        self._set_status("Result cleared")

    def _on_cancel_request(self, _event):
        if self._active_token:
            self._active_token.cancel()
        self._set_status("Cancellingâ€¦")

    def _on_optimize_tool(self, _event):
        self._last_prompt = "optimize placement"
        if hasattr(self, "task_choice"):
            self.task_choice.SetSelection(1)
            self._on_task_changed(None)
        self._set_status("Optimizing placementâ€¦")
        data = self._board_dict()
        if not data.get("components"):
            wx.MessageBox("No PCB components found on the active board.", "Assistant", wx.OK | wx.ICON_INFORMATION)
            return
        data["thermal_aware"] = CONFIG.thermal_aware
        self._queue_request(
            f"{CONFIG.backend_url}/placement/optimize?algorithm=auto",
            data, "optimize")

    def _on_dfm_tool(self, _event):
        self._last_prompt = "dfm check"
        if hasattr(self, "task_choice"):
            self.task_choice.SetSelection(2)
            self._on_task_changed(None)
        self._set_status("Running DFM checkâ€¦")
        data = self._board_dict()
        if not data.get("components"):
            wx.MessageBox("No PCB components found on the active board.", "Assistant", wx.OK | wx.ICON_INFORMATION)
            return
        self._queue_request(
            f"{CONFIG.backend_url}/dfm/check",
            data, "dfm")

    def _on_generate_tool(self, _event):
        prompt = wx.GetTextFromUser(
            "Describe the circuit:", "AI Circuit Generation",
            "555 timer astable LED blinker", parent=self)
        if not prompt:
            return
        if hasattr(self, "task_choice"):
            self.task_choice.SetSelection(0)
        self.prompt_ctrl.SetValue(prompt)
        self.notebook.SetSelection(0)
        self._on_execute_prompt(None)

    # â”€â”€ Result Processors â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _apply_placement_result(self, result: Any):
        if not isinstance(result, dict):
            self._append_result("Optimization failed: invalid response format")
            return
        if result.get("success") is False:
            self._append_result(f"Optimization failed: {result.get('error')}")
            return

        positions = result.get("positions", {}) if isinstance(result, dict) else {}
        if not positions:
            self._append_result("Optimization returned no positions.")
            return

        try:
            pcbnew.SaveBoard(self.board.GetFileName(), self.board)
        except Exception:
            pass

        for fp in self.board.GetFootprints():
            ref = fp.GetReference()
            if ref in positions:
                pos = positions[ref]
                fp.SetPosition(pcbnew.VECTOR2I(
                    pcbnew.FromMM(pos["x"]), pcbnew.FromMM(pos["y"])))
                if "rotation" in pos:
                    try:
                        fp.SetOrientationDegrees(pos["rotation"])
                    except Exception:
                        pass
        pcbnew.Refresh()
        self._extract_board_data()

        m = result.get("metrics", {})
        wire_mm = float(m.get("wirelength_mm", m.get("wirelength", 0.0)) or 0.0)
        thermal = float(m.get("thermal_score", 0.0) or 0.0)
        density = float(m.get("density_score", 0.0) or 0.0)
        net_count = int(m.get("net_count", 0) or 0)
        comp_count = int(m.get("component_count", len(positions)) or len(positions))

        txt = (f"âœ…  Optimization complete\n\n"
               f"Components:    {comp_count}\n"
               f"Nets:          {net_count}\n"
               f"Wirelength:    {wire_mm:.1f} mm\n"
               f"Thermal score: {thermal:.1f}\n"
               f"Density score: {density:.1f}\n"
               f"Time:          {m.get('time_ms', 0):.0f} ms")

        if net_count == 0:
            txt += "\n\nNote: 0 nets detected, so wirelength optimization is not applicable yet."
        elif comp_count < 2:
            txt += "\n\nNote: only one component detected; placement optimization effect is limited."
        self._append_result(txt)
        self._set_status("Optimization complete")
        self._add_history("optimize placement", txt, "optimize")

    def _apply_dfm_result(self, result: Any):
        raw = result if isinstance(result, list) else result.get("violations", [])
        self.dfm_violations = [
            DFMViolation(
                severity   = v.get("severity", "warning"),
                vtype      = v.get("type", "violation"),
                message    = v.get("message", ""),
                components = v.get("components", []),
            )
            for v in raw if isinstance(v, dict)
        ]
        self._refresh_dfm_list()
        self.notebook.SetSelection(3)   # jump to DFM tab

        # Highlight errors on canvas
        err_refs: Set[str] = set()
        for v in self.dfm_violations:
            if v.severity.lower() in ("critical", "error"):
                err_refs.update(v.components)
        if err_refs:
            self.canvas.highlight_refs(err_refs)

        txt = f"{len(self.dfm_violations)} DFM issues found."
        self._set_status(txt)
        self._add_history("dfm check", txt, "dfm")

    def _apply_generate_result(self, result: Any):
        if not isinstance(result, dict):
            self._append_result("Generate: invalid response")
            return
        if not result.get("success"):
            msg = f"Generate failed: {result.get('error','unknown')}"
            self._append_result(msg)
            self._add_history(getattr(self, "_last_prompt", ""), msg, "generate")
            return

        cdata       = result.get("circuit_data") or {}
        n_comps     = len(cdata.get("components", []))
        n_nets      = len(cdata.get("connections", []))

        summary = (f"âœ…  Generation complete\n"
                   f"Method:     {result.get('generation_method', 'unknown')}\n"
                   f"Template:   {result.get('template_used','LLM')}\n"
                   f"Components: {n_comps}\n"
                   f"Nets:       {n_nets}\n"
                   f"Time:       {result.get('generation_time_ms',0):.1f} ms")

        warnings = result.get("warnings") or []
        if isinstance(warnings, list) and warnings:
            summary += "\nWarnings:"
            for warning in warnings[:8]:
                summary += f"\n  - {warning}"

        download_url = str(result.get("download_url", "")).strip()
        if download_url:
            summary += f"\nSchematic:  {download_url}"

        if n_comps:
            imp = self._import_circuit(cdata)
            summary += (f"\n\nImported to PCB:\n"
                        f"  Created: {imp['created']}\n"
                        f"  Updated: {imp['updated']}\n"
                        f"  Pads connected: {imp['connected_pads']}")
            if imp["warnings"]:
                summary += "\n  Warnings:\n    " + "\n    ".join(imp["warnings"][:10])

            if imp["created"] == 0 and imp["updated"] == 0:
                msg = "Generate succeeded but no components were imported to PCB."
                if imp["warnings"]:
                    msg += "\n\nDetails:\n" + "\n".join(imp["warnings"][:12])
                wx.MessageBox(msg, "Import Failed", wx.OK | wx.ICON_ERROR)
            elif imp["warnings"]:
                wx.MessageBox(
                    "Imported with warnings:\n\n" + "\n".join(imp["warnings"][:12]),
                    "Import Warnings",
                    wx.OK | wx.ICON_WARNING,
                )

        self._append_result(summary)
        self._set_status("Generation complete")
        self._add_history(getattr(self, "_last_prompt", ""), summary, "generate")

        if download_url:
            ask = wx.MessageBox(
                "Open generated schematic file now?",
                "Open Schematic",
                wx.YES_NO | wx.ICON_QUESTION,
            )
            if ask == wx.YES:
                ok, msg = self._open_generated_schematic(download_url)
                if not ok:
                    wx.MessageBox(f"Could not open schematic:\n{msg}",
                                  "Open Schematic", wx.OK | wx.ICON_WARNING)

    def _import_circuit(self, cdata: Dict[str, Any]) -> Dict[str, Any]:
        """Apply generated circuit data to the live KiCad board."""
        fps_map: Dict[str, Any] = {
            fp.GetReference(): fp for fp in self.board.GetFootprints()}
        created = updated = connected_pads = 0
        warnings: List[str] = []

        for comp in cdata.get("components", []):
            if not isinstance(comp, dict):
                continue
            ref = str(comp.get("ref", "")).strip()
            if not ref:
                warnings.append("Skipped component with no ref")
                continue

            fp = fps_map.get(ref)
            if fp is None:
                fp, err = load_footprint_for_component(comp)
                if fp is None:
                    warnings.append(f"{ref}: {err}")
                    continue
                self.board.Add(fp)
                fps_map[ref] = fp
                created += 1
            else:
                updated += 1

            fp.SetReference(ref)
            fp.SetValue(str(comp.get("value", "")))
            fp.SetPosition(pcbnew.VECTOR2I(
                pcbnew.FromMM(float(comp.get("x", 0))),
                pcbnew.FromMM(float(comp.get("y", 0)))))
            rot = float(comp.get("rotation", 0))
            try:
                fp.SetOrientationDegrees(rot)
            except Exception:
                try:
                    fp.SetOrientation(int(rot * 10))
                except Exception:
                    pass
            layer = str(comp.get("layer", "top")).lower()
            flipped = fp.IsFlipped()
            if layer in ("bottom", "b.cu") and not flipped:
                try:
                    fp.Flip(fp.GetPosition(), False)
                except Exception:
                    pass
            elif layer not in ("bottom", "b.cu") and flipped:
                try:
                    fp.Flip(fp.GetPosition(), False)
                except Exception:
                    pass

        for conn in cdata.get("connections", []):
            if not isinstance(conn, dict):
                continue
            net_name = str(conn.get("net", "")).strip()
            if not net_name:
                continue
            net = ensure_board_net(self.board, net_name)
            for pin in conn.get("pins", []):
                if not isinstance(pin, dict):
                    continue
                ref    = str(pin.get("ref", "")).strip()
                pin_no = str(pin.get("pin", "")).strip()
                fp     = fps_map.get(ref)
                if not fp or not pin_no:
                    continue
                try:
                    pad = _resolve_pad_for_pin(fp, pin_no)
                    if pad:
                        pad.SetNet(net)
                        connected_pads += 1
                    else:
                        warnings.append(f"{ref}.{pin_no}: pad not found")
                except Exception as exc:
                    warnings.append(f"{ref}.{pin_no}: {exc}")

        try:
            self.board.BuildConnectivity()
        except Exception:
            pass
        pcbnew.Refresh()
        self._extract_board_data()
        return {"created": created, "updated": updated,
                "connected_pads": connected_pads, "warnings": warnings}

    def _open_generated_schematic(self, download_url: str) -> Tuple[bool, str]:
        """Download generated schematic from backend and open with OS handler."""
        if not download_url:
            return False, "missing download URL"

        try:
            if download_url.startswith("http://") or download_url.startswith("https://"):
                full_url = download_url
            else:
                full_url = urljoin(CONFIG.backend_url.rstrip("/") + "/", download_url.lstrip("/"))

            req = urllib.request.Request(full_url, method="GET")
            with urllib.request.urlopen(req, timeout=max(10, CONFIG.request_timeout)) as resp:
                data = resp.read()

            filename = os.path.basename(full_url.split("?", 1)[0]) or "generated.kicad_sch"
            out_dir = os.path.join(os.path.expanduser("~"), ".ai_pcb_assistant", "downloads")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, filename)

            with open(out_path, "wb") as f:
                f.write(data)

            if os.name == "nt":
                os.startfile(out_path)  # type: ignore[attr-defined]
            elif hasattr(subprocess, "Popen"):
                subprocess.Popen(["xdg-open", out_path])

            return True, out_path
        except Exception as exc:
            return False, str(exc)

    # â”€â”€ Component / Net / Constraint Events â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _on_component_selected(self, event):
        ref = event.GetText()
        self.canvas.highlight_refs({ref})
        self._set_status(f"Selected: {ref}")
        event.Skip()

    def _on_comp_col_click(self, event):
        col = event.GetColumn()
        keys = [None, "ref","value","footprint","x","y","layer","is_fixed"]
        key  = keys[col] if col < len(keys) else None
        if key:
            self.components.sort(key=lambda c: str(getattr(c, key, "")))
            self._refresh_comp_list(self.comp_search.GetValue())
        event.Skip()

    def _on_comp_filter(self, _event):
        self._refresh_comp_list(self.comp_search.GetValue())

    def _on_comp_filter_clear(self, _event):
        self.comp_search.Clear()
        self._refresh_comp_list()

    def _on_net_selected(self, event):
        idx = event.GetIndex()
        if 0 <= idx < len(self.nets):
            net = self.nets[idx]
            refs = {p["ref"] for p in net.pins}
            self.canvas.highlight_refs(refs)
        event.Skip()

    def _on_mark_net_critical(self, _event):
        idx = self.net_list.GetFirstSelected()
        if idx < 0 or idx >= len(self.nets):
            return
        self.nets[idx].is_critical = not self.nets[idx].is_critical
        self._refresh_net_list()

    def _on_dfm_item_selected(self, event):
        idx = event.GetIndex()
        if 0 <= idx < len(self.dfm_violations):
            v = self.dfm_violations[idx]
            self.canvas.highlight_refs(set(v.components))
        event.Skip()

    def _on_toggle_fixed(self, _event):
        refs = self._selected_comp_refs()
        if not refs:
            wx.MessageBox("Select component(s) first.", "Info")
            return
        n = 0
        for fp in self.board.GetFootprints():
            if fp.GetReference() in refs:
                fp.SetLocked(not fp.IsLocked())
                n += 1
        if n:
            pcbnew.Refresh()
            self._extract_board_data()
            self._set_status(f"Lock toggled for {n} component(s)")

    def _on_select_on_board(self, _event):
        """Zoom board to selected footprint."""
        refs = self._selected_comp_refs()
        if not refs:
            return
        for fp in self.board.GetFootprints():
            if fp.GetReference() in refs:
                try:
                    pcbnew.GetBoard().SetModified()
                    view = pcbnew.GetCurrentSelection()
                    if hasattr(view, "Add"):
                        view.Add(fp)
                except Exception:
                    pass

    def _on_add_constraint(self, _event):
        refs = self._selected_comp_refs()
        if not refs:
            wx.MessageBox("Select component(s) from the Components tab first.", "Info")
            return
        choices = ["fixed", "spacing", "alignment"]
        with wx.SingleChoiceDialog(self, "Constraint type:", "Add Constraint", choices) as dlg:
            if dlg.ShowModal() != wx.ID_OK:
                return
            ctype = choices[dlg.GetSelection()]
        params: Dict[str, Any] = {}
        if ctype == "spacing":
            val = wx.GetTextFromUser("Min spacing (mm):", "Spacing", "1.0", self)
            if not val:
                return
            try:
                params["min_mm"] = max(0.05, float(val))
            except ValueError:
                wx.MessageBox("Invalid value.", "Error", wx.OK | wx.ICON_ERROR)
                return
        self.constraints.append(Constraint(ctype=ctype, refs=refs, params=params))
        self._refresh_constraint_list()
        self._set_status(f"Added {ctype} constraint")

    def _on_toggle_constraint(self, _event):
        idx = self.constraint_list.GetSelection()
        if idx == wx.NOT_FOUND or idx >= len(self.constraints):
            return
        self.constraints[idx].enabled = not self.constraints[idx].enabled
        self._refresh_constraint_list()

    def _on_delete_constraint(self, _event):
        idx = self.constraint_list.GetSelection()
        if idx == wx.NOT_FOUND or idx >= len(self.constraints):
            return
        del self.constraints[idx]
        self._refresh_constraint_list()

    def _on_history_selected(self, event):
        idx = event.GetSelection()
        if 0 <= idx < len(self.history):
            self.history_detail.SetValue(self.history[idx].result)

    def _on_toggle_ratsnest(self, event):
        CONFIG.show_ratsnest = bool(event.IsChecked())
        CONFIG.save()
        self.canvas.show_ratsnest = CONFIG.show_ratsnest
        self.canvas._redraw()

    def _on_toggle_thermal(self, event):
        CONFIG.show_thermal = bool(event.IsChecked())
        CONFIG.save()
        self.canvas.show_thermal = CONFIG.show_thermal
        self.canvas._redraw()

    def _on_settings(self, _event):
        dlg = SettingsDialog(self)
        if dlg.ShowModal() == wx.ID_OK:
            dlg.apply()
            self.canvas.show_ratsnest = CONFIG.show_ratsnest
            self.canvas.show_thermal  = CONFIG.show_thermal
            self.canvas._redraw()
            self._update_backend_status()
            self._set_status("Settings saved")
        dlg.Destroy()

    def _on_refresh(self, _event):
        self._sync_active_board(force=True)
        self._extract_board_data()
        self._update_backend_status()

    def _on_export(self, _event):
        self._on_export_dfm(None)

    def _on_export_dfm(self, _event):
        if not self.dfm_violations:
            wx.MessageBox("No DFM results to export. Run DFM check first.", "Info")
            return
        with wx.FileDialog(self, "Export DFM Report", wildcard="CSV (*.csv)|*.csv",
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as dlg:
            if dlg.ShowModal() != wx.ID_OK:
                return
            path = dlg.GetPath()
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["Severity", "Type", "Message", "Components"])
                for v in self.dfm_violations:
                    w.writerow([v.severity, v.vtype, v.message,
                                 ", ".join(v.components)])
            self._set_status(f"Exported {len(self.dfm_violations)} violations â†’ {path}")
        except Exception as exc:
            wx.MessageBox(f"Export failed: {exc}", "Error", wx.OK | wx.ICON_ERROR)

    def _on_highlight_dfm(self, _event):
        idx = self.dfm_list.GetFirstSelected()
        if idx < 0 or idx >= len(self.dfm_violations):
            self._set_status("Select a DFM issue first.")
            return
        v = self.dfm_violations[idx]
        self.canvas.highlight_refs(set(v.components))

    def _on_exit(self, _event):
        self.Close()

    def _on_close(self, event):
        if hasattr(self, "_timer") and self._timer.IsRunning():
            self._timer.Stop()
        event.Skip()

    # â”€â”€ History â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _add_history(self, prompt: str, result: str, req_type: str):
        entry = HistoryEntry(prompt=prompt, result=result, req_type=req_type)
        self.history.insert(0, entry)
        if len(self.history) > MAX_HISTORY:
            self.history = self.history[:MAX_HISTORY]
        self.history_list.Insert(entry.header(), 0)
        while self.history_list.GetCount() > MAX_HISTORY:
            self.history_list.Delete(self.history_list.GetCount() - 1)

    def _clear_history(self):
        self.history.clear()
        self.history_list.Clear()
        self.history_detail.Clear()

    # â”€â”€ Misc Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _selected_comp_refs(self) -> List[str]:
        refs, idx = [], self.comp_list.GetFirstSelected()
        while idx != -1:
            refs.append(self.comp_list.GetItemText(idx))
            idx = self.comp_list.GetNextSelected(idx)
        return refs

    def _set_status(self, msg: str):
        self.SetStatusText(msg, 0)

    def _append_result(self, text: str):
        existing = self.result_ctrl.GetValue().strip()
        if existing:
            stamp = time.strftime("%H:%M:%S", time.localtime())
            self.result_ctrl.SetValue(f"{existing}\n\n--- [{stamp}] ---\n{text}")
        else:
            self.result_ctrl.SetValue(text)
        self.notebook.SetSelection(0)

    def _sync_active_board(self, force: bool = False) -> bool:
        """Bind frame to KiCad's currently active board when available."""
        try:
            active = pcbnew.GetBoard()
        except Exception:
            active = None
        if active is None:
            return False

        if force or active is not self.board:
            self.board = active
            return True
        return False


# â”€â”€ Action Plugin Entry Point â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class AIPlacementPlugin(pcbnew.ActionPlugin):
    def defaults(self):
        self.name                = f"AI PCB Assistant v{PLUGIN_VERSION}"
        self.category            = "AI Tools"
        self.description         = "AI-powered placement, DFM, and schematic generation"
        self.show_toolbar_button = True
        icon = os.path.join(os.path.dirname(__file__), "icon_32x32.png")
        self.icon_file_name      = icon if os.path.exists(icon) else ""

    def Run(self):
        board = pcbnew.GetBoard()
        if board is None:
            wx.MessageBox("No board is open.", "Error", wx.OK | wx.ICON_ERROR)
            return

        # Reuse existing window if alive
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

        if not _check_backend_url(CONFIG.backend_url):
            dlg = BackendSetupDialog(None)
            if dlg.ShowModal() != wx.ID_OK:
                dlg.Destroy()
                return
            dlg.Destroy()

        self._frame = AIPCBFrame(None, board)
        self._frame.Show()
        self._frame.Raise()


# Registration is handled by pcbnew_action.py wrapper.