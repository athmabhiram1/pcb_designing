"""
AI PCB Assistant - Advanced KiCad Action Plugin v2.0
Professional-grade PCB design automation with real-time AI assistance.
"""

from __future__ import annotations

import json
import logging
import math
import os
import queue
import re
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urljoin

import pcbnew
import wx
import wx.grid
import wx.lib.scrolledpanel as scrolled
from wx.lib.floatcanvas import FloatCanvas, NavCanvas, Resources

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class PluginConfig:
    """Plugin configuration with persistence."""
    backend_url: str = "http://localhost:8765"
    request_timeout: int = 120
    auto_refresh: bool = True
    show_ratsnest: bool = True
    placement_step_size: float = 1.0  # mm
    thermal_aware: bool = True
    constraint_driven: bool = True
    
    # File paths
    config_dir: str = field(default_factory=lambda: os.path.join(
        os.path.expanduser("~"), ".ai_pcb_assistant"
    ))
    
    def __post_init__(self):
        os.makedirs(self.config_dir, exist_ok=True)
    
    def save(self):
        """Save configuration to disk."""
        path = os.path.join(self.config_dir, "config.json")
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)
    
    @classmethod
    def load(cls) -> PluginConfig:
        """Load configuration from disk."""
        path = os.path.join(os.path.expanduser("~"), ".ai_pcb_assistant", "config.json")
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
                return cls(**data)
        return cls()

CONFIG = PluginConfig.load()


# ── Data Structures ───────────────────────────────────────────────────────────

class NetType(Enum):
    """Classification of nets for optimization."""
    SIGNAL = auto()
    POWER = auto()
    GROUND = auto()
    CLOCK = auto()
    DIFFERENTIAL = auto()
    HIGH_SPEED = auto()
    ANALOG = auto()


@dataclass
class NetInfo:
    """Enhanced net information."""
    name: str
    code: int
    net_type: NetType
    pins: List[Dict[str, str]] = field(default_factory=list)
    length_mm: float = 0.0
    is_critical: bool = False
    matched_length: Optional[str] = None  # For differential pairs
    
    @property
    def component_count(self) -> int:
        return len(set(p["ref"] for p in self.pins))


@dataclass
class ComponentInfo:
    """Enhanced component information."""
    ref: str
    value: str
    footprint: str
    x: float
    y: float
    rotation: float
    layer: str
    width: float = 0.0
    height: float = 0.0
    pins: List[Dict] = field(default_factory=list)
    power_dissipation: float = 0.0
    is_fixed: bool = False
    cluster_id: Optional[str] = None


@dataclass
class Constraint:
    """Placement constraint."""
    type: str  # "fixed", "region", "alignment", "spacing"
    refs: List[str]
    params: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


# ── Async HTTP Client ─────────────────────────────────────────────────────────

class AsyncHTTPClient:
    """Non-blocking HTTP client for backend communication."""
    
    def __init__(self):
        self._queue: queue.Queue = queue.Queue()
        self._results: Dict[str, Any] = {}
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
    
    def _worker(self):
        """Background worker thread."""
        while True:
            task = self._queue.get()
            if task is None:
                break
            
            request_id, url, data, callback = task
            try:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(data).encode('utf-8') if data else None,
                    headers={"Content-Type": "application/json"},
                    method="POST" if data else "GET",
                )
                
                with urllib.request.urlopen(req, timeout=CONFIG.request_timeout) as resp:
                    result = json.loads(resp.read().decode())
                    self._results[request_id] = ("success", result)
                    
            except Exception as e:
                self._results[request_id] = ("error", str(e))
            
            if callback:
                wx.CallAfter(callback, request_id)
    
    def request(self, url: str, data: Optional[Dict] = None, 
                callback: Optional[Callable] = None) -> str:
        """Queue a request and return request ID."""
        request_id = f"{time.time():.6f}"
        self._queue.put((request_id, url, data, callback))
        return request_id
    
    def get_result(self, request_id: str) -> Tuple[str, Any]:
        """Get result for a request."""
        return self._results.pop(request_id, ("pending", None))
    
    def close(self):
        """Shutdown the worker thread."""
        self._queue.put(None)

# Global client
HTTP_CLIENT = AsyncHTTPClient()


# ── Visualization Canvas ──────────────────────────────────────────────────────

class PlacementPreviewCanvas(NavCanvas.NavCanvas):
    """Interactive canvas for placement preview."""
    
    def __init__(self, parent):
        super().__init__(parent, size=(400, 400))
        self.board_width = 100.0
        self.board_height = 80.0
        self.scale = 4.0  # pixels per mm
        self.components: Dict[str, Any] = {}
        self.nets: List[NetInfo] = []
        self.selected_refs: Set[str] = set()
        self.show_ratsnest = True
        
        self.Bind(wx.EVT_SIZE, self.on_size)
    
    def set_board_dimensions(self, width: float, height: float):
        """Update board outline."""
        self.board_width = width
        self.board_height = height
        self._draw_board()
    
    def update_components(self, components: List[ComponentInfo], 
                         nets: Optional[List[NetInfo]] = None):
        """Update component display."""
        self.components = {c.ref: c for c in components}
        if nets:
            self.nets = nets
        self._draw()
    
    def highlight_component(self, ref: str, color: str = "red"):
        """Highlight a specific component."""
        if ref in self.components:
            self.selected_refs.add(ref)
            self._draw()
    
    def _draw_board(self):
        """Draw board outline."""
        self.Canvas.ClearAll()
        
        # Board outline
        w = self.board_width * self.scale
        h = self.board_height * self.scale
        rect = self.Canvas.AddRectangle(
            (-w/2, -h/2), (w, h),
            LineColor="black", LineWidth=2, FillColor="darkgreen", FillStyle="CrossHatch"
        )
        
        self.Canvas.ZoomToBB()
    
    def _draw(self):
        """Draw all components and connections."""
        self._draw_board()
        
        # Draw nets (ratsnest)
        if self.show_ratsnest and self.nets:
            self._draw_ratsnest()
        
        # Draw components
        for ref, comp in self.components.items():
            x = (comp.x - self.board_width/2) * self.scale
            y = -(comp.y - self.board_height/2) * self.scale  # Flip Y for canvas
            
            # Color by type
            color = self._get_component_color(comp)
            if ref in self.selected_refs:
                color = "red"
            
            # Draw component body
            w = max(2.0, comp.width * self.scale * 0.8)
            h = max(2.0, comp.height * self.scale * 0.8)
            
            rect = self.Canvas.AddRectangle(
                (x - w/2, y - h/2), (w, h),
                LineColor="black", LineWidth=1, FillColor=color
            )
            
            # Draw reference
            self.Canvas.AddText(ref, (x, y), Size=8, Color="white", Position="cc")
        
        self.Canvas.Draw()
    
    def _draw_ratsnest(self):
        """Draw airwires between connected components."""
        for net in self.nets:
            if len(net.pins) < 2:
                continue
            
            # Get component positions
            positions = []
            for pin in net.pins:
                if pin["ref"] in self.components:
                    comp = self.components[pin["ref"]]
                    x = (comp.x - self.board_width/2) * self.scale
                    y = -(comp.y - self.board_height/2) * self.scale
                    positions.append((x, y))
            
            # Draw lines between all pairs (simplified)
            color = self._get_net_color(net)
            for i in range(len(positions) - 1):
                self.Canvas.AddLine(
                    positions[i], positions[i+1],
                    LineColor=color, LineWidth=1, LineStyle="Dot"
                )
    
    def _get_component_color(self, comp: ComponentInfo) -> str:
        """Get color based on component type."""
        if comp.ref.startswith('U'):
            return "blue"
        elif comp.ref.startswith('R'):
            return "lightblue"
        elif comp.ref.startswith('C'):
            return "yellow"
        elif comp.ref.startswith('L'):
            return "orange"
        elif comp.ref[0] in 'JP':
            return "green"
        return "gray"
    
    def _get_net_color(self, net: NetInfo) -> str:
        """Get color based on net type."""
        if net.net_type == NetType.POWER:
            return "red"
        elif net.net_type == NetType.GROUND:
            return "green"
        elif net.net_type == NetType.CLOCK:
            return "purple"
        elif net.net_type == NetType.DIFFERENTIAL:
            return "orange"
        return "lightgray"
    
    def on_size(self, event):
        self.Canvas.ZoomToBB()
        event.Skip()


# ── Main Plugin Class ─────────────────────────────────────────────────────────

class AIPlacementPlugin(pcbnew.ActionPlugin):
    """Advanced KiCad plugin for AI-powered PCB design."""
    
    def defaults(self):
        self.name = "AI PCB Assistant Pro"
        self.category = "AI Tools"
        self.description = "Advanced AI-powered placement, routing, and DFM"
        self.show_toolbar_button = True
        self.icon_file_name = os.path.join(
            os.path.dirname(__file__), "icon_32x32.png"
        )
    
    def Run(self):
        """Main entry point."""
        board = pcbnew.GetBoard()
        if board is None:
            wx.MessageBox("No board is open.", "Error", wx.OK | wx.ICON_ERROR)
            return
        
        # Check backend with timeout
        if not self._check_backend():
            dlg = BackendSetupDialog(None)
            if dlg.ShowModal() != wx.ID_OK:
                dlg.Destroy()
                return
            dlg.Destroy()
        
        # Show main window
        frame = AIPCBFrame(None, board)
        frame.Show()
        frame.Raise()
    
    def _check_backend(self) -> bool:
        """Check if backend is available."""
        try:
            req = urllib.request.Request(
                f"{CONFIG.backend_url}/health", 
                method="GET"
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode())
                return data.get("status") in ("ok", "healthy", "degraded")
        except Exception as e:
            logger.warning(f"Backend check failed: {e}")
            return False


# ── Backend Setup Dialog ──────────────────────────────────────────────────────

class BackendSetupDialog(wx.Dialog):
    """Dialog for backend installation/configuration."""
    
    def __init__(self, parent):
        super().__init__(parent, title="AI Backend Setup", size=(500, 400))
        self._init_ui()
    
    def _init_ui(self):
        panel = wx.Panel(self)
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Header
        header = wx.StaticText(panel, label="AI Backend Not Detected")
        header.SetFont(wx.Font(14, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        sizer.Add(header, 0, wx.ALL | wx.ALIGN_CENTER, 10)
        
        # Options
        sizer.Add(wx.StaticText(panel, label="Choose an option:"), 0, wx.LEFT | wx.TOP, 10)
        
        self.rb_local = wx.RadioButton(panel, label="Start local backend (recommended)", style=wx.RB_GROUP)
        sizer.Add(self.rb_local, 0, wx.ALL, 5)
        
        self.rb_remote = wx.RadioButton(panel, label="Connect to remote backend")
        sizer.Add(self.rb_remote, 0, wx.ALL, 5)
        
        self.txt_url = wx.TextCtrl(panel, value=CONFIG.backend_url)
        sizer.Add(self.txt_url, 0, wx.EXPAND | wx.ALL, 5)
        
        # Auto-install option
        self.chk_install = wx.CheckBox(panel, label="Download and install backend automatically")
        self.chk_install.SetValue(True)
        sizer.Add(self.chk_install, 0, wx.ALL, 10)
        
        # Buttons
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        btn_ok = wx.Button(panel, wx.ID_OK, label="Setup")
        btn_cancel = wx.Button(panel, wx.ID_CANCEL)
        btn_sizer.Add(btn_ok, 0, wx.ALL, 5)
        btn_sizer.Add(btn_cancel, 0, wx.ALL, 5)
        sizer.Add(btn_sizer, 0, wx.ALIGN_CENTER | wx.ALL, 10)
        
        panel.SetSizer(sizer)
        
        # Bind events
        btn_ok.Bind(wx.EVT_BUTTON, self._on_setup)
    
    def _on_setup(self, event):
        if self.rb_remote.GetValue():
            CONFIG.backend_url = self.txt_url.GetValue()
            CONFIG.save()
        
        if self.chk_install.GetValue() and self.rb_local.GetValue():
            self._install_backend()
        
        self.EndModal(wx.ID_OK)
    
    def _install_backend(self):
        """Trigger backend installation."""
        # Would download and install backend
        wx.MessageBox("Backend installation would start here.", "Info")


# ── Main Application Frame ────────────────────────────────────────────────────

class AIPCBFrame(wx.Frame):
    """Main application window with advanced features."""
    
    def __init__(self, parent, board: pcbnew.BOARD):
        super().__init__(parent, title="AI PCB Assistant Pro", size=(1200, 800))
        self.board = board
        self.components: List[ComponentInfo] = []
        self.nets: List[NetInfo] = []
        self.constraints: List[Constraint] = []
        self.pending_requests: Set[str] = set()
        self.request_types: Dict[str, str] = {}
        
        self._init_ui()
        self._extract_board_data()
        self._start_refresh_timer()
    
    def _init_ui(self):
        """Initialize the main UI."""
        self.CreateStatusBar()
        self.SetStatusText("Ready")
        
        # Menu bar
        menubar = wx.MenuBar()
        file_menu = wx.Menu()
        file_menu.Append(wx.ID_EXIT, "E&xit\tCtrl+Q")
        self.Bind(wx.EVT_MENU, self._on_exit, id=wx.ID_EXIT)
        menubar.Append(file_menu, "&File")
        
        tools_menu = wx.Menu()
        tools_menu.Append(1001, "&Settings\tCtrl+,")
        tools_menu.Append(1002, "&Refresh Board Data\tF5")
        self.Bind(wx.EVT_MENU, self._on_settings, id=1001)
        self.Bind(wx.EVT_MENU, self._on_refresh, id=1002)
        menubar.Append(tools_menu, "&Tools")
        
        self.SetMenuBar(menubar)
        
        # Main splitter
        splitter = wx.SplitterWindow(self)
        
        # Left panel: Controls
        left_panel = self._create_left_panel(splitter)
        
        # Right panel: Visualization
        right_panel = self._create_right_panel(splitter)
        
        splitter.SplitVertically(left_panel, right_panel, 400)
        splitter.SetMinimumPaneSize(300)
        
        # Toolbar
        toolbar = self.CreateToolBar(wx.TB_HORIZONTAL | wx.TB_TEXT)
        toolbar.AddTool(101, "Optimize", wx.ArtProvider.GetBitmap(wx.ART_EXECUTABLE_FILE), "Optimize Placement")
        toolbar.AddTool(102, "DFM Check", wx.ArtProvider.GetBitmap(wx.ART_TICK_MARK), "Check DFM")
        toolbar.AddTool(103, "Generate", wx.ArtProvider.GetBitmap(wx.ART_NEW), "Generate Circuit")
        toolbar.Realize()
        
        self.Bind(wx.EVT_TOOL, self._on_optimize_tool, id=101)
        self.Bind(wx.EVT_TOOL, self._on_dfm_tool, id=102)
        self.Bind(wx.EVT_TOOL, self._on_generate_tool, id=103)
    
    def _create_left_panel(self, parent) -> wx.Panel:
        """Create left control panel."""
        panel = scrolled.ScrolledPanel(parent, size=(400, -1))
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # AI Assistant section
        box = wx.StaticBox(panel, label="AI Assistant")
        box_sizer = wx.StaticBoxSizer(box, wx.VERTICAL)
        
        self.prompt_ctrl = wx.TextCtrl(panel, style=wx.TE_MULTILINE, size=(-1, 100))
        self.prompt_ctrl.SetHint("Describe what you want to do...\n"
                                "Examples:\n"
                                "- 'Optimize for minimal EMI'\n"
                                "- 'Place decoupling caps near ICs'\n"
                                "- 'Group analog components together'")
        box_sizer.Add(self.prompt_ctrl, 0, wx.EXPAND | wx.ALL, 5)
        
        btn_run = wx.Button(panel, label="Execute")
        btn_run.Bind(wx.EVT_BUTTON, self._on_execute_prompt)
        box_sizer.Add(btn_run, 0, wx.ALIGN_RIGHT | wx.ALL, 5)
        
        sizer.Add(box_sizer, 0, wx.EXPAND | wx.ALL, 10)
        
        # Component list
        box2 = wx.StaticBox(panel, label="Components")
        box2_sizer = wx.StaticBoxSizer(box2, wx.VERTICAL)
        
        self.comp_list = wx.ListCtrl(panel, style=wx.LC_REPORT)
        self.comp_list.AppendColumn("Ref", width=60)
        self.comp_list.AppendColumn("Value", width=100)
        self.comp_list.AppendColumn("Footprint", width=120)
        self.comp_list.AppendColumn("X", width=50)
        self.comp_list.AppendColumn("Y", width=50)
        self.comp_list.AppendColumn("Fixed", width=50)
        box2_sizer.Add(self.comp_list, 1, wx.EXPAND | wx.ALL, 5)
        
        btn_fix = wx.Button(panel, label="Toggle Fixed")
        btn_fix.Bind(wx.EVT_BUTTON, self._on_toggle_fixed)
        box2_sizer.Add(btn_fix, 0, wx.ALL, 5)
        
        sizer.Add(box2_sizer, 1, wx.EXPAND | wx.ALL, 10)
        
        # Constraints
        box3 = wx.StaticBox(panel, label="Constraints")
        box3_sizer = wx.StaticBoxSizer(box3, wx.VERTICAL)
        
        self.constraint_list = wx.ListBox(panel)
        box3_sizer.Add(self.constraint_list, 1, wx.EXPAND | wx.ALL, 5)
        
        btn_add_constraint = wx.Button(panel, label="Add Constraint")
        btn_add_constraint.Bind(wx.EVT_BUTTON, self._on_add_constraint)
        box3_sizer.Add(btn_add_constraint, 0, wx.ALL, 5)
        
        sizer.Add(box3_sizer, 0, wx.EXPAND | wx.ALL, 10)
        
        panel.SetSizer(sizer)
        panel.SetupScrolling()
        return panel
    
    def _create_right_panel(self, parent) -> wx.Panel:
        """Create right visualization panel."""
        panel = wx.Panel(parent)
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Toolbar for view options
        tb = wx.ToolBar(panel, style=wx.TB_HORIZONTAL)
        self.chk_ratsnest = tb.AddCheckTool(201, "Show Ratsnest", wx.NullBitmap)
        self.chk_ratsnest.SetToggle(CONFIG.show_ratsnest)
        tb.Realize()
        sizer.Add(tb, 0, wx.EXPAND)
        
        # Canvas
        self.canvas = PlacementPreviewCanvas(panel)
        sizer.Add(self.canvas, 1, wx.EXPAND)
        
        # Info panel
        self.info_text = wx.TextCtrl(panel, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(-1, 100))
        sizer.Add(self.info_text, 0, wx.EXPAND | wx.ALL, 5)
        
        panel.SetSizer(sizer)
        return panel
    
    def _start_refresh_timer(self):
        """Start timer for async result checking."""
        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self._on_timer, self.timer)
        self.timer.Start(100)  # 100ms interval
    
    def _on_timer(self, event):
        """Check for async results."""
        completed = []
        for req_id in list(self.pending_requests):
            status, result = HTTP_CLIENT.get_result(req_id)
            if status != "pending":
                completed.append((req_id, status, result))
        
        for req_id, status, result in completed:
            self.pending_requests.discard(req_id)
            self._handle_async_result(req_id, status, result)
    
    def _handle_async_result(self, req_id: str, status: str, result: Any):
        """Handle completed async request."""
        req_type = self.request_types.pop(req_id, "")

        if status == "error":
            self.SetStatusText(f"Error: {result}")
            return
        
        if req_type == "optimize":
            self._apply_placement_result(result)
        elif req_type == "dfm":
            self._show_dfm_result(result)
        elif req_type == "generate":
            self._show_generate_result(result)
    
    def _extract_board_data(self):
        """Extract comprehensive board data."""
        self.components = []
        self.nets = []
        
        # Extract components with detailed info
        for fp in self.board.GetFootprints():
            pos = fp.GetPosition()
            
            # Get footprint dimensions
            bbox = fp.GetBoundingBox(False)
            width = pcbnew.ToMM(bbox.GetWidth())
            height = pcbnew.ToMM(bbox.GetHeight())
            
            # Extract pins
            pins = []
            for pad in fp.GetPads():
                pins.append({
                    "number": pad.GetNumber(),
                    "net": pad.GetNetCode(),
                    "x": pcbnew.ToMM(pad.GetPosition().x),
                    "y": pcbnew.ToMM(pad.GetPosition().y),
                })
            
            comp = ComponentInfo(
                ref=fp.GetReference(),
                value=fp.GetValue(),
                footprint=fp.GetFPID().GetLibItemName(),
                x=pcbnew.ToMM(pos.x),
                y=pcbnew.ToMM(pos.y),
                rotation=fp.GetOrientationDegrees(),
                layer="top" if fp.GetLayer() == pcbnew.F_Cu else "bottom",
                width=width,
                height=height,
                pins=pins,
                is_fixed=fp.IsLocked(),
            )
            self.components.append(comp)
        
        # Extract nets with classification
        net_info = self.board.GetNetInfo()
        for net_name, net in net_info.NetsByName().items():
            if net.GetNetCode() == 0:
                continue
            
            # Classify net
            net_type = self._classify_net(net_name)
            
            pins = []
            for pad in net.GetPads():
                pins.append({
                    "ref": pad.GetParent().GetReference(),
                    "pin": pad.GetNumber(),
                })
            
            if len(pins) >= 2:
                self.nets.append(NetInfo(
                    name=net_name,
                    code=net.GetNetCode(),
                    net_type=net_type,
                    pins=pins,
                ))
        
        # Update UI
        self._update_component_list()
        self._update_canvas()
        self.SetStatusText(f"Loaded {len(self.components)} components, {len(self.nets)} nets")
    
    def _classify_net(self, name: str) -> NetType:
        """Classify net by name patterns."""
        name_upper = name.upper()
        
        # Power nets
        if any(p in name_upper for p in ["VCC", "VDD", "3V3", "5V", "1V8", "PWR"]):
            return NetType.POWER
        
        # Ground
        if any(g in name_upper for g in ["GND", "VSS", "AGND", "DGND"]):
            return NetType.GROUND
        
        # Clock
        if any(c in name_upper for c in ["CLK", "CLOCK", "OSC"]):
            return NetType.CLOCK
        
        # Differential (avoid false positives from single-char +/-)
        if any(d in name_upper for d in ["DIFF", "_DP", "_DN", "USB_D", "CAN_", "LVDS"]):
            return NetType.DIFFERENTIAL
        
        # High speed
        if any(h in name_upper for h in ["USB", "HDMI", "ETH", "LVDS"]):
            return NetType.HIGH_SPEED
        
        # Analog
        if any(a in name_upper for a in ["ADC", "DAC", "ANA", "SENSOR"]):
            return NetType.ANALOG
        
        return NetType.SIGNAL
    
    def _update_component_list(self):
        """Update the component list control."""
        self.comp_list.DeleteAllItems()
        for i, comp in enumerate(self.components):
            self.comp_list.InsertItem(i, comp.ref)
            self.comp_list.SetItem(i, 1, comp.value)
            self.comp_list.SetItem(i, 2, comp.footprint)
            self.comp_list.SetItem(i, 3, f"{comp.x:.1f}")
            self.comp_list.SetItem(i, 4, f"{comp.y:.1f}")
            self.comp_list.SetItem(i, 5, "Yes" if comp.is_fixed else "No")
    
    def _update_canvas(self):
        """Update the visualization canvas."""
        self.canvas.set_board_dimensions(
            pcbnew.ToMM(self.board.GetBoardEdgesBoundingBox().GetWidth()),
            pcbnew.ToMM(self.board.GetBoardEdgesBoundingBox().GetHeight())
        )
        self.canvas.update_components(self.components, self.nets)
    
    def _get_board_data_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backend API."""
        return {
            "components": [
                {
                    "ref": c.ref,
                    "value": c.value,
                    "footprint": c.footprint,
                    "x": c.x,
                    "y": c.y,
                    "rotation": c.rotation,
                    "layer": c.layer,
                    "width": c.width,
                    "height": c.height,
                    "fixed": c.is_fixed,
                    "power_dissipation": c.power_dissipation,
                }
                for c in self.components
            ],
            "connections": [
                {
                    "net": n.name,
                    "net_type": n.net_type.name.lower(),
                    "pins": n.pins,
                }
                for n in self.nets
            ],
            "constraints": [
                {
                    "type": c.type,
                    "refs": c.refs,
                    "params": c.params,
                }
                for c in self.constraints if c.enabled
            ],
            "board_width": pcbnew.ToMM(self.board.GetBoardEdgesBoundingBox().GetWidth()),
            "board_height": pcbnew.ToMM(self.board.GetBoardEdgesBoundingBox().GetHeight()),
        }
    
    def _on_execute_prompt(self, event):
        """Execute natural language prompt."""
        prompt = self.prompt_ctrl.GetValue().strip()
        if not prompt:
            return
        
        self.SetStatusText("Processing request...")
        
        data = self._get_board_data_dict()
        data["prompt"] = prompt
        
        req_id = HTTP_CLIENT.request(
            f"{CONFIG.backend_url}/generate",
            data,
            callback=self._on_request_complete
        )
        self.pending_requests.add(req_id)
        self.request_types[req_id] = "generate"
    
    def _on_optimize_tool(self, event):
        """Run placement optimization."""
        self.SetStatusText("Optimizing placement...")
        
        data = self._get_board_data_dict()
        data["thermal_aware"] = CONFIG.thermal_aware
        
        req_id = HTTP_CLIENT.request(
            f"{CONFIG.backend_url}/placement/optimize?algorithm=auto",
            data,
            callback=self._on_request_complete
        )
        self.pending_requests.add(req_id)
        self.request_types[req_id] = "optimize"
    
    def _on_dfm_tool(self, event):
        """Run DFM check."""
        self.SetStatusText("Checking DFM...")
        
        data = self._get_board_data_dict()
        
        req_id = HTTP_CLIENT.request(
            f"{CONFIG.backend_url}/dfm/check",
            data,
            callback=self._on_request_complete
        )
        self.pending_requests.add(req_id)
        self.request_types[req_id] = "dfm"
    
    def _on_request_complete(self, request_id: str):
        """Called when async request completes."""
        # Handled in timer
        pass
    
    def _apply_placement_result(self, result: Dict):
        """Apply placement optimization results."""
        if not isinstance(result, dict):
            wx.MessageBox("Optimization failed: invalid response format", "Error")
            return

        if result.get("success") is False:
            wx.MessageBox(f"Optimization failed: {result.get('error')}", "Error")
            return
        
        positions = result.get("positions", {})
        if not positions:
            wx.MessageBox("Optimization completed but returned no positions.", "Info")
            return
        
        # Create undo point (API varies by KiCad version)
        try:
            commit = pcbnew.BOARD_COMMIT()
            commit.Push("AI Placement Optimization")
        except Exception:
            try:
                pcbnew.SaveCopyInUndoList(self.board, pcbnew.UNDO_REDO_CHANGED)
            except Exception:
                pass  # Undo not critical — placement will still be applied
        
        # Apply positions
        for fp in self.board.GetFootprints():
            ref = fp.GetReference()
            if ref in positions:
                pos = positions[ref]
                new_x = pcbnew.FromMM(pos["x"])
                new_y = pcbnew.FromMM(pos["y"])
                fp.SetPosition(pcbnew.VECTOR2I(new_x, new_y))
                
                if "rotation" in pos:
                    fp.SetOrientationDegrees(pos["rotation"])
        
        pcbnew.Refresh()
        
        # Update local data
        self._extract_board_data()
        
        # Show metrics
        metrics = result.get("metrics", {})
        info = (
            f"Optimization complete!\n\n"
            f"Wirelength: {metrics.get('wirelength', 0):.1f} mm\n"
            f"Thermal score: {metrics.get('thermal_score', 0):.1f}\n"
            f"Density: {metrics.get('density_score', 0):.1f}%\n"
            f"Time: {metrics.get('time_ms', 0):.0f} ms"
        )
        self.info_text.SetValue(info)
        self.SetStatusText("Optimization complete")
    
    def _show_dfm_result(self, result: Dict):
        """Display DFM check results."""
        if isinstance(result, list):
            violations = result
        elif isinstance(result, dict):
            violations = result.get("violations", [])
        else:
            violations = []

        if not violations:
            self.info_text.SetValue("✓ No DFM issues found!")
            return
        
        text = f"DFM Issues Found: {len(violations)}\n\n"
        for v in violations[:20]:
            sev = v.get('severity', 'warning').upper()
            text += f"[{sev}] {v.get('type', 'violation')}: {v.get('message', 'No details')}\n"
        
        self.info_text.SetValue(text)
        
        # Highlight violations on canvas
        for v in violations:
            for ref in v.get('components', []):
                self.canvas.highlight_component(ref, "red")

    def _show_generate_result(self, result: Any):
        """Display generate endpoint response in the info panel."""
        if not isinstance(result, dict):
            self.info_text.SetValue("Generate failed: invalid response format")
            return

        if not result.get("success"):
            self.info_text.SetValue(f"Generate failed: {result.get('error', 'Unknown error')}")
            self.SetStatusText("Generate failed")
            return

        circuit_data = result.get("circuit_data") or {}
        component_count = len(circuit_data.get("components", []))
        net_count = len(circuit_data.get("connections", []))

        summary = (
            f"Generation complete!\n\n"
            f"Template: {result.get('template_used', 'LLM/custom')}\n"
            f"Components: {component_count}\n"
            f"Nets: {net_count}\n"
            f"Time: {result.get('generation_time_ms', 0):.1f} ms\n"
            f"Download: {result.get('download_url', 'n/a')}"
        )
        self.info_text.SetValue(summary)
        self.SetStatusText("Generation complete")
    
    def _on_toggle_fixed(self, event):
        """Toggle fixed status of selected components."""
        # Implementation...
        pass
    
    def _on_add_constraint(self, event):
        """Add placement constraint."""
        # Implementation...
        pass
    
    def _on_settings(self, event):
        """Show settings dialog."""
        # Implementation...
        pass
    
    def _on_refresh(self, event):
        """Refresh board data."""
        self._extract_board_data()
    
    def _on_generate_tool(self, event):
        """Generate schematic from description."""
        # Implementation...
        pass
    
    def _on_exit(self, event):
        self.Close()


# ── Legacy Dialog for compatibility ───────────────────────────────────────────

class AIAssistantDialog(wx.Dialog):
    """Legacy dialog for simple usage."""
    
    def __init__(self, parent, board):
        super().__init__(parent, title="AI PCB Assistant", size=(500, 400))
        self.board = board
        self._init_ui()
    
    def _init_ui(self):
        # Simple UI for basic usage
        panel = wx.Panel(self)
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        sizer.Add(wx.StaticText(panel, label="Use the full AI PCB Assistant window\n"
                                           "for advanced features."), 0, wx.ALL, 20)
        
        btn_open = wx.Button(panel, label="Open Full Interface")
        btn_open.Bind(wx.EVT_BUTTON, self._on_open_full)
        sizer.Add(btn_open, 0, wx.ALIGN_CENTER | wx.ALL, 10)
        
        panel.SetSizer(sizer)
    
    def _on_open_full(self, event):
        self.EndModal(wx.ID_OK)
        frame = AIPCBFrame(None, self.board)
        frame.Show()


# ── Registration ─────────────────────────────────────────────────────────────

plugin = AIPlacementPlugin()
plugin.register()