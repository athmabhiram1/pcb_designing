# AI PCB Design Platform - Week-by-Week Execution Guide
## Detailed Task Breakdown with Code Examples

**Version**: 1.0  
**Date**: February 13, 2026  
**Timeline**: 24 Weeks to Production MVP

---

## How to Use This Guide

**Each week includes:**
1. ✅ **Objectives**: Clear goals for the week
2. 📋 **Tasks**: Specific actionable items
3. 💻 **Code Examples**: Copy-paste starting points
4. 🧪 **Validation**: How to verify work is correct
5. ⏱️ **Time Estimates**: Realistic hour breakdowns
6. 🚧 **Blockers**: Potential issues to watch for

**Progress Tracking**:
- Mark tasks as ✅ Done, 🔄 In Progress, or ❌ Blocked
- Update daily in GitHub Issues
- Review at end of each week

---

## Phase 1: Foundation & Schematic MVP (Weeks 1-6)

### Week 1: Research & Development Environment Setup

**Objectives**:
- ✅ Understand KiCad IPC API thoroughly
- ✅ Set up complete development environment
- ✅ Validate all critical dependencies work
- ✅ Create project repository structure

**Tasks**:

#### Day 1-2: KiCad IPC API Research

**Hours**: 12h  
**Priority**: CRITICAL

```bash
# Install KiCad 9.0
sudo add-apt-repository ppa:kicad/kicad-9.0-nightly
sudo apt update
sudo apt install kicad

# Verify IPC API is available
kicad --help | grep ipc

# Read official documentation
firefox https://dev-docs.kicad.org/en/apis-and-binding/ipc-api/
```

**Deliverables**:
- [✅] KiCad 9.0 installed and working
- [✅] Documentation notes on IPC API (save as `docs/kicad-ipc-api-notes.md`)
- [✅] List of available IPC methods documented

**Validation**:
```bash
# Test KiCad launches with IPC API
kicad-cli version
# Should show v9.0+
```

#### Day 3: Python Environment & Dependencies

**Hours**: 6h  
**Priority**: HIGH

```bash
# Create project directory
mkdir ai-pcb-design-platform
cd ai-pcb-design-platform
git init

# Create Python virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install core dependencies
pip install --upgrade pip
pip install kicad-python==0.3.0  # IPC API client
pip install fastapi uvicorn pydantic
pip install llama-cpp-python --break-system-packages
pip install instructor pydantic

# Create requirements.txt
cat > requirements.txt << 'EOF'
# Core
fastapi==0.109.2
uvicorn[standard]==0.27.1
pydantic==2.6.1

# KiCad Integration
kicad-python==0.3.0

# LLM
llama-cpp-python==0.2.27
instructor==0.5.2

# Schematic Generation
skidl==1.2.0

# Placement Algorithms
scipy==1.12.0
networkx==3.2.1
numpy==1.26.3

# Simulation
PySpice==1.5

# Geometry
shapely==2.0.2
rtree==1.1.0

# Database
chromadb==0.4.22
sqlite3

# Testing
pytest==8.0.0
pytest-asyncio==0.23.4
httpx==0.26.0
EOF

pip install -r requirements.txt
```

**Deliverables**:
- [✅] Virtual environment created
- [✅] All dependencies installed without errors
- [✅] requirements.txt committed to Git

**Validation**:
```bash
# Test imports
python -c "import kicad; print('kicad-python OK')"
python -c "import fastapi; print('FastAPI OK')"
python -c "import llama_cpp; print('llama-cpp-python OK')"
python -c "import skidl; print('SKiDL OK')"

# All should print "OK"
```

#### Day 4-5: Project Structure Setup

**Hours**: 10h  
**Priority**: HIGH

```bash
# Create directory structure
mkdir -p {
    plugin,
    ai_backend/{services,models,utils},
    tests/{unit,integration},
    docs,
    scripts,
    data/{models,cache,examples}
}

# Create initial files
touch plugin/__init__.py
touch plugin/plugin.py
touch plugin/metadata.json

touch ai_backend/__init__.py
touch ai_backend/server.py
touch ai_backend/services/__init__.py

touch README.md
touch LICENSE
touch .gitignore
```

**File: `.gitignore`**
```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
*.egg-info/
.pytest_cache/

# Models (large files)
data/models/*.gguf
data/models/*.onnx
*.bin
*.safetensors

# IDEs
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# KiCad
*.kicad_pcb-bak
*.kicad_sch-bak
*-cache.lib
*-rescue.lib

# Logs
*.log
logs/
```

**File: `README.md`**
```markdown
# AI PCB Design Platform

Open-source, local-first AI assistant for KiCad PCB design.

## Features

- 🤖 AI-powered schematic generation (natural language → circuit)
- 📐 Intelligent component placement optimization
- 🔌 Automated routing with FreeRouting
- ✅ Real-time DFM checking (JLCPCB, PCBWay rules)
- 🔬 SPICE simulation integration

## Installation

See [INSTALL.md](docs/INSTALL.md) for detailed instructions.

## Development

### Prerequisites
- Python 3.10+
- KiCad 9.0+
- 16GB RAM recommended
- Linux/macOS (Windows support coming soon)

### Setup
\`\`\`bash
# Clone repository
git clone https://github.com/yourusername/ai-pcb-design-platform.git
cd ai-pcb-design-platform

# Install dependencies
pip install -r requirements.txt

# Download models
./scripts/download_models.sh

# Run backend
python ai_backend/server.py

# Install plugin (separate terminal)
./scripts/install_plugin.sh
\`\`\`

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## License

MIT License - see [LICENSE](LICENSE)

## Contributing

Contributions welcome! See [CONTRIBUTING.md](docs/CONTRIBUTING.md)
```

**Deliverables**:
- [✅] Complete directory structure
- [✅] README.md with basic info
- [✅] .gitignore configured
- [✅] Initial commit to Git

**Validation**:
```bash
# Check structure
tree -L 2

# Should show:
# .
# ├── ai_backend/
# │   ├── __init__.py
# │   ├── server.py
# │   └── services/
# ├── plugin/
# │   ├── __init__.py
# │   ├── plugin.py
# │   └── metadata.json
# ├── tests/
# ├── docs/
# ├── requirements.txt
# └── README.md

# Make initial commit
git add .
git commit -m "Initial project structure"
```

---

### Week 2: KiCad Plugin Shell & IPC Integration

**Objectives**:
- ✅ Create minimal KiCad plugin that loads in KiCad 9.0
- ✅ Establish IPC connection to running KiCad instance
- ✅ Read board state via IPC API
- ✅ Test basic read/write operations

**Tasks**:

#### Day 1-2: Plugin Metadata & Entry Point

**Hours**: 10h  
**Priority**: CRITICAL

**File: `plugin/metadata.json`**
```json
{
  "$schema": "https://go.kicad.org/pcm/schemas/v1",
  "name": "AI PCB Designer",
  "description": "AI-powered assistant for PCB design (schematic generation, placement, routing)",
  "description_full": "Open-source AI assistant that generates schematics from natural language, optimizes component placement, and automates routing. Runs 100% locally with no cloud dependency.",
  "identifier": "com.github.yourusername.ai-pcb-designer",
  "type": "plugin",
  "author": {
    "name": "Your Name",
    "contact": {
      "email": "your.email@example.com"
    }
  },
  "maintainer": {
    "name": "Your Name",
    "contact": {
      "email": "your.email@example.com"
    }
  },
  "license": "MIT",
  "resources": {
    "homepage": "https://github.com/yourusername/ai-pcb-design-platform"
  },
  "versions": [
    {
      "version": "0.1.0",
      "status": "testing",
      "kicad_version": "9.0",
      "download_url": "https://github.com/yourusername/ai-pcb-design-platform/releases/download/v0.1.0/plugin.zip",
      "download_size": 50000,
      "download_sha256": "...",
      "install_size": 100000,
      "platforms": ["linux", "macos", "windows"]
    }
  ]
}
```

**File: `plugin/__init__.py`**
```python
"""
AI PCB Designer KiCad Plugin

This plugin provides AI-powered assistance for PCB design:
- Schematic generation from natural language
- Intelligent component placement
- Automated routing
- Real-time DFM checking
"""

__version__ = "0.1.0"
__author__ = "Your Name"

# Register plugin with KiCad
import pcbnew

class AIPCBDesignerPlugin(pcbnew.ActionPlugin):
    """Main plugin class registered with KiCad."""
    
    def defaults(self):
        """Plugin metadata shown in KiCad UI."""
        self.name = "AI PCB Designer"
        self.category = "AI Tools"
        self.description = "AI-powered PCB design assistant"
        self.show_toolbar_button = True
        self.icon_file_name = "icon.png"  # Optional
    
    def Run(self):
        """Called when user clicks plugin button."""
        from .plugin import run_plugin
        run_plugin()

# Register plugin
AIPCBDesignerPlugin().register()
```

**File: `plugin/plugin.py`**
```python
"""
Main plugin logic - handles user interaction and AI backend communication.
"""

import os
import sys
import wx
import requests
from typing import Optional

# Configuration
BACKEND_URL = "http://localhost:8765"

def check_backend_running() -> bool:
    """Check if AI backend is running."""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def show_backend_error():
    """Show error dialog if backend is not running."""
    dlg = wx.MessageDialog(
        None,
        "AI Backend is not running!\n\n"
        "Please start the backend:\n"
        "  python ai_backend/server.py\n\n"
        "Or install it:\n"
        "  ./scripts/install_backend.sh",
        "AI Backend Not Found",
        wx.OK | wx.ICON_ERROR
    )
    dlg.ShowModal()
    dlg.Destroy()

def run_plugin():
    """Main plugin entry point."""
    # Check if backend is running
    if not check_backend_running():
        show_backend_error()
        return
    
    # Show main dialog
    app = wx.App()
    dialog = AIPCBDesignerDialog(None)
    dialog.ShowModal()
    dialog.Destroy()

class AIPCBDesignerDialog(wx.Dialog):
    """Main plugin dialog."""
    
    def __init__(self, parent):
        wx.Dialog.__init__(
            self, parent,
            title="AI PCB Designer",
            size=(600, 400)
        )
        
        # Create UI
        panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)
        
        # Title
        title = wx.StaticText(panel, label="AI PCB Design Assistant")
        title_font = title.GetFont()
        title_font.PointSize += 3
        title_font = title_font.Bold()
        title.SetFont(title_font)
        vbox.Add(title, flag=wx.ALL|wx.ALIGN_CENTER, border=10)
        
        # Prompt input
        prompt_label = wx.StaticText(panel, label="Describe your circuit:")
        vbox.Add(prompt_label, flag=wx.LEFT|wx.RIGHT|wx.TOP, border=10)
        
        self.prompt_text = wx.TextCtrl(
            panel,
            style=wx.TE_MULTILINE,
            value="Example: 555 timer astable circuit blinking an LED at 1Hz"
        )
        vbox.Add(self.prompt_text, proportion=1, flag=wx.EXPAND|wx.ALL, border=10)
        
        # Constraints (optional)
        constraints_label = wx.StaticText(panel, label="Constraints (optional):")
        vbox.Add(constraints_label, flag=wx.LEFT|wx.RIGHT, border=10)
        
        constraints_box = wx.BoxSizer(wx.HORIZONTAL)
        
        voltage_label = wx.StaticText(panel, label="Supply Voltage:")
        constraints_box.Add(voltage_label, flag=wx.ALIGN_CENTER_VERTICAL|wx.LEFT, border=10)
        
        self.voltage_text = wx.TextCtrl(panel, value="5V", size=(60, -1))
        constraints_box.Add(self.voltage_text, flag=wx.LEFT, border=5)
        
        vbox.Add(constraints_box, flag=wx.EXPAND|wx.ALL, border=5)
        
        # Buttons
        button_box = wx.BoxSizer(wx.HORIZONTAL)
        
        generate_btn = wx.Button(panel, label="Generate Schematic")
        generate_btn.Bind(wx.EVT_BUTTON, self.on_generate)
        button_box.Add(generate_btn, flag=wx.ALL, border=5)
        
        cancel_btn = wx.Button(panel, wx.ID_CANCEL, label="Close")
        button_box.Add(cancel_btn, flag=wx.ALL, border=5)
        
        vbox.Add(button_box, flag=wx.ALIGN_RIGHT|wx.ALL, border=10)
        
        panel.SetSizer(vbox)
        
        # Status bar
        self.status = self.CreateStatusBar()
        self.status.SetStatusText("Ready")
    
    def on_generate(self, event):
        """Called when user clicks Generate button."""
        prompt = self.prompt_text.GetValue()
        voltage = self.voltage_text.GetValue()
        
        if not prompt:
            wx.MessageBox("Please enter a circuit description", "Error", wx.OK | wx.ICON_ERROR)
            return
        
        # Show progress
        self.status.SetStatusText("Generating circuit...")
        wx.SafeYield()
        
        try:
            # Call backend API
            response = requests.post(
                f"{BACKEND_URL}/api/generate/schematic",
                json={
                    "prompt": prompt,
                    "constraints": {"supply_voltage": voltage},
                    "output_format": "skidl"
                },
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    # TODO: Apply schematic to KiCad
                    self.status.SetStatusText(f"Generated in {data.get('generation_time', 0):.1f}s")
                    wx.MessageBox(
                        f"Schematic generated successfully!\n\n"
                        f"Components: {len(data.get('components', []))}\n"
                        f"Time: {data.get('generation_time', 0):.1f}s",
                        "Success",
                        wx.OK | wx.ICON_INFORMATION
                    )
                else:
                    error = data.get("error", "Unknown error")
                    self.status.SetStatusText("Generation failed")
                    wx.MessageBox(f"Error: {error}", "Generation Failed", wx.OK | wx.ICON_ERROR)
            else:
                self.status.SetStatusText("Backend error")
                wx.MessageBox(
                    f"Backend returned error {response.status_code}",
                    "Error",
                    wx.OK | wx.ICON_ERROR
                )
        
        except requests.Timeout:
            self.status.SetStatusText("Request timed out")
            wx.MessageBox(
                "Request timed out (>60s). Backend may be overloaded.",
                "Timeout",
                wx.OK | wx.ICON_ERROR
            )
        
        except Exception as e:
            self.status.SetStatusText("Error")
            wx.MessageBox(f"Error: {str(e)}", "Error", wx.OK | wx.ICON_ERROR)
```

**Deliverables**:
- [✅] Plugin metadata created
- [✅] Plugin loads in KiCad without errors
- [✅] Dialog opens when clicking plugin button
- [✅] Backend health check works

**Validation**:
```bash
# Install plugin for testing
ln -s $(pwd)/plugin ~/.kicad/9.0/3rdparty/plugins/ai-pcb-designer

# Start KiCad
kicad

# In KiCad:
# 1. Open PCB Editor
# 2. Tools → External Plugins → Refresh Plugins
# 3. Click "AI PCB Designer" button in toolbar
# 4. Dialog should open

# Check logs
tail -f ~/.kicad/9.0/kicad.log
```

#### Day 3-5: IPC API Integration

**Hours**: 15h  
**Priority**: CRITICAL

**File: `plugin/ipc_client.py`**
```python
"""
KiCad IPC API client - handles communication with KiCad via IPC.
"""

import os
from kicad import KiCad, Board, Schematic
from typing import Optional, List, Dict, Any

class KiCadIPCClient:
    """Client for KiCad IPC API."""
    
    def __init__(self):
        """Initialize IPC client."""
        # Read environment variables set by KiCad
        self.socket_path = os.environ.get('KICAD_API_SOCKET')
        self.token = os.environ.get('KICAD_API_TOKEN')
        
        if not self.socket_path:
            raise RuntimeError(
                "KICAD_API_SOCKET not set. "
                "Plugin must be run from within KiCad."
            )
        
        # Connect to KiCad
        self.kicad = KiCad(self.socket_path, self.token)
        print(f"Connected to KiCad via {self.socket_path}")
    
    def get_board(self) -> Optional[Board]:
        """Get current PCB board."""
        try:
            board = self.kicad.get_board()
            print(f"Got board: {board.filename if board else 'None'}")
            return board
        except Exception as e:
            print(f"Error getting board: {e}")
            return None
    
    def get_schematic(self) -> Optional[Schematic]:
        """Get current schematic."""
        try:
            schematic = self.kicad.get_schematic()
            print(f"Got schematic: {schematic.filename if schematic else 'None'}")
            return schematic
        except Exception as e:
            print(f"Error getting schematic: {e}")
            return None
    
    def add_component(
        self,
        board: Board,
        reference: str,
        footprint: str,
        x: float,
        y: float,
        rotation: float = 0.0
    ) -> bool:
        """Add component to board."""
        try:
            # Create footprint instance
            footprint_inst = board.add_footprint(
                footprint=footprint,
                reference=reference,
                position=(x, y),
                rotation=rotation
            )
            
            print(f"Added component {reference} at ({x}, {y})")
            return True
        
        except Exception as e:
            print(f"Error adding component: {e}")
            return False
    
    def get_board_info(self, board: Board) -> Dict[str, Any]:
        """Get board information."""
        return {
            'filename': board.filename,
            'width': board.get_width(),
            'height': board.get_height(),
            'layer_count': len(board.layers),
            'component_count': len(board.footprints),
            'net_count': len(board.nets)
        }
    
    def get_components(self, board: Board) -> List[Dict[str, Any]]:
        """Get all components on board."""
        components = []
        
        for footprint in board.footprints:
            components.append({
                'reference': footprint.reference,
                'value': footprint.value,
                'footprint': footprint.footprint_name,
                'x': footprint.position.x,
                'y': footprint.position.y,
                'rotation': footprint.rotation,
                'layer': footprint.layer
            })
        
        return components

# -------------------------------------------------------------------
# Testing
# -------------------------------------------------------------------

if __name__ == "__main__":
    # This will only work when run from KiCad plugin context
    try:
        client = KiCadIPCClient()
        
        # Test getting board
        board = client.get_board()
        if board:
            info = client.get_board_info(board)
            print(f"Board info: {info}")
            
            components = client.get_components(board)
            print(f"Found {len(components)} components")
            for comp in components[:5]:  # Show first 5
                print(f"  {comp['reference']}: {comp['value']} @ ({comp['x']}, {comp['y']})")
        else:
            print("No board open")
    
    except RuntimeError as e:
        print(f"Error: {e}")
        print("This script must be run from within KiCad plugin context")
```

**Updated: `plugin/plugin.py`** (integrate IPC client)
```python
# ... (previous imports)
from .ipc_client import KiCadIPCClient

class AIPCBDesignerDialog(wx.Dialog):
    def __init__(self, parent):
        # ... (previous init code)
        
        # Initialize IPC client
        try:
            self.ipc_client = KiCadIPCClient()
            self.board = self.ipc_client.get_board()
            
            if self.board:
                info = self.ipc_client.get_board_info(self.board)
                self.status.SetStatusText(
                    f"Connected to board: {info['component_count']} components"
                )
            else:
                self.status.SetStatusText("No board open - please create/open a PCB")
        
        except Exception as e:
            self.status.SetStatusText(f"IPC error: {e}")
            wx.MessageBox(
                f"Could not connect to KiCad:\n{e}",
                "IPC Error",
                wx.OK | wx.ICON_ERROR
            )
    
    # ... (rest of code)
```

**Deliverables**:
- [✅] IPC client successfully connects to KiCad
- [✅] Can read board state (components, nets)
- [✅] Can add test component to board
- [✅] No errors in KiCad logs

**Validation**:
```python
# Test script (run from KiCad plugin)
from plugin.ipc_client import KiCadIPCClient

client = KiCadIPCClient()
board = client.get_board()

if board:
    # Print board info
    info = client.get_board_info(board)
    print(f"Board: {info['filename']}")
    print(f"Size: {info['width']}x{info['height']} mm")
    print(f"Components: {info['component_count']}")
    
    # Add test component
    success = client.add_component(
        board=board,
        reference="TEST1",
        footprint="Resistor_SMD:R_0805_2012Metric",
        x=50.0,
        y=50.0,
        rotation=0.0
    )
    
    if success:
        print("✅ Successfully added test component")
    else:
        print("❌ Failed to add component")
else:
    print("⚠️ No board open")
```

---

### Week 3: FastAPI Backend + LLM Integration

**Objectives**:
- ✅ Create working FastAPI server
- ✅ Load and test DeepSeek-Coder 7B model
- ✅ Implement structured output with instructor
- ✅ Test end-to-end: UI → Backend → LLM → Response

**[Continues with detailed day-by-day breakdown...]**

---

## Summary of All 24 Weeks

| Week | Phase | Focus | Deliverable |
|------|-------|-------|-------------|
| 1 | Setup | Environment & Research | Dev environment ready |
| 2 | Plugin | KiCad IPC integration | Plugin loads, reads board |
| 3 | Backend | FastAPI + LLM | Backend responds to requests |
| 4 | Schematic | SKiDL generation | End-to-end schematic creation |
| 5 | Testing | Phase 1 validation | 80% test coverage |
| 6 | Buffer | Fix issues, refactor | Phase 1 complete |
| 7-8 | Placement | Heuristic algorithms | Placement works on simple boards |
| 9-10 | Routing | FreeRouting integration | Auto-routing functional |
| 11-12 | DFM | Real-time checking | DFM violations detected |
| 13-14 | Simulation | ngspice integration | Basic simulation works |
| 15-16 | Polish | UI/UX improvements | Professional user experience |
| 17-18 | Testing | Integration tests | <5 critical bugs |
| 19-20 | Documentation | User guides, videos | Complete documentation |
| 21-22 | Packaging | Installers for OS | One-click install |
| 23 | Beta | Public testing | 50+ beta users |
| 24 | Release | Bug fixes, launch | v0.1.0 released |

---

**END OF WEEK-BY-WEEK GUIDE**

For complete implementation:
1. Follow each week sequentially
2. Don't skip validation steps
3. Commit code daily
4. Review weekly progress
5. Adjust timeline as needed

**Next Steps**:
→ Start Week 1 Day 1 immediately
→ Set up project tracking (GitHub Projects)
→ Join KiCad Discord for community support
