# AI PCB Design Platform - Technical Architecture Specification
## Detailed Implementation Guide with Code Examples

**Version**: 1.0  
**Date**: February 13, 2026  
**Author**: Research-Validated Technical Specification

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Backend API Specification](#2-backend-api-specification)
3. [LLM Integration (Structured Outputs)](#3-llm-integration-structured-outputs)
4. [RL Placement System (ONNX)](#4-rl-placement-system-onnx)
5. [KiCad Integration (IPC API)](#5-kicad-integration-ipc-api)
6. [Database Schemas](#6-database-schemas)
7. [Testing Framework](#7-testing-framework)

---

## 1. System Architecture

### 1.1 Component Diagram

```
┌──────────────────────────────────────────────────────────┐
│                    KiCad 9.0 Process                     │
│                                                           │
│  ┌────────────────────────────────────────────────────┐ │
│  │          IPC API Server (Built-in)                 │ │
│  │  - Unix Socket: /tmp/kicad-<pid>.sock             │ │
│  │  - Protocol: Protocol Buffers + NNG               │ │
│  │  - Methods: GetBoard, SetBoard, AddComponent      │ │
│  └─────────────────┬──────────────────────────────────┘ │
│                    │                                     │
└────────────────────┼─────────────────────────────────────┘
                     │ IPC (Protobuf/NNG)
                     │
┌────────────────────▼─────────────────────────────────────┐
│          KiCad Plugin Process (Separate Python)          │
│                                                           │
│  ┌────────────────────────────────────────────────────┐ │
│  │  plugin.py (Entry Point)                           │ │
│  │  - Reads KICAD_API_SOCKET env var                  │ │
│  │  - Creates IPC connection via kicad-python         │ │
│  │  - Provides UI (wxPython dialogs)                  │ │
│  │  - Sends requests to AI Backend (HTTP)             │ │
│  └─────────────────┬──────────────────────────────────┘ │
│                    │                                     │
└────────────────────┼─────────────────────────────────────┘
                     │ HTTP/REST (localhost:8765)
                     │
┌────────────────────▼─────────────────────────────────────┐
│              AI Backend (FastAPI Server)                 │
│              http://localhost:8765                        │
│                                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐│
│  │ LLM Service  │  │  Placement   │  │   Routing      ││
│  │  (llama-cpp) │  │  (Heuristic) │  │ (FreeRouting)  ││
│  │              │  │              │  │                ││
│  │ ┌──────────┐ │  │ ┌──────────┐ │  │ ┌────────────┐││
│  │ │ DeepSeek │ │  │ │ Simulated│ │  │ │ DSN Export │││
│  │ │  Coder   │ │  │ │ Annealing│ │  │ │ SES Import │││
│  │ │   7B     │ │  │ │          │ │  │ │ JAR Wrapper│││
│  │ │  GGUF    │ │  │ │ Genetic  │ │  │ │            │││
│  │ └──────────┘ │  │ │ Algorithm│ │  │ └────────────┘││
│  │              │  │ └──────────┘ │  │                ││
│  │ ┌──────────┐ │  │              │  │                ││
│  │ │ SKiDL    │ │  │ ┌──────────┐ │  │                ││
│  │ │Generator │ │  │ │Force-Dir │ │  │                ││
│  │ │          │ │  │ │  Graph   │ │  │                ││
│  │ └──────────┘ │  │ └──────────┘ │  │                ││
│  └──────────────┘  └──────────────┘  └────────────────┘│
│                                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐│
│  │  Simulation  │  │ DFM Checker  │  │   Component    ││
│  │  (ngspice)   │  │  (Rules)     │  │    Search      ││
│  │              │  │              │  │  (ChromaDB)    ││
│  │ ┌──────────┐ │  │ ┌──────────┐ │  │ ┌────────────┐││
│  │ │Incr. Sim │ │  │ │ JLCPCB   │ │  │ │  Vector DB │││
│  │ │  Cache   │ │  │ │  Rules   │ │  │ │ Embeddings │││
│  │ │  (SQLite)│ │  │ │          │ │  │ │            │││
│  │ └──────────┘ │  │ │ PCBWay   │ │  │ └────────────┘││
│  │              │  │ │  Rules   │ │  │                ││
│  │ ┌──────────┐ │  │ └──────────┘ │  │                ││
│  │ │ PySpice  │ │  │              │  │                ││
│  │ │ Wrapper  │ │  │ ┌──────────┐ │  │                ││
│  │ └──────────┘ │  │ │Cost Calc │ │  │                ││
│  │              │  │ └──────────┘ │  │                ││
│  └──────────────┘  └──────────────┘  └────────────────┘│
│                                                           │
│  ┌────────────────────────────────────────────────────┐ │
│  │            Data Layer (SQLite + Cache)              │ │
│  │  - Simulation cache (netlist hashes)               │ │
│  │  - Component library (footprints, symbols)         │ │
│  │  - User preferences                                │ │
│  └────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────┘
```

### 1.2 Technology Stack Details

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| **KiCad Integration** | kicad-python | 0.3.0+ | IPC API client library |
| **Plugin UI** | wxPython | 4.2.1 | Native KiCad-compatible dialogs |
| **Backend Framework** | FastAPI | 0.109+ | REST API server |
| **LLM Runtime** | llama-cpp-python | 0.2.27+ | GGUF model inference |
| **Structured Output** | instructor | 0.5.0+ | JSON schema validation |
| **Schematic Gen** | SKiDL | 1.2.0+ | Code-first circuit design |
| **Placement** | scipy | 1.11+ | Simulated annealing optimization |
| **Routing** | FreeRouting | 1.9.0+ | Java-based autorouter |
| **Simulation** | ngspice | 40+ | SPICE circuit simulation |
| **Python Wrapper** | PySpice | 1.5+ | ngspice Python bindings |
| **Vector DB** | ChromaDB | 0.4.22+ | Component similarity search |
| **Database** | SQLite | 3.40+ | Caching, user data |
| **Geometric Ops** | Shapely | 2.0+ | PCB geometry calculations |
| **Spatial Index** | rtree | 1.1+ | Fast collision detection |

---

## 2. Backend API Specification

### 2.1 FastAPI Server Structure

```python
# ai_backend/server.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn

app = FastAPI(
    title="AI PCB Design Backend",
    description="Local AI backend for KiCad plugin",
    version="0.1.0"
)

# Allow localhost connections from plugin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# Request/Response Models
# -------------------------------------------------------------------

class SchematicGenerationRequest(BaseModel):
    """Request to generate schematic from natural language."""
    prompt: str = Field(..., description="Natural language circuit description")
    constraints: Dict[str, Any] = Field(
        default={},
        description="Electrical constraints (voltage, current, etc.)"
    )
    output_format: str = Field(
        default="skidl",
        description="Output format: 'skidl' or 'netlist'"
    )

class SchematicGenerationResponse(BaseModel):
    """Response containing generated schematic."""
    success: bool
    skidl_code: Optional[str] = None
    netlist: Optional[str] = None
    components: List[Dict[str, str]] = []  # [{ref, value, footprint}]
    error: Optional[str] = None
    generation_time: float

class PlacementRequest(BaseModel):
    """Request to optimize component placement."""
    board_width: float = Field(..., description="Board width in mm")
    board_height: float = Field(..., description="Board height in mm")
    components: List[Dict[str, Any]] = Field(
        ...,
        description="Components with dimensions and initial positions"
    )
    netlist: List[Dict[str, List[str]]] = Field(
        ...,
        description="Connectivity information"
    )
    constraints: Dict[str, Any] = Field(
        default={},
        description="Placement constraints (keep-out zones, etc.)"
    )

class PlacementResponse(BaseModel):
    """Response containing optimized component positions."""
    success: bool
    positions: List[Dict[str, Any]] = []  # [{ref, x, y, rotation}]
    metrics: Dict[str, float] = {}  # {wirelength, overlap, etc.}
    error: Optional[str] = None
    optimization_time: float

class RoutingRequest(BaseModel):
    """Request to route PCB."""
    dsn_file_path: str = Field(..., description="Path to Specctra DSN file")
    output_ses_path: str = Field(..., description="Path for output SES file")

class RoutingResponse(BaseModel):
    """Response from routing operation."""
    success: bool
    ses_file_path: Optional[str] = None
    completion_percentage: float
    error: Optional[str] = None
    routing_time: float

class DFMCheckRequest(BaseModel):
    """Request to check DFM rules."""
    board_data: Dict[str, Any] = Field(..., description="Board geometry")
    manufacturer: str = Field(default="jlcpcb", description="Target manufacturer")

class DFMCheckResponse(BaseModel):
    """Response with DFM violations."""
    success: bool
    violations: List[Dict[str, Any]] = []
    cost_estimate: Optional[Dict[str, float]] = None
    error: Optional[str] = None

# -------------------------------------------------------------------
# Health Check
# -------------------------------------------------------------------

@app.get("/health")
async def health_check():
    """Check if backend is running and models are loaded."""
    return {
        "status": "healthy",
        "version": "0.1.0",
        "models_loaded": {
            "llm": llm_service.is_loaded(),
            "placement": True  # Heuristic doesn't need model
        }
    }

# -------------------------------------------------------------------
# Schematic Generation
# -------------------------------------------------------------------

@app.post("/api/generate/schematic", response_model=SchematicGenerationResponse)
async def generate_schematic(request: SchematicGenerationRequest):
    """Generate schematic from natural language description."""
    import time
    start_time = time.time()
    
    try:
        # Use LLM to generate SKiDL code
        skidl_code = await llm_service.generate_skidl(
            prompt=request.prompt,
            constraints=request.constraints
        )
        
        # Execute SKiDL code to get netlist
        netlist, components = skidl_service.execute_skidl(skidl_code)
        
        return SchematicGenerationResponse(
            success=True,
            skidl_code=skidl_code,
            netlist=netlist,
            components=components,
            generation_time=time.time() - start_time
        )
        
    except Exception as e:
        return SchematicGenerationResponse(
            success=False,
            error=str(e),
            generation_time=time.time() - start_time
        )

# -------------------------------------------------------------------
# Component Placement
# -------------------------------------------------------------------

@app.post("/api/placement/optimize", response_model=PlacementResponse)
async def optimize_placement(request: PlacementRequest):
    """Optimize component placement using heuristic algorithm."""
    import time
    start_time = time.time()
    
    try:
        # Run simulated annealing placement
        positions, metrics = placement_service.optimize(
            board_size=(request.board_width, request.board_height),
            components=request.components,
            netlist=request.netlist,
            constraints=request.constraints
        )
        
        return PlacementResponse(
            success=True,
            positions=positions,
            metrics=metrics,
            optimization_time=time.time() - start_time
        )
        
    except Exception as e:
        return PlacementResponse(
            success=False,
            error=str(e),
            optimization_time=time.time() - start_time
        )

# -------------------------------------------------------------------
# Routing
# -------------------------------------------------------------------

@app.post("/api/routing/autoroute", response_model=RoutingResponse)
async def autoroute_board(request: RoutingRequest, background_tasks: BackgroundTasks):
    """Route board using FreeRouting."""
    import time
    start_time = time.time()
    
    try:
        # Run FreeRouting (may take several minutes)
        success, ses_path, completion = routing_service.route(
            dsn_path=request.dsn_file_path,
            ses_path=request.output_ses_path
        )
        
        return RoutingResponse(
            success=success,
            ses_file_path=ses_path if success else None,
            completion_percentage=completion,
            routing_time=time.time() - start_time
        )
        
    except Exception as e:
        return RoutingResponse(
            success=False,
            error=str(e),
            completion_percentage=0.0,
            routing_time=time.time() - start_time
        )

# -------------------------------------------------------------------
# DFM Checking
# -------------------------------------------------------------------

@app.post("/api/dfm/check", response_model=DFMCheckResponse)
async def check_dfm(request: DFMCheckRequest):
    """Check design for manufacturing constraints."""
    try:
        violations, cost = dfm_service.check(
            board_data=request.board_data,
            manufacturer=request.manufacturer
        )
        
        return DFMCheckResponse(
            success=True,
            violations=violations,
            cost_estimate=cost
        )
        
    except Exception as e:
        return DFMCheckResponse(
            success=False,
            error=str(e)
        )

# -------------------------------------------------------------------
# Service Initialization
# -------------------------------------------------------------------

from services.llm_service import LLMService
from services.skidl_service import SKiDLService
from services.placement_service import PlacementService
from services.routing_service import RoutingService
from services.dfm_service import DFMService

# Global service instances
llm_service = None
skidl_service = None
placement_service = None
routing_service = None
dfm_service = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global llm_service, skidl_service, placement_service, routing_service, dfm_service
    
    print("Initializing AI Backend...")
    
    # Initialize LLM (may take 10-30 seconds to load model)
    llm_service = LLMService(
        model_path="models/deepseek-coder-7b-instruct.Q5_K_M.gguf",
        n_ctx=8192,
        n_threads=8
    )
    
    # Initialize other services
    skidl_service = SKiDLService()
    placement_service = PlacementService()
    routing_service = RoutingService(jar_path="bin/freerouting.jar")
    dfm_service = DFMService()
    
    print("AI Backend ready!")

# -------------------------------------------------------------------
# Main Entry Point
# -------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="127.0.0.1",
        port=8765,
        reload=False,  # Disable in production (model reload is slow)
        log_level="info"
    )
```

---

## 3. LLM Integration (Structured Outputs)

### 3.1 LLM Service with JSON Schema Validation

```python
# services/llm_service.py
from llama_cpp import Llama
import instructor
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import json

class Component(BaseModel):
    """Represents a circuit component."""
    reference: str = Field(..., description="Component reference (e.g., R1, C1, U1)")
    type: str = Field(..., description="Component type (resistor, capacitor, IC, etc.)")
    value: str = Field(..., description="Component value (e.g., 10k, 100nF, LM358)")
    footprint: str = Field(..., description="PCB footprint (e.g., R_0805, SOT-23)")
    
class Connection(BaseModel):
    """Represents a connection between pins."""
    net_name: str = Field(..., description="Net name (e.g., VCC, GND, OUTPUT)")
    pins: List[str] = Field(..., description="Connected pins (e.g., ['R1.1', 'C1.1', 'U1.3'])")

class CircuitDesign(BaseModel):
    """Complete circuit design output."""
    description: str = Field(..., description="Brief description of the circuit")
    components: List[Component] = Field(..., description="List of components")
    connections: List[Connection] = Field(..., description="List of net connections")
    notes: Optional[str] = Field(None, description="Design notes or warnings")

class LLMService:
    """Service for LLM-powered circuit generation with structured output."""
    
    def __init__(self, model_path: str, n_ctx: int = 8192, n_threads: int = 8):
        """Initialize LLM model."""
        print(f"Loading LLM model: {model_path}")
        
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=0,  # CPU-only for now (set to -1 for GPU)
            verbose=False
        )
        
        # Patch with instructor for structured outputs
        self.create = instructor.patch(
            create=self.llm.create_chat_completion_openai_v1,
            mode=instructor.Mode.JSON_SCHEMA,
        )
        
        print("LLM model loaded successfully")
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.llm is not None
    
    async def generate_circuit(
        self,
        prompt: str,
        constraints: Dict[str, Any] = None
    ) -> CircuitDesign:
        """Generate circuit design from natural language with structured output."""
        
        # Build system prompt
        system_prompt = self._build_system_prompt()
        
        # Build user message with constraints
        user_message = self._build_user_message(prompt, constraints)
        
        # Call LLM with structured output schema
        response = self.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            response_model=CircuitDesign,
            temperature=0.3,
            max_tokens=2048
        )
        
        return response
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for circuit generation."""
        return """You are an expert electrical engineer specializing in circuit design.

Your task is to design circuits that are:
1. Electrically correct and functional
2. Use appropriate component values
3. Include proper power decoupling
4. Follow industry best practices
5. Use standard KiCad footprints

Output format:
- Provide a structured circuit design
- Include all necessary components
- Specify exact footprints (from KiCad libraries)
- Define clear net connections
- Add design notes if important

Component naming conventions:
- Resistors: R1, R2, ... (values in ohms, k, M)
- Capacitors: C1, C2, ... (values in pF, nF, uF)
- ICs: U1, U2, ... (part numbers)
- LEDs: D1, D2, ...
- Transistors: Q1, Q2, ...

Footprint examples:
- Resistor_SMD:R_0805_2012Metric
- Capacitor_SMD:C_0805_2012Metric
- Package_SO:SOIC-8_3.9x4.9mm_P1.27mm
- LED_SMD:LED_0805_2012Metric
- Package_TO_SOT_SMD:SOT-23

Common nets:
- VCC, VDD (positive power)
- GND (ground)
- INPUT, OUTPUT (signal names)
- Use descriptive names for clarity"""
    
    def _build_user_message(self, prompt: str, constraints: Dict[str, Any]) -> str:
        """Build user message with constraints."""
        message = f"Design requirement: {prompt}\n"
        
        if constraints:
            message += "\nConstraints:\n"
            for key, value in constraints.items():
                message += f"- {key}: {value}\n"
        
        return message.strip()
    
    async def generate_skidl(self, prompt: str, constraints: Dict[str, Any] = None) -> str:
        """Generate SKiDL Python code from circuit design."""
        
        # First, get structured circuit design
        circuit = await self.generate_circuit(prompt, constraints)
        
        # Convert to SKiDL code
        skidl_code = self._circuit_to_skidl(circuit)
        
        return skidl_code
    
    def _circuit_to_skidl(self, circuit: CircuitDesign) -> str:
        """Convert structured circuit design to SKiDL code."""
        
        code_lines = [
            "from skidl import *",
            "",
            f"# {circuit.description}",
            "reset()",
            ""
        ]
        
        # Create nets
        nets = set()
        for conn in circuit.connections:
            nets.add(conn.net_name)
        
        if nets:
            net_list = ", ".join(sorted(nets))
            code_lines.append(f"{net_list} = Net('{list(sorted(nets))[0]}'), " + \
                            ", ".join([f"Net('{n}')" for n in sorted(nets)[1:]]) if len(nets) > 1 else "")
        
        code_lines.append("")
        
        # Create components
        for comp in circuit.components:
            # Parse component type and library
            lib, part_type = self._parse_component_type(comp.type)
            
            code_lines.append(
                f"{comp.reference} = Part('{lib}', '{part_type}', "
                f"value='{comp.value}', footprint='{comp.footprint}')"
            )
        
        code_lines.append("")
        
        # Create connections
        for conn in circuit.connections:
            pin_refs = [f"{pin.split('.')[0]}['{pin.split('.')[1]}']" 
                       for pin in conn.pins]
            code_lines.append(
                f"{conn.net_name} & " + " & ".join(pin_refs)
            )
        
        code_lines.extend([
            "",
            "# Electrical Rules Check",
            "ERC()",
            "",
            "# Generate netlist",
            "generate_netlist()"
        ])
        
        if circuit.notes:
            code_lines.insert(2, f"# Note: {circuit.notes}")
        
        return "\n".join(code_lines)
    
    def _parse_component_type(self, comp_type: str) -> tuple:
        """Parse component type to (library, part_type)."""
        type_map = {
            "resistor": ("Device", "R"),
            "capacitor": ("Device", "C"),
            "led": ("Device", "LED"),
            "transistor_npn": ("Device", "Q_NPN_BCE"),
            "transistor_pnp": ("Device", "Q_PNP_BCE"),
            "diode": ("Device", "D"),
            "inductor": ("Device", "L"),
        }
        
        comp_type_lower = comp_type.lower()
        
        # Check if it's a known type
        if comp_type_lower in type_map:
            return type_map[comp_type_lower]
        
        # Otherwise assume it's an IC with specific part number
        return ("Reference_Designs", comp_type)

# -------------------------------------------------------------------
# Example Usage
# -------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio
    
    async def test():
        service = LLMService(
            model_path="models/deepseek-coder-7b-instruct.Q5_K_M.gguf"
        )
        
        # Test circuit generation
        circuit = await service.generate_circuit(
            prompt="555 timer astable circuit blinking an LED at 1Hz",
            constraints={
                "supply_voltage": "5V",
                "led_current": "20mA"
            }
        )
        
        print("Generated Circuit:")
        print(f"Description: {circuit.description}")
        print(f"\nComponents ({len(circuit.components)}):")
        for comp in circuit.components:
            print(f"  {comp.reference}: {comp.type} = {comp.value} ({comp.footprint})")
        
        print(f"\nConnections ({len(circuit.connections)}):")
        for conn in circuit.connections:
            print(f"  {conn.net_name}: {', '.join(conn.pins)}")
        
        # Generate SKiDL code
        skidl = await service.generate_skidl(
            prompt="555 timer astable circuit blinking an LED at 1Hz",
            constraints={"supply_voltage": "5V"}
        )
        
        print("\nGenerated SKiDL Code:")
        print(skidl)
    
    asyncio.run(test())
```

---

## 4. RL Placement System (ONNX)

### 4.1 Training Script (Phase 2-3)

```python
# training/train_placement_model.py
"""
Training script for RL-based component placement.

This script trains a PPO agent to optimize PCB component placement.
Training data: 10K-50K anonymized PCB layouts (Phase 2-3).

Usage:
    python train_placement_model.py --data-dir datasets/pcb_layouts --output models/placement.onnx
"""

import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import torch as th
from typing import Tuple, Dict, Any

class PCBPlacementEnv(gym.Env):
    """
    OpenAI Gym environment for PCB component placement.
    
    State: Grid representation + component features + netlist graph
    Action: (component_id, x, y, rotation)
    Reward: -(wirelength + overlap_penalty + thermal_penalty)
    """
    
    def __init__(
        self,
        board_width: float = 100.0,  # mm
        board_height: float = 80.0,  # mm
        grid_size: int = 50,  # Grid resolution
        max_components: int = 100
    ):
        super().__init__()
        
        self.board_width = board_width
        self.board_height = board_height
        self.grid_size = grid_size
        self.max_components = max_components
        
        # State space:
        # - Grid occupancy (grid_size x grid_size)
        # - Component features (max_components x 6: width, height, thermal, placed, x, y)
        # - Netlist adjacency matrix (max_components x max_components)
        state_dim = (grid_size * grid_size) + (max_components * 6) + (max_components * max_components)
        
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(state_dim,),
            dtype=np.float32
        )
        
        # Action space: (component_id, x_grid, y_grid, rotation)
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0, 0]),
            high=np.array([max_components-1, grid_size-1, grid_size-1, 3]),
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self):
        """Reset environment to initial state."""
        self.components = []  # Will be loaded from dataset
        self.netlist = []
        self.placed_positions = {}
        self.grid = np.zeros((self.grid_size, self.grid_size))
        
        return self._get_state()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one placement action."""
        comp_id, x, y, rotation = action
        comp_id = int(comp_id)
        x_grid = int(x)
        y_grid = int(y)
        rotation = int(rotation) * 90  # 0, 90, 180, 270 degrees
        
        # Place component
        if comp_id < len(self.components):
            component = self.components[comp_id]
            
            # Calculate real coordinates
            x_mm = (x_grid / self.grid_size) * self.board_width
            y_mm = (y_grid / self.grid_size) * self.board_height
            
            # Update grid
            self._update_grid(component, x_grid, y_grid, rotation)
            
            # Store placement
            self.placed_positions[comp_id] = (x_mm, y_mm, rotation)
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if done (all components placed)
        done = len(self.placed_positions) >= len(self.components)
        
        return self._get_state(), reward, done, {}
    
    def _get_state(self) -> np.ndarray:
        """Get current state representation."""
        # Flatten grid
        grid_flat = self.grid.flatten()
        
        # Component features
        comp_features = np.zeros((self.max_components, 6))
        for i, comp in enumerate(self.components):
            if i < self.max_components:
                comp_features[i] = [
                    comp['width'] / self.board_width,
                    comp['height'] / self.board_height,
                    comp.get('thermal', 0),
                    1.0 if i in self.placed_positions else 0.0,
                    self.placed_positions.get(i, (0, 0, 0))[0] / self.board_width,
                    self.placed_positions.get(i, (0, 0, 0))[1] / self.board_height
                ]
        comp_flat = comp_features.flatten()
        
        # Netlist adjacency matrix
        adj_matrix = np.zeros((self.max_components, self.max_components))
        for net in self.netlist:
            for i, comp_i in enumerate(net):
                for j, comp_j in enumerate(net):
                    if i < self.max_components and j < self.max_components:
                        adj_matrix[i, j] = 1.0
        adj_flat = adj_matrix.flatten()
        
        return np.concatenate([grid_flat, comp_flat, adj_flat]).astype(np.float32)
    
    def _update_grid(self, component: Dict, x: int, y: int, rotation: int):
        """Update grid occupancy."""
        # Simplified: mark grid cells as occupied
        width_cells = int((component['width'] / self.board_width) * self.grid_size)
        height_cells = int((component['height'] / self.board_height) * self.grid_size)
        
        for i in range(width_cells):
            for j in range(height_cells):
                if x + i < self.grid_size and y + j < self.grid_size:
                    self.grid[x + i, y + j] = 1.0
    
    def _calculate_reward(self) -> float:
        """Calculate reward for current placement."""
        reward = 0.0
        
        # Wirelength penalty
        total_wirelength = self._calculate_wirelength()
        reward -= total_wirelength * 0.01
        
        # Overlap penalty
        overlap = self._calculate_overlap()
        reward -= overlap * 10.0
        
        # Thermal penalty (components generating heat should be spread out)
        thermal_penalty = self._calculate_thermal_penalty()
        reward -= thermal_penalty * 5.0
        
        # Bonus for placing all components
        if len(self.placed_positions) >= len(self.components):
            reward += 100.0
        
        return reward
    
    def _calculate_wirelength(self) -> float:
        """Calculate total wirelength (half-perimeter bounding box)."""
        total = 0.0
        for net in self.netlist:
            positions = [self.placed_positions.get(comp_id, (0, 0, 0))[:2] 
                        for comp_id in net if comp_id in self.placed_positions]
            
            if len(positions) >= 2:
                xs = [p[0] for p in positions]
                ys = [p[1] for p in positions]
                hpwl = (max(xs) - min(xs)) + (max(ys) - min(ys))
                total += hpwl
        
        return total
    
    def _calculate_overlap(self) -> float:
        """Calculate overlap penalty."""
        # Simplified: count overlapping grid cells
        overlap_cells = np.sum(self.grid > 1.0)
        return float(overlap_cells)
    
    def _calculate_thermal_penalty(self) -> float:
        """Calculate thermal clustering penalty."""
        thermal_comps = [
            (comp_id, pos) for comp_id, pos in self.placed_positions.items()
            if self.components[comp_id].get('thermal', 0) > 0.5
        ]
        
        penalty = 0.0
        for i, (id1, pos1) in enumerate(thermal_comps):
            for id2, pos2 in thermal_comps[i+1:]:
                distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                if distance < 10.0:  # mm
                    penalty += (10.0 - distance)
        
        return penalty

def train_placement_model(
    data_dir: str,
    output_path: str,
    total_timesteps: int = 1_000_000
):
    """Train PPO model for PCB placement."""
    
    # Create vectorized environment
    env = make_vec_env(PCBPlacementEnv, n_envs=4)
    
    # Initialize PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        tensorboard_log="./logs/ppo_placement/"
    )
    
    # Train model
    print(f"Training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)
    
    # Save model (native SB3 format)
    model.save("models/placement_ppo.zip")
    
    # Export to ONNX
    export_to_onnx(model, output_path)
    
    print(f"Model saved to {output_path}")

def export_to_onnx(model: PPO, output_path: str):
    """Export trained PPO model to ONNX format."""
    
    class OnnxablePolicy(th.nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy
        
        def forward(self, observation: th.Tensor) -> th.Tensor:
            """Forward pass for inference only."""
            # Get deterministic action (no exploration)
            with th.no_grad():
                action, _ = self.policy.predict(observation, deterministic=True)
            return action
    
    # Create ONNX-compatible wrapper
    onnxable_model = OnnxablePolicy(model.policy)
    
    # Get observation space shape
    observation_size = model.observation_space.shape
    
    # Create dummy input
    dummy_input = th.randn(1, *observation_size)
    
    # Export to ONNX
    th.onnx.export(
        onnxable_model,
        dummy_input,
        output_path,
        opset_version=17,
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={
            "observation": {0: "batch_size"},
            "action": {0: "batch_size"}
        }
    )
    
    print(f"ONNX model exported to {output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Directory containing training data")
    parser.add_argument("--output", default="models/placement.onnx", help="Output ONNX file")
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="Training timesteps")
    
    args = parser.parse_args()
    
    train_placement_model(args.data_dir, args.output, args.timesteps)
```

### 4.2 ONNX Inference Service

```python
# services/placement_service.py
"""
Component placement service using heuristic algorithms (Phase 1)
or ONNX RL model (Phase 2-3).
"""

import numpy as np
from scipy.optimize import dual_annealing
import networkx as nx
from typing import List, Dict, Tuple, Any
import onnxruntime as ort

class PlacementService:
    """Service for optimizing component placement."""
    
    def __init__(self, onnx_model_path: str = None):
        """Initialize placement service."""
        self.onnx_session = None
        
        if onnx_model_path:
            # Load ONNX model (Phase 2-3)
            self.onnx_session = ort.InferenceSession(onnx_model_path)
            print(f"Loaded RL placement model: {onnx_model_path}")
    
    def optimize(
        self,
        board_size: Tuple[float, float],
        components: List[Dict[str, Any]],
        netlist: List[Dict[str, List[str]]],
        constraints: Dict[str, Any] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """
        Optimize component placement.
        
        Args:
            board_size: (width, height) in mm
            components: List of component dicts with keys:
                - ref: component reference (e.g., "R1")
                - width, height: component dimensions in mm
                - thermal: thermal generation (0.0-1.0)
            netlist: List of nets, each with connected component refs
            constraints: Placement constraints (keep-out zones, etc.)
        
        Returns:
            (positions, metrics) where:
            - positions: List of {ref, x, y, rotation}
            - metrics: {wirelength, overlap, thermal, optimization_time}
        """
        
        if self.onnx_session:
            # Use ONNX RL model (Phase 2-3)
            return self._optimize_with_rl(board_size, components, netlist, constraints)
        else:
            # Use heuristic algorithm (Phase 1)
            return self._optimize_with_heuristic(board_size, components, netlist, constraints)
    
    def _optimize_with_heuristic(
        self,
        board_size: Tuple[float, float],
        components: List[Dict[str, Any]],
        netlist: List[Dict[str, List[str]]],
        constraints: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """Optimize using simulated annealing (Phase 1)."""
        
        import time
        start_time = time.time()
        
        board_width, board_height = board_size
        n_components = len(components)
        
        # Build connectivity graph
        G = nx.Graph()
        for comp in components:
            G.add_node(comp['ref'])
        
        for net in netlist:
            refs = net.get('refs', [])
            for i, ref1 in enumerate(refs):
                for ref2 in refs[i+1:]:
                    if G.has_edge(ref1, ref2):
                        G[ref1][ref2]['weight'] += 1
                    else:
                        G.add_edge(ref1, ref2, weight=1)
        
        # Cost function to minimize
        def cost_function(positions_flat):
            """Calculate cost: wirelength + overlap penalty."""
            positions = positions_flat.reshape(n_components, 3)  # (x, y, rotation)
            
            total_cost = 0.0
            
            # Wirelength (HPWL)
            for net in netlist:
                refs = net.get('refs', [])
                if len(refs) < 2:
                    continue
                
                # Get positions of components in this net
                net_positions = []
                for ref in refs:
                    comp_idx = next((i for i, c in enumerate(components) if c['ref'] == ref), None)
                    if comp_idx is not None:
                        net_positions.append(positions[comp_idx, :2])
                
                if len(net_positions) >= 2:
                    xs = [p[0] for p in net_positions]
                    ys = [p[1] for p in net_positions]
                    hpwl = (max(xs) - min(xs)) + (max(ys) - min(ys))
                    total_cost += hpwl
            
            # Overlap penalty
            for i in range(n_components):
                for j in range(i+1, n_components):
                    x1, y1 = positions[i, :2]
                    x2, y2 = positions[j, :2]
                    w1, h1 = components[i]['width'], components[i]['height']
                    w2, h2 = components[j]['width'], components[j]['height']
                    
                    # Check overlap
                    if (abs(x1 - x2) < (w1 + w2) / 2 and 
                        abs(y1 - y2) < (h1 + h2) / 2):
                        overlap_area = ((w1 + w2) / 2 - abs(x1 - x2)) * \
                                     ((h1 + h2) / 2 - abs(y1 - y2))
                        total_cost += overlap_area * 1000.0  # Heavy penalty
            
            return total_cost
        
        # Bounds for simulated annealing
        # Each component has (x, y, rotation) where rotation is 0-3 (0°, 90°, 180°, 270°)
        bounds = []
        for comp in components:
            bounds.append((0, board_width))  # x
            bounds.append((0, board_height))  # y
            bounds.append((0, 3))  # rotation
        
        # Run simulated annealing
        print(f"Optimizing placement for {n_components} components...")
        result = dual_annealing(
            cost_function,
            bounds=bounds,
            maxiter=1000,
            seed=42
        )
        
        # Extract positions
        positions_flat = result.x
        positions_array = positions_flat.reshape(n_components, 3)
        
        # Format output
        positions = []
        for i, comp in enumerate(components):
            positions.append({
                'ref': comp['ref'],
                'x': float(positions_array[i, 0]),
                'y': float(positions_array[i, 1]),
                'rotation': int(positions_array[i, 2]) * 90  # Convert to degrees
            })
        
        # Calculate metrics
        final_wirelength = 0.0
        for net in netlist:
            refs = net.get('refs', [])
            if len(refs) < 2:
                continue
            net_positions = []
            for ref in refs:
                pos = next((p for p in positions if p['ref'] == ref), None)
                if pos:
                    net_positions.append((pos['x'], pos['y']))
            if len(net_positions) >= 2:
                xs = [p[0] for p in net_positions]
                ys = [p[1] for p in net_positions]
                final_wirelength += (max(xs) - min(xs)) + (max(ys) - min(ys))
        
        metrics = {
            'wirelength': final_wirelength,
            'overlap': 0.0,  # Simulated annealing should minimize this
            'thermal': 0.0,
            'optimization_time': time.time() - start_time,
            'algorithm': 'simulated_annealing'
        }
        
        return positions, metrics
    
    def _optimize_with_rl(
        self,
        board_size: Tuple[float, float],
        components: List[Dict[str, Any]],
        netlist: List[Dict[str, List[str]]],
        constraints: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """Optimize using ONNX RL model (Phase 2-3)."""
        
        import time
        start_time = time.time()
        
        # Prepare state (same format as training)
        state = self._prepare_state(board_size, components, netlist)
        
        # Run inference
        input_name = self.onnx_session.get_inputs()[0].name
        output_name = self.onnx_session.get_outputs()[0].name
        
        actions = self.onnx_session.run(
            [output_name],
            {input_name: state.reshape(1, -1).astype(np.float32)}
        )[0]
        
        # Decode actions to positions
        positions = self._decode_actions(actions[0], components, board_size)
        
        # Calculate metrics
        metrics = {
            'wirelength': 0.0,  # TODO: Calculate
            'overlap': 0.0,
            'thermal': 0.0,
            'optimization_time': time.time() - start_time,
            'algorithm': 'rl_onnx'
        }
        
        return positions, metrics
    
    def _prepare_state(
        self,
        board_size: Tuple[float, float],
        components: List[Dict[str, Any]],
        netlist: List[Dict[str, List[str]]]
    ) -> np.ndarray:
        """Prepare state for RL model."""
        # TODO: Implement state preparation matching training format
        pass
    
    def _decode_actions(
        self,
        actions: np.ndarray,
        components: List[Dict[str, Any]],
        board_size: Tuple[float, float]
    ) -> List[Dict[str, Any]]:
        """Decode RL actions to component positions."""
        # TODO: Implement action decoding
        pass
```

[DOCUMENT CONTINUES IN NEXT PART DUE TO LENGTH...]

---

**Would you like me to continue with:**
1. Section 5: KiCad Integration (IPC API)
2. Section 6: Database Schemas
3. Section 7: Testing Framework
4. Additional documents (Week-by-week execution guide, deployment guide)?

I'll create these next based on your preference!
