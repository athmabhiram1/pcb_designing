"""
Advanced Placement Engine v2.0 - Multi-Objective PCB Component Placement
Uses ONNX Runtime for inference with analytical fallback and full netlist integration.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import tempfile
import time
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any, Callable, Dict, List, Optional, Protocol, Set, Tuple, Union,
    Iterator, Iterable
)

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PlacementConfig:
    """Placement hyperparameters and constraints."""
    # Wirelength optimization
    WIRE_LENGTH_WEIGHT: float = 0.5
    HPWL_WEIGHT: float = 0.3  # Half-perimeter wirelength
    MST_WEIGHT: float = 0.2   # Minimum spanning tree
    
    # Thermal optimization
    THERMAL_WEIGHT: float = 0.3
    THERMAL_DIFFUSION_ITERATIONS: int = 50
    MAX_TEMPERATURE: float = 85.0  # °C
    
    # Density control
    DENSITY_WEIGHT: float = 0.2
    TARGET_DENSITY: float = 0.7
    BIN_SIZE: float = 5.0  # mm
    
    # Constraints
    EDGE_MARGIN: float = 5.0  # mm from board edge
    MIN_COMPONENT_SPACING: float = 0.5  # mm
    
    # RL parameters
    RL_ITERATIONS: int = 100
    REFINEMENT_ITERATIONS: int = 20
    
    # Parallel processing
    MAX_WORKERS: int = 4
    
    # Numerical stability
    EPSILON: float = 1e-6

CONFIG = PlacementConfig()


# ── Data Structures ───────────────────────────────────────────────────────────

class NetType(Enum):
    """Classification of nets for weighted optimization."""
    POWER = auto()      # VCC, VDD - minimize IR drop
    GROUND = auto()     # GND - minimize ground bounce
    SIGNAL = auto()     # General signals
    CLOCK = auto()      # Critical timing - minimize skew
    DIFFERENTIAL = auto()  # Pairs - match lengths
    HIGH_SPEED = auto()    # >100MHz - controlled impedance
    ANALOG = auto()     # Sensitive - isolation


@dataclass
class Pin:
    """Component pin with electrical properties."""
    ref: str
    pin: str
    x_offset: float = 0.0  # Relative to component center
    y_offset: float = 0.0
    net: Optional[str] = None
    electrical_type: str = "unspecified"  # input, output, power, etc.


@dataclass 
class Component:
    """Physical component with placement state."""
    ref: str
    value: str
    footprint: str
    x: float = 0.0
    y: float = 0.0
    rotation: float = 0.0
    layer: str = "top"
    
    # Physical properties
    width: float = 5.0   # mm
    height: float = 5.0  # mm
    pins: List[Pin] = field(default_factory=list)
    power_dissipation: float = 0.0  # Watts
    max_temperature: float = 125.0  # °C
    thermal_resistance: float = 50.0  # °C/W (junction to ambient)
    
    # Constraints
    fixed: bool = False
    keepout_radius: float = 0.0  # mm - exclusion zone
    allowed_layers: Set[str] = field(default_factory=lambda: {"top", "bottom"})
    
    # Grouping
    cluster_id: Optional[str] = None
    
    def get_pin_world_pos(self, pin: Pin) -> Tuple[float, float]:
        """Calculate absolute pin position with rotation."""
        # Rotate pin offset
        rad = math.radians(self.rotation)
        rx = pin.x_offset * math.cos(rad) - pin.y_offset * math.sin(rad)
        ry = pin.x_offset * math.sin(rad) + pin.y_offset * math.cos(rad)
        return self.x + rx, self.y + ry
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Return (min_x, min_y, max_x, max_y)."""
        hw, hh = self.width / 2, self.height / 2
        return (self.x - hw, self.y - hh, self.x + hw, self.y + hh)


@dataclass
class Net:
    """Electrical net with connectivity graph."""
    name: str
    pins: List[Pin] = field(default_factory=list)
    net_type: NetType = NetType.SIGNAL
    weight: float = 1.0  # Optimization weight
    critical_length: Optional[float] = None  # mm - max allowed length
    
    @property
    def components(self) -> Set[str]:
        return {p.ref for p in self.pins}
    
    def get_bounding_box(self, comp_positions: Dict[str, Tuple[float, float]]) -> Tuple[float, float, float, float]:
        """Calculate half-perimeter wirelength bounding box."""
        xs, ys = [], []
        for pin in self.pins:
            if pin.ref in comp_positions:
                x, y = comp_positions[pin.ref]
                # Add pin offset
                xs.append(x + pin.x_offset)
                ys.append(y + pin.y_offset)
        
        if not xs:
            return (0, 0, 0, 0)
        return (min(xs), min(ys), max(xs), max(ys))


@dataclass
class PlacementSolution:
    """Complete placement solution with metrics."""
    positions: Dict[str, Tuple[float, float, float]]  # ref -> (x, y, rotation)
    wirelength: float = 0.0
    thermal_score: float = 0.0
    density_score: float = 0.0
    constraint_violations: int = 0
    convergence_delta: float = float('inf')
    computation_time_ms: float = 0.0
    
    @property
    def total_cost(self) -> float:
        return (
            CONFIG.WIRE_LENGTH_WEIGHT * self.wirelength +
            CONFIG.THERMAL_WEIGHT * self.thermal_score +
            CONFIG.DENSITY_WEIGHT * self.density_score
        )


# ── Connectivity Graph ────────────────────────────────────────────────────────

class ConnectivityGraph:
    """Advanced graph representation of circuit connectivity."""
    
    def __init__(self, nets: List[Net], components: List[Component]):
        self.nets = {n.name: n for n in nets}
        self.components = {c.ref: c for c in components}
        
        # Build adjacency with weights
        self.adjacency: Dict[str, Dict[str, float]] = {c.ref: {} for c in components}
        self._build_graph()
        
        # Precompute clustering
        self.clusters: Dict[str, Set[str]] = {}
        self._compute_clusters()
    
    def _build_graph(self) -> None:
        """Build weighted adjacency from nets."""
        for net in self.nets.values():
            # Weight inversely by fanout (star nets have lower per-edge weight)
            edge_weight = net.weight / max(len(net.components) - 1, 1)
            
            refs = list(net.components)
            for i, ref1 in enumerate(refs):
                for ref2 in refs[i+1:]:
                    if ref1 in self.adjacency and ref2 in self.adjacency:
                        w = edge_weight
                        if net.net_type == NetType.CLOCK:
                            w *= 3.0  # Prioritize clock nets
                        elif net.net_type in (NetType.POWER, NetType.GROUND):
                            w *= 2.0  # Important for IR drop
                        
                        self.adjacency[ref1][ref2] = self.adjacency[ref1].get(ref2, 0) + w
                        self.adjacency[ref2][ref1] = self.adjacency[ref2].get(ref1, 0) + w
    
    def _compute_clusters(self) -> None:
        """Detect natural component clusters using union-find."""
        parent = {ref: ref for ref in self.adjacency}
        
        def find(x: str) -> str:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x: str, y: str) -> None:
            parent[find(x)] = find(y)
        
        # Union components with strong connections
        for ref1, neighbors in self.adjacency.items():
            for ref2, weight in neighbors.items():
                if weight > 1.0:  # Strong connection threshold
                    union(ref1, ref2)
        
        # Build clusters
        clusters: Dict[str, Set[str]] = {}
        for ref in self.adjacency:
            root = find(ref)
            if root not in clusters:
                clusters[root] = set()
            clusters[root].add(ref)
        
        self.clusters = {f"cluster_{i}": members for i, members in enumerate(clusters.values())}
        
        # Assign cluster IDs to components
        for cluster_id, members in self.clusters.items():
            for ref in members:
                if ref in self.components:
                    self.components[ref].cluster_id = cluster_id
    
    def get_neighbors(self, ref: str) -> Dict[str, float]:
        """Get weighted neighbors of a component."""
        return self.adjacency.get(ref, {})
    
    def get_cluster_bounding_box(self, cluster_id: str) -> Optional[Tuple[float, float, float, float]]:
        """Get bounding box of a cluster."""
        if cluster_id not in self.clusters:
            return None
        
        refs = self.clusters[cluster_id]
        xs, ys = [], []
        for ref in refs:
            if ref in self.components:
                c = self.components[ref]
                xs.append(c.x)
                ys.append(c.y)
        
        if not xs:
            return None
        return (min(xs), min(ys), max(xs), max(ys))


# ── Thermal Model ─────────────────────────────────────────────────────────────

class ThermalModel:
    """Finite difference thermal simulation for PCB."""
    
    def __init__(self, width: float, height: float, resolution: float = 2.0):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.nx = int(width / resolution) + 1
        self.ny = int(height / resolution) + 1
        
        # Temperature field (°C)
        self.temperature = np.full((self.ny, self.nx), 25.0)  # Ambient
        self.power_map = np.zeros((self.ny, self.nx))
        self.conductivity = 1.0  # Simplified copper conductivity
        
        # Thermal vias and planes
        self.high_conductivity_zones: List[Tuple[int, int, int, int]] = []  # x1,y1,x2,y2 in grid coords
    
    def add_component(self, comp: Component) -> None:
        """Add component power dissipation to thermal map."""
        # Map to grid
        gx1 = max(0, int((comp.x - comp.width/2) / self.resolution))
        gy1 = max(0, int((comp.y - comp.height/2) / self.resolution))
        gx2 = min(self.nx-1, int((comp.x + comp.width/2) / self.resolution))
        gy2 = min(self.ny-1, int((comp.y + comp.height/2) / self.resolution))
        
        power_density = comp.power_dissipation / ((gx2-gx1+1) * (gy2-gy1+1))
        self.power_map[gy1:gy2+1, gx1:gx2+1] += power_density
    
    def solve_steady_state(self, iterations: int = 100) -> None:
        """Solve thermal diffusion using Gauss-Seidel iteration."""
        alpha = 0.25  # Thermal diffusivity factor
        
        for _ in range(iterations):
            new_temp = self.temperature.copy()
            
            # Interior points
            new_temp[1:-1, 1:-1] = (
                self.temperature[1:-1, 1:-1] * (1 - alpha) +
                alpha * 0.25 * (
                    self.temperature[:-2, 1:-1] +
                    self.temperature[2:, 1:-1] +
                    self.temperature[1:-1, :-2] +
                    self.temperature[1:-1, 2:]
                ) +
                self.power_map[1:-1, 1:-1] * 0.1  # Heat input
            )
            
            # Boundary conditions (convective cooling at edges)
            new_temp[0, :] = new_temp[1, :] * 0.9 + 25.0 * 0.1  # Top
            new_temp[-1, :] = new_temp[-2, :] * 0.9 + 25.0 * 0.1  # Bottom
            new_temp[:, 0] = new_temp[:, 1] * 0.9 + 25.0 * 0.1  # Left
            new_temp[:, -1] = new_temp[:, -2] * 0.9 + 25.0 * 0.1  # Right
            
            self.temperature = new_temp
    
    def get_component_temperature(self, comp: Component) -> float:
        """Get maximum temperature at component location."""
        gx = int(comp.x / self.resolution)
        gy = int(comp.y / self.resolution)
        
        if 0 <= gx < self.nx and 0 <= gy < self.ny:
            return float(self.temperature[gy, gx])
        return 25.0
    
    def get_hotspots(self, threshold: float = 70.0) -> List[Tuple[float, float, float]]:
        """Return list of (x, y, temp) for hotspots above threshold."""
        hotspots = []
        for iy in range(self.ny):
            for ix in range(self.nx):
                if self.temperature[iy, ix] > threshold:
                    x = ix * self.resolution
                    y = iy * self.resolution
                    hotspots.append((x, y, float(self.temperature[iy, ix])))
        return hotspots


# ── Analytical Placement (Quadratic Solver) ───────────────────────────────────

class QuadraticPlacer:
    """Analytical placement using quadratic wirelength minimization."""
    
    def __init__(self, graph: ConnectivityGraph, board_width: float, board_height: float):
        self.graph = graph
        self.width = board_width
        self.height = board_height
        self.n = len(graph.components)
        self.refs = list(graph.adjacency.keys())
        self.ref_to_idx = {r: i for i, r in enumerate(self.refs)}
    
    def solve(self, fixed_positions: Optional[Dict[str, Tuple[float, float]]] = None) -> Dict[str, Tuple[float, float]]:
        """
        Solve using conjugate gradient on quadratic objective.
        
        Minimizes: 0.5 * sum(w_ij * ((xi-xj)² + (yi-yj)²))
        Subject to: fixed component constraints
        """
        # Build connectivity matrix (Laplacian)
        L = np.zeros((self.n, self.n))
        b_x = np.zeros(self.n)
        b_y = np.zeros(self.n)
        
        for i, ref_i in enumerate(self.refs):
            for ref_j, weight in self.graph.get_neighbors(ref_i).items():
                j = self.ref_to_idx[ref_j]
                L[i, i] += weight
                L[i, j] -= weight
        
        # Handle fixed components
        free_vars = []
        for i, ref in enumerate(self.refs):
            if fixed_positions and ref in fixed_positions:
                fx, fy = fixed_positions[ref]
                b_x -= L[i, i] * fx
                b_y -= L[i, i] * fy
                L[i, :] = 0
                L[:, i] = 0
                L[i, i] = 1
                b_x[i] = fx
                b_y[i] = fy
            else:
                free_vars.append(i)
        
        # Solve linear system (sparse would be better for large systems)
        try:
            x_pos = np.linalg.solve(L, b_x)
            y_pos = np.linalg.solve(L, b_y)
        except np.linalg.LinAlgError:
            # Fallback to least squares if singular
            x_pos = np.linalg.lstsq(L, b_x, rcond=None)[0]
            y_pos = np.linalg.lstsq(L, b_y, rcond=None)[0]
        
        # Clip to board boundaries
        positions = {}
        for i, ref in enumerate(self.refs):
            x = max(CONFIG.EDGE_MARGIN, min(self.width - CONFIG.EDGE_MARGIN, x_pos[i]))
            y = max(CONFIG.EDGE_MARGIN, min(self.height - CONFIG.EDGE_MARGIN, y_pos[i]))
            positions[ref] = (x, y)
        
        return positions


# ── RL Engine Interface ───────────────────────────────────────────────────────

class RLEngine(ABC):
    """Abstract base for RL placement engines."""
    
    @abstractmethod
    def predict(self, state: np.ndarray) -> np.ndarray:
        """Predict next placement action."""
        pass
    
    @abstractmethod
    def load(self, path: str) -> bool:
        """Load model from path."""
        pass


class ONNXRLEngine(RLEngine):
    """ONNX Runtime-based RL engine."""
    
    def __init__(self):
        self.session = None
        self.input_name = None
    
    def load(self, path: str) -> bool:
        try:
            import onnxruntime as ort
            
            if not os.path.exists(path):
                logger.warning(f"Model not found: {path}")
                return False
            
            providers = ['CPUExecutionProvider']
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers.insert(0, 'CUDAExecutionProvider')
            
            self.session = ort.InferenceSession(path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            logger.info(f"Loaded RL model from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load RL model: {e}")
            return False
    
    def predict(self, state: np.ndarray) -> np.ndarray:
        if self.session is None:
            raise RuntimeError("Model not loaded")
        
        if state.ndim == 3:
            state = state[np.newaxis, ...]
        
        outputs = self.session.run(None, {self.input_name: state.astype(np.float32)})
        return outputs[0][0]


# ── Main Placement Engine ─────────────────────────────────────────────────────

class PlacementEngine:
    """
    Unified placement engine combining RL, analytical, and rule-based methods.
    """
    
    def __init__(self):
        self.rl_engine: Optional[RLEngine] = None
        self.config = CONFIG
    
    def load_rl_model(self, path: Optional[str] = None) -> bool:
        """Load RL model."""
        model_path = path or os.path.join(
            os.path.dirname(__file__), "..", "models", "placement_model.onnx"
        )
        
        engine = ONNXRLEngine()
        if engine.load(model_path):
            self.rl_engine = engine
            return True
        return False
    
    def optimize(
        self,
        components: List[Component],
        nets: List[Net],
        board_width: float,
        board_height: float,
        method: str = "auto",
        fixed_components: Optional[Set[str]] = None,
    ) -> PlacementSolution:
        """
        Main optimization entry point.
        
        Methods:
        - auto: Choose best available (RL + analytical refinement)
        - rl: Reinforcement learning only
        - analytical: Quadratic wirelength minimization
        - thermal: Thermal-driven placement
        - hybrid: Multi-objective optimization
        """
        t0 = time.time()
        
        # Build connectivity graph
        graph = ConnectivityGraph(nets, components)
        
        # Determine method
        if method == "auto":
            if self.rl_engine:
                method = "hybrid"
            else:
                method = "analytical"
        
        # Execute chosen method
        if method == "hybrid":
            positions = self._hybrid_optimize(graph, components, board_width, board_height)
        elif method == "rl" and self.rl_engine:
            positions = self._rl_optimize(graph, components, board_width, board_height)
        elif method == "analytical":
            positions = self._analytical_optimize(graph, board_width, board_height, fixed_components)
        elif method == "thermal":
            positions = self._thermal_optimize(graph, components, board_width, board_height)
        else:
            positions = self._rule_based_fallback(components, board_width, board_height)
        
        # Build solution
        solution = self._build_solution(positions, components, nets, board_width, board_height)
        solution.computation_time_ms = (time.time() - t0) * 1000
        
        return solution
    
    def _hybrid_optimize(
        self,
        graph: ConnectivityGraph,
        components: List[Component],
        width: float,
        height: float
    ) -> Dict[str, Tuple[float, float, float]]:
        """Combine RL global optimization with analytical refinement."""
        # Step 1: RL for global placement
        rl_positions = self._rl_optimize(graph, components, width, height)
        
        # Step 2: Analytical refinement for wirelength
        fixed = {ref: (pos[0], pos[1]) for ref, pos in rl_positions.items()}
        quad = QuadraticPlacer(graph, width, height)
        refined = quad.solve(fixed)
        
        # Merge: Use RL for rotation, analytical for position
        result = {}
        for ref in rl_positions:
            rx, ry, rrot = rl_positions[ref]
            if ref in refined:
                qx, qy = refined[ref]
                # Weighted combination
                fx = 0.7 * qx + 0.3 * rx
                fy = 0.7 * qy + 0.3 * ry
                result[ref] = (fx, fy, rrot)
            else:
                result[ref] = (rx, ry, rrot)
        
        # Step 3: Wirelength-driven refinement
        result = self._refine_wirelength(result, graph, width, height)
        
        return result
    
    def _rl_optimize(
        self,
        graph: ConnectivityGraph,
        components: List[Component],
        width: float,
        height: float
    ) -> Dict[str, Tuple[float, float, float]]:
        """RL-based sequential placement."""
        if not self.rl_engine:
            raise RuntimeError("RL engine not loaded")
        
        # Sort components by connectivity (most connected first)
        sorted_comps = sorted(
            components,
            key=lambda c: len(graph.get_neighbors(c.ref)),
            reverse=True
        )
        
        # Initialize grid state
        grid_res = 2.0  # mm per cell
        grid_w, grid_h = int(width / grid_res), int(height / grid_res)
        state = np.zeros((4, grid_h, grid_w), dtype=np.float32)
        
        positions = {}
        
        for comp in sorted_comps:
            # Update state with current placements
            self._update_state(state, positions, grid_res, width, height)
            
            # Predict
            action = self.rl_engine.predict(state)
            
            # Denormalize
            x = float(action[0]) * width
            y = float(action[1]) * height
            rot = float(action[2] % 360) if len(action) > 2 else 0.0
            
            # Enforce margins
            x = max(CONFIG.EDGE_MARGIN, min(width - CONFIG.EDGE_MARGIN, x))
            y = max(CONFIG.EDGE_MARGIN, min(height - CONFIG.EDGE_MARGIN, y))
            
            positions[comp.ref] = (x, y, rot)
            
            # Update occupied channel
            gx, gy = int(x / grid_res), int(y / grid_res)
            if 0 <= gx < grid_w and 0 <= gy < grid_h:
                state[0, gy, gx] = 1.0
        
        # Post-process with wirelength refinement
        return self._refine_wirelength(positions, graph, width, height)
    
    def _analytical_optimize(
        self,
        graph: ConnectivityGraph,
        width: float,
        height: float,
        fixed: Optional[Set[str]] = None
    ) -> Dict[str, Tuple[float, float, float]]:
        """Pure analytical placement."""
        fixed_pos = None
        if fixed:
            fixed_pos = {ref: (graph.components[ref].x, graph.components[ref].y) 
                        for ref in fixed if ref in graph.components}
        
        placer = QuadraticPlacer(graph, width, height)
        positions_2d = placer.solve(fixed_pos)
        
        # Add default rotation
        return {ref: (x, y, 0.0) for ref, (x, y) in positions_2d.items()}
    
    def _thermal_optimize(
        self,
        graph: ConnectivityGraph,
        components: List[Component],
        width: float,
        height: float
    ) -> Dict[str, Tuple[float, float, float]]:
        """Thermal-aware placement spreading hot components."""
        # Start with analytical solution
        positions = self._analytical_optimize(graph, width, height)
        
        # Identify hot components
        hot_comps = [c for c in components if c.power_dissipation > 0.5]
        
        if len(hot_comps) < 2:
            return {ref: (pos[0], pos[1], 0.0) for ref, pos in positions.items()}
        
        # Iterative thermal spreading
        thermal = ThermalModel(width, height)
        
        for iteration in range(20):
            # Update thermal model
            thermal.power_map.fill(0)
            for comp in components:
                if comp.ref in positions:
                    c = copy.copy(comp)
                    c.x, c.y = positions[comp.ref][0], positions[comp.ref][1]
                    thermal.add_component(c)
            
            thermal.solve_steady_state(10)
            
            # Repel hot components from each other
            for i, c1 in enumerate(hot_comps):
                if c1.ref not in positions:
                    continue
                x1, y1 = positions[c1.ref][0], positions[c1.ref][1]
                t1 = thermal.get_component_temperature(c1)
                
                for c2 in hot_comps[i+1:]:
                    if c2.ref not in positions:
                        continue
                    x2, y2 = positions[c2.ref][0], positions[c2.ref][1]
                    
                    dist = math.hypot(x1 - x2, y1 - y2)
                    if dist < 20.0 and dist > 0:  # Too close
                        # Repel
                        dx = (x1 - x2) / dist * 2.0
                        dy = (y1 - y2) / dist * 2.0
                        
                        positions[c1.ref] = (
                            max(CONFIG.EDGE_MARGIN, min(width - CONFIG.EDGE_MARGIN, x1 + dx)),
                            max(CONFIG.EDGE_MARGIN, min(height - CONFIG.EDGE_MARGIN, y1 + dy)),
                            positions[c1.ref][2]
                        )
                        positions[c2.ref] = (
                            max(CONFIG.EDGE_MARGIN, min(width - CONFIG.EDGE_MARGIN, x2 - dx)),
                            max(CONFIG.EDGE_MARGIN, min(height - CONFIG.EDGE_MARGIN, y2 - dy)),
                            positions[c2.ref][2]
                        )
        
        return positions
    
    def _refine_wirelength(
        self,
        positions: Dict[str, Tuple[float, float, float]],
        graph: ConnectivityGraph,
        width: float,
        height: float
    ) -> Dict[str, Tuple[float, float, float]]:
        """Force-directed refinement to minimize wirelength."""
        pos = {ref: [x, y, rot] for ref, (x, y, rot) in positions.items()}
        
        for _ in range(CONFIG.REFINEMENT_ITERATIONS):
            forces: Dict[str, List[float]] = {ref: [0.0, 0.0] for ref in pos}
            
            # Attractive forces from connections
            for ref, neighbors in graph.adjacency.items():
                if ref not in pos:
                    continue
                x1, y1 = pos[ref][0], pos[ref][1]
                
                for nref, weight in neighbors.items():
                    if nref not in pos:
                        continue
                    x2, y2 = pos[nref][0], pos[nref][1]
                    
                    dx, dy = x2 - x1, y2 - y1
                    dist = math.hypot(dx, dy)
                    if dist > 0:
                        # Spring force toward ideal distance (10mm)
                        ideal = 10.0
                        force = weight * (dist - ideal) / dist
                        fx = dx * force * 0.1
                        fy = dy * force * 0.1
                        
                        forces[ref][0] += fx
                        forces[ref][1] += fy
            
            # Repulsive forces (density control)
            refs = list(pos.keys())
            for i, r1 in enumerate(refs):
                for r2 in refs[i+1:]:
                    x1, y1 = pos[r1][0], pos[r1][1]
                    x2, y2 = pos[r2][0], pos[r2][1]
                    dx, dy = x1 - x2, y1 - y2
                    dist = math.hypot(dx, dy)
                    
                    if dist < 15.0 and dist > 0:
                        force = (15.0 - dist) / dist * 0.5
                        fx, fy = dx * force, dy * force
                        
                        forces[r1][0] += fx
                        forces[r1][1] += fy
                        forces[r2][0] -= fx
                        forces[r2][1] -= fy
            
            # Apply forces
            for ref in pos:
                pos[ref][0] = max(CONFIG.EDGE_MARGIN, 
                                 min(width - CONFIG.EDGE_MARGIN, 
                                     pos[ref][0] + forces[ref][0]))
                pos[ref][1] = max(CONFIG.EDGE_MARGIN,
                                 min(height - CONFIG.EDGE_MARGIN,
                                     pos[ref][1] + forces[ref][1]))
        
        return {ref: (p[0], p[1], p[2]) for ref, p in pos.items()}
    
    def _rule_based_fallback(
        self,
        components: List[Component],
        width: float,
        height: float
    ) -> Dict[str, Tuple[float, float, float]]:
        """Intelligent rule-based placement when ML is unavailable."""
        # Group by type
        ics = [c for c in components if c.ref.startswith('U')]
        passives = [c for c in components if c.ref[0] in 'RLC']
        connectors = [c for c in components if c.ref[0] in 'JP']
        others = [c for c in components if c not in ics + passives + connectors]
        
        positions = {}
        
        # ICs in center with spacing
        grid = 15.0
        cols = max(1, int(math.sqrt(len(ics))))
        for i, ic in enumerate(ics):
            row, col = divmod(i, cols)
            x = width/2 + (col - cols/2) * grid
            y = height/2 + (row - len(ics)/cols/2) * grid
            positions[ic.ref] = (x, y, 0.0)
        
        # Passives in rows around ICs
        passive_cols = 8
        for i, p in enumerate(passives):
            row, col = divmod(i, passive_cols)
            # Alternate sides
            side = 1 if row % 2 == 0 else -1
            x = width/2 + side * (30 + col * 5)
            y = 10 + row * 5
            positions[p.ref] = (x, y, 0.0 if side == 1 else 180.0)
        
        # Connectors on edges
        for i, conn in enumerate(connectors):
            if i % 2 == 0:
                positions[conn.ref] = (5, 10 + i * 10, 270.0)
            else:
                positions[conn.ref] = (width - 5, 10 + i * 10, 90.0)
        
        # Others fill remaining space
        for i, o in enumerate(others):
            x = 20 + (i % 5) * 10
            y = height - 20 - (i // 5) * 10
            positions[o.ref] = (x, y, 0.0)
        
        return positions
    
    def _update_state(
        self,
        state: np.ndarray,
        positions: Dict[str, Tuple[float, float, float]],
        grid_res: float,
        width: float,
        height: float
    ) -> None:
        """Update RL state grid with current placements."""
        state.fill(0.0)
        for ref, (x, y, rot) in positions.items():
            gx = int(x / grid_res)
            gy = int(y / grid_res)
            if 0 <= gx < state.shape[2] and 0 <= gy < state.shape[1]:
                state[0, gy, gx] = 1.0  # Occupied
                state[1, gy, gx] = rot / 360.0  # Normalized rotation
    
    def _build_solution(
        self,
        positions: Dict[str, Tuple[float, float, float]],
        components: List[Component],
        nets: List[Net],
        width: float,
        height: float
    ) -> PlacementSolution:
        """Calculate placement metrics."""
        # Update component positions
        for comp in components:
            if comp.ref in positions:
                x, y, rot = positions[comp.ref]
                comp.x, comp.y, comp.rotation = x, y, rot
        
        # Calculate wirelength (HPWL)
        total_hpwl = 0.0
        for net in nets:
            bb = net.get_bounding_box({c.ref: (c.x, c.y) for c in components})
            if bb[2] > bb[0] and bb[3] > bb[1]:
                hpwl = (bb[2] - bb[0]) + (bb[3] - bb[1])
                total_hpwl += hpwl * net.weight
        
        # Thermal score
        thermal = ThermalModel(width, height)
        for comp in components:
            thermal.add_component(comp)
        thermal.solve_steady_state(20)
        max_temp = np.max(thermal.temperature)
        thermal_score = max(0, (max_temp - 25) / 60 * 100)  # Normalize to 0-100
        
        # Density score
        density_map = np.zeros((int(height/CONFIG.BIN_SIZE), int(width/CONFIG.BIN_SIZE)))
        for comp in components:
            bx = int(comp.x / CONFIG.BIN_SIZE)
            by = int(comp.y / CONFIG.BIN_SIZE)
            if 0 <= bx < density_map.shape[1] and 0 <= by < density_map.shape[0]:
                density_map[by, bx] += comp.area / (CONFIG.BIN_SIZE ** 2)
        
        max_density = np.max(density_map) if density_map.size > 0 else 0
        density_score = max_density * 100  # Percentage
        
        return PlacementSolution(
            positions=positions,
            wirelength=total_hpwl,
            thermal_score=thermal_score,
            density_score=density_score,
            constraint_violations=0  # Would check actual constraints
        )


# ── Public API ────────────────────────────────────────────────────────────────

def load_placement_model(model_path: Optional[str] = None) -> Optional[PlacementEngine]:
    """Factory function to create and load placement engine."""
    engine = PlacementEngine()
    engine.load_rl_model(model_path)
    return engine


def optimize_with_rl(
    model: PlacementEngine,
    board_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Legacy-compatible interface for RL optimization.
    
    Args:
        model: Loaded PlacementEngine
        board_data: Dict with components, nets, board dimensions
    
    Returns:
        Dict with success, positions, improvement, metrics
    """
    try:
        # Parse input
        components = [
            Component(
                ref=c["ref"],
                value=c.get("value", ""),
                footprint=c.get("footprint", ""),
                x=c.get("x", 0.0),
                y=c.get("y", 0.0),
                rotation=c.get("rotation", 0.0),
                layer=c.get("layer", "top"),
                width=c.get("width", 5.0),
                height=c.get("height", 5.0),
                power_dissipation=c.get("power_dissipation", 0.0),
                fixed=c.get("fixed", False)
            )
            for c in board_data.get("components", [])
        ]
        
        nets = []
        for n in board_data.get("connections", []):
            pins = [
                Pin(ref=p["ref"], pin=p["pin"])
                for p in n.get("pins", [])
            ]
            net_type = NetType.SIGNAL
            if any(kw in n["net"].upper() for kw in ["VCC", "VDD", "PWR"]):
                net_type = NetType.POWER
            elif "GND" in n["net"].upper():
                net_type = NetType.GROUND
            elif "CLK" in n["net"].upper():
                net_type = NetType.CLOCK
            
            nets.append(Net(
                name=n["net"],
                pins=pins,
                net_type=net_type
            ))
        
        width = board_data.get("board_width", 100.0)
        height = board_data.get("board_height", 80.0)
        
        # Run optimization
        solution = model.optimize(
            components=components,
            nets=nets,
            board_width=width,
            board_height=height,
            method="hybrid"
        )
        
        # Format output
        positions = {
            ref: {"x": x, "y": y, "rotation": rot}
            for ref, (x, y, rot) in solution.positions.items()
        }
        
        return {
            "success": True,
            "positions": positions,
            "improvement": (
                f"Hybrid optimization: WL={solution.wirelength:.1f}mm, "
                f"Thermal={solution.thermal_score:.1f}, "
                f"Density={solution.density_score:.1f}%"
            ),
            "metrics": {
                "wirelength": solution.wirelength,
                "thermal_score": solution.thermal_score,
                "density_score": solution.density_score,
                "total_cost": solution.total_cost,
                "time_ms": solution.computation_time_ms
            }
        }
        
    except Exception as e:
        logger.exception("Placement optimization failed")
        return {
            "success": False,
            "error": str(e)
        }


def optimize_with_rules(board_data: Dict[str, Any]) -> Dict[str, Any]:
    """Rule-based fallback for when ML is unavailable."""
    engine = PlacementEngine()
    
    components = [
        Component(
            ref=c["ref"],
            value=c.get("value", ""),
            footprint=c.get("footprint", "")
        )
        for c in board_data.get("components", [])
    ]
    
    width = board_data.get("board_width", 100.0)
    height = board_data.get("board_height", 80.0)
    
    positions = engine._rule_based_fallback(components, width, height)
    
    return {
        "success": True,
        "positions": {
            ref: {"x": x, "y": y, "rotation": rot}
            for ref, (x, y, rot) in positions.items()
        },
        "improvement": "Rule-based optimization (no ML model)",
        "metrics": {}
    }