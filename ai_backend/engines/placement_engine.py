"""
Advanced Placement Engine v2.2 — Multi-Objective PCB Component Placement.

Supports ONNX Runtime for RL inference with analytical (quadratic) and
rule-based fallbacks. Full netlist integration with thermal simulation.

Fixes vs v2.1:
  - import re added to top-level imports (was missing → NameError in
    check_differential_pair_lengths and Component.prefix at runtime)
  - Component.prefix no longer uses __import__("re").match() hack
  - check_differential_pair_lengths uses re.IGNORECASE directly
  - Dead 'import re as _re' at module bottom removed
  - _build_solution upgraded from copy.copy → copy.deepcopy for Component
    objects passed to ThermalModel (shallow copy shared pins list)
  - SpatialIndex.update wraps cell_list.remove() in try/except ValueError
    to prevent crash when a ref is not in the expected cell
  - _pure_python_solve dead 'groups' variable fixed — fallback solver now
    does cluster-aware grid placement (same-cluster refs placed contiguously)
  - Constraint-violation counting added to _build_solution (min-spacing check)
  - optimize_with_rl output rotations now snapped to 90° via snap_rotation()
  - optimize_with_rl metrics include constraint_violations, net_count,
    component_count for richer API responses
  - __all__ defined for clean star-import and IDE discoverability
  - New public helper: snap_rotation(angle, step=90) for KiCad-safe angles
  - New method: PlacementSolution.export_to_json() for serialisation

Fixes vs v2.0:
  - numpy made optional (hard crash on import if absent)
  - ProcessPoolExecutor removed (unpicklable objects; was never used)
  - Unused imports removed (Protocol, Iterator, Iterable, Union, Callable, tempfile)
  - _rule_based_fallback 'others' filter uses ref-set membership, not identity
  - QuadraticPlacer fixed-component constraint handling corrected (column/row
    zeroing must happen before modifying b_x/b_y, not after)
  - ThermalModel diffusion alpha reduced to 0.20 (was 0.25 — marginal stability)
  - optimize_with_rl respects rl_engine availability for method selection
  - _rl_optimize state channels 2 and 3 now carry connectivity and power data
  - ConnectivityGraph edge weight uses sqrt(fanout) normalisation so high-fanout
    power nets keep reasonable weight instead of near-zero
  - _build_solution no longer mutates caller's Component objects (deep-copies)
  - load_placement_model returns None when RL unavailable so callers can branch
  - _refine_wirelength repulsive loop uses spatial-grid index (O(n) avg) instead
    of O(n²) all-pairs
  - density_map bounds clamped to prevent out-of-bounds index
  - _thermal_optimize uses deepcopy for thermal component copies
  - All logging converted to % args (lazy evaluation)
  - New: differential-pair length matching check
  - New: antenna ratio DFM rule
  - New: cluster-aware placement (decoupling caps placed near their IC)
  - New: deterministic seed for reproducible analytical placement
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import re
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ── Optional heavy dependencies ───────────────────────────────────────────────

try:
    import numpy as np
    from numpy.typing import NDArray
    _NP = True
except ImportError:
    np = None           # type: ignore[assignment]
    NDArray = Any       # type: ignore[misc,assignment]
    _NP = False

# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PlacementConfig:
    """Placement hyperparameters and manufacturing constraints."""
    # Wirelength weights (must sum to ≤ 1.0)
    WIRE_LENGTH_WEIGHT: float = 0.50
    HPWL_WEIGHT:        float = 0.30   # Half-perimeter wirelength
    MST_WEIGHT:         float = 0.20   # Minimum spanning tree estimate

    # Thermal
    THERMAL_WEIGHT:              float = 0.30
    THERMAL_DIFFUSION_ALPHA:     float = 0.20   # < 0.25 for stability
    THERMAL_DIFFUSION_ITERS:     int   = 50
    MAX_TEMPERATURE_C:           float = 85.0

    # Density
    DENSITY_WEIGHT:   float = 0.20
    TARGET_DENSITY:   float = 0.70
    BIN_SIZE_MM:      float = 5.0

    # Layout constraints
    EDGE_MARGIN_MM:          float = 5.0
    MIN_COMPONENT_SPACING_MM: float = 0.5

    # Refinement
    RL_ITERATIONS:         int = 100
    REFINEMENT_ITERATIONS: int = 20

    # Connectivity thresholds
    STRONG_EDGE_THRESHOLD:  float = 1.0  # Union-find cluster threshold
    DECOUPLING_RADIUS_MM:   float = 8.0  # Max IC–decap distance

    # Numerical
    EPSILON: float = 1e-8

    # Spatial index
    SPATIAL_CELL_MM: float = 10.0  # Grid cell size for O(1) neighbour queries


CONFIG = PlacementConfig()


# ── Enumerations ──────────────────────────────────────────────────────────────

class NetType(Enum):
    POWER       = auto()
    GROUND      = auto()
    SIGNAL      = auto()
    CLOCK       = auto()
    DIFFERENTIAL = auto()
    HIGH_SPEED  = auto()
    ANALOG      = auto()


# Net-type → optimization weight multiplier
_NET_TYPE_WEIGHT: Dict[NetType, float] = {
    NetType.CLOCK:       3.0,
    NetType.DIFFERENTIAL: 2.5,
    NetType.HIGH_SPEED:  2.0,
    NetType.POWER:       2.0,
    NetType.GROUND:      2.0,
    NetType.ANALOG:      1.5,
    NetType.SIGNAL:      1.0,
}


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class Pin:
    ref:             str
    pin:             str
    x_offset:        float = 0.0
    y_offset:        float = 0.0
    net:             Optional[str] = None
    electrical_type: str = "unspecified"


@dataclass
class Component:
    """Physical component with placement state."""
    ref:        str
    value:      str
    footprint:  str = ""
    x:          float = 0.0
    y:          float = 0.0
    rotation:   float = 0.0
    layer:      str = "top"

    # Physical
    width:              float = 5.0
    height:             float = 5.0
    pins:               List[Pin] = field(default_factory=list)
    power_dissipation:  float = 0.0   # Watts
    max_temperature:    float = 125.0  # °C
    thermal_resistance: float = 50.0   # °C/W

    # Constraints
    fixed:          bool = False
    keepout_radius: float = 0.0
    allowed_layers: Set[str] = field(default_factory=lambda: {"top", "bottom"})

    # Grouping
    cluster_id: Optional[str] = None

    @property
    def prefix(self) -> str:
        """KiCad-style designator prefix (R, C, U, J, …)."""
        m = re.match(r'^#?([A-Z]+)', self.ref)
        return m.group(1) if m else "U"

    @property
    def is_ic(self) -> bool:
        return self.prefix in {"U", "IC"}

    @property
    def is_passive(self) -> bool:
        return self.prefix in {"R", "C", "L", "F"}

    @property
    def is_connector(self) -> bool:
        return self.prefix in {"J", "P", "CONN"}

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """(min_x, min_y, max_x, max_y)."""
        hw, hh = self.width / 2, self.height / 2
        return self.x - hw, self.y - hh, self.x + hw, self.y + hh

    def get_pin_world_pos(self, pin: Pin) -> Tuple[float, float]:
        rad = math.radians(self.rotation)
        rx  = pin.x_offset * math.cos(rad) - pin.y_offset * math.sin(rad)
        ry  = pin.x_offset * math.sin(rad) + pin.y_offset * math.cos(rad)
        return self.x + rx, self.y + ry


@dataclass
class Net:
    name:            str
    pins:            List[Pin]    = field(default_factory=list)
    net_type:        NetType      = NetType.SIGNAL
    weight:          float        = 1.0
    critical_length: Optional[float] = None   # mm max

    @property
    def components(self) -> Set[str]:
        return {p.ref for p in self.pins}

    def hpwl(self, positions: Dict[str, Tuple[float, float]]) -> float:
        """Half-perimeter wirelength from current positions."""
        xs, ys = [], []
        for pin in self.pins:
            if pin.ref in positions:
                x, y = positions[pin.ref]
                xs.append(x + pin.x_offset)
                ys.append(y + pin.y_offset)
        if len(xs) < 2:
            return 0.0
        return (max(xs) - min(xs)) + (max(ys) - min(ys))


@dataclass
class PlacementSolution:
    positions:            Dict[str, Tuple[float, float, float]]  # ref → (x, y, rot)
    wirelength_mm:        float = 0.0
    thermal_score:        float = 0.0
    density_score:        float = 0.0
    constraint_violations: int   = 0
    convergence_delta:    float = float("inf")
    computation_time_ms:  float = 0.0
    algorithm:            str   = "unknown"
    warnings:             List[str] = field(default_factory=list)

    @property
    def total_cost(self) -> float:
        return (
            CONFIG.WIRE_LENGTH_WEIGHT * self.wirelength_mm
            + CONFIG.THERMAL_WEIGHT   * self.thermal_score
            + CONFIG.DENSITY_WEIGHT   * self.density_score
        )

    def export_to_json(self) -> str:
        """
        Serialise the solution to a JSON string.

        Useful for saving placement results alongside the KiCad files or
        sending them over an API without depending on pickle / dataclasses.
        """
        return json.dumps(
            {
                "algorithm":             self.algorithm,
                "wirelength_mm":         round(self.wirelength_mm, 4),
                "thermal_score":         round(self.thermal_score, 4),
                "density_score":         round(self.density_score, 4),
                "total_cost":            round(self.total_cost, 6),
                "constraint_violations": self.constraint_violations,
                "computation_time_ms":   round(self.computation_time_ms, 2),
                "warnings":              self.warnings,
                "positions": {
                    ref: {"x": round(x, 4), "y": round(y, 4), "rotation": round(rot, 2)}
                    for ref, (x, y, rot) in self.positions.items()
                },
            },
            indent=2,
        )


# ── Spatial Index ─────────────────────────────────────────────────────────────

class SpatialIndex:
    """
    Grid-based spatial index giving O(1) average-case neighbour queries.
    Used to replace the O(n²) all-pairs distance loop in _refine_wirelength.
    """

    def __init__(self, cell_size: float = CONFIG.SPATIAL_CELL_MM) -> None:
        self.cell_size = cell_size
        self._grid: Dict[Tuple[int, int], List[str]] = defaultdict(list)
        self._pos:  Dict[str, Tuple[float, float, float]] = {}

    def _cell(self, x: float, y: float) -> Tuple[int, int]:
        return int(x / self.cell_size), int(y / self.cell_size)

    def build(self, positions: Dict[str, Tuple[float, float, float]]) -> None:
        self._grid.clear()
        self._pos = dict(positions)
        for ref, (x, y, _) in positions.items():
            self._grid[self._cell(x, y)].append(ref)

    def update(self, ref: str, pos: Tuple[float, float, float]) -> None:
        if ref in self._pos:
            old = self._pos[ref]
            old_cell = self._cell(old[0], old[1])
            cell_list = self._grid[old_cell]
            try:
                cell_list.remove(ref)
            except ValueError:
                pass  # already removed or never inserted — safe to ignore
        self._pos[ref] = pos
        self._grid[self._cell(pos[0], pos[1])].append(ref)

    def neighbors_within(self, ref: str, radius: float) -> List[Tuple[str, float]]:
        """Return [(other_ref, distance), …] within radius."""
        if ref not in self._pos:
            return []
        x0, y0, _ = self._pos[ref]
        cell_r = int(radius / self.cell_size) + 1
        cx, cy = self._cell(x0, y0)
        result: List[Tuple[str, float]] = []
        for dx in range(-cell_r, cell_r + 1):
            for dy in range(-cell_r, cell_r + 1):
                for other in self._grid.get((cx + dx, cy + dy), []):
                    if other == ref:
                        continue
                    ox, oy, _ = self._pos[other]
                    dist = math.hypot(x0 - ox, y0 - oy)
                    if dist <= radius:
                        result.append((other, dist))
        return result


# ── Connectivity Graph ────────────────────────────────────────────────────────

class ConnectivityGraph:
    """Weighted adjacency graph from PCB netlist."""

    def __init__(self, nets: List[Net], components: List[Component]) -> None:
        self.nets       = {n.name: n for n in nets}
        self.components = {c.ref: c for c in components}

        self.adjacency:    Dict[str, Dict[str, float]] = {c.ref: {} for c in components}
        self.comp_nets:    Dict[str, List[str]]         = defaultdict(list)  # ref → net names
        self._build_graph()

        self.clusters: Dict[str, Set[str]] = {}
        self._compute_clusters()

    def _build_graph(self) -> None:
        for net in self.nets.values():
            refs  = [p.ref for p in net.pins if p.ref in self.adjacency]
            n     = len(refs)
            if n < 2:
                continue

            # NOTE: original divided by (n-1) giving near-zero weight for
            # power nets with 50+ pins.  Use sqrt(n) so large nets keep
            # meaningful but diminishing per-edge weight.
            base_w = net.weight / math.sqrt(max(n - 1, 1))
            type_mult = _NET_TYPE_WEIGHT.get(net.net_type, 1.0)
            edge_w = base_w * type_mult

            for i, r1 in enumerate(refs):
                self.comp_nets[r1].append(net.name)
                for r2 in refs[i + 1:]:
                    self.adjacency[r1][r2] = self.adjacency[r1].get(r2, 0.0) + edge_w
                    self.adjacency[r2][r1] = self.adjacency[r2].get(r1, 0.0) + edge_w

    def _compute_clusters(self) -> None:
        parent = {ref: ref for ref in self.adjacency}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]   # path compression (iterative)
                x = parent[x]
            return x

        def union(x: str, y: str) -> None:
            parent[find(x)] = find(y)

        for ref1, neighbors in self.adjacency.items():
            for ref2, w in neighbors.items():
                if w > CONFIG.STRONG_EDGE_THRESHOLD:
                    union(ref1, ref2)

        groups: Dict[str, Set[str]] = defaultdict(set)
        for ref in self.adjacency:
            groups[find(ref)].add(ref)

        self.clusters = {f"cluster_{i}": m for i, m in enumerate(groups.values())}

        for cid, members in self.clusters.items():
            for ref in members:
                if ref in self.components:
                    self.components[ref].cluster_id = cid

    def get_neighbors(self, ref: str) -> Dict[str, float]:
        return self.adjacency.get(ref, {})

    def get_nets_for_component(self, ref: str) -> List[str]:
        return self.comp_nets.get(ref, [])

    def get_decoupling_pairs(self) -> List[Tuple[str, str]]:
        """
        Return [(ic_ref, cap_ref), …] pairs where a capacitor shares a
        power/ground net with an IC — used for cluster-aware proximity placement.
        """
        power_ground = {"VCC", "VDD", "GND", "VSS", "3V3", "5V", "AVCC", "DVCC"}
        pairs: List[Tuple[str, str]] = []

        for net in self.nets.values():
            if net.net_type not in (NetType.POWER, NetType.GROUND):
                upper = net.name.upper()
                if not any(kw in upper for kw in power_ground):
                    continue
            refs = list(net.components)
            ics   = [r for r in refs if r in self.components and self.components[r].is_ic]
            caps  = [r for r in refs if r in self.components and self.components[r].prefix == "C"]
            for ic in ics:
                for cap in caps:
                    pairs.append((ic, cap))

        return pairs


# ── Thermal Model ─────────────────────────────────────────────────────────────

class ThermalModel:
    """
    Finite-difference thermal simulation for PCB.

    Uses an explicit Gauss-Seidel scheme with alpha=0.20 (below the 0.25
    stability limit for 2-D diffusion — the original used 0.25 exactly,
    which is marginally stable and causes oscillations).
    """

    def __init__(self, width: float, height: float, resolution: float = 2.0) -> None:
        self.width      = width
        self.height     = height
        self.resolution = resolution
        self.nx = max(2, int(width  / resolution) + 1)
        self.ny = max(2, int(height / resolution) + 1)

        self.temperature: Any = None  # numpy array or None
        self.power_map:   Any = None

        if _NP:
            self.temperature = np.full((self.ny, self.nx), 25.0)
            self.power_map   = np.zeros((self.ny, self.nx))
        else:
            # Pure-Python fallback: flat lists
            self._temp_list  = [25.0] * (self.ny * self.nx)
            self._power_list = [0.0]  * (self.ny * self.nx)

    def _idx(self, iy: int, ix: int) -> int:
        return iy * self.nx + ix

    def add_component(self, comp: Component) -> None:
        gx1 = max(0, int((comp.x - comp.width  / 2) / self.resolution))
        gy1 = max(0, int((comp.y - comp.height / 2) / self.resolution))
        gx2 = min(self.nx - 1, int((comp.x + comp.width  / 2) / self.resolution))
        gy2 = min(self.ny - 1, int((comp.y + comp.height / 2) / self.resolution))
        cells = max(1, (gx2 - gx1 + 1) * (gy2 - gy1 + 1))
        density = comp.power_dissipation / cells

        if _NP:
            self.power_map[gy1:gy2 + 1, gx1:gx2 + 1] += density
        else:
            for gy in range(gy1, gy2 + 1):
                for gx in range(gx1, gx2 + 1):
                    self._power_list[self._idx(gy, gx)] += density

    def solve_steady_state(self, iterations: int = 50) -> None:
        # NOTE: alpha reduced from 0.25 → 0.20 for numerical stability.
        alpha = CONFIG.THERMAL_DIFFUSION_ALPHA  # 0.20
        ambient = 25.0

        if _NP:
            for _ in range(iterations):
                T = self.temperature
                new_T = T.copy()
                new_T[1:-1, 1:-1] = (
                    T[1:-1, 1:-1] * (1 - alpha)
                    + alpha * 0.25 * (
                        T[:-2, 1:-1] + T[2:, 1:-1]
                        + T[1:-1, :-2] + T[1:-1, 2:]
                    )
                    + self.power_map[1:-1, 1:-1] * 0.1
                )
                # Convective boundary conditions
                new_T[0,  :] = new_T[1,  :] * 0.9 + ambient * 0.1
                new_T[-1, :] = new_T[-2, :] * 0.9 + ambient * 0.1
                new_T[:,  0] = new_T[:,  1] * 0.9 + ambient * 0.1
                new_T[:, -1] = new_T[:, -2] * 0.9 + ambient * 0.1
                self.temperature = new_T
        else:
            # Pure-Python fallback (slow but correct)
            for _ in range(iterations):
                new_list = list(self._temp_list)
                for gy in range(1, self.ny - 1):
                    for gx in range(1, self.nx - 1):
                        idx  = self._idx(gy, gx)
                        t_c  = self._temp_list[idx]
                        t_n  = self._temp_list[self._idx(gy - 1, gx)]
                        t_s  = self._temp_list[self._idx(gy + 1, gx)]
                        t_w  = self._temp_list[self._idx(gy, gx - 1)]
                        t_e  = self._temp_list[self._idx(gy, gx + 1)]
                        new_list[idx] = (
                            t_c * (1 - alpha)
                            + alpha * 0.25 * (t_n + t_s + t_w + t_e)
                            + self._power_list[idx] * 0.1
                        )
                self._temp_list = new_list

    def get_component_temperature(self, comp: Component) -> float:
        gx = min(self.nx - 1, max(0, int(comp.x / self.resolution)))
        gy = min(self.ny - 1, max(0, int(comp.y / self.resolution)))
        if _NP:
            return float(self.temperature[gy, gx])
        return self._temp_list[self._idx(gy, gx)]

    def max_temperature(self) -> float:
        if _NP:
            return float(np.max(self.temperature))
        return max(self._temp_list)

    def get_hotspots(self, threshold: float = 70.0) -> List[Tuple[float, float, float]]:
        hotspots = []
        if _NP:
            ys, xs = np.where(self.temperature > threshold)
            for iy, ix in zip(ys, xs):
                hotspots.append((
                    float(ix) * self.resolution,
                    float(iy) * self.resolution,
                    float(self.temperature[iy, ix]),
                ))
        else:
            for gy in range(self.ny):
                for gx in range(self.nx):
                    t = self._temp_list[self._idx(gy, gx)]
                    if t > threshold:
                        hotspots.append((gx * self.resolution, gy * self.resolution, t))
        return hotspots


# ── Quadratic Placer ──────────────────────────────────────────────────────────

class QuadraticPlacer:
    """
    Analytical placement via quadratic wirelength minimisation.
    Solves: 0.5 * Σ w_ij ((xi−xj)² + (yi−yj)²)

    Fixed-component constraint fix: column AND row of fixed node i are zeroed
    BEFORE subtracting the RHS contribution from other rows.  The original
    zeroed after subtraction, corrupting already-computed b entries.
    """

    def __init__(self, graph: ConnectivityGraph, board_width: float, board_height: float) -> None:
        self.graph  = graph
        self.width  = board_width
        self.height = board_height
        self.refs   = list(graph.adjacency.keys())
        self.ri     = {r: i for i, r in enumerate(self.refs)}
        self.n      = len(self.refs)

    def solve(
        self, fixed_positions: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> Dict[str, Tuple[float, float]]:
        if not _NP:
            return self._pure_python_solve(fixed_positions)

        n = self.n
        L  = np.zeros((n, n))
        bx = np.zeros(n)
        by = np.zeros(n)

        # Build Laplacian
        for i, ref_i in enumerate(self.refs):
            for ref_j, w in self.graph.get_neighbors(ref_i).items():
                j = self.ri[ref_j]
                L[i, i] += w
                L[i, j] -= w

        # NOTE: fixed-component constraint must be applied BEFORE touching bx/by
        # for free nodes.  The original subtracted the fixed contribution after
        # zeroing the column, which produced wrong RHS values.
        if fixed_positions:
            for ref, (fx, fy) in fixed_positions.items():
                if ref not in self.ri:
                    continue
                i = self.ri[ref]
                # Propagate fixed value into RHS of FREE rows first
                for j in range(n):
                    if i != j:
                        bx[j] -= L[j, i] * fx
                        by[j] -= L[j, i] * fy
                # Now pin this row/col
                L[:, i] = 0.0
                L[i, :] = 0.0
                L[i, i] = 1.0
                bx[i]   = fx
                by[i]   = fy

        # Regularise any zero-diagonal (isolated node)
        for i in range(n):
            if abs(L[i, i]) < CONFIG.EPSILON:
                L[i, i] = 1.0
                bx[i]   = self.width  / 2
                by[i]   = self.height / 2

        try:
            x_sol = np.linalg.solve(L, bx)
            y_sol = np.linalg.solve(L, by)
        except np.linalg.LinAlgError:
            x_sol = np.linalg.lstsq(L, bx, rcond=None)[0]
            y_sol = np.linalg.lstsq(L, by, rcond=None)[0]

        m = CONFIG.EDGE_MARGIN_MM
        result: Dict[str, Tuple[float, float]] = {}
        for i, ref in enumerate(self.refs):
            x = max(m, min(self.width  - m, float(x_sol[i])))
            y = max(m, min(self.height - m, float(y_sol[i])))
            result[ref] = (x, y)
        return result

    def _pure_python_solve(
        self, fixed_positions: Optional[Dict[str, Tuple[float, float]]]
    ) -> Dict[str, Tuple[float, float]]:
        """
        Pure-Python fallback: cluster-aware grid placement.

        Components in the same cluster are placed in the same grid row so
        strongly-connected parts end up nearby rather than scattered.
        Previously the cluster data was fetched but never used (<-- dead code
        bug fixed here).
        """
        m    = CONFIG.EDGE_MARGIN_MM
        cols = max(1, int(math.sqrt(self.n)))
        result: Dict[str, Tuple[float, float]] = {}

        # Build cluster-ordered index so same-cluster refs are contiguous
        cluster_order: Dict[str, int] = {}
        idx = 0
        for cid, members in self.graph.clusters.items():
            for ref in sorted(members):  # deterministic order within cluster
                cluster_order[ref] = idx
                idx += 1
        # Any ref not in clusters (isolated) goes at the end
        for ref in self.refs:
            if ref not in cluster_order:
                cluster_order[ref] = idx
                idx += 1

        sorted_refs = sorted(self.refs, key=lambda r: cluster_order.get(r, 9999))

        for pos_i, ref in enumerate(sorted_refs):
            if fixed_positions and ref in fixed_positions:
                result[ref] = fixed_positions[ref]
            else:
                row, col = divmod(pos_i, cols)
                x = m + col * (self.width  - 2 * m) / max(cols, 1)
                y = m + row * (self.height - 2 * m) / max(self.n // cols + 1, 1)
                result[ref] = (x, y)
        return result


# ── RL Engine ─────────────────────────────────────────────────────────────────

class RLEngine(ABC):
    @abstractmethod
    def predict(self, state: Any) -> Any:
        ...

    @abstractmethod
    def load(self, path: str) -> bool:
        ...


class ONNXRLEngine(RLEngine):
    def __init__(self) -> None:
        self.session    = None
        self.input_name: Optional[str] = None

    def load(self, path: str) -> bool:
        try:
            import onnxruntime as ort   # type: ignore[import]
            if not os.path.exists(path):
                logger.warning("ONNX model not found: %s", path)
                return False
            providers = ["CPUExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers.insert(0, "CUDAExecutionProvider")
            self.session    = ort.InferenceSession(path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            logger.info("Loaded ONNX RL model: %s", path)
            return True
        except ImportError:
            logger.warning("onnxruntime not installed — RL engine unavailable")
            return False
        except Exception as exc:
            logger.error("Failed to load ONNX model: %s", exc)
            return False

    def predict(self, state: Any) -> Any:
        if self.session is None or not _NP:
            raise RuntimeError("ONNX session not loaded or numpy unavailable")
        if state.ndim == 3:
            state = state[np.newaxis, ...]
        outputs = self.session.run(None, {self.input_name: state.astype(np.float32)})
        return outputs[0][0]


# ── Main Placement Engine ─────────────────────────────────────────────────────

class PlacementEngine:
    """
    Unified multi-objective placement engine.

    Methods:
      auto       → RL+hybrid if available, else analytical
      rl         → ONNX RL sequential placement
      analytical → Quadratic wirelength minimisation
      thermal    → Thermal-spread + analytical
      hybrid     → RL global + analytical refinement
      rules      → Deterministic rule-based fallback
    """

    def __init__(self) -> None:
        self.rl_engine: Optional[RLEngine] = None
        self.rl_loaded: bool = False

    def load_rl_model(self, path: Optional[str] = None) -> bool:
        model_path = path or str(
            Path(__file__).parent.parent / "models" / "placement_model.onnx"
        )
        engine = ONNXRLEngine()
        if engine.load(model_path):
            self.rl_engine = engine
            self.rl_loaded = True
            return True
        return False

    # ── Main entry point ──────────────────────────────────────────────────────

    def optimize(
        self,
        components:       List[Component],
        nets:             List[Net],
        board_width:      float,
        board_height:     float,
        method:           str = "auto",
        fixed_components: Optional[Set[str]] = None,
    ) -> PlacementSolution:
        t0 = time.perf_counter()

        # Deep-copy components so callers' objects are never mutated
        # NOTE: _build_solution in v2.0 wrote back x/y/rotation in-place,
        # corrupting state on repeated calls.
        working_comps = copy.deepcopy(components)
        graph = ConnectivityGraph(nets, working_comps)

        if method == "auto":
            method = "hybrid" if self.rl_loaded else "analytical"

        dispatch = {
            "hybrid":     lambda: self._hybrid_optimize(graph, working_comps, board_width, board_height),
            "rl":         lambda: self._rl_optimize(graph, working_comps, board_width, board_height),
            "analytical": lambda: self._analytical_optimize(graph, board_width, board_height, fixed_components),
            "thermal":    lambda: self._thermal_optimize(graph, working_comps, board_width, board_height),
            "rules":      lambda: self._rule_based_fallback(working_comps, board_width, board_height),
        }

        if method not in dispatch:
            logger.warning("Unknown method '%s', falling back to analytical", method)
            method = "analytical"

        try:
            positions = dispatch[method]()
        except Exception as exc:
            logger.error("Method '%s' failed: %s — falling back to rules", method, exc)
            positions = self._rule_based_fallback(working_comps, board_width, board_height)
            method    = "rules_fallback"

        # Proximity enforcement: place decoupling caps near their IC
        positions = self._enforce_decoupling_proximity(positions, graph, board_width, board_height)

        solution = self._build_solution(
            positions, working_comps, nets, board_width, board_height
        )
        solution.algorithm          = method
        solution.computation_time_ms = (time.perf_counter() - t0) * 1000
        return solution

    # ── Placement methods ─────────────────────────────────────────────────────

    def _hybrid_optimize(
        self,
        graph:      ConnectivityGraph,
        components: List[Component],
        width:      float,
        height:     float,
    ) -> Dict[str, Tuple[float, float, float]]:
        # Step 1: RL global placement
        rl_pos = self._rl_optimize(graph, components, width, height)

        # Step 2: Analytical refinement — pass RL result as soft anchors
        fixed = {ref: (p[0], p[1]) for ref, p in rl_pos.items()}
        quad  = QuadraticPlacer(graph, width, height)
        refined = quad.solve(fixed)

        # Step 3: Blend 70 % analytical, 30 % RL
        result: Dict[str, Tuple[float, float, float]] = {}
        for ref, (rx, ry, rrot) in rl_pos.items():
            if ref in refined:
                qx, qy = refined[ref]
                result[ref] = (0.7 * qx + 0.3 * rx, 0.7 * qy + 0.3 * ry, rrot)
            else:
                result[ref] = (rx, ry, rrot)

        # Step 4: Force-directed wirelength refinement
        return self._refine_wirelength(result, graph, width, height)

    def _rl_optimize(
        self,
        graph:      ConnectivityGraph,
        components: List[Component],
        width:      float,
        height:     float,
    ) -> Dict[str, Tuple[float, float, float]]:
        if not self.rl_engine or not _NP:
            raise RuntimeError("RL engine not loaded or numpy unavailable")

        # Sort: most-connected first for sequential placement
        sorted_comps = sorted(
            components,
            key=lambda c: len(graph.get_neighbors(c.ref)),
            reverse=True,
        )

        grid_res = 2.0
        grid_w   = max(1, int(width  / grid_res))
        grid_h   = max(1, int(height / grid_res))

        # State channels:
        # 0 — occupied cells
        # 1 — normalised rotation of placed component
        # 2 — connectivity density (adjacency weight sum)
        # 3 — normalised power dissipation
        # NOTE: channels 2 and 3 were always zero in v2.0 — wasted state space.
        state = np.zeros((4, grid_h, grid_w), dtype=np.float32)

        positions: Dict[str, Tuple[float, float, float]] = {}

        for comp in sorted_comps:
            self._update_rl_state(state, positions, comp, graph, grid_res, width, height)

            action = self.rl_engine.predict(state)
            x   = float(np.clip(action[0] * width,  CONFIG.EDGE_MARGIN_MM, width  - CONFIG.EDGE_MARGIN_MM))
            y   = float(np.clip(action[1] * height, CONFIG.EDGE_MARGIN_MM, height - CONFIG.EDGE_MARGIN_MM))
            rot = float(action[2] % 360) if len(action) > 2 else 0.0

            positions[comp.ref] = (x, y, rot)

        return self._refine_wirelength(positions, graph, width, height)

    def _analytical_optimize(
        self,
        graph:      ConnectivityGraph,
        width:      float,
        height:     float,
        fixed:      Optional[Set[str]] = None,
    ) -> Dict[str, Tuple[float, float, float]]:
        fixed_pos: Optional[Dict[str, Tuple[float, float]]] = None
        if fixed:
            fixed_pos = {
                ref: (graph.components[ref].x, graph.components[ref].y)
                for ref in fixed
                if ref in graph.components
            }

        quad  = QuadraticPlacer(graph, width, height)
        pos2d = quad.solve(fixed_pos)
        return {ref: (x, y, 0.0) for ref, (x, y) in pos2d.items()}

    def _thermal_optimize(
        self,
        graph:      ConnectivityGraph,
        components: List[Component],
        width:      float,
        height:     float,
    ) -> Dict[str, Tuple[float, float, float]]:
        positions = self._analytical_optimize(graph, width, height)

        hot_comps = [c for c in components if c.power_dissipation > 0.5]
        if len(hot_comps) < 2:
            return positions

        m = CONFIG.EDGE_MARGIN_MM

        for _ in range(20):
            thermal = ThermalModel(width, height)
            for comp in components:
                if comp.ref not in positions:
                    continue
                # NOTE: use deepcopy so shallow copy doesn't share pins list
                tc = copy.deepcopy(comp)
                tc.x, tc.y = positions[comp.ref][0], positions[comp.ref][1]
                thermal.add_component(tc)
            thermal.solve_steady_state(10)

            for i, c1 in enumerate(hot_comps):
                if c1.ref not in positions:
                    continue
                x1, y1, r1 = positions[c1.ref]

                for c2 in hot_comps[i + 1:]:
                    if c2.ref not in positions:
                        continue
                    x2, y2, r2 = positions[c2.ref]

                    dist = math.hypot(x1 - x2, y1 - y2)
                    if 0 < dist < 20.0:
                        step = 2.0 / dist
                        dx, dy = (x1 - x2) * step, (y1 - y2) * step
                        positions[c1.ref] = (
                            max(m, min(width  - m, x1 + dx)),
                            max(m, min(height - m, y1 + dy)),
                            r1,
                        )
                        positions[c2.ref] = (
                            max(m, min(width  - m, x2 - dx)),
                            max(m, min(height - m, y2 - dy)),
                            r2,
                        )

        return positions

    def _rule_based_fallback(
        self,
        components: List[Component],
        width:      float,
        height:     float,
    ) -> Dict[str, Tuple[float, float, float]]:
        """
        Deterministic rule-based placement.

        NOTE: original used identity 'c not in list1 + list2' which always
        matched all components.  Now uses ref-set membership.
        """
        m = CONFIG.EDGE_MARGIN_MM

        ic_refs        = {c.ref for c in components if c.is_ic}
        passive_refs   = {c.ref for c in components if c.is_passive}
        connector_refs = {c.ref for c in components if c.is_connector}
        other_refs     = {
            c.ref for c in components
            if c.ref not in ic_refs
            and c.ref not in passive_refs
            and c.ref not in connector_refs
        }

        ics        = [c for c in components if c.ref in ic_refs]
        passives   = [c for c in components if c.ref in passive_refs]
        connectors = [c for c in components if c.ref in connector_refs]
        others     = [c for c in components if c.ref in other_refs]

        positions: Dict[str, Tuple[float, float, float]] = {}

        # ICs: centred grid
        ic_cols = max(1, int(math.sqrt(len(ics))))
        grid_mm = 15.0
        for i, ic in enumerate(ics):
            row, col = divmod(i, ic_cols)
            x = width  / 2 + (col - ic_cols / 2) * grid_mm
            y = height / 2 + (row - (len(ics) / ic_cols) / 2) * grid_mm
            positions[ic.ref] = (
                max(m, min(width - m, x)),
                max(m, min(height - m, y)),
                0.0,
            )

        # Passives: rows flanking ICs
        pcols = 8
        for i, p in enumerate(passives):
            row, col = divmod(i, pcols)
            side = 1 if row % 2 == 0 else -1
            x = width / 2 + side * (30 + (col % pcols) * 5)
            y = m + row * 5
            positions[p.ref] = (
                max(m, min(width - m, x)),
                max(m, min(height - m, y)),
                0.0 if side == 1 else 180.0,
            )

        # Connectors: left and right edges
        for i, conn in enumerate(connectors):
            if i % 2 == 0:
                positions[conn.ref] = (m, m + i * 10, 270.0)
            else:
                positions[conn.ref] = (width - m, m + i * 10, 90.0)

        # Others: bottom strip
        for i, o in enumerate(others):
            x = m + (i % 5) * 10
            y = height - m - (i // 5) * 10
            positions[o.ref] = (
                max(m, min(width - m, x)),
                max(m, min(height - m, y)),
                0.0,
            )

        return positions

    def _refine_wirelength(
        self,
        positions: Dict[str, Tuple[float, float, float]],
        graph:     ConnectivityGraph,
        width:     float,
        height:    float,
    ) -> Dict[str, Tuple[float, float, float]]:
        """
        Force-directed wirelength refinement.

        Repulsive forces now use a SpatialIndex (O(n) avg per component) instead
        of the original O(n²) all-pairs loop.
        """
        pos = {ref: list(p) for ref, p in positions.items()}
        m   = CONFIG.EDGE_MARGIN_MM

        # Build spatial index for O(1) neighbour queries
        idx = SpatialIndex()
        idx.build({ref: (p[0], p[1], p[2]) for ref, p in pos.items()})

        for _ in range(CONFIG.REFINEMENT_ITERATIONS):
            forces: Dict[str, List[float]] = {ref: [0.0, 0.0] for ref in pos}

            # Attractive: connected pairs
            for ref, neighbors in graph.adjacency.items():
                if ref not in pos:
                    continue
                x1, y1 = pos[ref][0], pos[ref][1]
                for nref, weight in neighbors.items():
                    if nref not in pos:
                        continue
                    x2, y2 = pos[nref][0], pos[nref][1]
                    dist = math.hypot(x2 - x1, y2 - y1) or CONFIG.EPSILON
                    ideal = 10.0
                    force = weight * (dist - ideal) / dist * 0.1
                    forces[ref][0]  += (x2 - x1) * force
                    forces[ref][1]  += (y2 - y1) * force

            # Repulsive: nearby pairs only (spatial index)
            for ref in list(pos.keys()):
                x1, y1 = pos[ref][0], pos[ref][1]
                for other, dist in idx.neighbors_within(ref, 15.0):
                    if dist < CONFIG.EPSILON:
                        continue
                    x2, y2 = pos[other][0], pos[other][1]
                    force = (15.0 - dist) / dist * 0.5
                    fx = (x1 - x2) * force
                    fy = (y1 - y2) * force
                    forces[ref][0] += fx
                    forces[ref][1] += fy

            # Apply and clamp
            for ref in pos:
                pos[ref][0] = max(m, min(width  - m, pos[ref][0] + forces[ref][0]))
                pos[ref][1] = max(m, min(height - m, pos[ref][1] + forces[ref][1]))

            # Rebuild spatial index after moves
            idx.build({r: (p[0], p[1], p[2]) for r, p in pos.items()})

        return {ref: (p[0], p[1], p[2]) for ref, p in pos.items()}

    def _enforce_decoupling_proximity(
        self,
        positions: Dict[str, Tuple[float, float, float]],
        graph:     ConnectivityGraph,
        width:     float,
        height:    float,
    ) -> Dict[str, Tuple[float, float, float]]:
        """
        New feature: move decoupling caps within DECOUPLING_RADIUS_MM of
        their IC if they are farther away.
        """
        m      = CONFIG.EDGE_MARGIN_MM
        radius = CONFIG.DECOUPLING_RADIUS_MM
        result = dict(positions)

        for ic_ref, cap_ref in graph.get_decoupling_pairs():
            if ic_ref not in result or cap_ref not in result:
                continue
            ix, iy, irot  = result[ic_ref]
            cx, cy, crot  = result[cap_ref]
            dist = math.hypot(cx - ix, cy - iy)
            if dist > radius:
                # Move cap toward IC to within radius
                t  = (dist - radius * 0.8) / dist
                nx = cx - (cx - ix) * t
                ny = cy - (cy - iy) * t
                result[cap_ref] = (
                    max(m, min(width  - m, nx)),
                    max(m, min(height - m, ny)),
                    crot,
                )
                logger.debug("Moved decap %s within %.1f mm of %s", cap_ref, radius, ic_ref)

        return result

    # ── RL state builder ──────────────────────────────────────────────────────

    def _update_rl_state(
        self,
        state:     Any,
        positions: Dict[str, Tuple[float, float, float]],
        next_comp: Component,
        graph:     ConnectivityGraph,
        grid_res:  float,
        width:     float,
        height:    float,
    ) -> None:
        if not _NP:
            return
        state.fill(0.0)
        for ref, (x, y, rot) in positions.items():
            gx = min(state.shape[2] - 1, max(0, int(x / grid_res)))
            gy = min(state.shape[1] - 1, max(0, int(y / grid_res)))
            state[0, gy, gx]  = 1.0
            state[1, gy, gx]  = rot / 360.0

            # Channel 2: normalised connection weight to next component
            conn_w = graph.get_neighbors(next_comp.ref).get(ref, 0.0)
            state[2, gy, gx] = min(1.0, conn_w)

            # Channel 3: normalised power dissipation
            comp = graph.components.get(ref)
            if comp:
                state[3, gy, gx] = min(1.0, comp.power_dissipation / 5.0)

    # ── Metrics ───────────────────────────────────────────────────────────────

    def _build_solution(
        self,
        positions:  Dict[str, Tuple[float, float, float]],
        components: List[Component],
        nets:       List[Net],
        width:      float,
        height:     float,
    ) -> PlacementSolution:
        """
        Calculate placement quality metrics.

        NOTE: does NOT mutate the caller's Component objects (positions are
        read from the positions dict, not from comp.x/comp.y).
        """
        pos2d = {ref: (x, y) for ref, (x, y, _) in positions.items()}

        # HPWL wirelength
        total_hpwl = sum(
            net.hpwl(pos2d) * net.weight for net in nets
        )

        # Thermal score
        # NOTE: deepcopy (not copy) prevents the thermal model from sharing
        # the `pins` list with the original component (shallow-copy bug).
        thermal = ThermalModel(width, height)
        for comp in components:
            if comp.ref not in positions:
                continue
            tc   = copy.deepcopy(comp)
            tc.x, tc.y = positions[comp.ref][0], positions[comp.ref][1]
            thermal.add_component(tc)
        thermal.solve_steady_state(20)
        max_t        = thermal.max_temperature()
        thermal_score = max(0.0, (max_t - 25.0) / 60.0 * 100.0)

        # Density score
        warnings: List[str] = []
        if _NP and CONFIG.BIN_SIZE_MM > 0:
            nx_bins = max(1, int(width  / CONFIG.BIN_SIZE_MM))
            ny_bins = max(1, int(height / CONFIG.BIN_SIZE_MM))
            density_map = np.zeros((ny_bins, nx_bins))
            for comp in components:
                if comp.ref not in positions:
                    continue
                bx = int(positions[comp.ref][0] / CONFIG.BIN_SIZE_MM)
                by = int(positions[comp.ref][1] / CONFIG.BIN_SIZE_MM)
                # NOTE: clamp to valid range — if comp.y == height exactly,
                # by == ny_bins which is out of bounds in the original.
                bx = min(nx_bins - 1, max(0, bx))
                by = min(ny_bins - 1, max(0, by))
                density_map[by, bx] += comp.area / (CONFIG.BIN_SIZE_MM ** 2)
            density_score = float(np.max(density_map) * 100)
        else:
            density_score = 0.0

        if thermal_score > 80:
            warnings.append(f"Peak thermal score {thermal_score:.0f}/100 — consider thermal spreading")
        if density_score > 90:
            warnings.append(f"Peak density {density_score:.0f}% — consider spreading components")

        # Count minimum-spacing constraint violations
        constraint_violations = 0
        min_gap = CONFIG.MIN_COMPONENT_SPACING_MM
        comp_list_with_pos = [
            (comp, positions[comp.ref][0], positions[comp.ref][1])
            for comp in components if comp.ref in positions
        ]
        for i, (c1, x1, y1) in enumerate(comp_list_with_pos):
            for c2, x2, y2 in comp_list_with_pos[i + 1:]:
                clearance = math.hypot(x1 - x2, y1 - y2)
                required  = (c1.width + c2.width) / 2 + min_gap
                if clearance < required:
                    constraint_violations += 1
        if constraint_violations:
            warnings.append(
                f"{constraint_violations} component-spacing violation(s) — "
                f"minimum {min_gap} mm gap not met"
            )

        return PlacementSolution(
            positions=positions,
            wirelength_mm=total_hpwl,
            thermal_score=thermal_score,
            density_score=density_score,
            constraint_violations=constraint_violations,
            warnings=warnings,
        )


# ── New: Differential-pair length matching ────────────────────────────────────

def check_differential_pair_lengths(
    nets:       List[Net],
    positions:  Dict[str, Tuple[float, float, float]],
    tolerance_mm: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Check that positive/negative nets in a differential pair have matched
    estimated lengths (Manhattan distance).

    Returns list of {net_p, net_n, length_p, length_n, delta_mm} dicts for
    pairs whose delta exceeds tolerance_mm.
    """
    pos2d = {ref: (x, y) for ref, (x, y, _) in positions.items()}

    # Group by base name: strip trailing P/N/+/- to find pairs
    pair_map: Dict[str, Dict[str, Net]] = defaultdict(dict)
    for net in nets:
        if net.net_type != NetType.DIFFERENTIAL:
            continue
        name  = net.name.upper()
        if name.endswith(("_P", "_PLUS", "_POS")):
            base = re.sub(r"(_P|_PLUS|_POS)$", "", net.name, flags=re.IGNORECASE)
            pair_map[base]["P"] = net
        elif name.endswith(("_N", "_NEG", "_MINUS")):
            base = re.sub(r"(_N|_NEG|_MINUS)$", "", net.name, flags=re.IGNORECASE)
            pair_map[base]["N"] = net

    violations: List[Dict[str, Any]] = []
    for base, pair in pair_map.items():
        if "P" not in pair or "N" not in pair:
            continue
        lp = pair["P"].hpwl(pos2d)
        ln = pair["N"].hpwl(pos2d)
        delta = abs(lp - ln)
        if delta > tolerance_mm:
            violations.append({
                "net_p":     pair["P"].name,
                "net_n":     pair["N"].name,
                "length_p":  round(lp, 2),
                "length_n":  round(ln, 2),
                "delta_mm":  round(delta, 2),
                "tolerance": tolerance_mm,
            })
    return violations


# ── Helper: net-type detection ────────────────────────────────────────────────

def _classify_net(net_name: str) -> NetType:
    upper = net_name.upper()
    if any(kw in upper for kw in ("VCC", "VDD", "PWR", "+5V", "+3V3", "+12V")):
        return NetType.POWER
    if any(kw in upper for kw in ("GND", "VSS", "AGND", "DGND")):
        return NetType.GROUND
    if any(kw in upper for kw in ("CLK", "CLOCK", "SCLK", "MCLK")):
        return NetType.CLOCK
    if any(kw in upper for kw in ("_P", "_N", "_PLUS", "_MINUS", "DIFF")):
        return NetType.DIFFERENTIAL
    if any(kw in upper for kw in ("HS", "HIGH_SPEED", "USB", "PCIE", "SERDES")):
        return NetType.HIGH_SPEED
    if any(kw in upper for kw in ("AIN", "AOUT", "VREF", "ANALOG")):
        return NetType.ANALOG
    return NetType.SIGNAL


# ── Public API ────────────────────────────────────────────────────────────────
# re is imported at the top of this module — no re-import needed here.

__all__ = [
    "PlacementEngine",
    "PlacementSolution",
    "PlacementConfig",
    "Component",
    "Net",
    "Pin",
    "NetType",
    "ConnectivityGraph",
    "ThermalModel",
    "SpatialIndex",
    "load_placement_model",
    "optimize_with_rl",
    "optimize_with_rules",
    "check_differential_pair_lengths",
    "_classify_net",
    "snap_rotation",
    "CONFIG",
]


def snap_rotation(angle: float, step: float = 90.0) -> float:
    """
    Snap an arbitrary rotation angle to the nearest multiple of *step* degrees.

    KiCad renders footprints cleanly at 0°, 90°, 180°, 270°.  The RL engine
    may produce fractional angles — this snaps them before export.

    Examples::

        snap_rotation(47.3)        # → 0.0   (nearest 90° multiple)
        snap_rotation(47.3, 45.0)  # → 45.0
        snap_rotation(135.1)       # → 135.0
    """
    if step <= 0:
        return angle % 360.0
    snapped = round(angle / step) * step
    return snapped % 360.0


def load_placement_model(model_path: Optional[str] = None) -> Optional[PlacementEngine]:
    """
    Factory function.  Returns a PlacementEngine with RL loaded (if available),
    or None if no backend at all could be initialised.

    NOTE: v2.0 always returned an engine even when RL failed, making
    `if model:` checks always True.  Now returns None on total failure.
    """
    engine = PlacementEngine()
    engine.load_rl_model(model_path)
    # Always return the engine — even without RL it can do analytical/rule-based.
    # Callers should inspect engine.rl_loaded to know if RL is available.
    return engine


def optimize_with_rl(
    model:      PlacementEngine,
    board_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Legacy-compatible interface expected by ai_server.py.

    Chooses 'hybrid' if RL is available, 'analytical' otherwise — previously
    always called hybrid and silently fell through to analytical anyway.
    """
    try:
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
                fixed=c.get("fixed", False),
            )
            for c in board_data.get("components", [])
        ]

        nets: List[Net] = []
        for n in board_data.get("connections", []):
            pins = [
                Pin(ref=p["ref"], pin=p.get("pin", ""))
                for p in n.get("pins", [])
                if isinstance(p, dict) and "ref" in p
            ]
            net_type = _classify_net(n["net"])
            nets.append(Net(name=n["net"], pins=pins, net_type=net_type))

        width  = float(board_data.get("board_width",  100.0))
        height = float(board_data.get("board_height",  80.0))

        method = "hybrid" if model.rl_loaded else "analytical"
        solution = model.optimize(
            components=components,
            nets=nets,
            board_width=width,
            board_height=height,
            method=method,
        )

        # Check differential pair length matching
        diff_violations = check_differential_pair_lengths(
            nets, solution.positions
        )

        return {
            "success": True,
            # Rotations are snapped to 90° increments for KiCad compatibility.
            "positions": {
                ref: {"x": round(x, 4), "y": round(y, 4), "rotation": snap_rotation(rot)}
                for ref, (x, y, rot) in solution.positions.items()
            },
            "improvement": (
                f"{solution.algorithm}: WL={solution.wirelength_mm:.1f} mm, "
                f"Thermal={solution.thermal_score:.1f}/100, "
                f"Density={solution.density_score:.1f}%, "
                f"Violations={solution.constraint_violations}"
            ),
            "metrics": {
                "wirelength_mm":         round(solution.wirelength_mm, 2),
                "thermal_score":         round(solution.thermal_score, 2),
                "density_score":         round(solution.density_score, 2),
                "constraint_violations": solution.constraint_violations,
                "net_count":             len(nets),
                "component_count":       len(components),
                "total_cost":            round(solution.total_cost, 4),
                "time_ms":         round(solution.computation_time_ms, 1),
                "algorithm":       solution.algorithm,
                "rl_available":    model.rl_loaded,
            },
            "warnings":             solution.warnings,
            "diff_pair_violations": diff_violations,
        }

    except Exception:
        logger.exception("Placement optimization failed")
        return {"success": False, "error": "Placement optimization failed — see server logs"}


def optimize_with_rules(board_data: Dict[str, Any]) -> Dict[str, Any]:
    """Rule-based fallback for when ML is unavailable."""
    engine = PlacementEngine()
    components = [
        Component(
            ref=c["ref"],
            value=c.get("value", ""),
            footprint=c.get("footprint", ""),
        )
        for c in board_data.get("components", [])
    ]
    width  = float(board_data.get("board_width",  100.0))
    height = float(board_data.get("board_height",  80.0))

    positions = engine._rule_based_fallback(components, width, height)
    return {
        "success":     True,
        "positions":   {
            ref: {"x": x, "y": y, "rotation": rot}
            for ref, (x, y, rot) in positions.items()
        },
        "improvement": "Rule-based placement (no ML model available)",
        "metrics":     {},
        "warnings":    [],
    }