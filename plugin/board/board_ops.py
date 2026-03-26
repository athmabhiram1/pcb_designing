from __future__ import annotations

import re
from typing import Any, Dict, List

try:
    import pcbnew as _pcbnew
except ImportError:
    _pcbnew = None  # type: ignore[assignment]


def _require_pcbnew() -> None:
    """Raise RuntimeError if pcbnew is not available (i.e. running outside KiCad)."""
    if _pcbnew is None:
        raise RuntimeError("pcbnew is only available inside KiCad")


class BoardReader:
    """Extract board data without any UI dependencies."""

    def __init__(self, board: Any):
        self.board = board

    def read_components(self) -> List[Dict[str, Any]]:
        _require_pcbnew()
        out: List[Dict[str, Any]] = []
        for fp in self.board.GetFootprints():
            pos = fp.GetPosition()
            out.append(
                {
                    "ref": fp.GetReference(),
                    "value": fp.GetValue(),
                    "x": _pcbnew.ToMM(pos.x),
                    "y": _pcbnew.ToMM(pos.y),
                    "layer": "top" if fp.GetLayer() == _pcbnew.F_Cu else "bottom",
                }
            )
        return out

    def read_nets(self) -> List[Dict[str, Any]]:
        _require_pcbnew()
        nets: List[Dict[str, Any]] = []
        for net_name, net in self.board.GetNetInfo().NetsByName().items():
            if net.GetNetCode() == 0:
                continue
            pins: List[Dict[str, str]] = []
            for pad in net.GetPads():
                pins.append({"ref": pad.GetParent().GetReference(), "pin": pad.GetNumber()})
            if len(pins) >= 2:
                nets.append({"net": str(net_name), "pins": pins})
        return nets


class BoardWriter:
    """Write placement/net updates to a KiCad board without UI concerns."""

    def __init__(self, board: Any):
        self.board = board

    def apply_positions(self, positions: Dict[str, Dict[str, float]]) -> int:
        _require_pcbnew()
        moved = 0
        for fp in self.board.GetFootprints():
            ref = fp.GetReference()
            pos = positions.get(ref)
            if not pos:
                continue
            fp.SetPosition(_pcbnew.VECTOR2I(_pcbnew.FromMM(pos["x"]), _pcbnew.FromMM(pos["y"])))
            if "rotation" in pos:
                try:
                    fp.SetOrientationDegrees(float(pos["rotation"]))
                except Exception:
                    pass
            moved += 1
        if moved:
            _pcbnew.Refresh()
        return moved


class BoardImporter:
    """Import generated circuit_data into board footprints/pads."""

    def __init__(self, board: Any):
        self.board = board

    def import_circuit(self, circuit_data: Dict[str, Any]) -> Dict[str, Any]:
        _require_pcbnew()
        warnings: List[str] = []
        connected = 0

        footprint_map = {fp.GetReference(): fp for fp in self.board.GetFootprints()}
        for conn in circuit_data.get("connections", []):
            net_name = str(conn.get("net", "")).strip()
            if not net_name:
                continue
            net = self._ensure_net(net_name)
            for pin in conn.get("pins", []):
                if not isinstance(pin, dict):
                    continue
                ref = str(pin.get("ref", "")).strip()
                pin_no = str(pin.get("pin", "")).strip()
                fp = footprint_map.get(ref)
                if not fp:
                    continue
                try:
                    pad = self._resolve_pad_for_pin(fp, pin_no)
                    if pad:
                        pad.SetNet(net)
                        connected += 1
                    else:
                        warnings.append(f"{ref}.{pin_no}: pad not found")
                except Exception as exc:
                    warnings.append(f"{ref}.{pin_no}: {exc}")

        try:
            self.board.BuildConnectivity()
        except Exception:
            pass
        _pcbnew.Refresh()
        return {"connected_pads": connected, "warnings": warnings}

    @staticmethod
    def _pin_alias_candidates(pin: str) -> List[str]:
        token = re.sub(r"[^A-Z0-9+]", "", str(pin).upper())
        if not token:
            return []
        if token.isdigit():
            return [token]

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
        }
        out = aliases.get(token, []).copy()
        m = re.search(r"(\d+)$", token)
        if m:
            out.insert(0, m.group(1))
        dedup: List[str] = []
        for n in out:
            if n and n not in dedup:
                dedup.append(n)
        return dedup

    def _resolve_pad_for_pin(self, fp: Any, pin: str) -> Any:
        pin_raw = str(pin).strip()
        if not pin_raw:
            return None

        try:
            pad = fp.FindPadByNumber(pin_raw)
            if pad:
                return pad
        except Exception:
            pass

        for candidate in self._pin_alias_candidates(pin_raw):
            try:
                pad = fp.FindPadByNumber(candidate)
                if pad:
                    return pad
            except Exception:
                continue

        return None

    def _ensure_net(self, net_name: str) -> Any:
        _require_pcbnew()
        try:
            found = self.board.FindNet(net_name)
            if found:
                return found
        except Exception:
            pass
        net = _pcbnew.NETINFO_ITEM(self.board, net_name)
        self.board.Add(net)
        return net
