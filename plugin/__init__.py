"""AI PCB Assistant package entrypoint.

This file stays lightweight so KiCad can import quickly.
Registration is delegated to pcbnew_action.py.
"""

__version__ = "0.1.0"

try:
    # Import side-effect: registers ActionPlugin wrapper.
    from .pcbnew_action import AIPlacementPluginWrapper as _Wrapper  # noqa: F401
except Exception as _e:
    import sys as _sys
    print(f"[AI PCB Assistant] Failed to load plugin entry point: {_e}", file=_sys.stderr)
