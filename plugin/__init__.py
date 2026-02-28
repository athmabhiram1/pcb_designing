"""AI PCB Assistant - KiCad Action Plugin.

This __init__.py intentionally triggers registration so that KiCad's
scripting/plugins loader (which only imports __init__.py for subdirectory
packages) picks up the ActionPlugin subclass.
"""

__version__ = "0.1.0"

try:
    # This import runs pcbnew_action.py, which calls
    # AIPlacementPluginWrapper().register() at module level.
    from .pcbnew_action import AIPlacementPluginWrapper as _Wrapper  # noqa: F401
except Exception as _e:
    import sys as _sys
    print(f"[AI PCB Assistant] Failed to load plugin entry point: {_e}",
          file=_sys.stderr)
