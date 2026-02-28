"""AI PCB Assistant - KiCad Action Plugin entry point for KiCad 9 PCM."""

from __future__ import annotations

import os

import pcbnew


def _get_real_plugin_class():
    """Import AIPlacementPlugin regardless of how this file was loaded.

    KiCad 9 PCM loads pcbnew_action.py as part of a package, so relative
    imports work.  KiCad's legacy scripting/plugins loader may load it as a
    standalone module via __init__.py, in which case we fall back to adding
    the directory to sys.path and using an absolute import.
    """
    try:
        from .plugin import AIPlacementPlugin  # loaded as package member
        return AIPlacementPlugin
    except ImportError:
        pass

    import sys
    _dir = os.path.dirname(os.path.abspath(__file__))
    if _dir not in sys.path:
        sys.path.insert(0, _dir)
    from plugin import AIPlacementPlugin  # type: ignore  # loaded standalone
    return AIPlacementPlugin


class AIPlacementPluginWrapper(pcbnew.ActionPlugin):
    @staticmethod
    def _resolve_icon_path() -> str:
        base_dir = os.path.dirname(__file__)
        candidates = [
            os.path.join(base_dir, "resources", "icon.png"),
            os.path.join(base_dir, "..", "resources", "icon.png"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return os.path.abspath(path)
        return ""

    def defaults(self):
        self.name = "AI PCB Assistant Pro"
        self.category = "AI Tools"
        self.description = "Advanced AI-powered placement, routing, and DFM"
        self.show_toolbar_button = True
        self.icon_file_name = self._resolve_icon_path()

    def Run(self):
        plugin_class = _get_real_plugin_class()
        # Re-use existing delegate so its _frame reference survives across calls.
        if not hasattr(self, "_delegate") or self._delegate is None:
            self._delegate = plugin_class()
            self._delegate.defaults()
        self._delegate.Run()


AIPlacementPluginWrapper().register()
