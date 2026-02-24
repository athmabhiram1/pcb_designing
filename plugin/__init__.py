"""
AI PCB Assistant - KiCad Action Plugin
Entry point for the KiCad plugin system.
"""
from .plugin import AIPlacementPlugin

# Register plugin with KiCad
AIPlacementPlugin().register()
