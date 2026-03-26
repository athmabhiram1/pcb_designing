from __future__ import annotations

import sys
from pathlib import Path

# Ensure ai_backend/ is importable as top-level module path during tests.
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
