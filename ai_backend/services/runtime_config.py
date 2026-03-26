from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class CorsSettings:
    allow_origins: List[str]
    allow_credentials: bool


def parse_cors_settings(origins_env: str, credentials_env: str = "true") -> CorsSettings:
    """Parse CORS environment values and prevent wildcard+credentials mismatch."""
    origins = [o.strip() for o in origins_env.split(",") if o.strip()]
    if not origins:
        origins = ["http://127.0.0.1:3000", "http://localhost:3000"]

    allow_credentials = credentials_env.strip().lower() in {"1", "true", "yes", "on"}
    if "*" in origins and allow_credentials:
        # Browsers reject wildcard origins with credentials enabled.
        allow_credentials = False

    return CorsSettings(allow_origins=origins, allow_credentials=allow_credentials)
