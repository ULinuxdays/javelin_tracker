from __future__ import annotations

import os
from typing import Any

PRIMARY_PREFIX = "THROWS_TRACKER_"
LEGACY_PREFIX = "JAVELIN_TRACKER_"


def get_env(name: str, default: str | None = None) -> str | None:
    """
    Resolve configuration environment variables.

    Prefers the Throws Tracker prefix while remaining backwards compatible with
    the historical Javelin Tracker names to avoid breaking existing deployments.
    """
    for prefix in (PRIMARY_PREFIX, LEGACY_PREFIX):
        value = os.getenv(f"{prefix}{name}")
        if value is not None:
            return value
    return default


def set_default(name: str, value: Any) -> None:
    """
    Assign a default environment variable if neither the new nor legacy names are set.
    """
    if get_env(name) is not None:
        return
    os.environ[f"{PRIMARY_PREFIX}{name}"] = str(value)
