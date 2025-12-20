"""javelin_tracker package."""

from importlib import metadata
from typing import Any

try:
    __version__ = metadata.version("throws-tracker")
except metadata.PackageNotFoundError:  # pragma: no cover - fallback for local edits
    __version__ = "0.0.0"

__all__ = ["app", "__version__", "biomechanics"]


def __getattr__(name: str) -> Any:  # pragma: no cover - exercised indirectly
    if name == "app":
        from .cli import app

        return app
    if name == "biomechanics":
        from . import biomechanics

        return biomechanics
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
