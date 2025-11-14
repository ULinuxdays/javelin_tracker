"""javelin_tracker package."""

from importlib import metadata

from .cli import app

try:
    __version__ = metadata.version("throws-tracker")
except metadata.PackageNotFoundError:  # pragma: no cover - fallback for local edits
    __version__ = "0.0.0"

__all__ = ["app", "__version__"]
