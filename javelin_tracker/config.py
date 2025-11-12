from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from .env import get_env

try:  # pragma: no cover - Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - Python <3.11
    try:
        import tomli as tomllib  # type: ignore
    except ModuleNotFoundError:
        tomllib = None  # type: ignore

DEFAULT_ALLOWED_EVENTS: tuple[str, ...] = ("javelin", "discus", "shot", "hammer")


@dataclass(frozen=True)
class ACWRThresholds:
    sweet_min: float = 0.8
    sweet_max: float = 1.3
    high: float = 1.5


@dataclass(frozen=True)
class AppConfig:
    allowed_events: tuple[str, ...] = DEFAULT_ALLOWED_EVENTS
    acwr_thresholds: ACWRThresholds = ACWRThresholds()


def _config_path() -> Path | None:
    """Resolve the TOML configuration file, if present."""
    env_override = get_env("CONFIG")
    if env_override:
        path = Path(env_override).expanduser()
        return path if path.exists() else None

    for candidate in ("config/throws_tracker.toml", "config/javelin_tracker.toml"):
        default_path = Path(candidate)
        if default_path.exists():
            return default_path
    return None


def _load_toml(path: Path) -> Mapping[str, Any]:
    if tomllib is None:
        raise RuntimeError("TOML configuration requires Python 3.11+ or the 'tomli' package.")
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _coerce_allowed_events(raw: Any) -> tuple[str, ...]:
    if not raw:
        return DEFAULT_ALLOWED_EVENTS
    if isinstance(raw, str):
        entries = [entry.strip() for entry in raw.split(",")]
    elif isinstance(raw, (list, tuple, set)):
        entries = [str(entry).strip() for entry in raw]
    else:
        return DEFAULT_ALLOWED_EVENTS
    cleaned = tuple(event for event in entries if event)
    return cleaned or DEFAULT_ALLOWED_EVENTS


def _coerce_thresholds(raw: Mapping[str, Any] | None) -> ACWRThresholds:
    base = ACWRThresholds()
    if not raw:
        return base
    try:
        sweet_min = float(raw.get("sweet_min", base.sweet_min))
        sweet_max = float(raw.get("sweet_max", base.sweet_max))
        high = float(raw.get("high", base.high))
    except (TypeError, ValueError):
        return base
    return ACWRThresholds(sweet_min=sweet_min, sweet_max=sweet_max, high=high)


def _build_config(raw: Mapping[str, Any]) -> AppConfig:
    allowed_events = _coerce_allowed_events(raw.get("allowed_events"))
    thresholds_section = raw.get("acwr_thresholds")
    thresholds = _coerce_thresholds(thresholds_section if isinstance(thresholds_section, Mapping) else None)
    return AppConfig(allowed_events=allowed_events, acwr_thresholds=thresholds)


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Load configuration once, falling back to built-in defaults."""
    path = _config_path()
    if not path:
        return AppConfig()
    data = _load_toml(path)
    return _build_config(data)


def as_dict() -> dict[str, Any]:
    """Return the effective configuration for debug/CLI display."""
    config = get_config()
    return {
        "allowed_events": list(config.allowed_events),
        "acwr_thresholds": {
            "sweet_min": config.acwr_thresholds.sweet_min,
            "sweet_max": config.acwr_thresholds.sweet_max,
            "high": config.acwr_thresholds.high,
        },
        "source": str(_config_path() or "defaults"),
    }
