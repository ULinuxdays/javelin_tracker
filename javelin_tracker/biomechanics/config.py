"""Configuration for biomechanical analysis of javelin throws.

Settings include:
- JAVELIN_KEY_JOINTS: MediaPipe landmark indices used for angles/kinematics.
- CONFIDENCE_THRESHOLD: Minimum landmark confidence to accept a point.
- VALID_FRAME_THRESHOLD: Minimum fraction of valid pose points to use a frame.
- FRAME_SMOOTHING_WINDOW: Rolling window size for metric smoothing.
- ELITE_DATABASE_PATH / VIDEO_STORAGE_PATH: Relative storage locations.
- PHASE_TIMINGS: Percent ranges (0-1) for approach, delivery, release phases.
- OUTLIER_THRESHOLDS: Cutoffs for z-score, angle jumps, and position jumps.

All values can be overridden via environment variables to ease experimentation.
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

from javelin_tracker.env import get_env

try:  # Optional heavy dependency (pose estimation)
    import mediapipe as mp  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    mp = None  # type: ignore[assignment]

def _has_mediapipe_solutions() -> bool:
    """Return True when `mediapipe.solutions` APIs are available.

    Newer MediaPipe wheels (notably some cp312 macOS builds) ship only the Tasks
    API (`mediapipe.tasks`) and do not include `mediapipe.solutions.*`.
    """
    if mp is None:
        return False
    try:
        solutions = getattr(mp, "solutions", None)
        return bool(solutions and getattr(solutions, "holistic", None))
    except Exception:
        return False


HAS_MEDIAPIPE_SOLUTIONS = _has_mediapipe_solutions()

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - fallback when tomllib missing
    tomllib = None  # type: ignore[assignment]


def _configure_logger() -> logging.Logger:
    logger = logging.getLogger("javelin_tracker.biomechanics")
    level_name = get_env("LOG_LEVEL") or os.getenv("LOG_LEVEL")
    if level_name:
        level = getattr(logging, level_name.upper(), logging.INFO)
        logger.setLevel(level)
    return logger


BIOMECHANICS_LOGGER = _configure_logger()
logger = BIOMECHANICS_LOGGER

# MediaPipe landmark mapping for key joints used in angle/velocity calculations.
#
# MediaPipe is optional: for lightweight environments, fall back to the known
# PoseLandmark indices used by the Holistic/Pose models so pure-numpy metrics
# can still run and config can be imported without `mediapipe` installed.
if HAS_MEDIAPIPE_SOLUTIONS:
    POSE_LANDMARK = mp.solutions.holistic.PoseLandmark
    JAVELIN_KEY_JOINTS: Dict[str, Tuple[int, int]] = {
        "shoulders": (POSE_LANDMARK.LEFT_SHOULDER.value, POSE_LANDMARK.RIGHT_SHOULDER.value),
        "elbows": (POSE_LANDMARK.LEFT_ELBOW.value, POSE_LANDMARK.RIGHT_ELBOW.value),
        "wrists": (POSE_LANDMARK.LEFT_WRIST.value, POSE_LANDMARK.RIGHT_WRIST.value),
        "hips": (POSE_LANDMARK.LEFT_HIP.value, POSE_LANDMARK.RIGHT_HIP.value),
        "knees": (POSE_LANDMARK.LEFT_KNEE.value, POSE_LANDMARK.RIGHT_KNEE.value),
    }
else:  # pragma: no cover - only when mediapipe unavailable
    POSE_LANDMARK = None
    # https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
    # LEFT_SHOULDER=11, RIGHT_SHOULDER=12, LEFT_ELBOW=13, RIGHT_ELBOW=14, LEFT_WRIST=15, RIGHT_WRIST=16,
    # LEFT_HIP=23, RIGHT_HIP=24, LEFT_KNEE=25, RIGHT_KNEE=26
    JAVELIN_KEY_JOINTS = {
        "shoulders": (11, 12),
        "elbows": (13, 14),
        "wrists": (15, 16),
        "hips": (23, 24),
        "knees": (25, 26),
    }


def _get_env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_env_path(key: str, default: str) -> Path:
    return Path(os.getenv(key, default)).expanduser()


def _get_env_range(key: str, default: Tuple[float, float]) -> Tuple[float, float]:
    raw = os.getenv(key)
    if not raw:
        return default
    for sep in (",", ":", "-"):
        if sep in raw:
            try:
                start_str, end_str = raw.split(sep)
                return float(start_str), float(end_str)
            except ValueError:
                break
    return default


def _coerce_range_tuple(value: Any, default: Tuple[float, float]) -> Tuple[float, float]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return float(value[0]), float(value[1])
        except (TypeError, ValueError):
            return default
    if isinstance(value, str):
        for sep in (",", ":", "-"):
            if sep in value:
                try:
                    start_str, end_str = value.split(sep)
                    return float(start_str), float(end_str)
                except ValueError:
                    continue
    return default


def _load_toml_file(path: Path) -> Dict[str, Any]:
    if tomllib:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    try:
        import toml  # type: ignore
    except ModuleNotFoundError as exc:
        raise ImportError(
            "tomllib is unavailable; install the 'toml' package to load TOML configs."
        ) from exc
    return toml.load(path)


@dataclass(frozen=True)
class PhaseTimings:
    """Percentage ranges (0-1) marking the throw phases."""

    approach: Tuple[float, float]
    delivery: Tuple[float, float]
    release: Tuple[float, float]


@dataclass(frozen=True)
class OutlierThresholds:
    """Thresholds used to flag spurious kinematic values."""

    max_zscore: float
    max_angle_jump_deg: float
    max_position_jump_m: float


# Core thresholds for deciding valid frames/landmarks.
CONFIDENCE_THRESHOLD: float = _get_env_float("JAVELIN_CONFIDENCE_THRESHOLD", 0.5)
VALID_FRAME_THRESHOLD: float = _get_env_float("JAVELIN_VALID_FRAME_THRESHOLD", 0.8)
FRAME_SMOOTHING_WINDOW: int = _get_env_int("JAVELIN_FRAME_SMOOTHING_WINDOW", 5)

# Storage defaults (relative paths are resolved from project root).
ELITE_DATABASE_PATH: Path = _get_env_path("JAVELIN_ELITE_DATABASE_PATH", "data/elite_throws")
VIDEO_STORAGE_PATH: Path = _get_env_path("JAVELIN_VIDEO_STORAGE_PATH", "data/user_videos")

# Phase timing defaults can be tuned via env (e.g., "0.0-0.65").
PHASE_TIMINGS = PhaseTimings(
    approach=_get_env_range("JAVELIN_PHASE_APPROACH", (0.0, 0.65)),
    delivery=_get_env_range("JAVELIN_PHASE_DELIVERY", (0.65, 0.9)),
    release=_get_env_range("JAVELIN_PHASE_RELEASE", (0.9, 1.0)),
)

# Outlier detection tuning for metrics smoothing and anomaly detection.
OUTLIER_THRESHOLDS = OutlierThresholds(
    max_zscore=_get_env_float("JAVELIN_OUTLIER_MAX_ZSCORE", 3.0),
    max_angle_jump_deg=_get_env_float("JAVELIN_OUTLIER_MAX_ANGLE_JUMP_DEG", 35.0),
    max_position_jump_m=_get_env_float("JAVELIN_OUTLIER_MAX_POSITION_JUMP_M", 0.5),
)

__all__ = [
    "JAVELIN_KEY_JOINTS",
    "CONFIDENCE_THRESHOLD",
    "VALID_FRAME_THRESHOLD",
    "FRAME_SMOOTHING_WINDOW",
    "ELITE_DATABASE_PATH",
    "VIDEO_STORAGE_PATH",
    "PHASE_TIMINGS",
    "OUTLIER_THRESHOLDS",
    "load_config_from_file",
    "validate_config_values",
    "validate_mediapipe_indices",
    "print_config",
]


def load_config_from_file(config_path: Path) -> Dict[str, Any]:
    """Load biomechanics config from TOML or JSON and apply env var overrides.

    Env vars take precedence over file values. Supports either a root-level
    mapping or a [biomechanics] table/object in the config file.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    if not path.is_file():
        raise ValueError(f"Expected a config file, but got a directory: {path}")

    suffix = path.suffix.lower()
    if suffix == ".toml":
        raw_config = _load_toml_file(path)
    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            raw_config = json.load(handle)
    else:
        raise ValueError(f"Unsupported config format for {path}; expected .toml or .json.")

    config_body = raw_config.get("biomechanics", raw_config) if isinstance(raw_config, dict) else raw_config
    if not isinstance(config_body, dict):
        raise ValueError("Invalid config structure; expected a dict or a [biomechanics] section.")

    phase_cfg = config_body.get("phase_timings", {}) if isinstance(config_body.get("phase_timings", {}), dict) else {}
    outlier_cfg = (
        config_body.get("outlier_thresholds", {}) if isinstance(config_body.get("outlier_thresholds", {}), dict) else {}
    )

    return {
        "JAVELIN_KEY_JOINTS": config_body.get("javelin_key_joints", JAVELIN_KEY_JOINTS),
        "CONFIDENCE_THRESHOLD": _get_env_float(
            "JAVELIN_CONFIDENCE_THRESHOLD", float(config_body.get("confidence_threshold", 0.5))
        ),
        "VALID_FRAME_THRESHOLD": _get_env_float(
            "JAVELIN_VALID_FRAME_THRESHOLD", float(config_body.get("valid_frame_threshold", 0.8))
        ),
        "FRAME_SMOOTHING_WINDOW": _get_env_int(
            "JAVELIN_FRAME_SMOOTHING_WINDOW", int(config_body.get("frame_smoothing_window", 5))
        ),
        "ELITE_DATABASE_PATH": _get_env_path(
            "JAVELIN_ELITE_DATABASE_PATH", str(config_body.get("elite_database_path", "data/elite_throws"))
        ),
        "VIDEO_STORAGE_PATH": _get_env_path(
            "JAVELIN_VIDEO_STORAGE_PATH", str(config_body.get("video_storage_path", "data/user_videos"))
        ),
        "PHASE_TIMINGS": PhaseTimings(
            approach=_get_env_range(
                "JAVELIN_PHASE_APPROACH", _coerce_range_tuple(phase_cfg.get("approach"), (0.0, 0.65))
            ),
            delivery=_get_env_range(
                "JAVELIN_PHASE_DELIVERY", _coerce_range_tuple(phase_cfg.get("delivery"), (0.65, 0.9))
            ),
            release=_get_env_range(
                "JAVELIN_PHASE_RELEASE", _coerce_range_tuple(phase_cfg.get("release"), (0.9, 1.0))
            ),
        ),
        "OUTLIER_THRESHOLDS": OutlierThresholds(
            max_zscore=_get_env_float("JAVELIN_OUTLIER_MAX_ZSCORE", float(outlier_cfg.get("max_zscore", 3.0))),
            max_angle_jump_deg=_get_env_float(
                "JAVELIN_OUTLIER_MAX_ANGLE_JUMP_DEG", float(outlier_cfg.get("max_angle_jump_deg", 35.0))
            ),
            max_position_jump_m=_get_env_float(
                "JAVELIN_OUTLIER_MAX_POSITION_JUMP_M", float(outlier_cfg.get("max_position_jump_m", 0.5))
            ),
        ),
    }


def _check_thresholds() -> None:
    for name, value in (
        ("CONFIDENCE_THRESHOLD", CONFIDENCE_THRESHOLD),
        ("VALID_FRAME_THRESHOLD", VALID_FRAME_THRESHOLD),
    ):
        if not 0.0 <= value <= 1.0:
            warnings.warn(
                f"{name}={value} is outside [0,1]; please correct the environment or config.",
                RuntimeWarning,
                stacklevel=2,
            )
            logger.warning("%s is outside [0,1]: %s", name, value)


def _check_phase_timings() -> None:
    phases = [
        ("approach", PHASE_TIMINGS.approach),
        ("delivery", PHASE_TIMINGS.delivery),
        ("release", PHASE_TIMINGS.release),
    ]
    total_span = 0.0
    for label, (start, end) in phases:
        if not (0.0 <= start <= end <= 1.0):
            warnings.warn(
                f"Phase '{label}' range {start}-{end} is outside [0,1] or invalid; adjust config.",
                RuntimeWarning,
                stacklevel=2,
            )
            logger.warning("Phase '%s' range invalid/outside [0,1]: %s-%s", label, start, end)
        total_span += max(0.0, end - start)
    if not 0.95 <= total_span <= 1.05:
        warnings.warn(
            f"Phase timings span {total_span:.2f} (expected ~1.0). Check approach/delivery/release ranges.",
            RuntimeWarning,
            stacklevel=2,
        )
        logger.warning("Phase timing span suspicious: %.2f (expected ~1.0)", total_span)


def _check_paths() -> None:
    for name, path in (
        ("ELITE_DATABASE_PATH", ELITE_DATABASE_PATH),
        ("VIDEO_STORAGE_PATH", VIDEO_STORAGE_PATH),
    ):
        resolved = path.expanduser()
        target = resolved if resolved.is_dir() else resolved.parent
        if not target.exists():
            warnings.warn(
                f"{name} target '{resolved}' does not exist. Ensure parent '{target}' is created/writable.",
                RuntimeWarning,
                stacklevel=2,
            )
            logger.warning("%s target does not exist: %s (parent %s)", name, resolved, target)
        if target.exists():
            if not os.access(target, os.R_OK):
                warnings.warn(
                    f"{name} target '{target}' is not readable; check permissions.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                logger.warning("%s target not readable: %s", name, target)
            if not os.access(target, os.W_OK):
                warnings.warn(
                    f"{name} target '{target}' is not writable; check permissions.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                logger.warning("%s target not writable: %s", name, target)


def _check_smoothing_window() -> None:
    if FRAME_SMOOTHING_WINDOW <= 0:
        warnings.warn(
            f"FRAME_SMOOTHING_WINDOW={FRAME_SMOOTHING_WINDOW} is non-positive; expected positive odd integer.",
            RuntimeWarning,
            stacklevel=2,
        )
        logger.warning("FRAME_SMOOTHING_WINDOW is non-positive: %s", FRAME_SMOOTHING_WINDOW)
    if FRAME_SMOOTHING_WINDOW % 2 == 0:
        warnings.warn(
            f"FRAME_SMOOTHING_WINDOW={FRAME_SMOOTHING_WINDOW} is even; smoothing kernels often require odd sizes.",
            RuntimeWarning,
            stacklevel=2,
        )
        logger.warning("FRAME_SMOOTHING_WINDOW is even (prefer odd): %s", FRAME_SMOOTHING_WINDOW)


def validate_config_values() -> None:
    """Validate current config values and emit warnings for suspicious settings."""
    _check_thresholds()
    _check_phase_timings()
    _check_paths()
    _check_smoothing_window()


def validate_mediapipe_indices() -> None:
    """Verify JAVELIN_KEY_JOINTS matches current MediaPipe Holistic indices."""
    if mp is None or not HAS_MEDIAPIPE_SOLUTIONS:  # pragma: no cover - optional dependency
        warnings.warn(
            "mediapipe solutions are unavailable; cannot validate landmark indices against the model.",
            RuntimeWarning,
            stacklevel=2,
        )
        logger.warning("mediapipe solutions unavailable; skipping validate_mediapipe_indices().")
        return
    pose_enum = mp.solutions.holistic.PoseLandmark
    expected = {
        "shoulders": (pose_enum.LEFT_SHOULDER.value, pose_enum.RIGHT_SHOULDER.value),
        "elbows": (pose_enum.LEFT_ELBOW.value, pose_enum.RIGHT_ELBOW.value),
        "wrists": (pose_enum.LEFT_WRIST.value, pose_enum.RIGHT_WRIST.value),
        "hips": (pose_enum.LEFT_HIP.value, pose_enum.RIGHT_HIP.value),
        "knees": (pose_enum.LEFT_KNEE.value, pose_enum.RIGHT_KNEE.value),
    }
    for key, expected_pair in expected.items():
        configured = JAVELIN_KEY_JOINTS.get(key)
        if configured != expected_pair:
            warnings.warn(
                f"{key} indices differ from MediaPipe Holistic ({configured} != {expected_pair}). "
                "Update JAVELIN_KEY_JOINTS to match the current model.",
                RuntimeWarning,
                stacklevel=2,
            )
            logger.warning(
                "%s indices mismatch MediaPipe Holistic (configured=%s, expected=%s)",
                key,
                configured,
                expected_pair,
            )


def print_config() -> None:
    """Print configuration values for debugging purposes."""
    print("Javelin biomechanics configuration:")
    print(f"  Key joints: {JAVELIN_KEY_JOINTS}")
    print(f"  Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"  Valid frame threshold: {VALID_FRAME_THRESHOLD}")
    print(f"  Frame smoothing window: {FRAME_SMOOTHING_WINDOW}")
    print(f"  Elite database path: {ELITE_DATABASE_PATH}")
    print(f"  Video storage path: {VIDEO_STORAGE_PATH}")
    print(
        "  Phase timings (approach, delivery, release): "
        f"{PHASE_TIMINGS.approach}, {PHASE_TIMINGS.delivery}, {PHASE_TIMINGS.release}"
    )
    print(
        "  Outlier thresholds (z-score, angle jump deg, position jump m): "
        f"{OUTLIER_THRESHOLDS.max_zscore}, "
        f"{OUTLIER_THRESHOLDS.max_angle_jump_deg}, "
        f"{OUTLIER_THRESHOLDS.max_position_jump_m}"
    )


# Run validation at import to surface misconfigurations early.
validate_config_values()
validate_mediapipe_indices()
