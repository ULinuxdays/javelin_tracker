"""Biomechanics utilities for javelin throw analysis.

This module is intentionally **lazy-imported** so lightweight tooling (like the
elite metadata initializer) can be imported without optional heavy dependencies
required by pose estimation (e.g., `mediapipe`).
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "PoseDetector",
    "PosePipeline",
    "detect_throw_phases",
    "plot_phase_velocities",
    "PhaseResult",
    "extract_frames",
    "get_video_metadata",
    "validate_video_readable",
    "smooth_landmarks",
    "smooth_trajectory",
    "plot_trajectory_overlay",
    "validate_pose_quality",
    "check_anatomical_plausibility",
    "plot_quality_issues",
    "JAVELIN_KEY_JOINTS",
    "BIOMECHANICS_LOGGER",
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

_CONFIG_EXPORTS = {
    "CONFIDENCE_THRESHOLD",
    "ELITE_DATABASE_PATH",
    "FRAME_SMOOTHING_WINDOW",
    "JAVELIN_KEY_JOINTS",
    "BIOMECHANICS_LOGGER",
    "load_config_from_file",
    "OUTLIER_THRESHOLDS",
    "PHASE_TIMINGS",
    "VALID_FRAME_THRESHOLD",
    "VIDEO_STORAGE_PATH",
    "print_config",
    "validate_config_values",
    "validate_mediapipe_indices",
}

_UTILS_EXPORTS = {
    "extract_frames",
    "get_video_metadata",
    "validate_video_readable",
    "smooth_landmarks",
    "smooth_trajectory",
    "plot_trajectory_overlay",
    "validate_pose_quality",
    "check_anatomical_plausibility",
    "plot_quality_issues",
}

_METRICS_EXPORTS = {"detect_throw_phases", "plot_phase_velocities", "PhaseResult"}


def __getattr__(name: str) -> Any:  # pragma: no cover - exercised indirectly
    if name in _CONFIG_EXPORTS:
        from . import config as _config

        return getattr(_config, name)
    if name in _UTILS_EXPORTS:
        from . import utils as _utils

        return getattr(_utils, name)
    if name in _METRICS_EXPORTS:
        from . import metrics as _metrics

        return getattr(_metrics, name)
    if name == "PoseDetector":
        from .pose_estimation import PoseDetector

        return PoseDetector
    if name == "PosePipeline":
        from .pose_estimation.pipeline import PosePipeline

        return PosePipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover - trivial
    return sorted(set(list(globals()) + __all__))
