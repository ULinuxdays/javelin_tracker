"""Metrics computation package for biomechanics analyses.

This package is intentionally **lazy-imported** so lightweight tooling can use
pure-numpy metrics (e.g. joint angles) without requiring optional heavy
dependencies used by video/pose pipelines.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "detect_throw_phases",
    "plot_phase_velocities",
    "PhaseResult",
    "compute_joint_angle",
    "compute_frame_angles",
    "compute_trajectory_angles",
    "compute_velocity",
    "compute_acceleration",
    "compute_joint_velocities",
    "compute_velocity_profiles_dataframe",
    "get_velocity_peaks",
    "plot_joint_velocity_profiles",
    "compute_throw_metrics",
    "PhaseBoundaries",
    "MetricsPipeline",
    "normalize_angles",
    "normalize_positions",
    "normalize_velocities",
    "denormalize_positions",
    "denormalize_velocities",
    "handle_variable_fps",
    "normalize_athlete_metrics",
    "NormalizationContext",
]

_PHASE_EXPORTS = {"detect_throw_phases", "plot_phase_velocities", "PhaseResult"}
_ANGLE_EXPORTS = {"compute_joint_angle", "compute_frame_angles", "compute_trajectory_angles"}
_KINEMATICS_EXPORTS = {
    "compute_velocity",
    "compute_acceleration",
    "compute_joint_velocities",
    "compute_velocity_profiles_dataframe",
    "get_velocity_peaks",
    "plot_joint_velocity_profiles",
}
_THROW_EXPORTS = {"compute_throw_metrics", "PhaseBoundaries"}
_NORMALIZATION_EXPORTS = {
    "normalize_angles",
    "normalize_positions",
    "normalize_velocities",
    "denormalize_positions",
    "denormalize_velocities",
    "handle_variable_fps",
    "normalize_athlete_metrics",
    "NormalizationContext",
}

_PIPELINE_EXPORTS = {"MetricsPipeline"}


def __getattr__(name: str) -> Any:  # pragma: no cover - exercised indirectly
    if name in _PHASE_EXPORTS:
        try:
            from . import phase_detection as _phase_detection
        except Exception as exc:  # pragma: no cover - optional heavy deps
            raise ImportError(
                "Phase detection requires optional biomechanics dependencies (e.g. mediapipe/opencv)."
            ) from exc
        return getattr(_phase_detection, name)
    if name in _ANGLE_EXPORTS:
        from . import angles as _angles

        return getattr(_angles, name)
    if name in _KINEMATICS_EXPORTS:
        from . import kinematics as _kinematics

        return getattr(_kinematics, name)
    if name in _THROW_EXPORTS:
        from . import throw_metrics as _throw_metrics

        return getattr(_throw_metrics, name)
    if name in _NORMALIZATION_EXPORTS:
        from . import normalization as _normalization

        return getattr(_normalization, name)
    if name in _PIPELINE_EXPORTS:
        from . import pipeline as _pipeline

        return getattr(_pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover - trivial
    return sorted(set(list(globals()) + __all__))
