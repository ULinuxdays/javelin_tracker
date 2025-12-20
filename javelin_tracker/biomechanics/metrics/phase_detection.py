"""Detect throw phases (approach, delivery, release) from pose trajectories.

Uses smoothed landmarks to derive joint velocities and identify phase boundaries:
- Approach start: onset of forward motion (hip velocity surpasses threshold).
- Delivery start: elbow extension velocity peak.
- Release: hand velocity peak.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from javelin_tracker.biomechanics import BIOMECHANICS_LOGGER as logger, PHASE_TIMINGS
from javelin_tracker.biomechanics.config import JAVELIN_KEY_JOINTS


@dataclass(frozen=True)
class PhaseResult:
    approach_start_frame: Optional[int]
    delivery_start_frame: Optional[int]
    release_frame: Optional[int]
    phase_durations_ms: Dict[str, float]
    confidences: Dict[str, float]


def _central_diff(traj: np.ndarray, fps: float) -> np.ndarray:
    dt = 1.0 / fps
    return np.gradient(traj, dt, axis=0)


def _velocity_magnitude(trajectory: np.ndarray, fps: float) -> np.ndarray:
    vel = _central_diff(trajectory, fps)
    return np.linalg.norm(vel, axis=-1)


def _normalized_threshold(signal: np.ndarray, rel: float) -> float:
    return float(np.nanmax(signal) * rel) if np.size(signal) else 0.0


def detect_throw_phases(
    landmarks_trajectory: np.ndarray,
    fps: float,
    confidence_scores: Optional[np.ndarray] = None,
    *,
    approach_rel_thresh: float = 0.25,
    elbow_rel_thresh: float = 0.5,
    hand_rel_thresh: float = 0.7,
) -> PhaseResult:
    """Detect throw phase boundaries from smoothed pose landmarks.

    Args:
        landmarks_trajectory: np.ndarray (n_frames, 33, 3) of smoothed coordinates.
        fps: sampling rate.
        confidence_scores: optional (n_frames, 33) confidence per landmark.
        approach_rel_thresh: fraction of max hip velocity to mark approach onset.
        elbow_rel_thresh: fraction of max elbow velocity to mark delivery start.
        hand_rel_thresh: fraction of max hand velocity to mark release.

    Returns:
        PhaseResult with frame indices (or None) and durations in ms.
    """
    if landmarks_trajectory.ndim != 3 or landmarks_trajectory.shape[1:] != (33, 3):
        raise ValueError("landmarks_trajectory must be (n_frames, 33, 3).")
    n_frames = landmarks_trajectory.shape[0]
    if n_frames < 3:
        logger.warning("Not enough frames to detect phases.")
        return PhaseResult(None, None, None, {}, {})

    hips_idx = JAVELIN_KEY_JOINTS["hips"]
    elbows_idx = JAVELIN_KEY_JOINTS["elbows"]
    wrists_idx = JAVELIN_KEY_JOINTS["wrists"]

    hip_vel = _velocity_magnitude(landmarks_trajectory[:, hips_idx, :].mean(axis=1), fps)
    elbow_vel = _velocity_magnitude(landmarks_trajectory[:, elbows_idx, :].mean(axis=1), fps)
    hand_vel = _velocity_magnitude(landmarks_trajectory[:, wrists_idx, :].mean(axis=1), fps)

    hip_thresh = _normalized_threshold(hip_vel, approach_rel_thresh)
    elbow_thresh = _normalized_threshold(elbow_vel, elbow_rel_thresh)
    hand_thresh = _normalized_threshold(hand_vel, hand_rel_thresh)

    approach_start = int(np.argmax(hip_vel > hip_thresh)) if np.any(hip_vel > hip_thresh) else None
    delivery_start = int(np.argmax(elbow_vel > elbow_thresh)) if np.any(elbow_vel > elbow_thresh) else None
    release_frame = int(np.argmax(hand_vel > hand_thresh)) if np.any(hand_vel > hand_thresh) else None

    # Confidence: normalized velocity at detected frame.
    confidences = {}
    if approach_start is not None:
        confidences["approach_start"] = float(hip_vel[approach_start] / (np.nanmax(hip_vel) + 1e-9))
    if delivery_start is not None:
        confidences["delivery_start"] = float(elbow_vel[delivery_start] / (np.nanmax(elbow_vel) + 1e-9))
    if release_frame is not None:
        confidences["release"] = float(hand_vel[release_frame] / (np.nanmax(hand_vel) + 1e-9))

    # Durations based on detected frames; fall back to phase timing ratios if missing.
    durations_ms: Dict[str, float] = {}
    if approach_start is not None and delivery_start is not None:
        durations_ms["approach"] = max(0.0, (delivery_start - approach_start) / fps * 1000.0)
    if delivery_start is not None and release_frame is not None:
        durations_ms["delivery"] = max(0.0, (release_frame - delivery_start) / fps * 1000.0)
    if approach_start is not None and release_frame is not None:
        durations_ms["total"] = max(0.0, (release_frame - approach_start) / fps * 1000.0)
    elif release_frame is not None:
        durations_ms["total"] = float(release_frame / fps * 1000.0)
    else:
        # Fallback using configured phase timing ratios if nothing detected.
        durations_ms["total"] = float(PHASE_TIMINGS.release[1] * n_frames / fps * 1000.0)

    return PhaseResult(
        approach_start_frame=approach_start,
        delivery_start_frame=delivery_start,
        release_frame=release_frame,
        phase_durations_ms=durations_ms,
        confidences=confidences,
    )


def plot_phase_velocities(
    hip_vel: np.ndarray,
    elbow_vel: np.ndarray,
    hand_vel: np.ndarray,
    phases: PhaseResult,
    fps: float,
    title: str = "Throw phase detection",
) -> plt.Figure:
    """Plot velocity curves with detected phase boundaries for debugging."""
    t = np.arange(len(hip_vel)) / fps
    fig, ax = plt.subplots()
    ax.plot(t, hip_vel, label="hips")
    ax.plot(t, elbow_vel, label="elbows")
    ax.plot(t, hand_vel, label="hands")

    def _mark(frame: Optional[int], label: str, color: str):
        if frame is not None:
            ax.axvline(frame / fps, color=color, linestyle="--", alpha=0.7, label=label)

    _mark(phases.approach_start_frame, "approach start", "tab:green")
    _mark(phases.delivery_start_frame, "delivery start", "tab:orange")
    _mark(phases.release_frame, "release", "tab:red")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity magnitude")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig
