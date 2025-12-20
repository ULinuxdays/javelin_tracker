"""Velocity/acceleration computations for pose-based biomechanics analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_CONFIDENCE_THRESHOLD = 0.5


def _ensure_3d(points: np.ndarray) -> np.ndarray:
    if points.shape[-1] == 3:
        return points
    if points.shape[-1] == 2:
        pad = np.zeros(points.shape[:-1] + (1,), dtype=float)
        return np.concatenate([points, pad], axis=-1)
    raise ValueError(f"Expected last dimension 2 or 3, got {points.shape[-1]}")


def _as_float_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=float)


def _safe_divide(numer: np.ndarray, denom: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = numer / denom
    out = np.where(np.isfinite(out), out, np.nan)
    return out


def compute_velocity(trajectory_3d: Any, timestamps: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute velocity vectors and magnitudes using central differences.

    Central frames use:
        v[i] = (x[i+1] - x[i-1]) / (t[i+1] - t[i-1])

    Endpoints use forward/backward differences.

    Args:
        trajectory_3d: shape (n_frames, ..., 2|3) coordinates.
        timestamps: length n_frames time values (seconds).

    Returns:
        speed: shape (n_frames, ...) velocity magnitude.
        velocity: shape (n_frames, ..., 3) velocity vectors.
    """
    coords = _ensure_3d(_as_float_array(trajectory_3d))
    if coords.shape[0] != len(timestamps):
        raise ValueError("trajectory_3d and timestamps must have the same length")

    n = int(coords.shape[0])
    vel = np.full_like(coords, np.nan, dtype=float)
    t = _as_float_array(timestamps).reshape(-1)

    if n < 2:
        speed = np.linalg.norm(vel, axis=-1)
        return speed, vel

    dt0 = t[1] - t[0]
    dt1 = t[-1] - t[-2]
    vel[0] = _safe_divide(coords[1] - coords[0], dt0)
    vel[-1] = _safe_divide(coords[-1] - coords[-2], dt1)

    if n > 2:
        denom = (t[2:] - t[:-2]).reshape((n - 2,) + (1,) * (coords.ndim - 1))
        vel[1:-1] = _safe_divide(coords[2:] - coords[:-2], denom)

    speed = np.linalg.norm(vel, axis=-1)
    return speed, vel


def compute_acceleration(velocity_trajectory: Any, timestamps: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute acceleration vectors and magnitudes using central differences."""
    return compute_velocity(velocity_trajectory, timestamps)


def _infer_joint_names() -> list[str]:
    try:  # Optional: use mediapipe names if available.
        import mediapipe as mp  # type: ignore

        names = [lm.name.lower() for lm in mp.solutions.holistic.PoseLandmark]  # type: ignore[attr-defined]
        if len(names) == 33:
            return names
    except Exception:
        pass
    return [f"joint_{i}" for i in range(33)]


def _confidence_array(landmarks: np.ndarray) -> Optional[np.ndarray]:
    if landmarks.shape[-1] == 4:
        return landmarks[..., 3]
    return None


def _coords_array(landmarks: np.ndarray) -> np.ndarray:
    if landmarks.shape[-1] == 4:
        return landmarks[..., :3]
    return _ensure_3d(landmarks)


@dataclass(frozen=True)
class _TrajectoryFrame:
    frame_idx: int
    timestamp_ms: float
    landmarks: Sequence[Any]
    valid: bool


def _iter_trajectory_frames(landmarks_trajectory: Any, fps: float) -> Iterable[_TrajectoryFrame]:
    if isinstance(landmarks_trajectory, Mapping) and "frames" in landmarks_trajectory:
        landmarks_trajectory = landmarks_trajectory["frames"]

    if isinstance(landmarks_trajectory, np.ndarray):
        for idx in range(int(landmarks_trajectory.shape[0])):
            yield _TrajectoryFrame(idx, idx * (1000.0 / fps), landmarks_trajectory[idx], True)
        return

    if isinstance(landmarks_trajectory, Sequence) and landmarks_trajectory:
        first = landmarks_trajectory[0]
        if isinstance(first, Mapping) and "landmarks" in first:
            for idx, frame in enumerate(landmarks_trajectory):
                if not isinstance(frame, Mapping):
                    continue
                landmarks = frame.get("landmarks")
                if not isinstance(landmarks, Sequence):
                    continue
                frame_idx = int(frame.get("frame_idx", idx))
                timestamp_ms = float(frame.get("timestamp_ms", frame_idx * (1000.0 / fps)))
                valid = bool(frame.get("valid", True))
                yield _TrajectoryFrame(frame_idx, timestamp_ms, landmarks, valid)
            return
        for idx, landmarks in enumerate(landmarks_trajectory):
            yield _TrajectoryFrame(idx, idx * (1000.0 / fps), landmarks, True)


def _extract_landmarks_and_timestamps(
    landmarks_trajectory: Any,
    fps: float,
    *,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frames = list(_iter_trajectory_frames(landmarks_trajectory, float(fps)))
    if not frames:
        return (
            np.zeros((0, 33, 3), dtype=float),
            np.zeros((0, 33), dtype=float),
            np.zeros((0,), dtype=float),
            np.zeros((0,), dtype=int),
        )

    n_frames = len(frames)
    landmarks = np.full((n_frames, 33, 4), np.nan, dtype=float)
    timestamps_ms = np.zeros((n_frames,), dtype=float)
    frame_indices = np.zeros((n_frames,), dtype=int)
    for i, frame in enumerate(frames):
        frame_indices[i] = int(frame.frame_idx)
        timestamps_ms[i] = float(frame.timestamp_ms)
        if not frame.valid:
            continue
        if len(frame.landmarks) != 33:
            continue
        try:
            arr = np.asarray(frame.landmarks, dtype=float)
        except Exception:
            continue
        if arr.ndim != 2 or arr.shape[0] != 33:
            continue
        if arr.shape[1] == 4:
            landmarks[i, :, :] = arr
        elif arr.shape[1] == 3:
            landmarks[i, :, :3] = arr
            landmarks[i, :, 3] = 1.0
        elif arr.shape[1] == 2:
            landmarks[i, :, 0:2] = arr
            landmarks[i, :, 2] = 0.0
            landmarks[i, :, 3] = 1.0

    conf = landmarks[:, :, 3]
    coords = landmarks[:, :, :3]
    invalid = ~(np.isfinite(conf) & (conf >= float(confidence_threshold)))
    coords = coords.copy()
    coords[invalid] = np.nan
    return coords, conf, timestamps_ms, frame_indices


def _nearest_index_for_frame(frame_indices: np.ndarray, frame: int) -> int:
    if frame_indices.size == 0:
        return 0
    idx = int(np.argmin(np.abs(frame_indices.astype(float) - float(frame))))
    return max(0, min(idx, int(frame_indices.size - 1)))


def _detect_release_frame(coords: np.ndarray, fps: float) -> int:
    n_frames = int(coords.shape[0])
    if n_frames <= 0:
        return 0

    try:
        from javelin_tracker.biomechanics.metrics.phase_detection import detect_throw_phases

        result = detect_throw_phases(coords, fps=float(fps))
        if getattr(result, "release_frame", None) is not None:
            return int(result.release_frame)
    except Exception:
        pass

    return int(round(0.90 * (n_frames - 1)))


def _timing_phrase(joint_label: str, delta_ms: Optional[float]) -> Optional[str]:
    if delta_ms is None or not math.isfinite(float(delta_ms)):
        return None
    rounded = int(round(abs(float(delta_ms))))
    if rounded == 0:
        return f"{joint_label} reaches max speed at release"
    when = "before" if float(delta_ms) < 0 else "after"
    return f"{joint_label} reaches max speed {rounded}ms {when} release"


def compute_joint_velocities(
    landmarks_trajectory: Any,
    fps: float,
    *,
    joint_names: Optional[Sequence[str]] = None,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    release_frame: Optional[int] = None,
) -> Dict[str, Dict[str, object]]:
    """Compute per-joint velocity magnitudes and peak timings."""
    coords, _conf, timestamps_ms, frame_indices = _extract_landmarks_and_timestamps(
        landmarks_trajectory, float(fps), confidence_threshold=float(confidence_threshold)
    )
    if coords.shape[0] == 0:
        return {}

    timestamps_s = timestamps_ms / 1000.0
    speeds, _vectors = compute_velocity(coords, timestamps_s)

    n_frames, n_joints = int(coords.shape[0]), int(coords.shape[1])
    names = list(joint_names) if joint_names is not None else _infer_joint_names()
    if len(names) != n_joints:
        names = [f"joint_{i}" for i in range(n_joints)]

    if release_frame is not None:
        rel_index = _nearest_index_for_frame(frame_indices, int(release_frame))
    else:
        rel_index = int(_detect_release_frame(coords, float(fps)))
        rel_index = max(0, min(rel_index, n_frames - 1))
    rel_frame_value = int(frame_indices[rel_index]) if frame_indices.size else int(rel_index)

    summary: Dict[str, Dict[str, object]] = {}
    for j in range(n_joints):
        joint = names[j]
        joint_speeds = speeds[:, j]
        finite = np.isfinite(joint_speeds)
        if not np.any(finite):
            peak_vel = float("nan")
            peak_index = -1
        else:
            peak_index = int(np.nanargmax(joint_speeds))
            peak_vel = float(joint_speeds[peak_index])

        peak_offset_ms: Optional[float] = None
        if peak_index >= 0 and math.isfinite(peak_vel):
            peak_offset_ms = float(timestamps_ms[peak_index] - timestamps_ms[rel_index])

        label = joint.replace("_", " ")
        if "wrist" in joint:
            timing_summary = _timing_phrase("Wrist", peak_offset_ms)
        else:
            timing_summary = _timing_phrase(label.title(), peak_offset_ms)

        peak_frame_value = (
            int(frame_indices[peak_index]) if peak_index >= 0 and frame_indices.size else int(peak_index)
        )

        summary[joint] = {
            "velocities": [float(v) for v in joint_speeds.tolist()],
            "timestamps": [float(t) for t in timestamps_ms.tolist()],
            "peak_velocity": peak_vel,
            "peak_frame": peak_frame_value,
            "peak_time_ms": None if peak_index < 0 else float(timestamps_ms[peak_index]),
            "peak_relative_to_release_ms": peak_offset_ms,
            "timing_summary": timing_summary,
            "release_frame": rel_frame_value,
            "release_time_ms": float(timestamps_ms[rel_index]),
        }

    return summary


def compute_velocity_profiles_dataframe(
    landmarks_trajectory: Any,
    fps: float,
    *,
    joint_names: Optional[Sequence[str]] = None,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> pd.DataFrame:
    """Return per-joint velocity vectors/magnitudes as a long-form DataFrame."""
    coords, _conf, timestamps_ms, frame_indices = _extract_landmarks_and_timestamps(
        landmarks_trajectory, float(fps), confidence_threshold=float(confidence_threshold)
    )
    if coords.shape[0] == 0:
        return pd.DataFrame(
            columns=["frame", "timestamp_ms", "joint_index", "joint", "vx", "vy", "vz", "speed", "valid"]
        )

    timestamps_s = timestamps_ms / 1000.0
    speeds, vectors = compute_velocity(coords, timestamps_s)
    n_frames, n_joints = int(coords.shape[0]), int(coords.shape[1])

    names = list(joint_names) if joint_names is not None else _infer_joint_names()
    if len(names) != n_joints:
        names = [f"joint_{i}" for i in range(n_joints)]

    frame_idx = frame_indices.astype(int, copy=False) if frame_indices.size else np.arange(n_frames, dtype=int)
    records: list[dict[str, object]] = []
    for j in range(n_joints):
        joint = names[j]
        for i in range(n_frames):
            vx, vy, vz = vectors[i, j, :].tolist()
            speed = float(speeds[i, j])
            records.append(
                {
                    "frame": int(frame_idx[i]),
                    "timestamp_ms": float(timestamps_ms[i]),
                    "joint_index": int(j),
                    "joint": str(joint),
                    "vx": float(vx) if math.isfinite(float(vx)) else float("nan"),
                    "vy": float(vy) if math.isfinite(float(vy)) else float("nan"),
                    "vz": float(vz) if math.isfinite(float(vz)) else float("nan"),
                    "speed": speed,
                    "valid": bool(math.isfinite(speed)),
                }
            )

    df = pd.DataFrame.from_records(
        records, columns=["frame", "timestamp_ms", "joint_index", "joint", "vx", "vy", "vz", "speed", "valid"]
    )
    return df.sort_values(["joint_index", "frame"]).reset_index(drop=True)


def get_velocity_peaks(kinematics_df: pd.DataFrame) -> list[tuple[str, float, int]]:
    """Return (joint, peak_velocity, frame_idx) sorted by peak_velocity desc."""
    required = {"joint", "speed", "frame"}
    if not required.issubset(set(kinematics_df.columns)):
        raise ValueError(f"kinematics_df must include columns: {sorted(required)}")

    peaks: list[tuple[str, float, int]] = []
    for joint, group in kinematics_df.groupby("joint"):
        speeds = group["speed"].astype(float)
        if speeds.notna().any():
            idx = int(speeds.idxmax())
            peak = float(kinematics_df.loc[idx, "speed"])
            frame = int(kinematics_df.loc[idx, "frame"])
            peaks.append((str(joint), peak, frame))

    peaks.sort(key=lambda item: item[1], reverse=True)
    return peaks


def plot_joint_velocity_profiles(
    kinematics_df: pd.DataFrame,
    *,
    joints: Optional[Sequence[str]] = None,
    max_joints: Optional[int] = None,
):
    """Plot velocity magnitude over time for each joint."""
    import matplotlib

    try:
        matplotlib.use("Agg", force=False)
    except Exception:
        pass
    import matplotlib.pyplot as plt

    if kinematics_df.empty:
        return None

    df = kinematics_df.copy()
    df["timestamp_ms"] = df["timestamp_ms"].astype(float)
    df["speed"] = df["speed"].astype(float)

    if joints is not None:
        df = df[df["joint"].isin(list(joints))]

    joint_order = sorted(df["joint"].unique())
    if max_joints is not None:
        joint_order = joint_order[: int(max_joints)]
        df = df[df["joint"].isin(joint_order)]

    fig, ax = plt.subplots(figsize=(10, 5))
    for joint in joint_order:
        subset = df[df["joint"] == joint]
        ax.plot(subset["timestamp_ms"], subset["speed"], label=str(joint), linewidth=1.0, alpha=0.9)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Speed (rel units / s)")
    ax.set_title("Joint velocity profiles")
    ax.grid(True, alpha=0.25)
    if len(joint_order) <= 12:
        ax.legend(loc="upper right", fontsize=8)
    return fig


__all__ = [
    "compute_velocity",
    "compute_acceleration",
    "compute_joint_velocities",
    "compute_velocity_profiles_dataframe",
    "get_velocity_peaks",
    "plot_joint_velocity_profiles",
]
