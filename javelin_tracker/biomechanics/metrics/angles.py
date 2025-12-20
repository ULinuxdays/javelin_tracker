"""Joint angle computations for pose-based biomechanics analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_CONFIDENCE_THRESHOLD = 0.5

DEFAULT_JAVELIN_KEY_JOINTS: dict[str, tuple[int, int]] = {
    "shoulders": (11, 12),
    "elbows": (13, 14),
    "wrists": (15, 16),
    "hips": (23, 24),
    "knees": (25, 26),
}

DEFAULT_ANKLES = (27, 28)


def _coerce_point(point: Any) -> tuple[np.ndarray, Optional[float]]:
    arr = np.asarray(point, dtype=float).reshape(-1)
    if arr.size < 2:
        return np.array([math.nan, math.nan, math.nan], dtype=float), None

    confidence: Optional[float] = None
    coords = arr
    if arr.size == 4:
        coords = arr[:3]
        confidence = float(arr[3])
    elif arr.size == 3:
        coords = arr[:3]
    else:
        coords = arr[:2]

    if coords.size == 2:
        coords = np.array([coords[0], coords[1], 0.0], dtype=float)
    else:
        coords = np.array([coords[0], coords[1], coords[2]], dtype=float)

    return coords, confidence


def _low_confidence(conf: Optional[float], threshold: float) -> bool:
    if conf is None:
        return False
    if not math.isfinite(float(conf)):
        return True
    return float(conf) < float(threshold)


def compute_joint_angle(p1: Any, p2: Any, p3: Any) -> float:
    """Compute the angle at p2 formed by points p1-p2-p3 (degrees).

    Accepts 2D or 3D points. If points include a confidence channel in the
    4th position (x, y, z, conf), returns NaN when any confidence is below
    `DEFAULT_CONFIDENCE_THRESHOLD`.
    """
    point1, conf1 = _coerce_point(p1)
    point2, conf2 = _coerce_point(p2)
    point3, conf3 = _coerce_point(p3)

    if any(_low_confidence(conf, DEFAULT_CONFIDENCE_THRESHOLD) for conf in (conf1, conf2, conf3)):
        return float("nan")
    if not np.all(np.isfinite([*point1, *point2, *point3])):
        return float("nan")

    vector1 = point1 - point2
    vector2 = point3 - point2
    norm1 = float(np.linalg.norm(vector1))
    norm2 = float(np.linalg.norm(vector2))
    if norm1 == 0.0 or norm2 == 0.0:
        return float("nan")

    cosine = float(np.dot(vector1, vector2) / (norm1 * norm2))
    cosine = float(np.clip(cosine, -1.0, 1.0))
    angle_rad = float(np.arccos(cosine))
    return float(np.degrees(angle_rad))


def _get_config_attr(config: Any, name: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        if name in config:
            return config[name]
        lowered = name.lower()
        if lowered in config:
            return config[lowered]
    return getattr(config, name, default)


def _landmark_at(landmarks: Any, index: int) -> Any:
    return landmarks[index]


def _confidence_at(point: Any) -> Optional[float]:
    arr = np.asarray(point, dtype=float).reshape(-1)
    if arr.size == 4:
        return float(arr[3])
    return None


def _coords3(point: Any) -> np.ndarray:
    coords, _conf = _coerce_point(point)
    return coords


def _quat_from_unit_vectors(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Quaternion rotating u to v (w, x, y, z). u and v must be unit vectors."""
    dot_uv = float(np.clip(np.dot(u, v), -1.0, 1.0))
    if dot_uv > 0.999999:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    if dot_uv < -0.999999:
        axis = np.cross(u, np.array([1.0, 0.0, 0.0], dtype=float))
        if np.linalg.norm(axis) < 1e-9:
            axis = np.cross(u, np.array([0.0, 1.0, 0.0], dtype=float))
        axis = axis / max(float(np.linalg.norm(axis)), 1e-9)
        return np.array([0.0, axis[0], axis[1], axis[2]], dtype=float)
    axis = np.cross(u, v)
    w = math.sqrt((1.0 + dot_uv) * 0.5)
    denom = 2.0 * w if w else 1e-9
    xyz = axis / denom
    return np.array([w, xyz[0], xyz[1], xyz[2]], dtype=float)


def _quat_angle_deg(q: np.ndarray) -> float:
    w = float(np.clip(q[0], -1.0, 1.0))
    return float(np.degrees(2.0 * math.acos(w)))


def _vector_rotation_angle_deg(
    p1: Any, p2: Any, p3: Any, p4: Any, *, confidence_threshold: float
) -> float:
    """Angle between vectors (p1->p2) and (p3->p4) using a quaternion rotation."""
    confs = [_confidence_at(p) for p in (p1, p2, p3, p4)]
    if any(_low_confidence(c, confidence_threshold) for c in confs):
        return float("nan")

    v1 = _coords3(p2) - _coords3(p1)
    v2 = _coords3(p4) - _coords3(p3)
    if not np.all(np.isfinite([*v1, *v2])):
        return float("nan")
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 == 0.0 or n2 == 0.0:
        return float("nan")
    u = v1 / n1
    v = v2 / n2
    q = _quat_from_unit_vectors(u, v)
    return _quat_angle_deg(q)


def compute_frame_angles(landmarks_frame: Sequence[Any], config: Any) -> Dict[str, Dict[str, object]]:
    """Compute key javelin angles for a single frame.

    Returns:
        {angle_name: {"value_degrees": float, "valid": bool}}
    """
    key_joints = _get_config_attr(config, "JAVELIN_KEY_JOINTS", DEFAULT_JAVELIN_KEY_JOINTS) or DEFAULT_JAVELIN_KEY_JOINTS
    confidence_threshold = float(_get_config_attr(config, "CONFIDENCE_THRESHOLD", DEFAULT_CONFIDENCE_THRESHOLD))

    left_shoulder, right_shoulder = key_joints.get("shoulders", DEFAULT_JAVELIN_KEY_JOINTS["shoulders"])
    left_elbow, right_elbow = key_joints.get("elbows", DEFAULT_JAVELIN_KEY_JOINTS["elbows"])
    left_wrist, right_wrist = key_joints.get("wrists", DEFAULT_JAVELIN_KEY_JOINTS["wrists"])
    left_hip, right_hip = key_joints.get("hips", DEFAULT_JAVELIN_KEY_JOINTS["hips"])
    left_knee, right_knee = key_joints.get("knees", DEFAULT_JAVELIN_KEY_JOINTS["knees"])
    _left_ankle, right_ankle = DEFAULT_ANKLES

    results: Dict[str, Dict[str, object]] = {}

    def add(name: str, value: float) -> None:
        results[name] = {"value_degrees": float(value), "valid": bool(math.isfinite(float(value)))}

    rs = _landmark_at(landmarks_frame, right_shoulder)
    re = _landmark_at(landmarks_frame, right_elbow)
    rw = _landmark_at(landmarks_frame, right_wrist)
    rh = _landmark_at(landmarks_frame, right_hip)
    rk = _landmark_at(landmarks_frame, right_knee)
    ra = _landmark_at(landmarks_frame, right_ankle)
    ls = _landmark_at(landmarks_frame, left_shoulder)
    lh = _landmark_at(landmarks_frame, left_hip)

    for name, points in (
        ("right_elbow", (rs, re, rw)),
        ("right_shoulder", (re, rs, rh)),
        ("right_hip", (rs, rh, rk)),
        ("right_knee", (rh, rk, ra)),
    ):
        confs = [_confidence_at(p) for p in points]
        if any(_low_confidence(conf, confidence_threshold) for conf in confs):
            add(name, float("nan"))
            continue
        add(name, compute_joint_angle(*points))

    thorax = _vector_rotation_angle_deg(
        lh,
        rh,
        ls,
        rs,
        confidence_threshold=confidence_threshold,
    )
    add("thorax_rotation", thorax)

    return results


@dataclass(frozen=True)
class _TrajectoryFrame:
    frame_idx: int
    timestamp_ms: float
    landmarks: Sequence[Any]


def _iter_trajectory_frames(landmarks_trajectory: Any, fps: float) -> Iterable[_TrajectoryFrame]:
    if isinstance(landmarks_trajectory, np.ndarray):
        for idx in range(int(landmarks_trajectory.shape[0])):
            yield _TrajectoryFrame(idx, idx * (1000.0 / fps), landmarks_trajectory[idx])
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
                yield _TrajectoryFrame(frame_idx, timestamp_ms, landmarks)
            return
        for idx, landmarks in enumerate(landmarks_trajectory):
            yield _TrajectoryFrame(idx, idx * (1000.0 / fps), landmarks)


def compute_trajectory_angles(landmarks_trajectory: Any, fps: float) -> pd.DataFrame:
    """Compute angles for an entire trajectory and return a long-form DataFrame.

    Output columns:
        frame, timestamp_ms, angle_name, value_degrees, valid
    """
    records: list[dict[str, object]] = []
    for frame in _iter_trajectory_frames(landmarks_trajectory, fps=float(fps)):
        angles = compute_frame_angles(frame.landmarks, config=None)
        for angle_name, entry in angles.items():
            records.append(
                {
                    "frame": int(frame.frame_idx),
                    "timestamp_ms": float(frame.timestamp_ms),
                    "angle_name": str(angle_name),
                    "value_degrees": float(entry["value_degrees"]),
                    "valid": bool(entry["valid"]),
                }
            )

    df = pd.DataFrame.from_records(records, columns=["frame", "timestamp_ms", "angle_name", "value_degrees", "valid"])
    if df.empty:
        return df
    df = df.sort_values(["frame", "angle_name"]).reset_index(drop=True)
    return df


__all__ = ["compute_joint_angle", "compute_frame_angles", "compute_trajectory_angles"]

