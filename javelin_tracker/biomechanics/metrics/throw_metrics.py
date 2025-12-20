"""Throw-specific metrics computed from a processed pose trajectory.

This module focuses on a *single throw* and produces coach-facing summaries:
- Phase durations (ms + % of total)
- Release metrics (wrist speed, segment heights, angles at release)
- Power chain lag (hip -> shoulder -> wrist peak timing)
- Symmetry scores (left vs right)
- Center-of-mass proxy displacement (hip/shoulder midpoint trajectory length)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from javelin_tracker.biomechanics.metrics.angles import compute_joint_angle
from javelin_tracker.biomechanics.metrics.kinematics import compute_velocity


DEFAULT_CONFIDENCE_THRESHOLD = 0.5

# MediaPipe pose indices used when mediapipe isn't available.
DEFAULT_JAVELIN_KEY_JOINTS: dict[str, tuple[int, int]] = {
    "shoulders": (11, 12),
    "elbows": (13, 14),
    "wrists": (15, 16),
    "hips": (23, 24),
    "knees": (25, 26),
}
DEFAULT_ANKLES = (27, 28)


@dataclass(frozen=True)
class PhaseBoundaries:
    approach_start: int
    delivery_start: int
    release_frame: int


def _as_float_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=float)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_phase_boundaries(phase_boundaries: Any, n_frames: int) -> PhaseBoundaries:
    if phase_boundaries is None:
        return _fallback_phase_boundaries(n_frames)

    if isinstance(phase_boundaries, PhaseBoundaries):
        return _clamp_phase_boundaries(phase_boundaries, n_frames)

    if isinstance(phase_boundaries, Mapping):
        approach_start = _safe_int(
            phase_boundaries.get("approach_start_frame", phase_boundaries.get("approach_start", 0)),
            0,
        )
        delivery_start = _safe_int(
            phase_boundaries.get("delivery_start_frame", phase_boundaries.get("delivery_start", int(round(0.65 * (n_frames - 1))))),
            int(round(0.65 * (n_frames - 1))),
        )
        release_frame = _safe_int(
            phase_boundaries.get("release_frame", phase_boundaries.get("release", int(round(0.90 * (n_frames - 1))))),
            int(round(0.90 * (n_frames - 1))),
        )
        return _clamp_phase_boundaries(PhaseBoundaries(approach_start, delivery_start, release_frame), n_frames)

    for name in ("approach_start_frame", "delivery_start_frame", "release_frame"):
        if hasattr(phase_boundaries, name):
            approach_start = _safe_int(getattr(phase_boundaries, "approach_start_frame", 0), 0)
            delivery_start = _safe_int(
                getattr(phase_boundaries, "delivery_start_frame", int(round(0.65 * (n_frames - 1)))),
                int(round(0.65 * (n_frames - 1))),
            )
            release_frame = _safe_int(getattr(phase_boundaries, "release_frame", int(round(0.90 * (n_frames - 1)))), int(round(0.90 * (n_frames - 1))))
            return _clamp_phase_boundaries(PhaseBoundaries(approach_start, delivery_start, release_frame), n_frames)

    return _fallback_phase_boundaries(n_frames)


def _fallback_phase_boundaries(n_frames: int) -> PhaseBoundaries:
    if n_frames <= 1:
        return PhaseBoundaries(0, 0, 0)
    delivery_start = int(round(0.65 * (n_frames - 1)))
    release_frame = int(round(0.90 * (n_frames - 1)))
    return _clamp_phase_boundaries(PhaseBoundaries(0, delivery_start, release_frame), n_frames)


def _clamp_phase_boundaries(boundaries: PhaseBoundaries, n_frames: int) -> PhaseBoundaries:
    if n_frames <= 0:
        return PhaseBoundaries(0, 0, 0)
    approach = max(0, min(int(boundaries.approach_start), n_frames - 1))
    delivery = max(approach, min(int(boundaries.delivery_start), n_frames - 1))
    release = max(delivery, min(int(boundaries.release_frame), n_frames - 1))
    return PhaseBoundaries(approach, delivery, release)


@dataclass(frozen=True)
class _TrajectoryFrame:
    frame_idx: int
    timestamp_ms: float
    landmarks: Sequence[Any]
    valid: bool


def _iter_trajectory_frames(landmarks: Any, fps: float) -> Iterable[_TrajectoryFrame]:
    if isinstance(landmarks, Mapping) and "frames" in landmarks:
        landmarks = landmarks["frames"]

    if isinstance(landmarks, np.ndarray):
        for idx in range(int(landmarks.shape[0])):
            yield _TrajectoryFrame(idx, idx * (1000.0 / fps), landmarks[idx], True)
        return

    if isinstance(landmarks, Sequence) and landmarks:
        first = landmarks[0]
        if isinstance(first, Mapping) and "landmarks" in first:
            for idx, frame in enumerate(landmarks):
                if not isinstance(frame, Mapping):
                    continue
                raw = frame.get("landmarks")
                if not isinstance(raw, Sequence):
                    continue
                frame_idx = _safe_int(frame.get("frame_idx", idx), idx)
                timestamp_ms = float(frame.get("timestamp_ms", frame_idx * (1000.0 / fps)))
                valid = bool(frame.get("valid", True))
                yield _TrajectoryFrame(frame_idx, timestamp_ms, raw, valid)
            return
        for idx, frame_landmarks in enumerate(landmarks):
            yield _TrajectoryFrame(idx, idx * (1000.0 / fps), frame_landmarks, True)


def _extract_landmarks_array(
    landmarks: Any,
    fps: float,
    *,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames = list(_iter_trajectory_frames(landmarks, float(fps)))
    if not frames:
        return (
            np.zeros((0, 33, 4), dtype=float),
            np.zeros((0,), dtype=float),
            np.zeros((0,), dtype=float),
        )

    n_frames = len(frames)
    out = np.full((n_frames, 33, 4), np.nan, dtype=float)
    timestamps_ms = np.zeros((n_frames,), dtype=float)
    valid_mask = np.zeros((n_frames,), dtype=bool)

    for i, frame in enumerate(frames):
        timestamps_ms[i] = float(frame.timestamp_ms)
        valid_mask[i] = bool(frame.valid)
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
            out[i, :, :] = arr
        elif arr.shape[1] == 3:
            out[i, :, :3] = arr
            out[i, :, 3] = 1.0
        elif arr.shape[1] == 2:
            out[i, :, 0:2] = arr
            out[i, :, 2] = 0.0
            out[i, :, 3] = 1.0

    # Apply confidence threshold and frame validity.
    conf = out[:, :, 3]
    coords = out[:, :, :3].copy()
    invalid = ~(valid_mask[:, None] & np.isfinite(conf) & (conf >= float(confidence_threshold)))
    coords[invalid] = np.nan
    out[:, :, :3] = coords
    return out, timestamps_ms, valid_mask.astype(float)


def _angle_deg_2d(p_left: np.ndarray, p_right: np.ndarray) -> float:
    delta = p_right[:2] - p_left[:2]
    if not np.all(np.isfinite(delta)):
        return float("nan")
    return float(np.degrees(np.arctan2(delta[1], delta[0])))


def _signed_angle_about_axis_deg(v1: np.ndarray, v2: np.ndarray, axis: np.ndarray) -> float:
    """Signed angle between v1 and v2 around axis (degrees)."""
    v1 = np.asarray(v1, dtype=float).reshape(-1)
    v2 = np.asarray(v2, dtype=float).reshape(-1)
    axis = np.asarray(axis, dtype=float).reshape(-1)
    if v1.size < 3 or v2.size < 3 or axis.size < 3:
        return float("nan")
    v1 = v1[:3]
    v2 = v2[:3]
    axis = axis[:3]
    if not np.all(np.isfinite([*v1, *v2, *axis])):
        return float("nan")
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-9:
        return float("nan")
    axis_u = axis / axis_norm

    v1p = v1 - float(np.dot(v1, axis_u)) * axis_u
    v2p = v2 - float(np.dot(v2, axis_u)) * axis_u
    n1 = float(np.linalg.norm(v1p))
    n2 = float(np.linalg.norm(v2p))
    if n1 <= 1e-9 or n2 <= 1e-9:
        return float("nan")

    cross = np.cross(v1p, v2p)
    sin = float(np.dot(cross, axis_u))
    cos = float(np.dot(v1p, v2p))
    return float(np.degrees(np.arctan2(sin, cos)))


def _throwing_side(
    left_wrist_speed: np.ndarray, right_wrist_speed: np.ndarray, release_frame: int, delivery_slice: slice
) -> str:
    idx = max(0, min(int(release_frame), int(left_wrist_speed.size - 1)))
    left = float(left_wrist_speed[idx]) if idx >= 0 and idx < left_wrist_speed.size else float("nan")
    right = float(right_wrist_speed[idx]) if idx >= 0 and idx < right_wrist_speed.size else float("nan")
    if math.isfinite(left) and math.isfinite(right):
        return "right" if right >= left else "left"

    # Fall back to delivery-peak.
    left_peak = float(np.nanmax(left_wrist_speed[delivery_slice])) if left_wrist_speed.size else float("nan")
    right_peak = float(np.nanmax(right_wrist_speed[delivery_slice])) if right_wrist_speed.size else float("nan")
    if not math.isfinite(left_peak) and not math.isfinite(right_peak):
        return "right"
    if not math.isfinite(left_peak):
        return "right"
    if not math.isfinite(right_peak):
        return "left"
    return "right" if right_peak >= left_peak else "left"


def _score_from_diff(diff: float, scale: float) -> float:
    if not math.isfinite(diff) or diff < 0:
        return float("nan")
    denom = max(float(scale), 1e-9)
    return float(1.0 / (1.0 + (diff / denom)))


def _mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return float("nan")
    mask = np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(a[mask] - b[mask])))


def _path_length(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return 0.0
    diffs = np.diff(points, axis=0)
    finite = np.isfinite(diffs).all(axis=1)
    if not np.any(finite):
        return 0.0
    seg = np.linalg.norm(diffs[finite], axis=1)
    return float(np.sum(seg))


_THROW_ID_SLUG_RE = re.compile(r"[^A-Za-z0-9_]+")


def _extract_throw_id(payload_or_frames: Any) -> str:
    if isinstance(payload_or_frames, Mapping):
        for key in ("throw_id", "video_id", "id"):
            value = payload_or_frames.get(key)
            if value:
                text = str(value).strip().replace(" ", "_")
                return _THROW_ID_SLUG_RE.sub("_", text).strip("_") or "unknown"
    return "unknown"


def compute_throw_metrics(
    landmarks: Any,
    phase_boundaries: Any,
    fps: float,
) -> Dict[str, object]:
    """Compute throw-level summary metrics for a processed pose trajectory."""
    fps = float(fps) if fps else 30.0
    payload = landmarks
    throw_id = _extract_throw_id(payload)

    arr, timestamps_ms, valid_mask = _extract_landmarks_array(payload, fps)
    n_frames = int(arr.shape[0])
    if n_frames == 0:
        return {
            "throw_id": throw_id,
            "phase_durations": {},
            "release_metrics": {},
            "power_chain_lag_ms": float("nan"),
            "symmetry": {},
            "com_displacement_m": float("nan"),
            "confidence_flags": {"has_data": False},
            "warnings": ["No frames/landmarks available."],
        }

    boundaries = _coerce_phase_boundaries(phase_boundaries, n_frames)
    approach_start, delivery_start, release_frame = boundaries.approach_start, boundaries.delivery_start, boundaries.release_frame

    total_ms = (n_frames / fps) * 1000.0
    approach_ms = max(0.0, (delivery_start - approach_start) / fps * 1000.0)
    delivery_ms = max(0.0, (release_frame - delivery_start) / fps * 1000.0)
    release_ms = max(0.0, (n_frames - release_frame) / fps * 1000.0)

    def pct(ms: float) -> float:
        return float((ms / total_ms) * 100.0) if total_ms > 0 else float("nan")

    phase_durations = {
        "approach_ms": float(approach_ms),
        "delivery_ms": float(delivery_ms),
        "release_ms": float(release_ms),
        "total_ms": float(total_ms),
        "approach_pct": pct(approach_ms),
        "delivery_pct": pct(delivery_ms),
        "release_pct": pct(release_ms),
        "frames": n_frames,
        "approach_start_frame": int(approach_start),
        "delivery_start_frame": int(delivery_start),
        "release_frame": int(release_frame),
    }

    key = DEFAULT_JAVELIN_KEY_JOINTS
    ls, rs = key["shoulders"]
    le, re = key["elbows"]
    lw, rw = key["wrists"]
    lh, rh = key["hips"]
    lk, rk = key["knees"]
    la, ra = DEFAULT_ANKLES

    coords = arr[:, :, :3]
    conf = arr[:, :, 3]
    timestamps_s = timestamps_ms / 1000.0

    delivery_slice = slice(delivery_start, max(delivery_start + 1, release_frame + 1))

    # Determine throwing side by wrist speed near release.
    left_wrist_speed, _ = compute_velocity(coords[:, lw, :], timestamps_s)
    right_wrist_speed, _ = compute_velocity(coords[:, rw, :], timestamps_s)
    side = _throwing_side(left_wrist_speed, right_wrist_speed, release_frame, delivery_slice)

    if side == "right":
        shoulder_idx, elbow_idx, wrist_idx, hip_idx = rs, re, rw, rh
        wrist_speed_series = right_wrist_speed
    else:
        shoulder_idx, elbow_idx, wrist_idx, hip_idx = ls, le, lw, lh
        wrist_speed_series = left_wrist_speed

    # Release metrics with confidence gating at the release frame.
    def rel_conf_ok(index: int) -> bool:
        c = float(conf[release_frame, index]) if release_frame < conf.shape[0] else float("nan")
        return bool(math.isfinite(c) and c >= DEFAULT_CONFIDENCE_THRESHOLD)

    wrist_speed = float(wrist_speed_series[release_frame]) if release_frame < wrist_speed_series.size else float("nan")
    if not rel_conf_ok(wrist_idx):
        wrist_speed = float("nan")

    wrist_y = float(coords[release_frame, wrist_idx, 1])
    elbow_y = float(coords[release_frame, elbow_idx, 1])
    hand_height = -wrist_y if math.isfinite(wrist_y) else float("nan")
    elbow_height = -elbow_y if math.isfinite(elbow_y) else float("nan")
    if not rel_conf_ok(wrist_idx):
        hand_height = float("nan")
    if not rel_conf_ok(elbow_idx):
        elbow_height = float("nan")

    # Shoulder angle at release: elbow -> shoulder -> ipsilateral hip.
    shoulder_angle = compute_joint_angle(
        arr[release_frame, elbow_idx, :],
        arr[release_frame, shoulder_idx, :],
        arr[release_frame, hip_idx, :],
    )

    # Hip rotation at release: prefer a 3D "pelvis vs shoulders" twist around the
    # trunk axis (more robust to camera direction). Fall back to image-plane hip
    # line angle when trunk-axis math is ill-conditioned.
    hip_rotation = float("nan")
    hip_rotation_method = "none"
    if rel_conf_ok(lh) and rel_conf_ok(rh) and rel_conf_ok(ls) and rel_conf_ok(rs):
        hip_line = coords[release_frame, rh, :] - coords[release_frame, lh, :]
        shoulder_line = coords[release_frame, rs, :] - coords[release_frame, ls, :]
        shoulders_mid = (coords[release_frame, ls, :] + coords[release_frame, rs, :]) / 2.0
        hips_mid = (coords[release_frame, lh, :] + coords[release_frame, rh, :]) / 2.0
        trunk_axis = shoulders_mid - hips_mid
        hip_rotation = abs(_signed_angle_about_axis_deg(hip_line, shoulder_line, trunk_axis))
        hip_rotation_method = "body_axis"

    if not math.isfinite(float(hip_rotation)):
        if rel_conf_ok(lh) and rel_conf_ok(rh):
            hip_rotation = _angle_deg_2d(coords[release_frame, lh, :], coords[release_frame, rh, :])
            hip_rotation_method = "image_plane"
        else:
            hip_rotation = float("nan")
            hip_rotation_method = "none"

    release_metrics = {
        "throwing_side": side,
        "release_frame": int(release_frame),
        "release_timestamp_ms": float(timestamps_ms[release_frame]),
        "wrist_speed": wrist_speed,
        "hand_height": hand_height,
        "elbow_height": elbow_height,
        "shoulder_angle_deg": float(shoulder_angle),
        "hip_rotation_deg": float(hip_rotation),
        "hip_rotation_method": hip_rotation_method,
        "valid": bool(
            math.isfinite(wrist_speed)
            and math.isfinite(hand_height)
            and math.isfinite(elbow_height)
            and math.isfinite(float(shoulder_angle))
            and math.isfinite(float(hip_rotation))
        ),
    }

    # Power chain lag: peak hip (midpoint) -> peak shoulder -> peak wrist (within delivery).
    hip_mid = (coords[:, lh, :] + coords[:, rh, :]) / 2.0
    hip_speed, _ = compute_velocity(hip_mid, timestamps_s)
    shoulder_speed, _ = compute_velocity(coords[:, shoulder_idx, :], timestamps_s)

    def peak_in_segment(series: np.ndarray, seg: slice) -> tuple[int, float]:
        subset = series[seg]
        if subset.size == 0 or not np.isfinite(subset).any():
            if series.size == 0 or not np.isfinite(series).any():
                return -1, float("nan")
            idx = int(np.nanargmax(series))
            return idx, float(series[idx])
        local = int(np.nanargmax(subset))
        idx = int((seg.start or 0) + local)
        return idx, float(series[idx])

    hip_peak_frame, hip_peak_vel = peak_in_segment(hip_speed, delivery_slice)
    shoulder_peak_frame, shoulder_peak_vel = peak_in_segment(shoulder_speed, delivery_slice)
    wrist_peak_frame, wrist_peak_vel = peak_in_segment(wrist_speed_series, delivery_slice)

    def time_ms(idx: int) -> float:
        if idx < 0 or idx >= timestamps_ms.size:
            return float("nan")
        return float(timestamps_ms[idx])

    hip_peak_time = time_ms(hip_peak_frame)
    shoulder_peak_time = time_ms(shoulder_peak_frame)
    wrist_peak_time = time_ms(wrist_peak_frame)

    hip_to_shoulder = shoulder_peak_time - hip_peak_time
    shoulder_to_wrist = wrist_peak_time - shoulder_peak_time
    hip_to_wrist = wrist_peak_time - hip_peak_time

    power_chain_valid = all(
        math.isfinite(v)
        for v in (
            hip_peak_time,
            shoulder_peak_time,
            wrist_peak_time,
            hip_to_shoulder,
            shoulder_to_wrist,
            hip_to_wrist,
        )
    )
    expected_ok = bool(power_chain_valid and 20.0 <= hip_to_shoulder <= 50.0 and 20.0 <= shoulder_to_wrist <= 50.0)

    power_chain = {
        "hip_peak_frame": hip_peak_frame,
        "hip_peak_velocity": hip_peak_vel,
        "shoulder_peak_frame": shoulder_peak_frame,
        "shoulder_peak_velocity": shoulder_peak_vel,
        "wrist_peak_frame": wrist_peak_frame,
        "wrist_peak_velocity": wrist_peak_vel,
        "hip_to_shoulder_ms": float(hip_to_shoulder) if power_chain_valid else float("nan"),
        "shoulder_to_wrist_ms": float(shoulder_to_wrist) if power_chain_valid else float("nan"),
        "hip_to_wrist_ms": float(hip_to_wrist) if power_chain_valid else float("nan"),
        "expected_range_ms": [20.0, 50.0],
        "within_expected_range": expected_ok,
    }

    # Symmetry scores (knee angles + shoulder/knee heights).
    left_knee_angles: list[float] = []
    right_knee_angles: list[float] = []
    for i in range(n_frames):
        left_knee_angles.append(
            compute_joint_angle(arr[i, lh, :], arr[i, lk, :], arr[i, la, :])
        )
        right_knee_angles.append(
            compute_joint_angle(arr[i, rh, :], arr[i, rk, :], arr[i, ra, :])
        )

    left_knee_angles_arr = _as_float_array(left_knee_angles)
    right_knee_angles_arr = _as_float_array(right_knee_angles)
    knee_angle_diff = _mean_abs_diff(left_knee_angles_arr, right_knee_angles_arr)
    knee_angle_score = _score_from_diff(knee_angle_diff, scale=45.0)

    shoulder_height_diff = _mean_abs_diff(coords[:, ls, 1], coords[:, rs, 1])
    knee_height_diff = _mean_abs_diff(coords[:, lk, 1], coords[:, rk, 1])

    torso_len = np.linalg.norm(((coords[:, ls, :] + coords[:, rs, :]) / 2.0) - hip_mid, axis=1)
    torso_scale = float(np.nanmedian(torso_len)) if np.isfinite(torso_len).any() else 1.0

    leg_len = np.linalg.norm(coords[:, lh, :] - coords[:, lk, :], axis=1)
    leg_scale = float(np.nanmedian(leg_len)) if np.isfinite(leg_len).any() else torso_scale

    shoulder_height_score = _score_from_diff(shoulder_height_diff, scale=torso_scale)
    knee_height_score = _score_from_diff(knee_height_diff, scale=leg_scale)

    symmetry = {
        "knee_angle_score": knee_angle_score,
        "shoulder_height_score": shoulder_height_score,
        "knee_height_score": knee_height_score,
        "knee_angle_mean_abs_diff_deg": knee_angle_diff,
        "shoulder_height_mean_abs_diff": shoulder_height_diff,
        "knee_height_mean_abs_diff": knee_height_diff,
    }
    symmetry_valid = all(math.isfinite(float(symmetry[k])) for k in ("knee_angle_score", "shoulder_height_score", "knee_height_score"))

    # Center-of-mass proxy displacement: midpoint between hips and shoulders.
    shoulders_mid = (coords[:, ls, :] + coords[:, rs, :]) / 2.0
    com = (hip_mid + shoulders_mid) / 2.0
    com_displacement = _path_length(com)
    com_valid = bool(np.isfinite(com).all(axis=1).sum() >= 2)

    warnings: list[str] = []
    if not release_metrics["valid"]:
        warnings.append("Release metrics incomplete/low-confidence at release frame.")
    if not power_chain_valid:
        warnings.append("Power chain lag unavailable (insufficient velocity data).")
    elif not expected_ok:
        warnings.append("Power chain lag outside expected 20-50ms window; review timing/phase boundaries.")
    if not symmetry_valid:
        warnings.append("Symmetry metrics incomplete (insufficient bilateral data).")
    if not com_valid:
        warnings.append("COM displacement unreliable (insufficient valid frames).")

    return {
        "throw_id": throw_id,
        "phase_durations": phase_durations,
        "release_metrics": release_metrics,
        "power_chain_lag_ms": float(power_chain.get("hip_to_wrist_ms", float("nan"))),
        "power_chain": power_chain,
        "symmetry": symmetry,
        "com_displacement_m": float(com_displacement),
        "confidence_flags": {
            "has_data": True,
            "release_metrics_valid": bool(release_metrics["valid"]),
            "power_chain_valid": bool(power_chain_valid),
            "symmetry_valid": bool(symmetry_valid),
            "com_displacement_valid": bool(com_valid),
        },
        "warnings": warnings,
    }


__all__ = ["compute_throw_metrics", "PhaseBoundaries"]
