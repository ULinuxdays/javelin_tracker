"""Filtering utilities for smoothing pose landmark trajectories.

Savitzky–Golay filtering reduces jitter while preserving sharp throwing
movements such as the release phase.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


def _ensure_valid_window(window_length: int, polyorder: int, min_len: int) -> int:
    if window_length <= polyorder:
        raise ValueError("window_length must be greater than polyorder.")
    if window_length % 2 == 0:
        raise ValueError("window_length must be odd.")
    if min_len <= 0:
        raise ValueError("Trajectory length must be positive.")
    if min_len < window_length:
        # Expand to next odd length greater than min_len if needed.
        window_length = min_len if min_len % 2 == 1 else min_len + 1
    return window_length


def _pad_for_window(data: np.ndarray, target_len: int) -> np.ndarray:
    pad_total = max(0, target_len - data.shape[0])
    if pad_total == 0:
        return data
    # Split padding across start/end, pad with edge values.
    pad_before = pad_total // 2
    pad_after = pad_total - pad_before
    pad_width = [(pad_before, pad_after)] + [(0, 0) for _ in range(data.ndim - 1)]
    return np.pad(data, pad_width=pad_width, mode="edge")


def fill_low_confidence_landmarks(
    landmarks_array: np.ndarray,
    confidence: np.ndarray,
    *,
    confidence_threshold: float = 0.5,
) -> np.ndarray:
    """Fill low-confidence landmark coordinates via temporal interpolation.

    This improves limb tracking stability when a joint temporarily drops out
    (often returning (0,0,0) with confidence ~0). Only coordinates are modified;
    callers should preserve original confidence values.
    """
    if landmarks_array.ndim != 3 or landmarks_array.shape[1:] != (33, 3):
        raise ValueError("landmarks_array must have shape (n_frames, 33, 3).")

    n_frames = int(landmarks_array.shape[0])
    if n_frames == 0:
        return landmarks_array

    conf = np.asarray(confidence, dtype=float)
    if conf.ndim == 1:
        conf = np.broadcast_to(conf[:, None], (n_frames, 33))
    if conf.shape != (n_frames, 33):
        raise ValueError("confidence must be shape (n_frames,) or (n_frames, 33).")

    out = landmarks_array.astype(float, copy=True)
    x = np.arange(n_frames, dtype=float)
    thresh = float(confidence_threshold)

    for joint_idx in range(33):
        good = np.isfinite(conf[:, joint_idx]) & (conf[:, joint_idx] >= thresh)
        if not np.any(good):
            continue

        for axis in range(3):
            series = out[:, joint_idx, axis].astype(float, copy=True)
            series[~good] = np.nan
            valid = np.isfinite(series)
            if not np.any(valid):
                continue
            if int(np.sum(valid)) == 1:
                out[:, joint_idx, axis] = float(series[valid][0])
            else:
                out[:, joint_idx, axis] = np.interp(x, x[valid], series[valid])

    return out


def bone_length_outlier_mask(
    landmarks_array: np.ndarray,
    confidence: np.ndarray,
    *,
    confidence_threshold: float = 0.5,
    bones: Optional[Sequence[tuple[int, int]]] = None,
    ratio_range: tuple[float, float] = (0.60, 1.40),
    min_samples: int = 10,
) -> np.ndarray:
    """Flag joints whose bone lengths deviate strongly from typical values.

    This heuristic improves stability when an occluded limb causes the pose model
    to "snap" the wrist/elbow to an unrelated location. Even when the model
    reports high confidence, these snaps often violate bone-length consistency.

    Returns:
        Boolean mask of shape (n_frames, 33) where True indicates the joint
        should be treated as unreliable/missing for that frame.
    """
    if landmarks_array.ndim != 3 or landmarks_array.shape[1:] != (33, 3):
        raise ValueError("landmarks_array must have shape (n_frames, 33, 3).")

    n_frames = int(landmarks_array.shape[0])
    if n_frames == 0:
        return np.zeros((0, 33), dtype=bool)

    conf = np.asarray(confidence, dtype=float)
    if conf.ndim == 1:
        conf = np.broadcast_to(conf[:, None], (n_frames, 33))
    if conf.shape != (n_frames, 33):
        raise ValueError("confidence must be shape (n_frames,) or (n_frames, 33).")

    default_bones = (
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
        (23, 25),
        (25, 27),
        (24, 26),
        (26, 28),
    )
    bone_pairs = list(bones) if bones is not None else list(default_bones)

    ratio_lo, ratio_hi = ratio_range
    ratio_lo = float(ratio_lo)
    ratio_hi = float(ratio_hi)
    if ratio_lo <= 0 or ratio_hi <= 0 or ratio_hi <= ratio_lo:
        raise ValueError("ratio_range must be positive and increasing, e.g. (0.6, 1.4).")

    out = np.zeros((n_frames, 33), dtype=bool)
    thresh = float(confidence_threshold)
    coords = landmarks_array.astype(float, copy=False)

    for proximal, distal in bone_pairs:
        if not (0 <= proximal < 33 and 0 <= distal < 33):
            continue
        ok = (
            np.isfinite(conf[:, proximal])
            & np.isfinite(conf[:, distal])
            & (conf[:, proximal] >= thresh)
            & (conf[:, distal] >= thresh)
        )
        if not np.any(ok):
            continue
        diffs = coords[:, distal, :] - coords[:, proximal, :]
        lengths = np.linalg.norm(diffs, axis=1)
        valid_lengths = lengths[ok & np.isfinite(lengths)]
        if int(valid_lengths.size) < int(min_samples):
            continue
        median = float(np.median(valid_lengths))
        if not np.isfinite(median) or median <= 0:
            continue
        bad = ok & ((lengths < median * ratio_lo) | (lengths > median * ratio_hi))
        out[bad, distal] = True

    return out


def joint_displacement_outlier_mask(
    landmarks_array: np.ndarray,
    confidence: np.ndarray,
    *,
    confidence_threshold: float = 0.5,
    joint_indices: Optional[Sequence[int]] = None,
    iqr_factor: float = 8.0,
    absolute_threshold: float = 0.18,
    min_samples: int = 12,
) -> np.ndarray:
    """Flag joints whose per-frame displacement is implausibly large.

    This catches the common failure mode where a landmark "snaps" to a random
    background pixel (or the javelin) while keeping bone lengths plausible.

    Returns:
        Boolean mask of shape (n_frames, 33) where True indicates the joint
        should be treated as unreliable/missing for that frame.
    """
    if landmarks_array.ndim != 3 or landmarks_array.shape[1:] != (33, 3):
        raise ValueError("landmarks_array must have shape (n_frames, 33, 3).")

    n_frames = int(landmarks_array.shape[0])
    if n_frames == 0:
        return np.zeros((0, 33), dtype=bool)

    conf = np.asarray(confidence, dtype=float)
    if conf.ndim == 1:
        conf = np.broadcast_to(conf[:, None], (n_frames, 33))
    if conf.shape != (n_frames, 33):
        raise ValueError("confidence must be shape (n_frames,) or (n_frames, 33).")

    use_joints = list(joint_indices) if joint_indices is not None else list(range(33))
    thresh = float(confidence_threshold)
    coords = landmarks_array.astype(float, copy=False)
    out = np.zeros((n_frames, 33), dtype=bool)

    iqr_factor = float(max(0.0, iqr_factor))
    absolute_threshold = float(max(0.0, absolute_threshold))
    min_samples = int(max(4, min_samples))

    for joint_idx in use_joints:
        if not (0 <= int(joint_idx) < 33):
            continue
        c = conf[:, int(joint_idx)]
        z = coords[:, int(joint_idx), :]
        valid = np.isfinite(z).all(axis=1) & np.isfinite(c) & (c >= thresh)
        if int(np.sum(valid)) < min_samples:
            continue
        diffs = z[1:, :] - z[:-1, :]
        disp = np.linalg.norm(diffs, axis=1)
        valid_pairs = valid[1:] & valid[:-1] & np.isfinite(disp)
        samples = disp[valid_pairs]
        if int(samples.size) < min_samples:
            continue
        q25, q75 = np.quantile(samples, [0.25, 0.75])
        iqr = float(q75 - q25)
        med = float(np.median(samples))
        thr = max(absolute_threshold, med + iqr_factor * iqr)
        if not np.isfinite(thr) or thr <= 0:
            continue
        bad = np.zeros((n_frames,), dtype=bool)
        bad[1:] = valid_pairs & (disp > thr)
        out[bad, int(joint_idx)] = True

    return out


def kalman_smooth_landmarks(
    landmarks_array: np.ndarray,
    confidence: np.ndarray,
    *,
    fps: float,
    confidence_threshold: float = 0.5,
    process_noise: float = 1e-4,
    measurement_noise: float = 5e-4,
    mahalanobis_threshold: float = 25.0,
) -> np.ndarray:
    """Kalman-smooth landmark trajectories with confidence-aware gating.

    Uses a constant-velocity Kalman filter per joint and rejects implausible
    jumps via a Mahalanobis-distance gate. This is effective for occlusion
    cases (e.g., arm behind head) where pose models can latch onto the wrong
    pixels.
    """
    if landmarks_array.ndim != 3 or landmarks_array.shape[1:] != (33, 3):
        raise ValueError("landmarks_array must have shape (n_frames, 33, 3).")

    n_frames = int(landmarks_array.shape[0])
    if n_frames == 0:
        return landmarks_array

    conf = np.asarray(confidence, dtype=float)
    if conf.ndim == 1:
        conf = np.broadcast_to(conf[:, None], (n_frames, 33))
    if conf.shape != (n_frames, 33):
        raise ValueError("confidence must be shape (n_frames,) or (n_frames, 33).")

    dt = 1.0 / float(fps or 30.0)
    # State: [x,y,z,vx,vy,vz]
    A = np.array(
        [
            [1.0, 0.0, 0.0, dt, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, dt, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, dt],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    H = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    I6 = np.eye(6, dtype=float)
    q = float(process_noise)
    Q = np.diag([q, q, q, q * 10.0, q * 10.0, q * 10.0]).astype(float)
    r0 = float(measurement_noise)
    maha_thresh = float(mahalanobis_threshold)
    thresh = float(confidence_threshold)

    out = np.full_like(landmarks_array, np.nan, dtype=float)
    coords = landmarks_array.astype(float, copy=False)

    for joint_idx in range(33):
        z_all = coords[:, joint_idx, :]
        c_all = conf[:, joint_idx]
        valid = np.isfinite(z_all).all(axis=1) & np.isfinite(c_all)
        if not np.any(valid):
            continue
        first = int(np.argmax(valid))
        x = np.zeros((6,), dtype=float)
        x[:3] = z_all[first]
        P = np.eye(6, dtype=float) * 1.0

        for t in range(n_frames):
            x = A @ x
            P = A @ P @ A.T + Q

            z = z_all[t]
            c = float(c_all[t]) if np.isfinite(c_all[t]) else 0.0
            if not np.all(np.isfinite(z)):
                out[t, joint_idx, :] = x[:3]
                continue

            # Confidence -> measurement noise (low conf => high noise).
            c_eff = max(0.05, min(1.0, c))
            if c < thresh:
                c_eff = min(c_eff, 0.25)
            r = r0 / max(c_eff, 1e-3)
            R = np.eye(3, dtype=float) * r

            y = z - (H @ x)
            S = H @ P @ H.T + R
            try:
                sol = np.linalg.solve(S, y)
                maha = float(y.T @ sol)
            except Exception:
                maha = float("inf")

            if np.isfinite(maha) and maha <= maha_thresh:
                K = P @ H.T @ np.linalg.inv(S)
                x = x + K @ y
                P = (I6 - (K @ H)) @ P

            out[t, joint_idx, :] = x[:3]

    missing = ~np.isfinite(out).all(axis=2)
    if np.any(missing):
        out = out.copy()
        out[missing] = coords[missing]

    return out


def smooth_landmarks(
    landmarks_array: np.ndarray,
    window_length: int = 5,
    polyorder: int = 2,
    confidence: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Apply Savitzky–Golay smoothing to pose landmarks over time.

    Args:
        landmarks_array: Shape (n_frames, 33, 3) array of (x, y, z) coords.
        window_length: Odd window size for smoothing.
        polyorder: Polynomial order for the filter.
        confidence: Optional per-landmark confidence weights; shape (n_frames,)
            or (n_frames, 33). Values in [0, 1]; lower confidence blends more
            of the original unsmoothed signal back in.

    Returns:
        Smoothed array with the same shape as landmarks_array.

    Edge handling: if n_frames < window_length, pads using edge values to allow
    filtering without shrinking the series.
    """
    if landmarks_array.ndim != 3 or landmarks_array.shape[1:] != (33, 3):
        raise ValueError("landmarks_array must have shape (n_frames, 33, 3).")

    n_frames = landmarks_array.shape[0]
    window_length = _ensure_valid_window(window_length, polyorder, n_frames)
    padded = _pad_for_window(landmarks_array, window_length)

    # Apply filter on each coordinate axis independently across frames.
    smoothed = savgol_filter(padded, window_length=window_length, polyorder=polyorder, axis=0, mode="interp")
    smoothed = smoothed[:n_frames]

    if confidence is not None:
        conf = np.asarray(confidence, dtype=float)
        if conf.ndim == 1:
            conf = np.broadcast_to(conf[:, None], (n_frames, 33))
        if conf.shape != (n_frames, 33):
            raise ValueError("confidence must be shape (n_frames,) or (n_frames, 33).")
        weights = np.clip(conf, 0.0, 1.0)[..., None]
        smoothed = weights * smoothed + (1.0 - weights) * landmarks_array

    return smoothed


def smooth_trajectory(trajectory_2d: np.ndarray, window_length: int = 5, polyorder: int = 2) -> np.ndarray:
    """Smooth a single joint trajectory (x, y) over time."""
    if trajectory_2d.ndim != 2 or trajectory_2d.shape[1] not in (2, 3):
        raise ValueError("trajectory_2d must have shape (n_frames, 2) or (n_frames, 3).")
    n_frames = trajectory_2d.shape[0]
    window_length = _ensure_valid_window(window_length, polyorder, n_frames)
    padded = _pad_for_window(trajectory_2d, window_length)
    smoothed = savgol_filter(padded, window_length=window_length, polyorder=polyorder, axis=0, mode="interp")
    return smoothed[:n_frames]


def plot_trajectory_overlay(
    original: np.ndarray,
    smoothed: np.ndarray,
    title: str = "Trajectory smoothing",
    labels: Tuple[str, str] = ("x", "y"),
) -> plt.Figure:
    """Plot original vs smoothed trajectories for quick visual debugging."""
    if original.shape[1] < 2 or smoothed.shape[1] < 2:
        raise ValueError("original and smoothed must have at least two columns (x, y).")
    fig, ax = plt.subplots()
    ax.plot(original[:, 0], original[:, 1], "o-", alpha=0.4, label="original")
    ax.plot(smoothed[:, 0], smoothed[:, 1], "o-", alpha=0.8, label="smoothed")
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig
