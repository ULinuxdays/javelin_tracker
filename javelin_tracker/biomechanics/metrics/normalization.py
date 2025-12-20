"""Normalization utilities for comparing biomechanics metrics across videos.

The core challenge for cross-video comparisons is that videos vary in:
- athlete size / camera distance (scale)
- frame rate (timing)

This module normalizes:
- positions to *body-height units* (dimensionless)
- velocities to *meters/second* when an athlete height in meters is available,
  otherwise to *body-heights/second* (still comparable across videos)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_TARGET_FPS = 30.0
DEFAULT_CONFIDENCE_THRESHOLD = 0.5

DEFAULT_JAVELIN_KEY_JOINTS: dict[str, tuple[int, int]] = {
    "shoulders": (11, 12),
    "elbows": (13, 14),
    "wrists": (15, 16),
    "hips": (23, 24),
    "knees": (25, 26),
}
DEFAULT_ANKLES = (27, 28)


def normalize_angles(angles_df: pd.DataFrame) -> pd.DataFrame:
    """Angles are already in degrees; returns a cleaned copy."""
    df = angles_df.copy()
    if "value_degrees" in df.columns:
        df["value_degrees"] = pd.to_numeric(df["value_degrees"], errors="coerce")
    if "timestamp_ms" in df.columns:
        df["timestamp_ms"] = pd.to_numeric(df["timestamp_ms"], errors="coerce")
    return df


def _as_landmarks_array(landmarks: Any) -> np.ndarray:
    """Return (n_frames, 33, 4) array (x,y,z,conf)."""
    if isinstance(landmarks, np.ndarray):
        arr = np.asarray(landmarks, dtype=float)
        if arr.ndim == 2:
            arr = arr[None, ...]
        if arr.ndim != 3 or arr.shape[1] != 33:
            raise ValueError("Expected landmarks shape (n_frames, 33, C)")
        if arr.shape[2] == 4:
            return arr
        if arr.shape[2] == 3:
            conf = np.ones((arr.shape[0], 33, 1), dtype=float)
            return np.concatenate([arr, conf], axis=2)
        if arr.shape[2] == 2:
            z = np.zeros((arr.shape[0], 33, 1), dtype=float)
            conf = np.ones((arr.shape[0], 33, 1), dtype=float)
            return np.concatenate([arr, z, conf], axis=2)
        raise ValueError("Expected landmarks channels 2, 3, or 4.")

    if isinstance(landmarks, Mapping) and "frames" in landmarks:
        frames = landmarks["frames"]
    else:
        frames = landmarks

    if not isinstance(frames, Sequence):
        raise ValueError("Unsupported landmarks format; expected ndarray or sequence of frames.")

    rows: list[np.ndarray] = []
    for frame in frames:
        if isinstance(frame, Mapping) and "landmarks" in frame:
            raw = frame["landmarks"]
        else:
            raw = frame
        if not isinstance(raw, Sequence) or len(raw) != 33:
            continue
        try:
            lm = np.asarray(raw, dtype=float)
        except Exception:
            continue
        if lm.ndim != 2 or lm.shape[0] != 33:
            continue
        if lm.shape[1] == 4:
            rows.append(lm)
        elif lm.shape[1] == 3:
            conf = np.ones((33, 1), dtype=float)
            rows.append(np.concatenate([lm, conf], axis=1))
        elif lm.shape[1] == 2:
            z = np.zeros((33, 1), dtype=float)
            conf = np.ones((33, 1), dtype=float)
            rows.append(np.concatenate([lm, z, conf], axis=1))

    if not rows:
        return np.zeros((0, 33, 4), dtype=float)
    return np.stack(rows, axis=0)


def normalize_positions(landmarks: Any, body_height_px: float) -> np.ndarray:
    """Normalize landmark coordinates to body-height units.

    Args:
        landmarks: frames/array of (x,y,z[,conf]) in *pixel coordinates*.
        body_height_px: athlete body height measured in pixels (scale factor).

    Returns:
        (n_frames, 33, 4) landmarks with x/y/z scaled by body_height_px.
    """
    arr = _as_landmarks_array(landmarks).copy()
    scale = float(body_height_px)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("body_height_px must be a positive finite number")
    arr[:, :, :3] = arr[:, :, :3] / scale
    return arr


def denormalize_positions(normalized_landmarks: Any, body_height_px: float) -> np.ndarray:
    """Inverse of normalize_positions (body-height units -> pixels)."""
    arr = _as_landmarks_array(normalized_landmarks).copy()
    scale = float(body_height_px)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("body_height_px must be a positive finite number")
    arr[:, :, :3] = arr[:, :, :3] * scale
    return arr


def normalize_velocities(
    velocities: Any,
    body_height_px: float,
    fps: float,
    *,
    input_unit: str = "per_second",
    athlete_height_m: float | None = None,
) -> Any:
    """Normalize velocities to meters/second (or body-heights/second).

    If `athlete_height_m` is provided, scale is:
        (px/s) * (athlete_height_m / body_height_px) -> m/s

    Otherwise, output is in body-heights/second (still cross-video comparable):
        (px/s) / body_height_px -> body_heights/s

    input_unit:
        - "per_second": velocities are in px/s already
        - "per_frame": velocities are in px/frame and will be multiplied by fps
    """
    scale_px = float(body_height_px)
    if not math.isfinite(scale_px) or scale_px <= 0:
        raise ValueError("body_height_px must be a positive finite number")

    fps_val = float(fps) if fps else DEFAULT_TARGET_FPS
    if not math.isfinite(fps_val) or fps_val <= 0:
        raise ValueError("fps must be a positive finite number")

    unit = (input_unit or "per_second").strip().lower()
    if unit not in {"per_second", "per_frame"}:
        raise ValueError("input_unit must be 'per_second' or 'per_frame'")

    per_second_factor = fps_val if unit == "per_frame" else 1.0
    meter_factor = float(athlete_height_m) if athlete_height_m is not None else 1.0
    factor = per_second_factor * (meter_factor / scale_px)

    if isinstance(velocities, pd.DataFrame):
        df = velocities.copy()
        for col in ("vx", "vy", "vz", "speed"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce") * factor
        return df

    if isinstance(velocities, Mapping):
        out: dict[str, object] = {}
        for key, value in velocities.items():
            if key in {"velocities", "vx", "vy", "vz", "speed"} and isinstance(value, Sequence):
                arr = np.asarray(value, dtype=float) * factor
                out[key] = [float(v) for v in arr.tolist()]
            else:
                out[key] = value
        return out

    arr = np.asarray(velocities, dtype=float)
    return arr * factor


def denormalize_velocities(
    velocities: Any,
    body_height_px: float,
    fps: float,
    *,
    output_unit: str = "per_second",
    athlete_height_m: float | None = None,
) -> Any:
    """Inverse of normalize_velocities (m/s -> px/s or px/frame).

    output_unit:
        - "per_second": return px/s
        - "per_frame": return px/frame
    """
    scale_px = float(body_height_px)
    if not math.isfinite(scale_px) or scale_px <= 0:
        raise ValueError("body_height_px must be a positive finite number")

    fps_val = float(fps) if fps else DEFAULT_TARGET_FPS
    if not math.isfinite(fps_val) or fps_val <= 0:
        raise ValueError("fps must be a positive finite number")

    unit = (output_unit or "per_second").strip().lower()
    if unit not in {"per_second", "per_frame"}:
        raise ValueError("output_unit must be 'per_second' or 'per_frame'")

    meter_factor = float(athlete_height_m) if athlete_height_m is not None else 1.0
    factor = scale_px / max(meter_factor, 1e-9)
    per_frame_factor = 1.0 / fps_val if unit == "per_frame" else 1.0
    total_factor = factor * per_frame_factor

    if isinstance(velocities, pd.DataFrame):
        df = velocities.copy()
        for col in ("vx", "vy", "vz", "speed"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce") * total_factor
        return df

    if isinstance(velocities, Mapping):
        out: dict[str, object] = {}
        for key, value in velocities.items():
            if key in {"velocities", "vx", "vy", "vz", "speed"} and isinstance(value, Sequence):
                arr = np.asarray(value, dtype=float) * total_factor
                out[key] = [float(v) for v in arr.tolist()]
            else:
                out[key] = value
        return out

    arr = np.asarray(velocities, dtype=float)
    return arr * total_factor


def handle_variable_fps(
    metrics: pd.DataFrame,
    original_fps: float,
    target_fps: float = DEFAULT_TARGET_FPS,
) -> pd.DataFrame:
    """Resample a long-form metrics DataFrame to a target FPS.

    Expects a DataFrame with at least:
      - either timestamp_ms, or frame + original_fps

    Resampling is performed per-group when a name column exists:
      - angle_name for angles, joint for kinematics, otherwise whole DataFrame.
    """
    df = metrics.copy()
    orig_fps = float(original_fps) if original_fps else DEFAULT_TARGET_FPS
    tgt_fps = float(target_fps) if target_fps else DEFAULT_TARGET_FPS

    if "timestamp_ms" not in df.columns:
        if "frame" not in df.columns:
            raise ValueError("metrics must include timestamp_ms or frame")
        df["timestamp_ms"] = pd.to_numeric(df["frame"], errors="coerce") * (1000.0 / orig_fps)

    df["timestamp_ms"] = pd.to_numeric(df["timestamp_ms"], errors="coerce")
    df = df.dropna(subset=["timestamp_ms"])
    if df.empty or math.isclose(orig_fps, tgt_fps):
        return df

    dt_ms = 1000.0 / tgt_fps
    t_min = float(df["timestamp_ms"].min())
    t_max = float(df["timestamp_ms"].max())
    grid = np.arange(t_min, t_max + 0.5 * dt_ms, dt_ms, dtype=float)

    group_cols: list[str] = []
    for candidate in ("angle_name", "joint"):
        if candidate in df.columns:
            group_cols.append(candidate)
            break

    exclude = {"frame", "timestamp_ms"} | set(group_cols)
    numeric_cols = [c for c in df.columns if c not in exclude]
    numeric_cols = [c for c in numeric_cols if pd.api.types.is_numeric_dtype(df[c]) or c == "valid"]

    records: list[dict[str, object]] = []
    if group_cols:
        groups = df.groupby(group_cols, dropna=False)
    else:
        groups = [((), df)]

    for key, group in groups:
        group = group.sort_values("timestamp_ms")
        src_t = group["timestamp_ms"].to_numpy(dtype=float)
        for i, t in enumerate(grid):
            rec: dict[str, object] = {"timestamp_ms": float(t), "frame": int(round((t - t_min) / dt_ms))}
            if group_cols:
                if not isinstance(key, tuple):
                    key = (key,)
                for col, val in zip(group_cols, key):
                    rec[col] = val

            valid_any = False
            for col in numeric_cols:
                if col == "valid":
                    continue
                src_y = pd.to_numeric(group[col], errors="coerce").to_numpy(dtype=float)
                finite = np.isfinite(src_y) & np.isfinite(src_t)
                if finite.sum() < 2:
                    rec[col] = float("nan")
                    continue
                rec[col] = float(np.interp(t, src_t[finite], src_y[finite]))
                valid_any = valid_any or math.isfinite(float(rec[col]))
            if "valid" in group.columns:
                rec["valid"] = bool(valid_any)
            records.append(rec)

    out = pd.DataFrame.from_records(records)
    return out.sort_values(group_cols + ["frame"] if group_cols else ["frame"]).reset_index(drop=True)


@dataclass(frozen=True)
class NormalizationContext:
    body_height_px: float
    athlete_height_m: Optional[float]
    fps: float
    width_px: Optional[int]
    height_px: Optional[int]


def _extract_athlete_height_m(video_metadata: Mapping[str, object]) -> Optional[float]:
    for key in ("athlete_height_m", "body_height_m"):
        value = video_metadata.get(key)
        if value is None or value == "":
            continue
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(val) and val > 0:
            return val

    value_cm = video_metadata.get("athlete_height_cm")
    if value_cm:
        try:
            cm = float(value_cm)
            if math.isfinite(cm) and cm > 0:
                return cm / 100.0
        except (TypeError, ValueError):
            pass
    return None


def _convert_landmarks_to_pixels(landmarks: np.ndarray, width_px: int, height_px: int) -> np.ndarray:
    out = landmarks.copy()
    if out.size == 0:
        return out
    out[:, :, 0] = out[:, :, 0] * float(width_px)
    out[:, :, 1] = out[:, :, 1] * float(height_px)
    out[:, :, 2] = out[:, :, 2] * float(width_px)
    return out


def _estimate_body_height_px(landmarks_px: np.ndarray) -> float:
    if landmarks_px.size == 0:
        return float("nan")
    lh, rh = DEFAULT_JAVELIN_KEY_JOINTS["hips"]
    ls, rs = DEFAULT_JAVELIN_KEY_JOINTS["shoulders"]
    la, ra = DEFAULT_ANKLES

    coords = landmarks_px[:, :, :3]
    conf = landmarks_px[:, :, 3]

    valid = np.isfinite(coords[:, :, :2]).all(axis=2) & (conf >= DEFAULT_CONFIDENCE_THRESHOLD)
    shoulders_mid_y = np.where(valid[:, [ls, rs]].all(axis=1), (coords[:, ls, 1] + coords[:, rs, 1]) / 2.0, np.nan)
    ankles_mid_y = np.where(valid[:, [la, ra]].all(axis=1), (coords[:, la, 1] + coords[:, ra, 1]) / 2.0, np.nan)

    diffs = np.abs(ankles_mid_y - shoulders_mid_y)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return float("nan")
    return float(np.median(diffs))


def normalize_athlete_metrics(raw_metrics: Mapping[str, object], video_metadata: Mapping[str, object]) -> Dict[str, object]:
    """Orchestrate normalization for a pose payload and associated metrics.

    Expected inputs:
      - raw_metrics may include a pose payload under `frames` and optionally
        precomputed `angles_df` / `kinematics_df` DataFrames.
      - video_metadata should include fps and optionally width/height.

    Returns a dict with:
      - context: NormalizationContext
      - normalized_landmarks: (n_frames,33,4) in body-height units
      - normalized_angles_df / normalized_kinematics_df when provided
    """
    fps = float(video_metadata.get("fps") or raw_metrics.get("fps") or DEFAULT_TARGET_FPS)
    width_px = video_metadata.get("width")
    height_px = video_metadata.get("height")
    width_int = int(width_px) if isinstance(width_px, (int, float)) and width_px else None
    height_int = int(height_px) if isinstance(height_px, (int, float)) and height_px else None

    athlete_height_m = _extract_athlete_height_m(video_metadata)

    if "frames" in raw_metrics or "landmarks" in raw_metrics:
        landmarks_source: Any = raw_metrics
    elif "pose" in raw_metrics:
        landmarks_source = raw_metrics["pose"]
    else:
        raise ValueError("raw_metrics must include pose landmarks under 'frames', 'landmarks', or 'pose'.")

    landmarks_arr = _as_landmarks_array(landmarks_source)
    landmarks_px = landmarks_arr
    if width_int and height_int:
        # Heuristic: if coords look normalized (mostly <=2), convert to pixels.
        max_xy = np.nanmax(np.abs(landmarks_arr[:, :, :2])) if landmarks_arr.size else 0.0
        if max_xy <= 2.0:
            landmarks_px = _convert_landmarks_to_pixels(landmarks_arr, width_int, height_int)

    body_height_px = float(video_metadata.get("body_height_px") or _estimate_body_height_px(landmarks_px))
    if not math.isfinite(body_height_px) or body_height_px <= 0:
        body_height_px = float(height_int or 1.0)

    ctx = NormalizationContext(
        body_height_px=body_height_px,
        athlete_height_m=athlete_height_m,
        fps=fps,
        width_px=width_int,
        height_px=height_int,
    )

    normalized_landmarks = normalize_positions(landmarks_px, ctx.body_height_px)

    out: Dict[str, object] = {"context": ctx, "normalized_landmarks": normalized_landmarks}

    angles_df = raw_metrics.get("angles_df")
    if isinstance(angles_df, pd.DataFrame):
        out["normalized_angles_df"] = normalize_angles(angles_df)

    kin_df = raw_metrics.get("kinematics_df")
    if isinstance(kin_df, pd.DataFrame):
        out["normalized_kinematics_df"] = normalize_velocities(
            kin_df,
            ctx.body_height_px,
            ctx.fps,
            input_unit="per_second",
            athlete_height_m=ctx.athlete_height_m,
        )

    return out


__all__ = [
    "normalize_angles",
    "normalize_positions",
    "normalize_velocities",
    "handle_variable_fps",
    "normalize_athlete_metrics",
    "denormalize_positions",
    "denormalize_velocities",
    "NormalizationContext",
]
