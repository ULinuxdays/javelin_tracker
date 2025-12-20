"""Elite reference profile computation for biomechanics metrics.

This module turns processed elite pose JSONs into a **reference profile** used
as a baseline for comparing developing athletes against elite movement patterns.

It computes a kinematic feature set (angles, velocities, timing, symmetry) per
throw, groups features by throw phase (approach, delivery, release), and then
aggregates across throws into mean/std/min/max summaries.
"""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:  # Optional dependency chain (mediapipe/cv2) may be absent in lightweight envs.
    from javelin_tracker.biomechanics.metrics.phase_detection import detect_throw_phases as _detect_throw_phases
except Exception:  # pragma: no cover - exercised when optional deps missing
    _detect_throw_phases = None  # type: ignore[assignment]

try:  # Prefer project-wide joint mapping when available.
    from javelin_tracker.biomechanics.config import JAVELIN_KEY_JOINTS as _JAVELIN_KEY_JOINTS
except Exception:  # pragma: no cover - exercised when optional deps missing
    _JAVELIN_KEY_JOINTS = {
        "shoulders": (11, 12),
        "elbows": (13, 14),
        "wrists": (15, 16),
        "hips": (23, 24),
        "knees": (25, 26),
    }

JAVELIN_KEY_JOINTS = _JAVELIN_KEY_JOINTS

PHASE_ORDER = ("approach", "delivery", "release")


def _stats(values: List[float], phase: str, unit: str) -> Optional[Dict[str, object]]:
    """Aggregate per-throw values into summary stats.

    Confidence is computed from relative variability:
        confidence = 1 / (1 + std / max(abs(mean), eps))
    """
    if not values:
        return None
    finite = [v for v in values if v is not None and isinstance(v, (int, float)) and math.isfinite(float(v))]
    if not finite:
        return None
    mean_val = float(mean(finite))
    std_val = float(pstdev(finite)) if len(finite) > 1 else 0.0
    scale = max(abs(mean_val), 1e-9)
    variability = std_val / scale if scale > 1e-9 else std_val
    confidence = float(1.0 / (1.0 + variability))
    return {
        "mean": mean_val,
        "std": std_val,
        "min": float(min(finite)),
        "max": float(max(finite)),
        "n_samples": len(finite),
        "phase": phase,
        "unit": unit,
        "confidence": confidence,
    }


def _slug_id_part(value: str) -> str:
    text = (value or "").strip().replace(" ", "_")
    text = re.sub(r"[^A-Za-z0-9_]+", "_", text)
    return text.strip("_") or "unknown"


def _normalize_style_label(value: str | None) -> str:
    text = (value or "").strip().lower()
    return text or "unknown"


def _style_filename_slug(style: str) -> str:
    text = (style or "").strip().lower().replace(" ", "_")
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    return text.strip("_") or "unknown"


def _pose_file_for_metadata_row(row: Mapping[str, str], poses_dir: Path) -> Path:
    thrower = _slug_id_part(row.get("thrower_name", ""))
    throw_number = _slug_id_part(row.get("throw_number", ""))
    return poses_dir / f"{thrower}_{throw_number}.json"


def resolve_style_profile(
    style_name: str | None, style_profiles: Mapping[str, Dict[str, object]]
) -> Dict[str, object]:
    """Return a style profile if available, otherwise fall back to the overall profile."""
    style = _normalize_style_label(style_name)
    if style in style_profiles:
        return dict(style_profiles[style])
    return dict(style_profiles.get("overall", {}))


def _extract_pose_arrays(
    payload: Mapping[str, object],
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        return None

    metadata = payload.get("video_metadata", {}) or {}
    fps = float(metadata.get("fps") or 30.0)
    width = metadata.get("width")
    height = metadata.get("height")
    width_int = int(width) if isinstance(width, (int, float)) and width else None
    height_int = int(height) if isinstance(height, (int, float)) and height else None

    coords: list[np.ndarray] = []
    confs: list[np.ndarray] = []
    valid_flags: list[bool] = []

    for frame in frames:
        if not isinstance(frame, dict):
            continue
        landmarks = frame.get("landmarks")
        if not isinstance(landmarks, list) or len(landmarks) != 33:
            continue
        try:
            coords.append(np.array([[float(p[0]), float(p[1]), float(p[2])] for p in landmarks], dtype=float))
            confs.append(np.array([float(p[3]) for p in landmarks], dtype=float))
            valid_flags.append(bool(frame.get("valid", True)))
        except Exception:
            continue

    if not coords:
        return None

    coords_arr = np.stack(coords, axis=0)

    # If landmarks are normalized (x,y in [0..1]), convert to pixel space to make
    # metrics more consistent across camera zoom and aspect ratios.
    if width_int and height_int and width_int > 0 and height_int > 0:
        try:
            max_xy = float(np.nanmax(np.abs(coords_arr[:, :, :2]))) if coords_arr.size else 0.0
        except Exception:
            max_xy = 0.0
        if math.isfinite(max_xy) and max_xy <= 2.0:
            coords_arr = coords_arr.copy()
            coords_arr[:, :, 0] = coords_arr[:, :, 0] * float(width_int)
            coords_arr[:, :, 1] = coords_arr[:, :, 1] * float(height_int)
            coords_arr[:, :, 2] = coords_arr[:, :, 2] * float(width_int)

    return coords_arr, np.stack(confs, axis=0), np.array(valid_flags, dtype=bool), fps


def _signed_angle_about_axis_deg(v1: np.ndarray, v2: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Signed angle between v1 and v2 around axis (degrees), vectorized over frames."""
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    axis = np.asarray(axis, dtype=float)
    if v1.shape != v2.shape or v1.shape != axis.shape or v1.ndim != 2 or v1.shape[1] != 3:
        raise ValueError("v1/v2/axis must be (n_frames, 3) arrays.")

    axis_norm = np.linalg.norm(axis, axis=1)
    axis_unit = np.where(axis_norm[:, None] > 1e-9, axis / axis_norm[:, None], np.nan)

    # Project v1/v2 onto plane orthogonal to axis.
    v1_dot = np.einsum("ij,ij->i", v1, axis_unit)
    v2_dot = np.einsum("ij,ij->i", v2, axis_unit)
    v1p = v1 - v1_dot[:, None] * axis_unit
    v2p = v2 - v2_dot[:, None] * axis_unit

    v1p_norm = np.linalg.norm(v1p, axis=1)
    v2p_norm = np.linalg.norm(v2p, axis=1)
    ok = (v1p_norm > 1e-9) & (v2p_norm > 1e-9) & np.isfinite(axis_unit).all(axis=1)

    cross = np.cross(v1p, v2p)
    sin = np.einsum("ij,ij->i", cross, axis_unit)
    cos = np.einsum("ij,ij->i", v1p, v2p)
    angle = np.degrees(np.arctan2(sin, cos))
    angle = np.where(ok & np.isfinite(angle), angle, np.nan)
    return angle


def _angle_series_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle at point b for each frame (degrees)."""
    ba = a - b
    bc = c - b
    ba_norm = np.linalg.norm(ba, axis=1)
    bc_norm = np.linalg.norm(bc, axis=1)
    denom = ba_norm * bc_norm
    denom = np.where(denom == 0, np.nan, denom)
    cos_theta = np.einsum("ij,ij->i", ba, bc) / denom
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def _speed_series(coords_arr: np.ndarray, fps: float, index: int) -> np.ndarray:
    traj = coords_arr[:, index, :]
    vel = np.gradient(traj, 1.0 / fps, axis=0)
    return np.linalg.norm(vel, axis=1)


def _line_angle_deg(p_left: np.ndarray, p_right: np.ndarray) -> np.ndarray:
    """Angle of a left->right line in the x/y plane (degrees)."""
    delta = p_right[:, :2] - p_left[:, :2]
    return np.degrees(np.arctan2(delta[:, 1], delta[:, 0]))


def _unwrap_angle_diff_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest signed difference between angles a and b (degrees)."""
    diff = (a - b + 180.0) % 360.0 - 180.0
    return diff


def _select(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    subset = values[mask] if mask.size == values.size else values
    subset = subset[np.isfinite(subset)]
    return subset


def _reduce(values: np.ndarray, mask: np.ndarray, reducer: str) -> Optional[float]:
    subset = _select(values, mask)
    if subset.size == 0:
        return None
    if reducer == "mean":
        return float(np.mean(subset))
    if reducer == "max":
        return float(np.max(subset))
    if reducer == "min":
        return float(np.min(subset))
    raise ValueError(f"Unknown reducer: {reducer}")


def _fallback_phase_indices(n_frames: int) -> Tuple[int, int, int]:
    if n_frames <= 1:
        return 0, 0, 0
    delivery_start = int(round(0.65 * (n_frames - 1)))
    release_frame = int(round(0.90 * (n_frames - 1)))
    return 0, max(0, min(delivery_start, n_frames - 1)), max(0, min(release_frame, n_frames - 1))


def _phase_indices(coords_arr: np.ndarray, fps: float) -> Tuple[int, int, int]:
    n_frames = int(coords_arr.shape[0])
    fallback_approach, fallback_delivery, fallback_release = _fallback_phase_indices(n_frames)
    approach_start, delivery_start, release_frame = fallback_approach, fallback_delivery, fallback_release

    if _detect_throw_phases is not None:
        try:
            res = _detect_throw_phases(coords_arr, fps)
            if getattr(res, "approach_start_frame", None) is not None:
                approach_start = int(res.approach_start_frame)
            if getattr(res, "delivery_start_frame", None) is not None:
                delivery_start = int(res.delivery_start_frame)
            if getattr(res, "release_frame", None) is not None:
                release_frame = int(res.release_frame)
        except Exception:  # pragma: no cover - defensive for optional deps
            pass

    approach_start = max(0, min(int(approach_start), n_frames - 1))
    delivery_start = max(approach_start, min(int(delivery_start), n_frames - 1))
    release_frame = max(delivery_start, min(int(release_frame), n_frames - 1))

    # If phase detection collapses phases (e.g. 0/0/0 on synthetic or low-signal
    # trajectories), fall back to a heuristic split so per-phase metrics exist.
    if n_frames >= 3 and (delivery_start <= approach_start or release_frame <= delivery_start):
        return fallback_approach, fallback_delivery, fallback_release

    return approach_start, delivery_start, release_frame


def _segments(approach_start: int, delivery_start: int, release_frame: int, n_frames: int) -> Dict[str, Tuple[int, int]]:
    return {
        "approach": (approach_start, delivery_start),
        "delivery": (delivery_start, release_frame),
        "release": (release_frame, n_frames),
    }


def _throwing_side(wrist_speed_left: np.ndarray, wrist_speed_right: np.ndarray, release_frame: int) -> str:
    idx = max(0, min(int(release_frame), wrist_speed_left.size - 1))
    left = float(wrist_speed_left[idx]) if wrist_speed_left.size else 0.0
    right = float(wrist_speed_right[idx]) if wrist_speed_right.size else 0.0
    return "right" if right >= left else "left"


def _compute_throw_metrics(payload: Mapping[str, object]) -> Dict[str, Tuple[float, str, str]]:
    extracted = _extract_pose_arrays(payload)
    if extracted is None:
        return {}

    coords_arr, conf_arr, valid_mask, fps = extracted
    n_frames = int(coords_arr.shape[0])
    if n_frames < 3:
        return {}

    ls, rs = JAVELIN_KEY_JOINTS["shoulders"]
    le, re = JAVELIN_KEY_JOINTS["elbows"]
    lw, rw = JAVELIN_KEY_JOINTS["wrists"]
    lh, rh = JAVELIN_KEY_JOINTS["hips"]

    approach_start, delivery_start, release_frame = _phase_indices(coords_arr, fps)
    segments = _segments(approach_start, delivery_start, release_frame, n_frames)

    # Identify throwing side by wrist speed at release.
    wrist_speed_left = _speed_series(coords_arr, fps, lw)
    wrist_speed_right = _speed_series(coords_arr, fps, rw)
    side = _throwing_side(wrist_speed_left, wrist_speed_right, release_frame)

    if side == "right":
        shoulder_idx, elbow_idx, wrist_idx = rs, re, rw
        wrist_speed_throw = wrist_speed_right
        hip_anchor_idx = rh
    else:
        shoulder_idx, elbow_idx, wrist_idx = ls, le, lw
        wrist_speed_throw = wrist_speed_left
        hip_anchor_idx = lh

    elbow_speed_throw = _speed_series(coords_arr, fps, elbow_idx)
    hip_mid = (coords_arr[:, lh, :] + coords_arr[:, rh, :]) / 2.0
    hip_vel = np.gradient(hip_mid, 1.0 / fps, axis=0)
    hip_speed = np.linalg.norm(hip_vel, axis=1)

    # Angles.
    elbow_flexion_throw = _angle_series_deg(
        coords_arr[:, shoulder_idx, :], coords_arr[:, elbow_idx, :], coords_arr[:, wrist_idx, :]
    )
    shoulder_angle_throw = _angle_series_deg(
        coords_arr[:, elbow_idx, :],
        coords_arr[:, shoulder_idx, :],
        coords_arr[:, hip_anchor_idx, :],  # ipsilateral hip as anchor
    )

    # Trunk lean (3D): angle between trunk axis and image-vertical (0=upright).
    #
    # Using 3D reduces camera yaw sensitivity (side vs quarter-front views) compared
    # to x/y-plane-only computations.
    shoulders_mid = (coords_arr[:, ls, :] + coords_arr[:, rs, :]) / 2.0
    hips_mid = hip_mid
    trunk_vec = shoulders_mid - hips_mid
    trunk_norm = np.linalg.norm(trunk_vec, axis=1)
    trunk_norm = np.where(trunk_norm > 1e-9, trunk_norm, np.nan)
    cos_from_vertical = (-trunk_vec[:, 1]) / trunk_norm  # y increases downward in image coords
    cos_from_vertical = np.clip(cos_from_vertical, -1.0, 1.0)
    trunk_lean_deg = np.degrees(np.arccos(cos_from_vertical))

    # Symmetry.
    shoulder_height_asym = np.abs(coords_arr[:, ls, 1] - coords_arr[:, rs, 1])
    elbow_flexion_left = _angle_series_deg(
        coords_arr[:, ls, :], coords_arr[:, le, :], coords_arr[:, lw, :]
    )
    elbow_flexion_right = _angle_series_deg(
        coords_arr[:, rs, :], coords_arr[:, re, :], coords_arr[:, rw, :]
    )
    elbow_flexion_asym = np.abs(elbow_flexion_left - elbow_flexion_right)

    # Hip/shoulder separation (3D twist about the trunk axis). This is more robust
    # to camera direction than comparing 2D line angles in the image plane.
    hip_line = coords_arr[:, rh, :] - coords_arr[:, lh, :]
    shoulder_line = coords_arr[:, rs, :] - coords_arr[:, ls, :]
    separation_signed = _signed_angle_about_axis_deg(hip_line, shoulder_line, trunk_vec)
    separation = np.abs(separation_signed)

    metrics: Dict[str, Tuple[float, str, str]] = {}

    def add_metric(phase: str, name: str, value: Optional[float], unit: str) -> None:
        if value is None:
            return
        if not math.isfinite(float(value)):
            return
        metrics[f"{phase}.{name}"] = (float(value), phase, unit)

    # Overall pose confidence (unitless).
    add_metric("overall", "mean_pose_confidence", float(np.mean(conf_arr)), "unitless")

    # Timing.
    for phase in PHASE_ORDER:
        start, end = segments[phase]
        dur_ms = max(0.0, (end - start) / fps * 1000.0)
        add_metric(phase, "duration_ms", dur_ms, "ms")

    # Release-specific point metrics.
    release_idx = max(0, min(release_frame, n_frames - 1))
    add_metric("release", "throwing_wrist_speed_at_release", float(wrist_speed_throw[release_idx]), "rel_units/s")
    add_metric("release", "throwing_elbow_flexion_deg_at_release", float(elbow_flexion_throw[release_idx]), "degrees")

    # Per-phase aggregates: speeds.
    for phase in PHASE_ORDER:
        start, end = segments[phase]
        if end <= start:
            continue
        mask = valid_mask[start:end]
        add_metric(phase, "throwing_wrist_speed_mean", _reduce(wrist_speed_throw[start:end], mask, "mean"), "rel_units/s")
        add_metric(phase, "throwing_wrist_speed_max", _reduce(wrist_speed_throw[start:end], mask, "max"), "rel_units/s")
        add_metric(phase, "throwing_elbow_speed_mean", _reduce(elbow_speed_throw[start:end], mask, "mean"), "rel_units/s")
        add_metric(phase, "throwing_elbow_speed_max", _reduce(elbow_speed_throw[start:end], mask, "max"), "rel_units/s")
        add_metric(phase, "hip_speed_mean", _reduce(hip_speed[start:end], mask, "mean"), "rel_units/s")
        add_metric(phase, "hip_speed_max", _reduce(hip_speed[start:end], mask, "max"), "rel_units/s")

    # Per-phase aggregates: angles + symmetry.
    for phase in PHASE_ORDER:
        start, end = segments[phase]
        if end <= start:
            continue
        mask = valid_mask[start:end]
        add_metric(
            phase,
            "throwing_elbow_flexion_deg_mean",
            _reduce(elbow_flexion_throw[start:end], mask, "mean"),
            "degrees",
        )
        add_metric(
            phase,
            "throwing_elbow_flexion_deg_min",
            _reduce(elbow_flexion_throw[start:end], mask, "min"),
            "degrees",
        )
        add_metric(
            phase,
            "throwing_shoulder_angle_deg_mean",
            _reduce(shoulder_angle_throw[start:end], mask, "mean"),
            "degrees",
        )
        add_metric(phase, "trunk_lean_deg_mean", _reduce(trunk_lean_deg[start:end], mask, "mean"), "degrees")
        add_metric(
            phase,
            "shoulder_height_asymmetry_mean",
            _reduce(shoulder_height_asym[start:end], mask, "mean"),
            "rel_units",
        )
        add_metric(
            phase,
            "elbow_flexion_asymmetry_deg_mean",
            _reduce(elbow_flexion_asym[start:end], mask, "mean"),
            "degrees",
        )
        add_metric(
            phase,
            "shoulder_hip_separation_deg_mean",
            _reduce(separation[start:end], mask, "mean"),
            "degrees",
        )

    # Record selected side as a unitless metric (0=left, 1=right) for debugging.
    add_metric("overall", "throwing_side_right", 1.0 if side == "right" else 0.0, "unitless")
    return metrics


def compute_throw_metrics(payload: Mapping[str, object]) -> Dict[str, Tuple[float, str, str]]:
    """Compute per-throw kinematic metrics from a processed pose payload.

    Returns a mapping of `metric_key -> (value, phase, unit)` where metric_key is
    phase-prefixed (e.g. `delivery.throwing_wrist_speed_max`).
    """
    return _compute_throw_metrics(payload)


def compute_elite_reference_profile(poses_dir: Path, *, output_path: Path | None = None) -> Dict[str, Dict[str, object]]:
    """Aggregate elite pose JSONs into a reference profile (baseline for comparisons)."""
    poses_dir = Path(poses_dir)
    files = sorted(poses_dir.glob("*.json"))
    aggregates: Dict[str, List[float]] = {}
    phases: Dict[str, str] = {}
    units: Dict[str, str] = {}

    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Skipping %s: %s", path, exc)
            continue

        metrics = _compute_throw_metrics(payload)
        if not metrics:
            continue
        for key, (value, phase, unit) in metrics.items():
            aggregates.setdefault(key, []).append(value)
            phases[key] = phase
            units[key] = unit

    reference: Dict[str, Dict[str, object]] = {}
    for metric, values in aggregates.items():
        stats = _stats(values, phases.get(metric, "overall"), units.get(metric, "unitless"))
        if stats:
            reference[metric] = stats

    default_base = Path(__file__).resolve().parents[3] / "data" / "biomechanics" / "elite_database"
    out_path = Path(output_path) if output_path is not None else (default_base / "reference_profile.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(reference, indent=2), encoding="utf-8")
    logger.info("Elite reference profile saved to %s (%s metrics)", out_path, len(reference))
    return reference


def compute_style_profiles(
    poses_dir: Path, metadata_csv: Path, *, output_dir: Path | None = None
) -> Dict[str, Dict[str, object]]:
    """Compute reference profiles grouped by throwing style, plus overall.

    Style labels are read from the `throwing_style` column in `elite_metadata.csv`.
    Missing/blank values are treated as `unknown`.
    """
    poses_dir = Path(poses_dir)
    metadata_csv = Path(metadata_csv)
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")

    import csv

    rows = list(csv.DictReader(metadata_csv.open("r", encoding="utf-8")))
    style_entries: Dict[str, List[Tuple[Path, str]]] = {}
    for row in rows:
        style = _normalize_style_label(row.get("throwing_style"))
        thrower_name = (row.get("thrower_name") or "").strip()
        pose_file = _pose_file_for_metadata_row(row, poses_dir)
        style_entries.setdefault(style, []).append((pose_file, thrower_name))

    profiles: Dict[str, Dict[str, object]] = {}
    default_base = Path(__file__).resolve().parents[3] / "data" / "biomechanics" / "elite_database"
    output_base = Path(output_dir) if output_dir is not None else default_base

    for style, entries in style_entries.items():
        existing = [(path, name) for path, name in entries if path.exists()]
        if len(existing) < 2:
            logger.warning("Skipping style %s: only %s samples", style, len(existing))
            continue
        files = [path for path, _ in existing]
        ref = compute_elite_reference_profile_from_files(files)
        profiles[style] = {
            "metrics": ref,
            "n_samples": len(files),
            "throwers": sorted({name for _, name in existing if name}),
        }
        out_path = output_base / f"reference_profile_{_style_filename_slug(style)}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(ref, indent=2), encoding="utf-8")
        logger.info("Style reference profile saved to %s (%s metrics)", out_path, len(ref))

    # Overall profile as fallback
    overall_entries: Dict[Path, str] = {}
    for entries in style_entries.values():
        for path, name in entries:
            if path.exists():
                overall_entries.setdefault(path, name)
    overall_files = list(overall_entries.keys())
    overall = compute_elite_reference_profile_from_files(overall_files)
    profiles["overall"] = {
        "metrics": overall,
        "n_samples": len(overall_files),
        "throwers": sorted({name for name in overall_entries.values() if name}),
    }
    out_all = output_base / "reference_profile_overall.json"
    out_all.parent.mkdir(parents=True, exist_ok=True)
    out_all.write_text(json.dumps(overall, indent=2), encoding="utf-8")
    return profiles


def compute_elite_reference_profile_from_files(files: List[Path]) -> Dict[str, Dict[str, object]]:
    aggregates: Dict[str, List[float]] = {}
    phases: Dict[str, str] = {}
    units: Dict[str, str] = {}

    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Skipping %s: %s", path, exc)
            continue
        metrics = _compute_throw_metrics(payload)
        if not metrics:
            continue
        for key, (value, phase, unit) in metrics.items():
            aggregates.setdefault(key, []).append(value)
            phases[key] = phase
            units[key] = unit

    reference: Dict[str, Dict[str, object]] = {}
    for metric, values in aggregates.items():
        stats = _stats(values, phases.get(metric, "overall"), units.get(metric, "unitless"))
        if stats:
            reference[metric] = stats
    return reference


def get_reference_value(metric_name: str, phase: str, reference_path: Path | None = None) -> Optional[Dict[str, float]]:
    """Retrieve mean/std for a metric/phase from the reference profile."""
    ref_path = reference_path or (
        Path(__file__).resolve().parents[3] / "data" / "biomechanics" / "elite_database" / "reference_profile_overall.json"
    )
    if not ref_path.exists():
        return None
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    candidates = [f"{phase}.{metric_name}", metric_name]
    for key in candidates:
        metric = ref.get(key)
        if metric and metric.get("phase") == phase:
            return {"mean": metric.get("mean"), "std": metric.get("std")}
    return None


__all__ = ["compute_elite_reference_profile", "get_reference_value", "resolve_style_profile"]
__all__ += ["compute_style_profiles", "compute_elite_reference_profile_from_files"]
__all__ += ["compute_throw_metrics"]
