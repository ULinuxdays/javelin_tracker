"""End-to-end metrics pipeline for processed pose JSON payloads.

This module consumes pose JSON emitted by `PosePipeline` (frames + video_metadata)
and computes:
- throw phases (approach, delivery, release)
- joint angle time series
- joint velocity time series
- throw-level summary metrics (release metrics, power-chain lag, symmetry, COM proxy)
- normalization context (for cross-video comparisons)

The pipeline is designed to be usable in lightweight environments that do *not*
have optional video/pose dependencies installed (e.g., OpenCV/MediaPipe). When
those dependencies are unavailable, phase detection and pose-quality validation
fall back to simple heuristics and the pipeline still produces metrics.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .angles import compute_trajectory_angles
from .kinematics import compute_joint_velocities, compute_velocity_profiles_dataframe
from .normalization import normalize_athlete_metrics
from .throw_metrics import compute_throw_metrics


DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_VALID_FRAME_THRESHOLD = 0.8


logger = logging.getLogger(__name__)

# We focus plots/metrics on the high-value "engaged" window around the throw
# (late delivery → release → brief follow-through) to avoid long run-up clips
# dominating time series charts.
DEFAULT_ENGAGED_PRE_SECONDS = 0.25
DEFAULT_ENGAGED_POST_SECONDS = 0.50


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_sessions_dir() -> Path:
    return _project_root() / "data" / "biomechanics" / "sessions"


def _slug(value: str) -> str:
    text = (value or "").strip().replace(" ", "_")
    out = []
    for ch in text:
        if ch.isalnum() or ch in {"_", "-"}:
            out.append(ch)
        else:
            out.append("_")
    cleaned = "".join(out).strip("_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned or "session"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any) -> Optional[float]:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(num):
        return None
    return float(num)


def _json_safe(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.generic):
        return _json_safe(value.item())

    if isinstance(value, pd.DataFrame):
        records = value.to_dict(orient="records")
        return [_json_safe(rec) for rec in records]

    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())

    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]

    # dataclasses (e.g., NormalizationContext)
    try:
        return _json_safe(asdict(value))  # type: ignore[arg-type]
    except Exception:
        pass

    return str(value)


def _load_pose_payload(pose_data_json: Any) -> tuple[dict[str, Any], Optional[Path]]:
    if isinstance(pose_data_json, Mapping):
        return dict(pose_data_json), None

    path = Path(pose_data_json)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("pose_data_json must decode to a JSON object.")
    return payload, path


def _extract_fps(payload: Mapping[str, Any]) -> float:
    video_meta = payload.get("video_metadata")
    if isinstance(video_meta, Mapping):
        fps = _safe_float(video_meta.get("fps"))
        if fps:
            return fps
    fps = _safe_float(payload.get("fps"))
    return float(fps) if fps else 30.0


def _extract_session_id(payload: Mapping[str, Any], source_path: Optional[Path]) -> str:
    for key in ("session_id", "video_id", "throw_id", "id"):
        raw = payload.get(key)
        if raw:
            return _slug(str(raw))
    if source_path is not None:
        return _slug(source_path.stem)
    return "session"


def _frame_landmarks(frame: Any) -> Optional[Sequence[Any]]:
    if isinstance(frame, Mapping):
        landmarks = frame.get("landmarks")
        if isinstance(landmarks, Sequence):
            return landmarks
        return None
    if isinstance(frame, Sequence):
        return frame
    return None


def _coerce_landmark_point(point: Any) -> tuple[float, float, float, float]:
    try:
        arr = np.asarray(point, dtype=float).reshape(-1)
    except Exception:
        return (math.nan, math.nan, math.nan, math.nan)

    if arr.size == 4:
        x, y, z, c = arr.tolist()
        return (float(x), float(y), float(z), float(c))
    if arr.size == 3:
        x, y, z = arr.tolist()
        return (float(x), float(y), float(z), 1.0)
    if arr.size == 2:
        x, y = arr.tolist()
        return (float(x), float(y), 0.0, 1.0)
    return (math.nan, math.nan, math.nan, math.nan)


def _extract_pose_arrays(
    frames: Sequence[Any],
    fps: float,
    *,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    valid_frame_threshold: float = DEFAULT_VALID_FRAME_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return coords (n,33,3), conf (n,33), timestamps_ms (n,), valid_frames (n,)."""
    n = len(frames)
    coords = np.full((n, 33, 3), np.nan, dtype=float)
    conf = np.full((n, 33), np.nan, dtype=float)
    timestamps_ms = np.full((n,), np.nan, dtype=float)
    valid = np.zeros((n,), dtype=bool)

    dt_ms = 1000.0 / float(fps or 30.0)
    for idx, frame in enumerate(frames):
        frame_idx = idx
        if isinstance(frame, Mapping):
            frame_idx = int(frame.get("frame_idx", idx))
            timestamps_ms[idx] = float(frame.get("timestamp_ms", frame_idx * dt_ms))
        else:
            timestamps_ms[idx] = float(frame_idx * dt_ms)

        landmarks = _frame_landmarks(frame)
        if not landmarks or len(landmarks) != 33:
            continue

        pts = [_coerce_landmark_point(p) for p in landmarks]
        arr = np.asarray(pts, dtype=float)  # (33,4)
        coords[idx, :, :] = arr[:, :3]
        conf[idx, :] = arr[:, 3]

        # Respect upstream valid flag when present; otherwise infer.
        if isinstance(frame, Mapping) and "valid" in frame:
            valid[idx] = bool(frame.get("valid"))
        else:
            finite = np.isfinite(conf[idx, :])
            present_ratio = float(np.mean(finite & (conf[idx, :] >= confidence_threshold))) if finite.any() else 0.0
            valid[idx] = present_ratio >= float(valid_frame_threshold)

    # Mask low-confidence landmarks.
    low = ~(np.isfinite(conf) & (conf >= float(confidence_threshold)))
    coords = coords.copy()
    coords[low] = np.nan
    return coords, conf, timestamps_ms, valid


def _fallback_phase_boundaries(n_frames: int) -> dict[str, int]:
    if n_frames <= 1:
        return {"approach_start_frame": 0, "delivery_start_frame": 0, "release_frame": 0}
    delivery = int(round(0.65 * (n_frames - 1)))
    release = int(round(0.90 * (n_frames - 1)))
    return {
        "approach_start_frame": 0,
        "delivery_start_frame": max(0, min(delivery, n_frames - 1)),
        "release_frame": max(0, min(release, n_frames - 1)),
    }


def _try_detect_phases(coords: np.ndarray, fps: float) -> tuple[dict[str, int], dict[str, object]]:
    """Best-effort phase detection; falls back when unavailable."""
    n_frames = int(coords.shape[0])
    fallback = _fallback_phase_boundaries(n_frames)
    details: dict[str, object] = {"method": "fallback", "confidences": {}, "notes": []}

    try:
        from javelin_tracker.biomechanics.metrics import detect_throw_phases

        result = detect_throw_phases(coords, fps=float(fps))
        approach = getattr(result, "approach_start_frame", None)
        delivery = getattr(result, "delivery_start_frame", None)
        release = getattr(result, "release_frame", None)

        boundaries = {
            "approach_start_frame": int(approach) if approach is not None else fallback["approach_start_frame"],
            "delivery_start_frame": int(delivery) if delivery is not None else fallback["delivery_start_frame"],
            "release_frame": int(release) if release is not None else fallback["release_frame"],
        }
        # Clamp monotonic order.
        boundaries["approach_start_frame"] = max(0, min(boundaries["approach_start_frame"], n_frames - 1))
        boundaries["delivery_start_frame"] = max(
            boundaries["approach_start_frame"], min(boundaries["delivery_start_frame"], n_frames - 1)
        )
        boundaries["release_frame"] = max(
            boundaries["delivery_start_frame"], min(boundaries["release_frame"], n_frames - 1)
        )
        details["method"] = "detect_throw_phases"
        details["confidences"] = getattr(result, "confidences", {}) or {}
        return boundaries, details
    except Exception as exc:
        details["notes"] = [f"Phase detection unavailable; using fallback. ({exc})"]
        return fallback, details


def _analysis_window_from_phases(
    phase_boundaries: Mapping[str, object],
    *,
    n_frames: int,
    fps: float,
    timestamps_ms: np.ndarray,
) -> dict[str, object]:
    if n_frames <= 0:
        return {
            "start_frame": 0,
            "end_frame": 0,
            "start_timestamp_ms": 0.0,
            "end_timestamp_ms": 0.0,
            "method": "empty",
        }

    release = phase_boundaries.get("release_frame")
    delivery = phase_boundaries.get("delivery_start_frame")
    try:
        release_frame = int(release) if release is not None else n_frames - 1
    except Exception:
        release_frame = n_frames - 1
    release_frame = max(0, min(release_frame, n_frames - 1))

    # If delivery is missing, use a 1s window before release.
    try:
        delivery_frame = int(delivery) if delivery is not None else max(0, release_frame - int(round(float(fps) * 1.0)))
    except Exception:
        delivery_frame = max(0, release_frame - int(round(float(fps) * 1.0)))
    delivery_frame = max(0, min(delivery_frame, release_frame))

    pre = int(round(float(fps) * DEFAULT_ENGAGED_PRE_SECONDS))
    post = int(round(float(fps) * DEFAULT_ENGAGED_POST_SECONDS))

    start_frame = max(0, delivery_frame - pre)
    end_frame = min(n_frames - 1, release_frame + post)

    start_ts = float(timestamps_ms[start_frame]) if start_frame < timestamps_ms.size else float(start_frame / fps * 1000.0)
    end_ts = float(timestamps_ms[end_frame]) if end_frame < timestamps_ms.size else float(end_frame / fps * 1000.0)

    return {
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "start_timestamp_ms": start_ts,
        "end_timestamp_ms": end_ts,
        "method": "delivery_release_window",
        "pre_seconds": float(DEFAULT_ENGAGED_PRE_SECONDS),
        "post_seconds": float(DEFAULT_ENGAGED_POST_SECONDS),
        "notes": "Angles/kinematics are filtered to this window to avoid long run-ups dominating charts.",
    }


def _maybe_convert_frames_to_pixels(
    frames: Sequence[object],
    video_metadata: Mapping[str, object],
) -> list[object]:
    """Convert normalized landmark coordinates to pixel coordinates when possible.

    Pose pipelines typically emit normalized coords (x,y in [0..1], z in width
    units). Downstream metrics (especially velocities/COM displacement) are more
    consistent across camera angles and resolutions when computed in pixel space.
    """
    width = video_metadata.get("width")
    height = video_metadata.get("height")
    if not isinstance(width, (int, float)) or not isinstance(height, (int, float)):
        return list(frames)
    width_int = int(width)
    height_int = int(height)
    if width_int <= 0 or height_int <= 0:
        return list(frames)

    # Heuristic: if coords look normalized (mostly <=2), convert to pixels.
    max_xy = 0.0
    checked = 0
    for frame in frames[: min(30, len(frames))]:
        if not isinstance(frame, Mapping):
            continue
        landmarks = frame.get("landmarks")
        if not isinstance(landmarks, Sequence) or len(landmarks) != 33:
            continue
        for p in landmarks:
            try:
                x = float(p[0])
                y = float(p[1])
            except Exception:
                continue
            if math.isfinite(x):
                max_xy = max(max_xy, abs(x))
            if math.isfinite(y):
                max_xy = max(max_xy, abs(y))
        checked += 1
        if checked >= 6:
            break
    if not math.isfinite(max_xy) or max_xy > 2.0:
        return list(frames)

    converted: list[object] = []
    for frame in frames:
        if not isinstance(frame, Mapping):
            converted.append(frame)
            continue
        landmarks = frame.get("landmarks")
        if not isinstance(landmarks, Sequence) or len(landmarks) != 33:
            converted.append(dict(frame))
            continue
        new_landmarks: list[tuple[float, float, float, float]] = []
        for p in landmarks:
            try:
                x = float(p[0]) * float(width_int)
                y = float(p[1]) * float(height_int)
                z = float(p[2]) * float(width_int) if len(p) > 2 else 0.0
                c = float(p[3]) if len(p) > 3 else 1.0
            except Exception:
                x, y, z, c = 0.0, 0.0, 0.0, 0.0
            new_landmarks.append((x, y, z, c))
        converted.append({**dict(frame), "landmarks": new_landmarks})
    return converted


def _pose_quality_summary(
    conf: np.ndarray,
    valid_frames: np.ndarray,
    *,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict[str, object]:
    finite = np.isfinite(conf)
    mean_conf = float(np.nanmean(conf[finite])) if finite.any() else float("nan")
    valid_ratio = float(np.mean(valid_frames)) if valid_frames.size else 0.0
    present_ratio = float(np.mean(finite & (conf >= float(confidence_threshold)))) if finite.any() else 0.0
    return {
        "mean_landmark_confidence": mean_conf,
        "valid_frame_ratio": valid_ratio,
        "landmark_presence_ratio": present_ratio,
    }


def _try_pose_quality_validation(payload: Mapping[str, Any]) -> tuple[Optional[Mapping[str, Any]], Optional[str]]:
    """Try to run validate_pose_quality (optional); returns (quality, error)."""
    try:
        from javelin_tracker.biomechanics.utils.validation import validate_pose_quality

        frames = payload.get("frames", [])
        video_meta = payload.get("video_metadata", {}) or {}
        if not isinstance(frames, list) or not isinstance(video_meta, Mapping):
            return None, "Invalid payload shape for validate_pose_quality."
        quality = validate_pose_quality(frames, dict(video_meta))
        return quality if isinstance(quality, Mapping) else None, None
    except Exception as exc:
        return None, str(exc)


def _angles_summary(df: pd.DataFrame) -> dict[str, object]:
    if df.empty:
        return {"n_samples": 0, "valid_ratio": 0.0, "per_angle_valid_ratio": {}}
    valid = df.get("valid")
    valid_ratio = float(valid.mean()) if isinstance(valid, pd.Series) else 0.0
    per_angle: dict[str, float] = {}
    if "angle_name" in df.columns and "valid" in df.columns:
        for name, group in df.groupby("angle_name"):
            per_angle[str(name)] = float(group["valid"].astype(bool).mean()) if not group.empty else 0.0
    return {"n_samples": int(df.shape[0]), "valid_ratio": valid_ratio, "per_angle_valid_ratio": per_angle}


def _kinematics_summary(df: pd.DataFrame) -> dict[str, object]:
    if df.empty:
        return {"n_samples": 0, "valid_ratio": 0.0, "per_joint_valid_ratio": {}}
    valid = df.get("valid")
    valid_ratio = float(valid.mean()) if isinstance(valid, pd.Series) else 0.0
    per_joint: dict[str, float] = {}
    if "joint" in df.columns and "valid" in df.columns:
        for name, group in df.groupby("joint"):
            per_joint[str(name)] = float(group["valid"].astype(bool).mean()) if not group.empty else 0.0
    return {"n_samples": int(df.shape[0]), "valid_ratio": valid_ratio, "per_joint_valid_ratio": per_joint}


class MetricsPipeline:
    """Compute and persist a full metrics bundle for a single pose payload."""

    def __init__(
        self,
        *,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        valid_frame_threshold: float = DEFAULT_VALID_FRAME_THRESHOLD,
        log: Optional[logging.Logger] = None,
    ) -> None:
        self.confidence_threshold = float(confidence_threshold)
        self.valid_frame_threshold = float(valid_frame_threshold)
        self.log = log or logger

    def compute_metrics(
        self,
        pose_data_json: Any,
        output_dir: Any,
        *,
        progress_callback: Callable[[float], None] | None = None,
    ) -> dict[str, object]:
        """Compute metrics and write `metrics.json` under output_dir/session_id/.

        Args:
            pose_data_json: dict payload or path to a pose JSON file.
            output_dir: base directory where session subfolders are written.

        Returns:
            Processing stats including output path.
        """
        def _progress(value: float) -> None:
            if progress_callback is None:
                return
            try:
                progress_callback(float(value))
            except Exception:
                return

        _progress(0.0)

        payload, source_path = _load_pose_payload(pose_data_json)
        fps = _extract_fps(payload)
        session_id = _extract_session_id(payload, source_path)

        base_dir = Path(output_dir) if output_dir is not None else _default_sessions_dir()
        session_dir = base_dir / session_id
        output_path = session_dir / "metrics.json"
        session_dir.mkdir(parents=True, exist_ok=True)

        self.log.info("Computing metrics for session=%s fps=%.2f", session_id, fps)

        frames = payload.get("frames", [])
        if not isinstance(frames, Sequence):
            frames = []
        video_meta = payload.get("video_metadata", {}) or {}
        frames_metrics = _maybe_convert_frames_to_pixels(frames, video_meta) if frames else []

        coords, conf, timestamps_ms, valid_frames = _extract_pose_arrays(
            frames_metrics,
            fps,
            confidence_threshold=self.confidence_threshold,
            valid_frame_threshold=self.valid_frame_threshold,
        )
        pose_summary = _pose_quality_summary(conf, valid_frames, confidence_threshold=self.confidence_threshold)
        _progress(0.12)

        phase_boundaries, phase_details = _try_detect_phases(coords, fps)
        self.log.info(
            "Phase boundaries (method=%s): approach=%s delivery=%s release=%s",
            phase_details.get("method"),
            phase_boundaries.get("approach_start_frame"),
            phase_boundaries.get("delivery_start_frame"),
            phase_boundaries.get("release_frame"),
        )
        analysis_window = _analysis_window_from_phases(
            phase_boundaries,
            n_frames=int(coords.shape[0]),
            fps=float(fps),
            timestamps_ms=timestamps_ms,
        )
        _progress(0.20)

        computed: dict[str, bool] = {"phases": True}
        errors: list[str] = []
        unreliable: list[str] = []

        angles_df = pd.DataFrame(columns=["frame", "timestamp_ms", "angle_name", "value_degrees", "valid"])
        kinematics_df = pd.DataFrame(
            columns=["frame", "timestamp_ms", "joint_index", "joint", "vx", "vy", "vz", "speed", "valid"]
        )
        joint_velocity_summary: dict[str, object] = {}
        throw_metrics: dict[str, object] = {}
        normalization: dict[str, object] = {}

        try:
            self.log.info("Computing joint angles...")
            angles_df = compute_trajectory_angles(frames_metrics, fps=float(fps))
            if not angles_df.empty:
                start_f = int(analysis_window.get("start_frame", 0))
                end_f = int(analysis_window.get("end_frame", int(coords.shape[0]) - 1))
                angles_df = angles_df[(angles_df["frame"] >= start_f) & (angles_df["frame"] <= end_f)]
            computed["angles"] = True
            _progress(0.40)
        except Exception as exc:
            computed["angles"] = False
            errors.append(f"angles: {exc}")
            self.log.exception("Angle computation failed: %s", exc)
            _progress(0.40)

        try:
            self.log.info("Computing joint velocities...")
            kinematics_df = compute_velocity_profiles_dataframe(
                frames_metrics, fps=float(fps), confidence_threshold=self.confidence_threshold
            )
            if not kinematics_df.empty:
                start_f = int(analysis_window.get("start_frame", 0))
                end_f = int(analysis_window.get("end_frame", int(coords.shape[0]) - 1))
                kinematics_df = kinematics_df[(kinematics_df["frame"] >= start_f) & (kinematics_df["frame"] <= end_f)]
            joint_velocity_summary = compute_joint_velocities(
                frames_metrics,
                fps=float(fps),
                confidence_threshold=self.confidence_threshold,
                release_frame=int(phase_boundaries.get("release_frame", 0)),
            )
            computed["kinematics"] = True
            _progress(0.65)
        except Exception as exc:
            computed["kinematics"] = False
            errors.append(f"kinematics: {exc}")
            self.log.exception("Kinematics computation failed: %s", exc)
            _progress(0.65)

        try:
            self.log.info("Computing throw summary metrics...")
            throw_metrics_payload = dict(payload)
            throw_metrics_payload["frames"] = frames_metrics
            throw_metrics = compute_throw_metrics(throw_metrics_payload, phase_boundaries, fps=float(fps))
            computed["throw_metrics"] = True
            _progress(0.80)
        except Exception as exc:
            computed["throw_metrics"] = False
            errors.append(f"throw_metrics: {exc}")
            self.log.exception("Throw metrics computation failed: %s", exc)
            _progress(0.80)

        try:
            self.log.info("Normalizing metrics for cross-video comparisons...")
            norm = normalize_athlete_metrics(
                {"frames": frames_metrics, "angles_df": angles_df, "kinematics_df": kinematics_df},
                video_meta or {"fps": fps},
            )
            ctx = norm.get("context")
            athlete_height_m: object = None
            if isinstance(ctx, Mapping):
                athlete_height_m = ctx.get("athlete_height_m")
            elif ctx is not None:
                athlete_height_m = getattr(ctx, "athlete_height_m", None)
            normalization = {
                "context": _json_safe(ctx) if ctx is not None else None,
                "velocity_unit": "m/s" if athlete_height_m else "body_heights/s",
            }
            normalized_angles_df = norm.get("normalized_angles_df")
            if isinstance(normalized_angles_df, pd.DataFrame):
                normalization["normalized_angles"] = normalized_angles_df
            normalized_kin_df = norm.get("normalized_kinematics_df")
            if isinstance(normalized_kin_df, pd.DataFrame):
                normalization["normalized_kinematics"] = normalized_kin_df
            computed["normalization"] = True
            _progress(0.93)
        except Exception as exc:
            computed["normalization"] = False
            errors.append(f"normalization: {exc}")
            self.log.exception("Normalization failed: %s", exc)
            _progress(0.93)

        quality, quality_error = _try_pose_quality_validation(payload)
        if quality is None and quality_error:
            self.log.debug("validate_pose_quality unavailable: %s", quality_error)

        # Lightweight reliability flags.
        if not math.isfinite(float(pose_summary.get("mean_landmark_confidence") or float("nan"))):
            unreliable.append("pose_confidence_unknown")
        if float(pose_summary.get("valid_frame_ratio") or 0.0) < float(self.valid_frame_threshold):
            unreliable.append("pose_low_valid_frame_ratio")

        angle_summary = _angles_summary(angles_df)
        kin_summary = _kinematics_summary(kinematics_df)
        if computed.get("angles") and float(angle_summary.get("valid_ratio") or 0.0) < 0.5:
            unreliable.append("angles_low_valid_ratio")
        if computed.get("kinematics") and float(kin_summary.get("valid_ratio") or 0.0) < 0.5:
            unreliable.append("kinematics_low_valid_ratio")

        throw_flags = {}
        if isinstance(throw_metrics, Mapping):
            throw_flags = throw_metrics.get("confidence_flags") if isinstance(throw_metrics.get("confidence_flags"), Mapping) else {}
        if throw_flags and not bool(throw_flags.get("release_metrics_valid", True)):
            unreliable.append("release_metrics_unreliable")

        stats = {
            "computed": computed,
            "errors": errors,
            "flagged_unreliable": sorted(set(unreliable)),
        }

        output_payload: dict[str, object] = {
            "schema_version": 1,
            "session_id": session_id,
            "video_id": payload.get("video_id"),
            "source_pose_path": str(source_path) if source_path else None,
            "generated_at": _now_iso(),
            "video_metadata": payload.get("video_metadata", {}),
            "pose_summary": pose_summary,
            "phase_boundaries": phase_boundaries,
            "phase_detection": phase_details,
            "analysis_window": analysis_window,
            "angles": {"data": angles_df, "summary": angle_summary},
            "kinematics": {
                "data": kinematics_df,
                "summary": kin_summary,
                "joint_velocity_summary": joint_velocity_summary,
            },
            "throw_metrics": throw_metrics,
            "normalization": normalization,
            "validation": {
                "pose_quality": quality,
                "pose_quality_error": quality_error if quality is None else None,
            },
            "processing_stats": stats,
        }

        output_path.write_text(json.dumps(_json_safe(output_payload), indent=2, allow_nan=False), encoding="utf-8")
        self.log.info("Wrote metrics: %s", output_path)
        _progress(1.0)
        return {"status": "success", "session_id": session_id, "output_path": str(output_path), **stats}
