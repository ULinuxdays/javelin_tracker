#!/usr/bin/env python3
"""Biomechanics reproduction harness for tracking bugs.

This script is designed to make pose-tracking bugs reproducible and reviewable
by enforcing:
- a stable debug-assets folder structure
- a naming convention for clips
- consistent pipeline settings
- exported overlay frames for visual proof

Usage:
  .venv/bin/python scripts/biomech_repro.py --help
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEBUG_ROOT = REPO_ROOT / "debug_assets"
VIDEOS_DIR = DEBUG_ROOT / "videos"
META_DIR = DEBUG_ROOT / "metadata"
RUNS_DIR = DEBUG_ROOT / "runs"

POSE_EDGES: list[tuple[int, int]] = [
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (23, 24),
    (11, 23),
    (12, 24),
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _slug(text: str) -> str:
    s = (text or "").strip().lower().replace(" ", "_")
    s = re.sub(r"[^a-z0-9_]+", "_", s).strip("_")
    return s or "unknown"


def _ensure_dirs() -> None:
    for p in (DEBUG_ROOT, VIDEOS_DIR, META_DIR, RUNS_DIR):
        p.mkdir(parents=True, exist_ok=True)


def _require_cv2() -> Any:
    try:
        import cv2  # type: ignore

        return cv2
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "OpenCV is required for this command. Install dependencies: `pip install -r requirements.txt`."
        ) from exc


def _video_metadata(path: Path) -> dict[str, Any]:
    # Prefer project helper to keep behavior aligned with the app.
    try:
        from javelin_tracker.biomechanics.utils.video import get_video_metadata

        return dict(get_video_metadata(path))
    except Exception:
        cv2 = _require_cv2()
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {path}")
        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            dur = float(frames / fps) if fps > 0 and frames > 0 else 0.0
            return {"width": width, "height": height, "fps": fps, "total_frames": frames, "duration_seconds": dur}
        finally:
            cap.release()


def _estimate_vfr(path: Path, *, sample_frames: int = 60) -> dict[str, Any]:
    """Best-effort variable-FPS detection using CAP_PROP_POS_MSEC deltas."""
    cv2 = _require_cv2()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"checked": False, "reason": "unopened"}
    try:
        ts: list[float] = []
        read = 0
        while read < int(sample_frames):
            ok = cap.grab()
            if not ok:
                break
            read += 1
            ms = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            if np.isfinite(ms) and ms > 0:
                ts.append(ms)
        if len(ts) < 6:
            return {"checked": True, "vfr_suspected": False, "reason": "insufficient_timestamps", "n": len(ts)}
        ts_arr = np.asarray(ts, dtype=float)
        dt = np.diff(ts_arr)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if dt.size < 4:
            return {"checked": True, "vfr_suspected": False, "reason": "insufficient_deltas", "n": int(dt.size)}
        cv = float(np.std(dt) / max(1e-9, float(np.mean(dt))))
        # Very loose heuristic: CV above ~0.08 usually indicates timestamps drifting.
        return {
            "checked": True,
            "vfr_suspected": bool(cv >= 0.08),
            "delta_ms_mean": float(np.mean(dt)),
            "delta_ms_std": float(np.std(dt)),
            "delta_ms_cv": cv,
            "n": int(dt.size),
        }
    finally:
        cap.release()


def _default_clip_name(
    *,
    clip_type: str,
    version: int,
    date_yyyymmdd: str,
    height: int,
    fps: float,
    view: str,
    camera: str,
) -> str:
    fps_int = int(round(float(fps))) if fps and float(fps) > 0 else 0
    return f"debug_{clip_type}_v{version}_{date_yyyymmdd}_{height}p_{fps_int}fps_{_slug(view)}_{_slug(camera)}.mp4"


def _find_next_version(pattern_prefix: str, *, max_search: int = 50) -> int:
    # Find next version by scanning existing files in VIDEOS_DIR.
    existing = {p.name for p in VIDEOS_DIR.glob(f"{pattern_prefix}_v*_*.mp4")}
    used: set[int] = set()
    for name in existing:
        m = re.search(r"_v(\d+)_", name)
        if not m:
            continue
        try:
            used.add(int(m.group(1)))
        except Exception:
            continue
    for v in range(1, max_search + 1):
        if v not in used:
            return v
    return max_search + 1


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def cmd_register(args: argparse.Namespace) -> None:
    _ensure_dirs()
    src = Path(args.source).expanduser().resolve()
    if not src.exists():
        raise SystemExit(f"Source video not found: {src}")
    if not src.is_file():
        raise SystemExit(f"Expected a file, got: {src}")

    meta = _video_metadata(src)
    date_str = args.date or datetime.now(timezone.utc).strftime("%Y%m%d")
    clip_type = (args.clip_type or "good").strip().lower()
    if clip_type not in {"good", "hard"}:
        raise SystemExit("--clip-type must be 'good' or 'hard'")

    prefix = f"debug_{clip_type}"
    version = int(args.version) if args.version else _find_next_version(prefix)
    out_name = _default_clip_name(
        clip_type=clip_type,
        version=version,
        date_yyyymmdd=date_str,
        height=int(meta.get("height") or 0),
        fps=float(meta.get("fps") or 0.0),
        view=args.view or "unknown",
        camera=args.camera or "unknown",
    )
    dest = VIDEOS_DIR / out_name
    if dest.exists() and not args.force:
        raise SystemExit(f"Destination already exists: {dest} (use --force or bump --version)")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)

    sha = _sha256(dest)
    vfr = _estimate_vfr(dest)
    payload = {
        "clip_type": clip_type,
        "version": version,
        "registered_at": _now_iso(),
        "source_path": str(src),
        "sha256": sha,
        "video_metadata": meta,
        "vfr_check": vfr,
        "labels": {
            "view": args.view or "unknown",
            "camera": args.camera or "unknown",
            "lighting": args.lighting or "",
            "people": args.people or "",
            "notes": args.notes or "",
        },
    }
    meta_path = META_DIR / f"{dest.stem}.json"
    _write_json(meta_path, payload)
    print(f"Registered: {dest}")
    print(f"Metadata:  {meta_path}")
    if vfr.get("vfr_suspected"):
        print("WARNING: variable-FPS suspected (timestamps not stable). Prefer constant-FPS exports for reproducibility.")


def _check_range(name: str, value: float, lo: float, hi: float) -> tuple[bool, str]:
    ok = (value >= lo) and (value <= hi)
    return ok, f"{name}={value} (expected {lo}..{hi})"


def cmd_validate(args: argparse.Namespace) -> None:
    _ensure_dirs()
    path = Path(args.video).expanduser()
    if not path.exists():
        raise SystemExit(f"Video not found: {path}")
    meta = _video_metadata(path)
    vfr = _estimate_vfr(path)

    width = int(meta.get("width") or 0)
    height = int(meta.get("height") or 0)
    fps = float(meta.get("fps") or 0.0)
    duration = float(meta.get("duration_seconds") or 0.0)

    checks: list[tuple[bool, str]] = []
    checks.append((width >= 1280 or height >= 720, f"resolution={width}x{height} (min 720p)"))
    checks.append(_check_range("fps", fps, 30.0, 240.0))
    checks.append(_check_range("duration_seconds", duration, 5.0, 12.0))
    checks.append((not bool(vfr.get("vfr_suspected")), f"variable_fps_suspected={vfr.get('vfr_suspected')}"))

    ok_all = all(ok for ok, _ in checks)
    print(f"Video: {path}")
    print(json.dumps({"video_metadata": meta, "vfr_check": vfr}, indent=2))
    print("\nChecks:")
    for ok, msg in checks:
        status = "OK" if ok else "FAIL"
        print(f"- {status}: {msg}")
    if not ok_all:
        raise SystemExit(2)


@dataclass(frozen=True)
class _Frame:
    frame_idx: int
    timestamp_ms: float
    image_bgr: Any


def _iter_frames_by_index(video_path: Path, indices: Sequence[int]) -> Iterable[_Frame]:
    cv2 = _require_cv2()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    try:
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            ts = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            yield _Frame(frame_idx=int(idx), timestamp_ms=ts, image_bgr=frame)
    finally:
        cap.release()


def _to_px(lm: Sequence[float], width: int, height: int) -> tuple[int, int, float]:
    x, y = float(lm[0]), float(lm[1])
    c = float(lm[3]) if len(lm) > 3 else 1.0
    px = int(round(x * width)) if 0.0 <= x <= 1.0 else int(round(x))
    py = int(round(y * height)) if 0.0 <= y <= 1.0 else int(round(y))
    return px, py, c


def _draw_overlay(
    frame_bgr: Any,
    landmarks: Sequence[Sequence[float]],
    *,
    min_conf: float = 0.25,
) -> Any:
    cv2 = _require_cv2()
    out = frame_bgr.copy()
    h, w = int(out.shape[0]), int(out.shape[1])
    pts = [_to_px(lm, w, h) for lm in landmarks]

    for a, b in POSE_EDGES:
        if not (0 <= a < len(pts) and 0 <= b < len(pts)):
            continue
        x1, y1, c1 = pts[a]
        x2, y2, c2 = pts[b]
        conf = float(min(c1, c2))
        if not np.isfinite(conf) or conf < float(min_conf):
            continue
        alpha = float(min(0.98, 0.15 + 0.83 * max(0.0, min(1.0, conf))))
        color = (int(246 * alpha), int(130 * alpha), int(59 * alpha))  # BGR-ish tint
        cv2.line(out, (x1, y1), (x2, y2), color=color, thickness=2, lineType=cv2.LINE_AA)

    for x, y, c in pts:
        if not np.isfinite(c) or float(c) < float(min_conf):
            continue
        alpha = float(min(0.98, 0.15 + 0.83 * max(0.0, min(1.0, float(c)))))
        color = (int(42 * alpha), int(23 * alpha), int(15 * alpha))
        cv2.circle(out, (int(x), int(y)), radius=3, color=color, thickness=-1, lineType=cv2.LINE_AA)

    return out


def _draw_skeleton_only(
    frame_shape: tuple[int, int],
    landmarks: Sequence[Sequence[float]],
    *,
    min_conf: float = 0.25,
) -> Any:
    cv2 = _require_cv2()
    h, w = int(frame_shape[0]), int(frame_shape[1])
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[:, :] = (32, 18, 11)  # dark background
    pts = [_to_px(lm, w, h) for lm in landmarks]

    for a, b in POSE_EDGES:
        if not (0 <= a < len(pts) and 0 <= b < len(pts)):
            continue
        x1, y1, c1 = pts[a]
        x2, y2, c2 = pts[b]
        conf = float(min(c1, c2))
        if not np.isfinite(conf) or conf < float(min_conf):
            continue
        alpha = float(min(0.98, 0.18 + 0.80 * max(0.0, min(1.0, conf))))
        color = (int(238 * alpha), int(211 * alpha), int(34 * alpha))
        cv2.line(out, (x1, y1), (x2, y2), color=color, thickness=2, lineType=cv2.LINE_AA)

    for x, y, c in pts:
        if not np.isfinite(c) or float(c) < float(min_conf):
            continue
        alpha = float(min(0.98, 0.18 + 0.80 * max(0.0, min(1.0, float(c)))))
        color = (int(252 * alpha), int(250 * alpha), int(248 * alpha))
        cv2.circle(out, (int(x), int(y)), radius=3, color=color, thickness=-1, lineType=cv2.LINE_AA)

    return out


def _candidate_frames_for_export(pose_payload: dict[str, Any], *, max_frames: int = 24) -> list[int]:
    frames = pose_payload.get("frames")
    if not isinstance(frames, list) or not frames:
        return []

    # Prefer engaged window from metrics if available.
    analysis_window = None
    try:
        analysis_window = pose_payload.get("analysis_window")  # not present in pose payload; metrics has it
    except Exception:
        analysis_window = None

    # Fallback: use last ~1.0s of the clip.
    fps = float((pose_payload.get("video_metadata") or {}).get("fps") or 30.0)
    n = int(len(frames))
    start = max(0, n - int(round(fps * 1.0)))
    end = n - 1
    indices = list(range(start, end + 1))
    if not indices:
        return []

    # Score frames by wrist/elbow displacement (find likely "snap" moments).
    def _coords(frame: dict[str, Any], joint: int) -> np.ndarray:
        lm = frame.get("landmarks")
        if not isinstance(lm, list) or len(lm) != 33:
            return np.array([np.nan, np.nan], dtype=float)
        try:
            x = float(lm[joint][0])
            y = float(lm[joint][1])
        except Exception:
            return np.array([np.nan, np.nan], dtype=float)
        return np.array([x, y], dtype=float)

    joints = (13, 14, 15, 16)  # elbows/wrists
    scores: list[tuple[float, int]] = []
    for i in indices[1:]:
        prev = frames[i - 1] if i - 1 >= 0 else None
        cur = frames[i]
        if not isinstance(prev, dict) or not isinstance(cur, dict):
            continue
        disp = 0.0
        parts = 0
        for j in joints:
            a = _coords(prev, j)
            b = _coords(cur, j)
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                continue
            d = float(np.linalg.norm(b - a))
            disp += d
            parts += 1
        if parts:
            scores.append((disp / parts, int(i)))
    scores.sort(reverse=True, key=lambda t: t[0])
    picked = [idx for _score, idx in scores[: max_frames]]
    # Always include a small sequential window around the highest-scoring frame.
    if picked:
        pivot = picked[0]
        window = list(range(max(0, pivot - 6), min(n, pivot + 7)))
        picked = sorted(set(picked + window))
    return picked[: max_frames]


def cmd_run(args: argparse.Namespace) -> None:
    _ensure_dirs()
    video_path = Path(args.video).expanduser()
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    # Configure deterministic-ish tracker settings for debugging.
    if args.time_scale is not None:
        os.environ["THROWS_TRACKER_POSE_TIME_SCALE"] = str(args.time_scale)
    if args.assumed_fps is not None:
        os.environ["THROWS_TRACKER_POSE_ASSUMED_FPS"] = str(args.assumed_fps)
    if args.model_variant:
        os.environ["THROWS_TRACKER_POSE_LANDMARKER_MODEL_VARIANT"] = str(args.model_variant)

    # Run pose pipeline.
    try:
        from javelin_tracker.biomechanics.pose_estimation import PosePipeline
        from javelin_tracker.biomechanics.metrics.pipeline import MetricsPipeline
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "Biomechanics pipeline imports failed. Ensure you installed deps: `pip install -r requirements.txt`."
        ) from exc

    clip_stem = video_path.stem
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = RUNS_DIR / clip_stem / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running PosePipeline on: {video_path}")
    pose_pipe = PosePipeline()
    pose_result = pose_pipe.process_video(video_path, video_id="pose", output_dir=out_dir)
    if pose_result.get("status") != "success":
        raise SystemExit(f"PosePipeline failed: {pose_result}")
    pose_json = Path(str(pose_result["output_path"]))
    pose_payload = json.loads(pose_json.read_text(encoding="utf-8"))

    print("Running MetricsPipeline...")
    metrics_pipe = MetricsPipeline()
    metrics_result = metrics_pipe.compute_metrics(pose_payload, output_dir=out_dir)
    metrics_json = Path(str(metrics_result.get("output_path")))
    metrics_payload = json.loads(metrics_json.read_text(encoding="utf-8")) if metrics_json.exists() else {}

    # Export overlay frames.
    overlay_dir = out_dir / "overlays"
    skeleton_dir = out_dir / "skeleton"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    skeleton_dir.mkdir(parents=True, exist_ok=True)

    indices: list[int]
    if args.frames:
        indices = [int(x) for x in args.frames.split(",") if str(x).strip()]
    else:
        indices = _candidate_frames_for_export(pose_payload, max_frames=int(args.max_overlay_frames))

    # If metrics computed an analysis window, also export that window (sampled).
    try:
        analysis = metrics_payload.get("analysis_window") if isinstance(metrics_payload, dict) else None
        if isinstance(analysis, dict) and "start_frame" in analysis and "end_frame" in analysis:
            start = int(analysis["start_frame"])
            end = int(analysis["end_frame"])
            step = max(1, int(args.window_step))
            window_idx = list(range(max(0, start), max(0, end) + 1, step))
            indices = sorted(set(indices + window_idx))
    except Exception:
        pass

    # Map pose frames by index for quick lookup.
    frames = pose_payload.get("frames", [])
    by_idx: dict[int, dict[str, Any]] = {}
    if isinstance(frames, list):
        for fr in frames:
            if not isinstance(fr, dict):
                continue
            try:
                fi = int(fr.get("frame_idx"))
            except Exception:
                continue
            by_idx[fi] = fr

    cv2 = _require_cv2()
    exported = 0
    for fr in _iter_frames_by_index(video_path, indices):
        pose_fr = by_idx.get(int(fr.frame_idx))
        if not pose_fr:
            continue
        landmarks = pose_fr.get("landmarks")
        if not isinstance(landmarks, list) or len(landmarks) != 33:
            continue
        overlay = _draw_overlay(fr.image_bgr, landmarks, min_conf=float(args.min_conf))
        skeleton = _draw_skeleton_only(fr.image_bgr.shape[:2], landmarks, min_conf=float(args.min_conf))
        out_name = f"frame_{int(fr.frame_idx):05d}.png"
        cv2.imwrite(str(overlay_dir / out_name), overlay)
        cv2.imwrite(str(skeleton_dir / out_name), skeleton)
        exported += 1

    print(f"Run dir:          {out_dir}")
    print(f"Pose JSON:        {pose_json}")
    print(f"Metrics JSON:     {metrics_json if metrics_json.exists() else '—'}")
    print(f"Overlay frames:   {overlay_dir} ({exported} PNGs)")
    print(f"Skeleton frames:  {skeleton_dir} ({exported} PNGs)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="biomech_repro", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    reg = sub.add_parser("register", help="Copy a clip into debug_assets/videos and write metadata.")
    reg.add_argument("--clip-type", choices=["good", "hard"], required=True)
    reg.add_argument("--source", required=True, help="Path to a local video file.")
    reg.add_argument("--view", default="unknown", help="e.g., side, three_quarter, front.")
    reg.add_argument("--camera", default="unknown", help="e.g., static, handheld, pan.")
    reg.add_argument("--date", default=None, help="YYYYMMDD (default: today UTC).")
    reg.add_argument("--version", default=None, help="Explicit version number (default: next available).")
    reg.add_argument("--lighting", default="", help="Optional label, e.g., daylight, indoor.")
    reg.add_argument("--people", default="", help="Optional label, e.g., single, crowd.")
    reg.add_argument("--notes", default="", help="Optional notes for metadata JSON.")
    reg.add_argument("--force", action="store_true", help="Overwrite if destination exists.")
    reg.set_defaults(func=cmd_register)

    val = sub.add_parser("validate", help="Validate a clip against the debug-clip checklist heuristics.")
    val.add_argument("video", help="Path to the debug clip.")
    val.set_defaults(func=cmd_validate)

    run = sub.add_parser("run", help="Run PosePipeline + MetricsPipeline and export overlay frames.")
    run.add_argument("video", help="Path to the debug clip.")
    run.add_argument("--run-id", default=None, help="Optional run-id (default: timestamp).")
    run.add_argument("--min-conf", default=0.25, type=float, help="Min confidence to draw landmarks.")
    run.add_argument("--frames", default=None, help="Comma-separated frame indices to export (overrides auto selection).")
    run.add_argument("--max-overlay-frames", default=24, type=int, help="Max auto-selected frames to export.")
    run.add_argument("--window-step", default=2, type=int, help="Sampling step when exporting the analysis window.")
    run.add_argument("--time-scale", default=None, type=float, help="Override THROWS_TRACKER_POSE_TIME_SCALE.")
    run.add_argument("--assumed-fps", default=None, type=float, help="Override THROWS_TRACKER_POSE_ASSUMED_FPS.")
    run.add_argument(
        "--model-variant",
        default=None,
        choices=["lite", "full", "heavy"],
        help="Override THROWS_TRACKER_POSE_LANDMARKER_MODEL_VARIANT.",
    )
    run.set_defaults(func=cmd_run)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    _ensure_dirs()
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

