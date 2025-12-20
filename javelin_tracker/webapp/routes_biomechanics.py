from __future__ import annotations

import json
import os
import shutil
import sqlite3
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from flask import g, jsonify, request, send_file

from javelin_tracker import storage
from javelin_tracker.biomechanics import BIOMECHANICS_LOGGER as logger
from javelin_tracker.biomechanics.comparison.reporter import generate_comparison_report
from javelin_tracker.biomechanics.feedback.generator import generate_feedback
from javelin_tracker.biomechanics.metrics.pipeline import MetricsPipeline

BASE_DIR = Path(__file__).resolve().parents[2]
BIOMECH_DIR = BASE_DIR / "data" / "biomechanics"
VIDEO_DIR = BIOMECH_DIR / "videos"
DB_PATH = BIOMECH_DIR / "biomechanics_jobs.db"
EXECUTOR = ThreadPoolExecutor(max_workers=2)


def _init_db() -> None:
    BIOMECH_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                status TEXT,
                percent REAL,
                error TEXT,
                created_at REAL
            )
            """
        )
        conn.commit()


def _update_job(
    job_id: str, status: str, percent: float = 0.0, error: str | None = None, session_id: str | None = None
) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO jobs (id, session_id, status, percent, error, created_at) VALUES (?, COALESCE(?, (SELECT session_id FROM jobs WHERE id=?)), ?, ?, ?, strftime('%s','now'))",
            (job_id, session_id, job_id, status, percent, error),
        )
        conn.commit()


def _set_job(job_id: str, session_id: str, status: str, percent: float = 0.0, error: str | None = None) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO jobs (id, session_id, status, percent, error, created_at) VALUES (?, ?, ?, ?, ?, strftime('%s','now'))",
            (job_id, session_id, status, percent, error),
        )
        conn.commit()


def _get_job(job_id: str) -> dict | None:
    if not DB_PATH.exists():
        return None
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("SELECT id, session_id, status, percent, error FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "session_id": row[1],
            "status": row[2],
            "percent_complete": float(row[3]),
            "error_message": row[4],
        }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _webapp_sessions_file(user_id: str) -> Path:
    return BASE_DIR / "data" / "webapp" / "userspace" / str(user_id) / "sessions.json"


def _update_session_biomechanics(
    session_id: str,
    *,
    user_id: str | None,
    patch: dict[str, object],
) -> None:
    if not user_id:
        return
    sessions_file = _webapp_sessions_file(user_id)
    sessions_file.parent.mkdir(parents=True, exist_ok=True)
    storage.update_session_by_id(session_id, patch, sessions_file=sessions_file)


def _try_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _process_video_job(job_id: str, session_id: str, video_path: Path, user_id: str | None = None) -> None:
    metrics_path: Path | None = None
    try:
        # Percent values are used by the frontend progress bar; keep them monotonic and
        # tied to real work where possible.
        POSE_START = 0.0
        POSE_END = 85.0
        QC_END = 88.0
        METRICS_END = 96.0
        REPORT_END = 98.0
        FEEDBACK_END = 99.0

        _update_job(job_id, "processing", POSE_START, session_id=session_id)
        try:
            from javelin_tracker.biomechanics.pose_estimation import PosePipeline  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "PosePipeline unavailable: install optional pose dependencies (opencv-python, mediapipe)."
            ) from exc

        pipeline = PosePipeline()
        last_pose_percent = -1
        last_pose_write = 0.0

        def _pose_progress(done_frames: int, total_frames: int) -> None:
            nonlocal last_pose_percent, last_pose_write
            total = int(total_frames or 0)
            done = int(done_frames or 0)
            if total <= 0:
                return
            frac = float(max(0.0, min(1.0, float(done) / float(total))))
            pct = float(POSE_START + frac * (POSE_END - POSE_START))
            pct_int = int(max(POSE_START, min(POSE_END - 1.0, pct)))
            now = time.monotonic()
            if pct_int <= last_pose_percent and (now - last_pose_write) < 1.0:
                return
            if pct_int > last_pose_percent:
                last_pose_percent = pct_int
                last_pose_write = now
                _update_job(job_id, "processing", float(pct_int), session_id=session_id)

        result = pipeline.process_video(
            video_path,
            video_id=job_id,
            output_dir=VIDEO_DIR,
            progress_callback=_pose_progress,
        )
        if result.get("status") != "success":
            raise RuntimeError(result.get("error") or "Processing failed.")

        _update_job(job_id, "processing", POSE_END, session_id=session_id)

        # Validate quality and attach to output JSON.
        output_path = Path(result["output_path"])
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        try:
            from javelin_tracker.biomechanics.utils.validation import validate_pose_quality  # type: ignore
        except Exception as exc:
            validate_pose_quality = None  # type: ignore[assignment]
            logger.warning("validate_pose_quality unavailable; skipping QC scoring: %s", exc)

        if validate_pose_quality is not None:
            quality = validate_pose_quality(payload.get("frames", []), payload.get("video_metadata", {}))
        else:
            quality = None
        payload["quality"] = quality
        payload["session_id"] = session_id
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        _update_job(job_id, "processing", QC_END, session_id=session_id)

        # Step 21: compute biomechanics metrics bundle for the session.
        metrics_pipeline = MetricsPipeline(log=logger)
        def _metrics_progress(frac: float) -> None:
            frac_f = float(frac) if frac is not None else 0.0
            frac_f = float(max(0.0, min(1.0, frac_f)))
            pct = float(QC_END + frac_f * (METRICS_END - QC_END))
            _update_job(job_id, "processing", float(min(METRICS_END - 1.0, pct)), session_id=session_id)

        metrics_result = metrics_pipeline.compute_metrics(
            payload,
            output_dir=BIOMECH_DIR / "sessions",
            progress_callback=_metrics_progress,
        )
        raw_metrics_path = metrics_result.get("output_path")
        if isinstance(raw_metrics_path, str) and raw_metrics_path.strip():
            metrics_path = Path(raw_metrics_path)
        else:
            metrics_path = None

        _update_job(job_id, "processing", METRICS_END, session_id=session_id)

        # Optional: comparison + feedback when an elite reference exists.
        report_payload: dict[str, object] | None = None
        feedback_payload: list[dict[str, object]] | None = None
        elite_profile = BASE_DIR / "data" / "biomechanics" / "elite_database" / "reference_profile_overall.json"
        if elite_profile.exists() and elite_profile.stat().st_size > 2:
            try:
                report_payload = generate_comparison_report(payload, elite_profile, athlete_id=str(session_id), style="overall")
                if metrics_path is not None:
                    _try_write_json(metrics_path.with_name("comparison_report.json"), report_payload)
                _update_job(job_id, "processing", REPORT_END, session_id=session_id)
            except Exception as exc:
                logger.warning("Comparison report skipped for session %s: %s", session_id, exc)

            if report_payload is not None:
                try:
                    rules_path = BASE_DIR / "config" / "feedback_rules.json"
                    feedback_payload = generate_feedback(report_payload, metrics_path, rules_path)
                    if metrics_path is not None:
                        _try_write_json(metrics_path.with_name("feedback.json"), feedback_payload)
                    _update_job(job_id, "processing", FEEDBACK_END, session_id=session_id)
                except Exception as exc:
                    logger.warning("Feedback generation skipped for session %s: %s", session_id, exc)

        if metrics_path is not None and metrics_path.exists():
            try:
                relative_metrics_path = str(metrics_path.relative_to(BASE_DIR))
            except Exception:
                relative_metrics_path = str(metrics_path)
        else:
            relative_metrics_path = None

        _update_session_biomechanics(
            session_id,
            user_id=user_id,
            patch={
                "video_id": job_id,
                "biomechanics_analysis_id": job_id,
                "biomechanics_status": "complete",
                "biomechanics_timestamp": _now_iso(),
                "biomechanics_result_path": relative_metrics_path,
            },
        )

        _update_job(job_id, "complete", 100.0, session_id=session_id)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Processing job %s failed: %s", job_id, exc)
        _update_job(job_id, "error", 0.0, error=str(exc), session_id=session_id)
        try:
            _update_session_biomechanics(
                session_id,
                user_id=user_id,
                patch={
                    "video_id": job_id,
                    "biomechanics_analysis_id": job_id,
                    "biomechanics_status": "failed",
                    "biomechanics_timestamp": _now_iso(),
                    "biomechanics_result_path": None,
                },
            )
        except Exception:
            logger.debug("Failed to persist session biomechanics status for %s", session_id)


def register_biomechanics_api(app) -> None:
    _init_db()

    @app.get("/frontend/BiomechanicsViewer.jsx")
    def biomechanics_viewer_frontend():
        """Serve the runtime React biomechanics viewer (no build step required)."""
        jsx_path = Path(__file__).resolve().parent / "frontend" / "BiomechanicsViewer.jsx"
        if not jsx_path.exists():
            return jsonify({"error": "Frontend asset not found"}), 404
        return send_file(jsx_path, mimetype="text/plain", conditional=True)

    @app.get("/api/sessions/<session_id>/biomechanics")
    def get_session_biomechanics(session_id: str):
        user = getattr(g, "user", None)
        user_id = str(user.get("id")) if isinstance(user, dict) and user.get("id") else None
        sessions_file = _webapp_sessions_file(user_id) if user_id else None
        session = storage.get_session_by_id(session_id, sessions_file=sessions_file)
        if session is None:
            return jsonify({"error": "Session not found"}), 404

        status = str(session.get("biomechanics_status") or "").strip().lower()
        job_id = session.get("biomechanics_analysis_id") or session.get("video_id")

        if status in {"queued", "pending", "processing"}:
            job = _get_job(str(job_id)) if job_id else None
            if job:
                return (
                    jsonify(
                        {
                            "status": job["status"],
                            "percent_complete": job["percent_complete"],
                            "error_message": job.get("error_message"),
                        }
                    ),
                    202,
                )
            return jsonify({"status": status, "percent_complete": 0.0, "error_message": None}), 202

        if status == "complete":
            bundle = storage.get_session_biomechanics(session_id, sessions_file=sessions_file) or {}
            metrics = bundle.get("metrics")
            if not isinstance(metrics, dict):
                return jsonify({"error": "Biomechanics results missing"}), 404
            return (
                jsonify(
                    {
                        "metrics": metrics,
                        "comparison": bundle.get("comparison_report"),
                        "feedback": bundle.get("feedback"),
                    }
                ),
                200,
            )

        # Not analyzed (or failed without cached results).
        if status in {"failed", "error"}:
            job = _get_job(str(job_id)) if job_id else None
            return (
                jsonify(
                    {
                        "error": "Biomechanics analysis failed",
                        "status": status,
                        "error_message": job.get("error_message") if job else None,
                    }
                ),
                404,
            )

        return jsonify({"error": "Biomechanics not analyzed"}), 404

    @app.post("/api/sessions/<session_id>/upload-video")
    def upload_video(session_id: str):
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file provided."}), 400

        filename = file.filename or ""
        if not filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
            return jsonify({"error": "Unsupported file type."}), 400

        file.seek(0, os.SEEK_END)
        size = file.tell()
        file.seek(0)
        if size > 100 * 1024 * 1024:
            return jsonify({"error": "File too large (max 100MB)."}), 400

        session_dir = VIDEO_DIR / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        target = session_dir / "raw.mp4"
        try:
            file.save(target)
        except OSError as exc:
            logger.error("Failed to save video: %s", exc)
            return jsonify({"error": "Failed to save video to disk."}), 500

        # Cleanup if disk issues (basic check).
        if not target.exists() or target.stat().st_size == 0:
            try:
                target.unlink(missing_ok=True)
            except Exception:
                pass
            return jsonify({"error": "File could not be saved."}), 500

        job_id = uuid.uuid4().hex[:16]
        _set_job(job_id, session_id, "queued", 0.0)

        user = getattr(g, "user", None)
        user_id = str(user.get("id")) if isinstance(user, dict) and user.get("id") else None
        if user_id:
            try:
                _update_session_biomechanics(
                    session_id,
                    user_id=user_id,
                    patch={
                        "video_id": job_id,
                        "biomechanics_analysis_id": job_id,
                        "biomechanics_status": "pending",
                        "biomechanics_timestamp": None,
                        "biomechanics_result_path": None,
                    },
                )
            except Exception:
                logger.debug("Failed to mark session %s biomechanics as pending.", session_id)

        if user_id:
            EXECUTOR.submit(_process_video_job, job_id, session_id, target, user_id)
        else:
            EXECUTOR.submit(_process_video_job, job_id, session_id, target)

        return jsonify(
            {
                "status": "processing",
                "video_id": job_id,
                "session_id": session_id,
                "estimated_time_seconds": 30,
            }
        )

    @app.get("/api/sessions/<session_id>/biomechanics/progress")
    def biomechanics_progress(session_id: str):
        job_id = request.args.get("video_id") or request.args.get("job_id")
        if not job_id:
            user = getattr(g, "user", None)
            user_id = str(user.get("id")) if isinstance(user, dict) and user.get("id") else None
            sessions_file = _webapp_sessions_file(user_id) if user_id else None
            session = storage.get_session_by_id(session_id, sessions_file=sessions_file)
            if session is None:
                return jsonify({"error": "Session not found"}), 404
            job_id = session.get("biomechanics_analysis_id") or session.get("video_id")
        if not job_id:
            return jsonify({"error": "No biomechanics job found for session"}), 404

        job = _get_job(str(job_id))
        if not job or job.get("session_id") not in (session_id, "", None):
            return jsonify({"error": "Job not found"}), 404

        http_status = 202 if job["status"] in {"queued", "processing"} else 200
        return (
            jsonify(
                {
                    "status": job["status"],
                    "percent_complete": job["percent_complete"],
                    "error_message": job.get("error_message"),
                }
            ),
            http_status,
        )

    @app.post("/api/sessions/<session_id>/biomechanics/rerun")
    def biomechanics_rerun(session_id: str):
        user = getattr(g, "user", None)
        user_id = str(user.get("id")) if isinstance(user, dict) and user.get("id") else None
        if not user_id:
            return jsonify({"error": "Login required"}), 400

        sessions_file = _webapp_sessions_file(user_id)
        session = storage.get_session_by_id(session_id, sessions_file=sessions_file)
        if session is None:
            return jsonify({"error": "Session not found"}), 404

        raw_video = VIDEO_DIR / session_id / "raw.mp4"
        if not raw_video.exists():
            return jsonify({"error": "No uploaded video found for session"}), 404

        # Delete cached results (metrics + derived artifacts).
        metrics_dir = BIOMECH_DIR / "sessions" / session_id
        if metrics_dir.exists():
            shutil.rmtree(metrics_dir, ignore_errors=True)

        existing_metrics_path = session.get("biomechanics_result_path")
        if isinstance(existing_metrics_path, str) and existing_metrics_path.strip():
            try:
                candidate = Path(existing_metrics_path.strip())
                if not candidate.is_absolute():
                    candidate = BASE_DIR / candidate
                if candidate.exists():
                    shutil.rmtree(candidate.parent, ignore_errors=True)
            except Exception:
                logger.debug("Failed to remove cached metrics path for session %s", session_id)

        # Delete old pose output dir (best-effort).
        prior_job_id = session.get("biomechanics_analysis_id") or session.get("video_id")
        if prior_job_id:
            try:
                prior_dir = VIDEO_DIR / str(prior_job_id)
                if prior_dir.exists() and prior_dir.is_dir():
                    shutil.rmtree(prior_dir, ignore_errors=True)
            except Exception:
                logger.debug("Failed to remove cached pose output for session %s", session_id)

        job_id = uuid.uuid4().hex[:16]
        _set_job(job_id, session_id, "queued", 0.0)
        storage.update_session_by_id(
            session_id,
            {
                "video_id": job_id,
                "biomechanics_analysis_id": job_id,
                "biomechanics_status": "pending",
                "biomechanics_timestamp": None,
                "biomechanics_result_path": None,
            },
            sessions_file=sessions_file,
        )

        EXECUTOR.submit(_process_video_job, job_id, session_id, raw_video, user_id)
        return jsonify({"status": "processing", "session_id": session_id, "video_id": job_id}), 202

    @app.get("/api/sessions/<session_id>/biomechanics/video")
    def biomechanics_video(session_id: str):
        video_path = VIDEO_DIR / session_id / "raw.mp4"
        if not video_path.exists():
            return jsonify({"error": "Video not found"}), 404
        return send_file(video_path, mimetype="video/mp4", conditional=True)

    @app.get("/api/sessions/<session_id>/biomechanics/pose")
    def biomechanics_pose(session_id: str):
        user = getattr(g, "user", None)
        user_id = str(user.get("id")) if isinstance(user, dict) and user.get("id") else None
        sessions_file = _webapp_sessions_file(user_id) if user_id else None
        session = storage.get_session_by_id(session_id, sessions_file=sessions_file)
        if session is None:
            return jsonify({"error": "Session not found"}), 404

        job_id = session.get("biomechanics_analysis_id") or session.get("video_id")
        if not job_id:
            return jsonify({"error": "No biomechanics job found for session"}), 404

        pose_path = VIDEO_DIR / str(job_id) / "pose_data.json"
        if not pose_path.exists():
            return jsonify({"error": "Pose data not found"}), 404

        stride_raw = request.args.get("stride", "1")
        try:
            stride = int(stride_raw)
        except Exception:
            return jsonify({"error": "stride must be an integer"}), 400
        if stride < 1:
            return jsonify({"error": "stride must be >= 1"}), 400

        try:
            payload = json.loads(pose_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error("Failed to load pose payload for session=%s: %s", session_id, exc)
            return jsonify({"error": "Pose data is invalid"}), 400

        frames = payload.get("frames", [])
        if isinstance(frames, list) and stride > 1:
            frames = frames[::stride]

        # Return a slim payload for UI consumption.
        slim_frames = []
        if isinstance(frames, list):
            for frame in frames:
                if not isinstance(frame, dict):
                    continue
                slim_frames.append(
                    {
                        "frame_idx": frame.get("frame_idx"),
                        "timestamp_ms": frame.get("timestamp_ms"),
                        "landmarks": frame.get("landmarks"),
                        "valid": frame.get("valid", True),
                    }
                )

        return (
            jsonify(
                {
                    "video_id": payload.get("video_id"),
                    "video_metadata": payload.get("video_metadata", {}),
                    "frames": slim_frames,
                    "stride": stride,
                }
            ),
            200,
        )

    @app.get("/api/elite-db/reference/<style>")
    def elite_reference(style: str):
        style_norm = (style or "").strip().lower().replace(" ", "_")
        if style_norm in {"", "default", "all"}:
            return jsonify({"error": "style is required"}), 400
        if style_norm in {"overall", "global"}:
            filename = "reference_profile_overall.json"
        else:
            filename = f"reference_profile_{style_norm}.json"
        path = BASE_DIR / "data" / "biomechanics" / "elite_database" / filename
        if not path.exists():
            return jsonify({"error": "Reference profile not found"}), 404
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error("Failed to load elite reference %s: %s", path, exc)
            return jsonify({"error": "Reference profile is invalid"}), 400
        return jsonify(payload), 200

    @app.post("/api/elite-db/set-style/<path:athlete>")
    def set_athlete_style(athlete: str):
        user = getattr(g, "user", None)
        if not isinstance(user, dict) or not user.get("id"):
            return jsonify({"error": "Login required"}), 400

        payload = request.get_json(force=True) or {}
        style = payload.get("throwing_style", payload.get("style"))
        style_text = str(style).strip() if style is not None else ""
        if not style_text:
            return jsonify({"error": "throwing_style is required"}), 400

        athlete_name = (athlete or "").strip()
        if not athlete_name:
            return jsonify({"error": "athlete is required"}), 400

        with storage.open_database() as conn:
            row = conn.execute(
                "SELECT id, name FROM Athletes WHERE lower(name) = lower(?) LIMIT 1",
                (athlete_name,),
            ).fetchone()
            if not row:
                return jsonify({"error": "Athlete not found"}), 404
            athlete_id, canonical_name = int(row[0]), str(row[1])
            conn.execute(
                "UPDATE Athletes SET throwing_style = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (style_text, athlete_id),
            )
            conn.commit()

        logger.info("Set throwing_style=%s for athlete=%s (%s)", style_text, canonical_name, athlete_id)
        return jsonify({"athlete_id": athlete_id, "athlete": canonical_name, "throwing_style": style_text}), 200
