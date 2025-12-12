from __future__ import annotations

import json
import csv
import logging
import sqlite3
import os
import uuid
from datetime import datetime, date, timedelta
from functools import wraps
from pathlib import Path
from typing import Any, Iterable
from collections import defaultdict
from datetime import timedelta as datetime_timedelta

from flask import (
    Flask,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    session as flask_session,
    url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash

from .. import services, storage
from ..reports import generate_weekly_report
from ..constants import DEFAULT_ATHLETE_PLACEHOLDER
from ..env import LEGACY_PREFIX, PRIMARY_PREFIX
from ..models import DEFAULT_EVENT

BASE_DIR = Path(__file__).resolve().parents[2]
WEBAPP_DATA = BASE_DIR / "data" / "webapp"
USERS_FILE = WEBAPP_DATA / "users.json"


FAILED_LOGINS: dict[str, list[datetime]] = defaultdict(list)
MAX_FAILED_ATTEMPTS = 5
FAILED_WINDOW_MINUTES = 10


def create_app() -> Flask:
    WEBAPP_DATA.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    app = Flask(
        __name__,
        static_folder=str(Path(__file__).parent / "static"),
        template_folder=str(Path(__file__).parent / "templates"),
    )
    env_secret = os.environ.get("THROWS_TRACKER_SECRET") or os.environ.get("JAVELIN_TRACKER_SECRET") or os.environ.get(
        "SECRET_KEY"
    )
    if not env_secret and os.environ.get("FLASK_ENV") == "production":
        raise RuntimeError("SECRET_KEY/THROWS_TRACKER_SECRET must be set in production.")
    app.secret_key = env_secret or "dev-secret"
    app.config.update(
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=os.environ.get("SESSION_COOKIE_SECURE", "0") == "1",
        PREFERRED_URL_SCHEME="https" if os.environ.get("FORCE_HTTPS", "0") == "1" else "http",
    )

    @app.before_request
    def load_user() -> None:
        user_id = flask_session.get("user_id")
        g.user = None
        if user_id:
            g.user = _get_user_by_id(user_id)
            if g.user:
                # Ensure every request is scoped to the active user's datastore
                _set_user_env(user_id)
            _ensure_default_team(user_id)

    @app.context_processor
    def inject_onboarding():
        onboarding = None
        try:
            if getattr(g, "user", None):
                onboarding = _compute_onboarding_progress(g.user["id"])
        except Exception:
            onboarding = None
        return {"dashboard_onboarding": onboarding}

    register_routes(app)
    register_api(app)
    return app


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not getattr(g, "user", None):
            return redirect(url_for("login"))
        return view(*args, **kwargs)

    return wrapped


def require_role(*roles: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            user = getattr(g, "user", None)
            role = user.get("role") if user else None
            if role not in roles:
                return jsonify({"error": "forbidden"}), 403
            return func(*args, **kwargs)

        return wrapper

    return decorator


def register_routes(app: Flask) -> None:
    @app.route("/")
    def index():
        if g.get("user"):
            return redirect(url_for("dashboard"))
        return redirect(url_for("login"))

    @app.route("/login", methods=["GET", "POST"])
    def login():
        error = None
        if request.method == "POST":
            email = (request.form.get("email") or "").strip().lower()
            password = request.form.get("password") or ""
            remote_addr = request.remote_addr or "unknown"
            if _is_rate_limited(remote_addr):
                error = "Too many attempts. Try again in a few minutes."
                return render_template("login.html", error=error, page_slug="login"), 429
            user = _get_user_by_email(email)
            if not user or not check_password_hash(user["password_hash"], password):
                error = "Invalid email or password."
                _record_failed_login(remote_addr)
            else:
                _clear_failed_login(remote_addr)
                flask_session["user_id"] = user["id"]
                return redirect(url_for("dashboard"))
        return render_template("login.html", error=error, page_slug="login")

    @app.route("/register", methods=["GET", "POST"])
    def register():
        error = None
        if request.method == "POST":
            name = (request.form.get("name") or "").strip() or "Coach"
            email = (request.form.get("email") or "").strip().lower()
            password = request.form.get("password") or ""
            if not email or not password:
                error = "Email and password are required."
            elif len(password) < 8:
                error = "Use a password with at least 8 characters."
            elif _get_user_by_email(email):
                error = "Account already exists for that email."
            else:
                user = _create_user(name=name, email=email, password=password)
                flask_session["user_id"] = user["id"]
                return redirect(url_for("dashboard"))
        return render_template("register.html", error=error, page_slug="register")

    @app.post("/logout")
    def auth_logout():
        flask_session.clear()
        return redirect(url_for("login"))

    @app.route("/dashboard")
    @login_required
    def dashboard():
        demo_seeded = _maybe_seed_demo_data(g.user["id"])
        onboarding = _compute_onboarding_progress(g.user["id"])
        return render_template(
            "dashboard.html",
            page_slug="dashboard",
            demo_seeded=demo_seeded,
            dashboard_onboarding=onboarding,
            acute_days=7,
            chronic_days=28,
        )

    @app.route("/sessions")
    @login_required
    def sessions_page():
        return render_template("sessions.html", page_slug="sessions")

    @app.route("/logs")
    @login_required
    def logs_page():
        return render_template("logs.html", page_slug="logs")

    @app.route("/analytics")
    @login_required
    def analytics_page():
        return render_template("analytics.html", page_slug="analytics", acute_days=7, chronic_days=28)

    @app.route("/throwai")
    @login_required
    def training_page():
        return render_template("throwai.html", page_slug="throwai")

    @app.route("/weight-room")
    @login_required
    def weight_room_page():
        return render_template("weight_room.html", page_slug="weightroom")

    @app.route("/reports")
    @login_required
    def reports_page():
        return render_template("reports.html", page_slug="reports", acute_days=7, chronic_days=28)

    @app.route("/athletes")
    @login_required
    def athletes_page():
        return render_template("athletes.html", page_slug="athletes")

    @app.route("/help/metrics")
    @login_required
    def metrics_help():
        return render_template("metrics_explainer.html", page_slug="metrics_help", page_title="Load & readiness")

    @app.route("/quickstart")
    @login_required
    def quickstart_page():
        return render_template("quickstart.html", page_slug="quickstart", page_title="Quick start guide")

    @app.route("/privacy")
    def privacy():
        return render_template("privacy.html", page_slug="privacy", page_title="Privacy & data")

    @app.route("/health")
    def health():
        return jsonify({"status": "ok"}), 200


def register_api(app: Flask) -> None:
    @app.get("/api/sessions")
    @login_required
    def api_sessions():
        sessions = _load_sessions(g.user["id"])
        return jsonify({"sessions": sessions})

    @app.post("/api/sessions")
    @login_required
    def api_create_session():
        payload = request.get_json(force=True) or {}
        athlete_name = (payload.get("athlete") or DEFAULT_ATHLETE_PLACEHOLDER).strip() or DEFAULT_ATHLETE_PLACEHOLDER
        team_name = (payload.get("team") or "Unassigned").strip() or "Unassigned"
        team_id = _get_or_create_team(g.user["id"], team_name)
        athlete_id = _get_or_create_athlete(g.user["id"], athlete_name, team_id)
        try:
            result = services.build_session_from_inputs(
                date_text=payload.get("date"),
                best=payload.get("best"),
                throws=payload.get("throws"),
                rpe=payload.get("rpe"),
                notes=payload.get("notes"),
                tags=payload.get("tags"),
                duration_minutes=payload.get("duration_minutes"),
                athlete=athlete_name,
                team=team_name,
                event=payload.get("event") or DEFAULT_EVENT,
                implement_weight_kg=payload.get("implement_weight_kg"),
                technique=payload.get("technique"),
                fouls=payload.get("fouls"),
            )
        except Exception as exc:  # pragma: no cover - surfacing to client
            return jsonify({"error": str(exc)}), 400

        session_dict = result.session.to_dict()
        session_dict["id"] = payload.get("id") or uuid.uuid4().hex[:16]
        session_dict["athlete_hash"] = uuid.uuid5(uuid.NAMESPACE_DNS, athlete_name).hex[:12]
        session_dict["event"] = payload.get("event") or DEFAULT_EVENT
        session_dict["team"] = team_name
        session_dict["team_id"] = team_id
        _append_session(g.user["id"], session_dict)
        _purge_demo_data_if_needed(g.user["id"])
        # Keep athlete table in sync by logging throw distance if present
        if session_dict.get("best"):
            storage.log_throw_distance(athlete_id, session_dict["date"], session_dict["event"], session_dict["best"])
        summary = _build_summary(g.user["id"], group="week")
        return jsonify({"session": session_dict, "summary": summary})

    @app.delete("/api/sessions/<session_id>")
    @login_required
    def api_delete_session(session_id: str):
        removed = _delete_session(g.user["id"], session_id)
        if not removed:
            return jsonify({"error": "Session not found"}), 404
        return jsonify({"status": "ok"})

    @app.get("/api/summary")
    @login_required
    def api_summary():
        group = request.args.get("group", "week")
        try:
            summary = _build_summary(g.user["id"], group=group)
        except Exception as exc:  # pragma: no cover
            return jsonify({"error": str(exc)}), 400
        return jsonify(summary)

    @app.get("/api/summary/table")
    @login_required
    def api_summary_table():
        group = request.args.get("group", "week")
        summary = _build_summary(g.user["id"], group=group)
        table = summary.get("table", "")
        return jsonify({"table": table})

    @app.post("/api/reports/weekly")
    @login_required
    def api_weekly_report():
        payload = request.get_json(force=True) or {}
        athlete = (payload.get("athlete") or "").strip()
        if not athlete:
            return jsonify({"error": "athlete is required"}), 400
        events_raw = payload.get("events") or []
        if isinstance(events_raw, str):
            events = [chunk.strip() for chunk in events_raw.split(",") if chunk.strip()]
        else:
            events = [str(e).strip() for e in events_raw if str(e).strip()]
        team_name = (payload.get("team") or "").strip() or None
        school_name = (payload.get("school") or "").strip() or None
        team_filter = (payload.get("team") or "").strip() or None
        week_ending_raw = (payload.get("week_ending") or "").strip()
        week_ending: date | None = None
        if week_ending_raw:
            try:
                week_ending = datetime.fromisoformat(week_ending_raw).date()
            except Exception:
                return jsonify({"error": "week_ending must be YYYY-MM-DD"}), 400

        _set_user_env(g.user["id"])
        sessions = storage.load_sessions()
        if team_filter:
            sessions = [s for s in sessions if (s.get("team") or "").lower() == team_filter.lower()]
        output_dir = WEBAPP_DATA / "userspace" / g.user["id"] / "reports"
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            pdf_paths = generate_weekly_report(
                sessions,
                athlete=athlete,
                events=events or None,
                week_ending=week_ending,
                output_dir=output_dir,
                team_name=team_name,
                school_name=school_name,
            )
        except Exception as exc:  # pragma: no cover
            return jsonify({"error": str(exc)}), 400
        _record_report_generated(g.user["id"])
        return jsonify(
            {
                "message": f"Saved {len(pdf_paths)} PDF report(s) to {output_dir}",
                "files": [str(path) for path in pdf_paths],
            }
        )

    @app.get("/api/strength-logs")
    @login_required
    def api_strength_logs():
        logs = _load_strength_logs(g.user["id"])
        return jsonify({"logs": logs})

    @app.post("/api/training/log-lift")
    @login_required
    def api_log_lift():
        payload = request.get_json(force=True) or {}
        athlete = (payload.get("athlete") or "").strip() or DEFAULT_ATHLETE_PLACEHOLDER
        athlete_id = _get_or_create_athlete(g.user["id"], athlete)
        try:
            storage.log_strength_workout(
                athlete_id,
                payload.get("date") or datetime.now().date().isoformat(),
                payload.get("exercise") or "exercise",
                float(payload.get("weight_kg") or 0),
                int(payload.get("reps") or 0),
                notes=payload.get("notes"),
            )
        except Exception as exc:  # pragma: no cover
            return jsonify({"error": str(exc)}), 400
        return jsonify({"message": "Set logged."})

    @app.post("/api/training/plan")
    @login_required
    def api_training_plan():
        payload = request.get_json(force=True) or {}
        athlete_name = (payload.get("athlete") or "").strip()
        if not athlete_name:
            return jsonify({"error": "athlete is required"}), 400
        athlete_id = _get_or_create_athlete(g.user["id"], athlete_name)
        profile = _fetch_athlete_profile(g.user["id"], athlete_id)
        plan = storage.generate_workout(profile or {"id": athlete_id, "name": athlete_name})
        return jsonify({"plan": plan})

    @app.post("/api/forecast")
    @login_required
    def api_forecast():
        payload = request.get_json(force=True) or {}
        athlete_name = (payload.get("athlete") or "").strip()
        metric = payload.get("metric") or "throw_distance"
        days = int(payload.get("days") or 14)
        if not athlete_name:
            return jsonify({"error": "athlete is required"}), 400
        athlete_id = _lookup_athlete_id(g.user["id"], athlete_name)
        if athlete_id is None:
            athlete_id = _get_or_create_athlete(g.user["id"], athlete_name)
        _backfill_throw_logs(g.user["id"], athlete_name, athlete_id)
        forecast = storage.predict_trends(athlete_id, metric, days_ahead=days)
        profile = _fetch_athlete_profile(g.user["id"], athlete_id)
        return jsonify({"forecast": forecast, "profile": profile})

    @app.get("/api/athletes")
    @login_required
    def api_athletes():
        athletes = _list_athletes(g.user["id"])
        return jsonify({"athletes": athletes})

    @app.get("/api/athletes/profiles")
    @login_required
    def api_athlete_profiles():
        athletes = _list_athletes(g.user["id"], include_profile=True)
        return jsonify({"athletes": athletes})

    @app.post("/api/athletes/<int:athlete_id>/profile")
    @login_required
    def api_update_athlete(athlete_id: int):
        payload = request.get_json(force=True) or {}
        benchmarks = {}
        if payload.get("bench_1rm_kg"):
            benchmarks["bench press"] = float(payload.get("bench_1rm_kg"))
        if payload.get("squat_1rm_kg"):
            benchmarks["back squat"] = float(payload.get("squat_1rm_kg"))
        try:
            team_id = None
            if payload.get("team_id"):
                try:
                    team_id = int(payload["team_id"])
                except Exception:
                    team_id = _lookup_team_id_by_name(g.user["id"], str(payload.get("team_id")))
            storage.update_athlete_profile(
                athlete_id,
                height_cm=_float_or_none(payload.get("height_cm")),
                weight_kg=_float_or_none(payload.get("weight_kg")),
                new_strength_benchmarks=benchmarks or None,
                notes=payload.get("notes"),
            )
            if team_id:
                _update_athlete_team(g.user["id"], athlete_id, team_id)
        except Exception as exc:  # pragma: no cover
            return jsonify({"error": str(exc)}), 400
        athlete = _fetch_athlete_profile(g.user["id"], athlete_id)
        return jsonify({"athlete": athlete})

    @app.post("/api/import/roster")
    @login_required
    @require_role("head_coach")
    def api_import_roster():
        if "file" not in request.files:
            return jsonify({"error": "Missing CSV file"}), 400
        file = request.files["file"]
        rows = file.read().decode("utf-8").splitlines()
        summary = {"created": 0, "errors": []}
        reader = csv.DictReader(rows)
        required = {"first_name", "last_name"}
        if not required.issubset(set(reader.fieldnames or [])):
            return jsonify({"error": "CSV must include first_name,last_name columns"}), 400
        for index, row in enumerate(reader, start=2):
            try:
                name = f"{row.get('first_name','').strip()} {row.get('last_name','').strip()}".strip()
                if not name:
                    raise ValueError("Name missing")
                team_name = (row.get("team_name") or "Unassigned").strip() or "Unassigned"
                team_id = _get_or_create_team(g.user["id"], team_name)
                athlete_id = _get_or_create_athlete(g.user["id"], name, team_id)
                if team_id:
                    _update_athlete_team(g.user["id"], athlete_id, team_id)
                height = _float_or_none(row.get("height_cm"))
                weight = _float_or_none(row.get("weight_kg"))
                storage.update_athlete_profile(athlete_id, height_cm=height, weight_kg=weight)
                summary["created"] += 1
            except Exception as exc:
                summary["errors"].append({"row": index, "error": str(exc)})
        return jsonify(summary)

    @app.post("/api/import/sessions")
    @login_required
    @require_role("head_coach")
    def api_import_sessions():
        if "file" not in request.files:
            return jsonify({"error": "Missing CSV file"}), 400
        file = request.files["file"]
        rows = file.read().decode("utf-8").splitlines()
        reader = csv.DictReader(rows)
        required = {"date", "athlete_name", "event"}
        if not required.issubset(set(reader.fieldnames or [])):
            return jsonify({"error": "CSV must include date, athlete_name, event"}), 400
        summary = {"created": 0, "errors": []}
        for index, row in enumerate(reader, start=2):
            try:
                athlete = (row.get("athlete_name") or DEFAULT_ATHLETE_PLACEHOLDER).strip()
                team_name = (row.get("team_name") or "Unassigned").strip() or "Unassigned"
                team_id = _get_or_create_team(g.user["id"], team_name)
                athlete_id = _get_or_create_athlete(g.user["id"], athlete, team_id)
                if team_id:
                    _update_athlete_team(g.user["id"], athlete_id, team_id)
                session_result = services.build_session_from_inputs(
                    date_text=row.get("date"),
                    best=row.get("best_distance") or row.get("best"),
                    throws=row.get("throws") or "",
                    rpe=row.get("rpe"),
                    notes=row.get("notes"),
                    tags=row.get("tags"),
                    duration_minutes=row.get("duration_minutes"),
                    athlete=athlete,
                    team=team_name,
                    event=row.get("event") or DEFAULT_EVENT,
                    implement_weight_kg=row.get("implement_weight_kg"),
                    technique=row.get("technique"),
                    fouls=row.get("fouls"),
                )
                session_payload = session_result.session.to_dict()
                session_payload["athlete"] = athlete
                session_payload["team"] = team_name
                session_payload["id"] = uuid.uuid4().hex[:16]
                session_payload["athlete_hash"] = uuid.uuid5(uuid.NAMESPACE_DNS, athlete).hex[:12]
                session_payload["team_id"] = team_id
                _append_session(g.user["id"], session_payload)
                summary["created"] += 1
            except Exception as exc:
                summary["errors"].append({"row": index, "error": str(exc)})
        return jsonify(summary)

    @app.get("/api/export/sessions")
    @login_required
    def api_export_sessions():
        team = request.args.get("team")
        sessions = _load_sessions(g.user["id"])
        if team and team.lower() != "all":
            sessions = [s for s in sessions if (s.get("team") or "").lower() == team.lower()]
        output_dir = WEBAPP_DATA / "userspace" / g.user["id"] / "exports"
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"sessions_export_{datetime.utcnow().isoformat()}.csv"
        storage.export_sessions_csv(path, sessions)
        return jsonify({"file": str(path)})

    @app.get("/api/teams")
    @login_required
    def api_teams():
        teams = _list_teams(g.user["id"])
        return jsonify({"teams": teams})

    @app.post("/api/teams")
    @login_required
    @require_role("head_coach")
    def api_create_team():
        payload = request.get_json(force=True) or {}
        name = (payload.get("name") or "").strip()
        if not name:
            return jsonify({"error": "Team name is required."}), 400
        team_id = _get_or_create_team(g.user["id"], name)
        return jsonify({"team": {"id": team_id, "name": name}})


def _set_user_env(user_id: str) -> None:
    user_dir = WEBAPP_DATA / "userspace" / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    db_file = user_dir / "throws_tracker.db"
    sessions_file = user_dir / "sessions.json"
    onboarding_file = user_dir / "onboarding.json"
    os.environ[f"{PRIMARY_PREFIX}DATA_DIR"] = str(user_dir)
    os.environ[f"{PRIMARY_PREFIX}DB_FILE"] = str(db_file)
    os.environ[f"{PRIMARY_PREFIX}SESSIONS_FILE"] = str(sessions_file)
    os.environ[f"{LEGACY_PREFIX}DATA_DIR"] = str(user_dir)
    os.environ[f"{LEGACY_PREFIX}DB_FILE"] = str(db_file)
    os.environ[f"{LEGACY_PREFIX}SESSIONS_FILE"] = str(sessions_file)
    if not onboarding_file.exists():
        onboarding_file.write_text(json.dumps({}, indent=2))


def _load_sessions(user_id: str) -> list[dict[str, Any]]:
    _set_user_env(user_id)
    sessions = storage.load_sessions()
    # Fill required defaults for downstream analytics
    for record in sessions:
        record.setdefault("event", DEFAULT_EVENT)
        record.setdefault("team", "Unassigned")
        record.setdefault("id", uuid.uuid4().hex[:16])
        if record.get("best") is None:
            throws = record.get("throws") or []
            try:
                numeric = [float(v) for v in throws if v is not None]
                if numeric:
                    record["best"] = max(numeric)
            except Exception:
                pass
    return sessions


def _append_session(user_id: str, session: dict[str, Any]) -> None:
    _set_user_env(user_id)
    storage.append_session(session)


def _delete_session(user_id: str, session_id: str) -> bool:
    _set_user_env(user_id)
    sessions = storage.load_sessions()
    filtered = [row for row in sessions if str(row.get("id")) != str(session_id)]
    if len(filtered) == len(sessions):
        return False
    storage.save_sessions(filtered)
    return True


def _build_summary(user_id: str, *, group: str) -> dict[str, Any]:
    _set_user_env(user_id)
    sessions = _load_sessions(user_id)
    report = services.build_summary_report(sessions, group=group)
    rows = [
        {
            "scope": row.scope,
            "athlete": row.athlete,
            "event": row.event,
            "label": row.label,
            "sessions": row.sessions,
            "best": row.best_throw,
            "avgRpe": row.avg_rpe,
            "load": row.total_load,
            "throws": row.throw_volume,
            "acwrRolling": row.acwr_rolling,
            "acwrEwma": row.acwr_ewma,
            "monotony": row.monotony_7d,
            "strain": row.strain_7d,
            "risk": row.risk_flag,
            "marker": row.marker,
        }
        for row in report.rows
    ]
    rollup_rows = [
        {
            "scope": row.scope,
            "athlete": row.athlete,
            "event": row.event,
            "label": row.label,
            "sessions": row.sessions,
            "best": row.best_throw,
            "avgRpe": row.avg_rpe,
            "load": row.total_load,
            "throws": row.throw_volume,
            "acwrRolling": row.acwr_rolling,
            "acwrEwma": row.acwr_ewma,
            "monotony": row.monotony_7d,
            "strain": row.strain_7d,
            "risk": row.risk_flag,
            "marker": row.marker,
        }
        for row in report.rollup_rows
    ]
    personal_best = (
        None
        if report.personal_best is None
        else {"date": report.personal_best[0].isoformat(), "best": report.personal_best[1]}
    )
    personal_bests = [
        {"athlete": athlete, "event": event, "date": date.isoformat(), "best": best}
        for (athlete, event), (date, best) in report.personal_bests.items()
    ]
    high_risk = {
        f"{athlete}|{event}": [d.isoformat() for d in dates]
        for (athlete, event), dates in report.high_risk_dates.items()
    }
    totals = {
        "sessions": report.total_sessions,
        "throws": report.total_throws,
        "load": report.total_load,
        "averageRpe": report.average_rpe,
        "dateRange": {"start": report.date_range[0].isoformat(), "end": report.date_range[1].isoformat()}
        if report.date_range
        else None,
    }
    series = _build_series(sessions)
    table = services.render_summary_table(report)
    return {
        "summary": {
            "rows": rows,
            "rollupRows": rollup_rows,
            "totals": totals,
            "personalBests": personal_bests,
            "highRiskDates": high_risk,
        },
        "personalBest": personal_best,
        "series": series,
        "table": table,
    }


def _build_series(sessions: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    from ..analysis import group_by_week, mean_best_by_group, median_best_by_group, total_throws_by_group

    series: list[dict[str, Any]] = []
    groups = group_by_week(sessions)
    mean_map = mean_best_by_group(groups)
    median_map = median_best_by_group(groups)
    throws_map = total_throws_by_group(groups)
    for label in sorted(groups.keys()):
        series.append(
            {
                "label": label,
                "meanBest": mean_map.get(label, 0),
                "medianBest": median_map.get(label, 0),
                "throwVolume": throws_map.get(label, 0),
            }
        )
    return series


def _load_strength_logs(user_id: str) -> list[dict[str, Any]]:
    _set_user_env(user_id)
    logs: list[dict[str, Any]] = []
    with storage.open_database(readonly=True) as conn:
        rows = conn.execute(
            """
            SELECT l.id, l.logged_at, l.exercise, l.load_kg, l.reps, l.notes, a.name
            FROM StrengthLogs l
            JOIN Athletes a ON l.athlete_id = a.id
            ORDER BY l.logged_at DESC
            """
        ).fetchall()
        for row in rows:
            log_id, logged_at, exercise, load_kg, reps, notes, athlete = row
            logs.append(
                {
                    "id": log_id,
                    "date": str(logged_at),
                    "exercise": exercise,
                    "load_kg": load_kg,
                    "reps": reps,
                    "notes": notes,
                    "athlete": athlete,
                }
            )
    return logs


def _list_athletes(user_id: str, *, include_profile: bool = False) -> list[dict[str, Any]]:
    _set_user_env(user_id)
    _sync_athletes_from_sessions(user_id)
    athletes: list[dict[str, Any]] = []
    with storage.open_database(readonly=True) as conn:
        rows = conn.execute(
            """
            SELECT a.id, a.name, a.height_cm, a.weight_kg, a.strength_benchmarks, a.notes, a.team_id, t.name
            FROM Athletes a
            LEFT JOIN Teams t ON a.team_id = t.id
            ORDER BY a.name
            """
        ).fetchall()
        for row in rows:
            athlete_id, name, height, weight, benchmarks_json, notes, team_id, team_name = row
            entry: dict[str, Any] = {"id": athlete_id, "name": name, "team_id": team_id, "team": team_name}
            if include_profile:
                entry.update(
                    {
                        "height_cm": height,
                        "weight_kg": weight,
                        "bench_1rm_kg": _coerce_benchmark(benchmarks_json, "bench press"),
                        "squat_1rm_kg": _coerce_benchmark(benchmarks_json, "back squat"),
                        "bmi": _bmi(height, weight),
                        "notes": notes,
                    }
                )
            athletes.append(entry)
    return athletes


def _fetch_athlete_profile(user_id: str, athlete_id: int) -> dict[str, Any] | None:
    _set_user_env(user_id)
    with storage.open_database(readonly=True) as conn:
        row = conn.execute(
            "SELECT id, name, height_cm, weight_kg, strength_benchmarks, notes FROM Athletes WHERE id = ?",
            (athlete_id,),
        ).fetchone()
        if not row:
            return None
        athlete_id, name, height, weight, benchmarks_json, notes = row
        profile = {
            "id": athlete_id,
            "name": name,
            "height_cm": height,
            "weight_kg": weight,
            "bench_1rm_kg": _coerce_benchmark(benchmarks_json, "bench press"),
            "squat_1rm_kg": _coerce_benchmark(benchmarks_json, "back squat"),
            "bmi": _bmi(height, weight),
            "notes": notes,
        }
    return profile


def _coerce_benchmark(raw_json: Any, exercise: str) -> float | None:
    try:
        data = json.loads(raw_json) if isinstance(raw_json, str) else raw_json
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    value = data.get(exercise)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _bmi(height_cm: Any, weight_kg: Any) -> float | None:
    try:
        if height_cm and weight_kg:
            h_m = float(height_cm) / 100.0
            if h_m > 0:
                return float(weight_kg) / (h_m * h_m)
    except Exception:
        return None
    return None


def _get_user_by_email(email: str) -> dict[str, Any] | None:
    users = _load_users()
    for user in users:
        if user["email"] == email:
            user.setdefault("role", "head_coach")
            return user
    return None


def _get_user_by_id(user_id: str) -> dict[str, Any] | None:
    users = _load_users()
    for user in users:
        if user["id"] == user_id:
            user.setdefault("role", "head_coach")
            return user
    return None


def _create_user(*, name: str, email: str, password: str) -> dict[str, Any]:
    users = _load_users()
    new_user = {
        "id": uuid.uuid4().hex[:16],
        "name": name,
        "email": email,
        "password_hash": generate_password_hash(password),
        "created_at": datetime.utcnow().isoformat(),
        "role": "head_coach",
    }
    users.append(new_user)
    _save_users(users)
    # Prime the user data dir
    _set_user_env(new_user["id"])
    return new_user


def _load_users() -> list[dict[str, Any]]:
    if not USERS_FILE.exists():
        USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
        USERS_FILE.write_text(json.dumps({"users": []}, indent=2))
    raw = USERS_FILE.read_text().strip() or '{"users": []}'
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = {"users": []}
    return payload.get("users", [])


def _save_users(users: list[dict[str, Any]]) -> None:
    USERS_FILE.write_text(json.dumps({"users": users}, indent=2))


def _lookup_athlete_id(user_id: str, name: str) -> int | None:
    _set_user_env(user_id)
    with storage.open_database(readonly=True) as conn:
        row = conn.execute("SELECT id FROM Athletes WHERE name = ? LIMIT 1", (name,)).fetchone()
        if row:
            return int(row[0])
    return None


def _get_or_create_athlete(user_id: str, name: str, team_id: int | None = None) -> int:
    clean = (name or DEFAULT_ATHLETE_PLACEHOLDER).strip() or DEFAULT_ATHLETE_PLACEHOLDER
    _set_user_env(user_id)
    with storage.open_database() as conn:
        row = conn.execute("SELECT id FROM Athletes WHERE name = ? LIMIT 1", (clean,)).fetchone()
        if row:
            return int(row[0])
        cursor = conn.execute("INSERT INTO Athletes (name, team_id) VALUES (?, ?)", (clean, team_id))
        conn.commit()
        return int(cursor.lastrowid)


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value) if value not in (None, "", "null") else None
    except (TypeError, ValueError):
        return None


def _sync_athletes_from_sessions(user_id: str) -> None:
    """
    Ensure all athlete names present in session logs also exist in the Athletes table.
    This keeps dropdowns populated even for legacy data before profiles were created.
    """
    _set_user_env(user_id)
    names = { (row.get("athlete") or "").strip() for row in _load_sessions(user_id) }
    names = {name for name in names if name}
    if not names:
        return
    for name in names:
        _get_or_create_athlete(user_id, name)


def _backfill_throw_logs(user_id: str, athlete_name: str, athlete_id: int) -> None:
    """
    For legacy data where throw distances were only stored in sessions.json,
    insert throw logs so forecasting has a proper time series.
    """
    _set_user_env(user_id)
    sessions = [row for row in _load_sessions(user_id) if (row.get("athlete") or "").strip() == athlete_name]
    if not sessions:
        return
    with storage.open_database(readonly=True) as conn:
        existing = conn.execute(
            "SELECT logged_at, event, distance FROM ThrowLogs WHERE athlete_id = ?",
            (athlete_id,),
        ).fetchall()
    existing_keys = {(str(dt).split("T")[0], ev or "", float(dist)) for dt, ev, dist in existing}

    inserted = 0
    for session in sessions:
        date_key = (session.get("date") or "").strip()
        best = session.get("best")
        if best is None:
            throws_list = session.get("throws") or []
            try:
                throws_numeric = [float(v) for v in throws_list if v is not None]
                best = max(throws_numeric) if throws_numeric else None
            except Exception:
                best = None
        event = (session.get("event") or DEFAULT_EVENT).strip().lower() or DEFAULT_EVENT
        if not date_key or best is None:
            continue
        key = (date_key, event, float(best))
        if key in existing_keys:
            continue
        try:
            storage.log_throw_distance(athlete_id, date_key, event, float(best))
            inserted += 1
        except Exception:
            continue
    if inserted:
        # Refresh profile aggregates
        try:
            storage.update_athlete_profile(athlete_id)
        except Exception:
            pass


def _list_teams(user_id: str) -> list[dict[str, Any]]:
    _set_user_env(user_id)
    _ensure_default_team(user_id)
    with storage.open_database(readonly=True) as conn:
        rows = conn.execute(
            "SELECT id, name, program_name, school_name, color, short_code FROM Teams ORDER BY name"
        ).fetchall()
    return [
        {
          "id": row[0],
          "name": row[1],
          "program_name": row[2],
          "school_name": row[3],
          "color": row[4],
          "short_code": row[5],
        }
        for row in rows
    ]


def _get_or_create_team(user_id: str, name: str | None) -> int:
    clean = (name or "Unassigned").strip() or "Unassigned"
    _set_user_env(user_id)
    try:
        with storage.open_database() as conn:
            row = conn.execute("SELECT id FROM Teams WHERE name = ? LIMIT 1", (clean,)).fetchone()
            if row:
                return int(row[0])
            cursor = conn.execute("INSERT INTO Teams (name) VALUES (?)", (clean,))
            conn.commit()
            return int(cursor.lastrowid)
    except sqlite3.OperationalError as exc:
        if "no such table: Teams" not in str(exc):
            raise
        # legacy database without Teams table; rebuild schema then retry once
        storage._ensure_database()
        with storage.open_database() as conn:
            row = conn.execute("SELECT id FROM Teams WHERE name = ? LIMIT 1", (clean,)).fetchone()
            if row:
                return int(row[0])
            cursor = conn.execute("INSERT INTO Teams (name) VALUES (?)", (clean,))
            conn.commit()
            return int(cursor.lastrowid)


def _ensure_default_team(user_id: str) -> int:
    return _get_or_create_team(user_id, "Unassigned")


def _lookup_team_id_by_name(user_id: str, name: str) -> int | None:
    clean = (name or "").strip()
    if not clean:
        return None
    _set_user_env(user_id)
    with storage.open_database(readonly=True) as conn:
        row = conn.execute("SELECT id FROM Teams WHERE name = ? LIMIT 1", (clean,)).fetchone()
        if row:
            return int(row[0])
    return None


def _update_athlete_team(user_id: str, athlete_id: int, team_id: int) -> None:
    _set_user_env(user_id)
    with storage.open_database() as conn:
        conn.execute("UPDATE Athletes SET team_id = ? WHERE id = ?", (team_id, athlete_id))
        conn.commit()


def _record_failed_login(ip: str) -> None:
    now = datetime.utcnow()
    FAILED_LOGINS[ip].append(now)
    cutoff = now - datetime_timedelta(minutes=FAILED_WINDOW_MINUTES)
    FAILED_LOGINS[ip] = [ts for ts in FAILED_LOGINS[ip] if ts >= cutoff]


def _clear_failed_login(ip: str) -> None:
    FAILED_LOGINS.pop(ip, None)


def _is_rate_limited(ip: str) -> bool:
    now = datetime.utcnow()
    cutoff = now - datetime_timedelta(minutes=FAILED_WINDOW_MINUTES)
    attempts = FAILED_LOGINS.get(ip, [])
    recent = [ts for ts in attempts if ts >= cutoff]
    FAILED_LOGINS[ip] = recent
    return len(recent) >= MAX_FAILED_ATTEMPTS


def _load_onboarding_state(user_id: str) -> dict[str, Any]:
    _set_user_env(user_id)
    state_file = WEBAPP_DATA / "userspace" / user_id / "onboarding.json"
    try:
        raw = state_file.read_text()
        return json.loads(raw) if raw else {}
    except Exception:
        return {}


def _save_onboarding_state(user_id: str, state: dict[str, Any]) -> None:
    _set_user_env(user_id)
    state_file = WEBAPP_DATA / "userspace" / user_id / "onboarding.json"
    current = _load_onboarding_state(user_id)
    current.update(state)
    state_file.write_text(json.dumps(current, indent=2))


def _record_report_generated(user_id: str) -> None:
    _save_onboarding_state(
        user_id,
        {
            "last_report_generated": datetime.utcnow().isoformat(),
        },
    )


def _maybe_seed_demo_data(user_id: str) -> bool:
    flag = os.environ.get("ENABLE_DEMO_SEED", "1").lower()
    if flag in {"0", "false", "no"}:
        return False
    athletes = _list_athletes(user_id)
    sessions = _load_sessions(user_id)
    if athletes or sessions:
        return False
    _set_user_env(user_id)
    demo_team_id = _get_or_create_team(user_id, "Demo Team")
    demo_athletes = [
        {"name": "Demo Athlete A", "height_cm": 185, "weight_kg": 86, "team_id": demo_team_id},
        {"name": "Demo Athlete B", "height_cm": 178, "weight_kg": 82, "team_id": demo_team_id},
    ]
    for entry in demo_athletes:
        athlete_id = _get_or_create_athlete(user_id, entry["name"], entry.get("team_id"))
        try:
            storage.update_athlete_profile(
                athlete_id,
                height_cm=entry.get("height_cm"),
                weight_kg=entry.get("weight_kg"),
                notes="Demo profile",
            )
        except Exception:
            continue
    base_date = datetime.utcnow().date()
    demo_sessions = []
    throws_template = [61.2, 62.0, 63.1, 64.4]
    for i in range(7):
        day = base_date - timedelta(days=i)
        athlete = demo_athletes[i % len(demo_athletes)]["name"]
        athlete_team_id = demo_athletes[i % len(demo_athletes)]["team_id"]
        # Give demo athlete A an upward trend and demo athlete B a downward trend
        if athlete.endswith("A"):
            best = 60.0 + i * 0.8
        else:
            best = 66.0 - i * 0.7
        session = services.build_session_from_inputs(
            date_text=day.isoformat(),
            best=best,
            throws=",".join(str(v) for v in throws_template),
            rpe=6 + (i % 3),
            notes="Demo session data to preview analytics.",
            tags="demo, seeded",
            duration_minutes=70 - i,
            athlete=athlete,
            event=DEFAULT_EVENT,
            team="Demo Squad",
            implement_weight_kg=0.8,
            technique="Full run",
            fouls=0,
        )
        session_dict = session.session.to_dict()
        session_dict["id"] = uuid.uuid4().hex[:16]
        session_dict["athlete"] = athlete
        session_dict["athlete_hash"] = uuid.uuid5(uuid.NAMESPACE_DNS, athlete).hex[:12]
        session_dict["event"] = DEFAULT_EVENT
        session_dict["team"] = "Demo Team"
        session_dict["team_id"] = athlete_team_id
        session_dict["tags"] = ["demo", "seeded"]
        demo_sessions.append(session_dict)
    for entry in demo_sessions:
        _append_session(user_id, entry)
        if entry.get("best"):
            try:
                athlete_id = _lookup_athlete_id(user_id, entry.get("athlete") or "")
                if athlete_id:
                    storage.log_throw_distance(athlete_id, entry["date"], entry["event"], float(entry["best"]))
            except Exception:
                pass
    _save_onboarding_state(user_id, {"demo_seeded": True})
    return True


def _purge_demo_data_if_needed(user_id: str) -> None:
    """
    Remove demo data once a real session/athlete exists so coaches are not confused.
    """
    sessions = _load_sessions(user_id)
    athletes = _list_athletes(user_id)
    has_real_session = any(not _is_demo_session(s) for s in sessions)
    has_real_athlete = any(not _is_demo_name(a.get("name", "")) for a in athletes)
    if not (has_real_session or has_real_athlete):
        return
    demo_sessions = [s for s in sessions if _is_demo_session(s)]
    if demo_sessions:
        remaining = [s for s in sessions if not _is_demo_session(s)]
        storage.save_sessions(remaining)
    with storage.open_database() as conn:
        try:
            conn.execute("DELETE FROM Athletes WHERE name LIKE 'Demo %'")
            conn.commit()
        except Exception:
            pass
    state = _load_onboarding_state(user_id)
    state["demo_seeded"] = False
    _save_onboarding_state(user_id, state)


def _compute_onboarding_progress(user_id: str) -> dict[str, Any]:
    athletes = _list_athletes(user_id)
    sessions = _load_sessions(user_id)
    state = _load_onboarding_state(user_id)
    today = datetime.utcnow().date()
    has_real_athletes = any(not _is_demo_name(a.get("name", "")) for a in athletes)
    has_recent_session = False
    for session in sessions:
        if _is_demo_session(session):
            continue
        when = _parse_session_date(session)
        if when and when >= today - timedelta(days=7):
            has_recent_session = True
            break
    has_reports = bool(state.get("last_report_generated"))
    reports_dir = WEBAPP_DATA / "userspace" / user_id / "reports"
    if not has_reports and reports_dir.exists():
        has_reports = any(reports_dir.glob("*.pdf"))
    onboarding_complete = has_real_athletes and has_recent_session and has_reports
    demo_data_active = bool(sessions or athletes) and not has_real_athletes and not has_recent_session
    return {
        "has_athletes": has_real_athletes,
        "has_sessions": has_recent_session,
        "has_reports": has_reports,
        "onboarding_complete": onboarding_complete,
        "demo_data_active": demo_data_active or bool(state.get("demo_seeded")),
    }


def _is_demo_name(name: str) -> bool:
    return name.lower().startswith("demo")


def _is_demo_session(session: dict[str, Any]) -> bool:
    athlete_name = (session.get("athlete") or "").strip().lower()
    tags = session.get("tags") or []
    tag_string = ",".join(tags).lower() if isinstance(tags, list) else str(tags).lower()
    return _is_demo_name(athlete_name) or "demo" in tag_string


def _parse_session_date(session: dict[str, Any]) -> date | None:
    raw = (session.get("date") or "").split("T")[0]
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw).date()
    except Exception:
        return None


# Convenience entry point for `flask run` style discovery
app = create_app()
