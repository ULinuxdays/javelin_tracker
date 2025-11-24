from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, date
from functools import wraps
from pathlib import Path
from typing import Any, Iterable

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

from .. import analysis, services, storage
from ..reports import generate_weekly_report
from ..constants import DEFAULT_ATHLETE_PLACEHOLDER
from ..env import LEGACY_PREFIX, PRIMARY_PREFIX
from ..models import DEFAULT_EVENT, Session

BASE_DIR = Path(__file__).resolve().parents[2]
WEBAPP_DATA = BASE_DIR / "data" / "webapp"
USERS_FILE = WEBAPP_DATA / "users.json"


def create_app() -> Flask:
    WEBAPP_DATA.mkdir(parents=True, exist_ok=True)
    app = Flask(
        __name__,
        static_folder=str(Path(__file__).parent / "static"),
        template_folder=str(Path(__file__).parent / "templates"),
    )
    app.secret_key = os.environ.get("THROWS_TRACKER_SECRET", "dev-secret")

    @app.before_request
    def load_user() -> None:
        user_id = flask_session.get("user_id")
        g.user = None
        if user_id:
            g.user = _get_user_by_id(user_id)
            if g.user:
                # Ensure every request is scoped to the active user's datastore
                _set_user_env(user_id)

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
            user = _get_user_by_email(email)
            if not user or not check_password_hash(user["password_hash"], password):
                error = "Invalid email or password."
            else:
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
        return render_template("dashboard.html", page_slug="dashboard")

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
        return render_template("analytics.html", page_slug="analytics")

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
        return render_template("reports.html", page_slug="reports")

    @app.route("/athletes")
    @login_required
    def athletes_page():
        return render_template("athletes.html", page_slug="athletes")


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
        athlete_id = _get_or_create_athlete(g.user["id"], athlete_name)
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
                team=payload.get("team"),
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
        _append_session(g.user["id"], session_dict)
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
        week_ending_raw = (payload.get("week_ending") or "").strip()
        week_ending: date | None = None
        if week_ending_raw:
            try:
                week_ending = datetime.fromisoformat(week_ending_raw).date()
            except Exception:
                return jsonify({"error": "week_ending must be YYYY-MM-DD"}), 400

        _set_user_env(g.user["id"])
        sessions = storage.load_sessions()
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
            storage.update_athlete_profile(
                athlete_id,
                height_cm=_float_or_none(payload.get("height_cm")),
                weight_kg=_float_or_none(payload.get("weight_kg")),
                new_strength_benchmarks=benchmarks or None,
                notes=payload.get("notes"),
            )
        except Exception as exc:  # pragma: no cover
            return jsonify({"error": str(exc)}), 400
        athlete = _fetch_athlete_profile(g.user["id"], athlete_id)
        return jsonify({"athlete": athlete})


def _set_user_env(user_id: str) -> None:
    user_dir = WEBAPP_DATA / "userspace" / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    db_file = user_dir / "throws_tracker.db"
    sessions_file = user_dir / "sessions.json"
    os.environ[f"{PRIMARY_PREFIX}DATA_DIR"] = str(user_dir)
    os.environ[f"{PRIMARY_PREFIX}DB_FILE"] = str(db_file)
    os.environ[f"{PRIMARY_PREFIX}SESSIONS_FILE"] = str(sessions_file)
    os.environ[f"{LEGACY_PREFIX}DATA_DIR"] = str(user_dir)
    os.environ[f"{LEGACY_PREFIX}DB_FILE"] = str(db_file)
    os.environ[f"{LEGACY_PREFIX}SESSIONS_FILE"] = str(sessions_file)


def _load_sessions(user_id: str) -> list[dict[str, Any]]:
    _set_user_env(user_id)
    sessions = storage.load_sessions()
    # Fill required defaults for downstream analytics
    for record in sessions:
        record.setdefault("event", DEFAULT_EVENT)
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
            "SELECT id, name, height_cm, weight_kg, strength_benchmarks, notes FROM Athletes ORDER BY name"
        ).fetchall()
        for row in rows:
            athlete_id, name, height, weight, benchmarks_json, notes = row
            entry: dict[str, Any] = {"id": athlete_id, "name": name}
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
            return user
    return None


def _get_user_by_id(user_id: str) -> dict[str, Any] | None:
    users = _load_users()
    for user in users:
        if user["id"] == user_id:
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


def _get_or_create_athlete(user_id: str, name: str) -> int:
    clean = (name or DEFAULT_ATHLETE_PLACEHOLDER).strip() or DEFAULT_ATHLETE_PLACEHOLDER
    _set_user_env(user_id)
    with storage.open_database() as conn:
        row = conn.execute("SELECT id FROM Athletes WHERE name = ? LIMIT 1", (clean,)).fetchone()
        if row:
            return int(row[0])
        cursor = conn.execute("INSERT INTO Athletes (name) VALUES (?)", (clean,))
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


# Convenience entry point for `flask run` style discovery
app = create_app()
