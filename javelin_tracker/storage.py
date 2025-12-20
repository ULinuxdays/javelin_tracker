from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterable, Iterator, List, Mapping, Tuple
import sqlite3
import numpy as np
import logging

from .env import get_env
from .models import (
    CURRENT_SCHEMA_VERSION,
    DEFAULT_EVENT,
    ValidationError,
    calculate_bmi,
    calculate_session_load,
    parse_iso_date,
    validate_duration,
)
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_DB_FILENAME = "throws_tracker.db"
_DB_INITIALISED_FOR: Path | None = None
LOGGER = logging.getLogger(__name__)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _data_dir() -> Path:
    override = get_env("DATA_DIR")
    base = Path(override).expanduser() if override else DEFAULT_DATA_DIR
    base.mkdir(parents=True, exist_ok=True)
    return base


def _sessions_file() -> Path:
    override = get_env("SESSIONS_FILE")
    if override:
        target = Path(override).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        return target
    return _data_dir() / "sessions.json"


def _save_sessions_to_file(sessions_file: Path, sessions: Iterable[Any]) -> None:
    sessions_file.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(list(sessions), indent=2, sort_keys=True) + "\n"

    with NamedTemporaryFile("w", dir=sessions_file.parent, delete=False, encoding="utf-8") as tmp:
        tmp.write(payload)
        temp_path = Path(tmp.name)
    temp_path.replace(sessions_file)


def _load_sessions_from_file(sessions_file: Path) -> List[Any]:
    if not sessions_file.exists():
        sessions_file.parent.mkdir(parents=True, exist_ok=True)
        sessions_file.write_text("[]\n", encoding="utf-8")
        return []

    raw = sessions_file.read_text(encoding="utf-8").strip() or "[]"
    try:
        sessions = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse {sessions_file}: {exc}") from exc

    if not isinstance(sessions, list):
        raise ValueError(f"{sessions_file} must contain a JSON list")

    upgraded, changed = _migrate_sessions(sessions)
    if changed:
        _save_sessions_to_file(sessions_file, upgraded)
    return upgraded


def _database_file() -> Path:
    override = get_env("DB_FILE")
    if override:
        path = Path(override).expanduser()
    else:
        path = _data_dir() / DEFAULT_DB_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, alter_sql: str) -> None:
    try:
        cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
        if any(col[1] == column for col in cols):
            return
        conn.execute(alter_sql)
        conn.commit()
    except Exception:
        return


def load_sessions() -> List[Any]:
    return _load_sessions_from_file(_sessions_file())


def _ensure_database() -> None:
    global _DB_INITIALISED_FOR
    db_path = _database_file()
    if _DB_INITIALISED_FOR is not None:
        try:
            if Path(_DB_INITIALISED_FOR).resolve() == db_path.resolve():
                return
        except Exception:
            if Path(_DB_INITIALISED_FOR) == db_path:
                return
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            PRAGMA foreign_keys = ON;

            CREATE TABLE IF NOT EXISTS Teams (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                program_name TEXT,
                school_name TEXT,
                color TEXT,
                short_code TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS Athletes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                team_id INTEGER,
                throwing_style TEXT,
                height_cm REAL,
                weight_kg REAL,
                bmi REAL,
                strength_benchmarks TEXT,
                throw_distances TEXT,
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (team_id) REFERENCES Teams(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS StrengthLogs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                athlete_id INTEGER NOT NULL,
                team_id INTEGER,
                logged_at TEXT NOT NULL,
                exercise TEXT NOT NULL,
                load_kg REAL,
                reps INTEGER,
                notes TEXT,
                FOREIGN KEY (athlete_id) REFERENCES Athletes(id) ON DELETE CASCADE,
                FOREIGN KEY (team_id) REFERENCES Teams(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS ThrowLogs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                athlete_id INTEGER NOT NULL,
                event TEXT NOT NULL,
                logged_at TEXT NOT NULL,
                distance REAL NOT NULL,
                attempt INTEGER,
                notes TEXT,
                FOREIGN KEY (athlete_id) REFERENCES Athletes(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_strengthlogs_athlete_date
                ON StrengthLogs (athlete_id, logged_at);

            CREATE INDEX IF NOT EXISTS idx_throwlogs_athlete_date
                ON ThrowLogs (athlete_id, logged_at);

            CREATE TABLE IF NOT EXISTS AthleteTrends (
                athlete_id INTEGER PRIMARY KEY,
                snapshot TEXT NOT NULL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (athlete_id) REFERENCES Athletes(id) ON DELETE CASCADE
            );
            """
        )
        _ensure_column(conn, "Athletes", "team_id", "ALTER TABLE Athletes ADD COLUMN team_id INTEGER")
        _ensure_column(conn, "Athletes", "throwing_style", "ALTER TABLE Athletes ADD COLUMN throwing_style TEXT")
        _ensure_column(conn, "StrengthLogs", "team_id", "ALTER TABLE StrengthLogs ADD COLUMN team_id INTEGER")
    finally:
        conn.close()
    _DB_INITIALISED_FOR = db_path


def save_sessions(sessions: Iterable[Any]) -> None:
    _save_sessions_to_file(_sessions_file(), sessions)


def get_session_by_id(session_id: str, *, sessions_file: Path | str | None = None) -> dict[str, Any] | None:
    """Return a session record matching `id` or None when missing."""
    if not session_id:
        return None
    path = Path(sessions_file).expanduser() if sessions_file is not None else _sessions_file()
    sessions = _load_sessions_from_file(path) if sessions_file is not None else load_sessions()
    for record in sessions:
        if isinstance(record, dict) and str(record.get("id")) == str(session_id):
            return dict(record)
    return None


def update_session_by_id(
    session_id: str,
    patch: Mapping[str, Any],
    *,
    sessions_file: Path | str | None = None,
) -> dict[str, Any] | None:
    """Update a session record in place and persist changes."""
    if not session_id:
        return None
    path = Path(sessions_file).expanduser() if sessions_file is not None else _sessions_file()
    sessions = _load_sessions_from_file(path) if sessions_file is not None else load_sessions()

    updated: dict[str, Any] | None = None
    for record in sessions:
        if not isinstance(record, dict) or str(record.get("id")) != str(session_id):
            continue
        record.update(dict(patch))
        updated = dict(record)
        break

    if updated is None:
        return None

    if sessions_file is not None:
        _save_sessions_to_file(path, sessions)
    else:
        save_sessions(sessions)
    return updated


def _project_root() -> Path:
    return DEFAULT_DATA_DIR.parent


def _resolve_project_path(path: str | Path) -> Path:
    p = path if isinstance(path, Path) else Path(path)
    return p if p.is_absolute() else _project_root() / p


def get_session_biomechanics(session_id: str, *, sessions_file: Path | str | None = None) -> dict[str, Any] | None:
    """Load biomechanics artifacts (metrics, report, feedback) for a session."""
    session = get_session_by_id(session_id, sessions_file=sessions_file)
    if session is None:
        return None

    raw_path = session.get("biomechanics_result_path")
    metrics_path: Path
    if isinstance(raw_path, str) and raw_path.strip():
        metrics_path = _resolve_project_path(raw_path.strip())
    else:
        metrics_path = _project_root() / "data" / "biomechanics" / "sessions" / str(session_id) / "metrics.json"

    output: dict[str, Any] = {
        "session_id": str(session_id),
        "video_id": session.get("video_id"),
        "biomechanics_analysis_id": session.get("biomechanics_analysis_id"),
        "biomechanics_status": session.get("biomechanics_status"),
        "biomechanics_timestamp": session.get("biomechanics_timestamp"),
        "biomechanics_result_path": str(metrics_path) if metrics_path else None,
        "metrics": None,
        "comparison_report": None,
        "feedback": None,
        "errors": [],
    }

    def _try_load_json(path: Path) -> Any:
        return json.loads(path.read_text(encoding="utf-8"))

    if metrics_path.exists():
        try:
            output["metrics"] = _try_load_json(metrics_path)
        except Exception as exc:
            output["errors"].append(f"metrics: {exc}")
    else:
        output["errors"].append(f"metrics_missing: {metrics_path}")

    report_path = metrics_path.with_name("comparison_report.json")
    if report_path.exists():
        try:
            output["comparison_report"] = _try_load_json(report_path)
        except Exception as exc:
            output["errors"].append(f"comparison_report: {exc}")

    feedback_path = metrics_path.with_name("feedback.json")
    if feedback_path.exists():
        try:
            output["feedback"] = _try_load_json(feedback_path)
        except Exception as exc:
            output["errors"].append(f"feedback: {exc}")

    return output


def append_session(session: Any) -> List[Any]:
    sessions = load_sessions()
    sessions.append(session)
    save_sessions(sessions)
    return sessions


def export_csv(path: Path | str) -> Path:
    sessions = load_sessions()
    flattened = [_flatten(record) for record in sessions]
    fieldnames = sorted({key for row in flattened for key in row.keys()})

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flattened)
    return target


def export_sessions_csv(path: Path | str, sessions: Iterable[dict[str, Any]]) -> Path:
    flattened = [_flatten(record) for record in sessions]
    fieldnames = sorted({key for row in flattened for key in row.keys()})
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flattened)
    return target


def _flatten(payload: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(payload, dict):
        items = {}
        for key, value in payload.items():
            full_key = f"{prefix}.{key}" if prefix else str(key)
            items.update(_flatten(value, full_key))
        return items

    if isinstance(payload, list):
        if all(isinstance(item, dict) for item in payload):
            items = {}
            for index, value in enumerate(payload):
                full_key = f"{prefix}[{index}]"
                items.update(_flatten(value, full_key))
            return items
        return {prefix: ";".join(map(str, payload)) if prefix else list(payload)}

    return {prefix: payload} if prefix else {"value": payload}


@contextmanager
def open_database(readonly: bool = False) -> Iterator[sqlite3.Connection]:
    """Context manager yielding a SQLite connection with ensured schema."""
    _ensure_database()
    db_path = _database_file()
    if readonly:
        uri = f"file:{db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
    else:
        conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
    finally:
        conn.close()


def log_strength_workout(
    athlete_id: int,
    logged_date: Any,
    exercise: str,
    weight_kg: float,
    reps: int,
    notes: str | None = None,
) -> int:
    """
    Append a strength-training record for the specified athlete.

    Parameters
    ----------
    athlete_id:
        Primary key of the athlete in the Athletes table.
    logged_date:
        ISO8601 string or date-like object for when the session occurred.
    exercise:
        Name of the lift (e.g., "Back Squat").
    weight_kg:
        Load lifted in kilograms.
    reps:
        Repetitions completed.
    notes:
        Optional free-text notes about the set.
    """
    exercise_name = (exercise or "").strip()
    if not exercise_name:
        raise ValueError("exercise is required.")
    iso_date = parse_iso_date(logged_date, field="logged_at").isoformat()
    load_value = float(weight_kg)
    reps_value = int(reps)

    with open_database() as conn:
        cursor = conn.execute(
            """
            INSERT INTO StrengthLogs (athlete_id, logged_at, exercise, load_kg, reps, notes)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (athlete_id, iso_date, exercise_name, load_value, reps_value, notes),
        )
        conn.commit()
        row_id = cursor.lastrowid
    _refresh_profile_and_trends(athlete_id)
    return row_id


def log_throw_distance(
    athlete_id: int,
    logged_date: Any,
    event: str,
    distance: float,
    attempt: int | None = None,
    notes: str | None = None,
) -> int:
    """
    Record a throwing performance for the specified athlete.

    Parameters
    ----------
    athlete_id:
        Primary key in Athletes table.
    logged_date:
        ISO date string or date-like object when the throw was recorded.
    event:
        Event label (e.g., "javelin", "shot", "discus").
    distance:
        Best distance in metres for the attempt/session.
    attempt:
        Optional attempt index for multi-throw sessions.
    notes:
        Optional notes about wind, runway, etc.
    """
    event_name = (event or "").strip().lower()
    if not event_name:
        raise ValueError("event is required.")
    iso_date = parse_iso_date(logged_date, field="logged_at").isoformat()
    distance_value = float(distance)

    with open_database() as conn:
        cursor = conn.execute(
            """
            INSERT INTO ThrowLogs (athlete_id, event, logged_at, distance, attempt, notes)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (athlete_id, event_name, iso_date, distance_value, attempt, notes),
        )
        conn.commit()
        row_id = cursor.lastrowid
    _refresh_profile_and_trends(athlete_id)
    return row_id


def update_athlete_profile(
    athlete_id: int,
    height_cm: float | None = None,
    weight_kg: float | None = None,
    new_strength_benchmarks: dict[str, float] | None = None,
    notes: str | None = None,
) -> None:
    """
    Update core athlete attributes as new measurements are captured.

    Parameters
    ----------
    athlete_id:
        Primary key for the athlete row.
    height_cm / weight_kg:
        Optional new measurements. BMI is recalculated automatically when both
        height and weight are present in the payload or existing record.
    new_strength_benchmarks:
        Dict of updated max lifts (exercise -> kg). Merged with existing data.
    notes:
        Optional note string to overwrite the stored value.
    """
    updates: list[str] = []
    params: list[Any] = []

    with open_database() as conn:
        cursor = conn.execute(
            "SELECT height_cm, weight_kg, strength_benchmarks FROM Athletes WHERE id = ?",
            (athlete_id,),
        )
        row = cursor.fetchone()
        if row is None:
            raise ValueError(f"Athlete {athlete_id} not found.")

        current_height, current_weight, strength_json = row
        updated_height = height_cm if height_cm is not None else current_height
        updated_weight = weight_kg if weight_kg is not None else current_weight

        if height_cm is not None:
            updates.append("height_cm = ?")
            params.append(float(height_cm))
        if weight_kg is not None:
            updates.append("weight_kg = ?")
            params.append(float(weight_kg))

        bmi_value: float | None = None
        if updated_height and updated_weight:
            try:
                bmi_value = calculate_bmi(float(updated_height), float(updated_weight))
            except ValidationError:
                bmi_value = None
        if bmi_value is not None:
            updates.append("bmi = ?")
            params.append(bmi_value)

        if new_strength_benchmarks:
            existing = json.loads(strength_json) if strength_json else {}
            existing.update(new_strength_benchmarks)
            updates.append("strength_benchmarks = ?")
            params.append(json.dumps(existing, sort_keys=True))

        if notes is not None:
            updates.append("notes = ?")
            params.append(notes)

        if not updates:
            return

        updates.append("updated_at = CURRENT_TIMESTAMP")
        sql = f"UPDATE Athletes SET {', '.join(updates)} WHERE id = ?"
        params.append(athlete_id)
        conn.execute(sql, params)
        conn.commit()


def analyze_trends(
    athlete_id: int,
    *,
    recent_days: int = 28,
    baseline_days: int = 28,
) -> dict[str, Any]:
    """
    Inspect throw and strength history to surface trends/plateaus.

    Returns rolling averages and simple linear trend slopes (per day).
    """
    now = _now_utc()
    recent_start = now - timedelta(days=recent_days)
    baseline_start = recent_start - timedelta(days=baseline_days)

    with open_database(readonly=True) as conn:
        strength_rows = conn.execute(
            """
            SELECT logged_at, load_kg, reps
            FROM StrengthLogs
            WHERE athlete_id = ? AND load_kg IS NOT NULL
            ORDER BY logged_at
            """,
            (athlete_id,),
        ).fetchall()
        throw_rows = conn.execute(
            """
            SELECT logged_at, distance
            FROM ThrowLogs
            WHERE athlete_id = ?
            ORDER BY logged_at
            """,
            (athlete_id,),
        ).fetchall()

    strength_records = _strength_time_series(strength_rows)
    throw_records = _throw_time_series(throw_rows)

    strength_recent = _window_values(strength_records, recent_start, now)
    strength_baseline = _window_values(strength_records, baseline_start, recent_start)
    throws_recent = _window_values(throw_records, recent_start, now)
    throws_baseline = _window_values(throw_records, baseline_start, recent_start)

    strength_trend = {
        "recent_avg_load": _average(strength_recent),
        "baseline_avg_load": _average(strength_baseline),
        "load_change_pct": _percent_change(
            _average(strength_recent), _average(strength_baseline)
        ),
        "trend_slope_per_day": _trend_slope(strength_records),
        "coefficient_of_variation": _coefficient_of_variation(strength_recent),
        "effect_size": _effect_size(strength_recent, strength_baseline),
        "n_recent": len(strength_recent),
        "n_baseline": len(strength_baseline),
    }
    throw_trend = {
        "recent_avg_distance": _average(throws_recent),
        "baseline_avg_distance": _average(throws_baseline),
        "distance_change_pct": _percent_change(
            _average(throws_recent), _average(throws_baseline)
        ),
        "trend_slope_per_day": _trend_slope(throw_records),
        "coefficient_of_variation": _coefficient_of_variation(throws_recent),
        "effect_size": _effect_size(throws_recent, throws_baseline),
        "n_recent": len(throws_recent),
        "n_baseline": len(throws_baseline),
    }

    return {
        "strength": strength_trend,
        "throws": throw_trend,
        "recent_window_days": recent_days,
        "baseline_window_days": baseline_days,
    }


def _coerce_timestamp(value: Any) -> datetime | None:
    """Parse timestamps and normalise to UTC."""
    try:
        timestamp = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    else:
        timestamp = timestamp.astimezone(timezone.utc)
    return timestamp


def _collapse_by_day(
    records: list[tuple[datetime, float]],
    reducer,
) -> list[tuple[datetime, float]]:
    """Collapse multiple entries on the same date using the supplied reducer."""
    buckets: dict[datetime.date, list[float]] = {}
    tzinfo = timezone.utc
    for ts, value in records:
        tzinfo = ts.tzinfo or tzinfo
        buckets.setdefault(ts.date(), []).append(float(value))
    collapsed: list[tuple[datetime, float]] = []
    for day, values in buckets.items():
        combined = reducer(values)
        collapsed.append((datetime.combine(day, datetime.min.time(), tzinfo), float(combined)))
    collapsed.sort(key=lambda item: item[0])
    return collapsed


def _strength_time_series(rows: list[tuple[Any, Any, Any]]) -> list[tuple[datetime, float]]:
    """
    Convert strength logs into a per-day tonnage series (load × reps).
    Multiple sets on the same day are summed to represent total stimulus.
    """
    entries: list[tuple[datetime, float]] = []
    for logged_at, load, reps in rows:
        ts = _coerce_timestamp(logged_at)
        if ts is None or load is None:
            continue
        try:
            rep_count = max(1, int(reps)) if reps is not None else 1
        except (TypeError, ValueError):
            rep_count = 1
        try:
            value = float(load) * rep_count
        except (TypeError, ValueError):
            continue
        entries.append((ts, value))
    return _collapse_by_day(entries, sum)


def _throw_time_series(rows: list[tuple[Any, Any]]) -> list[tuple[datetime, float]]:
    """
    Convert throw logs into a per-day best-distance series.
    """
    entries: list[tuple[datetime, float]] = []
    for logged_at, value in rows:
        ts = _coerce_timestamp(logged_at)
        if ts is None or value is None:
            continue
        try:
            distance = float(value)
        except (TypeError, ValueError):
            continue
        entries.append((ts, distance))
    # Use max distance per day to emphasise performance signal.
    return _collapse_by_day(entries, max)


def _to_time_series(rows: list[tuple[str, float]]) -> list[tuple[datetime, float]]:
    """Legacy helper used elsewhere (no daily collapsing)."""
    series: list[tuple[datetime, float]] = []
    for logged_at, value in rows:
        if value is None:
            continue
        ts = _coerce_timestamp(logged_at)
        if ts is None:
            continue
        series.append((ts, float(value)))
    return series


def _window_values(
    records: list[tuple[datetime, float]],
    start: datetime,
    end: datetime,
) -> list[float]:
    return [value for logged_at, value in records if start <= logged_at < end]


def _average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _variance(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = _average(values)
    if mean is None:
        return None
    return sum((value - mean) ** 2 for value in values) / len(values)


def _percent_change(current: float | None, previous: float | None) -> float | None:
    if current is None or previous is None or previous == 0:
        return None
    return ((current - previous) / previous) * 100.0


def _coefficient_of_variation(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = _average(values)
    if mean is None or mean == 0:
        return None
    var = _variance(values)
    if var is None:
        return None
    return math.sqrt(var) / mean


def _effect_size(recent: list[float], baseline: list[float]) -> float | None:
    """
    Compute a simple Cohen's d between recent and baseline windows.
    """
    if len(recent) < 2 or len(baseline) < 2:
        return None
    mean_recent = _average(recent)
    mean_base = _average(baseline)
    if mean_recent is None or mean_base is None:
        return None
    var_recent = _variance(recent)
    var_base = _variance(baseline)
    if var_recent is None or var_base is None:
        return None
    pooled = ((len(recent) - 1) * var_recent + (len(baseline) - 1) * var_base) / max(
        1, (len(recent) + len(baseline) - 2)
    )
    if pooled <= 0:
        return None
    return (mean_recent - mean_base) / math.sqrt(pooled)


def _trend_slope(records: list[tuple[datetime, float]]) -> float | None:
    if len(records) < 2:
        return None
    xs = [ts.timestamp() for ts, _ in records]
    ys = [value for _, value in records]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        return None
    slope_per_second = numerator / denominator
    return slope_per_second * 86400  # convert to per-day change


def adjust_workload(
    athlete_profile: Mapping[str, Any],
    recent_performance: Mapping[str, Any],
    *,
    strength_threshold_pct: float = 5.0,
    throw_threshold_pct: float = 2.0,
    strength_increment_pct: float = 0.025,
    throw_increment_pct: float = 0.01,
) -> dict[str, Any]:
    """
    Suggest progressive overload adjustments based on recent trends.

    If the athlete improved beyond the specified thresholds, nudge the
    recommended load/reps or distance slightly higher to maintain stimulus.
    """
    strength_info = recent_performance.get("strength", {}) if recent_performance else {}
    throw_info = recent_performance.get("throws", {}) if recent_performance else {}

    benchmarks = _coerce_mapping(athlete_profile.get("strength_benchmarks"))
    throw_history = _coerce_mapping(athlete_profile.get("throw_distances"))

    recommendations: dict[str, Any] = {"strength": {}, "throws": {}}

    strength_effect = strength_info.get("effect_size")
    throw_effect = throw_info.get("effect_size")
    strength_cv = strength_info.get("coefficient_of_variation")
    throw_cv = throw_info.get("coefficient_of_variation")
    strength_samples = strength_info.get("n_recent") or 0
    throw_samples = throw_info.get("n_recent") or 0

    strength_change = strength_info.get("load_change_pct")
    if (
        strength_change is not None
        and strength_change >= strength_threshold_pct
        and benchmarks
        and strength_samples >= 3
        and (strength_effect is None or strength_effect >= 0.15)
    ):
        suggestions: dict[str, dict[str, float]] = {}
        for lift, current_max in benchmarks.items():
            try:
                current_value = float(current_max)
            except (TypeError, ValueError):
                continue
            effect_multiplier = 1 + min(0.6, max(0.0, (strength_effect or 0.0)))
            target = current_value * (1 + strength_increment_pct * effect_multiplier)
            suggestions[lift] = {
                "current_max": current_value,
                "target_max": round(target, 2),
                "increase_pct": round(strength_increment_pct * effect_multiplier * 100.0, 2),
                "signal_strength": round(strength_effect, 2) if strength_effect is not None else None,
            }
        if suggestions:
            recommendations["strength"] = {
                "message": "Performance trending up with a meaningful signal; increase loads slightly to maintain overload stimulus.",
                "suggestions": suggestions,
            }

    throw_change = throw_info.get("distance_change_pct")
    if (
        throw_change is not None
        and throw_change >= throw_threshold_pct
        and throw_history
        and throw_samples >= 3
        and (throw_effect is None or throw_effect >= 0.1)
    ):
        throw_suggestions: dict[str, dict[str, float]] = {}
        for event, distances in throw_history.items():
            values = distances if isinstance(distances, list) else []
            numeric = [float(value) for value in values if isinstance(value, (int, float))]
            if not numeric:
                continue
            recent_best = max(numeric)
            effect_multiplier = 1 + min(0.5, max(0.0, (throw_effect or 0.0)))
            target = recent_best * (1 + throw_increment_pct * effect_multiplier)
            throw_suggestions[event] = {
                "recent_best": recent_best,
                "target_best": round(target, 2),
                "increase_pct": round(throw_increment_pct * effect_multiplier * 100.0, 2),
                "signal_strength": round(throw_effect, 2) if throw_effect is not None else None,
            }
        if throw_suggestions:
            recommendations["throws"] = {
                "message": "Throw distances improving; aim for a modest increase next session.",
                "suggestions": throw_suggestions,
            }

    # Add readiness multipliers (0.9–1.1) derived from slopes and deltas for more nuanced dosing.
    def _clip(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    strength_slope = strength_info.get("trend_slope_per_day") or 0.0
    throw_slope = throw_info.get("trend_slope_per_day") or 0.0
    strength_delta = (strength_info.get("load_change_pct") or 0.0) / 100.0
    throw_delta = (throw_info.get("distance_change_pct") or 0.0) / 100.0
    variability_penalty_strength = 0.0
    variability_penalty_throw = 0.0
    if strength_cv and strength_cv > 0.35:
        variability_penalty_strength = min(0.08, (strength_cv - 0.35) * 0.2)
    if throw_cv and throw_cv > 0.25:
        variability_penalty_throw = min(0.06, (throw_cv - 0.25) * 0.15)

    # Convert to soft multipliers: small increases for upward trends, small decreases for downward trends.
    strength_factor = 1.0 + _clip(
        0.5 * strength_delta + 0.0005 * strength_slope + 0.05 * (strength_effect or 0.0) - variability_penalty_strength,
        -0.12,
        0.12,
    )
    throw_factor = 1.0 + _clip(
        0.5 * throw_delta + 0.0005 * throw_slope + 0.04 * (throw_effect or 0.0) - variability_penalty_throw,
        -0.1,
        0.1,
    )

    recommendations["strength"]["readiness_multiplier"] = round(strength_factor, 3)
    recommendations["throws"]["readiness_multiplier"] = round(throw_factor, 3)

    # Flag deload when trends decline notably
    recommendations["deload"] = bool(
        (strength_delta < -0.05)
        or (throw_delta < -0.03)
        or (strength_effect is not None and strength_effect < -0.35)
        or (throw_effect is not None and throw_effect < -0.25)
        or (strength_cv is not None and strength_cv > 0.45)
    )

    return recommendations


def _coerce_mapping(payload: Any) -> dict[str, Any]:
    if payload is None:
        return {}
    if isinstance(payload, str):
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}
    if isinstance(payload, Mapping):
        return dict(payload)
    return {}


def generate_workout(
    athlete_profile: Mapping[str, Any],
    session_date: Any | None = None,
) -> list[dict[str, Any]]:
    """
    Assemble a personalised workout plan using profile metrics and load trends.
    """
    profile = dict(athlete_profile or {})
    benchmarks = _coerce_mapping(profile.get("strength_benchmarks"))
    throw_history = _coerce_mapping(profile.get("throw_distances"))
    bmi = profile.get("bmi")
    height = profile.get("height_cm")
    weight = profile.get("weight_kg")
    if bmi in (None, "", 0) and height and weight:
        try:
            bmi = calculate_bmi(float(height), float(weight))
        except ValidationError:
            bmi = None

    athlete_id = profile.get("id") or profile.get("athlete_id")
    trends = analyze_trends(athlete_id) if athlete_id else None
    throw_trend_info = trends.get("throws") if trends else {}
    workload = adjust_workload(profile, trends) if trends else {}
    # Optional per-lift suggestions (kept for compatibility)
    strength_adjustments = workload.get("strength", {}).get("suggestions", {})
    _ = strength_adjustments

    # Adjust training intensity for higher BMI athletes to manage joint load.
    bmi_modifier = 1.0
    if bmi and bmi >= 32:
        bmi_modifier = 0.92
    elif bmi and bmi <= 20:
        bmi_modifier = 1.05

    # Daily undulating variation: heavy / moderate / light based on weekday.
    today = _now_utc().date() if session_date is None else (
        session_date.date() if isinstance(session_date, datetime) else session_date
    )
    try:
        weekday = today.weekday()  # 0=Mon
    except Exception:
        weekday = 0

    if weekday in (0, 3):  # Mon/Thu: strength focus
        rep_schemes = {
            "back squat": (4, 6, 2),  # sets, reps, RIR
            "bench press": (4, 6, 2),
            "deadlift": (3, 5, 2),
            "push press": (4, 5, 2),
        }
    elif weekday in (1, 5):  # Tue/Sat: power/speed
        rep_schemes = {
            "back squat": (5, 3, 3),
            "bench press": (6, 3, 3),
            "deadlift": (4, 3, 3),
            "push press": (6, 3, 2),
        }
    else:  # Wed/Fri/Sun: hypertrophy/technique
        rep_schemes = {
            "back squat": (4, 10, 3),
            "bench press": (4, 10, 3),
            "deadlift": (3, 8, 3),
            "push press": (3, 8, 3),
        }

    template = list(rep_schemes.items())  # (name, (sets, reps, RIR))

    session_plan: list[dict[str, Any]] = []
    readiness_mult = (workload.get("strength", {}) or {}).get("readiness_multiplier", 1.0)
    deload = workload.get("deload", False)
    strength_scale = readiness_mult * (0.92 if deload else 1.0) * bmi_modifier

    # Helper to estimate 1RM from logs if no benchmark present.
    def _estimate_one_rm(exercise: str) -> float | None:
        value = benchmarks.get(exercise)
        try:
            if isinstance(value, str):
                value = float(value)
        except ValueError:
            value = None
        if value and value > 0:
            # Treat benchmark as best set load; approximate 1RM conservatively by +10%.
            return float(value) * 1.10
        # Fall back to log-derived estimate
        try:
            with open_database(readonly=True) as conn:
                rows = conn.execute(
                    "SELECT load_kg, reps FROM StrengthLogs WHERE athlete_id = ? AND exercise = ? AND load_kg IS NOT NULL AND reps IS NOT NULL",
                    (athlete_id, exercise),
                ).fetchall()
        except Exception:
            rows = []
        ests: list[float] = []
        for load, reps in rows:
            try:
                load_v = float(load)
                reps_v = int(reps)
            except (TypeError, ValueError):
                continue
            # Epley: 1RM ≈ w * (1 + r/30)
            ests.append(load_v * (1.0 + reps_v / 30.0))
        return max(ests) if ests else None

    for name, (sets, target_reps, rir) in template:
        one_rm = _estimate_one_rm(name)
        if one_rm:
            # Compute training weight targeting RIR using Epley inverse: w = 1RM / (1 + (reps+RIR)/30)
            denom = 1.0 + (target_reps + max(0, int(rir))) / 30.0
            base_weight = one_rm / denom
            training_weight = max(0.0, base_weight * strength_scale)
            session_plan.append(
                {
                    "name": name,
                    "sets": sets,
                    "reps": target_reps,
                    "rir_target": rir,
                    "target_weight_kg": round(training_weight, 1),
                    "intensity_pct_1rm": round((training_weight / one_rm) * 100.0, 1),
                    "is_strength": True,
                }
            )
        else:
            # Bodyweight tempo fallback
            session_plan.append(
                {
                    "name": f"Tempo {name.title()} (bodyweight)",
                    "sets": sets,
                    "reps": target_reps + 2,
                    "target_weight_kg": None,
                    "is_strength": True,
                }
            )

    # Add accessory/core work scaled by BMI.
    accessory_reps = 12 if bmi and bmi < 25 else 10
    session_plan.extend(
        [
            {"name": "Pull-ups or Assisted Rows", "sets": 3, "reps": accessory_reps},
            {"name": "Single-leg RDL", "sets": 3, "reps": accessory_reps},
            {"name": "Med-ball Rotational Throw", "sets": 4, "reps": 8},
            {"name": "Plank", "sets": 3, "duration_seconds": 60},
        ]
    )

    # Event-specific target derived from throw history and readiness.
    if throw_history:
        primary_event = max(
            throw_history.items(),
            key=lambda item: max(
                (float(val) for val in (item[1] if isinstance(item[1], list) else [])),
                default=0,
            ),
        )[0]
        distances = throw_history.get(primary_event, [])
        numeric = [
            float(value)
            for value in (distances if isinstance(distances, list) else [])
            if isinstance(value, (int, float))
        ]
        if numeric:
            best = max(numeric)
            increment_pct = workload.get("throws", {}).get("suggestions", {}).get(
                primary_event, {}
            ).get("increase_pct", 1.0)
            throw_readiness = (workload.get("throws", {}) or {}).get("readiness_multiplier", 1.0)
            target = best * (1 + (increment_pct / 100.0)) * throw_readiness
            session_plan.append(
                {
                    "name": f"{primary_event.title()} technical session",
                    "sets": 6,
                    "reps": 3,
                    "target_distance_m": round(target, 2),
                }
            )
            # Add weighted ball/implement drill scaling with best throw.
            throw_change = throw_trend_info.get("distance_change_pct")
            # Heavier implements when plateau/downtrend; lighter for speed when trending up.
            if throw_change is None:
                drill_load_pct = 0.14
            elif throw_change >= 2:
                drill_load_pct = 0.10
            elif throw_change <= -2:
                drill_load_pct = 0.18
            else:
                drill_load_pct = 0.12
            if throw_change is None or throw_change <= 0:
                drill_sets = 5
                drill_reps = 6
                drill_note = "Distance plateau detected; focus on heavier implements."
            else:
                drill_sets = 4
                drill_reps = 5
                drill_note = "Maintain tempo with moderate overload implement."
            session_plan.append(
                {
                    "name": "Weighted Ball Throws",
                    "sets": drill_sets,
                    "reps": drill_reps,
                    "implement_weight_pct_of_best": round(drill_load_pct * 100, 1),
                    "estimated_distance_target_m": round(best * (1 - 0.05), 2),
                    "notes": drill_note,
                    "is_strength": False,
                    "loggable": True,
                }
            )

    # Non-strength (cardio/mobility) recommendations – advisory only.
    session_plan.extend(
        [
            {
                "name": "Light Jog or Bike",
                "duration_minutes": 20,
                "intensity": "easy aerobic",
                "is_strength": False,
                "loggable": False,
            },
            {
                "name": "Full-body Mobility Flow",
                "duration_minutes": 15,
                "focus": "hips/shoulders/spine",
                "is_strength": False,
                "loggable": False,
            },
        ]
    )

    return session_plan


def print_workout_routine(routine: Iterable[Mapping[str, Any]]) -> str:
    """
    Format a workout routine for CLI display.

    Returns the string so callers can print or log it, but also prints by default.
    """
    lines: list[str] = []
    for exercise in routine:
        name = exercise.get("name", "Exercise")
        sets = exercise.get("sets")
        reps = exercise.get("reps")
        rep_range = exercise.get("rep_range")
        target_weight = exercise.get("target_weight_kg")
        duration = exercise.get("duration_minutes")
        distance = exercise.get("target_distance_m")
        notes = exercise.get("notes")
        loggable = exercise.get("loggable", True)

        line = f"- {name}"
        if sets:
            line += f" | {sets} sets"
        if reps:
            line += f" × {reps} reps"
        elif rep_range:
            line += f" × {rep_range[0]}-{rep_range[1]} reps"
        if target_weight:
            line += f" @ {target_weight:.1f} kg"
        if duration:
            line += f" | {duration} min"
        if distance:
            line += f" | target {distance:.2f} m"
        if notes:
            line += f" ({notes})"
        if not loggable:
            line += " [recommended]"
        lines.append(line)

    output = "\n".join(lines)
    print(output)
    return output


def routine_to_payload(
    routine: Iterable[Mapping[str, Any]],
    *,
    athlete_profile: Mapping[str, Any] | None = None,
    session_date: Any | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Prepare structured data for future printable/PDF exports.
    """
    if isinstance(session_date, datetime):
        date_iso = session_date.astimezone(timezone.utc).isoformat()
    elif session_date:
        date_iso = str(session_date)
    else:
        date_iso = _now_utc().date().isoformat()
    return {
        "athlete": dict(athlete_profile or {}),
        "session_date": date_iso,
        "metadata": dict(metadata or {}),
        "routine": [dict(item) for item in routine],
    }


def routine_to_json(
    routine: Iterable[Mapping[str, Any]],
    *,
    athlete_profile: Mapping[str, Any] | None = None,
    session_date: Any | None = None,
    metadata: Mapping[str, Any] | None = None,
    indent: int = 2,
) -> str:
    """
    JSON serialisation helper so printable/exportable outputs can be generated later.
    """
    payload = routine_to_payload(
        routine,
        athlete_profile=athlete_profile,
        session_date=session_date,
        metadata=metadata,
    )
    return json.dumps(payload, indent=indent, sort_keys=True)


# ------------------------------ Prediction Logic ---------------------------- #

VALID_METRICS = {"throw_distance", "bench_1rm", "squat_1rm", "session_load"}


def _collapse_same_day(series: list[tuple[datetime, float]], reducer) -> list[tuple[datetime, float]]:
    """Collapse multiple samples recorded on the same day."""
    buckets: dict[datetime.date, list[float]] = {}
    tzinfo = timezone.utc
    for ts, value in series:
        tzinfo = ts.tzinfo or tzinfo
        buckets.setdefault(ts.date(), []).append(float(value))
    collapsed: list[tuple[datetime, float]] = []
    for day, values in buckets.items():
        collapsed.append((datetime.combine(day, datetime.min.time(), tzinfo), reducer(values)))
    collapsed.sort(key=lambda item: item[0])
    return collapsed


def _winsorize_series(series: list[tuple[datetime, float]], z_limit: float = 3.5) -> list[tuple[datetime, float]]:
    """Clamp extreme values using a MAD-based winsorisation to tame outliers."""
    if len(series) < 4:
        return series
    values = np.array([val for _, val in series], dtype=float)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad == 0:
        return series
    scale = 1.4826 * mad  # convert MAD to approximate std
    lower = median - z_limit * scale
    upper = median + z_limit * scale
    clamped = np.clip(values, lower, upper)
    return [(series[idx][0], float(clamped[idx])) for idx in range(len(series))]


def predict_trends(
    athlete_id: int,
    metric: str,
    *,
    days_ahead: int = 14,
) -> dict[str, Any]:
    metric = metric.lower()
    if metric not in VALID_METRICS:
        raise ValueError(f"metric must be one of {sorted(VALID_METRICS)}")

    with open_database(readonly=True) as conn:
        series = _fetch_metric_series(conn, athlete_id, metric)
        if len(series) < 3:
            synthetic = _synthetic_series_from_profile(conn, athlete_id, metric)
            if synthetic:
                series = synthetic
    if series:
        reducer = sum if metric == "session_load" else max
        series = _collapse_same_day(series, reducer)
        series = _winsorize_series(series)

    if len(series) < 3:
        LOGGER.warning("Not enough data to model %s for athlete %s", metric, athlete_id)
        # If we have at least one point, return a flat projection instead of empty.
        if len(series) == 1:
            value = float(series[0][1])
            last_date = series[0][0]
            forecasts = [
                ((last_date + timedelta(days=step)).date().isoformat(), value)
                for step in range(1, days_ahead + 1)
            ]
            return {"forecasts": forecasts, "model": "flat", "confidence": 0.0, "trend": "flat"}
        if len(series) == 2:
            # Simple linear extrapolation for two points
            (t0, v0), (t1, v1) = series
            delta_days = max(1, (t1 - t0).days)
            slope = (float(v1) - float(v0)) / delta_days
            last_date = t1
            forecasts = []
            for step in range(1, days_ahead + 1):
                val = float(v1) + slope * step
                if val < 0:
                    val = 0.0
                forecasts.append(((last_date + timedelta(days=step)).date().isoformat(), val))
            trend = "up" if slope > 0 else "down" if slope < 0 else "flat"
            return {"forecasts": forecasts, "model": "linear_seed", "confidence": None, "trend": trend}
        return {"forecasts": [], "model": "insufficient", "confidence": None, "trend": "flat"}

    series.sort(key=lambda item: item[0])
    timestamps = np.array([(ts - series[0][0]).days for ts, _ in series], dtype=float)
    values = np.array([value for _, value in series], dtype=float)
    value_std = float(np.std(values)) if values.size else 0.0

    def _rmse(actual: np.ndarray, fit: np.ndarray) -> float:
        if actual.size == 0 or fit.size == 0:
            return float("nan")
        return float(np.sqrt(np.mean((actual - fit) ** 2)))

    def _forecast_holt(y: np.ndarray, m: int) -> tuple[np.ndarray, list[tuple[str, float]], float, float]:
        alphas = [0.2, 0.3, 0.4, 0.5, 0.7]
        betas = [0.1, 0.2, 0.3, 0.4]
        best_alpha = 0.3
        best_beta = 0.2
        best_fit: np.ndarray | None = None
        best_rmse = float("inf")
        for a in alphas:
            for b in betas:
                if len(y) == 1:
                    level_val = y[0]
                    trend_val = 0.0
                else:
                    level_val = y[0]
                    trend_val = y[1] - y[0]
                preds = np.zeros_like(y)
                preds[0] = y[0]
                for i in range(1, len(y)):
                    pred = level_val + trend_val
                    preds[i] = pred
                    prev_level = level_val
                    level_val = a * y[i] + (1 - a) * (level_val + trend_val)
                    trend_val = b * (level_val - prev_level) + (1 - b) * trend_val
                err = _rmse(y, preds)
                if err < best_rmse:
                    best_rmse = err
                    best_alpha = a
                    best_beta = b
                    best_fit = preds
        if best_fit is None:
            best_fit = y.copy()
        if len(y) == 1:
            level = y[0]
            trend = 0.0
        else:
            level = y[0]
            trend = y[1] - y[0]
            for i in range(1, len(y)):
                prev_level = level
                level = best_alpha * y[i] + (1 - best_alpha) * (level + trend)
                trend = best_beta * (level - prev_level) + (1 - best_beta) * trend
        fc: list[tuple[str, float]] = []
        last_date = series[-1][0]
        for step in range(1, days_ahead + 1):
            pred = float(level + step * trend)
            if pred < 0:
                pred = 0.0
            fc.append(((last_date + timedelta(days=step)).date().isoformat(), pred))
        return best_fit, fc, best_rmse, trend

    def _forecast_wls_linear(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, list[tuple[str, float]], float, float]:
        n = len(x)
        decay = 0.9
        weights = np.power(decay, (n - 1 - np.arange(n))).astype(float)
        w_sum = np.sum(weights)
        x_bar = float(np.sum(weights * x) / w_sum)
        y_bar = float(np.sum(weights * y) / w_sum)
        denom = float(np.sum(weights * (x - x_bar) ** 2))
        if denom == 0.0:
            slope = 0.0
        else:
            slope = float(np.sum(weights * (x - x_bar) * (y - y_bar)) / denom)
        intercept = y_bar - slope * x_bar
        fitted = intercept + slope * x
        rmse = _rmse(y, fitted)
        last_index = x[-1]
        fc: list[tuple[str, float]] = []
        last_date = series[-1][0]
        for step in range(1, days_ahead + 1):
            future_x = last_index + step
            pred = float(intercept + slope * future_x)
            if pred < 0:
                pred = 0.0
            fc.append(((last_date + timedelta(days=step)).date().isoformat(), pred))
        return fitted, fc, rmse, slope

    def _forecast_poly2(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, list[tuple[str, float]], float, float]:
        coeffs = np.polyfit(x, y, deg=2)
        fitted = np.polyval(coeffs, x)
        rmse = _rmse(y, fitted)
        last_index = x[-1]
        fc: list[tuple[str, float]] = []
        last_date = series[-1][0]
        for step in range(1, days_ahead + 1):
            future_x = last_index + step
            pred = float(np.polyval(coeffs, future_x))
            if pred < 0:
                pred = 0.0
            fc.append(((last_date + timedelta(days=step)).date().isoformat(), pred))
        slope_now = float(2 * coeffs[0] * last_index + coeffs[1])
        return fitted, fc, rmse, slope_now

    candidates: list[tuple[str, np.ndarray, list[tuple[str, float]], float, float]] = []
    holt_fit, holt_fc, holt_rmse, holt_slope = _forecast_holt(values, days_ahead)
    candidates.append(("holt_linear", holt_fit, holt_fc, holt_rmse, holt_slope))
    wls_fit, wls_fc, wls_rmse, wls_slope = _forecast_wls_linear(timestamps, values)
    candidates.append(("linear_wls", wls_fit, wls_fc, wls_rmse, wls_slope))
    if len(series) >= 20:
        poly_fit, poly_fc, poly_rmse, poly_slope = _forecast_poly2(timestamps, values)
        candidates.append(("poly2", poly_fit, poly_fc, poly_rmse, poly_slope))

    best = min(candidates, key=lambda item: (float("inf") if np.isnan(item[3]) else item[3]))
    best_model, best_fitted, forecasts, rmse, slope_est = best

    last_actual = values[-1]
    last_forecast = forecasts[-1][1] if forecasts else last_actual
    small_signal = value_std > 0 and abs(slope_est) < value_std * 0.01
    if slope_est > 0 and not small_signal and last_forecast > last_actual * 1.005:
        trend_direction = "up"
    elif slope_est < 0 and not small_signal and last_forecast < last_actual * 0.995:
        trend_direction = "down"
    else:
        trend_direction = "flat"
        LOGGER.warning("Trend appears flat for %s (athlete %s).", metric, athlete_id)

    return {
        "forecasts": forecasts,
        "model": best_model,
        "confidence": rmse,
        "trend": trend_direction,
    }


def _fetch_metric_series(
    conn: sqlite3.Connection,
    athlete_id: int,
    metric: str,
) -> list[tuple[datetime, float]]:
    if metric == "throw_distance":
        rows = conn.execute(
            """
            SELECT logged_at, distance
            FROM ThrowLogs
            WHERE athlete_id = ? AND distance IS NOT NULL
            ORDER BY logged_at
            """,
            (athlete_id,),
        ).fetchall()
        return _to_time_series(rows)

    if metric in {"bench_1rm", "squat_1rm"}:
        exercise = "bench press" if metric == "bench_1rm" else "back squat"
        rows = conn.execute(
            """
            SELECT logged_at, load_kg, reps
            FROM StrengthLogs
            WHERE athlete_id = ? AND exercise = ? AND load_kg IS NOT NULL AND reps IS NOT NULL
            ORDER BY logged_at
            """,
            (athlete_id, exercise),
        ).fetchall()
        one_rm_rows = []
        for logged_at, load, reps in rows:
            est_one_rm = float(load) * (1 + float(reps) / 30.0)
            one_rm_rows.append((logged_at, est_one_rm))
        return _to_time_series(one_rm_rows)

    if metric == "session_load":
        rows = conn.execute(
            """
            SELECT logged_at, load_kg, reps
            FROM StrengthLogs
            WHERE athlete_id = ? AND load_kg IS NOT NULL AND reps IS NOT NULL
            ORDER BY logged_at
            """,
            (athlete_id,),
        ).fetchall()
        load_per_day: dict[str, float] = {}
        for logged_at, load, reps in rows:
            date_key = str(logged_at).split("T")[0]
            load_per_day[date_key] = load_per_day.get(date_key, 0.0) + float(load) * float(reps)
        aggregate = [(date_key, value) for date_key, value in load_per_day.items()]
        return _to_time_series(aggregate)

    return []


def _synthetic_series_from_profile(
    conn: sqlite3.Connection,
    athlete_id: int,
    metric: str,
) -> list[tuple[datetime, float]]:
    """
    Build a minimal series from stored athlete profile data when logs are absent.
    This keeps forecasts usable after only entering profile benchmarks.
    """
    row = conn.execute(
        "SELECT strength_benchmarks, throw_distances, updated_at, created_at FROM Athletes WHERE id = ?",
        (athlete_id,),
    ).fetchone()
    if not row:
        return []
    strength_json, throws_json, updated_at, created_at = row
    def _coerce_mapping(text: Any) -> dict[str, Any]:
        if not text:
            return {}
        if isinstance(text, dict):
            return text
        if isinstance(text, str):
            # Try strict JSON first
            try:
                return json.loads(text)
            except Exception:
                pass
            # Fallback for loose "bench press:100,back squat:110" style strings
            parts = [chunk.strip() for chunk in text.strip("{}").split(",") if chunk.strip()]
            mapping: dict[str, Any] = {}
            for part in parts:
                if ":" in part:
                    k, v = part.split(":", 1)
                    try:
                        mapping[k.strip().strip('"')] = float(v.strip().strip('"'))
                    except Exception:
                        continue
            return mapping
        return {}

    strength = _coerce_mapping(strength_json)
    throws = _coerce_mapping(throws_json)

    base_date = None
    for ts in (updated_at, created_at):
        try:
            base_date = datetime.fromisoformat(str(ts))
            break
        except Exception:
            continue
    base_date = base_date or _now_utc()

    def _spread(values: list[float]) -> list[tuple[datetime, float]]:
        return [
            (base_date - timedelta(days=2 * idx), float(value))
            for idx, value in enumerate(values)
            if value is not None
        ]

    if metric == "throw_distance" and throws:
        primary = max(
            throws.items(),
            key=lambda item: len(item[1]) if isinstance(item[1], list) else 0,
        )[0]
        values_raw = throws.get(primary) or []
        numeric: list[float] = []
        for v in values_raw:
            try:
                numeric.append(float(v))
            except Exception:
                continue
        if numeric:
            return _spread(numeric[:5])

    if metric == "bench_1rm":
        bench = strength.get("bench press")
        if bench is not None:
            return _spread([float(bench)] * 3)

    if metric == "squat_1rm":
        squat = strength.get("back squat")
        if squat is not None:
            return _spread([float(squat)] * 3)

    if metric == "session_load":
        primary = None
        if throws:
            primary = max(
                throws.items(),
                key=lambda item: len(item[1]) if isinstance(item[1], list) else 0,
            )[0]
        distances = throws.get(primary) if primary else None
        numeric: list[float] = []
        if isinstance(distances, list):
            for v in distances:
                try:
                    numeric.append(float(v))
                except Exception:
                    continue
        if numeric:
            loads = [max(10.0, d * 1.5) for d in numeric[:3]]
            return _spread(loads)

    return []

def _fetch_athlete_profile(conn: sqlite3.Connection, athlete_id: int) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT id, name, height_cm, weight_kg, bmi, strength_benchmarks, throw_distances, notes
        FROM Athletes
        WHERE id = ?
        """,
        (athlete_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"Athlete {athlete_id} not found.")
    keys = (
        "id",
        "name",
        "height_cm",
        "weight_kg",
        "bmi",
        "strength_benchmarks",
        "throw_distances",
        "notes",
    )
    return {key: value for key, value in zip(keys, row)}


def get_daily_routine(athlete_id: int) -> list[dict[str, Any]]:
    """Convenience helper for CLI to fetch profile and build the routine."""
    with open_database(readonly=True) as conn:
        profile = _fetch_athlete_profile(conn, athlete_id)
    return generate_workout(profile, session_date=_now_utc())


def record_workout_results(
    routine_results: Iterable[Mapping[str, Any]],
    *,
    default_date: Any | None = None,
) -> list[int]:
    """
    Persist completed strength movements into StrengthLogs.

    Each routine result item should include:
        - athlete_id (int)
        - name (exercise name)
        - actual_weight_kg (float)
        - actual_reps (int)
        - date (optional override)
        - notes/RPE (optional)
    Non-strength entries (loggable=False) are ignored.
    """
    inserted_ids: list[int] = []
    last_athlete: int | None = None
    for entry in routine_results:
        if not entry.get("loggable", True):
            continue
        athlete_id = entry.get("athlete_id")
        if not athlete_id:
            continue
        exercise = entry.get("name") or entry.get("exercise")
        weight = entry.get("actual_weight_kg") or entry.get("target_weight_kg")
        reps = entry.get("actual_reps") or entry.get("reps")
        if exercise is None or weight is None or reps is None:
            continue
        logged_date = entry.get("date") or default_date or _now_utc().date().isoformat()
        notes = entry.get("notes") or entry.get("rpe")
        athlete_int = int(athlete_id)
        log_id = log_strength_workout(
            athlete_int,
            logged_date,
            exercise,
            float(weight),
            int(reps),
            notes=str(notes) if notes is not None else None,
        )
        inserted_ids.append(log_id)
        last_athlete = athlete_int
    if inserted_ids and last_athlete is not None:
        _refresh_profile_and_trends(last_athlete)
    return inserted_ids


def _refresh_profile_and_trends(athlete_id: int) -> None:
    """Recompute athlete aggregates and trend snapshots after new data."""
    with open_database() as conn:
        benchmark_rows = conn.execute(
            """
            SELECT exercise, MAX(load_kg)
            FROM StrengthLogs
            WHERE athlete_id = ? AND load_kg IS NOT NULL
            GROUP BY exercise
            """,
            (athlete_id,),
        ).fetchall()
        benchmarks = {
            exercise: float(max_load)
            for exercise, max_load in benchmark_rows
            if exercise and max_load is not None
        }

        throw_rows = conn.execute(
            """
            SELECT event, distance
            FROM ThrowLogs
            WHERE athlete_id = ? AND distance IS NOT NULL
            ORDER BY logged_at DESC
            """,
            (athlete_id,),
        ).fetchall()
        throws: dict[str, list[float]] = defaultdict(list)
        for event, distance in throw_rows:
            if event and distance is not None:
                if len(throws[event]) < 5:
                    throws[event].append(float(distance))

        updates: list[str] = []
        params: list[Any] = []
        if benchmarks:
            updates.append("strength_benchmarks = ?")
            params.append(json.dumps(benchmarks, sort_keys=True))
        if throws:
            updates.append("throw_distances = ?")
            params.append(json.dumps(throws, sort_keys=True))
        if updates:
            updates.append("updated_at = CURRENT_TIMESTAMP")
            sql = f"UPDATE Athletes SET {', '.join(updates)} WHERE id = ?"
            params.append(athlete_id)
            conn.execute(sql, params)

        trend_snapshot = analyze_trends(athlete_id)
        if trend_snapshot:
            conn.execute(
                """
                INSERT INTO AthleteTrends (athlete_id, snapshot)
                VALUES (?, ?)
                ON CONFLICT(athlete_id) DO UPDATE
                SET snapshot = excluded.snapshot,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (athlete_id, json.dumps(trend_snapshot, sort_keys=True)),
            )
        conn.commit()


def _migrate_sessions(sessions: list[Any]) -> Tuple[list[Any], bool]:
    """Upgrade legacy session records in-place when the schema evolves."""
    upgraded: list[Any] = []
    changed = False
    for record in sessions:
        if isinstance(record, dict):
            migrated, mutated = _migrate_record(record)
            upgraded.append(migrated)
            changed = changed or mutated
        else:
            upgraded.append(record)
    return upgraded, changed


def _migrate_record(record: dict[str, Any]) -> Tuple[dict[str, Any], bool]:
    mutated = False
    upgraded = dict(record)

    raw_event = upgraded.get("event")
    if isinstance(raw_event, str) and raw_event.strip():
        event = raw_event.strip().lower()
    else:
        event = DEFAULT_EVENT
    if upgraded.get("event") != event:
        upgraded["event"] = event
        mutated = True

    duration_raw = upgraded.get("duration_minutes", 0.0)
    try:
        duration_value = validate_duration(duration_raw, field="duration_minutes")
    except ValidationError:
        duration_value = 0.0
    if duration_value != duration_raw:
        upgraded["duration_minutes"] = duration_value
        mutated = True

    computed_load = calculate_session_load(upgraded.get("rpe"), float(duration_value))
    existing_load = upgraded.get("load")
    if not isinstance(existing_load, (int, float)) or not math.isclose(
        float(existing_load),
        computed_load,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        upgraded["load"] = computed_load
        mutated = True

    schema_raw = upgraded.get("schema_version")
    if isinstance(schema_raw, int) and schema_raw > CURRENT_SCHEMA_VERSION:
        schema_value = schema_raw
    else:
        schema_value = CURRENT_SCHEMA_VERSION
    if schema_raw != schema_value:
        upgraded["schema_version"] = schema_value
        mutated = True

    for key in (
        "video_id",
        "biomechanics_analysis_id",
        "biomechanics_status",
        "biomechanics_timestamp",
        "biomechanics_result_path",
    ):
        if key not in upgraded:
            upgraded[key] = None
            mutated = True

    return upgraded, mutated
