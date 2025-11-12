from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import pandas as pd

from .constants import DEFAULT_ATHLETE_PLACEHOLDER
from .config import get_config
from .models import (
    CURRENT_SCHEMA_VERSION,
    DEFAULT_EVENT,
    Session,
    ValidationError,
    calculate_session_load,
    coerce_number,
    parse_iso_date,
    parse_tags,
    parse_throws,
    validate_distance,
    validate_duration,
)


def _thresholds() -> tuple[float, float, float]:
    cfg = get_config().acwr_thresholds
    return cfg.sweet_min, cfg.sweet_max, cfg.high


def sessions_to_dataframe(sessions: Sequence[Mapping[str, object] | Session]) -> pd.DataFrame:
    """Normalise raw sessions into a pandas DataFrame."""
    records: list[dict[str, object]] = []
    for item in sessions:
        if isinstance(item, Session):
            payload = item.to_dict()
            payload.setdefault("duration_minutes", item.duration_minutes)
        elif isinstance(item, Mapping):
            payload = dict(item)
        else:
            raise TypeError(f"Unsupported session type: {type(item)!r}")

        session_date = parse_iso_date(payload.get("date"), field="date")
        event = _normalise_event(payload.get("event"))
        throws = _normalise_throws(payload.get("throws", []))
        best = validate_distance(payload.get("best"), field="best")
        rpe = payload.get("rpe")
        duration = validate_duration(payload.get("duration_minutes", 0.0), field="duration_minutes")
        load = _calculate_load(rpe, duration)
        athlete = str(payload.get("athlete") or "").strip() or DEFAULT_ATHLETE_PLACEHOLDER
        team_raw = payload.get("team")
        team = str(team_raw).strip() if isinstance(team_raw, str) else team_raw
        team_value = team if team else None
        implement_weight = _optional_float(payload.get("implement_weight_kg"))
        technique = _optional_str(payload.get("technique"))
        fouls = _optional_int(payload.get("fouls"))
        schema_raw = payload.get("schema_version")
        try:
            schema_version = int(schema_raw)
        except (TypeError, ValueError):
            schema_version = CURRENT_SCHEMA_VERSION

        records.append(
            {
                "date": pd.to_datetime(session_date),
                "best": best,
                "throws": throws,
                "throws_count": len(throws),
                "rpe": float(rpe) if rpe is not None else pd.NA,
                "duration_minutes": duration,
                "load": load,
                "notes": payload.get("notes") or "",
                "tags": parse_tags(payload.get("tags", []), field="tags"),
                "athlete": athlete,
                "team": team_value,
                "event": event,
                "implement_weight_kg": implement_weight,
                "technique": technique,
                "fouls": fouls,
                "schema_version": schema_version,
            }
        )

    df = pd.DataFrame(records)
    sort_keys = [key for key in ("athlete", "event", "date") if key in df.columns]
    df.sort_values(sort_keys, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def compute_daily_metrics(
    df_sessions: pd.DataFrame,
    group_keys: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Aggregate session dataframe into daily workload metrics."""
    group_keys = [key for key in (group_keys or []) if key in df_sessions.columns]
    base_columns = group_keys + [
        "date",
        "load",
        "session_count",
        "acute_load",
        "chronic_load",
        "acwr_rolling",
        "acute_ewma",
        "chronic_ewma",
        "acwr_ewma",
        "risk_flag",
    ]
    if df_sessions.empty:
        return pd.DataFrame(columns=base_columns)

    if not group_keys:
        return _compute_daily_metrics_single(df_sessions)

    rows: list[dict[str, object]] = []
    grouped = df_sessions.groupby(group_keys, dropna=False)
    for key, group in grouped:
        daily = _compute_daily_metrics_single(group)
        key_tuple = _ensure_tuple(key)
        for idx, column in enumerate(group_keys):
            daily[column] = key_tuple[idx]
        rows.extend(daily.to_dict("records"))

    if not rows:
        return pd.DataFrame(columns=base_columns)

    combined = pd.DataFrame(rows)
    ordered = group_keys + [col for col in combined.columns if col not in group_keys]
    return combined[ordered]


def _compute_daily_metrics_single(df_sessions: pd.DataFrame) -> pd.DataFrame:
    daily = (
        df_sessions.groupby("date", as_index=False)
        .agg(
            load=("load", "sum"),
            session_count=("date", "count"),
        )
        .set_index("date")
    )

    daily["acute_load"] = daily["load"].rolling(window=7, min_periods=1).sum()
    daily["chronic_load"] = daily["load"].rolling(window=28, min_periods=1).mean()
    daily["acwr_rolling"] = _safe_divide(daily["acute_load"], daily["chronic_load"])

    daily["acute_ewma"] = daily["load"].ewm(span=7, adjust=False).mean()
    daily["chronic_ewma"] = daily["load"].ewm(span=28, adjust=False).mean()
    daily["acwr_ewma"] = _safe_divide(daily["acute_ewma"], daily["chronic_ewma"])

    daily["risk_flag"] = daily.apply(
        lambda row: _risk_category(row["acwr_rolling"], row["acwr_ewma"]),
        axis=1,
    )
    return daily.reset_index()


def compute_weekly_summary(
    df_sessions: pd.DataFrame,
    daily_metrics: pd.DataFrame,
    group_keys: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Generate weekly aggregates incorporating ACWR metrics."""
    group_keys = [key for key in (group_keys or []) if key in df_sessions.columns]
    base_columns = group_keys + [
        "label",
        "start_date",
        "end_date",
        "sessions",
        "best_throw",
        "avg_rpe",
        "total_load",
        "throw_volume",
        "acwr_rolling",
        "acwr_ewma",
        "risk_flag",
    ]
    if df_sessions.empty:
        return pd.DataFrame(columns=base_columns)

    iso = df_sessions["date"].dt.isocalendar()
    df_sessions = df_sessions.assign(
        iso_year=iso.year,
        iso_week=iso.week,
        label=iso.year.astype(str) + "-W" + iso.week.map(lambda value: f"{int(value):02d}"),
    )

    group_cols = group_keys + ["iso_year", "iso_week", "label"]
    weekly = (
        df_sessions.groupby(group_cols)
        .agg(
            start_date=("date", "min"),
            end_date=("date", "max"),
            sessions=("date", "count"),
            best_throw=("best", "max"),
            avg_rpe=("rpe", "mean"),
            total_load=("load", "sum"),
            throw_volume=("throws_count", "sum"),
            team=("team", lambda values: _most_common(values)),
        )
        .reset_index()
        .drop(columns=["iso_year", "iso_week"])
    )

    if not daily_metrics.empty:
        daily_iso = daily_metrics.copy()
        iso_daily = pd.to_datetime(daily_iso["date"]).dt.isocalendar()
        daily_iso["label"] = iso_daily.year.astype(str) + "-W" + iso_daily.week.map(lambda value: f"{int(value):02d}")
        merge_keys = group_keys + ["label"]
        acwr_by_week = (
            daily_iso.groupby(merge_keys)
            .agg(
                acwr_rolling=("acwr_rolling", "last"),
                acwr_ewma=("acwr_ewma", "last"),
                risk_flag=("risk_flag", lambda flags: _aggregate_weekly_risk(list(flags))),
            )
            .reset_index()
        )
        weekly = weekly.merge(acwr_by_week, on=merge_keys, how="left")
    else:
        weekly["acwr_rolling"] = pd.NA
        weekly["acwr_ewma"] = pd.NA
        weekly["risk_flag"] = ""

    for column in ("avg_rpe", "total_load", "acwr_rolling", "acwr_ewma"):
        if column in weekly.columns:
            weekly[column] = pd.to_numeric(weekly[column], errors="coerce")
    weekly["avg_rpe"] = weekly["avg_rpe"].round(2)
    weekly["total_load"] = weekly["total_load"].round(1)
    weekly["acwr_rolling"] = weekly["acwr_rolling"].round(2)
    weekly["acwr_ewma"] = weekly["acwr_ewma"].round(2)
    ordered = group_keys + [col for col in weekly.columns if col not in group_keys]
    return weekly[ordered]


def _normalise_throws(throws: Iterable[object]) -> list[float]:
    if isinstance(throws, str):
        values = parse_throws(throws, field="throws")
    else:
        values = list(throws)
    return [validate_distance(value, field="throw") for value in values]


def _normalise_event(value: object | None) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip().lower()
    return DEFAULT_EVENT


def _optional_float(value: object | None) -> float | None:
    if value is None:
        return None
    try:
        return float(coerce_number(value, field="value", allow_empty=True))
    except ValidationError:
        return None


def _optional_int(value: object | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        coerced = coerce_number(value, field="value", allow_float=False)
    except ValidationError:
        return None
    return int(coerced)


def _optional_str(value: object | None) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return None


def _ensure_tuple(key: object) -> tuple:
    if isinstance(key, tuple):
        return key
    return (key,)


def _calculate_load(rpe: object | None, duration: float) -> float:
    if rpe is None or pd.isna(rpe):
        return 0.0
    return calculate_session_load(rpe, float(duration))


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    ratio = numerator / denominator.replace({0: pd.NA})
    return ratio.replace([pd.NA, pd.NaT], pd.NA)


def _risk_category(acwr_roll: float, acwr_ewma: float) -> str:
    lower, sweet_max, high = _thresholds()
    values = [value for value in (acwr_roll, acwr_ewma) if pd.notna(value)]
    if not values:
        return ""
    if any(value > high or value < lower for value in values):
        return "HIGH"
    if any(value > sweet_max for value in values):
        return "MODERATE"
    return "LOW"


def _aggregate_weekly_risk(flags: list[str]) -> str:
    flags = [flag for flag in flags if flag]
    if not flags:
        return ""
    if "HIGH" in flags:
        return "HIGH"
    if "MODERATE" in flags:
        return "MODERATE"
    return "LOW"


def _most_common(values: Iterable[object]) -> object | None:
    counts: dict[object, int] = {}
    for value in values:
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        if isinstance(value, str):
            key = value.strip()
        elif pd.isna(value):
            continue
        else:
            key = value
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.get)
