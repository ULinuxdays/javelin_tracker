from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .constants import DEFAULT_ATHLETE_PLACEHOLDER
from .metrics import compute_daily_metrics, compute_weekly_summary, sessions_to_dataframe
from .models import (
    Session,
    ValidationError,
    clamp_rpe,
    parse_iso_date,
    parse_tags,
    parse_throws,
    validate_distance,
    validate_duration,
)

def _normalise_identity(value: str | None, *, field: str, default: str | None = None) -> str:
    text = (value or "").strip()
    if text:
        return text
    if default:
        return default
    raise ValidationError(f"{field} is required.")


@dataclass(frozen=True)
class LogResult:
    """Structured outcome of parsing log arguments."""

    session: Session
    throws: list[float]

    @property
    def confirmation(self) -> str:
        return (
            f"[{self.session.athlete}] "
            f"Logged {self.session.date.isoformat()} best {self.session.best:.1f} m."
        )

    @property
    def verbose_tokens(self) -> list[str]:
        tokens: list[str] = []
        tokens.append(f"athlete={self.session.athlete}")
        if self.session.team:
            tokens.append(f"team={self.session.team}")
        if self.throws:
            tokens.append("throws=" + ", ".join(f"{value:.1f}" for value in self.throws))
        if self.session.tags:
            tokens.append("tags=" + ", ".join(self.session.tags))
        if self.session.rpe is not None:
            tokens.append(f"rpe={self.session.rpe}")
        if self.session.duration_minutes:
            load = self.session_load
            tokens.append(f"duration={self.session.duration_minutes:.1f} min")
            if load is not None:
                tokens.append(f"load={load:.1f} AU")
        if self.session.notes:
            tokens.append("notes recorded")
        return tokens

    @property
    def session_load(self) -> float | None:
        if self.session.rpe is None or self.session.duration_minutes <= 0:
            return None
        return float(self.session.rpe) * self.session.duration_minutes


@dataclass(frozen=True)
class SummaryRow:
    marker: str
    athlete: str
    team: str | None
    label: str
    sessions: int
    best_throw: float | None
    avg_rpe: float | None
    total_load: float
    throw_volume: int
    acwr_rolling: float | None
    acwr_ewma: float | None
    risk_flag: str


@dataclass(frozen=True)
class SummaryReport:
    rows: list[SummaryRow]
    total_sessions: int
    total_throws: int
    total_load: float
    average_rpe: float | None
    date_range: tuple[date, date] | None
    personal_best: tuple[date, float] | None
    personal_bests: dict[str, tuple[date, float]]
    pb_groups: set[str]
    high_risk_dates: dict[str, list[date]]
    filters: dict[str, Any]


def build_session_from_inputs(
    *,
    date_text: str | None,
    best: float | None,
    throws: str | None,
    rpe: int | None,
    notes: str | None,
    tags: str | None,
    duration_minutes: float | None,
    athlete: str | None,
    team: str | None,
) -> LogResult:
    """Convert CLI inputs into a validated session payload."""
    session_date = parse_iso_date(date_text, field="date") if date_text else date.today()
    throws_tokens = parse_throws(throws, field="throws")
    throws_distances = _normalise_throw_distances(throws_tokens)

    best_distance = (
        validate_distance(best, field="best")
        if best is not None
        else _resolve_best_from_throws(throws_distances)
    )

    rpe_value = clamp_rpe(rpe, field="rpe") if rpe is not None else None
    tag_list = parse_tags(tags, field="tags")
    note_value = notes.strip() if notes else None
    duration_value = (
        validate_duration(duration_minutes, field="duration_minutes")
        if duration_minutes is not None
        else 0.0
    )
    athlete_name = _normalise_identity(athlete, field="athlete", default=DEFAULT_ATHLETE_PLACEHOLDER)
    team_name = team.strip() if team and team.strip() else None

    session = Session(
        date=session_date,
        best=best_distance,
        athlete=athlete_name,
        team=team_name,
        throws=throws_distances,
        rpe=rpe_value,
        duration_minutes=duration_value,
        notes=note_value,
        tags=tag_list,
    )
    return LogResult(session=session, throws=throws_distances)


def build_summary_report(
    sessions: Sequence[Mapping[str, Any]],
    *,
    group: str,
    filters: dict[str, Any] | None = None,
) -> SummaryReport:
    """Produce aggregated statistics for summary displays."""
    normalised_group = group.lower()
    if normalised_group not in {"week", "month"}:
        raise ValueError("Grouping must be 'week' or 'month'.")

    df_sessions = sessions_to_dataframe(sessions)
    group_keys = ["athlete"]
    daily = compute_daily_metrics(df_sessions, group_keys=group_keys)
    weekly = compute_weekly_summary(df_sessions, daily, group_keys=group_keys)
    summary_df = (
        weekly
        if normalised_group == "week"
        else _convert_weekly_to_monthly(weekly, group_keys=["athlete", "team"])
    )

    personal_bests = _personal_best_from_df(df_sessions)
    pb_groups = _pb_groups_from_summary(summary_df, personal_bests)
    overall_pb = _overall_best_from_map(personal_bests)

    rows = [
        _summary_row_from_record(record, pb_groups) for record in summary_df.to_dict("records")
    ]

    total_sessions = int(df_sessions.shape[0])
    total_throws = int(df_sessions["throws_count"].sum()) if not df_sessions.empty else 0
    total_load = float(df_sessions["load"].sum()) if not df_sessions.empty else 0.0
    avg_rpe = (
        float(df_sessions["rpe"].dropna().mean())
        if not df_sessions.empty and df_sessions["rpe"].dropna().size
        else None
    )

    date_range: tuple[date, date] | None = None
    if not df_sessions.empty:
        start = df_sessions["date"].min().date()
        end = df_sessions["date"].max().date()
        date_range = (start, end)

    high_risk_dates: dict[str, list[date]] = defaultdict(list)
    if not daily.empty and "risk_flag" in daily:
        for _, record in daily.iterrows():
            if record.get("risk_flag") == "HIGH":
                athlete = str(record.get("athlete") or DEFAULT_ATHLETE_PLACEHOLDER)
                high_risk_dates[athlete].append(pd.Timestamp(record["date"]).date())
        for athlete in list(high_risk_dates.keys()):
            unique_dates = sorted(set(high_risk_dates[athlete]))
            high_risk_dates[athlete] = unique_dates

    return SummaryReport(
        rows=rows,
        total_sessions=total_sessions,
        total_throws=total_throws,
        total_load=total_load,
        average_rpe=avg_rpe,
        date_range=date_range,
        personal_best=overall_pb,
        personal_bests=personal_bests,
        pb_groups=pb_groups,
        high_risk_dates=high_risk_dates,
        filters=filters or {},
    )


def render_summary_table(report: SummaryReport) -> str:
    """Render a fixed-width table for summary rows."""
    headers = (
        "marker",
        "athlete",
        "team",
        "group",
        "sessions",
        "best",
        "avg_rpe",
        "load",
        "throws",
        "acwr_roll",
        "acwr_ewma",
        "risk",
    )
    rows = [
        {
            "marker": row.marker,
            "athlete": row.athlete,
            "team": row.team or "",
            "group": row.label,
            "sessions": str(row.sessions),
            "best": f"{row.best_throw:.1f}" if row.best_throw is not None else "n/a",
            "avg_rpe": f"{row.avg_rpe:.1f}" if row.avg_rpe is not None else "n/a",
            "load": f"{row.total_load:.1f}",
            "throws": str(row.throw_volume),
            "acwr_roll": f"{row.acwr_rolling:.2f}" if row.acwr_rolling is not None else "n/a",
            "acwr_ewma": f"{row.acwr_ewma:.2f}" if row.acwr_ewma is not None else "n/a",
            "risk": row.risk_flag or "",
        }
        for row in report.rows
    ]
    widths = {key: len(key) for key in headers}
    for row in rows:
        for key in headers:
            widths[key] = max(widths[key], len(row[key]))

    def _format_line(values: Mapping[str, str]) -> str:
        return "  ".join(values[key].rjust(widths[key]) for key in headers)

    header_line = "  ".join(key.upper().rjust(widths[key]) for key in headers)
    body = "\n".join(_format_line(row) for row in rows)
    return "\n".join(filter(None, [header_line, body]))


def generate_plots(
    sessions: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path,
) -> list[Path]:
    """Create distance line plot and weekly volume bar chart."""

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised via CLI
        raise RuntimeError("matplotlib is required to generate plots.") from exc

    df_sessions = sessions_to_dataframe(sessions)
    if df_sessions.empty:
        raise ValueError("No sessions available to plot.")

    daily = compute_daily_metrics(df_sessions)
    weekly = compute_weekly_summary(df_sessions, daily)

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    distance_path = _create_distance_plot(df_sessions, output_dir, timestamp, plt)
    volume_path = _create_volume_plot(weekly, output_dir, timestamp, plt)
    return [distance_path, volume_path]


def build_export_dataframe(
    sessions: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Prepare a DataFrame ready for CSV/Parquet export."""
    df_sessions = sessions_to_dataframe(sessions)
    if df_sessions.empty:
        return df_sessions

    export_df = df_sessions.copy()
    export_df["avg_throw"] = export_df["throws"].apply(
        lambda values: round(sum(values) / len(values), 2) if values else pd.NA
    )
    export_df["date"] = export_df["date"].dt.date
    export_df["throws"] = export_df["throws"].apply(lambda values: ", ".join(f"{value:.1f}" for value in values))
    export_df["tags"] = export_df["tags"].apply(lambda tags: ", ".join(tags))
    export_df["load"] = export_df["load"].round(1)
    export_df.rename(
        columns={
            "best": "best_throw",
        },
        inplace=True,
    )
    return export_df[
        [
            "athlete",
            "team",
            "date",
            "best_throw",
            "throws",
            "throws_count",
            "rpe",
            "duration_minutes",
            "load",
            "avg_throw",
            "tags",
            "notes",
        ]
    ]


def load_seed_payload(source: Path) -> list[dict[str, Any]]:
    """Load seed sessions from disk."""
    if not source.exists():
        raise FileNotFoundError(f"Seed file not found: {source}")

    raw_text = source.read_text(encoding="utf-8")
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError("Seed file is not valid JSON.") from exc

    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ValueError("Seed data must be a JSON list of session objects.")
    return payload


def _create_distance_plot(
    df_sessions: pd.DataFrame,
    output_dir: Path,
    timestamp: str,
    plt: Any,
) -> Path:
    ordered = df_sessions.sort_values("date")
    dates = ordered["date"].dt.date
    bests = ordered["best"]

    line_path = output_dir / f"best_distance_{timestamp}.png"
    fig, ax = plt.subplots()
    ax.plot(dates, bests, marker="o", linewidth=2)
    ax.set_title("Best Distance Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Best Distance (m)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(line_path, dpi=150)
    plt.close(fig)
    return line_path


def _create_volume_plot(
    weekly: pd.DataFrame,
    output_dir: Path,
    timestamp: str,
    plt: Any,
) -> Path:
    if weekly.empty:
        labels: list[str] = []
        volumes: list[int] = []
    else:
        labels = weekly["label"].tolist()
        volumes = weekly["throw_volume"].astype(int).tolist()

    bar_path = output_dir / f"weekly_volume_{timestamp}.png"
    fig, ax = plt.subplots()
    ax.bar(labels, volumes, color="#4C72B0")
    ax.set_title("Weekly Throw Volume")
    ax.set_xlabel("Week")
    ax.set_ylabel("Throws Recorded")
    if labels:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(bar_path, dpi=150)
    plt.close(fig)
    return bar_path


def _normalise_throw_distances(tokens: Sequence[Any]) -> list[float]:
    if isinstance(tokens, str):
        values = parse_throws(tokens, field="throws")
    else:
        values = list(tokens)
    distances: list[float] = []
    for index, token in enumerate(values, start=1):
        distances.append(validate_distance(token, field=f"throws[{index}]"))
    return distances


def _resolve_best_from_throws(throws: Sequence[float]) -> float:
    if not throws:
        raise ValidationError("Provide --best or at least one value via --throws.")
    return max(throws)


def _summary_row_from_record(record: Mapping[str, Any], pb_groups: set[str]) -> SummaryRow:
    label = _compose_label(record)
    primary_label = str(
        record.get("label") or record.get("week_label") or record.get("month_label") or label
    )
    best_throw = _as_optional_float(record.get("best_throw"))
    avg_rpe = _as_optional_float(record.get("avg_rpe"))
    total_load = float(record.get("total_load") or 0.0)
    throw_volume = int(record.get("throw_volume") or 0)
    acwr_rolling = _as_optional_float(record.get("acwr_rolling"))
    acwr_ewma = _as_optional_float(record.get("acwr_ewma"))
    risk_flag = str(record.get("risk_flag") or "").upper()
    athlete = str(record.get("athlete") or DEFAULT_ATHLETE_PLACEHOLDER)
    team = record.get("team") or None

    marker_parts: list[str] = []
    pb_key = f"{athlete}|{primary_label}"
    if pb_key in pb_groups or f"{athlete}|{label}" in pb_groups:
        marker_parts.append("★ PB")
    if risk_flag == "HIGH":
        marker_parts.append("⚠ High Risk")
    marker = " ".join(marker_parts)

    return SummaryRow(
        marker=marker,
        athlete=athlete,
        team=team,
        label=label,
        sessions=int(record.get("sessions") or 0),
        best_throw=best_throw,
        avg_rpe=avg_rpe,
        total_load=round(total_load, 1),
        throw_volume=throw_volume,
        acwr_rolling=acwr_rolling,
        acwr_ewma=acwr_ewma,
        risk_flag=risk_flag,
    )


def _compose_label(record: Mapping[str, Any]) -> str:
    start = (
        pd.to_datetime(record.get("start_date")).date()
        if record.get("start_date") is not None
        else None
    )
    end = (
        pd.to_datetime(record.get("end_date")).date()
        if record.get("end_date") is not None
        else None
    )
    primary = record.get("label") or record.get("week_label") or record.get("month_label")
    if start and end:
        if isinstance(primary, str) and primary:
            label_text = primary
        else:
            label_text = start.strftime("%Y-%m-%d")
        return f"{label_text} ({start:%b %d}–{end:%b %d})"
    return str(primary or "n/a")


def _as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, int)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _convert_weekly_to_monthly(weekly: pd.DataFrame, group_keys: Sequence[str]) -> pd.DataFrame:
    if weekly.empty:
        return weekly.assign(label=[], month_label=[])

    monthly = weekly.copy()
    monthly["month"] = pd.to_datetime(monthly["start_date"]).dt.to_period("M")

    aggregated_rows = []
    grouping = list(group_keys) + ["month"]
    for _, group in monthly.groupby(grouping):
        month = group["month"].iloc[0]
        sessions = int(group["sessions"].sum())
        weights = group["sessions"]
        avg_rpe = None
        valid_rpe = group["avg_rpe"].notna() & weights.gt(0)
        if valid_rpe.any():
            avg_rpe = float(
                (group.loc[valid_rpe, "avg_rpe"] * weights[valid_rpe]).sum()
                / weights[valid_rpe].sum()
            )

        row = {
            "label": month.strftime("%Y-%m"),
            "month_label": month.strftime("%Y-%m"),
            "start_date": group["start_date"].min(),
            "end_date": group["end_date"].max(),
            "sessions": sessions,
            "best_throw": group["best_throw"].max(),
            "avg_rpe": avg_rpe,
            "total_load": group["total_load"].sum(),
            "throw_volume": int(group["throw_volume"].sum()),
            "acwr_rolling": group["acwr_rolling"].dropna().iloc[-1]
            if group["acwr_rolling"].dropna().any()
            else None,
            "acwr_ewma": group["acwr_ewma"].dropna().iloc[-1]
            if group["acwr_ewma"].dropna().any()
            else None,
            "risk_flag": _aggregate_risk(group["risk_flag"].tolist()),
        }
        for key in group_keys:
            row[key] = group[key].iloc[0]
        row["team"] = group["team"].iloc[0] if "team" in group.columns else None
        aggregated_rows.append(row)

    return pd.DataFrame(aggregated_rows)


def _personal_best_from_df(df_sessions: pd.DataFrame) -> dict[str, tuple[date, float]]:
    if df_sessions.empty or "athlete" not in df_sessions.columns:
        return {}
    bests: dict[str, tuple[date, float]] = {}
    for athlete, group in df_sessions.groupby("athlete"):
        idx = group["best"].idxmax()
        if pd.isna(idx):
            continue
        row = group.loc[idx]
        bests[str(athlete)] = (row["date"].date(), float(row["best"]))
    return bests


def _pb_groups_from_summary(
    summary_df: pd.DataFrame,
    pb_map: dict[str, tuple[date, float]],
) -> set[str]:
    labels: set[str] = set()
    if summary_df.empty or not pb_map:
        return labels

    for athlete, (pb_date, _) in pb_map.items():
        for _, record in summary_df.iterrows():
            start = (
                pd.to_datetime(record.get("start_date")).date()
                if record.get("start_date") is not None
                else None
            )
            end = (
                pd.to_datetime(record.get("end_date")).date()
                if record.get("end_date") is not None
                else None
            )
            if start and end and start <= pb_date <= end:
                label = record.get("label") or record.get("week_label") or record.get("month_label")
                labels.add(f"{athlete}|{label}")
                break
    return labels


def _aggregate_risk(flags: list[Any]) -> str:
    normalised = [str(flag).upper() for flag in flags if str(flag).strip()]
    if not normalised:
        return ""
    if "HIGH" in normalised:
        return "HIGH"
    if "MODERATE" in normalised:
        return "MODERATE"
    return "LOW"


def _overall_best_from_map(pb_map: dict[str, tuple[date, float]]) -> tuple[date, float] | None:
    if not pb_map:
        return None
    athlete, (pb_date, pb_value) = max(pb_map.items(), key=lambda item: item[1][1])
    return (pb_date, pb_value)
