from __future__ import annotations

import tempfile
import math
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .metrics import compute_daily_metrics, compute_weekly_summary, sessions_to_dataframe
from .models import Session


@dataclass(frozen=True)
class WeeklyReportStats:
    athlete: str
    team: str | None
    brand_line: str | None
    week_label: str
    week_start: date
    week_end: date
    session_count: int
    best_throw: float
    total_load: float
    average_rpe: float | None
    throw_volume: int
    acwr_rolling: float | None
    acwr_ewma: float | None
    risk_flag: str
    high_risk_dates: list[date]
    risk_rows: list[dict[str, Any]]


def generate_weekly_report(
    sessions: Sequence[Mapping[str, Any] | Session],
    *,
    athlete: str,
    week_ending: date | None = None,
    output_dir: Path = Path("reports"),
    team_name: str | None = None,
    school_name: str | None = None,
) -> Path:
    """
    Build a weekly PDF report include workload metrics, plots, and risk highlights.
    """

    df_sessions = sessions_to_dataframe(sessions)
    if df_sessions.empty:
        raise ValueError("No sessions available to summarise.")

    athlete_key = athlete.strip().lower()
    df_sessions = df_sessions[df_sessions["athlete"].str.lower() == athlete_key]
    if df_sessions.empty:
        raise ValueError(f"No sessions found for athlete '{athlete}'.")

    if team_name:
        team_scope = team_name
    else:
        non_empty_teams = df_sessions["team"].dropna()
        team_scope = non_empty_teams.iloc[0] if not non_empty_teams.empty else None

    week_start, week_end = _resolve_week_range(week_ending)
    df_week = _slice_week(df_sessions, week_start, week_end)
    if df_week.empty:
        raise ValueError("No sessions recorded during the requested week.")

    daily_metrics = compute_daily_metrics(df_sessions, group_keys=["athlete"])
    weekly_metrics = compute_weekly_summary(df_sessions, daily_metrics, group_keys=["athlete"])
    stats = _extract_weekly_stats(
        df_week,
        daily_metrics,
        weekly_metrics,
        week_end,
        athlete=athlete,
        team=team_scope,
        brand_line=_compose_brand_line(athlete, team_scope, school_name),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    slug = _slugify(athlete)
    pdf_path = output_dir / f"weekly_{slug}_{stats.week_start.isoformat()}_{stats.week_end.isoformat()}.pdf"

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        session_plot = _create_weekly_load_plot(df_week, tmp_dir_path)
        risk_plot = _create_acwr_risk_figure(tmp_dir_path)
        _build_pdf(pdf_path, stats, session_plot, risk_plot, df_week)

    return pdf_path


def _resolve_week_range(week_ending: date | None) -> tuple[date, date]:
    today = date.today()
    if week_ending is None:
        days_since_sunday = (today.weekday() + 1) % 7  # Monday=0, Sunday=6
        week_end = today - timedelta(days=days_since_sunday)
    else:
        week_end = week_ending
    week_start = week_end - timedelta(days=6)
    return week_start, week_end


def _slice_week(df_sessions: pd.DataFrame, week_start: date, week_end: date) -> pd.DataFrame:
    mask = (df_sessions["date"] >= pd.Timestamp(week_start)) & (
        df_sessions["date"] <= pd.Timestamp(week_end)
    )
    return df_sessions.loc[mask].copy()


def _extract_weekly_stats(
    df_week: pd.DataFrame,
    daily_metrics: pd.DataFrame,
    weekly_metrics: pd.DataFrame,
    week_end: date,
    *,
    athlete: str,
    team: str | None,
    brand_line: str | None,
) -> WeeklyReportStats:
    week_iso = date.isocalendar(week_end)
    week_label = f"{week_iso.year}-W{week_iso.week:02d}"
    athlete_name = athlete
    team_name = team

    summary_row = weekly_metrics.loc[
        (weekly_metrics["label"] == week_label) & (weekly_metrics["athlete"].str.lower() == athlete_name.lower())
    ].copy()
    if summary_row.empty:
        raise ValueError("Could not locate aggregated metrics for the requested week.")
    summary = summary_row.iloc[0]

    high_risk_dates = (
        daily_metrics.loc[
            (daily_metrics["risk_flag"] == "HIGH")
            & (daily_metrics["date"] >= pd.Timestamp(summary["start_date"]))
            & (daily_metrics["date"] <= pd.Timestamp(summary["end_date"]))
        ]["date"]
        .dt.date.tolist()
    )

    athlete_daily = daily_metrics[daily_metrics["athlete"].str.lower() == athlete_name.lower()]
    weekly_daily = athlete_daily[
        (athlete_daily["date"] >= pd.Timestamp(summary["start_date"]))
        & (athlete_daily["date"] <= pd.Timestamp(summary["end_date"]))
    ].sort_values("date")
    risk_rows = [
        {
            "date": row["date"].date(),
            "load": float(row["load"]),
            "acwr_rolling": _safe_float(row.get("acwr_rolling")),
            "acwr_ewma": _safe_float(row.get("acwr_ewma")),
            "risk_flag": row.get("risk_flag") or "",
        }
        for _, row in weekly_daily.iterrows()
    ]

    avg_rpe = df_week["rpe"].dropna()

    return WeeklyReportStats(
        athlete=athlete_name,
        team=team_name,
        brand_line=brand_line,
        week_label=week_label,
        week_start=pd.Timestamp(summary["start_date"]).date(),
        week_end=pd.Timestamp(summary["end_date"]).date(),
        session_count=int(summary["sessions"]),
        best_throw=float(summary["best_throw"]),
        total_load=float(summary["total_load"]),
        average_rpe=float(avg_rpe.mean()) if not avg_rpe.empty else None,
        throw_volume=int(summary["throw_volume"]),
        acwr_rolling=_safe_float(summary["acwr_rolling"]),
        acwr_ewma=_safe_float(summary["acwr_ewma"]),
        risk_flag=str(summary["risk_flag"] or ""),
        high_risk_dates=high_risk_dates,
        risk_rows=risk_rows,
    )


def _create_weekly_load_plot(df_week: pd.DataFrame, tmp_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    plot_path = tmp_dir / "weekly_load.png"
    ordered = df_week.sort_values("date")
    fig, ax = plt.subplots()
    ax.bar(ordered["date"].dt.date, ordered["load"], color="#4C72B0")
    ax.set_title("Session Loads (AU)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Load (RPE × duration)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


def _create_acwr_risk_figure(tmp_dir: Path) -> Path:
    import matplotlib.pyplot as plt
    import numpy as np

    risk_path = tmp_dir / "acwr_risk.png"
    acwr_values = np.linspace(0.5, 2.0, 200)
    # Model a simple logistic curve for injury likelihood
    risk = 1 / (1 + np.exp(-6 * (acwr_values - 1.3)))

    fig, ax = plt.subplots()
    ax.plot(acwr_values, risk, color="#C92A2A", linewidth=2)
    ax.axvspan(0.8, 1.3, color="#51CF66", alpha=0.2, label="Low-risk zone (0.8–1.3)")
    ax.axvspan(0.0, 0.8, color="#FCC419", alpha=0.15, label="Underload")
    ax.axvspan(1.5, 2.0, color="#FF6B6B", alpha=0.2, label="High-risk spike (>1.5)")
    ax.set_xlabel("ACWR")
    ax.set_ylabel("Relative Injury Likelihood")
    ax.set_title("ACWR vs. Injury Risk (illustrative)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(risk_path, dpi=150)
    plt.close(fig)
    return risk_path


def _compose_brand_line(athlete: str, team: str | None, school: str | None) -> str | None:
    parts = [part for part in (school, team) if part]
    return " | ".join(parts) if parts else None


def _slugify(value: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "_" for char in value.strip())
    slug = "_".join(token for token in cleaned.split("_") if token)
    return slug or "athlete"


def _build_pdf(
    destination: Path,
    stats: WeeklyReportStats,
    session_plot: Path,
    risk_plot: Path,
    df_week: pd.DataFrame,
) -> None:
    story = []
    styles = getSampleStyleSheet()

    story.append(Paragraph(f"Weekly Summary: {stats.week_start} – {stats.week_end}", styles["Title"]))
    if stats.brand_line:
        story.append(Paragraph(stats.brand_line, styles["Heading4"]))
    story.append(
        Paragraph(
            f"Athlete: <b>{stats.athlete}</b>"
            + (f" | Team: {stats.team}" if stats.team else ""),
            styles["BodyText"],
        )
    )
    story.append(Spacer(1, 0.2 * inch))

    avg_rpe_text = f"{stats.average_rpe:.1f}" if stats.average_rpe is not None else "n/a"
    summary_text = (
        f"Recorded {stats.session_count} sessions with {stats.throw_volume} throws. "
        f"Total load was {stats.total_load:.1f} AU with average RPE {avg_rpe_text}."
    )
    story.append(Paragraph(summary_text, styles["BodyText"]))

    table_data = [
        ["Metric", "Value"],
        ["Best Throw (m)", f"{stats.best_throw:.2f}"],
        ["Total Load (AU)", f"{stats.total_load:.1f}"],
        ["Average RPE", f"{stats.average_rpe:.1f}" if stats.average_rpe is not None else "n/a"],
        ["Throw Volume", str(stats.throw_volume)],
        ["Rolling ACWR", f"{stats.acwr_rolling:.2f}" if stats.acwr_rolling is not None else "n/a"],
        ["EWMA ACWR", f"{stats.acwr_ewma:.2f}" if stats.acwr_ewma is not None else "n/a"],
        ["Risk Flag", stats.risk_flag or "LOW"],
    ]
    table = Table(table_data, hAlign="LEFT", colWidths=[2.5 * inch, 3.5 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F3C88")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ]
        )
    )
    story.append(Spacer(1, 0.2 * inch))
    story.append(table)
    story.append(Spacer(1, 0.3 * inch))

    if stats.risk_rows:
        story.append(Paragraph("Daily Load & Risk Flags", styles["Heading2"]))
        risk_table_data = [
            ["Date", "Load (AU)", "Rolling ACWR", "EWMA ACWR", "Risk"],
        ]
        for row in stats.risk_rows:
            risk_table_data.append(
                [
                    row["date"].isoformat(),
                    f"{row['load']:.1f}",
                    f"{row['acwr_rolling']:.2f}" if row["acwr_rolling"] is not None else "n/a",
                    f"{row['acwr_ewma']:.2f}" if row["acwr_ewma"] is not None else "n/a",
                    row["risk_flag"] or "LOW",
                ]
            )
        risk_table = Table(risk_table_data, hAlign="LEFT", colWidths=[1.3 * inch, 1.1 * inch, 1.4 * inch, 1.4 * inch, 1.1 * inch])
        risk_styles = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B7285")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ALIGN", (1, 1), (-2, -1), "RIGHT"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ]
        risk_palette = {
            "HIGH": colors.Color(0.99, 0.85, 0.85),
            "MODERATE": colors.Color(1.0, 0.95, 0.8),
            "LOW": colors.whitesmoke,
            "": colors.whitesmoke,
        }
        for idx, row in enumerate(stats.risk_rows, start=1):
            risk_styles.append(
                ("BACKGROUND", (0, idx), (-1, idx), risk_palette.get(row["risk_flag"], colors.whitesmoke))
            )
            if row["risk_flag"] == "HIGH":
                risk_styles.append(("TEXTCOLOR", (0, idx), (-1, idx), colors.HexColor("#C92A2A")))
        risk_table.setStyle(TableStyle(risk_styles))
        story.append(risk_table)
        story.append(Spacer(1, 0.3 * inch))

    story.append(Paragraph("Session Loads", styles["Heading2"]))
    story.append(Image(str(session_plot), width=6.5 * inch, height=3.5 * inch))

    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph("ACWR Risk Zones", styles["Heading2"]))
    story.append(Image(str(risk_plot), width=6.5 * inch, height=3.0 * inch))

    if stats.high_risk_dates:
        risk_dates = ", ".join(date.isoformat() for date in stats.high_risk_dates)
        story.append(Spacer(1, 0.2 * inch))
        story.append(
            Paragraph(
                f"<b>High-risk alerts:</b> {stats.athlete} recorded ACWR spikes on {risk_dates}. "
                "Consider adjusting workload to return to the 0.8–1.3 zone.",
                styles["BodyText"],
            )
        )

    story.append(Spacer(1, 0.2 * inch))
    notable_notes = df_week[df_week["notes"].astype(str).str.len() > 0][["date", "notes"]]
    if not notable_notes.empty:
        story.append(Paragraph("Session Notes", styles["Heading2"]))
        for _, row in notable_notes.iterrows():
            story.append(
                Paragraph(f"{row['date'].date().isoformat()}: {row['notes']}", styles["BodyText"])
            )

    doc = SimpleDocTemplate(str(destination), pagesize=letter, title=f"Weekly Report {stats.week_label}")
    doc.build(story)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
