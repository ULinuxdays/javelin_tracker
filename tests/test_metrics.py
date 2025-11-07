from __future__ import annotations

from datetime import date, timedelta
import pytest

from javelin_tracker.metrics import (
    compute_daily_metrics,
    compute_weekly_summary,
    sessions_to_dataframe,
)
from javelin_tracker.services import build_summary_report


def _make_sessions() -> list[dict[str, object]]:
    sessions: list[dict[str, object]] = []
    base_day = date(2024, 1, 1)
    for offset in range(30):
        day = base_day + timedelta(days=offset)
        if offset < 29:
            rpe = 5
            duration = 10.0
        else:
            rpe = 10
            duration = 25.0  # load spike on final day
        sessions.append(
            {
                "date": day.isoformat(),
                "best": 60.0 + offset * 0.2,
                "throws": [60.0 + offset * 0.2, 59.5 + offset * 0.2],
                "rpe": rpe,
                "duration_minutes": duration,
                "notes": "",
                "tags": ["training"],
                "athlete": "athlete-alpha",
                "team": "Team A",
            }
        )
    return sessions


def test_sessions_to_dataframe_load_calculation() -> None:
    df = sessions_to_dataframe(_make_sessions())
    assert not df.empty
    first_row = df.iloc[0]
    assert first_row["load"] == pytest.approx(first_row["rpe"] * first_row["duration_minutes"])


def test_daily_metrics_flags_high_acwr() -> None:
    df = sessions_to_dataframe(_make_sessions())
    daily = compute_daily_metrics(df)
    assert not daily.empty
    final_day = daily.iloc[-1]
    assert final_day["acwr_rolling"] > 1.5
    assert final_day["risk_flag"] == "HIGH"


def test_weekly_summary_includes_acwr_metrics() -> None:
    df = sessions_to_dataframe(_make_sessions())
    daily = compute_daily_metrics(df)
    weekly = compute_weekly_summary(df, daily)
    assert "acwr_rolling" in weekly.columns
    assert "acwr_ewma" in weekly.columns
    assert weekly["acwr_rolling"].notna().any()


def test_build_summary_report_tracks_high_risk_dates() -> None:
    sessions = _make_sessions()
    report = build_summary_report(sessions, group="week")
    assert report.total_sessions == len(sessions)
    assert report.high_risk_dates, "Expected high-risk dates due to load spike"
    alpha_dates = report.high_risk_dates.get("athlete-alpha")
    assert alpha_dates
    assert any(date.isoformat().startswith("2024-01") for date in alpha_dates)


def test_summary_rows_include_athlete_metadata() -> None:
    sessions = _make_sessions()
    sessions.append(
        {
            "date": date(2024, 2, 15).isoformat(),
            "best": 70.0,
            "throws": [70.0],
            "rpe": 6,
            "duration_minutes": 45.0,
            "athlete": "athlete-bravo",
            "team": "Team B",
        }
    )
    report = build_summary_report(sessions, group="week")
    assert any(row.athlete == "athlete-bravo" for row in report.rows)
    assert "athlete-bravo" in report.personal_bests
