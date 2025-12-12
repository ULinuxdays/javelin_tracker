from __future__ import annotations

from datetime import date, timedelta
import pytest

from javelin_tracker.metrics import (
    compute_daily_metrics,
    compute_weekly_summary,
    sessions_to_dataframe,
)
from javelin_tracker.services import build_summary_report, generate_plots


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
            duration = 35.0  # load spike on final day
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
    assert "monotony_7d" in daily.columns
    assert final_day["monotony_7d"] >= 0
    assert final_day["strain_7d"] >= 0


def test_weekly_summary_includes_acwr_metrics() -> None:
    df = sessions_to_dataframe(_make_sessions())
    daily = compute_daily_metrics(df)
    weekly = compute_weekly_summary(df, daily)
    assert "acwr_rolling" in weekly.columns
    assert "acwr_ewma" in weekly.columns
    assert "monotony_7d" in weekly.columns
    assert "strain_7d" in weekly.columns
    assert weekly["acwr_rolling"].notna().any()


def test_build_summary_report_tracks_high_risk_dates() -> None:
    sessions = _make_sessions()
    report = build_summary_report(sessions, group="week")
    assert report.total_sessions == len(sessions)
    assert report.high_risk_dates, "Expected high-risk dates due to load spike"
    alpha_dates = report.high_risk_dates.get(("athlete-alpha", "javelin"))
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
    assert ("athlete-bravo", "javelin") in report.personal_bests


def test_summary_report_includes_event_rollup() -> None:
    sessions = _make_sessions()
    sessions.extend(
        [
            {
                "date": date(2024, 1, 5).isoformat(),
                "best": 45.0,
                "throws": [44.0, 45.0],
                "athlete": "athlete-alpha",
                "event": "discus",
                "rpe": 6,
                "duration_minutes": 40,
            },
            {
                "date": date(2024, 1, 7).isoformat(),
                "best": 19.0,
                "throws": [18.5, 19.0],
                "athlete": "athlete-bravo",
                "event": "shot",
                "rpe": 7,
                "duration_minutes": 35,
            },
        ]
    )
    report = build_summary_report(sessions, group="week")
    assert report.rollup_rows, "Expected all-event rollup rows when no event filter is provided."
    assert any(row.event == "discus" for row in report.rows)
    assert sum(row.sessions for row in report.rollup_rows) == report.total_sessions


def test_generate_plots_include_event_labels(tmp_path, monkeypatch) -> None:
    sessions = [
        {
            "date": date(2024, 4, 1).isoformat(),
            "best": 60.0,
            "throws": [60.0, 59.5],
            "athlete": "alpha",
            "event": "javelin",
            "rpe": 6,
            "duration_minutes": 45.0,
        },
        {
            "date": date(2024, 4, 2).isoformat(),
            "best": 45.0,
            "throws": [44.0, 45.0],
            "athlete": "alpha",
            "event": "discus",
            "rpe": 5,
            "duration_minutes": 35.0,
        },
    ]

    import matplotlib.pyplot as plt

    captured_axes = []
    original_subplots = plt.subplots

    def capture_subplots(*args, **kwargs):
        fig, ax = original_subplots(*args, **kwargs)
        captured_axes.append(ax)
        return fig, ax

    monkeypatch.setattr("matplotlib.pyplot.subplots", capture_subplots)

    paths = generate_plots(sessions, output_dir=tmp_path)
    assert len(paths) == 2
    for path in paths:
        assert path.exists()

    legend_labels: list[str] = []
    for ax in captured_axes:
        _, labels = ax.get_legend_handles_labels()
        legend_labels.extend(labels)

    assert any("Javelin" in label for label in legend_labels)
    assert any("Discus" in label for label in legend_labels)
