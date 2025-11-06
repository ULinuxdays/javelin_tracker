from __future__ import annotations

from datetime import date

import pytest

from javelin_tracker.analysis import (
    group_by_week,
    median_best_by_group,
    personal_best,
    total_throws_by_group,
)
from javelin_tracker.models import clamp_rpe, parse_throws


def _build_sessions() -> list[dict[str, object]]:
    return [
        {
            "date": "2024-03-01",
            "best": 61.2,
            "throws": [59.8, 60.4, 61.2],
            "tags": ["technique"],
        },
        {
            "date": "2024-03-05",
            "best": 63.0,
            "throws": [62.1, 63.0],
            "tags": ["run-up"],
        },
        {
            "date": "2024-03-08",
            "best": 62.4,
            "throws": [61.9, 62.4],
            "tags": ["speed"],
        },
        {
            "date": "2024-03-12",
            "best": 64.1,
            "throws": [63.8, 64.1],
            "tags": ["competition"],
        },
    ]


def test_personal_best_returns_highest_distance_and_date() -> None:
    sessions = _build_sessions()
    pb_date, pb_distance = personal_best(sessions)

    assert pb_date == date.fromisoformat("2024-03-12")
    assert pb_distance == pytest.approx(64.1)


def test_group_by_week_clusters_sessions_by_iso_week() -> None:
    sessions = _build_sessions()
    groups = group_by_week(sessions)

    expected_keys = {
        f"{date.fromisoformat(session['date']).isocalendar()[0]}-W{date.fromisoformat(session['date']).isocalendar()[1]:02d}"
        for session in sessions
    }
    assert set(groups.keys()) == expected_keys

    # Ensure dates inside each group match the key they belong to
    for key, entries in groups.items():
        iso_year, iso_week = key.split("-W")
        for entry in entries:
            y, w, _ = entry["date"].isocalendar()
            assert str(y) == iso_year
            assert f"{w:02d}" == iso_week


def test_median_best_by_group_matches_statistics_median() -> None:
    sessions = _build_sessions()
    groups = group_by_week(sessions)
    medians = median_best_by_group(groups)

    from statistics import median

    for key, entries in groups.items():
        expected = median(entry["best"] for entry in entries)
        assert medians[key] == pytest.approx(expected)


def test_total_throw_volume_counts_per_group() -> None:
    sessions = _build_sessions()
    groups = group_by_week(sessions)
    volumes = total_throws_by_group(groups)

    for key, entries in groups.items():
        expected_total = sum(len(entry["throws"]) for entry in entries)
        assert volumes[key] == expected_total


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("61.0, 62.0, 63.5", ["61.0", "62.0", "63.5"]),
        (" 61.0 , , 62.5 ,63.0 ", ["61.0", "62.5", "63.0"]),
        (["61.0", " 62.0 "], ["61.0", "62.0"]),
        (None, []),
    ],
)
def test_parse_throws_normalises_entries(raw, expected) -> None:
    assert parse_throws(raw) == expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        (10, 10),
        (11, 10),
        (-2, 1),
        ("7", 7),
    ],
)
def test_clamp_rpe_enforces_bounds(raw, expected) -> None:
    assert clamp_rpe(raw) == expected
