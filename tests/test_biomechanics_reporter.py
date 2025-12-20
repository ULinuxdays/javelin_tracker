from __future__ import annotations

import json
from pathlib import Path

import pytest

from javelin_tracker.biomechanics.comparison.reporter import generate_comparison_report


def test_generate_comparison_report_outputs_expected_shape(tmp_path: Path) -> None:
    athlete = {
        "release.throwing_wrist_speed_at_release": 14.0,
        "approach.duration_ms": 1200.0,
        "delivery.throwing_wrist_speed_max": 8.0,
    }
    elite = {
        "release.throwing_wrist_speed_at_release": {"mean": 10.0, "std": 2.0, "phase": "release", "unit": "rel_units/s"},
        "approach.duration_ms": {"mean": 1000.0, "std": 100.0, "phase": "approach", "unit": "ms"},
        "delivery.throwing_wrist_speed_max": {"mean": 9.0, "std": 1.0, "phase": "delivery", "unit": "rel_units/s"},
    }

    report = generate_comparison_report(athlete, elite, athlete_id="athlete_1")
    assert report["athlete_id"] == "athlete_1"
    assert 0.0 <= float(report["overall_score"]) <= 100.0
    assert set(report["phase_breakdown"]) == {"approach", "delivery", "release"}
    assert isinstance(report["top_issues"], list)
    assert len(report["top_issues"]) <= 5
    assert isinstance(report["strengths"], list)
    assert isinstance(report["weaknesses"], list)
    assert int(report["n_metrics_scored"]) == 3

    # Top issues should be ranked.
    if len(report["top_issues"]) >= 2:
        assert int(report["top_issues"][0]["rank"]) == 1
        assert int(report["top_issues"][1]["rank"]) == 2


def test_generate_comparison_report_accepts_json_paths(tmp_path: Path) -> None:
    athlete_path = tmp_path / "athlete_metrics.json"
    elite_path = tmp_path / "elite_profile.json"

    athlete_path.write_text(json.dumps({"release.throwing_wrist_speed_at_release": 10.0}), encoding="utf-8")
    elite_path.write_text(
        json.dumps(
            {
                "release.throwing_wrist_speed_at_release": {
                    "mean": 10.0,
                    "std": 2.0,
                    "phase": "release",
                    "unit": "rel_units/s",
                }
            }
        ),
        encoding="utf-8",
    )

    report = generate_comparison_report(athlete_path, elite_path, athlete_id="athlete_x")
    assert report["overall_score"] == pytest.approx(100.0)


def test_generate_comparison_report_raises_when_no_overlap() -> None:
    athlete = {"release.some_metric": 1.0}
    elite = {"delivery.other_metric": {"mean": 0.0, "std": 1.0, "phase": "delivery", "unit": "x"}}
    with pytest.raises(ValueError, match="No comparable metrics"):
        generate_comparison_report(athlete, elite, athlete_id="athlete_2")

