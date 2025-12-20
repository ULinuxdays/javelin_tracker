from __future__ import annotations

from javelin_tracker.biomechanics.comparison.reporter import generate_comparison_report
from javelin_tracker.biomechanics.comparison.scoring import infer_athlete_style


def _style_profiles_fixture():
    # Minimal style profiles in the same shape as compute_style_profiles():
    # {style: {"metrics": {metric_key: {mean,std,...}}, "n_samples": int, "throwers": [...]}}
    return {
        "overall": {
            "metrics": {
                "release.throwing_wrist_speed_at_release": {
                    "mean": 10.0,
                    "std": 2.0,
                    "phase": "release",
                    "unit": "rel_units/s",
                    "confidence": 0.9,
                    "n_samples": 10,
                }
            },
            "n_samples": 10,
            "throwers": ["A", "B"],
        },
        "finnish": {
            "metrics": {
                "release.throwing_wrist_speed_at_release": {
                    "mean": 14.0,
                    "std": 1.0,
                    "phase": "release",
                    "unit": "rel_units/s",
                    "confidence": 0.9,
                    "n_samples": 3,
                }
            },
            "n_samples": 3,
            "throwers": ["C"],
        },
        # Insufficient samples -> should be skipped/fallback.
        "czech": {
            "metrics": {
                "release.throwing_wrist_speed_at_release": {
                    "mean": 8.0,
                    "std": 1.0,
                    "phase": "release",
                    "unit": "rel_units/s",
                    "confidence": 0.9,
                    "n_samples": 1,
                }
            },
            "n_samples": 1,
            "throwers": ["D"],
        },
    }


def test_infer_athlete_style_best_match() -> None:
    styles = _style_profiles_fixture()
    athlete = {"release.throwing_wrist_speed_at_release": 14.0}
    inferred = infer_athlete_style(athlete, styles)
    assert inferred["style_name"] == "finnish"
    assert inferred["metrics_compared"] == 1
    assert float(inferred["confidence"]) > 0.5


def test_generate_report_auto_style_uses_inferred_style() -> None:
    styles = _style_profiles_fixture()
    athlete = {"release.throwing_wrist_speed_at_release": 14.0}
    report = generate_comparison_report(athlete, styles["overall"]["metrics"], "ath_1", style="auto", all_style_profiles=styles)
    assert report["style_comparison"]["style_name"] == "finnish"
    assert report["style_comparison"]["overridden"] is False
    assert float(report["style_comparison"]["confidence"]) > 0.5
    assert int(report["style_comparison"]["metrics_compared"]) == int(report["n_metrics_scored"]) == 1


def test_generate_report_style_override_falls_back_to_overall() -> None:
    styles = _style_profiles_fixture()
    athlete = {"release.throwing_wrist_speed_at_release": 14.0}
    report = generate_comparison_report(athlete, styles["overall"]["metrics"], "ath_1", style="czech", all_style_profiles=styles)
    assert report["style_comparison"]["style_name"] == "overall"
    assert report["style_comparison"]["overridden"] is True
    assert "falling back" in str(report["style_comparison"]["note_about_style"]).lower()

