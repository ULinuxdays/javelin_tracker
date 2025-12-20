from __future__ import annotations

from javelin_tracker.biomechanics.feedback.generator import generate_feedback


def test_generate_feedback_ranks_by_impact_weighted_abs_z() -> None:
    comparison_report = {
        "top_issues": [
            # Higher |z| but low weight -> lower impact
            {
                "rank": 1,
                "metric": "approach.duration_ms",
                "phase": "approach",
                "unit": "ms",
                "athlete_value": 1300.0,
                "elite_mean": 1000.0,
                "elite_std": 100.0,
                "z_score": 3.0,
                "status": "poor",
                "weight": 0.5,
            },
            # Lower |z| but high weight -> higher impact
            {
                "rank": 2,
                "metric": "release.throwing_wrist_speed_at_release",
                "phase": "release",
                "unit": "rel_units/s",
                "athlete_value": 8.0,
                "elite_mean": 10.0,
                "elite_std": 1.0,
                "z_score": -2.0,
                "status": "poor",
                "weight": 2.0,
            },
        ]
    }
    throw_metrics = {
        "phase_durations": {
            "approach_start_frame": 0,
            "delivery_start_frame": 65,
            "release_frame": 90,
            "frames": 100,
        },
        "release_metrics": {"release_frame": 90},
        "power_chain_lag_ms": 120.0,
    }

    cues = generate_feedback(comparison_report, throw_metrics, rules=None)
    assert len(cues) == 2
    assert cues[0]["issue_rank"] == 1
    assert cues[0]["metric_name"] == "release.throwing_wrist_speed_at_release"
    assert cues[1]["metric_name"] == "approach.duration_ms"


def test_generate_feedback_includes_required_fields_and_frame_ranges() -> None:
    comparison_report = {
        "top_issues": [
            {
                "rank": 1,
                "metric": "release.throwing_wrist_speed_at_release",
                "phase": "release",
                "unit": "rel_units/s",
                "athlete_value": 8.0,
                "elite_mean": 10.0,
                "elite_std": 1.0,
                "z_score": -2.0,
                "status": "poor",
                "weight": 2.0,
            }
        ]
    }
    throw_metrics = {
        "phase_durations": {
            "approach_start_frame": 0,
            "delivery_start_frame": 65,
            "release_frame": 90,
            "frames": 100,
        },
        "release_metrics": {"release_frame": 90},
    }

    cues = generate_feedback(comparison_report, throw_metrics, rules=None)
    assert len(cues) == 1
    cue = cues[0]
    for key in (
        "issue_rank",
        "metric_name",
        "feedback_text",
        "ideal_range",
        "current_value",
        "suggested_correction",
        "drill_suggestion",
        "severity",
    ):
        assert key in cue
    assert cue["severity"] == "major"
    assert isinstance(cue["frame_ranges"], list)
    assert cue["frame_ranges"]
    # Should include a release-focused range around the release frame.
    release_ranges = [r for r in cue["frame_ranges"] if r.get("phase") == "release"]
    assert release_ranges
    rr = release_ranges[0]
    assert int(rr["start_frame"]) <= 90 <= int(rr["end_frame"])


def test_generate_feedback_uses_matching_rule_text_when_applicable() -> None:
    comparison_report = {
        "top_issues": [
            {
                "rank": 1,
                "metric": "release.throwing_elbow_flexion_deg_at_release",
                "phase": "release",
                "unit": "degrees",
                "athlete_value": 80.0,
                "elite_mean": 100.0,
                "elite_std": 10.0,
                "z_score": -2.0,
                "status": "poor",
                "weight": 1.0,
            }
        ]
    }
    throw_metrics = {
        "phase_durations": {"approach_start_frame": 0, "delivery_start_frame": 65, "release_frame": 90, "frames": 100},
        "release_metrics": {"release_frame": 90},
    }
    rules = {
        "metric_aliases": {"elbow_angle_at_release": "release.throwing_elbow_flexion_deg_at_release"},
        "rules": [
            {
                "rule_id": "elbow_extension_release",
                "priority": 10,
                "severity": "warning",
                "feedback_text": "Extend your elbow fully at release.",
                "condition": {"metric": "elbow_angle_at_release", "op": "<", "value": 85},
            }
        ],
    }

    cues = generate_feedback(comparison_report, throw_metrics, rules=rules)
    assert len(cues) == 1
    assert "Extend your elbow fully at release" in str(cues[0]["feedback_text"])

