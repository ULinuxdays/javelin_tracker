from __future__ import annotations

from javelin_tracker.biomechanics.feedback.rules import evaluate_rules


def test_evaluate_rules_atomic_with_alias() -> None:
    athlete = {"release.throwing_elbow_flexion_deg_at_release": 80.0}
    rules = {
        "metric_aliases": {"elbow_angle_at_release": "release.throwing_elbow_flexion_deg_at_release"},
        "rules": [
            {
                "rule_id": "elbow_extension_release",
                "priority": 10,
                "severity": "warning",
                "feedback_text": "Extend your elbow fully at release",
                "condition": {"metric": "elbow_angle_at_release", "op": "<", "value": 85},
            }
        ],
    }

    matches = evaluate_rules(athlete, rules)
    assert [m["rule_id"] for m in matches] == ["elbow_extension_release"]
    assert matches[0]["metric_value"] == 80.0
    assert matches[0]["ideal_value"] == 85


def test_evaluate_rules_supports_range_and_and_or_logic() -> None:
    athlete = {"a": 2.0, "b": 6.0, "c": 3.0, "speed": 11.0}
    rules = {
        "rules": [
            {
                "rule_id": "combo",
                "priority": 5,
                "severity": "info",
                "feedback_text": "combo cue",
                "condition": {
                    "all": [
                        {"metric": "a", "op": ">", "value": 1},
                        {"any": [{"metric": "b", "op": "<", "value": 5}, {"metric": "c", "op": "==", "value": 3}]},
                        {"metric": "speed", "op": "range", "min": 10, "max": 12},
                    ]
                },
            }
        ]
    }

    matches = evaluate_rules(athlete, rules)
    assert [m["rule_id"] for m in matches] == ["combo"]


def test_evaluate_rules_supports_dotted_path_lookup() -> None:
    athlete = {"throw_metrics": {"power_chain_lag_ms": "120"}}
    rules = {
        "rules": [
            {
                "rule_id": "lag",
                "priority": 5,
                "severity": "warning",
                "feedback_text": "Delay arm acceleration; let hips drive first",
                "condition": {"metric": "throw_metrics.power_chain_lag_ms", "op": ">", "value": 100},
            }
        ]
    }
    matches = evaluate_rules(athlete, rules)
    assert [m["rule_id"] for m in matches] == ["lag"]
    assert matches[0]["metric_value"] == 120.0


def test_evaluate_rules_skips_missing_metrics() -> None:
    athlete = {"some_metric": 1.0}
    rules = {
        "rules": [
            {
                "rule_id": "missing",
                "priority": 1,
                "severity": "info",
                "feedback_text": "nope",
                "condition": {"metric": "unknown_metric", "op": ">", "value": 0},
            }
        ]
    }
    assert evaluate_rules(athlete, rules) == []


def test_evaluate_rules_sorts_by_priority_then_severity() -> None:
    athlete = {"m": 1.0}
    rules = {
        "rules": [
            {
                "rule_id": "low",
                "priority": 1,
                "severity": "critical",
                "feedback_text": "low",
                "condition": {"metric": "m", "op": "==", "value": 1},
            },
            {
                "rule_id": "high",
                "priority": 10,
                "severity": "info",
                "feedback_text": "high",
                "condition": {"metric": "m", "op": "==", "value": 1},
            },
        ]
    }
    matches = evaluate_rules(athlete, rules)
    assert [m["rule_id"] for m in matches] == ["high", "low"]

