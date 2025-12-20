from __future__ import annotations

import math

import pytest

from javelin_tracker.biomechanics.comparison.scoring import (
    get_metric_description,
    get_metric_weight,
    score_all_metrics,
    score_metric,
)


def test_score_metric_status_thresholds_and_score() -> None:
    out = score_metric(10.0, 10.0, 2.0)
    assert out["z_score"] == pytest.approx(0.0)
    assert out["score"] == pytest.approx(100.0)
    assert out["status"] == "good"
    assert out["flagged"] is False

    out = score_metric(12.0, 10.0, 2.0)  # z=1.0
    assert out["z_score"] == pytest.approx(1.0)
    assert out["status"] == "warning"
    assert out["flagged"] is False

    out = score_metric(13.0, 10.0, 2.0)  # z=1.5
    assert out["z_score"] == pytest.approx(1.5)
    assert out["status"] == "warning"
    assert out["flagged"] is False

    out = score_metric(14.0, 10.0, 2.0)  # z=2.0
    assert out["z_score"] == pytest.approx(2.0)
    assert out["score"] == pytest.approx(80.0)
    assert out["status"] == "poor"
    assert out["flagged"] is True


def test_score_metric_handles_zero_std() -> None:
    out = score_metric(10.0, 10.0, 0.0)
    assert out["z_score"] == pytest.approx(0.0)
    assert out["score"] == pytest.approx(100.0)
    assert out["status"] == "good"

    out = score_metric(11.0, 10.0, 0.0)
    assert math.isinf(float(out["z_score"]))
    assert out["score"] == pytest.approx(0.0)
    assert out["status"] == "poor"
    assert out["flagged"] is True


def test_score_all_metrics_ranks_by_severity_and_weight() -> None:
    athlete = {
        "release.throwing_wrist_speed_at_release": 14.0,  # z=2.0 (poor)
        "approach.duration_ms": 1200.0,  # z=2.0 (poor) but lower weight
    }
    reference = {
        "release.throwing_wrist_speed_at_release": {
            "mean": 10.0,
            "std": 2.0,
            "phase": "release",
            "unit": "rel_units/s",
            "confidence": 0.9,
            "n_samples": 5,
        },
        "approach.duration_ms": {
            "mean": 1000.0,
            "std": 100.0,
            "phase": "approach",
            "unit": "ms",
            "confidence": 0.9,
            "n_samples": 5,
        },
    }

    scored = score_all_metrics(athlete, reference)
    assert len(scored) == 2
    assert scored[0]["metric"] == "release.throwing_wrist_speed_at_release"
    assert scored[0]["status"] == "poor"
    assert scored[1]["metric"] == "approach.duration_ms"


def test_get_metric_description_and_weight() -> None:
    desc = get_metric_description("release.throwing_wrist_speed_at_release")
    assert desc.lower().startswith("release phase:")

    w_release = get_metric_weight("release.throwing_wrist_speed_at_release")
    w_approach = get_metric_weight("approach.duration_ms")
    assert w_release > w_approach

