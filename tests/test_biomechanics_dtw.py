from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from javelin_tracker.biomechanics.comparison.dtw import (
    compare_throw_to_elite,
    compute_dtw_distance,
    dtw_with_path,
    plot_aligned_sequences,
)


def test_compute_dtw_distance_identical_sequences() -> None:
    seq = [10.0, 11.0, 12.0, 13.0, 14.0]
    dist = compute_dtw_distance(seq, seq)
    assert dist == pytest.approx(0.0)


def test_dtw_with_path_returns_valid_path() -> None:
    a = np.linspace(0.0, 1.0, 10)
    b = np.linspace(0.0, 1.0, 15)
    dist, path = dtw_with_path(a, b)
    assert math.isfinite(dist)
    assert path[0] == (0, 0)
    assert path[-1] == (len(a) - 1, len(b) - 1)
    assert len(path) >= max(len(a), len(b))


def test_compare_throw_to_elite_angles_dataframe() -> None:
    athlete = pd.DataFrame(
        {
            "frame": list(range(10)) * 2,
            "angle_name": ["right_elbow"] * 10 + ["thorax_rotation"] * 10,
            "value_degrees": np.concatenate([np.linspace(60.0, 120.0, 10), np.linspace(10.0, 30.0, 10)]),
            "valid": [True] * 20,
        }
    )
    elite = pd.DataFrame(
        {
            "frame": list(range(15)) * 2,
            "angle_name": ["right_elbow"] * 15 + ["thorax_rotation"] * 15,
            "value_degrees": np.concatenate([np.linspace(62.0, 118.0, 15), np.linspace(12.0, 28.0, 15)]),
            "valid": [True] * 30,
        }
    )

    result = compare_throw_to_elite({"angles": athlete}, {"angles": elite})
    assert set(result) >= {
        "dtw_distance",
        "normalized_distance",
        "alignment_quality",
        "reference_angle",
        "per_angle_distances",
    }
    assert math.isfinite(float(result["dtw_distance"]))
    assert 0.0 <= float(result["normalized_distance"]) <= 100.0
    assert 0.0 <= float(result["alignment_quality"]) <= 100.0
    assert "right_elbow" in result["per_angle_distances"]


def test_plot_aligned_sequences_returns_figure() -> None:
    a = np.sin(np.linspace(0.0, 2.0 * np.pi, 20))
    b = np.sin(np.linspace(0.0, 2.0 * np.pi, 30))
    fig = plot_aligned_sequences(a, b)
    assert fig is not None

