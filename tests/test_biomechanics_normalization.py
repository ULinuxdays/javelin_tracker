from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from javelin_tracker.biomechanics.metrics.normalization import (
    denormalize_positions,
    denormalize_velocities,
    handle_variable_fps,
    normalize_angles,
    normalize_athlete_metrics,
    normalize_positions,
    normalize_velocities,
)


def test_normalize_positions_roundtrip() -> None:
    landmarks = np.zeros((2, 33, 4), dtype=float)
    landmarks[:, :, 3] = 1.0
    landmarks[:, :, 0] = 100.0
    landmarks[:, :, 1] = 50.0
    landmarks[:, :, 2] = 10.0

    body_height_px = 200.0
    normalized = normalize_positions(landmarks, body_height_px)
    assert normalized.shape == landmarks.shape
    assert float(normalized[0, 0, 0]) == pytest.approx(0.5)
    assert float(normalized[0, 0, 1]) == pytest.approx(0.25)

    restored = denormalize_positions(normalized, body_height_px)
    assert np.allclose(restored[:, :, :3], landmarks[:, :, :3])
    assert np.allclose(restored[:, :, 3], landmarks[:, :, 3])


def test_normalize_velocities_and_denormalize_velocities_roundtrip() -> None:
    velocities_px_per_frame = np.array([0.0, 10.0, 20.0], dtype=float)
    body_height_px = 200.0
    fps = 50.0
    athlete_height_m = 2.0

    normalized = normalize_velocities(
        velocities_px_per_frame,
        body_height_px,
        fps,
        input_unit="per_frame",
        athlete_height_m=athlete_height_m,
    )
    # factor = fps*(athlete_height_m/body_height_px) = 50*(2/200)=0.5
    assert normalized[1] == pytest.approx(5.0)

    restored = denormalize_velocities(
        normalized,
        body_height_px,
        fps,
        output_unit="per_frame",
        athlete_height_m=athlete_height_m,
    )
    assert np.allclose(restored, velocities_px_per_frame)


def test_handle_variable_fps_resamples_dataframe() -> None:
    df = pd.DataFrame(
        {
            "frame": [0, 1, 2, 3, 4, 5],
            "angle_name": ["test"] * 6,
            "value_degrees": [0, 1, 2, 3, 4, 5],
        }
    )
    out = handle_variable_fps(df, original_fps=60.0, target_fps=30.0)
    assert "timestamp_ms" in out.columns
    assert out["angle_name"].nunique() == 1
    assert len(out) == 3
    assert float(out.iloc[1]["value_degrees"]) == pytest.approx(2.0)
    assert float(out.iloc[2]["value_degrees"]) == pytest.approx(4.0)


def test_normalize_athlete_metrics_converts_landmarks_and_velocities() -> None:
    width = 1000
    height = 1000
    fps = 50.0

    LS, RS = 11, 12
    LA, RA = 27, 28

    def make_frame(y_shoulder: float, y_ankle: float) -> dict[str, object]:
        landmarks = [(0.0, 0.0, 0.0, 1.0) for _ in range(33)]
        landmarks[LS] = (0.4, y_shoulder, 0.0, 1.0)
        landmarks[RS] = (0.6, y_shoulder, 0.0, 1.0)
        landmarks[LA] = (0.4, y_ankle, 0.0, 1.0)
        landmarks[RA] = (0.6, y_ankle, 0.0, 1.0)
        return {"landmarks": landmarks, "valid": True}

    frames = [make_frame(0.2, 0.8), make_frame(0.2, 0.8)]
    # speed in px/s; 60 px/s should become 0.2 m/s if body_height_px=600 and athlete_height_m=2.0.
    kin_df = pd.DataFrame({"frame": [0, 1], "timestamp_ms": [0.0, 20.0], "vx": [60.0, 60.0], "vy": [0.0, 0.0], "vz": [0.0, 0.0], "speed": [60.0, 60.0]})
    angles_df = pd.DataFrame({"frame": [0], "timestamp_ms": [0.0], "angle_name": ["x"], "value_degrees": [90.0], "valid": [True]})

    out = normalize_athlete_metrics(
        {"frames": frames, "kinematics_df": kin_df, "angles_df": angles_df},
        {"fps": fps, "width": width, "height": height, "athlete_height_m": 2.0},
    )
    ctx = out["context"]
    assert float(ctx.body_height_px) == pytest.approx(600.0)

    normalized_landmarks = out["normalized_landmarks"]
    # Shoulder y=0.2*1000=200px -> 200/600=0.333...
    assert float(normalized_landmarks[0, LS, 1]) == pytest.approx(200.0 / 600.0)

    norm_kin = out["normalized_kinematics_df"]
    assert float(norm_kin["speed"].iloc[0]) == pytest.approx(0.2)

    norm_angles = normalize_angles(angles_df)
    assert float(norm_angles["value_degrees"].iloc[0]) == pytest.approx(90.0)

