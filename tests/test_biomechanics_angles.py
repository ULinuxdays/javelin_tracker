from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest

from javelin_tracker.biomechanics.metrics.angles import (
    compute_frame_angles,
    compute_joint_angle,
    compute_trajectory_angles,
)


def test_compute_joint_angle_handles_2d_and_3d() -> None:
    angle_2d = compute_joint_angle((1.0, 0.0), (0.0, 0.0), (0.0, 1.0))
    assert angle_2d == pytest.approx(90.0)

    angle_3d = compute_joint_angle((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    assert angle_3d == pytest.approx(90.0)


def test_compute_joint_angle_returns_nan_on_low_confidence() -> None:
    angle = compute_joint_angle((1.0, 0.0, 0.0, 0.1), (0.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0))
    assert math.isnan(angle)


@dataclass(frozen=True)
class _DummyConfig:
    CONFIDENCE_THRESHOLD: float = 0.5
    JAVELIN_KEY_JOINTS: dict[str, tuple[int, int]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "JAVELIN_KEY_JOINTS",
            {
                "shoulders": (11, 12),
                "elbows": (13, 14),
                "wrists": (15, 16),
                "hips": (23, 24),
                "knees": (25, 26),
            },
        )


def test_compute_frame_angles_outputs_expected_keys() -> None:
    landmarks = np.zeros((33, 4), dtype=float)
    landmarks[:, 3] = 1.0

    LS, RS = 11, 12
    RE, RW = 14, 16
    LH, RH = 23, 24
    RK, RA = 26, 28

    # Right elbow 90 degrees: RS -> RE -> RW.
    landmarks[RS, :3] = (0.0, 0.0, 0.0)
    landmarks[RE, :3] = (1.0, 0.0, 0.0)
    landmarks[RW, :3] = (1.0, 1.0, 0.0)

    # Thorax rotation 90 degrees (hip line along x, shoulder line along y).
    landmarks[LS, :3] = (0.0, -2.0, 0.0)
    landmarks[LH, :3] = (-1.0, -1.0, 0.0)
    landmarks[RH, :3] = (1.0, -1.0, 0.0)

    # Right hip/knee/ankle for simple non-degenerate angles.
    landmarks[RK, :3] = (2.0, -2.0, 0.0)
    landmarks[RA, :3] = (3.0, -2.0, 0.0)

    angles = compute_frame_angles(landmarks, _DummyConfig())
    assert set(angles.keys()) == {"right_elbow", "right_shoulder", "right_hip", "right_knee", "thorax_rotation"}

    assert angles["right_elbow"]["valid"] is True
    assert float(angles["right_elbow"]["value_degrees"]) == pytest.approx(90.0)

    assert angles["thorax_rotation"]["valid"] is True
    assert float(angles["thorax_rotation"]["value_degrees"]) == pytest.approx(90.0)


def test_compute_trajectory_angles_returns_dataframe() -> None:
    landmarks0 = np.zeros((33, 4), dtype=float)
    landmarks0[:, 3] = 1.0
    landmarks1 = np.zeros((33, 4), dtype=float)
    landmarks1[:, 3] = 1.0

    RS, RE, RW = 12, 14, 16
    RH, RK, RA = 24, 26, 28
    LH, LS = 23, 11

    for lm in (landmarks0, landmarks1):
        lm[RS, :3] = (0.0, 0.0, 0.0)
        lm[RE, :3] = (1.0, 0.0, 0.0)
        lm[RW, :3] = (1.0, 1.0, 0.0)
        lm[RH, :3] = (0.0, -1.0, 0.0)
        lm[RK, :3] = (1.0, -2.0, 0.0)
        lm[RA, :3] = (2.0, -2.0, 0.0)
        lm[LH, :3] = (-1.0, 0.0, 0.0)
        lm[LS, :3] = (0.0, -1.0, 0.0)

    traj = np.stack([landmarks0, landmarks1], axis=0)
    df = compute_trajectory_angles(traj, fps=50.0)
    assert not df.empty
    assert set(df.columns) == {"frame", "timestamp_ms", "angle_name", "value_degrees", "valid"}
    assert df["frame"].min() == 0
    assert df["frame"].max() == 1
    assert df[df["frame"] == 1]["timestamp_ms"].iloc[0] == pytest.approx(20.0)
