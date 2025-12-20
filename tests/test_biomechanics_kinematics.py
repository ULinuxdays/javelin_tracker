from __future__ import annotations

import math

import numpy as np
import pytest

from javelin_tracker.biomechanics.metrics.kinematics import (
    compute_acceleration,
    compute_joint_velocities,
    compute_velocity,
    compute_velocity_profiles_dataframe,
    get_velocity_peaks,
    plot_joint_velocity_profiles,
)


def test_compute_velocity_linear_motion() -> None:
    # x(t)=t, dt=0.1s => velocity ~ 1.0 everywhere.
    t = np.arange(0, 1.0, 0.1, dtype=float)
    coords = np.stack([t, np.zeros_like(t), np.zeros_like(t)], axis=1)
    speed, vel = compute_velocity(coords, t.tolist())
    assert speed.shape == (len(t),)
    assert vel.shape == (len(t), 3)
    assert np.nanmin(speed) == pytest.approx(1.0)
    assert np.nanmax(speed) == pytest.approx(1.0)


def test_compute_acceleration_quadratic_motion() -> None:
    # x(t)=t^2 => v=2t, a=2 constant.
    t = np.arange(0, 1.0, 0.1, dtype=float)
    coords = np.stack([t**2, np.zeros_like(t), np.zeros_like(t)], axis=1)
    _, vel = compute_velocity(coords, t.tolist())
    acc_mag, acc = compute_acceleration(vel, t.tolist())
    assert acc.shape == (len(t), 3)
    # Interior points should be ~2; endpoints use one-sided and are less stable.
    assert np.nanmedian(acc_mag[2:-2]) == pytest.approx(2.0, rel=1e-2)


def test_compute_joint_velocities_and_peaks_dataframe() -> None:
    fps = 50.0
    n_frames = 10
    coords = np.zeros((n_frames, 33, 4), dtype=float)
    coords[:, :, 3] = 1.0

    # Move joint 16 (right wrist) linearly in x.
    for i in range(n_frames):
        coords[i, 16, 0] = i * 0.01

    joint_names = [f"joint_{i}" for i in range(33)]
    summary = compute_joint_velocities(coords, fps, joint_names=joint_names, release_frame=n_frames - 1)
    assert "joint_16" in summary

    wrist = summary["joint_16"]
    assert len(wrist["velocities"]) == n_frames
    assert len(wrist["timestamps"]) == n_frames
    assert wrist["peak_frame"] >= 0
    assert wrist["peak_velocity"] > 0
    assert wrist["peak_relative_to_release_ms"] is not None
    assert wrist["timing_summary"] is not None

    df = compute_velocity_profiles_dataframe(coords, fps, joint_names=joint_names)
    assert not df.empty
    assert set(df.columns) == {"frame", "timestamp_ms", "joint_index", "joint", "vx", "vy", "vz", "speed", "valid"}

    peaks = get_velocity_peaks(df)
    assert peaks, "Expected at least one peak entry"
    assert peaks[0][0] == "joint_16"
    assert math.isfinite(peaks[0][1])
    assert isinstance(peaks[0][2], int)

    fig = plot_joint_velocity_profiles(df, joints=["joint_16"])
    assert fig is not None
