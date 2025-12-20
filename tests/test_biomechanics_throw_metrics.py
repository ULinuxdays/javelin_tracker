from __future__ import annotations

import math

import numpy as np
import pytest

from javelin_tracker.biomechanics.metrics.throw_metrics import compute_throw_metrics


def _make_synthetic_throw(*, n_frames: int, fps: float) -> list[dict[str, object]]:
    dt_ms = 1000.0 / fps

    # MediaPipe indices (fallback values used across lightweight modules).
    LS, RS = 11, 12
    LE, RE = 13, 14
    LW, RW = 15, 16
    LH, RH = 23, 24
    LK, RK = 25, 26
    LA, RA = 27, 28

    hip_x = 0.0
    shoulder_x = 0.0
    wrist_x = 0.0
    frames: list[dict[str, object]] = []

    for i in range(n_frames):
        # Build simple piecewise motion patterns to create distinct velocity peaks.
        hip_dx = 0.001 if i < 11 else (0.02 if i < 13 else 0.001)
        shoulder_dx = 0.001 if i < 13 else (0.02 if i < 15 else 0.001)
        wrist_dx = 0.001 if i < 15 else (0.03 if i < 17 else 0.001)

        hip_x += hip_dx
        shoulder_x += shoulder_dx
        wrist_x += wrist_dx

        landmarks = [(0.0, 0.0, 0.0, 1.0) for _ in range(33)]
        # Hips (symmetric).
        landmarks[LH] = (hip_x - 0.10, 0.50, 0.0, 1.0)
        landmarks[RH] = (hip_x + 0.10, 0.50, 0.0, 1.0)
        # Shoulders (symmetric).
        landmarks[LS] = (shoulder_x - 0.10, 0.30, 0.0, 1.0)
        landmarks[RS] = (shoulder_x + 0.10, 0.30, 0.0, 1.0)
        # Arms (right is throwing).
        landmarks[RE] = (shoulder_x + 0.20, 0.30, 0.0, 1.0)
        landmarks[RW] = (wrist_x + 0.20, 0.40, 0.0, 1.0)
        landmarks[LE] = (shoulder_x - 0.20, 0.30, 0.0, 1.0)
        landmarks[LW] = (wrist_x - 0.20, 0.40, 0.0, 1.0)
        # Legs (straight, symmetric).
        landmarks[LK] = (hip_x - 0.10, 0.70, 0.0, 1.0)
        landmarks[RK] = (hip_x + 0.10, 0.70, 0.0, 1.0)
        landmarks[LA] = (hip_x - 0.10, 0.90, 0.0, 1.0)
        landmarks[RA] = (hip_x + 0.10, 0.90, 0.0, 1.0)

        frames.append(
            {
                "frame_idx": i,
                "timestamp_ms": i * dt_ms,
                "landmarks": landmarks,
                "valid": True,
            }
        )

    return frames


def test_compute_throw_metrics_phase_release_power_chain_symmetry_com() -> None:
    fps = 100.0
    n_frames = 30
    frames = _make_synthetic_throw(n_frames=n_frames, fps=fps)
    phase_bounds = {"approach_start_frame": 0, "delivery_start_frame": 10, "release_frame": 20}

    metrics = compute_throw_metrics({"video_id": "Test Throw 1", "frames": frames}, phase_bounds, fps)
    assert metrics["throw_id"] == "Test_Throw_1"

    phases = metrics["phase_durations"]
    assert phases["approach_ms"] == pytest.approx(100.0)
    assert phases["delivery_ms"] == pytest.approx(100.0)
    assert phases["release_ms"] == pytest.approx(100.0)
    assert phases["total_ms"] == pytest.approx(300.0)

    release = metrics["release_metrics"]
    assert release["valid"] is True
    assert release["throwing_side"] == "right"
    assert math.isfinite(float(release["wrist_speed"]))
    assert float(release["hand_height"]) == pytest.approx(-0.40)
    assert float(release["elbow_height"]) == pytest.approx(-0.30)
    assert math.isfinite(float(release["shoulder_angle_deg"]))
    assert float(release["hip_rotation_deg"]) == pytest.approx(0.0)

    power_chain = metrics["power_chain"]
    assert power_chain["within_expected_range"] is True
    assert 20.0 <= float(power_chain["hip_to_shoulder_ms"]) <= 50.0
    assert 20.0 <= float(power_chain["shoulder_to_wrist_ms"]) <= 50.0
    assert metrics["power_chain_lag_ms"] == pytest.approx(float(power_chain["hip_to_wrist_ms"]))

    symmetry = metrics["symmetry"]
    assert symmetry["shoulder_height_score"] == pytest.approx(1.0)
    assert symmetry["knee_height_score"] == pytest.approx(1.0)
    assert symmetry["knee_angle_score"] == pytest.approx(1.0)

    assert float(metrics["com_displacement_m"]) > 0.0

    flags = metrics["confidence_flags"]
    assert flags["release_metrics_valid"] is True
    assert flags["power_chain_valid"] is True
    assert flags["symmetry_valid"] is True
    assert flags["com_displacement_valid"] is True


def test_release_low_confidence_flags_release_metrics_invalid() -> None:
    fps = 100.0
    n_frames = 30
    frames = _make_synthetic_throw(n_frames=n_frames, fps=fps)
    phase_bounds = {"approach_start_frame": 0, "delivery_start_frame": 10, "release_frame": 20}

    # Drop throwing wrist confidence at release.
    RW = 16
    release_idx = 20
    lm = list(frames[release_idx]["landmarks"])  # type: ignore[index]
    x, y, z, _ = lm[RW]
    lm[RW] = (x, y, z, 0.1)
    frames[release_idx]["landmarks"] = lm  # type: ignore[index]

    metrics = compute_throw_metrics({"video_id": "Test Throw 2", "frames": frames}, phase_bounds, fps)
    release = metrics["release_metrics"]
    assert release["valid"] is False
    assert math.isnan(float(release["wrist_speed"]))
    assert math.isnan(float(release["hand_height"]))

