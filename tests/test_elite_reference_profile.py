from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from javelin_tracker.biomechanics.database.elite_reference import compute_elite_reference_profile, get_reference_value


def _make_pose_payload(*, fps: float, speed_scale: float) -> dict[str, object]:
    n_frames = 30
    coords = np.zeros((n_frames, 33, 3), dtype=float)

    # MediaPipe indices (fallback values used by elite_reference when mediapipe isn't installed).
    LS, RS = 11, 12
    LE, RE = 13, 14
    LW, RW = 15, 16
    LH, RH = 23, 24

    for i in range(n_frames):
        t = i * 0.01

        # Hips (slower).
        coords[i, LH, :] = (0.20 + t * 0.5, 0.60, 0.0)
        coords[i, RH, :] = (0.30 + t * 0.5, 0.60, 0.0)

        # Shoulders.
        coords[i, LS, :] = (0.18 + t * 0.2, 0.25, 0.0)
        coords[i, RS, :] = (0.32 + t * 0.2, 0.25, 0.0)

        # Left arm (non-throwing, slower).
        coords[i, LE, :] = (0.16 + t * 0.2, 0.30, 0.0)
        coords[i, LW, :] = (0.14 + t * 0.2, 0.35, 0.0)

        # Right arm (throwing, faster; speed_scale varies per throw).
        coords[i, RE, :] = (0.34 + t * 0.8 * speed_scale, 0.30, 0.0)
        coords[i, RW, :] = (0.36 + t * 1.2 * speed_scale, 0.35, 0.0)

    frames = []
    for i in range(n_frames):
        landmarks = [(float(coords[i, j, 0]), float(coords[i, j, 1]), float(coords[i, j, 2]), 1.0) for j in range(33)]
        frames.append(
            {
                "frame_idx": i,
                "timestamp_ms": i * (1000.0 / fps),
                "landmarks": landmarks,
                "valid": True,
            }
        )

    return {"video_metadata": {"fps": fps}, "frames": frames}


def test_compute_elite_reference_profile_writes_phase_metrics(tmp_path: Path) -> None:
    poses_dir = tmp_path / "poses"
    poses_dir.mkdir(parents=True)

    (poses_dir / "Throw_1.json").write_text(
        json.dumps(_make_pose_payload(fps=50.0, speed_scale=1.0)),
        encoding="utf-8",
    )
    (poses_dir / "Throw_2.json").write_text(
        json.dumps(_make_pose_payload(fps=50.0, speed_scale=2.0)),
        encoding="utf-8",
    )
    # Should be skipped gracefully.
    (poses_dir / "Bad.json").write_text(json.dumps({"frames": []}), encoding="utf-8")

    out_path = tmp_path / "reference_profile.json"
    reference = compute_elite_reference_profile(poses_dir, output_path=out_path)
    assert out_path.exists()
    assert reference, "Expected non-empty reference profile"

    # Spot-check phase-grouped metrics.
    delivery_key = "delivery.throwing_wrist_speed_max"
    assert delivery_key in reference
    assert reference[delivery_key]["phase"] == "delivery"
    assert reference[delivery_key]["unit"] == "rel_units/s"
    assert reference[delivery_key]["n_samples"] == 2
    assert reference[delivery_key]["std"] > 0

    release_key = "release.throwing_wrist_speed_at_release"
    assert release_key in reference
    assert reference[release_key]["phase"] == "release"

    # Query helper should find metrics by (metric_name, phase).
    found = get_reference_value("throwing_wrist_speed_max", "delivery", reference_path=out_path)
    assert found is not None
    assert found["mean"] > 0
    assert found["std"] >= 0


def _write_single_pose(tmp_path: Path, payload: dict[str, object]) -> Path:
    poses_dir = tmp_path / "poses"
    poses_dir.mkdir(parents=True, exist_ok=True)
    path = poses_dir / "Throw_1.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return poses_dir


def _payload_with_keypoints(
    *,
    fps: float,
    hips: tuple[tuple[float, float, float], tuple[float, float, float]],
    shoulders: tuple[tuple[float, float, float], tuple[float, float, float]],
    n_frames: int = 10,
) -> dict[str, object]:
    LH, RH = 23, 24
    LS, RS = 11, 12
    frames = []
    for i in range(n_frames):
        landmarks = [(0.0, 0.0, 0.0, 1.0) for _ in range(33)]
        landmarks[LH] = (*hips[0], 1.0)
        landmarks[RH] = (*hips[1], 1.0)
        landmarks[LS] = (*shoulders[0], 1.0)
        landmarks[RS] = (*shoulders[1], 1.0)
        frames.append({"frame_idx": i, "timestamp_ms": i * (1000.0 / fps), "landmarks": landmarks, "valid": True})
    return {"video_metadata": {"fps": fps}, "frames": frames}


def test_elite_reference_trunk_lean_uses_3d(tmp_path: Path) -> None:
    # Trunk axis has a Z component. A 2D-only trunk-lean computation would report 0°,
    # but the 3D metric should reflect the out-of-plane lean (~45°).
    payload = _payload_with_keypoints(
        fps=50.0,
        hips=((-0.1, 0.0, 0.0), (0.1, 0.0, 0.0)),
        shoulders=((-0.1, -1.0, 1.0), (0.1, -1.0, 1.0)),
    )
    poses_dir = _write_single_pose(tmp_path, payload)
    reference = compute_elite_reference_profile(poses_dir, output_path=tmp_path / "ref.json")
    lean = float(reference["approach.trunk_lean_deg_mean"]["mean"])
    assert lean == pytest.approx(45.0, abs=0.5)


def test_elite_reference_separation_is_trunk_axis_twist(tmp_path: Path) -> None:
    # Hip line is along +X. Shoulder line is rotated 30° around the (vertical) trunk axis
    # into the Z dimension. In the image X/Y plane both lines look identical (0°), but
    # the trunk-axis twist should still measure ~30°.
    deg = 30.0
    rad = np.deg2rad(deg)
    c = float(np.cos(rad))
    s = float(np.sin(rad))
    payload = _payload_with_keypoints(
        fps=50.0,
        hips=((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        shoulders=((-c, -1.0, -s), (c, -1.0, s)),
    )
    poses_dir = _write_single_pose(tmp_path, payload)
    reference = compute_elite_reference_profile(poses_dir, output_path=tmp_path / "ref.json")
    sep = float(reference["approach.shoulder_hip_separation_deg_mean"]["mean"])
    assert sep == pytest.approx(deg, abs=0.75)
