from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from javelin_tracker.biomechanics.database.elite_reference import compute_style_profiles, resolve_style_profile
from javelin_tracker.biomechanics.elite_database.init_elite_database import REQUIRED_COLUMNS


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
        coords[i, LH, :] = (0.20 + t * 0.5, 0.60, 0.0)
        coords[i, RH, :] = (0.30 + t * 0.5, 0.60, 0.0)
        coords[i, LS, :] = (0.18 + t * 0.2, 0.25, 0.0)
        coords[i, RS, :] = (0.32 + t * 0.2, 0.25, 0.0)
        coords[i, LE, :] = (0.16 + t * 0.2, 0.30, 0.0)
        coords[i, LW, :] = (0.14 + t * 0.2, 0.35, 0.0)
        coords[i, RE, :] = (0.34 + t * 0.8 * speed_scale, 0.30, 0.0)
        coords[i, RW, :] = (0.36 + t * 1.2 * speed_scale, 0.35, 0.0)

    frames = []
    for i in range(n_frames):
        landmarks = [(float(coords[i, j, 0]), float(coords[i, j, 1]), float(coords[i, j, 2]), 1.0) for j in range(33)]
        frames.append({"frame_idx": i, "timestamp_ms": i * (1000.0 / fps), "landmarks": landmarks, "valid": True})

    return {"video_metadata": {"fps": fps}, "frames": frames}


def _write_metadata(csv_path: Path, rows: list[dict[str, str]]) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def test_compute_style_profiles_writes_files_and_falls_back(tmp_path: Path) -> None:
    poses_dir = tmp_path / "poses"
    poses_dir.mkdir()

    (poses_dir / "Alice_Thrower_1.json").write_text(
        json.dumps(_make_pose_payload(fps=50.0, speed_scale=1.0)), encoding="utf-8"
    )
    (poses_dir / "Bob_Thrower_1.json").write_text(
        json.dumps(_make_pose_payload(fps=50.0, speed_scale=2.0)), encoding="utf-8"
    )
    (poses_dir / "Carol_Thrower_1.json").write_text(
        json.dumps(_make_pose_payload(fps=50.0, speed_scale=1.5)), encoding="utf-8"
    )
    (poses_dir / "Dan_Thrower_1.json").write_text(
        json.dumps(_make_pose_payload(fps=50.0, speed_scale=1.1)), encoding="utf-8"
    )

    metadata_csv = tmp_path / "elite_metadata.csv"
    _write_metadata(
        metadata_csv,
        [
            {
                "thrower_name": "Alice Thrower",
                "throw_number": "1",
                "distance_m": "90.0",
                "throwing_style": "Finnish",
                "video_path": "videos/Alice_Thrower_1.mp4",
                "fps": "50",
                "date_recorded": "2024-01-01",
                "source_url": "https://example.com/alice",
                "notes": "",
                "processed_status": "complete",
            },
            {
                "thrower_name": "Bob Thrower",
                "throw_number": "1",
                "distance_m": "88.0",
                "throwing_style": "finnish",
                "video_path": "videos/Bob_Thrower_1.mp4",
                "fps": "50",
                "date_recorded": "2024-01-02",
                "source_url": "https://example.com/bob",
                "notes": "",
                "processed_status": "complete",
            },
            {
                "thrower_name": "Carol Thrower",
                "throw_number": "1",
                "distance_m": "86.0",
                "throwing_style": "Czech",
                "video_path": "videos/Carol_Thrower_1.mp4",
                "fps": "50",
                "date_recorded": "2024-01-03",
                "source_url": "https://example.com/carol",
                "notes": "",
                "processed_status": "complete",
            },
            {
                "thrower_name": "Dan Thrower",
                "throw_number": "1",
                "distance_m": "84.0",
                "throwing_style": "",
                "video_path": "videos/Dan_Thrower_1.mp4",
                "fps": "50",
                "date_recorded": "2024-01-04",
                "source_url": "https://example.com/dan",
                "notes": "",
                "processed_status": "complete",
            },
        ],
    )

    output_dir = tmp_path / "out"
    profiles = compute_style_profiles(poses_dir, metadata_csv, output_dir=output_dir)
    assert "overall" in profiles
    assert "finnish" in profiles
    assert "czech" not in profiles  # insufficient samples
    assert "unknown" not in profiles  # insufficient samples

    assert profiles["finnish"]["n_samples"] == 2
    assert profiles["finnish"]["throwers"] == ["Alice Thrower", "Bob Thrower"]

    assert profiles["overall"]["n_samples"] == 4
    assert profiles["overall"]["throwers"] == ["Alice Thrower", "Bob Thrower", "Carol Thrower", "Dan Thrower"]

    assert (output_dir / "reference_profile_finnish.json").exists()
    assert (output_dir / "reference_profile_overall.json").exists()
    assert not (output_dir / "reference_profile_czech.json").exists()

    fallback = resolve_style_profile("czech", profiles)
    assert fallback["n_samples"] == profiles["overall"]["n_samples"]

