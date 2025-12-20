from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from javelin_tracker.biomechanics.database.elite_reference import compute_elite_reference_profile
from javelin_tracker.biomechanics.database.validation import (
    exclude_throw,
    load_exclusions,
    plot_metric_distribution,
    recompute_profiles_after_exclusions,
    validate_elite_profiles,
)
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


def test_validate_elite_profiles_detects_outlier_and_supports_exclusions(tmp_path: Path) -> None:
    poses_dir = tmp_path / "poses"
    poses_dir.mkdir()

    rows: list[dict[str, str]] = []
    for throw_number in range(1, 12):
        throw_id = f"Test_Thrower_{throw_number}"
        speed_scale = 1.0 if throw_number < 11 else 5.0
        (poses_dir / f"{throw_id}.json").write_text(
            json.dumps(_make_pose_payload(fps=50.0, speed_scale=speed_scale)),
            encoding="utf-8",
        )
        rows.append(
            {
                "thrower_name": "Test Thrower",
                "throw_number": str(throw_number),
                "distance_m": "85.0",
                "throwing_style": "Finnish",
                "video_path": f"videos/{throw_id}.mp4",
                "fps": "50",
                "date_recorded": "2024-01-01",
                "source_url": f"https://example.com/{throw_id}",
                "notes": "",
                "processed_status": "complete",
            }
        )

    metadata_csv = tmp_path / "elite_metadata.csv"
    _write_metadata(metadata_csv, rows)

    ref_path = tmp_path / "reference_profile_overall.json"
    compute_elite_reference_profile(poses_dir, output_path=ref_path)

    exclusions_path = tmp_path / "exclusions.json"
    log_path = tmp_path / "elite_database_qc.log"

    report = validate_elite_profiles(
        ref_path,
        None,
        poses_dir=poses_dir,
        metadata_csv=metadata_csv,
        exclusions_path=exclusions_path,
        log_path=log_path,
    )
    assert report["style_sample_counts"]["overall"] == 11
    assert report["style_sample_counts"]["finnish"] == 11

    outlier_id = "Test_Thrower_11"
    outlier_rows = [row for row in report["outliers"] if row.get("throw_id") == outlier_id]
    assert outlier_rows, "Expected the outlier throw to be flagged"
    assert any(row.get("metric") == "delivery.throwing_wrist_speed_max" for row in outlier_rows)

    exclude_throw(outlier_id, "Synthetic test exclusion", exclusions_path=exclusions_path, log_path=log_path)
    exclusions = load_exclusions(exclusions_path)
    assert outlier_id in exclusions

    report_after = validate_elite_profiles(
        ref_path,
        None,
        poses_dir=poses_dir,
        metadata_csv=metadata_csv,
        exclusions_path=exclusions_path,
        log_path=log_path,
    )
    assert report_after["style_sample_counts"]["overall"] == 10
    assert all(row.get("throw_id") != outlier_id for row in report_after["outliers"])

    output_dir = tmp_path / "out"
    profiles = recompute_profiles_after_exclusions(
        poses_dir=poses_dir,
        metadata_csv=metadata_csv,
        output_dir=output_dir,
        exclusions_path=exclusions_path,
        log_path=log_path,
    )
    assert profiles["overall"]["n_samples"] == 10
    assert profiles["finnish"]["n_samples"] == 10
    assert (output_dir / "reference_profile_overall.json").exists()
    assert (output_dir / "reference_profile_finnish.json").exists()

    overall = json.loads((output_dir / "reference_profile_overall.json").read_text(encoding="utf-8"))
    assert overall["delivery.throwing_wrist_speed_max"]["n_samples"] == 10

    fig = plot_metric_distribution(
        "delivery.throwing_wrist_speed_max",
        poses_dir=poses_dir,
        metadata_csv=metadata_csv,
        reference_path=ref_path,
        exclusions_path=exclusions_path,
    )
    assert fig is not None

