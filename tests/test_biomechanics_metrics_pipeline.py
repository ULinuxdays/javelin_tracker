from __future__ import annotations

import json
from pathlib import Path

import pytest

from javelin_tracker.biomechanics.metrics.pipeline import MetricsPipeline


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
        # Piecewise motion to create distinct velocity peaks.
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


def test_metrics_pipeline_writes_metrics_json(tmp_path: Path) -> None:
    fps = 100.0
    frames = _make_synthetic_throw(n_frames=30, fps=fps)
    payload = {
        "video_id": "Test Session 1",
        "video_metadata": {"fps": fps, "width": 1000, "height": 1000, "athlete_height_m": 2.0},
        "frames": frames,
    }

    pipeline = MetricsPipeline()
    result = pipeline.compute_metrics(payload, tmp_path)
    assert result["status"] == "success"

    output_path = Path(result["output_path"])
    assert output_path.exists()

    metrics = json.loads(output_path.read_text(encoding="utf-8"))
    assert metrics["session_id"] == "Test_Session_1"
    assert metrics["phase_boundaries"]["release_frame"] >= metrics["phase_boundaries"]["delivery_start_frame"]
    assert isinstance(metrics["angles"]["data"], list)
    assert isinstance(metrics["kinematics"]["data"], list)
    assert metrics["throw_metrics"]["throw_id"] == "Test_Session_1"
    assert "processing_stats" in metrics

    # Strict JSON should not contain NaN literals.
    assert "NaN" not in output_path.read_text(encoding="utf-8")


def test_metrics_pipeline_handles_empty_frames(tmp_path: Path) -> None:
    payload = {"video_id": "Empty", "video_metadata": {"fps": 30.0}, "frames": []}
    pipeline = MetricsPipeline()
    result = pipeline.compute_metrics(payload, tmp_path)
    assert result["status"] == "success"
    assert "pose_low_valid_frame_ratio" in result["flagged_unreliable"]


def test_metrics_pipeline_accepts_pose_json_path(tmp_path: Path) -> None:
    fps = 60.0
    frames = _make_synthetic_throw(n_frames=10, fps=fps)
    payload_path = tmp_path / "pose_data.json"
    payload_path.write_text(
        json.dumps({"video_metadata": {"fps": fps, "width": 1000, "height": 1000}, "frames": frames}),
        encoding="utf-8",
    )

    pipeline = MetricsPipeline()
    result = pipeline.compute_metrics(payload_path, tmp_path)
    assert result["status"] == "success"
    assert Path(result["output_path"]).exists()
    assert Path(result["output_path"]).parent.name == "pose_data"

