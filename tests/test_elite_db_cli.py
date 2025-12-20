from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

import javelin_tracker.cli as cli
from javelin_tracker.cli import app
from javelin_tracker.biomechanics.elite_database.init_elite_database import REQUIRED_COLUMNS


def _make_pose_payload(*, fps: float, speed_scale: float, video_id: str) -> dict[str, object]:
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

    return {
        "video_id": video_id,
        "video_metadata": {"fps": fps},
        "frames": frames,
        "processing_stats": {"total_frames": n_frames, "valid_frames": n_frames, "avg_confidence": 1.0},
        "warnings": [],
    }


class _StubPipeline:
    def process_video(self, video_path: str | Path, video_id: str, output_dir: str | Path, **_kwargs):
        out_root = Path(output_dir) / video_id
        out_root.mkdir(parents=True, exist_ok=True)
        out_path = out_root / "pose_data.json"
        payload = _make_pose_payload(fps=50.0, speed_scale=1.0, video_id=video_id)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return {"status": "success", "output_path": str(out_path), "warnings": []}


def _stub_validate_pose_quality(_frames, _video_metadata):
    return {"quality_score": 95.0, "is_valid": True, "issues": []}


def _write_metadata(csv_path: Path, rows: list[dict[str, str]]) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def test_elite_db_cli_commands(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()

    base_dir = tmp_path / "elite_db"
    videos_dir = base_dir / "videos"
    poses_dir = base_dir / "poses"
    videos_dir.mkdir(parents=True)
    poses_dir.mkdir(parents=True)

    video_a = videos_dir / "Alice_Thrower_1.mp4"
    video_b = videos_dir / "Bob_Thrower_1.mp4"
    video_a.write_bytes(b"")
    video_b.write_bytes(b"")

    metadata_csv = base_dir / "elite_metadata.csv"
    _write_metadata(
        metadata_csv,
        [
            {
                "thrower_name": "Alice Thrower",
                "throw_number": "1",
                "distance_m": "90.0",
                "throwing_style": "Finnish",
                "video_path": "videos/Alice_Thrower_1.mp4",
                "fps": "",
                "date_recorded": "2024-01-01",
                "source_url": "https://example.com/alice",
                "notes": "",
                "processed_status": "pending",
            },
            {
                "thrower_name": "Bob Thrower",
                "throw_number": "1",
                "distance_m": "88.0",
                "throwing_style": "Finnish",
                "video_path": "videos/Bob_Thrower_1.mp4",
                "fps": "",
                "date_recorded": "2024-01-02",
                "source_url": "https://example.com/bob",
                "notes": "",
                "processed_status": "pending",
            },
        ],
    )

    log_path = base_dir / "elite_database_qc.log"
    exclusions_path = base_dir / "exclusions.json"

    monkeypatch.setattr(cli, "_elite_db_get_pose_pipeline", lambda: _StubPipeline())
    monkeypatch.setattr(cli, "_elite_db_get_validate_pose_quality", lambda: _stub_validate_pose_quality)

    res_a = runner.invoke(
        app,
        [
            "elite-db",
            "process",
            str(video_a),
            "--poses-dir",
            str(poses_dir),
            "--metadata-csv",
            str(metadata_csv),
            "--log-path",
            str(log_path),
            "--register-metadata",
        ],
    )
    assert res_a.exit_code == 0, res_a.stdout
    assert (poses_dir / "Alice_Thrower_1.json").exists()

    res_b = runner.invoke(
        app,
        [
            "elite-db",
            "process",
            str(video_b),
            "--poses-dir",
            str(poses_dir),
            "--metadata-csv",
            str(metadata_csv),
            "--log-path",
            str(log_path),
            "--register-metadata",
        ],
    )
    assert res_b.exit_code == 0, res_b.stdout
    assert (poses_dir / "Bob_Thrower_1.json").exists()

    rows = list(csv.DictReader(metadata_csv.open("r", encoding="utf-8")))
    by_id = {f"{row['thrower_name'].replace(' ', '_')}_{row['throw_number']}": row for row in rows}
    assert by_id["Alice_Thrower_1"]["processed_status"] == "complete"
    assert by_id["Alice_Thrower_1"]["quality_score"] == "95.0"
    assert by_id["Alice_Thrower_1"]["fps"] == "50.0"

    styles = runner.invoke(
        app,
        [
            "elite-db",
            "list-styles",
            "--metadata-csv",
            str(metadata_csv),
            "--poses-dir",
            str(poses_dir),
            "--exclusions-path",
            str(exclusions_path),
        ],
    )
    assert styles.exit_code == 0, styles.stdout
    assert "finnish: 2" in styles.stdout.lower()

    recompute = runner.invoke(
        app,
        [
            "elite-db",
            "recompute",
            "--poses-dir",
            str(poses_dir),
            "--metadata-csv",
            str(metadata_csv),
            "--output-dir",
            str(base_dir),
            "--exclusions-path",
            str(exclusions_path),
            "--log-path",
            str(log_path),
        ],
    )
    assert recompute.exit_code == 0, recompute.stdout
    assert (base_dir / "reference_profile_overall.json").exists()
    assert (base_dir / "reference_profile_finnish.json").exists()

    validate = runner.invoke(
        app,
        [
            "elite-db",
            "validate",
            "--reference-path",
            str(base_dir / "reference_profile_overall.json"),
            "--poses-dir",
            str(poses_dir),
            "--metadata-csv",
            str(metadata_csv),
            "--exclusions-path",
            str(exclusions_path),
            "--log-path",
            str(log_path),
        ],
    )
    assert validate.exit_code == 0, validate.stdout
    report = json.loads(validate.stdout)
    assert "outliers" in report
    assert "low_confidence_metrics" in report

    exported = base_dir / "exported_profile.json"
    export = runner.invoke(
        app,
        [
            "elite-db",
            "export",
            "--reference-path",
            str(base_dir / "reference_profile_overall.json"),
            str(exported),
            "--log-path",
            str(log_path),
        ],
    )
    assert export.exit_code == 0, export.stdout
    assert exported.exists()

    exclude = runner.invoke(
        app,
        [
            "elite-db",
            "exclude",
            "Alice_Thrower_1",
            "test reason",
            "--exclusions-path",
            str(exclusions_path),
            "--log-path",
            str(log_path),
        ],
        input="y\n",
    )
    assert exclude.exit_code == 0, exclude.stdout
    exclusions = json.loads(exclusions_path.read_text(encoding="utf-8"))
    assert "Alice_Thrower_1" in exclusions.get("excluded", {})

