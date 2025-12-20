from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.process_elite_database import (
    DEFAULT_QUALITY_COLUMN,
    REQUIRED_COLUMNS,
    process_elite_database,
)


class StubPipeline:
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def process_video(self, video_path: Path, video_id: str, output_dir: str, *, max_retries: int = 2):
        self._calls.append(video_id)
        out_root = Path(output_dir) / video_id
        out_root.mkdir(parents=True, exist_ok=True)
        out_path = out_root / "pose_data.json"
        payload = {"video_id": video_id, "video_metadata": {"fps": 50}, "frames": []}
        out_path.write_text(json.dumps(payload), encoding="utf-8")
        return {"status": "success", "output_path": str(out_path)}


def _write_metadata(csv_path: Path, rows: list[dict[str, str]]) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _read_header_and_rows(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        rows = list(reader)
    return header, rows


def test_process_elite_database_updates_csv_and_writes_pose_files(tmp_path: Path) -> None:
    base = tmp_path / "data" / "biomechanics" / "elite_database"
    (base / "videos").mkdir(parents=True)
    (base / "poses").mkdir(parents=True)

    (base / "videos" / "Alice_Thrower_1.mp4").write_bytes(b"fake")
    (base / "videos" / "Bob_Thrower_1.mp4").write_bytes(b"fake")

    csv_path = base / "elite_metadata.csv"
    rows = [
        {
            "thrower_name": "Alice Thrower",
            "throw_number": "1",
            "distance_m": "90.0",
            "throwing_style": "full",
            "video_path": "videos/Alice_Thrower_1.mp4",
            "fps": "50",
            "date_recorded": "2024-01-01",
            "source_url": "https://example.com/alice",
            "notes": "",
            "processed_status": "pending",
        },
        {
            "thrower_name": "Bob Thrower",
            "throw_number": "1",
            "distance_m": "88.0",
            "throwing_style": "full",
            "video_path": "videos/Bob_Thrower_1.mp4",
            "fps": "50",
            "date_recorded": "2024-01-02",
            "source_url": "https://example.com/bob",
            "notes": "",
            "processed_status": "pending",
        },
        {
            "thrower_name": "Carol Thrower",
            "throw_number": "1",
            "distance_m": "85.0",
            "throwing_style": "full",
            "video_path": "videos/Carol_Thrower_1.mp4",
            "fps": "50",
            "date_recorded": "2024-01-03",
            "source_url": "https://example.com/carol",
            "notes": "",
            "processed_status": "pending",
        },
    ]
    _write_metadata(csv_path, rows)

    calls: list[str] = []

    def pipeline_factory() -> StubPipeline:
        return StubPipeline(calls)

    def validate_fn(_landmarks_data, _video_metadata):
        if calls and calls[-1].startswith("Bob_Thrower_"):
            return {"quality_score": 70.0, "is_valid": False, "issues": ["low confidence"]}
        return {"quality_score": 90.0, "is_valid": True, "issues": []}

    log_path = base / "elite_database_qc.log"
    summary = process_elite_database(
        base_dir=base,
        csv_path=csv_path,
        log_path=log_path,
        pipeline_factory=pipeline_factory,
        validate_fn=validate_fn,
        show_progress=False,
    )

    assert summary.format() == "Processed 1/3 videos, 1 flagged for review."
    assert calls == ["Alice_Thrower_1", "Bob_Thrower_1"]

    alice_pose = base / "poses" / "Alice_Thrower_1.json"
    bob_pose = base / "poses" / "Bob_Thrower_1.json"
    assert alice_pose.exists()
    assert bob_pose.exists()
    assert "quality" in json.loads(alice_pose.read_text(encoding="utf-8"))
    assert "quality" in json.loads(bob_pose.read_text(encoding="utf-8"))

    header, out_rows = _read_header_and_rows(csv_path)
    assert header[: len(REQUIRED_COLUMNS)] == REQUIRED_COLUMNS
    assert DEFAULT_QUALITY_COLUMN in header

    by_name = {r["thrower_name"]: r for r in out_rows}
    assert by_name["Alice Thrower"]["processed_status"] == "complete"
    assert float(by_name["Alice Thrower"][DEFAULT_QUALITY_COLUMN]) == 90.0
    assert by_name["Bob Thrower"]["processed_status"] == "complete"
    assert float(by_name["Bob Thrower"][DEFAULT_QUALITY_COLUMN]) == 70.0
    assert by_name["Carol Thrower"]["processed_status"] == "error"
    assert by_name["Carol Thrower"][DEFAULT_QUALITY_COLUMN] == ""

    log_text = log_path.read_text(encoding="utf-8")
    assert "QC issue [Bob_Thrower_1] low confidence" in log_text
    assert "Flagged for review [Bob_Thrower_1]" in log_text


def test_process_elite_database_is_idempotent_for_completed_rows(tmp_path: Path) -> None:
    base = tmp_path / "data" / "biomechanics" / "elite_database"
    (base / "videos").mkdir(parents=True)
    (base / "poses").mkdir(parents=True)

    (base / "videos" / "Alice_Thrower_1.mp4").write_bytes(b"fake")
    (base / "videos" / "Bob_Thrower_1.mp4").write_bytes(b"fake")

    csv_path = base / "elite_metadata.csv"
    _write_metadata(
        csv_path,
        [
            {
                "thrower_name": "Alice Thrower",
                "throw_number": "1",
                "distance_m": "90.0",
                "throwing_style": "full",
                "video_path": "videos/Alice_Thrower_1.mp4",
                "fps": "50",
                "date_recorded": "2024-01-01",
                "source_url": "https://example.com/alice",
                "notes": "",
                "processed_status": "pending",
            },
            {
                "thrower_name": "Bob Thrower",
                "throw_number": "1",
                "distance_m": "88.0",
                "throwing_style": "full",
                "video_path": "videos/Bob_Thrower_1.mp4",
                "fps": "50",
                "date_recorded": "2024-01-02",
                "source_url": "https://example.com/bob",
                "notes": "",
                "processed_status": "pending",
            },
        ],
    )

    calls: list[str] = []

    def pipeline_factory() -> StubPipeline:
        return StubPipeline(calls)

    def validate_fn(_landmarks_data, _video_metadata):
        return {"quality_score": 90.0, "is_valid": True, "issues": []}

    log_path = base / "elite_database_qc.log"
    process_elite_database(
        base_dir=base,
        csv_path=csv_path,
        log_path=log_path,
        pipeline_factory=pipeline_factory,
        validate_fn=validate_fn,
        show_progress=False,
    )
    assert calls == ["Alice_Thrower_1", "Bob_Thrower_1"]

    calls.clear()
    summary = process_elite_database(
        base_dir=base,
        csv_path=csv_path,
        log_path=log_path,
        pipeline_factory=pipeline_factory,
        validate_fn=validate_fn,
        show_progress=False,
    )
    assert calls == []
    assert summary.format() == "Processed 0/0 videos, 0 flagged for review."

