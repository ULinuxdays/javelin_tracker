from __future__ import annotations

import csv
from pathlib import Path

import pytest

from javelin_tracker.biomechanics.elite_database.init_elite_database import (
    REQUIRED_COLUMNS,
    init_elite_database,
    validate_elite_metadata,
)


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert list(reader.fieldnames or []) == REQUIRED_COLUMNS
        return list(reader)


def _write_rows(csv_path: Path, rows: list[dict[str, str]]) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def test_init_creates_structure_and_metadata(tmp_path: Path) -> None:
    csv_path = init_elite_database(root_dir=tmp_path)
    assert csv_path.exists()

    base = tmp_path / "data" / "biomechanics" / "elite_database"
    assert (base / "videos").is_dir()
    assert (base / "poses").is_dir()

    rows = _read_rows(csv_path)
    assert 5 <= len(rows) <= 10
    for row in rows:
        video_rel = row["video_path"]
        assert video_rel.startswith("videos/")
        assert not Path(video_rel).is_absolute()
        assert (base / video_rel).exists()


def test_init_is_idempotent_without_overwrite(tmp_path: Path) -> None:
    csv_path = init_elite_database(root_dir=tmp_path)
    rows = _read_rows(csv_path)
    rows[0]["notes"] = "custom coach note"
    _write_rows(csv_path, rows)

    original = csv_path.read_text(encoding="utf-8")
    csv_path_2 = init_elite_database(root_dir=tmp_path, overwrite=False)
    assert csv_path_2 == csv_path
    assert csv_path.read_text(encoding="utf-8") == original


def test_init_overwrite_restores_defaults(tmp_path: Path) -> None:
    csv_path = init_elite_database(root_dir=tmp_path)
    rows = _read_rows(csv_path)
    rows[0]["notes"] = "custom coach note"
    _write_rows(csv_path, rows)

    init_elite_database(root_dir=tmp_path, overwrite=True)
    rows_after = _read_rows(csv_path)
    assert rows_after[0]["notes"] != "custom coach note"


def test_validation_flags_missing_video_file(tmp_path: Path) -> None:
    csv_path = init_elite_database(root_dir=tmp_path)
    base = tmp_path / "data" / "biomechanics" / "elite_database"
    first_video = base / _read_rows(csv_path)[0]["video_path"]
    first_video.unlink()

    report = validate_elite_metadata(csv_path, create_placeholders=False)
    assert report.has_errors
    assert report.missing_files

    with pytest.raises(ValueError):
        init_elite_database(root_dir=tmp_path, overwrite=False, validate=True, create_placeholders=False)

