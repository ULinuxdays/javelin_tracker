"""Initialize and validate the elite biomechanics database.

Creates (if missing):
- `data/biomechanics/elite_database/videos/` – elite throw videos (placeholders in repo).
- `data/biomechanics/elite_database/poses/` – processed pose JSON outputs.
- `data/biomechanics/elite_database/elite_metadata.csv` – coach-editable metadata.

This project does not ship copyrighted videos; the tracked `.mp4` files are empty
placeholders so the directory structure and metadata stay reproducible.

Usage:
    python3 -m javelin_tracker.biomechanics.elite_database.init_elite_database --help
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence


REQUIRED_COLUMNS: List[str] = [
    "thrower_name",
    "throw_number",
    "distance_m",
    "throwing_style",
    "video_path",
    "fps",
    "date_recorded",
    "source_url",
    "notes",
    "processed_status",
]

OPTIONAL_COLUMNS: List[str] = ["quality_score"]


@dataclass(frozen=True)
class MetadataValidationReport:
    missing_files: List[str]
    placeholder_files: List[str]
    invalid_video_paths: List[str]
    header_error: str | None

    @property
    def has_errors(self) -> bool:
        return bool(self.missing_files or self.invalid_video_paths or self.header_error)

    def format(self) -> str:
        parts: List[str] = []
        if self.header_error:
            parts.append(f"CSV header error: {self.header_error}")
        if self.invalid_video_paths:
            parts.append("Invalid `video_path` values:\n  - " + "\n  - ".join(self.invalid_video_paths))
        if self.missing_files:
            parts.append("Missing video files:\n  - " + "\n  - ".join(self.missing_files))
        if self.placeholder_files:
            parts.append(
                "Placeholder (empty) video files detected:\n  - " + "\n  - ".join(self.placeholder_files)
            )
        return "\n".join(parts) if parts else "OK"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _elite_database_dir(root_dir: Path | None = None) -> Path:
    root = Path(root_dir) if root_dir is not None else _project_root()
    return root / "data" / "biomechanics" / "elite_database"


def _metadata_path(root_dir: Path | None = None) -> Path:
    return _elite_database_dir(root_dir) / "elite_metadata.csv"


def _default_records() -> List[Dict[str, object]]:
    return [
        {
            "thrower_name": "Johannes Vetter",
            "throw_number": 1,
            "distance_m": 97.76,
            "throwing_style": "full",
            "video_path": "videos/Johannes_Vetter_1.mp4",
            "fps": 50,
            "date_recorded": "2020-09-06",
            "source_url": "https://www.youtube.com/watch?v=LWwKZLDwPvE",
            "notes": "97.76m Chorzow",
            "processed_status": "pending",
        },
        {
            "thrower_name": "Neeraj Chopra",
            "throw_number": 1,
            "distance_m": 88.07,
            "throwing_style": "full",
            "video_path": "videos/Neeraj_Chopra_1.mp4",
            "fps": 50,
            "date_recorded": "2021-08-07",
            "source_url": "https://www.youtube.com/watch?v=8pXipPyTTKM",
            "notes": "Tokyo gold",
            "processed_status": "pending",
        },
        {
            "thrower_name": "Thomas Rohler",
            "throw_number": 1,
            "distance_m": 93.90,
            "throwing_style": "full",
            "video_path": "videos/Thomas_Rohler_1.mp4",
            "fps": 50,
            "date_recorded": "2017-05-05",
            "source_url": "https://www.youtube.com/watch?v=fvPDDQ3MeMc",
            "notes": "Doha DL 93.90m",
            "processed_status": "pending",
        },
        {
            "thrower_name": "Anderson Peters",
            "throw_number": 1,
            "distance_m": 93.07,
            "throwing_style": "full",
            "video_path": "videos/Anderson_Peters_1.mp4",
            "fps": 50,
            "date_recorded": "2022-05-13",
            "source_url": "https://www.youtube.com/watch?v=1fx9gGRA5N0",
            "notes": "Doha DL 93.07m",
            "processed_status": "pending",
        },
        {
            "thrower_name": "Keshorn Walcott",
            "throw_number": 1,
            "distance_m": 90.16,
            "throwing_style": "full",
            "video_path": "videos/Keshorn_Walcott_1.mp4",
            "fps": 50,
            "date_recorded": "2015-05-30",
            "source_url": "https://www.youtube.com/watch?v=ziSYTWcZi-M",
            "notes": "Diamond League best",
            "processed_status": "pending",
        },
        {
            "thrower_name": "Julian Weber",
            "throw_number": 1,
            "distance_m": 89.54,
            "throwing_style": "full",
            "video_path": "videos/Julian_Weber_1.mp4",
            "fps": 50,
            "date_recorded": "2022-08-21",
            "source_url": "https://www.youtube.com/watch?v=EJzY1v77YP8",
            "notes": "European champs",
            "processed_status": "pending",
        },
        {
            "thrower_name": "Jakub Vadlejch",
            "throw_number": 1,
            "distance_m": 90.88,
            "throwing_style": "full",
            "video_path": "videos/Jakub_Vadlejch_1.mp4",
            "fps": 50,
            "date_recorded": "2022-09-08",
            "source_url": "https://www.youtube.com/watch?v=wiwFcntuv5w",
            "notes": "Zagreb meeting",
            "processed_status": "pending",
        },
    ]


def _read_metadata_csv(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _write_metadata_csv(csv_path: Path, rows: List[Dict[str, object]], *, include_quality_score: bool) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = REQUIRED_COLUMNS + (OPTIONAL_COLUMNS if include_quality_score else [])
    normalised_rows: List[Dict[str, object]] = []
    for row in rows:
        record = {k: row.get(k, "") for k in fieldnames}
        normalised_rows.append(record)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalised_rows)


def validate_elite_metadata(
    csv_path: Path,
    *,
    create_placeholders: bool = False,
) -> MetadataValidationReport:
    base_dir = csv_path.parent
    if not csv_path.exists():
        return MetadataValidationReport(
            missing_files=[],
            placeholder_files=[],
            invalid_video_paths=[],
            header_error=f"CSV not found: {csv_path}",
        )

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])

    allowed_headers = [REQUIRED_COLUMNS, REQUIRED_COLUMNS + OPTIONAL_COLUMNS]
    header_error = None
    if header not in allowed_headers:
        header_error = (
            f"Expected columns {REQUIRED_COLUMNS} (optionally + {OPTIONAL_COLUMNS}), "
            f"but found {header}."
        )

    invalid_paths: List[str] = []
    missing_files: List[str] = []
    placeholder_files: List[str] = []

    rows = _read_metadata_csv(csv_path)
    for row in rows:
        raw_path = (row.get("video_path") or "").strip()
        if not raw_path:
            invalid_paths.append("(empty video_path)")
            continue
        if "\\" in raw_path:
            invalid_paths.append(f"{raw_path} (use forward slashes: videos/Foo_1.mp4)")
            continue
        path = Path(raw_path)
        if path.is_absolute():
            invalid_paths.append(f"{raw_path} (must be relative, e.g. videos/Foo_1.mp4)")
            continue
        if path.parts[:1] != ("videos",):
            invalid_paths.append(f"{raw_path} (must live under videos/)")
            continue
        if path.suffix.lower() != ".mp4":
            invalid_paths.append(f"{raw_path} (expected .mp4)")
            continue

        abs_path = base_dir / path
        if not abs_path.exists():
            if create_placeholders:
                abs_path.parent.mkdir(parents=True, exist_ok=True)
                abs_path.touch()
            else:
                missing_files.append(str(abs_path))
                continue
        if abs_path.exists() and abs_path.stat().st_size == 0:
            placeholder_files.append(str(abs_path))

    return MetadataValidationReport(
        missing_files=missing_files,
        placeholder_files=placeholder_files,
        invalid_video_paths=invalid_paths,
        header_error=header_error,
    )


def init_elite_database(
    *,
    root_dir: Path | None = None,
    overwrite: bool = False,
    include_quality_score: bool = False,
    validate: bool = True,
    create_placeholders: bool = False,
) -> Path:
    """Create/validate elite database dirs and metadata CSV.

    If `overwrite` is False and the metadata CSV exists, it is left untouched.
    """
    base = _elite_database_dir(root_dir)
    videos_dir = base / "videos"
    poses_dir = base / "poses"
    base.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    poses_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = base / "elite_metadata.csv"
    if overwrite or not metadata_path.exists():
        rows = _default_records()
        _write_metadata_csv(metadata_path, rows, include_quality_score=include_quality_score)
        create_placeholders = True

    if validate:
        report = validate_elite_metadata(metadata_path, create_placeholders=create_placeholders)
        if report.has_errors:
            raise ValueError(report.format())
        if report.placeholder_files:
            print(report.format(), file=sys.stderr)

    return metadata_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Initialize/validate the elite biomechanics database.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite elite_metadata.csv with default sample entries.",
    )
    parser.add_argument(
        "--include-quality-score",
        action="store_true",
        help="Add an optional `quality_score` column for downstream processing scripts.",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation of metadata and referenced video files.",
    )
    parser.add_argument(
        "--create-placeholders",
        action="store_true",
        help="Create placeholder (empty) video files for any missing `video_path` entries.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        path = init_elite_database(
            overwrite=args.overwrite,
            include_quality_score=args.include_quality_score,
            validate=not args.no_validate,
            create_placeholders=args.create_placeholders,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Elite database ready at {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
