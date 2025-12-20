#!/usr/bin/env python3
"""Process elite biomechanics videos through PosePipeline.

This script reads `data/biomechanics/elite_database/elite_metadata.csv`, processes any
rows that are not yet marked `processed_status=complete`, writes pose JSONs to
`data/biomechanics/elite_database/poses/{thrower_name}_{throw_number}.json`, and
updates the metadata CSV with `processed_status` and a numeric `quality_score`.

Quality/QC issues are appended to `data/biomechanics/elite_database/elite_database_qc.log`.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, MutableMapping, Optional, Sequence

from rich.progress import Progress

from javelin_tracker.biomechanics.elite_database.init_elite_database import REQUIRED_COLUMNS


DEFAULT_QUALITY_COLUMN = "quality_score"


@dataclass(frozen=True)
class RunSummary:
    processed: int
    flagged: int
    errors: int
    total: int

    def format(self) -> str:
        return f"Processed {self.processed}/{self.total} videos, {self.flagged} flagged for review."


def _slug(value: str) -> str:
    text = (value or "").strip().replace(" ", "_")
    text = re.sub(r"[^A-Za-z0-9_]+", "_", text)
    return text.strip("_") or "unknown"


def _video_id(row: Mapping[str, str]) -> str:
    thrower = _slug(row.get("thrower_name", ""))
    throw_number = _slug(row.get("throw_number", ""))
    return f"{thrower}_{throw_number}".strip("_")


def _pose_output_path(row: Mapping[str, str], poses_dir: Path) -> Path:
    return poses_dir / f"{_video_id(row)}.json"


def _resolve_video_path(row: Mapping[str, str], base_dir: Path) -> Path:
    raw = (row.get("video_path") or "").strip()
    path = Path(raw) if raw else Path()
    if path.is_absolute():
        return path
    return base_dir / path


def load_metadata(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    return fieldnames, rows


def _stable_fieldnames(existing: list[str], rows: list[dict[str, str]]) -> list[str]:
    fieldnames: list[str] = []
    for col in REQUIRED_COLUMNS:
        fieldnames.append(col)

    wants_quality = DEFAULT_QUALITY_COLUMN in existing or any(DEFAULT_QUALITY_COLUMN in row for row in rows)
    if wants_quality:
        fieldnames.append(DEFAULT_QUALITY_COLUMN)

    for col in existing:
        if col not in fieldnames:
            fieldnames.append(col)

    all_keys: set[str] = set()
    for row in rows:
        all_keys.update(row.keys())
    for col in sorted(all_keys):
        if col not in fieldnames:
            fieldnames.append(col)
    return fieldnames


def save_metadata(csv_path: Path, existing_fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    fieldnames = _stable_fieldnames(existing_fieldnames, rows)
    for row in rows:
        for col in fieldnames:
            row.setdefault(col, "")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _configure_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("elite_db")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def _default_pipeline_factory():
    from javelin_tracker.biomechanics.pose_estimation.pipeline import PosePipeline

    return PosePipeline()


def _default_validate_fn(landmarks_data: list[dict[str, object]], video_metadata: dict[str, object]) -> dict[str, object]:
    from javelin_tracker.biomechanics.utils.validation import validate_pose_quality

    return validate_pose_quality(landmarks_data, video_metadata)


def _coerce_quality_score(value: object) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return ""


def _log_quality_issues(log: logging.Logger, video_id: str, quality: Mapping[str, object]) -> None:
    issues = quality.get("issues")
    if isinstance(issues, list):
        for issue in issues:
            if issue:
                log.warning("QC issue [%s] %s", video_id, issue)


def process_row(
    row: MutableMapping[str, str],
    *,
    base_dir: Path,
    force: bool,
    dry_run: bool,
    log: logging.Logger,
    pipeline_factory: Callable[[], object],
    validate_fn: Callable[[list[dict[str, object]], dict[str, object]], dict[str, object]],
) -> str:
    poses_dir = base_dir / "poses"
    poses_dir.mkdir(parents=True, exist_ok=True)

    video_id = _video_id(row)
    video_path = _resolve_video_path(row, base_dir)
    pose_output = _pose_output_path(row, poses_dir)

    status = (row.get("processed_status") or "").strip().lower()
    if status == "complete" and not force and pose_output.exists():
        return "skipped"

    if not video_path.exists():
        log.error("Missing video [%s]: %s", video_id, video_path)
        row["processed_status"] = "error"
        row[DEFAULT_QUALITY_COLUMN] = ""
        return "error"

    if dry_run:
        log.info("DRY RUN: would process [%s] %s", video_id, video_path)
        return "dry-run"

    try:
        pipeline = pipeline_factory()
        result = pipeline.process_video(video_path, video_id=video_id, output_dir=str(poses_dir))  # type: ignore[attr-defined]
    except Exception as exc:
        log.exception("Processing exception [%s] (%s): %s", video_id, video_path, exc)
        row["processed_status"] = "error"
        row[DEFAULT_QUALITY_COLUMN] = ""
        return "error"

    if result.get("status") != "success":
        row["processed_status"] = "error"
        row[DEFAULT_QUALITY_COLUMN] = ""
        log.error("Processing failed [%s] (%s): %s", video_id, video_path, result.get("error"))
        return "error"

    try:
        payload = json.loads(Path(result["output_path"]).read_text(encoding="utf-8"))
    except Exception as exc:
        row["processed_status"] = "error"
        row[DEFAULT_QUALITY_COLUMN] = ""
        log.exception("Failed to read pipeline output [%s]: %s", video_id, exc)
        return "error"

    try:
        quality = validate_fn(payload.get("frames", []), payload.get("video_metadata", {}) or {})
    except Exception as exc:
        row["processed_status"] = "error"
        row[DEFAULT_QUALITY_COLUMN] = ""
        log.exception("Quality validation failed [%s]: %s", video_id, exc)
        return "error"

    _log_quality_issues(log, video_id, quality)
    payload["quality"] = quality
    pose_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    row["processed_status"] = "complete"
    row[DEFAULT_QUALITY_COLUMN] = _coerce_quality_score(quality.get("quality_score"))
    is_valid = bool(quality.get("is_valid"))
    if not is_valid:
        log.warning("Flagged for review [%s] (quality_score=%s)", video_id, row[DEFAULT_QUALITY_COLUMN])
        return "flagged"

    log.info("Processed [%s] (quality_score=%s)", video_id, row[DEFAULT_QUALITY_COLUMN])
    return "processed"


def _needs_processing(row: Mapping[str, str], poses_dir: Path, force: bool) -> bool:
    if force:
        return True
    status = (row.get("processed_status") or "").strip().lower()
    if status != "complete":
        return True
    return not _pose_output_path(row, poses_dir).exists()


def process_elite_database(
    *,
    base_dir: Path,
    csv_path: Path,
    log_path: Path,
    force_reprocess: bool = False,
    filter_name: Optional[str] = None,
    start_from: int = 0,
    dry_run: bool = False,
    pipeline_factory: Optional[Callable[[], object]] = None,
    validate_fn: Optional[Callable[[list[dict[str, object]], dict[str, object]], dict[str, object]]] = None,
    show_progress: bool = True,
) -> RunSummary:
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")

    existing_fieldnames, rows = load_metadata(csv_path)
    if not all(col in existing_fieldnames for col in REQUIRED_COLUMNS):
        raise ValueError(
            f"Invalid elite metadata CSV header; missing required columns. "
            f"Required={REQUIRED_COLUMNS} Found={existing_fieldnames}"
        )

    if DEFAULT_QUALITY_COLUMN not in existing_fieldnames:
        existing_fieldnames = existing_fieldnames + [DEFAULT_QUALITY_COLUMN]

    logger = _configure_logger(log_path)
    poses_dir = base_dir / "poses"
    poses_dir.mkdir(parents=True, exist_ok=True)

    pipeline_factory = pipeline_factory or _default_pipeline_factory
    validate_fn = validate_fn or _default_validate_fn

    selected_rows: list[MutableMapping[str, str]] = []
    for idx, row in enumerate(rows):
        if idx < start_from:
            continue
        if filter_name and filter_name.lower() not in (row.get("thrower_name") or "").lower():
            continue
        if _needs_processing(row, poses_dir, force_reprocess):
            selected_rows.append(row)

    processed = 0
    flagged = 0
    errors = 0
    total = len(selected_rows)

    def _iter_rows():
        if not show_progress:
            for row in selected_rows:
                yield row
            return
        with Progress() as progress:
            task = progress.add_task("Processing elite videos", total=total)
            for row in selected_rows:
                yield row
                progress.update(task, advance=1)

    for row in _iter_rows():
        result = process_row(
            row,
            base_dir=base_dir,
            force=force_reprocess,
            dry_run=dry_run,
            log=logger,
            pipeline_factory=pipeline_factory,
            validate_fn=validate_fn,
        )
        if result == "processed":
            processed += 1
        elif result == "flagged":
            flagged += 1
        elif result == "error":
            errors += 1

    if not dry_run:
        save_metadata(csv_path, existing_fieldnames, rows)

    summary = RunSummary(processed=processed, flagged=flagged, errors=errors, total=total)
    logger.info("Run summary: %s (errors=%s)", summary.format(), errors)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Process elite database videos through PosePipeline.")
    parser.add_argument("--force-reprocess", action="store_true", help="Reprocess even if marked complete.")
    parser.add_argument("--filter-name", help="Process only throwers matching this substring.")
    parser.add_argument("--start-from", type=int, default=0, help="Skip first N rows.")
    parser.add_argument("--dry-run", action="store_true", help="Do not run pipeline; just list actions.")
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[1] / "data" / "biomechanics" / "elite_database"
    csv_path = base / "elite_metadata.csv"
    log_path = base / "elite_database_qc.log"

    summary = process_elite_database(
        base_dir=base,
        csv_path=csv_path,
        log_path=log_path,
        force_reprocess=args.force_reprocess,
        filter_name=args.filter_name,
        start_from=args.start_from,
        dry_run=args.dry_run,
        show_progress=True,
    )
    print(summary.format())


if __name__ == "__main__":
    main()
