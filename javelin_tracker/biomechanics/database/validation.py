"""Elite database QC and validation utilities.

Phase 3 / Step 15 focuses on validating elite reference profiles:
- Flag *outlier throws* where a per-throw metric deviates >3 std from the mean.
- Flag *low-confidence metrics* with high variability.
- Persist coach-reviewed exclusions and support recomputing profiles without them.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

from javelin_tracker.biomechanics.database.elite_reference import compute_elite_reference_profile_from_files, compute_throw_metrics

ELITE_DATABASE_DIR = Path(__file__).resolve().parents[3] / "data" / "biomechanics" / "elite_database"
DEFAULT_POSES_DIR = ELITE_DATABASE_DIR / "poses"
DEFAULT_METADATA_CSV = ELITE_DATABASE_DIR / "elite_metadata.csv"
DEFAULT_REFERENCE_PATH = ELITE_DATABASE_DIR / "reference_profile_overall.json"

QC_LOG = ELITE_DATABASE_DIR / "elite_database_qc.log"
EXCLUSION_FILE = ELITE_DATABASE_DIR / "exclusions.json"

DEFAULT_OUTLIER_ZSCORE = 3.0
DEFAULT_MIN_STYLE_SAMPLES = 2


@dataclass(frozen=True)
class ProcessedThrow:
    throw_id: str
    thrower_name: str
    throwing_style: str
    pose_path: Path
    metrics: Dict[str, float]


def _configure_logger(log_path: Path) -> logging.Logger:
    log = logging.getLogger("elite_db.qc")
    log.setLevel(logging.INFO)
    log.propagate = False
    for handler in list(log.handlers):
        log.removeHandler(handler)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(handler)
    return log


def _slug(value: str) -> str:
    text = (value or "").strip().replace(" ", "_")
    text = re.sub(r"[^A-Za-z0-9_]+", "_", text)
    return text.strip("_") or "unknown"


def _throw_id_from_row(row: Mapping[str, str]) -> str:
    thrower = _slug(row.get("thrower_name", ""))
    throw_number = _slug(row.get("throw_number", ""))
    return f"{thrower}_{throw_number}".strip("_")


def _normalize_style(value: str | None) -> str:
    return (value or "").strip().lower() or "unknown"


def _style_filename_slug(style: str) -> str:
    text = (style or "").strip().lower().replace(" ", "_")
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    return text.strip("_") or "unknown"


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def load_exclusions(exclusions_path: Path = EXCLUSION_FILE) -> Dict[str, Dict[str, str]]:
    """Load excluded throws from disk.

    The on-disk format is:
      {"excluded": {"Thrower_1": {"reason": "...", "excluded_at": "..."}}}

    Older formats with string reasons are also supported.
    """
    raw = _load_json(exclusions_path)
    excluded = raw.get("excluded", {})
    normalized: Dict[str, Dict[str, str]] = {}
    if not isinstance(excluded, dict):
        return normalized
    for throw_id, entry in excluded.items():
        if not throw_id:
            continue
        if isinstance(entry, str):
            normalized[str(throw_id)] = {"reason": entry, "excluded_at": ""}
            continue
        if isinstance(entry, dict):
            reason = str(entry.get("reason") or "")
            excluded_at = str(entry.get("excluded_at") or "")
            normalized[str(throw_id)] = {"reason": reason, "excluded_at": excluded_at}
    return normalized


def _write_exclusions(exclusions_path: Path, exclusions: Mapping[str, Mapping[str, str]]) -> None:
    exclusions_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"excluded": {k: dict(v) for k, v in exclusions.items()}}
    exclusions_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def exclude_throw(
    throw_id: str,
    reason: str,
    *,
    exclusions_path: Path = EXCLUSION_FILE,
    log_path: Path = QC_LOG,
) -> None:
    """Mark a throw as excluded from reference calculations (coach-reviewed decision)."""
    log = _configure_logger(log_path)
    throw_id = (throw_id or "").strip()
    if not throw_id:
        raise ValueError("throw_id cannot be blank")

    exclusions = load_exclusions(exclusions_path)
    exclusions[throw_id] = {
        "reason": (reason or "").strip(),
        "excluded_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_exclusions(exclusions_path, exclusions)
    log.info("Excluded throw %s: %s", throw_id, reason)


def _iter_processed_metadata_rows(metadata_csv: Path) -> Iterable[dict[str, str]]:
    with metadata_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield {k: (v or "") for k, v in row.items()}


def _collect_processed_throws(
    *,
    poses_dir: Path,
    metadata_csv: Optional[Path],
    excluded_ids: set[str],
    log: logging.Logger,
) -> Tuple[List[ProcessedThrow], Dict[str, int], List[str]]:
    warnings: List[str] = []
    throws: List[ProcessedThrow] = []
    style_counts: Dict[str, int] = {}

    if metadata_csv is None or not metadata_csv.exists():
        if not poses_dir.exists():
            return [], {}, ["Poses directory not found."]
        for pose_file in sorted(poses_dir.glob("*.json")):
            throw_id = pose_file.stem
            if throw_id in excluded_ids:
                continue
            payload = _load_json(pose_file)
            tuples = compute_throw_metrics(payload)
            metrics = {k: float(v[0]) for k, v in tuples.items() if isinstance(v, tuple) and len(v) >= 1}
            if not metrics:
                warnings.append(f"Skipping {pose_file.name}: no metrics computed.")
                continue
            throws.append(
                ProcessedThrow(
                    throw_id=throw_id,
                    thrower_name=throw_id,
                    throwing_style="unknown",
                    pose_path=pose_file,
                    metrics=metrics,
                )
            )
            style_counts["unknown"] = style_counts.get("unknown", 0) + 1
        style_counts["overall"] = len(throws)
        return throws, style_counts, warnings

    for row in _iter_processed_metadata_rows(metadata_csv):
        status = (row.get("processed_status") or "").strip().lower()
        if status != "complete":
            continue
        throw_id = _throw_id_from_row(row)
        if throw_id in excluded_ids:
            continue

        pose_path = poses_dir / f"{throw_id}.json"
        if not pose_path.exists():
            warnings.append(f"Missing pose JSON for {throw_id}: expected {pose_path}")
            continue

        payload = _load_json(pose_path)
        tuples = compute_throw_metrics(payload)
        metrics = {k: float(v[0]) for k, v in tuples.items() if isinstance(v, tuple) and len(v) >= 1}
        if not metrics:
            warnings.append(f"Skipping {throw_id}: no metrics computed from {pose_path.name}")
            continue

        style = _normalize_style(row.get("throwing_style"))
        style_counts[style] = style_counts.get(style, 0) + 1

        throws.append(
            ProcessedThrow(
                throw_id=throw_id,
                thrower_name=(row.get("thrower_name") or "").strip(),
                throwing_style=style,
                pose_path=pose_path,
                metrics=metrics,
            )
        )

    style_counts["overall"] = len(throws)
    if not throws:
        warnings.append("No processed throws found (processed_status=complete).")
    return throws, style_counts, warnings


def validate_elite_profiles(
    reference_path: Path | None = None,
    style_profiles: Optional[Dict[str, Dict[str, object]]] = None,
    *,
    poses_dir: Path | None = None,
    metadata_csv: Path | None = None,
    exclusions_path: Path = EXCLUSION_FILE,
    log_path: Path = QC_LOG,
    outlier_zscore: float = DEFAULT_OUTLIER_ZSCORE,
) -> Dict[str, object]:
    """Validate elite reference profiles with per-throw outlier detection.

    Outlier definition: a throw is flagged when `abs(value - mean) / std > 3`
    for any metric, using the mean/std stored in the reference profile JSON.

    Low-confidence metrics are flagged when the coefficient of variation is high:
      `std > abs(mean)` (equivalently, `confidence < 0.5`).
    """
    ref_path = Path(reference_path) if reference_path is not None else DEFAULT_REFERENCE_PATH
    poses_dir = Path(poses_dir) if poses_dir is not None else DEFAULT_POSES_DIR
    metadata_csv = Path(metadata_csv) if metadata_csv is not None else DEFAULT_METADATA_CSV

    log = _configure_logger(log_path)
    excluded = load_exclusions(exclusions_path)
    excluded_ids = set(excluded.keys())

    reference = _load_json(ref_path)
    if not reference:
        log.warning("Reference profile missing or invalid: %s", ref_path)

    throws, style_counts, warnings = _collect_processed_throws(
        poses_dir=poses_dir,
        metadata_csv=metadata_csv if metadata_csv.exists() else None,
        excluded_ids=excluded_ids,
        log=log,
    )

    if not reference and throws:
        reference = compute_elite_reference_profile_from_files([t.pose_path for t in throws])

    low_confidence: List[str] = []
    for metric, stats in reference.items() if isinstance(reference, dict) else []:
        if not isinstance(stats, dict):
            continue
        mean_val = stats.get("mean")
        std_val = stats.get("std")
        if not isinstance(mean_val, (int, float)) or not isinstance(std_val, (int, float)):
            continue
        if not math.isfinite(float(mean_val)) or not math.isfinite(float(std_val)):
            continue
        if float(std_val) <= 0.0:
            continue
        confidence = stats.get("confidence")
        if isinstance(confidence, (int, float)) and math.isfinite(float(confidence)):
            if float(confidence) < 0.5:
                low_confidence.append(metric)
        else:
            scale = max(abs(float(mean_val)), 1e-9)
            if float(std_val) / scale > 1.0:
                low_confidence.append(metric)

    outliers: List[Dict[str, object]] = []
    for metric, stats in reference.items() if isinstance(reference, dict) else []:
        if not isinstance(stats, dict):
            continue
        mean_val = stats.get("mean")
        std_val = stats.get("std")
        if not isinstance(mean_val, (int, float)) or not isinstance(std_val, (int, float)):
            continue
        if float(std_val) <= 0.0:
            continue
        if not math.isfinite(float(mean_val)) or not math.isfinite(float(std_val)):
            continue

        for throw in throws:
            value = throw.metrics.get(metric)
            if value is None:
                continue
            if not math.isfinite(float(value)):
                continue
            z = abs(float(value) - float(mean_val)) / float(std_val) if float(std_val) else 0.0
            if z > outlier_zscore:
                outliers.append(
                    {
                        "throw_id": throw.throw_id,
                        "thrower_name": throw.thrower_name,
                        "throwing_style": throw.throwing_style,
                        "metric": metric,
                        "value": float(value),
                        "mean": float(mean_val),
                        "std": float(std_val),
                        "z_score": float(z),
                        "phase": stats.get("phase"),
                        "unit": stats.get("unit"),
                    }
                )

    recommendations: List[str] = []
    recommendations.extend(warnings)

    if outliers:
        recommendations.append(
            "Outliers detected; coach review is required before excluding throws. "
            "If excluding, record the decision with exclude_throw(throw_id, reason)."
        )
    if low_confidence:
        recommendations.append(
            "Low-confidence metrics detected (high variability). Consider collecting more samples or refining metric definitions."
        )

    if style_profiles:
        for style, entry in style_profiles.items():
            if style == "overall":
                continue
            count = int(entry.get("n_samples", 0)) if isinstance(entry, dict) else 0
            if count < 2:
                recommendations.append(f"Style '{style}' has insufficient samples ({count}); collect more data.")

    report = {
        "outliers": outliers,
        "low_confidence_metrics": sorted(set(low_confidence)),
        "style_sample_counts": style_counts,
        "recommendations": recommendations or ["Profiles validated; no critical issues detected."],
    }

    log.info("QC report: %s", json.dumps(report, sort_keys=True))
    return report


def recompute_profiles_after_exclusions(
    *,
    poses_dir: Path | None = None,
    metadata_csv: Path | None = None,
    output_dir: Path | None = None,
    exclusions_path: Path = EXCLUSION_FILE,
    log_path: Path = QC_LOG,
    min_style_samples: int = DEFAULT_MIN_STYLE_SAMPLES,
) -> Dict[str, Dict[str, object]]:
    """Recompute overall + per-style reference profiles, excluding coach-flagged throws."""
    poses_dir = Path(poses_dir) if poses_dir is not None else DEFAULT_POSES_DIR
    metadata_csv = Path(metadata_csv) if metadata_csv is not None else DEFAULT_METADATA_CSV
    output_base = Path(output_dir) if output_dir is not None else ELITE_DATABASE_DIR

    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")

    log = _configure_logger(log_path)
    excluded = load_exclusions(exclusions_path)
    excluded_ids = set(excluded.keys())

    entries: Dict[str, List[Tuple[Path, str]]] = {}
    for row in _iter_processed_metadata_rows(metadata_csv):
        status = (row.get("processed_status") or "").strip().lower()
        if status != "complete":
            continue
        throw_id = _throw_id_from_row(row)
        if throw_id in excluded_ids:
            continue
        pose_path = poses_dir / f"{throw_id}.json"
        if not pose_path.exists():
            continue
        style = _normalize_style(row.get("throwing_style"))
        thrower_name = (row.get("thrower_name") or "").strip()
        entries.setdefault(style, []).append((pose_path, thrower_name))

    profiles: Dict[str, Dict[str, object]] = {}

    for style, files_and_names in entries.items():
        files = [p for p, _ in files_and_names]
        if len(files) < min_style_samples:
            log.warning("Skipping style %s: only %s samples after exclusions", style, len(files))
            continue
        ref = compute_elite_reference_profile_from_files(files)
        profiles[style] = {
            "metrics": ref,
            "n_samples": len(files),
            "throwers": sorted({name for _, name in files_and_names if name}),
        }
        out_path = output_base / f"reference_profile_{_style_filename_slug(style)}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(ref, indent=2), encoding="utf-8")
        log.info("Wrote style profile %s (%s samples)", style, len(files))

    overall_entries: Dict[Path, str] = {}
    for files_and_names in entries.values():
        for path, name in files_and_names:
            overall_entries.setdefault(path, name)
    overall_files = list(overall_entries.keys())
    overall_ref = compute_elite_reference_profile_from_files(overall_files)
    profiles["overall"] = {
        "metrics": overall_ref,
        "n_samples": len(overall_files),
        "throwers": sorted({name for name in overall_entries.values() if name}),
    }

    out_overall = output_base / "reference_profile_overall.json"
    out_overall.parent.mkdir(parents=True, exist_ok=True)
    out_overall.write_text(json.dumps(overall_ref, indent=2), encoding="utf-8")
    (output_base / "reference_profile.json").write_text(json.dumps(overall_ref, indent=2), encoding="utf-8")
    log.info("Wrote overall profile (%s samples) to %s", len(overall_files), out_overall)
    return profiles


def plot_metric_distribution(
    metric: str,
    *,
    poses_dir: Path | None = None,
    metadata_csv: Path | None = None,
    reference_path: Path | None = None,
    exclusions_path: Path = EXCLUSION_FILE,
    outlier_zscore: float = DEFAULT_OUTLIER_ZSCORE,
):
    """Plot metric distribution and highlight outlier throws."""
    import matplotlib

    try:  # Avoid interactive backend requirements in headless contexts.
        matplotlib.use("Agg", force=False)
    except Exception:
        pass
    import matplotlib.pyplot as plt

    poses_dir = Path(poses_dir) if poses_dir is not None else DEFAULT_POSES_DIR
    metadata_csv = Path(metadata_csv) if metadata_csv is not None else DEFAULT_METADATA_CSV
    ref_path = Path(reference_path) if reference_path is not None else DEFAULT_REFERENCE_PATH

    log = logging.getLogger(__name__)
    excluded = load_exclusions(exclusions_path)
    excluded_ids = set(excluded.keys())

    throws, _, _ = _collect_processed_throws(
        poses_dir=poses_dir,
        metadata_csv=metadata_csv if metadata_csv.exists() else None,
        excluded_ids=excluded_ids,
        log=log,
    )
    values: List[Tuple[str, float]] = []
    for throw in throws:
        if metric in throw.metrics:
            values.append((throw.throw_id, float(throw.metrics[metric])))
    if not values:
        return None

    metric_values = [v for _, v in values]
    reference = _load_json(ref_path)
    ref_stats = reference.get(metric) if isinstance(reference, dict) else None
    if isinstance(ref_stats, dict) and isinstance(ref_stats.get("mean"), (int, float)) and isinstance(ref_stats.get("std"), (int, float)):
        mean_val = float(ref_stats["mean"])
        std_val = float(ref_stats["std"])
    else:
        mean_val = float(mean(metric_values))
        std_val = float(pstdev(metric_values)) if len(metric_values) > 1 else 0.0

    outlier_ids = set()
    if std_val > 0:
        for throw_id, value in values:
            z = abs(value - mean_val) / std_val
            if z > outlier_zscore:
                outlier_ids.add(throw_id)

    fig, ax = plt.subplots(figsize=(10, 4))
    bins = min(25, max(5, int(math.sqrt(len(metric_values)))))
    ax.hist(metric_values, bins=bins, alpha=0.6, color="steelblue", edgecolor="white")
    ax.axvline(mean_val, color="green", label="mean")
    if std_val > 0:
        ax.axvline(mean_val + outlier_zscore * std_val, color="red", linestyle="--", label=f"+{outlier_zscore} std")
        ax.axvline(mean_val - outlier_zscore * std_val, color="red", linestyle="--", label=f"-{outlier_zscore} std")

    ax2 = ax.twinx()
    ax2.set_yticks([])
    ax2.scatter([v for _, v in values], [0] * len(values), s=18, color="black", alpha=0.6, label="throws")
    if outlier_ids:
        outlier_points = [(tid, v) for tid, v in values if tid in outlier_ids]
        ax2.scatter([v for _, v in outlier_points], [0] * len(outlier_points), s=30, color="red", label="outliers")
        for throw_id, value in outlier_points:
            ax2.annotate(throw_id, (value, 0), xytext=(0, 8), textcoords="offset points", rotation=90, ha="center", fontsize=8)

    ax.set_title(f"Elite metric distribution: {metric}")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    return fig


__all__ = [
    "validate_elite_profiles",
    "exclude_throw",
    "load_exclusions",
    "recompute_profiles_after_exclusions",
    "plot_metric_distribution",
]
