"""Comparison report generator for athlete-vs-elite biomechanics.

This module turns scored per-metric comparisons into a compact JSON payload
intended for UI display (overall score, phase breakdown, top issues).
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from javelin_tracker.biomechanics.comparison.scoring import (
    infer_athlete_style,
    resolve_style_reference,
    score_all_metrics,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_jsonish(value: Any) -> dict:
    if isinstance(value, Mapping):
        return dict(value)
    path = Path(value)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Expected JSON object at path.")
    return payload


def _coerce_float(value: Any) -> Optional[float]:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(num):
        return None
    return float(num)


def _coerce_reference_profile(elite_profile: Any) -> Mapping[str, Any]:
    profile = _load_jsonish(elite_profile)
    metrics = profile.get("metrics")
    if isinstance(metrics, Mapping):
        return metrics
    return profile


def _looks_like_style_profiles(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    for _key, entry in value.items():
        if isinstance(entry, Mapping) and isinstance(entry.get("metrics"), Mapping):
            return True
    return False


def _coerce_athlete_metrics(athlete_metrics: Any) -> Mapping[str, float]:
    payload = _load_jsonish(athlete_metrics)

    # Support passing a processed pose payload: compute metrics on the fly.
    if "frames" in payload and "video_metadata" in payload:
        try:
            from javelin_tracker.biomechanics.database.elite_reference import compute_throw_metrics

            tuples = compute_throw_metrics(payload)
            out: dict[str, float] = {}
            for key, tup in tuples.items():
                if isinstance(tup, (tuple, list)) and tup:
                    num = _coerce_float(tup[0])
                    if num is not None:
                        out[str(key)] = num
            return out
        except Exception:
            # Fall through to best-effort parsing below.
            pass

    metrics = payload.get("metrics")
    if isinstance(metrics, Mapping):
        payload = dict(metrics)

    out: dict[str, float] = {}
    for key, value in payload.items():
        if isinstance(value, (tuple, list)) and value:
            value = value[0]
        num = _coerce_float(value)
        if num is None:
            continue
        out[str(key)] = num
    return out


def _weighted_average(pairs: list[tuple[float, float]]) -> Optional[float]:
    if not pairs:
        return None
    total_w = 0.0
    total = 0.0
    for score, weight in pairs:
        if not math.isfinite(float(score)) or not math.isfinite(float(weight)):
            continue
        if weight <= 0.0:
            continue
        total += float(score) * float(weight)
        total_w += float(weight)
    if total_w <= 0.0:
        return None
    return float(total / total_w)


def generate_comparison_report(
    athlete_metrics: Any,
    elite_profile: Any,
    athlete_id: str,
    *,
    top_n: int = 5,
    style: str | None = "overall",
    all_style_profiles: Any | None = None,
) -> dict[str, object]:
    """Generate a UI-friendly comparison report (athlete vs elite profile)."""
    athlete_vals = _coerce_athlete_metrics(athlete_metrics)
    elite_payload = _load_jsonish(elite_profile)

    style_profiles = all_style_profiles
    if style_profiles is None and _looks_like_style_profiles(elite_payload):
        style_profiles = elite_payload

    requested_style = (style or "overall").strip()
    requested_norm = requested_style.lower()
    infer_requested = requested_norm in {"", "auto", "infer", "detect"}

    style_info: dict[str, object] = {}
    overridden = False
    if style_profiles is not None and isinstance(style_profiles, Mapping):
        if infer_requested:
            style_info = infer_athlete_style(athlete_vals, style_profiles)
            selected_style = str(style_info.get("style_name") or "overall")
        else:
            overridden = True
            resolved = resolve_style_reference(requested_style, style_profiles)
            selected_style = str(resolved.get("style_name") or "overall")
            style_info = {
                "style_name": selected_style,
                "n_elite_samples": resolved.get("n_elite_samples"),
                "metrics_compared": 0,
                "note_about_style": f"Coach override: {resolved.get('note_about_style')}",
                "confidence": 1.0,
            }

        resolved = resolve_style_reference(str(style_info.get("style_name") or "overall"), style_profiles)
        selected_style = str(resolved.get("style_name") or "overall")
        reference = _coerce_reference_profile(resolved.get("metrics") or {})
        note = str(style_info.get("note_about_style") or resolved.get("note_about_style") or "")
        confidence = float(style_info.get("confidence") or 0.0)
        n_elite_samples = style_info.get("n_elite_samples") or resolved.get("n_elite_samples")
    else:
        selected_style = (requested_style or "overall").strip().lower() or "overall"
        reference = _coerce_reference_profile(elite_payload)
        note = "Style profiles not provided; using the supplied elite reference profile."
        confidence = 0.0
        n_elite_samples = None

    scored = score_all_metrics(athlete_vals, style_profiles or reference, style=selected_style)
    if not scored:
        raise ValueError("No comparable metrics found between athlete_metrics and elite_profile.")

    missing = 0
    for metric_key, stats in reference.items():
        if not isinstance(metric_key, str) or not isinstance(stats, Mapping):
            continue
        if metric_key not in athlete_vals:
            missing += 1

    # Rank deviations by severity (absolute z-score magnitude).
    def abs_z(item: Mapping[str, object]) -> float:
        try:
            return abs(float(item.get("z_score", 0.0)))
        except (TypeError, ValueError):
            return 0.0

    scored_sorted_by_z = sorted(scored, key=lambda e: (abs_z(e), float(e.get("weight", 1.0))), reverse=True)
    top_issues_detailed = []
    for idx, item in enumerate(scored_sorted_by_z[: int(top_n)], start=1):
        top_issues_detailed.append(
            {
                "rank": idx,
                "metric": item["metric"],
                "description": item.get("description"),
                "phase": item.get("phase"),
                "unit": item.get("unit"),
                "athlete_value": item.get("athlete_value"),
                "elite_mean": item.get("elite_mean"),
                "elite_std": item.get("elite_std"),
                "z_score": item.get("z_score"),
                "score": item.get("score"),
                "status": item.get("status"),
                "weight": item.get("weight"),
            }
        )

    # Strengths/weaknesses.
    strengths = [m for m in scored if str(m.get("status")) == "good"]
    weaknesses = [m for m in scored if str(m.get("status")) in {"warning", "poor"}]

    strengths.sort(key=lambda e: (float(e.get("score", 0.0)), float(e.get("weight", 1.0))), reverse=True)
    weaknesses.sort(key=lambda e: (abs_z(e) * float(e.get("weight", 1.0))), reverse=True)

    strengths_out = [
        {
            "metric": m["metric"],
            "description": m.get("description"),
            "phase": m.get("phase"),
            "unit": m.get("unit"),
            "athlete_value": m.get("athlete_value"),
            "elite_mean": m.get("elite_mean"),
            "elite_std": m.get("elite_std"),
            "z_score": m.get("z_score"),
            "score": m.get("score"),
            "weight": m.get("weight"),
        }
        for m in strengths[: int(top_n)]
    ]
    weaknesses_out = [
        {
            "metric": m["metric"],
            "description": m.get("description"),
            "phase": m.get("phase"),
            "unit": m.get("unit"),
            "athlete_value": m.get("athlete_value"),
            "elite_mean": m.get("elite_mean"),
            "elite_std": m.get("elite_std"),
            "z_score": m.get("z_score"),
            "score": m.get("score"),
            "status": m.get("status"),
            "flagged": m.get("flagged"),
            "weight": m.get("weight"),
        }
        for m in weaknesses[: int(top_n)]
    ]

    # Phase breakdown.
    phase_pairs: dict[str, list[tuple[float, float]]] = {"approach": [], "delivery": [], "release": []}
    overall_pairs: list[tuple[float, float]] = []
    for m in scored:
        phase = str(m.get("phase") or "")
        score = _coerce_float(m.get("score"))
        weight = _coerce_float(m.get("weight")) or 0.0
        if score is None:
            continue
        overall_pairs.append((score, weight))
        if phase in phase_pairs:
            phase_pairs[phase].append((score, weight))

    phase_breakdown = {phase: _weighted_average(pairs) for phase, pairs in phase_pairs.items()}
    overall_score = _weighted_average(overall_pairs)

    return {
        "athlete_id": (athlete_id or "").strip() or "unknown",
        "generated_at": _now_iso(),
        "style_comparison": {
            "style_name": str(selected_style),
            "n_elite_samples": n_elite_samples,
            "metrics_compared": len(scored),
            "note_about_style": note,
            "confidence": confidence,
            "overridden": bool(overridden),
            "requested_style": requested_style,
        },
        "overall_score": overall_score,
        "phase_breakdown": phase_breakdown,
        "top_issues": top_issues_detailed,
        "strengths": strengths_out,
        "weaknesses": weaknesses_out,
        "n_metrics_scored": len(scored),
        "missing_metrics": int(missing),
    }


__all__ = ["generate_comparison_report"]
