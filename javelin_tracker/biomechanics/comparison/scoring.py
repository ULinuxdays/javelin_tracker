"""Elite reference scoring helpers (per-metric z-scores and coach-friendly scoring).

This module converts athlete metric values into a simple 0–100 score relative to
an elite reference distribution (mean/std). It is intended for coach-facing
feedback, not scientific inference.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional


def _coerce_float(value: Any) -> Optional[float]:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(num):
        return None
    return float(num)


def _normalize_style(style: str | None) -> str:
    """Normalize a coach-entered style label to a stable key."""
    text = (style or "").strip().lower()
    return text or "overall"


def _looks_like_style_profiles(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    # style_profiles shape is {style: {"metrics": {...}, "n_samples": ...}}
    for _key, entry in value.items():
        if isinstance(entry, Mapping) and isinstance(entry.get("metrics"), Mapping):
            return True
    return False


def _extract_metrics_and_samples(profile_or_metrics: Mapping[str, Any]) -> tuple[Mapping[str, Any], Optional[int]]:
    """Return (metrics_mapping, n_samples) from either a wrapper or raw profile."""
    metrics = profile_or_metrics.get("metrics")
    if isinstance(metrics, Mapping):
        n = profile_or_metrics.get("n_samples")
        try:
            n_samples = int(n) if n is not None else None
        except (TypeError, ValueError):
            n_samples = None
        return metrics, n_samples

    # Assume it's already a metrics mapping (metric_key -> stats).
    n_samples: Optional[int] = None
    candidates: list[int] = []
    for stats in profile_or_metrics.values():
        if not isinstance(stats, Mapping):
            continue
        raw = stats.get("n_samples")
        try:
            if raw is not None:
                candidates.append(int(raw))
        except (TypeError, ValueError):
            continue
    if candidates:
        n_samples = max(candidates)
    return profile_or_metrics, n_samples


def resolve_style_reference(
    style: str | None,
    all_style_profiles: Mapping[str, Any],
    *,
    fallback_style: str = "overall",
    min_samples: int = 2,
) -> dict[str, object]:
    """Resolve a reference profile for a given style with fallback to overall."""
    requested = _normalize_style(style)
    fallback = _normalize_style(fallback_style)

    selected = requested
    entry = all_style_profiles.get(selected)
    if not isinstance(entry, Mapping):
        selected = fallback
        entry = all_style_profiles.get(selected)

    if isinstance(entry, Mapping):
        metrics, n_samples = _extract_metrics_and_samples(entry)
        if selected != fallback and (n_samples is None or n_samples < int(min_samples)):
            selected = fallback
            entry = all_style_profiles.get(selected)
            if isinstance(entry, Mapping):
                metrics, n_samples = _extract_metrics_and_samples(entry)
            else:
                metrics, n_samples = {}, None
        note = (
            f"Using style '{selected}'."
            if selected == requested
            else f"Style '{requested}' unavailable/insufficient; falling back to '{selected}'."
        )
        return {
            "style_name": selected,
            "metrics": dict(metrics),
            "n_elite_samples": n_samples,
            "note_about_style": note,
        }

    return {
        "style_name": fallback,
        "metrics": {},
        "n_elite_samples": None,
        "note_about_style": f"No style profiles available; using '{fallback}'.",
    }


def score_metric(athlete_value: Any, elite_mean: Any, elite_std: Any, *, style: str | None = "overall") -> Dict[str, object]:
    """Score a single metric value vs elite mean/std using a z-score.

    Z-score:
        z = (value - mean) / std

    Flag if:
        |z| > 1.5

    Score (0–100):
        score = max(0, 100 - 10*|z|)

    Status:
        - "good" when |z| < 1
        - "warning" when 1 <= |z| <= 1.5
        - "poor" when |z| > 1.5
    """
    athlete = _coerce_float(athlete_value)
    mean = _coerce_float(elite_mean)
    std = _coerce_float(elite_std)
    if athlete is None or mean is None or std is None:
        raise ValueError("athlete_value, elite_mean, and elite_std must be finite numbers.")

    diff = athlete - mean
    if std <= 0.0:
        z = 0.0 if abs(diff) <= 1e-9 * max(1.0, abs(mean)) else math.copysign(math.inf, diff)
    else:
        z = diff / std

    abs_z = abs(float(z))
    score = max(0.0, 100.0 - 10.0 * abs_z)
    flagged = bool(abs_z > 1.5)

    if abs_z < 1.0:
        status = "good"
    elif abs_z <= 1.5:
        status = "warning"
    else:
        status = "poor"

    return {
        "athlete_value": athlete,
        "elite_mean": mean,
        "elite_std": std,
        "z_score": float(z),
        "score": float(score),
        "status": status,
        "flagged": flagged,
        "style": _normalize_style(style),
    }


def get_metric_weight(metric_name: str) -> float:
    """Return a heuristic importance weight for ranking metrics."""
    name = (metric_name or "").strip().lower()
    if not name:
        return 1.0

    # Explicit low-importance / debug metrics.
    if "throwing_side_right" in name:
        return 0.1
    if "mean_pose_confidence" in name:
        return 0.2

    # Phase baseline importance.
    if name.startswith("release."):
        weight = 1.6
    elif name.startswith("delivery."):
        weight = 1.3
    elif name.startswith("approach."):
        weight = 0.9
    elif name.startswith("overall."):
        weight = 0.4
    else:
        weight = 1.0

    # Specific focus areas for javelin performance.
    if "at_release" in name:
        weight += 0.8
    if "wrist_speed" in name:
        weight += 0.6
    if "hip_speed" in name:
        weight += 0.3
    if "shoulder_hip_separation" in name:
        weight += 0.4
    if "asymmetry" in name:
        weight += 0.3
    if "duration_ms" in name:
        weight *= 0.6

    return float(weight)


_METRIC_DESCRIPTIONS: dict[str, str] = {
    "duration_ms": "Time spent in this phase (milliseconds).",
    "throwing_wrist_speed_at_release": "Throwing wrist speed at the release frame (proxy for release speed).",
    "throwing_elbow_flexion_deg_at_release": "Throwing elbow flexion at release (degrees).",
    "throwing_wrist_speed_mean": "Average throwing wrist speed during the phase.",
    "throwing_wrist_speed_max": "Peak throwing wrist speed during the phase.",
    "throwing_elbow_speed_mean": "Average throwing elbow speed during the phase.",
    "throwing_elbow_speed_max": "Peak throwing elbow speed during the phase.",
    "hip_speed_mean": "Average hip midpoint speed during the phase.",
    "hip_speed_max": "Peak hip midpoint speed during the phase.",
    "throwing_elbow_flexion_deg_mean": "Average throwing elbow flexion during the phase (degrees).",
    "throwing_elbow_flexion_deg_min": "Minimum throwing elbow flexion during the phase (degrees; lower means more extension).",
    "throwing_shoulder_angle_deg_mean": "Average throwing-side shoulder angle during the phase (degrees).",
    "trunk_lean_deg_mean": "Average trunk lean during the phase (degrees from vertical).",
    "shoulder_height_asymmetry_mean": "Average left/right shoulder height difference (lower is more symmetrical).",
    "elbow_flexion_asymmetry_deg_mean": "Average left/right elbow flexion difference (degrees; lower is more symmetrical).",
    "shoulder_hip_separation_deg_mean": "Average separation between shoulders and hips (hip–shoulder dissociation).",
    "mean_pose_confidence": "Average pose-landmark confidence for the video (data quality).",
    "throwing_side_right": "Debug: 1=right-handed throw detected, 0=left-handed throw detected.",
}


def get_metric_description(metric_name: str) -> str:
    """Return a coach-friendly description for a metric key."""
    raw = (metric_name or "").strip()
    if not raw:
        return ""

    phase = ""
    name = raw
    if "." in raw:
        phase, name = raw.split(".", 1)

    desc = _METRIC_DESCRIPTIONS.get(name.lower())
    if not desc:
        # Fallback: humanize.
        human = name.replace("_", " ").strip()
        desc = f"{human}."

    phase_norm = phase.strip().lower()
    if phase_norm in {"approach", "delivery", "release"}:
        return f"{phase_norm.title()} phase: {desc}"
    return desc


def _coerce_reference_profile(reference: Any) -> Mapping[str, Any]:
    if isinstance(reference, Mapping):
        metrics = reference.get("metrics")
        if isinstance(metrics, Mapping):
            return metrics
        return reference
    raise ValueError("elite_reference_profile must be a mapping (or a style profile with a 'metrics' mapping).")


def _coerce_athlete_metrics(metrics: Any) -> Mapping[str, Any]:
    if isinstance(metrics, Mapping):
        inner = metrics.get("metrics")
        if isinstance(inner, Mapping):
            return inner
        return metrics
    raise ValueError("athlete_metrics must be a mapping.")


def score_all_metrics(
    athlete_metrics: Any,
    elite_reference_profile: Any,
    *,
    style: str | None = "overall",
    all_style_profiles: Any | None = None,
) -> list[dict[str, object]]:
    """Score all athlete metrics that exist in the elite reference profile.

    Returns:
        A ranked list (worst/most-important first) of score entries.
    """
    athlete = _coerce_athlete_metrics(athlete_metrics)

    # Accept either a raw reference profile OR a style-profiles mapping.
    style_profiles = all_style_profiles
    if style_profiles is None and _looks_like_style_profiles(elite_reference_profile):
        style_profiles = elite_reference_profile

    if style_profiles is not None and isinstance(style_profiles, Mapping):
        resolved = resolve_style_reference(style, style_profiles)
        reference = _coerce_reference_profile(resolved["metrics"])
        style_key = str(resolved.get("style_name") or _normalize_style(style))
        n_style_samples = resolved.get("n_elite_samples")
        style_note = resolved.get("note_about_style")
    else:
        reference = _coerce_reference_profile(elite_reference_profile)
        style_key = _normalize_style(style)
        n_style_samples = None
        style_note = None

    athlete_values: dict[str, float] = {}
    for key, value in athlete.items():
        if isinstance(value, (tuple, list)) and value:
            value = value[0]
        num = _coerce_float(value)
        if num is None:
            continue
        athlete_values[str(key)] = num

    entries: list[dict[str, object]] = []
    missing: list[str] = []
    for metric_key, ref_stats in reference.items():
        if not isinstance(metric_key, str) or not isinstance(ref_stats, Mapping):
            continue
        athlete_value = athlete_values.get(metric_key)
        if athlete_value is None:
            missing.append(metric_key)
            continue

        mean = ref_stats.get("mean")
        std = ref_stats.get("std")
        try:
            scored = score_metric(athlete_value, mean, std, style=style_key)
        except ValueError:
            continue

        weight = get_metric_weight(metric_key)
        abs_z = abs(float(scored["z_score"]))
        impact = weight * abs_z

        entry: dict[str, object] = {
            "metric": metric_key,
            "description": get_metric_description(metric_key),
            "phase": ref_stats.get("phase"),
            "unit": ref_stats.get("unit"),
            "elite_confidence": ref_stats.get("confidence"),
            "elite_n_samples": ref_stats.get("n_samples"),
            "elite_style": style_key,
            "elite_style_n_samples": n_style_samples,
            "elite_style_note": style_note,
            "weight": float(weight),
            "impact": float(impact),
            "weighted_score": float(scored["score"]) * float(weight),
            **scored,
        }
        entries.append(entry)

    severity = {"poor": 2, "warning": 1, "good": 0}
    entries.sort(
        key=lambda e: (
            -severity.get(str(e.get("status")), 0),
            -float(e.get("impact") or 0.0),
            str(e.get("metric") or ""),
        )
    )
    return entries


__all__ = [
    "score_metric",
    "score_all_metrics",
    "infer_athlete_style",
    "resolve_style_reference",
    "get_metric_description",
    "get_metric_weight",
]


def infer_athlete_style(
    athlete_metrics: Any,
    all_style_profiles: Mapping[str, Any],
    *,
    fallback_style: str = "overall",
    min_samples: int = 2,
) -> dict[str, object]:
    """Infer throwing style by choosing the best-matching elite style profile.

    The best match is defined as the style with the *lowest weighted mean abs z-score*
    across all overlapping metrics. Confidence is reported as a 0–1 value derived
    from a softmax over (negative) costs.
    """
    athlete = _coerce_athlete_metrics(athlete_metrics)
    athlete_values: dict[str, float] = {}
    for key, value in athlete.items():
        if isinstance(value, (tuple, list)) and value:
            value = value[0]
        num = _coerce_float(value)
        if num is None:
            continue
        athlete_values[str(key)] = num

    fallback = _normalize_style(fallback_style)
    candidates: list[tuple[str, float, int, Optional[int]]] = []
    for style_key_raw, profile in all_style_profiles.items():
        style_key = _normalize_style(str(style_key_raw))
        if not isinstance(profile, Mapping):
            continue
        metrics, n_samples = _extract_metrics_and_samples(profile)
        if style_key != fallback and (n_samples is None or n_samples < int(min_samples)):
            continue
        if not isinstance(metrics, Mapping) or not metrics:
            continue

        pairs: list[tuple[float, float]] = []
        compared = 0
        for metric_key, stats in metrics.items():
            if not isinstance(metric_key, str) or metric_key not in athlete_values:
                continue
            if not isinstance(stats, Mapping):
                continue
            compared += 1
            try:
                scored = score_metric(athlete_values[metric_key], stats.get("mean"), stats.get("std"), style=style_key)
            except ValueError:
                continue
            abs_z = abs(float(scored["z_score"]))
            if not math.isfinite(abs_z):
                abs_z = 10.0
            weight = float(get_metric_weight(metric_key))
            pairs.append((abs_z, weight))

        if not pairs:
            continue

        total_w = sum(w for _v, w in pairs if math.isfinite(w) and w > 0)
        if total_w <= 0:
            continue
        cost = sum(v * w for v, w in pairs) / total_w
        candidates.append((style_key, float(cost), int(compared), n_samples))

    if not candidates:
        resolved = resolve_style_reference(fallback, all_style_profiles)
        return {
            "style_name": str(resolved.get("style_name") or fallback),
            "n_elite_samples": resolved.get("n_elite_samples"),
            "metrics_compared": 0,
            "note_about_style": "Style inference unavailable; using overall reference.",
            "confidence": 0.0,
        }

    candidates.sort(key=lambda row: row[1])
    best_style, best_cost, best_compared, best_samples = candidates[0]

    # Softmax confidence based on relative costs.
    min_cost = min(cost for _s, cost, _c, _n in candidates)
    weights = [math.exp(-(cost - min_cost)) for _s, cost, _c, _n in candidates]
    denom = sum(weights) or 1.0
    best_weight = weights[0]
    confidence = float(best_weight / denom)

    note = f"Inferred style '{best_style}' from {best_compared} metrics (confidence={confidence:.2f})."
    if best_style == fallback:
        note = f"Inferred 'overall' reference from {best_compared} metrics (confidence={confidence:.2f})."

    return {
        "style_name": best_style,
        "n_elite_samples": best_samples,
        "metrics_compared": best_compared,
        "note_about_style": note,
        "confidence": confidence,
    }
