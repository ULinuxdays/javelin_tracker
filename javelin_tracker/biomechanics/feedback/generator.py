"""Actionable feedback generator for athlete-vs-elite comparisons.

This module turns a `comparison_report` (from `comparison.reporter`) and optional
`throw_metrics` (from `metrics.throw_metrics`) into ranked, coach-facing cues.

The intent is UI-friendly output: short, actionable guidance, plus context:
  - what metric is off (current vs ideal)
  - why it matters biomechanically
  - a concrete correction + suggested drill
  - optional frame ranges (phase windows) to review in video

Ranking philosophy:
  - Selection: uses the report's top issues (already ranked by |z|).
  - Final ranking: uses an "impact score" that multiplies |z| by an importance
    weight (heuristic proxy for distance impact).
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from javelin_tracker.biomechanics.comparison.scoring import get_metric_description, get_metric_weight
from javelin_tracker.biomechanics.feedback.rules import evaluate_rules, load_rules_config


def _load_jsonish(value: Any) -> dict:
    if isinstance(value, Mapping):
        return dict(value)
    path = Path(value)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Expected JSON object at path.")
    return payload


def _coerce_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(num):
        return None
    return float(num)


def _coerce_throw_metrics(value: Any) -> dict:
    """Accept either raw throw_metrics dict, metrics.json, or a JSON path."""
    if value is None:
        return {}
    payload = _load_jsonish(value)
    # metrics pipeline output uses "throw_metrics" key.
    inner = payload.get("throw_metrics")
    if isinstance(inner, Mapping):
        return dict(inner)
    return payload


def _ideal_range(mean: Optional[float], std: Optional[float], unit: Any) -> Optional[dict[str, object]]:
    if mean is None:
        return None
    if std is None or not math.isfinite(float(std)) or float(std) < 0:
        return {"min": float(mean), "max": float(mean), "unit": unit}
    return {"min": float(mean) - float(std), "max": float(mean) + float(std), "unit": unit}


def _directional_correction(current: Optional[float], ideal: Optional[dict[str, object]]) -> Optional[str]:
    if current is None or ideal is None:
        return None
    lo = _coerce_float(ideal.get("min"))
    hi = _coerce_float(ideal.get("max"))
    unit = ideal.get("unit")
    if lo is None or hi is None:
        return None
    low, high = (lo, hi) if lo <= hi else (hi, lo)
    if current < low:
        return f"Increase toward {low:.2f}–{high:.2f} {unit}."
    if current > high:
        return f"Decrease toward {low:.2f}–{high:.2f} {unit}."
    return f"Keep within {low:.2f}–{high:.2f} {unit}."


def _severity_from_issue(issue: Mapping[str, object]) -> str:
    status = str(issue.get("status") or "").strip().lower()
    if status == "poor":
        return "major"
    z = _coerce_float(issue.get("z_score"))
    if z is not None and abs(float(z)) > 1.5:
        return "major"
    return "minor"


def _phase_ranges_from_throw_metrics(throw_metrics: Mapping[str, object]) -> Optional[dict[str, tuple[int, int]]]:
    phase_durations = throw_metrics.get("phase_durations")
    if not isinstance(phase_durations, Mapping):
        return None
    n_frames = phase_durations.get("frames")
    approach = phase_durations.get("approach_start_frame")
    delivery = phase_durations.get("delivery_start_frame")
    release = phase_durations.get("release_frame")
    try:
        total = int(n_frames)
        a = int(approach)
        d = int(delivery)
        r = int(release)
    except (TypeError, ValueError):
        return None
    if total <= 0:
        return None
    a = max(0, min(a, total - 1))
    d = max(a, min(d, total - 1))
    r = max(d, min(r, total - 1))
    return {
        "approach": (a, max(a, d)),
        "delivery": (d, max(d, r)),
        "release": (r, total - 1),
        "overall": (0, total - 1),
    }


def _frame_ranges_for_metric(
    metric_name: str,
    issue_phase: str | None,
    throw_metrics: Mapping[str, object],
    *,
    window_frames: int = 5,
) -> list[dict[str, object]]:
    ranges = _phase_ranges_from_throw_metrics(throw_metrics)
    if not ranges:
        return []

    phase = (issue_phase or "").strip().lower()
    if not phase and isinstance(metric_name, str) and "." in metric_name:
        phase = metric_name.split(".", 1)[0].strip().lower()

    if phase not in ranges:
        phase = "overall"

    start, end = ranges[phase]
    out: list[dict[str, object]] = [{"phase": phase, "start_frame": int(start), "end_frame": int(end), "label": f"{phase} phase"}]

    # Highlight the release window for release-phase issues (or at-release metrics).
    release_frame = None
    rel = throw_metrics.get("release_metrics")
    if isinstance(rel, Mapping):
        rf = rel.get("release_frame")
        try:
            release_frame = int(rf) if rf is not None else None
        except (TypeError, ValueError):
            release_frame = None

    if release_frame is None:
        pd = throw_metrics.get("phase_durations")
        if isinstance(pd, Mapping) and pd.get("release_frame") is not None:
            try:
                release_frame = int(pd.get("release_frame"))
            except (TypeError, ValueError):
                release_frame = None

    if release_frame is not None:
        overall_start, overall_end = ranges["overall"]
        w = max(1, int(window_frames))
        w_start = max(overall_start, int(release_frame) - w)
        w_end = min(overall_end, int(release_frame) + w)
        if phase == "release" or "at_release" in (metric_name or ""):
            out.insert(0, {"phase": "release", "start_frame": w_start, "end_frame": w_end, "label": "release window"})

    return out


def _metric_family(metric_name: str) -> str:
    name = (metric_name or "").lower()
    if "elbow_flexion" in name:
        return "elbow_extension"
    if "wrist_speed" in name:
        return "wrist_speed"
    if "power_chain" in name or "lag" in name:
        return "power_chain"
    if "shoulder_hip_separation" in name:
        return "separation"
    if "hip_speed" in name:
        return "hip_speed"
    if "trunk_lean" in name:
        return "posture"
    if "asymmetry" in name:
        return "symmetry"
    if "duration_ms" in name:
        return "timing"
    return "general"


def _suggestions(metric_name: str, *, direction_hint: Optional[str]) -> tuple[str, Optional[str], str]:
    """Return (actionable_cue, drill_suggestion, why_it_matters)."""
    family = _metric_family(metric_name)
    why = get_metric_description(metric_name) or "This metric influences throw efficiency and consistency."

    if family == "elbow_extension":
        cue = "Work toward a longer lever at release: extend the throwing elbow through the strike."
        drill = "Cue: 'reach long'. Drills: short-approach strikes, towel/ballistic band whip, triceps band press-downs."
        return cue, drill, why
    if family == "wrist_speed":
        cue = "Increase speed through delivery and keep the hand fast into release (avoid decelerating early)."
        drill = "Drills: towel snaps, elastic-band 'whip' drills, 3–5 step throws focusing on a fast strike."
        return cue, drill, why
    if family == "power_chain":
        cue = "Improve sequencing: hips lead, then shoulders, then the arm/hand (avoid an early arm pull)."
        drill = "Drills: step-behind throws, medicine-ball hip-to-shoulder throws, separation holds → strike."
        return cue, drill, why
    if family == "separation":
        cue = "Create and hold hip–shoulder separation into delivery; keep hips ahead of shoulders before the strike."
        drill = "Drills: med-ball rotational throws, cross-step separation drills, 90–90 shoulder mobility work."
        return cue, drill, why
    if family == "hip_speed":
        cue = "Drive the hips more aggressively through delivery while staying tall and stable."
        drill = "Drills: run-up rhythm work, hip-drive medicine ball throws, sprint/plyo acceleration drills."
        return cue, drill, why
    if family == "posture":
        cue = "Maintain posture through delivery; avoid collapsing the trunk as you enter the strike."
        drill = "Drills: wall posture drill, tall-throw cues, single-leg stability with trunk control."
        return cue, drill, why
    if family == "symmetry":
        cue = "Reduce left/right collapse; keep shoulders and pelvis stable to transfer force cleanly."
        drill = "Drills: split-stance throws, single-leg stability, mirror work for level shoulders."
        return cue, drill, why
    if family == "timing":
        cue = "Tune phase timing for rhythm: avoid rushing delivery or stalling into the block."
        drill = "Drills: run-up rhythm (clap/count), reduced-approach throws, metronome cadence work."
        return cue, drill, why

    cue = "Bring this metric closer to the elite range while keeping the movement smooth and repeatable."
    if direction_hint:
        cue = f"{cue} {direction_hint}"
    return cue, None, why


def _extract_rule_resolved_metrics(detail: Any) -> set[str]:
    if not isinstance(detail, Mapping):
        return set()
    kind = str(detail.get("type") or "")
    if kind == "atomic":
        resolved = detail.get("resolved_metric")
        return {str(resolved)} if resolved else set()
    if kind in {"all", "any"}:
        key = "all" if kind == "all" else "any"
        out: set[str] = set()
        items = detail.get(key, [])
        if isinstance(items, list):
            for item in items:
                out |= _extract_rule_resolved_metrics(item)
        return out
    if kind == "not":
        return _extract_rule_resolved_metrics(detail.get("not"))
    return set()


def _format_numeric(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    abs_v = abs(value)
    if abs_v >= 100:
        return f"{value:.0f}"
    if abs_v >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _compact_range_text(ideal: Optional[dict[str, object]]) -> str:
    if not ideal:
        return "elite range: n/a"
    lo = _coerce_float(ideal.get("min"))
    hi = _coerce_float(ideal.get("max"))
    unit = ideal.get("unit") or ""
    if lo is None or hi is None:
        return "elite range: n/a"
    low, high = (lo, hi) if lo <= hi else (hi, lo)
    return f"elite range { _format_numeric(low) }–{ _format_numeric(high) } {unit}".strip()


def generate_feedback(
    comparison_report: Any,
    throw_metrics: Any | None,
    rules: Any | None,
) -> list[dict[str, object]]:
    """Generate ranked, UI-friendly coaching cues from a comparison report."""
    report = _load_jsonish(comparison_report)
    throw = _coerce_throw_metrics(throw_metrics)

    top_issues = report.get("top_issues", [])
    if not isinstance(top_issues, list) or not top_issues:
        return []

    # Select up to 5 issues (the report already ranks them by |z|).
    selected = [i for i in top_issues if isinstance(i, Mapping)]
    selected = selected[:5]

    # Build a metrics mapping for rule evaluation:
    # - include the metric keys themselves
    # - include the nested throw_metrics object for dotted-path lookup
    metrics_for_rules: dict[str, object] = {}
    for item in selected:
        metric_key = str(item.get("metric") or "").strip()
        if metric_key:
            metrics_for_rules[metric_key] = item.get("athlete_value")
    if throw:
        metrics_for_rules.setdefault("throw_metrics", throw)

    rule_matches: list[dict[str, object]] = []
    if rules is not None:
        cfg = load_rules_config(rules)
        rule_matches = evaluate_rules(metrics_for_rules, cfg)

    # Rank by impact proxy (|z| * weight).
    def impact(issue: Mapping[str, object]) -> float:
        z = _coerce_float(issue.get("z_score"))
        w = _coerce_float(issue.get("weight"))
        metric = str(issue.get("metric") or "")
        weight = float(w) if w is not None else float(get_metric_weight(metric))
        return float(abs(float(z)) * weight) if z is not None else 0.0

    ranked = sorted(selected, key=impact, reverse=True)

    cues: list[dict[str, object]] = []
    for idx, issue in enumerate(ranked, start=1):
        metric_name = str(issue.get("metric") or "").strip()
        phase = str(issue.get("phase") or "").strip().lower() or None
        unit = issue.get("unit")

        current = _coerce_float(issue.get("athlete_value"))
        mean = _coerce_float(issue.get("elite_mean"))
        std = _coerce_float(issue.get("elite_std"))
        z = _coerce_float(issue.get("z_score"))

        ideal = _ideal_range(mean, std, unit)
        correction = _directional_correction(current, ideal)

        base_cue, drill, why = _suggestions(metric_name, direction_hint=correction)

        # If a rule matched the same resolved metric, prefer the rule's cue text.
        rule_text: Optional[str] = None
        for match in rule_matches:
            resolved = _extract_rule_resolved_metrics(match.get("condition_matched"))
            if metric_name in resolved:
                rule_text = str(match.get("feedback_text") or "").strip() or None
                break

        feedback_text = rule_text or base_cue
        # Add lightweight numeric context (UI-friendly).
        if metric_name:
            numeric_bits = []
            if current is not None:
                numeric_bits.append(f"current {_format_numeric(current)} {unit}".strip())
            numeric_bits.append(_compact_range_text(ideal))
            if numeric_bits:
                feedback_text = f"{feedback_text} ({'; '.join(numeric_bits)})"

        cues.append(
            {
                "issue_rank": int(idx),
                "metric_name": metric_name,
                "phase": phase,
                "severity": _severity_from_issue(issue),
                "feedback_text": feedback_text,
                "why_it_matters": why,
                "ideal_range": ideal,
                "current_value": current,
                "suggested_correction": correction,
                "drill_suggestion": drill,
                "frame_ranges": _frame_ranges_for_metric(metric_name, phase, throw, window_frames=5) if throw else [],
                "impact_score": impact(issue),
                "z_score": z,
            }
        )

    return cues


__all__ = ["generate_feedback"]

