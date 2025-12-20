"""Feedback rule engine for coach-facing biomechanics cues.

Rules are evaluated against an athlete's computed metrics and return a list of
matched coaching cues. Rules can be provided as Python objects (dict/list) or
loaded from a JSON file (default: `config/feedback_rules.json` at project root).

Rule shape (JSON / dict):
    {
      "rule_id": "elbow_extension_release",
      "priority": 90,
      "severity": "warning",
      "feedback_text": "Extend your elbow fully at release.",
      "condition": {"metric": "elbow_angle_at_release", "op": "<", "value": 85, "unit": "degrees"}
    }

Condition operators:
  - Comparisons: >, <, >=, <=, ==, !=
  - Range checks: "range" / "between" / "in_range" (inclusive)
  - Logic: {"all": [..]} (AND), {"any": [..]} (OR), {"not": {...}}

Metric lookup:
  - If the exact `metric` key exists in the provided `athlete_metrics` mapping,
    that value is used.
  - Otherwise, `metric` is treated as a dotted path into nested dicts
    (e.g. `throw_metrics.power_chain_lag_ms`).
  - Optional metric aliases can map coach-friendly names to canonical keys.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Tuple

DEFAULT_RULES_PATH = Path(__file__).resolve().parents[3] / "config" / "feedback_rules.json"

_SEVERITY_RANK = {"info": 0, "warning": 1, "critical": 2}


DEFAULT_RULES_CONFIG: dict[str, object] = {
    "version": 1,
    "metric_aliases": {
        # Coach-friendly aliases -> metric keys/paths used by this project.
        "elbow_angle_at_release": [
            # Phase-prefixed metric (from elite_reference.compute_throw_metrics()).
            "release.throwing_elbow_flexion_deg_at_release",
        ],
        "power_chain_lag_ms": [
            # MetricsPipeline output path (from metrics/throw_metrics.py).
            "throw_metrics.power_chain_lag_ms",
            # Sometimes coaches may pass a flattened dict already.
            "power_chain_lag_ms",
        ],
    },
    "rules": [
        {
            "rule_id": "elbow_extension_release",
            "priority": 90,
            "severity": "warning",
            "feedback_text": "Extend your elbow fully at release.",
            "condition": {"metric": "elbow_angle_at_release", "op": "<", "value": 85, "unit": "degrees"},
        },
        {
            "rule_id": "power_chain_lag_slow",
            "priority": 80,
            "severity": "warning",
            "feedback_text": "Delay arm acceleration; let hips drive first.",
            "condition": {"metric": "power_chain_lag_ms", "op": ">", "value": 100, "unit": "ms"},
        },
    ],
    "templates": [
        {
            "rule_id": "template_low_release_speed",
            "priority": 50,
            "severity": "warning",
            "feedback_text": "Increase speed through delivery; aim for a faster wrist at release.",
            "condition": {"metric": "release.throwing_wrist_speed_at_release", "op": "<", "value": 0.0, "unit": "rel_units/s"},
        },
        {
            "rule_id": "template_low_shoulder_hip_separation",
            "priority": 50,
            "severity": "warning",
            "feedback_text": "Create more hip–shoulder separation before the strike; keep hips leading shoulders.",
            "condition": {
                "metric": "delivery.shoulder_hip_separation_deg_mean",
                "op": "<",
                "value": 0.0,
                "unit": "degrees",
            },
        },
        {
            "rule_id": "template_release_timing_range",
            "priority": 40,
            "severity": "info",
            "feedback_text": "Keep phase timing consistent; avoid rushing the delivery.",
            "condition": {"metric": "delivery.duration_ms", "op": "range", "min": 0.0, "max": 0.0, "unit": "ms"},
        },
        {
            "rule_id": "template_excessive_asymmetry",
            "priority": 40,
            "severity": "warning",
            "feedback_text": "Reduce left/right asymmetry; keep shoulders level and balanced through the throw.",
            "condition": {"metric": "release.shoulder_height_asymmetry_mean", "op": ">", "value": 0.0, "unit": "rel_units"},
        },
    ],
}


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


def _normalize_aliases(raw: Any) -> dict[str, list[str]]:
    aliases: dict[str, list[str]] = {}
    if not isinstance(raw, Mapping):
        return aliases
    for key, value in raw.items():
        if not key:
            continue
        if isinstance(value, str):
            aliases[str(key)] = [value]
        elif isinstance(value, Sequence):
            out: list[str] = []
            for item in value:
                if isinstance(item, str) and item.strip():
                    out.append(item.strip())
            if out:
                aliases[str(key)] = out
    return aliases


def load_rules_config(rules_config: Any | None = None) -> dict[str, object]:
    """Load/normalize rules config.

    Accepted inputs:
      - None: loads from DEFAULT_RULES_PATH if present; otherwise uses built-in defaults.
      - Path/str: JSON file path.
      - dict: either a full config (with "rules") or a single rule object.
      - list: a list of rule objects.
    """
    if rules_config is None:
        if DEFAULT_RULES_PATH.exists():
            try:
                rules_config = DEFAULT_RULES_PATH
            except Exception:
                rules_config = None
        else:
            return dict(DEFAULT_RULES_CONFIG)

    if isinstance(rules_config, (str, Path)):
        loaded = _load_jsonish(rules_config)
        if "rules" not in loaded:
            raise ValueError("Rules config JSON must include a top-level 'rules' list.")
        return loaded

    if isinstance(rules_config, list):
        return {"version": 1, "metric_aliases": {}, "rules": list(rules_config)}

    if isinstance(rules_config, Mapping):
        cfg = dict(rules_config)
        if "rules" in cfg:
            return cfg
        if "rule_id" in cfg and "condition" in cfg:
            return {"version": 1, "metric_aliases": {}, "rules": [cfg]}
        raise ValueError("Unsupported rules_config mapping; expected {'rules': [...]} or a single rule object.")

    raise ValueError("Unsupported rules_config type; expected None, path, dict, or list.")


def _lookup_metric(metrics: Mapping[str, Any], metric_key: str) -> Any:
    if metric_key in metrics:
        return metrics[metric_key]

    # Fallback: dotted-path into nested dicts.
    current: Any = metrics
    for part in (metric_key or "").split("."):
        if not part:
            return None
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _resolve_metric_candidates(metric: str, aliases: Mapping[str, Sequence[str]]) -> list[str]:
    metric = str(metric or "").strip()
    if not metric:
        return []
    candidates = [metric]
    aliased = aliases.get(metric)
    if isinstance(aliased, str):
        if aliased.strip():
            candidates.append(aliased.strip())
    elif isinstance(aliased, Sequence):
        for item in aliased:
            if isinstance(item, str) and item.strip():
                candidates.append(item.strip())
    # Preserve order, de-dup.
    out: list[str] = []
    seen: set[str] = set()
    for key in candidates:
        if key not in seen:
            out.append(key)
            seen.add(key)
    return out


def _coerce_range(condition: Mapping[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    if "value" in condition:
        value = condition.get("value")
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
            lo = _coerce_float(value[0])
            hi = _coerce_float(value[1])
            return lo, hi
    return _coerce_float(condition.get("min")), _coerce_float(condition.get("max"))


def _ideal_value(condition: Mapping[str, Any], op: str) -> object:
    for key in ("ideal_value", "ideal"):
        if key in condition:
            return condition.get(key)
    if op in {"range", "between", "in_range"}:
        lo, hi = _coerce_range(condition)
        return {"min": lo, "max": hi}
    return condition.get("value")


def _eval_atomic_condition(
    condition: Mapping[str, Any],
    metrics: Mapping[str, Any],
    *,
    aliases: Mapping[str, Sequence[str]],
) -> tuple[bool, dict[str, object], list[dict[str, object]]]:
    metric = str(condition.get("metric") or "").strip()
    op_raw = str(condition.get("op") or "").strip().lower()
    op = op_raw or "=="
    unit = condition.get("unit")

    candidates = _resolve_metric_candidates(metric, aliases)
    resolved_metric: Optional[str] = None
    observed_raw: Any = None
    observed: Optional[float] = None
    for candidate in candidates:
        value = _lookup_metric(metrics, candidate)
        num = _coerce_float(value)
        if num is None:
            continue
        resolved_metric = candidate
        observed_raw = value
        observed = num
        break

    # If we didn't find a numeric value, treat as not matched.
    if observed is None or resolved_metric is None:
        detail = {
            "type": "atomic",
            "metric": metric,
            "resolved_metric": resolved_metric,
            "op": op,
            "target": condition.get("value"),
            "unit": unit,
            "observed": None,
            "result": False,
            "error": "metric_missing_or_non_numeric",
        }
        return False, detail, []

    ideal = _ideal_value(condition, op)
    target = condition.get("value")
    target_num = _coerce_float(target)

    result = False
    try:
        if op in {">", "gt"}:
            result = bool(target_num is not None and observed > target_num)
        elif op in {"<", "lt"}:
            result = bool(target_num is not None and observed < target_num)
        elif op in {">=", "gte"}:
            result = bool(target_num is not None and observed >= target_num)
        elif op in {"<=", "lte"}:
            result = bool(target_num is not None and observed <= target_num)
        elif op in {"==", "eq"}:
            result = bool(target_num is not None and observed == target_num)
        elif op in {"!=", "ne"}:
            result = bool(target_num is not None and observed != target_num)
        elif op in {"range", "between", "in_range"}:
            lo, hi = _coerce_range(condition)
            if lo is None or hi is None:
                result = False
            else:
                low, high = (lo, hi) if lo <= hi else (hi, lo)
                result = bool(low <= observed <= high)
        else:
            result = False
    except Exception:
        result = False

    detail = {
        "type": "atomic",
        "metric": metric,
        "resolved_metric": resolved_metric,
        "op": op,
        "target": condition.get("value") if op not in {"range", "between", "in_range"} else {"min": condition.get("min"), "max": condition.get("max")},
        "unit": unit,
        "observed": observed,
        "observed_raw": observed_raw,
        "ideal_value": ideal,
        "result": bool(result),
    }

    atomic = [
        {
            "metric": metric,
            "resolved_metric": resolved_metric,
            "observed": observed,
            "ideal_value": ideal,
            "unit": unit,
        }
    ]
    return bool(result), detail, atomic


def _eval_condition(
    condition: Any,
    metrics: Mapping[str, Any],
    *,
    aliases: Mapping[str, Sequence[str]],
) -> tuple[bool, dict[str, object], list[dict[str, object]]]:
    if not isinstance(condition, Mapping):
        return False, {"type": "invalid", "result": False, "error": "condition_not_a_mapping"}, []

    if "all" in condition:
        raw = condition.get("all")
        if not isinstance(raw, Sequence):
            return False, {"type": "all", "result": False, "error": "all_must_be_a_list"}, []
        details: list[dict[str, object]] = []
        atoms: list[dict[str, object]] = []
        results: list[bool] = []
        for item in raw:
            ok, detail, atomic = _eval_condition(item, metrics, aliases=aliases)
            details.append(detail)
            atoms.extend(atomic)
            results.append(bool(ok))
        out = bool(all(results)) if results else False
        return out, {"type": "all", "all": details, "result": out}, atoms

    if "any" in condition:
        raw = condition.get("any")
        if not isinstance(raw, Sequence):
            return False, {"type": "any", "result": False, "error": "any_must_be_a_list"}, []
        details = []
        atoms: list[dict[str, object]] = []
        results: list[bool] = []
        for item in raw:
            ok, detail, atomic = _eval_condition(item, metrics, aliases=aliases)
            details.append(detail)
            atoms.extend(atomic)
            results.append(bool(ok))
        out = bool(any(results)) if results else False
        return out, {"type": "any", "any": details, "result": out}, atoms

    if "not" in condition:
        ok, detail, atomic = _eval_condition(condition.get("not"), metrics, aliases=aliases)
        out = not bool(ok)
        return out, {"type": "not", "not": detail, "result": out}, atomic

    if "metric" in condition:
        return _eval_atomic_condition(condition, metrics, aliases=aliases)

    return False, {"type": "invalid", "result": False, "error": "unknown_condition_shape"}, []


def evaluate_rules(athlete_metrics: Any, rules_config: Any | None = None) -> list[dict[str, object]]:
    """Evaluate rules against athlete metrics and return matched feedback entries."""
    metrics = _load_jsonish(athlete_metrics)
    cfg = load_rules_config(rules_config)

    aliases = _normalize_aliases(cfg.get("metric_aliases"))
    rules_raw = cfg.get("rules", [])
    if not isinstance(rules_raw, list):
        raise ValueError("rules_config['rules'] must be a list.")

    matches: list[dict[str, object]] = []
    for rule in rules_raw:
        if not isinstance(rule, Mapping):
            continue
        rule_id = str(rule.get("rule_id") or "").strip()
        if not rule_id:
            continue
        feedback_text = str(rule.get("feedback_text") or rule.get("text") or "").strip()
        if not feedback_text:
            continue
        condition = rule.get("condition")
        if condition is None:
            continue

        ok, detail, atomic = _eval_condition(condition, metrics, aliases=aliases)
        if not ok:
            continue

        # Determine summary metric_value/ideal_value for the rule.
        metric_value: object = None
        ideal_value: object = None
        if len(atomic) == 1:
            metric_value = atomic[0].get("observed")
            ideal_value = atomic[0].get("ideal_value")
        elif atomic:
            metric_value = {a.get("resolved_metric") or a.get("metric"): a.get("observed") for a in atomic}
            ideal_value = {a.get("resolved_metric") or a.get("metric"): a.get("ideal_value") for a in atomic}

        priority = int(rule.get("priority", 0)) if isinstance(rule.get("priority"), (int, float, str)) else 0
        severity = str(rule.get("severity") or "info").strip().lower() or "info"

        matches.append(
            {
                "rule_id": rule_id,
                "feedback_text": feedback_text,
                "condition_matched": detail,
                "metric_value": metric_value,
                "ideal_value": ideal_value,
                "priority": priority,
                "severity": severity,
            }
        )

    matches.sort(
        key=lambda m: (
            -int(m.get("priority") or 0),
            -_SEVERITY_RANK.get(str(m.get("severity") or "info"), 0),
            str(m.get("rule_id") or ""),
        )
    )
    return matches


__all__ = ["DEFAULT_RULES_PATH", "DEFAULT_RULES_CONFIG", "load_rules_config", "evaluate_rules"]

