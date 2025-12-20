"""Tools for comparing biomechanics metrics across sessions or athletes.

Currently includes Dynamic Time Warping (DTW) utilities for comparing
variable-length angle trajectories.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "compute_dtw_distance",
    "dtw_with_path",
    "compare_throw_to_elite",
    "plot_aligned_sequences",
    "score_metric",
    "score_all_metrics",
    "infer_athlete_style",
    "resolve_style_reference",
    "get_metric_description",
    "get_metric_weight",
    "generate_comparison_report",
]

_DTW_EXPORTS = {
    "compute_dtw_distance",
    "dtw_with_path",
    "compare_throw_to_elite",
    "plot_aligned_sequences",
}

_SCORING_EXPORTS = {
    "score_metric",
    "score_all_metrics",
    "infer_athlete_style",
    "resolve_style_reference",
    "get_metric_description",
    "get_metric_weight",
}
_REPORT_EXPORTS = {"generate_comparison_report"}


def __getattr__(name: str) -> Any:  # pragma: no cover - exercised indirectly
    if name in _DTW_EXPORTS:
        from . import dtw as _dtw

        return getattr(_dtw, name)
    if name in _SCORING_EXPORTS:
        from . import scoring as _scoring

        return getattr(_scoring, name)
    if name in _REPORT_EXPORTS:
        from . import reporter as _reporter

        return getattr(_reporter, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover - trivial
    return sorted(set(list(globals()) + __all__))
