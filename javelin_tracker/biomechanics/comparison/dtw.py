"""Dynamic Time Warping (DTW) utilities for comparing throw biomechanics.

This module focuses on comparing *time series* that may have different lengths
by using Dynamic Time Warping. Typical inputs are joint-angle trajectories
exported by the biomechanics metrics pipeline.

When available, `dtaidistance` is used for fast DTW computation; a small
NumPy-based fallback keeps the project functional in lightweight test/dev
environments where optional dependencies are not installed.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _as_array(sequence: Any) -> np.ndarray:
    arr = np.asarray(sequence, dtype=float)
    if arr.ndim == 0:
        raise ValueError("DTW sequence must be 1D or 2D.")
    if arr.ndim == 1:
        if arr.size == 0:
            raise ValueError("DTW sequence cannot be empty.")
        return arr
    if arr.ndim == 2:
        if arr.shape[0] == 0:
            raise ValueError("DTW sequence cannot be empty.")
        return arr
    raise ValueError("DTW sequence must be 1D or 2D.")


def _z_normalize(arr: np.ndarray) -> np.ndarray:
    """Zero-mean / unit-variance normalization (per-sequence, per-feature)."""
    x = np.asarray(arr, dtype=float)
    if x.ndim == 1:
        mean = float(np.nanmean(x)) if np.isfinite(x).any() else 0.0
        std = float(np.nanstd(x)) if np.isfinite(x).any() else 1.0
        if not math.isfinite(std) or std <= 1e-12:
            std = 1.0
        out = (x - mean) / std
    else:
        mean = np.nanmean(x, axis=0)
        std = np.nanstd(x, axis=0)
        std = np.where(np.isfinite(std) & (std > 1e-12), std, 1.0)
        out = (x - mean) / std
    out = np.where(np.isfinite(out), out, 0.0)
    return out


def _frame_distance(a: np.ndarray, b: np.ndarray, metric: str) -> float:
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if metric == "euclidean":
        return float(np.linalg.norm(diff))
    if metric in {"manhattan", "l1"}:
        return float(np.sum(np.abs(diff)))
    raise ValueError("metric must be 'euclidean' or 'manhattan'")


def _dtw_fallback_distance_and_path(
    seq_a: np.ndarray, seq_b: np.ndarray, *, metric: str, return_path: bool
) -> tuple[float, list[tuple[int, int]]]:
    a = np.asarray(seq_a, dtype=float)
    b = np.asarray(seq_b, dtype=float)

    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if b.ndim == 1:
        b = b.reshape(-1, 1)

    n, m = int(a.shape[0]), int(b.shape[0])
    dp = np.full((n + 1, m + 1), float("inf"), dtype=float)
    dp[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = _frame_distance(a[i - 1], b[j - 1], metric)
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])

    dist = float(dp[n, m])

    if not return_path:
        return dist, []

    path: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 or j > 0:
        path.append((max(0, i - 1), max(0, j - 1)))
        if i == 0:
            j -= 1
            continue
        if j == 0:
            i -= 1
            continue
        # Choose predecessor with minimal cumulative cost.
        candidates = [
            (dp[i - 1, j - 1], i - 1, j - 1),
            (dp[i - 1, j], i - 1, j),
            (dp[i, j - 1], i, j - 1),
        ]
        _, i, j = min(candidates, key=lambda t: float(t[0]))

    path.reverse()
    return dist, path


def compute_dtw_distance(sequence_a: Any, sequence_b: Any, metric: str = "euclidean") -> float:
    """Compute DTW distance between two sequences (0=identical, higher=different).

    Sequences are normalized (z-score) before DTW is computed.
    """
    metric_norm = (metric or "euclidean").strip().lower()
    a = _z_normalize(_as_array(sequence_a))
    b = _z_normalize(_as_array(sequence_b))

    # Fast path: dtaidistance for 1D sequences (common case: one angle time series).
    if a.ndim == 1 and b.ndim == 1 and metric_norm == "euclidean":
        try:
            from dtaidistance import dtw as _dtw  # type: ignore

            return float(_dtw.distance(a, b))
        except Exception:
            pass

    dist, _ = _dtw_fallback_distance_and_path(a, b, metric=metric_norm, return_path=False)
    return float(dist)


def dtw_with_path(seq_a: Any, seq_b: Any, metric: str = "euclidean") -> tuple[float, list[tuple[int, int]]]:
    """Return DTW distance and alignment path.

    The path is a list of `(index_a, index_b)` pairs in alignment order.
    """
    metric_norm = (metric or "euclidean").strip().lower()
    a = _z_normalize(_as_array(seq_a))
    b = _z_normalize(_as_array(seq_b))

    if a.ndim == 1 and b.ndim == 1 and metric_norm == "euclidean":
        try:
            from dtaidistance import dtw as _dtw  # type: ignore

            dist = float(_dtw.distance(a, b))
            path = [(int(i), int(j)) for i, j in _dtw.warping_path(a, b)]
            return dist, path
        except Exception:
            pass

    dist, path = _dtw_fallback_distance_and_path(a, b, metric=metric_norm, return_path=True)
    return float(dist), path


def _extract_angle_dataframe(metrics: Any) -> pd.DataFrame:
    if isinstance(metrics, pd.DataFrame):
        return metrics

    if isinstance(metrics, Mapping):
        angles = metrics.get("angles")
        if isinstance(angles, Mapping) and "data" in angles:
            angles = angles.get("data")
        if isinstance(angles, pd.DataFrame):
            return angles
        if isinstance(angles, Sequence):
            return pd.DataFrame.from_records(list(angles))

        angles_df = metrics.get("angles_df")
        if isinstance(angles_df, pd.DataFrame):
            return angles_df

    raise ValueError("Unsupported metrics format; expected angles data as DataFrame or list of records.")


def _angle_sequences(metrics: Any) -> dict[str, np.ndarray]:
    df = _extract_angle_dataframe(metrics).copy()
    required = {"frame", "angle_name", "value_degrees"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"Angles data must include columns: {sorted(required)}")

    df["frame"] = pd.to_numeric(df["frame"], errors="coerce")
    df["value_degrees"] = pd.to_numeric(df["value_degrees"], errors="coerce")
    if "valid" in df.columns:
        valid = df["valid"].astype(bool)
        df.loc[~valid, "value_degrees"] = np.nan

    out: dict[str, np.ndarray] = {}
    for angle_name, group in df.groupby("angle_name"):
        group_sorted = group.sort_values("frame")
        values = group_sorted["value_degrees"].to_numpy(dtype=float)
        if values.size:
            out[str(angle_name)] = values
    return out


def _alignment_quality(path: Sequence[tuple[int, int]], len_a: int, len_b: int) -> float:
    if len(path) < 2 or len_a <= 0 or len_b <= 0:
        return 0.0
    diag_steps = 0
    total_steps = len(path) - 1
    for (i0, j0), (i1, j1) in zip(path, path[1:]):
        if (i1 - i0) == 1 and (j1 - j0) == 1:
            diag_steps += 1
    diag_ratio = diag_steps / max(1, total_steps)
    length_ratio = min(len_a, len_b) / max(len_a, len_b)
    return float(100.0 * diag_ratio * length_ratio)


def _normalized_distance(dtw_distance: float, *, path_len: int, length_scale: int) -> float:
    if not math.isfinite(float(dtw_distance)):
        return float("nan")
    denom = max(1, int(path_len) if path_len else int(length_scale))
    mean_cost = float(dtw_distance) / denom
    score = 100.0 * (1.0 - math.exp(-max(0.0, mean_cost)))
    return float(np.clip(score, 0.0, 100.0))


def compare_throw_to_elite(
    athlete_metrics: Any,
    elite_metrics: Any,
    *,
    metric: str = "euclidean",
    reference_angle: Optional[str] = None,
) -> dict[str, object]:
    """Compare an athlete throw vs an elite throw using DTW over angle time series.

    The comparison aggregates DTW distances across all angles that exist in both
    inputs. A representative angle is used to compute an alignment path and an
    alignment-quality score.
    """
    metric_norm = (metric or "euclidean").strip().lower()
    athlete = _angle_sequences(athlete_metrics)
    elite = _angle_sequences(elite_metrics)

    common = sorted(set(athlete.keys()) & set(elite.keys()))
    if not common:
        raise ValueError("No common angles found between athlete_metrics and elite_metrics.")

    ref = reference_angle or ("right_elbow" if "right_elbow" in common else common[0])
    if ref not in common:
        ref = common[0]

    distances: dict[str, float] = {}
    for angle_name in common:
        distances[angle_name] = compute_dtw_distance(athlete[angle_name], elite[angle_name], metric=metric_norm)

    finite = [d for d in distances.values() if math.isfinite(float(d))]
    dtw_distance = float(np.mean(finite)) if finite else float("nan")

    ref_dist, ref_path = dtw_with_path(athlete[ref], elite[ref], metric=metric_norm)
    quality = _alignment_quality(ref_path, len(athlete[ref]), len(elite[ref]))
    normalized = _normalized_distance(
        dtw_distance,
        path_len=len(ref_path),
        length_scale=max(len(athlete[ref]), len(elite[ref]), 1),
    )

    return {
        "dtw_distance": dtw_distance,
        "normalized_distance": normalized,
        "alignment_quality": quality,
        "reference_angle": ref,
        "per_angle_distances": distances,
        "reference_distance": ref_dist,
    }


def plot_aligned_sequences(
    seq_a: Any,
    seq_b: Any,
    *,
    path: Optional[Sequence[tuple[int, int]]] = None,
    metric: str = "euclidean",
    labels: tuple[str, str] = ("athlete", "elite"),
    title: str = "DTW alignment",
    max_features: int = 4,
):
    """Plot two sequences after DTW alignment (using the alignment path)."""
    import matplotlib

    try:
        matplotlib.use("Agg", force=False)
    except Exception:
        pass
    import matplotlib.pyplot as plt

    a = _z_normalize(_as_array(seq_a))
    b = _z_normalize(_as_array(seq_b))

    dist = None
    if path is None:
        dist, path = dtw_with_path(a, b, metric=metric)
    assert path is not None

    aligned_a = np.asarray([a[i] for i, _j in path], dtype=float)
    aligned_b = np.asarray([b[j] for _i, j in path], dtype=float)

    if aligned_a.ndim == 1:
        aligned_a = aligned_a.reshape(-1, 1)
    if aligned_b.ndim == 1:
        aligned_b = aligned_b.reshape(-1, 1)

    n_features = int(min(aligned_a.shape[1], aligned_b.shape[1], max_features))
    fig, axes = plt.subplots(n_features, 1, figsize=(10, 3 * n_features), sharex=True)
    if n_features == 1:
        axes = [axes]

    x = np.arange(len(path), dtype=int)
    for k in range(n_features):
        ax = axes[k]
        ax.plot(x, aligned_a[:, k], label=labels[0], linewidth=1.5)
        ax.plot(x, aligned_b[:, k], label=labels[1], linewidth=1.5, alpha=0.85)
        ax.set_ylabel(f"feature_{k}")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")

    subtitle = f"{title}"
    if dist is not None and math.isfinite(float(dist)):
        subtitle = f"{title} (dtw_distance={dist:.3f})"
    fig.suptitle(subtitle)
    axes[-1].set_xlabel("Alignment step")
    fig.tight_layout()
    return fig


__all__ = [
    "compute_dtw_distance",
    "dtw_with_path",
    "compare_throw_to_elite",
    "plot_aligned_sequences",
]

