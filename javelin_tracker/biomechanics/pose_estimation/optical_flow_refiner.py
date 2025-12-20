"""Kinovea-style landmark refinement using optical flow.

MediaPipe pose estimation can "snap" an occluded limb (e.g., wrist behind head)
to an unrelated foreground pixel with high confidence. Kinovea mitigates this
in practice by tracking points across frames via optical flow.

This module provides an optional, lightweight refinement pass that:
- Tracks landmark 2D positions with Lucas–Kanade optical flow.
- Corrects left/right symmetric swaps based on temporal continuity.
- Replaces implausible per-frame jumps with the optical-flow prediction.

The refiner is intentionally conservative: it only intervenes when the current
pose measurement is low-confidence or far from the optical-flow prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np

from javelin_tracker.env import get_env


LandmarkTuple = Tuple[float, float, float, float]  # (x, y, z, confidence) normalized xy


@dataclass
class OpticalFlowRefineStats:
    """Simple counters to help debug refinement behavior."""

    frames: int = 0
    swaps: int = 0
    replacements: int = 0
    flow_failures: int = 0


def _as_pixel_xy(landmark: LandmarkTuple, *, width: int, height: int) -> Tuple[float, float] | None:
    x, y, _z, conf = landmark
    if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(conf)):
        return None
    if conf <= 0.0:
        return None
    # Support either normalized coords [0..1] or raw pixel coords.
    px = float(x) * float(width) if 0.0 <= float(x) <= 1.0 else float(x)
    py = float(y) * float(height) if 0.0 <= float(y) <= 1.0 else float(y)
    return (px, py)


def _to_normalized_xy(px: float, py: float, *, width: int, height: int) -> Tuple[float, float]:
    nx = float(px) / float(max(1, width))
    ny = float(py) / float(max(1, height))
    return (float(np.clip(nx, 0.0, 1.0)), float(np.clip(ny, 0.0, 1.0)))


def _swap_landmarks(landmarks: List[LandmarkTuple], idx_a: int, idx_b: int) -> None:
    landmarks[idx_a], landmarks[idx_b] = landmarks[idx_b], landmarks[idx_a]


def _pair_cost_px(
    current: List[LandmarkTuple],
    previous: List[LandmarkTuple],
    idx_a: int,
    idx_b: int,
    *,
    width: int,
    height: int,
) -> Tuple[float, float] | None:
    cur_a = _as_pixel_xy(current[idx_a], width=width, height=height)
    cur_b = _as_pixel_xy(current[idx_b], width=width, height=height)
    prev_a = _as_pixel_xy(previous[idx_a], width=width, height=height)
    prev_b = _as_pixel_xy(previous[idx_b], width=width, height=height)
    if cur_a is None or cur_b is None or prev_a is None or prev_b is None:
        return None
    cur_a = np.array(cur_a, dtype=float)
    cur_b = np.array(cur_b, dtype=float)
    prev_a = np.array(prev_a, dtype=float)
    prev_b = np.array(prev_b, dtype=float)

    no_swap = float(np.linalg.norm(cur_a - prev_a) + np.linalg.norm(cur_b - prev_b))
    swapped = float(np.linalg.norm(cur_a - prev_b) + np.linalg.norm(cur_b - prev_a))
    return (no_swap, swapped)


def correct_symmetric_swaps(
    landmarks: List[LandmarkTuple],
    previous_landmarks: List[LandmarkTuple],
    *,
    width: int,
    height: int,
    pairs: Sequence[tuple[int, int]] | None = None,
    min_improvement_px: float = 18.0,
) -> int:
    """Correct left/right swaps for common symmetric landmark pairs.

    Returns:
        Number of swaps applied.
    """
    if len(landmarks) != 33 or len(previous_landmarks) != 33:
        return 0

    default_pairs: Sequence[tuple[int, int]] = (
        (11, 12),  # shoulders
        (13, 14),  # elbows
        (15, 16),  # wrists
        (23, 24),  # hips
        (25, 26),  # knees
        (27, 28),  # ankles
    )
    use_pairs = pairs or default_pairs

    swaps = 0
    threshold = float(min_improvement_px)
    for idx_a, idx_b in use_pairs:
        costs = _pair_cost_px(landmarks, previous_landmarks, idx_a, idx_b, width=width, height=height)
        if costs is None:
            continue
        no_swap, swapped = costs
        # Apply swap only if it is meaningfully better (avoid flapping).
        if swapped + threshold < no_swap:
            _swap_landmarks(landmarks, idx_a, idx_b)
            swaps += 1
    return swaps


class OpticalFlowLandmarkRefiner:
    """Refine landmark trajectories using per-frame optical flow."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        confidence_threshold: float = 0.5,
        replace_distance_fraction: float = 0.08,
        max_flow_error: float = 25.0,
        max_points: int = 33,
    ) -> None:
        self.enabled = bool(enabled)
        self.confidence_threshold = float(confidence_threshold)
        self.replace_distance_fraction = float(replace_distance_fraction)
        self.max_flow_error = float(max_flow_error)
        self.max_points = int(max_points)

        self._prev_gray: np.ndarray | None = None
        self._prev_landmarks: List[LandmarkTuple] | None = None
        self.stats = OpticalFlowRefineStats()

    def reset(self) -> None:
        self._prev_gray = None
        self._prev_landmarks = None
        self.stats = OpticalFlowRefineStats()

    def _replace_threshold_px(self, width: int, height: int) -> float:
        return float(max(8.0, self.replace_distance_fraction * float(min(width, height))))

    @staticmethod
    def _high_conf_replace_multiplier(idx: int) -> float:
        raw = get_env("POSE_OPTICAL_FLOW_HIGH_CONF_MULTIPLIER")
        if raw:
            try:
                return float(max(1.0, float(raw)))
            except Exception:
                pass
        # More aggressive for distal joints that commonly snap under occlusion.
        if int(idx) in {13, 14, 15, 16, 27, 28}:
            return 1.75
        return 2.25

    def refine(
        self,
        frame_bgr: np.ndarray,
        landmarks: List[LandmarkTuple],
        *,
        person_bbox: tuple[int, int, int, int] | None = None,
    ) -> List[LandmarkTuple]:
        """Return refined landmarks for the given frame.

        The returned list is always length 33. The input list is not mutated.
        """
        if not self.enabled:
            return landmarks
        if frame_bgr is None or not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
            return landmarks
        if len(landmarks) != 33:
            return landmarks

        height, width = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
        if height <= 2 or width <= 2:
            return landmarks

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        current = list(landmarks)
        self.stats.frames += 1

        gate_bbox: tuple[float, float, float, float] | None = None
        if person_bbox is not None:
            try:
                x0, y0, x1, y1 = (float(person_bbox[0]), float(person_bbox[1]), float(person_bbox[2]), float(person_bbox[3]))
            except Exception:
                x0 = y0 = x1 = y1 = 0.0
            if x1 > x0 + 2 and y1 > y0 + 2:
                pad = 0.12
                pad_x = (x1 - x0) * pad
                pad_y = (y1 - y0) * pad
                gx0 = max(-10.0, x0 - pad_x)
                gy0 = max(-10.0, y0 - pad_y)
                gx1 = min(float(width) + 10.0, x1 + pad_x)
                gy1 = min(float(height) + 10.0, y1 + pad_y)
                gate_bbox = (gx0, gy0, gx1, gy1)

        if self._prev_gray is None or self._prev_landmarks is None or self._prev_gray.shape != gray.shape:
            self._prev_gray = gray
            self._prev_landmarks = current
            return current

        # 1) Fix left/right swaps based on continuity before flow gating.
        swaps = correct_symmetric_swaps(current, self._prev_landmarks, width=width, height=height)
        self.stats.swaps += int(swaps)

        # 2) Track previous landmarks forward with optical flow.
        prev_points: list[tuple[float, float]] = []
        indices: list[int] = []
        for idx, lm in enumerate(self._prev_landmarks[: self.max_points]):
            xy = _as_pixel_xy(lm, width=width, height=height)
            if xy is None:
                continue
            prev_points.append(xy)
            indices.append(idx)

        if not prev_points:
            self.stats.flow_failures += 1
            self._prev_gray = gray
            self._prev_landmarks = current
            return current

        prev_pts = np.array(prev_points, dtype=np.float32).reshape(-1, 1, 2)
        next_pts, status, errors = cv2.calcOpticalFlowPyrLK(
            self._prev_gray,
            gray,
            prev_pts,
            None,
            # Larger window/pyramid improves robustness when the athlete moves quickly
            # (at the cost of a bit more CPU; we track only up to 33 points).
            winSize=(41, 41),
            maxLevel=5,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 35, 0.01),
        )

        if next_pts is None or status is None:
            self.stats.flow_failures += 1
            self._prev_gray = gray
            self._prev_landmarks = current
            return current

        replace_threshold_px = self._replace_threshold_px(width, height)
        max_err = float(self.max_flow_error)
        conf_thresh = float(self.confidence_threshold)

        status = status.reshape(-1)
        if errors is None:
            errors = np.zeros((len(status),), dtype=np.float32)
        errors = np.asarray(errors, dtype=np.float32).reshape(-1)

        for j, idx in enumerate(indices):
            if int(status[j]) != 1:
                continue
            err = float(errors[j]) if j < errors.shape[0] else 0.0
            if np.isfinite(err) and err > max_err:
                continue

            pred_px = float(next_pts[j, 0, 0])
            pred_py = float(next_pts[j, 0, 1])
            if not (np.isfinite(pred_px) and np.isfinite(pred_py)):
                continue
            if pred_px < -5 or pred_px > width + 5 or pred_py < -5 or pred_py > height + 5:
                continue
            if gate_bbox is not None:
                gx0, gy0, gx1, gy1 = gate_bbox
                if not (gx0 <= pred_px <= gx1 and gy0 <= pred_py <= gy1):
                    continue

            pose_x, pose_y, pose_z, pose_c = current[idx]
            pose_xy = _as_pixel_xy((pose_x, pose_y, pose_z, pose_c), width=width, height=height)
            prev_z = float(self._prev_landmarks[idx][2])
            prev_c = float(self._prev_landmarks[idx][3])

            should_replace = False
            if pose_xy is None:
                should_replace = True
            elif gate_bbox is not None:
                gx0, gy0, gx1, gy1 = gate_bbox
                if not (gx0 <= float(pose_xy[0]) <= gx1 and gy0 <= float(pose_xy[1]) <= gy1):
                    # If the current measurement is outside the athlete region, prefer
                    # the flow-predicted point (prevents latching onto the background).
                    should_replace = True
            elif float(pose_c) < conf_thresh:
                should_replace = True
            else:
                dx = float(pose_xy[0]) - pred_px
                dy = float(pose_xy[1]) - pred_py
                dist = float(np.hypot(dx, dy))
                if dist <= replace_threshold_px:
                    should_replace = False
                elif float(pose_c) >= 0.75:
                    # Trust high-confidence pose measurements unless they are wildly
                    # inconsistent with temporal continuity (common during occlusions).
                    should_replace = dist > (replace_threshold_px * self._high_conf_replace_multiplier(idx))
                else:
                    should_replace = True

            if not should_replace:
                continue

            nx, ny = _to_normalized_xy(pred_px, pred_py, width=width, height=height)
            use_z = float(pose_z) if np.isfinite(pose_z) and float(pose_c) >= 0.1 else prev_z
            # Mark as "usable but not perfect": slightly above the validity threshold.
            fallback_conf = max(conf_thresh + 0.05, min(0.85, max(float(pose_c), prev_c * 0.85)))
            current[idx] = (float(nx), float(ny), float(use_z), float(fallback_conf))
            self.stats.replacements += 1

        self._prev_gray = gray
        self._prev_landmarks = current
        return current


def iter_refined_landmarks(
    frames_bgr: Iterable[np.ndarray],
    landmarks_per_frame: Iterable[List[LandmarkTuple]],
    *,
    enabled: bool = True,
    confidence_threshold: float = 0.5,
) -> Tuple[List[List[LandmarkTuple]], OpticalFlowRefineStats]:
    """Convenience helper to refine an entire sequence (used by tests)."""
    refiner = OpticalFlowLandmarkRefiner(
        enabled=enabled,
        confidence_threshold=confidence_threshold,
    )
    refined: List[List[LandmarkTuple]] = []
    for frame, landmarks in zip(frames_bgr, landmarks_per_frame):
        refined.append(refiner.refine(frame, landmarks))
    return refined, refiner.stats
