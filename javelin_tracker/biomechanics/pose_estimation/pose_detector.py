"""Pose estimation utilities for javelin analysis.

This project supports two MediaPipe backends:
- **Solutions** API (legacy): `mediapipe.solutions.holistic.Holistic`
- **Tasks** API (newer wheels): `mediapipe.tasks.python.vision.PoseLandmarker`

Some recent MediaPipe wheels (notably certain cp312 macOS builds) ship only the
Tasks API and do not include `mediapipe.solutions.*`. The detector selects the
best available backend at runtime.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import mediapipe as mp
import numpy as np

from javelin_tracker.biomechanics import BIOMECHANICS_LOGGER
from javelin_tracker.env import get_env


logger = BIOMECHANICS_LOGGER

LandmarkTuple = Tuple[float, float, float, float]

_SEGMENTATION_CHANNEL_MISMATCH_WARNED = False
_SEGMENTATION_PY312_DISABLED_WARNED = False


class PoseDetector:
    """Wrapper around MediaPipe pose detection with cached model and thread safety.

    Thread-safety: internal access to the MediaPipe model is guarded by a lock
    so multiple threads can share one PoseDetector instance. Prefer reusing a
    single instance across threads instead of creating one per thread to reduce
    initialization overhead.
    """

    def __init__(self) -> None:
        self._backend = "unknown"
        self._pose_landmark = None
        self._holistic = None
        self._landmarker = None
        self._landmarker_running_mode = "IMAGE"
        self._prev_pose_center: tuple[float, float] | None = None

        backend_pref = (get_env("POSE_BACKEND") or "").strip().lower()
        if backend_pref in {"holistic", "solutions"}:
            if not self._has_solutions_backend():
                raise RuntimeError("Requested PoseDetector backend 'holistic' but mediapipe.solutions is unavailable.")
            self._init_holistic()
        elif backend_pref in {"pose_landmarker", "landmarker", "tasks"}:
            self._init_pose_landmarker()
        else:
            # Default: prefer Tasks PoseLandmarker (more robust temporal tracking + ROI support).
            if self._has_tasks_backend():
                self._init_pose_landmarker()
            elif self._has_solutions_backend():
                self._init_holistic()
            else:
                raise RuntimeError(
                    "No MediaPipe pose backend available. Install `mediapipe` (Tasks preferred) and `opencv-python`."
                )

        self._lock = threading.Lock()
        self._frame_counter = 0
        self._closed = False
        # When enabled, we crop frames around the tracked athlete to increase
        # effective resolution for the model (helpful when limbs occlude).
        self._roi: tuple[int, int, int, int] | None = None  # (x0, y0, x1, y1) in full-frame pixels

    @staticmethod
    def _has_solutions_backend() -> bool:
        try:
            solutions = getattr(mp, "solutions", None)
            return bool(solutions and getattr(solutions, "holistic", None))
        except Exception:
            return False

    @staticmethod
    def _has_tasks_backend() -> bool:
        try:
            from mediapipe.tasks.python.vision import PoseLandmarker  # noqa: F401

            return True
        except Exception:
            return False

    @staticmethod
    def _truthy(value: str | None, *, default: bool = False) -> bool:
        if value is None:
            return bool(default)
        return value.strip().lower() not in {"0", "false", "no", "off", ""}

    def _roi_enabled(self) -> bool:
        # Default ON for better tracking with consumer video clips.
        return self._truthy(get_env("POSE_ROI"), default=True)

    def _segmentation_enabled(self) -> bool:
        raw = get_env("POSE_SEGMENTATION")
        if raw is not None:
            return self._truthy(raw, default=True)

        # MediaPipe Tasks segmentation masks are known to hard-crash some Python 3.12
        # macOS builds when converting masks to numpy (C++ CHECK failure). Default
        # segmentation OFF in that environment; users can override explicitly via
        # THROWS_TRACKER_POSE_SEGMENTATION=true.
        if self._backend == "pose_landmarker" and sys.version_info >= (3, 12):
            global _SEGMENTATION_PY312_DISABLED_WARNED
            if not _SEGMENTATION_PY312_DISABLED_WARNED:
                logger.warning(
                    "Pose segmentation is disabled by default on Python 3.12 (MediaPipe Tasks mask conversion can crash). "
                    "Set THROWS_TRACKER_POSE_SEGMENTATION=true to force-enable, or keep it off for stability."
                )
                _SEGMENTATION_PY312_DISABLED_WARNED = True
            return False

        # Default ON elsewhere: helps keep landmarks on the athlete (not javelin/background).
        return True

    @staticmethod
    def _segmentation_threshold() -> float:
        raw = get_env("POSE_SEGMENTATION_THRESHOLD")
        if raw:
            try:
                return float(np.clip(float(raw), 0.0, 1.0))
            except Exception:
                pass
        return 0.4

    @staticmethod
    def _segmentation_dilate_px() -> int:
        raw = get_env("POSE_SEGMENTATION_DILATE_PX")
        if raw:
            try:
                return max(0, int(float(raw)))
            except Exception:
                pass
        return 8

    @staticmethod
    def _segmentation_min_pixels() -> int:
        raw = get_env("POSE_SEGMENTATION_MIN_PIXELS")
        if raw:
            try:
                return max(10, int(float(raw)))
            except Exception:
                pass
        return 250

    @staticmethod
    def _segmentation_mask_open_px() -> int:
        raw = get_env("POSE_SEGMENTATION_OPEN_PX")
        if raw:
            try:
                return max(0, int(float(raw)))
            except Exception:
                pass
        # Only used when a mask looks like it contains a long thin spur (e.g., javelin).
        return 2

    @staticmethod
    def _mask_spur_ratio_threshold() -> float:
        raw = get_env("POSE_MASK_SPUR_RATIO_THRESHOLD")
        if raw:
            try:
                return float(max(1.2, float(raw)))
            except Exception:
                pass
        return 3.5

    @staticmethod
    def _mask_spur_aspect_threshold() -> float:
        raw = get_env("POSE_MASK_SPUR_ASPECT_THRESHOLD")
        if raw:
            try:
                return float(max(1.2, float(raw)))
            except Exception:
                pass
        return 3.8

    @staticmethod
    def _mask_component_keep_ratio() -> float:
        raw = get_env("POSE_MASK_COMPONENT_KEEP_RATIO")
        if raw:
            try:
                return float(np.clip(float(raw), 0.0, 1.0))
            except Exception:
                pass
        # Keep components >= 6% of largest (helps keep arms if slightly disconnected).
        return 0.06

    @staticmethod
    def _mask_min_area_fraction_after_refine() -> float:
        raw = get_env("POSE_MASK_MIN_AREA_FRACTION_AFTER_REFINE")
        if raw:
            try:
                return float(np.clip(float(raw), 0.0, 1.0))
            except Exception:
                pass
        # Safety valve: don't accept refinement if it destroys the mask.
        return 0.55

    @staticmethod
    def _extract_segmentation_masks_tasks(results: object) -> List[np.ndarray]:
        try:
            masks = getattr(results, "segmentation_masks", None)
        except Exception:
            masks = None
        if not masks:
            return []
        out: List[np.ndarray] = []
        for mask_img in list(masks or []):
            # Defensive: some MediaPipe builds can return segmentation mask images
            # whose declared format and underlying channel count disagree. Calling
            # `numpy_view()` in that case can hard-abort the interpreter (C++ CHECK).
            try:
                fmt = getattr(mask_img, "image_format", None)
                channels = getattr(mask_img, "channels", None)
            except Exception:
                fmt = None
                channels = None

            expected_channels: int | None = None
            try:
                if fmt in {mp.ImageFormat.GRAY8, mp.ImageFormat.GRAY16, mp.ImageFormat.VEC32F1}:
                    expected_channels = 1
                elif fmt == mp.ImageFormat.VEC32F2:
                    expected_channels = 2
                elif fmt == mp.ImageFormat.SRGB:
                    expected_channels = 3
                elif fmt in {mp.ImageFormat.SRGBA, mp.ImageFormat.SRGBA64, mp.ImageFormat.VEC32F4}:
                    expected_channels = 4
            except Exception:
                expected_channels = None

            if expected_channels is not None and channels is not None:
                try:
                    if int(channels) != int(expected_channels):
                        global _SEGMENTATION_CHANNEL_MISMATCH_WARNED
                        if not _SEGMENTATION_CHANNEL_MISMATCH_WARNED:
                            logger.warning(
                                "Skipping segmentation masks due to channel mismatch (format=%s channels=%s expected=%s). "
                                "Set THROWS_TRACKER_POSE_SEGMENTATION=false to disable segmentation entirely.",
                                fmt,
                                channels,
                                expected_channels,
                            )
                            _SEGMENTATION_CHANNEL_MISMATCH_WARNED = True
                        continue
                except Exception:
                    pass

            try:
                arr = mask_img.numpy_view()
            except Exception:
                continue
            mask = np.asarray(arr)
            if mask.ndim == 3:
                mask = mask[..., 0]
            mask = mask.astype(np.float32, copy=False)
            if float(np.nanmax(mask)) > 1.5:
                mask = mask / 255.0
            out.append(mask)
        return out

    @classmethod
    def _binary_person_mask(cls, mask: np.ndarray) -> np.ndarray:
        if mask is None or not isinstance(mask, np.ndarray) or mask.size == 0:
            return np.zeros((0, 0), dtype=bool)
        m = np.asarray(mask, dtype=np.float32)
        if m.ndim != 2:
            m = m.reshape(m.shape[0], m.shape[1])
        thresh = float(cls._segmentation_threshold())
        binary = (m >= thresh).astype(np.uint8)

        # If the mask looks like a person + long thin spur (common when the javelin is
        # included), apply a gentle cleanup that removes thin structures while keeping
        # major limbs. This is intentionally conservative and only triggers when the
        # mask's bbox is very sparse (bbox_area >> mask_area) or extremely elongated.
        try:
            ys, xs = np.where(binary)
        except Exception:
            ys = xs = np.array([], dtype=int)
        if xs.size >= int(cls._segmentation_min_pixels()):
            try:
                w = int(xs.max() - xs.min() + 1)
                h = int(ys.max() - ys.min() + 1)
                bbox_area = float(max(1, w) * max(1, h))
                area = float(xs.size)
                sparsity_ratio = bbox_area / max(1.0, area)
                aspect = float(max(w / max(1, h), h / max(1, w)))
                if sparsity_ratio >= float(cls._mask_spur_ratio_threshold()) or aspect >= float(
                    cls._mask_spur_aspect_threshold()
                ):
                    open_px = int(cls._segmentation_mask_open_px())
                    if open_px > 0:
                        k = max(1, int(open_px) * 2 + 1)
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
                    else:
                        opened = binary

                    # Keep the largest connected component and any substantial limb-sized components.
                    try:
                        n_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(opened, connectivity=8)
                    except Exception:
                        n_labels = 0
                        labels = None
                        stats = None

                    refined = opened
                    if n_labels and n_labels > 1 and labels is not None and stats is not None:
                        areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float32, copy=False)
                        largest = float(np.max(areas)) if areas.size else 0.0
                        keep_ratio = float(cls._mask_component_keep_ratio())
                        keep = [i + 1 for i, a in enumerate(areas.tolist()) if largest > 0 and float(a) >= largest * keep_ratio]
                        if keep:
                            refined = np.isin(labels, keep).astype(np.uint8)

                    # Safety: don't accept refinement if it destroys the mask area.
                    original_area = float(np.sum(binary))
                    refined_area = float(np.sum(refined))
                    if original_area > 0 and refined_area >= original_area * float(cls._mask_min_area_fraction_after_refine()):
                        binary = refined.astype(np.uint8, copy=False)
            except Exception:
                pass

        dilate_px = int(cls._segmentation_dilate_px())
        if dilate_px > 0 and binary.size:
            k = max(1, int(dilate_px) * 2 + 1)
            kernel = np.ones((k, k), dtype=np.uint8)
            binary = cv2.dilate(binary, kernel, iterations=1)
        return binary.astype(bool)

    @classmethod
    def _bbox_from_person_mask(
        cls,
        binary_mask: np.ndarray,
        *,
        roi_used: tuple[int, int, int, int] | None,
        full_width: int,
        full_height: int,
    ) -> tuple[int, int, int, int] | None:
        if binary_mask is None or binary_mask.size == 0:
            return None
        ys, xs = np.where(binary_mask)
        if xs.size < int(cls._segmentation_min_pixels()):
            return None
        mask_h, mask_w = int(binary_mask.shape[0]), int(binary_mask.shape[1])
        if mask_h <= 0 or mask_w <= 0:
            return None
        xs_f = xs.astype(np.float32, copy=False)
        ys_f = ys.astype(np.float32, copy=False)

        x0 = float(xs_f.min())
        x1 = float(xs_f.max())
        y0 = float(ys_f.min())
        y1 = float(ys_f.max())

        # Robustness: segmentation masks can occasionally include thin, long spurs
        # (e.g., a javelin) that explode the bbox width and reduce effective ROI
        # resolution. Detect this via IQR and trim only when the tail is extreme.
        try:
            trim_q_raw = get_env("POSE_MASK_BBOX_TRIM_QUANTILE")
            trim_q = float(trim_q_raw) if trim_q_raw else 0.01
            trim_q = float(np.clip(trim_q, 0.0, 0.2))
        except Exception:
            trim_q = 0.01
        try:
            factor_raw = get_env("POSE_MASK_BBOX_IQR_FACTOR")
            iqr_factor = float(factor_raw) if factor_raw else 4.0
            iqr_factor = float(np.clip(iqr_factor, 1.5, 12.0))
        except Exception:
            iqr_factor = 4.0

        def _maybe_trim(values: np.ndarray, lo: float, hi: float) -> tuple[float, float]:
            if values.size < 200 or trim_q <= 0:
                return (float(lo), float(hi))
            q25, q75 = np.quantile(values, [0.25, 0.75])
            iqr = float(q75 - q25)
            span = float(hi - lo)
            if not (np.isfinite(iqr) and iqr > 0 and np.isfinite(span) and span > 0):
                return (float(lo), float(hi))
            if span <= iqr * iqr_factor:
                return (float(lo), float(hi))
            t0 = float(np.quantile(values, trim_q))
            t1 = float(np.quantile(values, 1.0 - trim_q))
            if not (np.isfinite(t0) and np.isfinite(t1) and t1 > t0 + 1.0):
                return (float(lo), float(hi))
            return (float(t0), float(t1))

        x0, x1 = _maybe_trim(xs_f, x0, x1)
        y0, y1 = _maybe_trim(ys_f, y0, y1)

        x0 = int(np.floor(x0))
        x1 = int(np.ceil(x1)) + 1
        y0 = int(np.floor(y0))
        y1 = int(np.ceil(y1)) + 1
        x0 = max(0, min(x0, mask_w - 1))
        y0 = max(0, min(y0, mask_h - 1))
        x1 = max(x0 + 1, min(x1, mask_w))
        y1 = max(y0 + 1, min(y1, mask_h))

        if roi_used is not None:
            rx0, ry0, rx1, ry1 = roi_used
            crop_w = max(1, int(rx1 - rx0))
            crop_h = max(1, int(ry1 - ry0))
            fx0 = int(round(float(rx0) + (float(x0) / float(mask_w)) * float(crop_w)))
            fx1 = int(round(float(rx0) + (float(x1) / float(mask_w)) * float(crop_w)))
            fy0 = int(round(float(ry0) + (float(y0) / float(mask_h)) * float(crop_h)))
            fy1 = int(round(float(ry0) + (float(y1) / float(mask_h)) * float(crop_h)))
        else:
            fx0 = int(round((float(x0) / float(mask_w)) * float(full_width)))
            fx1 = int(round((float(x1) / float(mask_w)) * float(full_width)))
            fy0 = int(round((float(y0) / float(mask_h)) * float(full_height)))
            fy1 = int(round((float(y1) / float(mask_h)) * float(full_height)))

        fx0 = max(0, min(fx0, int(full_width) - 1))
        fy0 = max(0, min(fy0, int(full_height) - 1))
        fx1 = max(fx0 + 1, min(fx1, int(full_width)))
        fy1 = max(fy0 + 1, min(fy1, int(full_height)))
        return (fx0, fy0, fx1, fy1)

    @staticmethod
    def _bbox_from_landmarks(
        landmarks: List[LandmarkTuple],
        *,
        full_width: int,
        full_height: int,
        confidence_threshold: float = 0.4,
    ) -> tuple[int, int, int, int] | None:
        if not landmarks or full_width <= 0 or full_height <= 0:
            return None
        mode = (get_env("POSE_ROI_MODE") or "").strip().lower()
        if mode in {"all", "full"}:
            indices = list(range(min(33, len(landmarks))))
        else:
            indices = [0, 11, 12, 13, 14, 23, 24, 25, 26]

        xs: list[float] = []
        ys: list[float] = []
        thresh = float(confidence_threshold)
        for idx in indices:
            if not (0 <= idx < len(landmarks)):
                continue
            x, y, _z, c = landmarks[idx]
            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(c)):
                continue
            if float(c) < thresh:
                continue
            xf = float(x)
            yf = float(y)
            px = xf * float(full_width) if 0.0 <= xf <= 1.0 else xf
            py = yf * float(full_height) if 0.0 <= yf <= 1.0 else yf
            xs.append(px)
            ys.append(py)
        if len(xs) < 4:
            return None

        min_x = float(np.clip(min(xs), 0.0, float(full_width - 1)))
        max_x = float(np.clip(max(xs), 0.0, float(full_width - 1)))
        min_y = float(np.clip(min(ys), 0.0, float(full_height - 1)))
        max_y = float(np.clip(max(ys), 0.0, float(full_height - 1)))
        box_w = max(2.0, max_x - min_x)
        box_h = max(2.0, max_y - min_y)

        # A small margin makes this bbox usable for gating, but keep it tighter
        # than the ROI crop so it doesn't include background spectators.
        pad = 0.12
        pad_x = box_w * pad
        pad_y = box_h * pad
        x0 = int(round(min_x - pad_x))
        y0 = int(round(min_y - pad_y))
        x1 = int(round(max_x + pad_x))
        y1 = int(round(max_y + pad_y))

        x0 = max(0, min(x0, full_width - 1))
        y0 = max(0, min(y0, full_height - 1))
        x1 = max(x0 + 1, min(x1, full_width))
        y1 = max(y0 + 1, min(y1, full_height))
        return (int(x0), int(y0), int(x1), int(y1))

    @classmethod
    def _filter_landmarks_to_person_mask(
        cls,
        landmarks: List[LandmarkTuple],
        binary_mask: np.ndarray,
        *,
        outside_confidence: float = 0.05,
    ) -> List[LandmarkTuple]:
        if not landmarks or binary_mask is None or binary_mask.size == 0:
            return landmarks
        mask_h, mask_w = int(binary_mask.shape[0]), int(binary_mask.shape[1])
        if mask_h <= 0 or mask_w <= 0:
            return landmarks

        out: List[LandmarkTuple] = []
        oc = float(outside_confidence)
        for x, y, z, c in landmarks:
            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(c)):
                out.append((float(x), float(y), float(z), float(c)))
                continue
            xf = float(x)
            yf = float(y)
            # Tasks landmarks are normalized to the input image.
            if not (0.0 <= xf <= 1.0 and 0.0 <= yf <= 1.0):
                out.append((xf, yf, float(z), float(c)))
                continue
            ix = int(round(xf * float(mask_w - 1)))
            iy = int(round(yf * float(mask_h - 1)))
            ix = max(0, min(ix, mask_w - 1))
            iy = max(0, min(iy, mask_h - 1))
            if not bool(binary_mask[iy, ix]):
                out.append((xf, yf, float(z), float(min(float(c), oc))))
            else:
                out.append((xf, yf, float(z), float(c)))
        return out

    @classmethod
    def _filter_landmarks_to_bbox(
        cls,
        landmarks: List[LandmarkTuple],
        bbox: tuple[int, int, int, int],
        *,
        full_width: int,
        full_height: int,
        outside_confidence: float = 0.05,
    ) -> List[LandmarkTuple]:
        if not landmarks or bbox is None:
            return landmarks
        x0, y0, x1, y1 = bbox
        if full_width <= 0 or full_height <= 0:
            return landmarks
        oc = float(outside_confidence)
        out: List[LandmarkTuple] = []
        for x, y, z, c in landmarks:
            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(c)):
                out.append((float(x), float(y), float(z), float(c)))
                continue
            xf = float(x)
            yf = float(y)
            px = xf * float(full_width) if 0.0 <= xf <= 1.0 else xf
            py = yf * float(full_height) if 0.0 <= yf <= 1.0 else yf
            if px < float(x0) or px > float(x1) or py < float(y0) or py > float(y1):
                out.append((xf, yf, float(z), float(min(float(c), oc))))
            else:
                out.append((xf, yf, float(z), float(c)))
        return out

    def _init_holistic(self) -> None:
        self._backend = "holistic"
        self._mp_holistic = mp.solutions.holistic
        self._pose_landmark = self._mp_holistic.PoseLandmark
        model_complexity = 2
        env_complexity = get_env("POSE_HOLISTIC_MODEL_COMPLEXITY")
        if env_complexity:
            try:
                model_complexity = int(float(env_complexity))
            except Exception:
                pass
        model_complexity = int(np.clip(model_complexity, 0, 2))
        # Cache the Holistic model for reuse across frames.
        self._holistic = self._mp_holistic.Holistic(
            static_image_mode=False,
            smooth_landmarks=True,
            model_complexity=model_complexity,
        )
        logger.info(
            "PoseDetector initialized with MediaPipe Holistic (smooth_landmarks=True, model_complexity=%s).",
            model_complexity,
        )

    def _init_pose_landmarker(self) -> None:
        self._backend = "pose_landmarker"
        self._landmarker = self._create_pose_landmarker()
        logger.info("PoseDetector initialized with MediaPipe Tasks PoseLandmarker.")

    @staticmethod
    def _remap_landmarks_from_roi(
        landmarks: List[LandmarkTuple],
        *,
        roi: tuple[int, int, int, int],
        full_width: int,
        full_height: int,
    ) -> List[LandmarkTuple]:
        x0, y0, x1, y1 = roi
        crop_w = max(1, int(x1 - x0))
        crop_h = max(1, int(y1 - y0))
        scale_z = float(crop_w) / float(max(1, full_width))

        out: List[LandmarkTuple] = []
        for x, y, z, c in landmarks:
            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z) and np.isfinite(c)):
                out.append((float(x), float(y), float(z), float(c)))
                continue

            # Tasks landmarks are normalized to the *cropped* image.
            fx = (float(x0) + float(x) * float(crop_w)) / float(max(1, full_width))
            fy = (float(y0) + float(y) * float(crop_h)) / float(max(1, full_height))
            fz = float(z) * scale_z
            out.append((float(np.clip(fx, 0.0, 1.0)), float(np.clip(fy, 0.0, 1.0)), float(fz), float(c)))
        return out

    def _propose_roi_from_landmarks(
        self,
        landmarks: List[LandmarkTuple],
        *,
        full_width: int,
        full_height: int,
        confidence_threshold: float = 0.4,
    ) -> tuple[int, int, int, int] | None:
        if full_width <= 0 or full_height <= 0:
            return None
        mode = (get_env("POSE_ROI_MODE") or "").strip().lower()
        if mode in {"all", "full"}:
            indices = list(range(min(33, len(landmarks))))
        else:
            # Default: use stable torso/upper-limb points to avoid ROI drift onto
            # the javelin/background when distal landmarks snap incorrectly.
            indices = [0, 11, 12, 13, 14, 23, 24, 25, 26]
        xs: list[float] = []
        ys: list[float] = []
        thresh = float(confidence_threshold)
        for idx in indices:
            if not (0 <= idx < len(landmarks)):
                continue
            x, y, _z, c = landmarks[idx]
            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(c)):
                continue
            if float(c) < thresh:
                continue
            xs.append(float(x) * float(full_width))
            ys.append(float(y) * float(full_height))
        if len(xs) < 4 and mode not in {"all", "full"}:
            # Fallback: if we don't have enough stable points, use all landmarks.
            for x, y, _z, c in landmarks:
                if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(c)):
                    continue
                if float(c) < thresh:
                    continue
                xs.append(float(x) * float(full_width))
                ys.append(float(y) * float(full_height))
        if len(xs) < 6:
            return None

        min_x = float(np.clip(min(xs), 0.0, float(full_width - 1)))
        max_x = float(np.clip(max(xs), 0.0, float(full_width - 1)))
        min_y = float(np.clip(min(ys), 0.0, float(full_height - 1)))
        max_y = float(np.clip(max(ys), 0.0, float(full_height - 1)))

        box_w = max(1.0, max_x - min_x)
        box_h = max(1.0, max_y - min_y)

        # Expand generously to keep arms/javelin in view.
        margin_x = box_w * 0.35
        margin_y = box_h * 0.40

        cx = (min_x + max_x) * 0.5
        cy = (min_y + max_y) * 0.5

        # Smaller default => more zoom-in on the athlete (better for far-field videos).
        min_size = 160
        env_min = get_env("POSE_ROI_MIN_SIZE")
        if env_min:
            try:
                min_size = max(64, int(float(env_min)))
            except Exception:
                pass

        target_w = max(float(min_size), box_w + margin_x * 2.0)
        target_h = max(float(min_size), box_h + margin_y * 2.0)

        x0 = int(round(cx - target_w * 0.5))
        y0 = int(round(cy - target_h * 0.5))
        x1 = int(round(cx + target_w * 0.5))
        y1 = int(round(cy + target_h * 0.5))

        # Clamp to frame bounds while preserving size as best-effort.
        x0 = max(0, min(x0, full_width - 1))
        y0 = max(0, min(y0, full_height - 1))
        x1 = max(x0 + 1, min(x1, full_width))
        y1 = max(y0 + 1, min(y1, full_height))

        return (int(x0), int(y0), int(x1), int(y1))

    def _propose_roi_from_bbox(
        self,
        bbox: tuple[int, int, int, int],
        *,
        full_width: int,
        full_height: int,
    ) -> tuple[int, int, int, int] | None:
        x0, y0, x1, y1 = bbox
        if full_width <= 0 or full_height <= 0:
            return None
        x0 = max(0, min(int(x0), int(full_width - 1)))
        y0 = max(0, min(int(y0), int(full_height - 1)))
        x1 = max(x0 + 1, min(int(x1), int(full_width)))
        y1 = max(y0 + 1, min(int(y1), int(full_height)))
        box_w = max(1.0, float(x1 - x0))
        box_h = max(1.0, float(y1 - y0))

        margin_x = box_w * 0.35
        margin_y = box_h * 0.40

        cx = (float(x0) + float(x1)) * 0.5
        cy = (float(y0) + float(y1)) * 0.5

        # Keep consistent with landmark-based ROI sizing (zoom-in helps limb tracking).
        min_size = 160
        env_min = get_env("POSE_ROI_MIN_SIZE")
        if env_min:
            try:
                min_size = max(64, int(float(env_min)))
            except Exception:
                pass

        target_w = max(float(min_size), box_w + margin_x * 2.0)
        target_h = max(float(min_size), box_h + margin_y * 2.0)

        rx0 = int(round(cx - target_w * 0.5))
        ry0 = int(round(cy - target_h * 0.5))
        rx1 = int(round(cx + target_w * 0.5))
        ry1 = int(round(cy + target_h * 0.5))

        rx0 = max(0, min(rx0, full_width - 1))
        ry0 = max(0, min(ry0, full_height - 1))
        rx1 = max(rx0 + 1, min(rx1, full_width))
        ry1 = max(ry0 + 1, min(ry1, full_height))

        return (int(rx0), int(ry0), int(rx1), int(ry1))

    @staticmethod
    def _normalize_model_variant(value: str | None) -> str:
        variant = (value or "").strip().lower()
        if not variant:
            # Accuracy-first default for sports clips; can be overridden to "full"/"lite".
            return "heavy"
        alias = {
            "light": "lite",
            "default": "full",
            "standard": "full",
        }.get(variant)
        return alias or variant

    def _model_variant(self) -> str:
        return self._normalize_model_variant(
            get_env("POSE_LANDMARKER_MODEL_VARIANT")
            or get_env("POSE_LANDMARKER_MODEL_SIZE")
        )

    def _default_pose_landmarker_url(self) -> str:
        # Official MediaPipe model variants: lite/full/heavy.
        variant = self._model_variant()
        if variant not in {"lite", "full", "heavy"}:
            logger.warning("Unknown pose landmarker variant '%s'; falling back to 'lite'.", variant)
            variant = "lite"
        return (
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            f"pose_landmarker_{variant}/float16/1/pose_landmarker_{variant}.task"
        )

    def _default_pose_landmarker_path(self) -> Path:
        base = Path(__file__).resolve().parents[3]
        variant = self._model_variant()
        if variant not in {"lite", "full", "heavy"}:
            variant = "lite"
        return base / "data" / "biomechanics" / "models" / f"pose_landmarker_{variant}.task"

    def _ensure_pose_landmarker_model(self) -> Path:
        env_path = get_env("POSE_LANDMARKER_MODEL_PATH") or get_env("POSE_LANDMARKER_MODEL")
        model_path = Path(env_path).expanduser() if env_path else self._default_pose_landmarker_path()
        model_path.parent.mkdir(parents=True, exist_ok=True)

        if model_path.exists() and model_path.stat().st_size > 1024:
            return model_path

        url = get_env("POSE_LANDMARKER_MODEL_URL") or self._default_pose_landmarker_url()
        logger.info("Downloading pose landmarker model to %s", model_path)
        tmp_path = model_path.with_suffix(model_path.suffix + ".tmp")
        try:
            with urllib.request.urlopen(url) as response, tmp_path.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            tmp_path.replace(model_path)
        except Exception as exc:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            raise RuntimeError(
                "PoseLandmarker model download failed. "
                f"Set THROWS_TRACKER_POSE_LANDMARKER_MODEL_PATH to a local .task file, "
                f"or THROWS_TRACKER_POSE_LANDMARKER_MODEL_URL to a reachable model URL. Error: {exc}"
            ) from exc

        return model_path

    def _create_pose_landmarker(self):
        try:
            from mediapipe.tasks.python.core.base_options import BaseOptions
            from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError("MediaPipe Tasks API unavailable; install a mediapipe build with tasks support.") from exc

        model_path = self._ensure_pose_landmarker_model()
        num_poses = 2
        env_num = get_env("POSE_NUM_POSES")
        if env_num:
            try:
                num_poses = int(float(env_num))
            except Exception:
                pass
        num_poses = int(np.clip(num_poses, 1, 4))

        def _env_conf(name: str, default: float) -> float:
            raw = get_env(name)
            if not raw:
                return float(default)
            try:
                return float(np.clip(float(raw), 0.0, 1.0))
            except Exception:
                return float(default)

        min_det = _env_conf("POSE_MIN_DETECTION_CONFIDENCE", 0.5)
        min_presence = _env_conf("POSE_MIN_PRESENCE_CONFIDENCE", 0.5)
        min_track = _env_conf("POSE_MIN_TRACKING_CONFIDENCE", 0.5)
        output_segmentation = bool(self._segmentation_enabled())

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.VIDEO,
            num_poses=num_poses,
            min_pose_detection_confidence=min_det,
            min_pose_presence_confidence=min_presence,
            min_tracking_confidence=min_track,
            output_segmentation_masks=output_segmentation,
        )
        self._landmarker_running_mode = "VIDEO"
        return PoseLandmarker.create_from_options(options)

    def process_frame(self, frame: np.ndarray, *, timestamp_ms: int | float | None = None) -> Dict[str, object]:
        """Run pose estimation on a BGR frame and return landmarks and metadata.

        Returns a dictionary containing:
        - frame_idx: incremental index assigned by this detector
        - timestamp: wall-clock time at processing
        - landmarks: list of 33 (x, y, z, confidence) tuples for pose landmarks
        - hands_data: dict with left/right hand landmarks as (x, y, z, confidence)
        - pose_confidence_avg: average confidence across pose landmarks

        Holistic provides both the 33 pose landmarks and detailed hand landmarks
        that are important for analyzing javelin release timing and grip.
        """
        if frame is None:
            raise ValueError("Invalid frame: received None. Possible video decode or codec issue.")
        if not isinstance(frame, np.ndarray):
            raise ValueError("Invalid frame type; expected numpy.ndarray from decoded video frames.")
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
            raise ValueError(
                "Invalid frame shape; expected non-empty BGR image. Ensure the video codec is supported."
            )

        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except cv2.error as exc:
            raise RuntimeError(f"Failed to convert frame to RGB; codec/decoding issue: {exc}") from exc

        full_h, full_w = int(frame.shape[0]), int(frame.shape[1])
        roi_used: tuple[int, int, int, int] | None = None

        with self._lock:
            frame_idx = self._frame_counter
            self._frame_counter += 1
            timestamp = time.time()
            logger.debug("Processing frame %s (shape=%s)", frame_idx, getattr(frame, "shape", None))

            if self._backend == "holistic":
                results = self._holistic.process(rgb_frame)  # type: ignore[union-attr]
            else:
                # Optional ROI crop to increase effective resolution for the model.
                crop_rgb = rgb_frame
                if self._roi_enabled() and self._roi is not None:
                    x0, y0, x1, y1 = self._roi
                    if 0 <= x0 < x1 <= full_w and 0 <= y0 < y1 <= full_h and (x1 - x0) > 2 and (y1 - y0) > 2:
                        roi_used = (int(x0), int(y0), int(x1), int(y1))
                        crop_rgb = rgb_frame[y0:y1, x0:x1]

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)

                # MediaPipe Tasks expects a monotonically increasing timestamp in ms for VIDEO mode.
                ts_ms = timestamp_ms
                if ts_ms is None:
                    ts_ms = frame_idx * 33
                ts_ms_int = max(0, int(float(ts_ms)))

                if self._landmarker_running_mode == "VIDEO" and hasattr(self._landmarker, "detect_for_video"):
                    results = self._landmarker.detect_for_video(mp_image, ts_ms_int)  # type: ignore[union-attr]
                else:
                    results = self._landmarker.detect(mp_image)  # type: ignore[union-attr]

                # If ROI cropping missed the athlete, fall back to the full frame.
                poses = getattr(results, "pose_landmarks", None)
                if roi_used is not None and not poses:
                    logger.debug("ROI crop produced no pose; retrying on full frame.")
                    roi_used = None
                    self._roi = None
                    mp_image_full = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    if self._landmarker_running_mode == "VIDEO" and hasattr(self._landmarker, "detect_for_video"):
                        results = self._landmarker.detect_for_video(mp_image_full, ts_ms_int)  # type: ignore[union-attr]
                    else:
                        results = self._landmarker.detect(mp_image_full)  # type: ignore[union-attr]

        seg_bbox: tuple[int, int, int, int] | None = None
        person_bbox: tuple[int, int, int, int] | None = None
        if self._backend == "holistic":
            pose_landmarks = self._extract_pose_landmarks(results)
            left_hand = self._extract_hand_landmarks(results.left_hand_landmarks)
            right_hand = self._extract_hand_landmarks(results.right_hand_landmarks)
            person_bbox = self._bbox_from_landmarks(pose_landmarks, full_width=full_w, full_height=full_h)
        else:
            pose_candidates_raw = self._extract_pose_landmarks_tasks_all(results)
            seg_masks = self._extract_segmentation_masks_tasks(results) if self._segmentation_enabled() else []
            binary_masks: List[np.ndarray] = []
            if seg_masks:
                binary_masks = [self._binary_person_mask(mask) for mask in seg_masks]
                for idx in range(min(len(pose_candidates_raw), len(binary_masks))):
                    pose_candidates_raw[idx] = self._filter_landmarks_to_person_mask(
                        pose_candidates_raw[idx],
                        binary_masks[idx],
                        outside_confidence=0.05,
                    )

            pose_candidates = pose_candidates_raw
            if roi_used is not None:
                pose_candidates = [
                    self._remap_landmarks_from_roi(
                        candidate,
                        roi=roi_used,
                        full_width=full_w,
                        full_height=full_h,
                    )
                    for candidate in pose_candidates
                ]
            pose_landmarks, best_idx = self._select_best_pose(pose_candidates)
            left_hand = []
            right_hand = []
            if binary_masks and 0 <= int(best_idx) < len(binary_masks):
                seg_bbox = self._bbox_from_person_mask(
                    binary_masks[int(best_idx)],
                    roi_used=roi_used,
                    full_width=full_w,
                    full_height=full_h,
                )

            person_bbox = seg_bbox
            if person_bbox is None and pose_landmarks:
                person_bbox = self._bbox_from_landmarks(pose_landmarks, full_width=full_w, full_height=full_h)

            if person_bbox is not None and pose_landmarks:
                # Additional safety: down-weight any landmark that falls outside the athlete bbox.
                # This helps prevent wrists/ankles snapping onto the javelin/background.
                pose_landmarks = self._filter_landmarks_to_bbox(
                    pose_landmarks,
                    person_bbox,
                    full_width=full_w,
                    full_height=full_h,
                    outside_confidence=0.05,
                )

            if person_bbox is not None:
                cx = (float(person_bbox[0]) + float(person_bbox[2])) * 0.5 / float(max(1, full_w))
                cy = (float(person_bbox[1]) + float(person_bbox[3])) * 0.5 / float(max(1, full_h))
                self._prev_pose_center = (float(np.clip(cx, 0.0, 1.0)), float(np.clip(cy, 0.0, 1.0)))
            elif pose_landmarks:
                center, _area, avg_conf = self._pose_summary(pose_landmarks)
                if np.isfinite(avg_conf) and avg_conf >= 0.1:
                    self._prev_pose_center = center

            # Update ROI for the next frame using the full-frame landmarks.
            if self._roi_enabled():
                proposed = None
                if seg_bbox is not None:
                    proposed = self._propose_roi_from_bbox(seg_bbox, full_width=full_w, full_height=full_h)
                if proposed is None:
                    proposed = self._propose_roi_from_landmarks(pose_landmarks, full_width=full_w, full_height=full_h)
                if proposed is not None:
                    # Lower default alpha => ROI follows fast throws better (still smoothable via env).
                    alpha = 0.45
                    env_alpha = get_env("POSE_ROI_SMOOTHING")
                    if env_alpha:
                        try:
                            alpha = float(env_alpha)
                        except Exception:
                            pass
                    alpha = float(np.clip(alpha, 0.0, 0.95))

                    with self._lock:
                        if self._roi is None:
                            self._roi = proposed
                        else:
                            ox0, oy0, ox1, oy1 = self._roi
                            px0, py0, px1, py1 = proposed
                            nx0 = int(round(alpha * ox0 + (1.0 - alpha) * px0))
                            ny0 = int(round(alpha * oy0 + (1.0 - alpha) * py0))
                            nx1 = int(round(alpha * ox1 + (1.0 - alpha) * px1))
                            ny1 = int(round(alpha * oy1 + (1.0 - alpha) * py1))
                            nx0 = max(0, min(nx0, full_w - 1))
                            ny0 = max(0, min(ny0, full_h - 1))
                            nx1 = max(nx0 + 1, min(nx1, full_w))
                            ny1 = max(ny0 + 1, min(ny1, full_h))
                            self._roi = (nx0, ny0, nx1, ny1)

        if not pose_landmarks:
            pose_landmarks = [(0.0, 0.0, 0.0, 0.0) for _ in range(33)]

        pose_confidence_avg = self._compute_average_confidence(pose_landmarks)
        logger.debug(
            "Frame %s processed: pose_landmarks=%s left_hand=%s right_hand=%s avg_conf=%.3f",
            frame_idx,
            bool(pose_landmarks),
            bool(left_hand),
            bool(right_hand),
            pose_confidence_avg,
        )

        return {
            "frame_idx": frame_idx,
            "timestamp": timestamp,
            "landmarks": pose_landmarks,
            "hands_data": {"left": left_hand, "right": right_hand},
            "pose_confidence_avg": pose_confidence_avg,
            "person_bbox": person_bbox,
        }

    def get_key_joints(self) -> Dict[str, Tuple[int, int]]:
        """Return indices for key joints used in kinematic analysis."""
        if self._backend != "holistic" or self._pose_landmark is None:
            # MediaPipe landmark indices used by both solutions/tasks pose models.
            return {
                "shoulders": (11, 12),
                "elbows": (13, 14),
                "wrists": (15, 16),
                "hips": (23, 24),
                "knees": (25, 26),
            }
        return {
            "shoulders": (
                self._pose_landmark.LEFT_SHOULDER.value,
                self._pose_landmark.RIGHT_SHOULDER.value,
            ),
            "elbows": (
                self._pose_landmark.LEFT_ELBOW.value,
                self._pose_landmark.RIGHT_ELBOW.value,
            ),
            "wrists": (
                self._pose_landmark.LEFT_WRIST.value,
                self._pose_landmark.RIGHT_WRIST.value,
            ),
            "hips": (
                self._pose_landmark.LEFT_HIP.value,
                self._pose_landmark.RIGHT_HIP.value,
            ),
            "knees": (
                self._pose_landmark.LEFT_KNEE.value,
                self._pose_landmark.RIGHT_KNEE.value,
            ),
        }

    def _extract_pose_landmarks(self, results: object) -> List[LandmarkTuple]:
        if results is None or results.pose_landmarks is None:
            return [(0.0, 0.0, 0.0, 0.0) for _ in self._pose_landmark]

        landmarks = results.pose_landmarks.landmark
        ordered_landmarks: List[LandmarkTuple] = []
        for landmark_enum in self._pose_landmark:
            lm = landmarks[landmark_enum.value]
            ordered_landmarks.append(self._landmark_to_tuple(lm))
        return ordered_landmarks

    @staticmethod
    def _pose_summary(landmarks: List[LandmarkTuple], *, conf_threshold: float = 0.4) -> tuple[tuple[float, float], float, float]:
        xs: list[float] = []
        ys: list[float] = []
        confs: list[float] = []
        thresh = float(conf_threshold)
        for x, y, _z, c in landmarks:
            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(c)):
                continue
            cf = float(c)
            confs.append(cf)
            if cf >= thresh:
                xs.append(float(x))
                ys.append(float(y))
        avg_conf = float(np.mean(confs)) if confs else 0.0
        if not xs or not ys:
            return (0.5, 0.5), 0.0, avg_conf
        min_x = float(np.clip(min(xs), 0.0, 1.0))
        max_x = float(np.clip(max(xs), 0.0, 1.0))
        min_y = float(np.clip(min(ys), 0.0, 1.0))
        max_y = float(np.clip(max(ys), 0.0, 1.0))
        area = max(0.0, (max_x - min_x) * (max_y - min_y))
        center = ((min_x + max_x) * 0.5, (min_y + max_y) * 0.5)
        return (float(center[0]), float(center[1])), float(area), avg_conf

    @staticmethod
    def _pose_plausibility_penalty(
        landmarks: List[LandmarkTuple],
        *,
        conf_threshold: float = 0.4,
    ) -> float:
        """Return a [0..1] penalty for anatomically implausible landmark layouts.

        This is used to pick the "best" pose when multiple candidates exist.
        Under fast motion/occlusion, pose models can produce a high-confidence
        wrist/elbow that snaps to the background or the javelin; those failures
        often violate simple limb-length and left/right symmetry constraints.
        """
        if not landmarks or len(landmarks) < 33:
            return 0.0

        thresh = float(conf_threshold)

        def _pt(i: int) -> np.ndarray | None:
            try:
                x, y, _z, c = landmarks[int(i)]
            except Exception:
                return None
            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(c)):
                return None
            if float(c) < thresh:
                return None
            return np.array([float(x), float(y)], dtype=float)

        def _dist(a: int, b: int) -> float | None:
            p = _pt(a)
            q = _pt(b)
            if p is None or q is None:
                return None
            d = float(np.linalg.norm(p - q))
            return d if np.isfinite(d) else None

        # Use a torso scale so thresholds are relative to the detected athlete size.
        shoulder_w = _dist(11, 12)
        hip_w = _dist(23, 24)
        torso_scale = max([v for v in (shoulder_w, hip_w) if v is not None] or [0.0])
        if not np.isfinite(torso_scale) or float(torso_scale) <= 1e-6:
            return 0.0

        checks = 0
        bad = 0

        def _check_length(length: float | None) -> None:
            nonlocal checks, bad
            if length is None or not np.isfinite(length):
                return
            checks += 1
            rel = float(length) / float(torso_scale)
            # Very conservative bounds: allow perspective/partial visibility but
            # reject "teleporting" joints.
            if rel < 0.20 or rel > 5.0:
                bad += 1

        def _check_symmetry(a: float | None, b: float | None) -> None:
            nonlocal checks, bad
            if a is None or b is None or not (np.isfinite(a) and np.isfinite(b)):
                return
            if float(a) <= 1e-9 or float(b) <= 1e-9:
                return
            checks += 1
            ratio = float(a) / float(b)
            ratio = max(ratio, 1.0 / ratio)
            if ratio > 1.9:
                bad += 1

        # Arm + leg bones.
        l_upper = _dist(11, 13)
        l_fore = _dist(13, 15)
        r_upper = _dist(12, 14)
        r_fore = _dist(14, 16)
        l_thigh = _dist(23, 25)
        l_shin = _dist(25, 27)
        r_thigh = _dist(24, 26)
        r_shin = _dist(26, 28)

        for length in (l_upper, l_fore, r_upper, r_fore, l_thigh, l_shin, r_thigh, r_shin):
            _check_length(length)

        # Left/right symmetry checks.
        _check_symmetry(l_upper, r_upper)
        _check_symmetry(l_fore, r_fore)
        _check_symmetry(l_thigh, r_thigh)
        _check_symmetry(l_shin, r_shin)

        if checks <= 0:
            return 0.0
        return float(max(0.0, min(1.0, float(bad) / float(checks))))

    def _select_best_pose(self, candidates: List[List[LandmarkTuple]]) -> tuple[List[LandmarkTuple], int]:
        if not candidates:
            return ([], 0)
        prev_center = self._prev_pose_center
        best_score = -1e9
        best: List[LandmarkTuple] = candidates[0]
        best_idx = 0
        for idx, pose in enumerate(candidates):
            center, area, avg_conf = self._pose_summary(pose)
            plaus_pen = self._pose_plausibility_penalty(pose)
            score = float(avg_conf) * 1.6 + float(area) * 0.8 - float(plaus_pen) * 1.2
            if prev_center is not None:
                dist = float(np.hypot(center[0] - prev_center[0], center[1] - prev_center[1]))
                score -= dist * 1.2
            if score > best_score:
                best_score = score
                best = pose
                best_idx = int(idx)
        return best, int(best_idx)

    def _extract_pose_landmarks_tasks_all(self, results: object) -> List[List[LandmarkTuple]]:
        try:
            poses = getattr(results, "pose_landmarks", None)
        except Exception:
            poses = None

        if not poses:
            return []

        candidates: list[object] = []
        if isinstance(poses, list):
            candidates = list(poses)
        else:
            candidates = [poses]

        out_all: List[List[LandmarkTuple]] = []
        for candidate in candidates:
            try:
                raw = list(candidate or [])
            except TypeError:
                raw = list(getattr(candidate, "landmark", []) or [])
            out: List[LandmarkTuple] = []
            for lm in raw:
                out.append(self._landmark_to_tuple(lm))
            if len(out) < 33:
                out.extend([(0.0, 0.0, 0.0, 0.0) for _ in range(33 - len(out))])
            out_all.append(out[:33])

        return out_all

    def _extract_hand_landmarks(self, hand_landmarks: object) -> List[LandmarkTuple]:
        if hand_landmarks is None:
            return []
        return [self._landmark_to_tuple(lm) for lm in hand_landmarks.landmark]

    @staticmethod
    def _landmark_to_tuple(landmark: object) -> LandmarkTuple:
        confidence = getattr(landmark, "visibility", getattr(landmark, "presence", 1.0))
        return (
            float(landmark.x),
            float(landmark.y),
            float(landmark.z),
            float(confidence),
        )

    @staticmethod
    def _compute_average_confidence(landmarks: List[LandmarkTuple]) -> float:
        if not landmarks:
            return 0.0
        total_confidence = sum(lm[3] for lm in landmarks)
        return total_confidence / len(landmarks)

    def close(self) -> None:
        """Release MediaPipe model resources."""
        with self._lock:
            if self._closed:
                return
            if self._backend == "holistic":
                if self._holistic is not None:
                    self._holistic.close()
                    self._holistic = None
            else:
                if self._landmarker is not None:
                    self._landmarker.close()
                    self._landmarker = None
            self._closed = True
            logger.info("PoseDetector resources released.")

    def __enter__(self) -> "PoseDetector":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            # Avoid raising during interpreter shutdown.
            pass
