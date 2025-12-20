"""End-to-end pose extraction pipeline for javelin biomechanics."""

from __future__ import annotations

import json
import signal
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from rich.progress import Progress

from javelin_tracker.biomechanics.config import (
    BIOMECHANICS_LOGGER as logger,
    CONFIDENCE_THRESHOLD,
    FRAME_SMOOTHING_WINDOW,
    VALID_FRAME_THRESHOLD,
)
from javelin_tracker.biomechanics.pose_estimation.pose_detector import PoseDetector
from javelin_tracker.biomechanics.pose_estimation.optical_flow_refiner import OpticalFlowLandmarkRefiner
from javelin_tracker.biomechanics.utils.filtering import (
    bone_length_outlier_mask,
    fill_low_confidence_landmarks,
    joint_displacement_outlier_mask,
    kalman_smooth_landmarks,
    smooth_landmarks,
)
from javelin_tracker.biomechanics.utils.video import extract_frames, get_video_metadata, validate_video_readable
from javelin_tracker.env import get_env


class PosePipeline:
    """Orchestrates frame extraction, pose detection, smoothing, and export."""

    def __init__(self, detector: Optional[PoseDetector] = None) -> None:
        self.detector = detector or PoseDetector()
        self._external_detector = detector is not None
        self._interrupted = False
        # `signal.signal()` only works in the main thread; the webapp runs this
        # pipeline inside a background thread via ThreadPoolExecutor.
        if threading.current_thread() is threading.main_thread():
            try:
                signal.signal(signal.SIGINT, self._handle_interrupt)
            except Exception:
                # Best-effort: interruptions are optional, so don't fail init.
                pass

    def _handle_interrupt(self, signum, frame) -> None:  # type: ignore[override]
        self._interrupted = True
        logger.warning("Interrupt received; will stop after current frame.")

    def process_video(
        self,
        video_path: str | Path,
        video_id: str,
        output_dir: str | Path,
        *,
        max_retries: int = 2,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> Dict[str, Any]:
        """Process a video and emit smoothed pose data to JSON."""
        def _truthy(value: str | None, *, default: bool = False) -> bool:
            if value is None:
                return bool(default)
            return value.strip().lower() not in {"0", "false", "no", "off", ""}

        path = Path(video_path)
        out_root = Path(output_dir) / video_id
        out_root.mkdir(parents=True, exist_ok=True)
        output_path = out_root / "pose_data.json"
        warnings: List[str] = []

        try:
            validation = validate_video_readable(path)
            metadata = get_video_metadata(path)
        except Exception as exc:
            logger.error("Video validation failed for %s: %s", path, exc)
            return {"status": "error", "error": str(exc), "output_path": str(output_path)}

        frames_records: List[Dict[str, Any]] = []
        coords_sequence: List[np.ndarray] = []
        confidence_sequence: List[np.ndarray] = []
        per_frame_conf_avg: List[float] = []
        flow_refiner: OpticalFlowLandmarkRefiner | None = None
        flow_enabled = False

        try:
            total_frames = int(validation.get("total_frames") or metadata.get("total_frames") or 0)
            assumed_fps_raw = get_env("POSE_ASSUMED_FPS")
            time_scale_raw = get_env("POSE_TIME_SCALE")
            assumed_fps: float | None = None
            time_scale: float | None = None
            if assumed_fps_raw:
                try:
                    assumed_fps = float(assumed_fps_raw)
                except Exception:
                    assumed_fps = None
            if time_scale_raw:
                try:
                    time_scale = float(time_scale_raw)
                except Exception:
                    time_scale = None
            # Default to a mild "slow-motion" tracker timestamp scaling for the
            # MediaPipe Tasks backend. This often improves stability for very fast
            # throws (arm behind head / rapid elbow-wrist motion) without changing
            # the actual video frames or the timestamps stored in the output JSON.
            if time_scale is None:
                backend = str(getattr(self.detector, "_backend", "") or "").strip().lower()
                if backend == "pose_landmarker":
                    time_scale = 0.5

            flow_enabled = _truthy(get_env("POSE_OPTICAL_FLOW"), default=True)
            if flow_enabled:
                flow_refiner = OpticalFlowLandmarkRefiner(
                    enabled=True,
                    confidence_threshold=float(CONFIDENCE_THRESHOLD),
                )

            processed_frames = 0
            last_progress_cb = 0.0
            def _report_progress() -> None:
                nonlocal last_progress_cb
                if progress_callback is None:
                    return
                if total_frames <= 0:
                    return
                now = time.monotonic()
                # Keep callback overhead low; webapp polls ~1.5s anyway.
                if now - last_progress_cb < 0.15 and processed_frames < total_frames:
                    return
                last_progress_cb = now
                try:
                    progress_callback(int(processed_frames), int(total_frames))
                except Exception:
                    # Progress reporting must never break processing.
                    pass

            with Progress() as progress:
                task = progress.add_task(f"Processing {path.name}", total=total_frames or None)
                for frame_idx, timestamp_ms, frame in extract_frames(path, fps=metadata["fps"]):
                    if self._interrupted:
                        warnings.append("Processing interrupted by user.")
                        break

                    retries = 0
                    result = None
                    # Optional "slow motion" timestamping for the pose tracker.
                    # Note: this does not change the video frames; it only affects
                    # the temporal assumptions of the tracking backend.
                    detector_ts_ms = float(timestamp_ms)
                    if assumed_fps and assumed_fps > 0:
                        detector_ts_ms = float(frame_idx) * (1000.0 / float(assumed_fps))
                    if time_scale and time_scale > 0:
                        detector_ts_ms = detector_ts_ms * float(time_scale)
                    while retries <= max_retries and result is None:
                        try:
                            result = self.detector.process_frame(frame, timestamp_ms=detector_ts_ms)
                        except Exception as exc:
                            retries += 1
                            logger.warning("Frame %s failed (attempt %s/%s): %s", frame_idx, retries, max_retries, exc)
                            if retries > max_retries:
                                warnings.append(f"Frame {frame_idx} failed after {max_retries} retries: {exc}")
                                break

                    if result is None:
                        processed_frames += 1
                        progress.update(task, advance=1)
                        _report_progress()
                        continue

                    landmarks = result["landmarks"]
                    if flow_refiner is not None and flow_refiner.enabled:
                        try:
                            landmarks = flow_refiner.refine(
                                frame,
                                list(landmarks),
                                person_bbox=result.get("person_bbox") if isinstance(result, dict) else None,
                            )
                        except Exception as exc:
                            warnings.append(f"Optical-flow refinement disabled after error: {exc}")
                            flow_refiner.enabled = False
                            flow_enabled = False
                    coords = np.array([[x, y, z] for (x, y, z, _conf) in landmarks], dtype=float)
                    confidences = np.array([conf for (_x, _y, _z, conf) in landmarks], dtype=float)

                    conf_valid_ratio = float(np.mean(confidences >= CONFIDENCE_THRESHOLD))
                    is_valid = conf_valid_ratio >= VALID_FRAME_THRESHOLD
                    if not is_valid:
                        warning_msg = (
                            f"Frame {frame_idx} low confidence: "
                            f"{conf_valid_ratio*100:.1f}% landmarks >= {CONFIDENCE_THRESHOLD}"
                        )
                        warnings.append(warning_msg)
                        logger.warning(warning_msg)

                    frames_records.append(
                        {
                            "frame_idx": frame_idx,
                            "timestamp_ms": timestamp_ms,
                            "landmarks": landmarks,
                            "valid": bool(is_valid),
                        }
                    )
                    coords_sequence.append(coords)
                    confidence_sequence.append(confidences)
                    per_frame_conf_avg.append(result["pose_confidence_avg"])
                    processed_frames += 1
                    progress.update(task, advance=1)
                    _report_progress()
        finally:
            if not self._external_detector:
                self.detector.close()

        if not frames_records:
            return {"status": "error", "error": "No frames processed.", "output_path": str(output_path)}

        # Apply smoothing to xyz coordinates; keep original confidences.
        coords_array = np.stack(coords_sequence, axis=0)
        conf_array = np.stack(confidence_sequence, axis=0)
        try:
            # Robust occlusion handling:
            # - detect bone-length outliers (common when an arm is behind the head)
            # - treat them as missing and interpolate
            # - apply Kalman smoothing with gating to reject implausible jumps
            arm_bones = ((11, 13), (13, 15), (12, 14), (14, 16))
            leg_bones = ((23, 25), (25, 27), (24, 26), (26, 28))
            outliers_bl_arm = bone_length_outlier_mask(
                coords_array,
                conf_array,
                confidence_threshold=CONFIDENCE_THRESHOLD,
                bones=arm_bones,
                ratio_range=(0.60, 1.40),
                min_samples=10,
            )
            outliers_bl_leg = bone_length_outlier_mask(
                coords_array,
                conf_array,
                confidence_threshold=CONFIDENCE_THRESHOLD,
                bones=leg_bones,
                ratio_range=(0.55, 1.55),
                min_samples=10,
            )
            outliers_bl = outliers_bl_arm | outliers_bl_leg

            fps_val = float(metadata.get("fps") or 30.0)
            abs_thr = 0.12 * (30.0 / max(1e-9, fps_val))  # normalized units per frame
            abs_thr = float(np.clip(abs_thr, 0.03, 0.20))
            outliers_disp = joint_displacement_outlier_mask(
                coords_array,
                conf_array,
                confidence_threshold=CONFIDENCE_THRESHOLD,
                joint_indices=(13, 14, 15, 16, 25, 26, 27, 28),
                iqr_factor=6.0,
                absolute_threshold=abs_thr,
                min_samples=12,
            )
            outliers = outliers_bl | outliers_disp
            if bool(np.any(outliers_bl)):
                warnings.append(
                    f"Detected {int(np.sum(outliers_bl))} bone-length outliers; applied occlusion-robust smoothing."
                )
            if bool(np.any(outliers_disp)):
                warnings.append(
                    f"Detected {int(np.sum(outliers_disp))} displacement outliers; filtered background snaps."
                )

            invalid = (~np.isfinite(conf_array)) | (conf_array < float(CONFIDENCE_THRESHOLD)) | outliers
            coords_masked = coords_array.copy()
            coords_masked[invalid] = np.nan
            conf_masked = conf_array.copy()
            conf_masked[invalid] = 0.0

            filled = fill_low_confidence_landmarks(
                coords_masked,
                conf_masked,
                confidence_threshold=CONFIDENCE_THRESHOLD,
            )

            # Use interpolated points as "weak" measurements to keep the filter stable.
            conf_for_kalman = conf_array.copy()
            conf_for_kalman[invalid] = 0.05
            kalman_fps = float(metadata.get("fps") or 30.0)
            if assumed_fps and assumed_fps > 0:
                kalman_fps = float(assumed_fps)
            if time_scale and time_scale > 0:
                kalman_fps = kalman_fps / float(time_scale)
            kalman = kalman_smooth_landmarks(
                filled,
                conf_for_kalman,
                fps=kalman_fps,
                confidence_threshold=CONFIDENCE_THRESHOLD,
                process_noise=1e-4,
                measurement_noise=5e-4,
                mahalanobis_threshold=25.0,
            )

            smoothed = smooth_landmarks(kalman, window_length=FRAME_SMOOTHING_WINDOW, polyorder=2)
        except Exception as exc:
            warnings.append(f"Smoothing failed; emitting raw landmarks. Error: {exc}")
            logger.warning("Smoothing failed: %s", exc)
            smoothed = coords_array

        # Reduce confidence for any measurement we had to "repair" so the UI can
        # avoid drawing landmarks that would otherwise appear to jump to random
        # background objects (even if MediaPipe reported high confidence).
        conf_out = conf_array.copy()
        conf_out[outliers] = np.minimum(conf_out[outliers], 0.02)
        conf_out[conf_array < float(CONFIDENCE_THRESHOLD)] = np.minimum(
            conf_out[conf_array < float(CONFIDENCE_THRESHOLD)], 0.10
        )

        # Inject smoothed coordinates back into frame records (with corrected conf).
        for idx, frame_record in enumerate(frames_records):
            sm_coords = smoothed[idx]
            confs = conf_out[idx]
            frame_record["landmarks"] = [
                (float(x), float(y), float(z), float(conf)) for (x, y, z), conf in zip(sm_coords, confs)
            ]
            try:
                conf_valid_ratio = float(np.mean(confs >= CONFIDENCE_THRESHOLD))
                frame_record["valid"] = bool(conf_valid_ratio >= VALID_FRAME_THRESHOLD)
            except Exception:
                pass

        processing_stats = {
            "total_frames": len(frames_records),
            "valid_frames": sum(1 for fr in frames_records if fr["valid"]),
            "avg_confidence": float(np.mean(per_frame_conf_avg)) if per_frame_conf_avg else 0.0,
        }
        if flow_refiner is not None:
            replacements = int(flow_refiner.stats.replacements)
            swaps = int(flow_refiner.stats.swaps)
            processing_stats["optical_flow_replacements"] = replacements
            processing_stats["optical_flow_swaps"] = swaps
            if replacements:
                warnings.append(f"Optical-flow refinement applied: {replacements} landmark replacements, {swaps} swaps.")

        output_payload = {
            "video_id": video_id,
            "video_metadata": metadata,
            "frames": frames_records,
            "processing_stats": processing_stats,
            "warnings": warnings,
            "tracking_settings": {
                "pose_assumed_fps": float(assumed_fps) if assumed_fps and assumed_fps > 0 else None,
                "pose_time_scale": float(time_scale) if time_scale and time_scale > 0 else None,
                "pose_optical_flow": bool(flow_enabled),
            },
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
        status = "aborted" if self._interrupted else "success"
        return {"status": status, "output_path": str(output_path), "warnings": warnings}
