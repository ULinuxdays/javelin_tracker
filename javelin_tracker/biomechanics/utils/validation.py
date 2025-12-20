"""Pose data quality validation utilities."""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from javelin_tracker.biomechanics.config import (
    CONFIDENCE_THRESHOLD,
    OUTLIER_THRESHOLDS,
    VALID_FRAME_THRESHOLD,
    JAVELIN_KEY_JOINTS,
)
from javelin_tracker.biomechanics.metrics.phase_detection import detect_throw_phases
from javelin_tracker.biomechanics import BIOMECHANICS_LOGGER as logger


def _pair_distance(coords: np.ndarray, idx_a: int, idx_b: int) -> float:
    return float(np.linalg.norm(coords[idx_a] - coords[idx_b]))


def check_anatomical_plausibility(landmarks_frame: np.ndarray) -> bool:
    """Check if limb proportions are human-like for a single frame."""
    if landmarks_frame.shape != (33, 3):
        return False

    shoulder_left, shoulder_right = JAVELIN_KEY_JOINTS["shoulders"]
    elbow_left, elbow_right = JAVELIN_KEY_JOINTS["elbows"]
    wrist_left, wrist_right = JAVELIN_KEY_JOINTS["wrists"]

    upper_arm_left = _pair_distance(landmarks_frame, shoulder_left, elbow_left)
    upper_arm_right = _pair_distance(landmarks_frame, shoulder_right, elbow_right)
    forearm_left = _pair_distance(landmarks_frame, elbow_left, wrist_left)
    forearm_right = _pair_distance(landmarks_frame, elbow_right, wrist_right)

    def _ratio_ok(a: float, b: float) -> bool:
        if a == 0 or b == 0:
            return False
        return 0.6 <= a / b <= 1.4

    return all(
        [
            _ratio_ok(upper_arm_left, upper_arm_right),
            _ratio_ok(forearm_left, forearm_right),
            _ratio_ok(upper_arm_left, forearm_left),
            _ratio_ok(upper_arm_right, forearm_right),
        ]
    )


def _presence_score(confidences: np.ndarray, key_indices: List[int]) -> Tuple[float, List[str]]:
    per_joint_presence = []
    issues = []
    for idx in key_indices:
        joint_conf = confidences[:, idx]
        present_ratio = float(np.mean(joint_conf >= CONFIDENCE_THRESHOLD))
        per_joint_presence.append(present_ratio)
        if present_ratio < 0.9:
            frames_missing = np.where(joint_conf < CONFIDENCE_THRESHOLD)[0]
            if frames_missing.size:
                start, end = frames_missing[0], frames_missing[-1]
                issues.append(
                    f"Joint {idx} confidence < {CONFIDENCE_THRESHOLD:.2f} in frames {start}-{end} "
                    f"({present_ratio*100:.1f}% present)."
                )
    return float(np.mean(per_joint_presence)) if per_joint_presence else 0.0, issues


def _distance_stability(coords: np.ndarray, idx_pairs: List[Tuple[int, int]]) -> Tuple[float, List[str]]:
    issues = []
    stabilities = []
    for a, b in idx_pairs:
        dists = np.linalg.norm(coords[:, a, :] - coords[:, b, :], axis=1)
        median = np.median(dists)
        if median == 0:
            continue
        rel_var = float(np.std(dists) / median)
        stabilities.append(max(0.0, 1.0 - rel_var))
        if rel_var > 0.05:
            issues.append(
                f"Distance variance >5% between landmarks {a}-{b} (std/median={rel_var:.3f})."
            )
    return (float(np.mean(stabilities)) if stabilities else 0.0, issues)


def _temporal_smoothness(coords: np.ndarray) -> Tuple[float, List[str]]:
    if coords.shape[0] < 2:
        return 1.0, []
    deltas = np.linalg.norm(np.diff(coords, axis=0), axis=2)  # (n-1, 33)
    mean_jump = np.mean(deltas, axis=1)
    median_jump = np.median(mean_jump)
    outlier_thresh = max(median_jump * 5, OUTLIER_THRESHOLDS.max_position_jump_m)
    outlier_frames = np.where(mean_jump > outlier_thresh)[0]
    issues = []
    if outlier_frames.size:
        spans = f"{outlier_frames[0]}-{outlier_frames[-1]}" if outlier_frames.size > 1 else f"{outlier_frames[0]}"
        issues.append(f"Sudden position jumps detected around frames {spans}.")
    smooth_score = float(np.clip(1.0 - (len(outlier_frames) / max(1, coords.shape[0])), 0.0, 1.0))
    return smooth_score, issues


def validate_pose_quality(
    landmarks_data: List[Dict[str, object]], video_metadata: Dict[str, object]
) -> Dict[str, object]:
    """Validate pose quality and return a weighted quality score with issues."""
    if not landmarks_data:
        return {"quality_score": 0, "is_valid": False, "issues": ["No landmarks data provided."]}

    fps = float(video_metadata.get("fps") or 30.0)
    coords = []
    confidences = []
    for frame in landmarks_data:
        lm = frame.get("landmarks")
        if lm is None or len(lm) != 33:
            continue
        coords.append(np.array([[p[0], p[1], p[2]] for p in lm], dtype=float))
        confidences.append(np.array([p[3] for p in lm], dtype=float))

    if not coords:
        return {"quality_score": 0, "is_valid": False, "issues": ["No valid frames found."]}

    coords_arr = np.stack(coords, axis=0)
    conf_arr = np.stack(confidences, axis=0)

    key_idxs = [
        JAVELIN_KEY_JOINTS["shoulders"][0],
        JAVELIN_KEY_JOINTS["shoulders"][1],
        JAVELIN_KEY_JOINTS["elbows"][0],
        JAVELIN_KEY_JOINTS["elbows"][1],
        JAVELIN_KEY_JOINTS["wrists"][0],
        JAVELIN_KEY_JOINTS["wrists"][1],
    ]

    presence_score, presence_issues = _presence_score(conf_arr, key_idxs)

    distance_pairs = [
        (JAVELIN_KEY_JOINTS["shoulders"][0], JAVELIN_KEY_JOINTS["elbows"][0]),
        (JAVELIN_KEY_JOINTS["shoulders"][1], JAVELIN_KEY_JOINTS["elbows"][1]),
        (JAVELIN_KEY_JOINTS["elbows"][0], JAVELIN_KEY_JOINTS["wrists"][0]),
        (JAVELIN_KEY_JOINTS["elbows"][1], JAVELIN_KEY_JOINTS["wrists"][1]),
    ]
    stability_score, stability_issues = _distance_stability(coords_arr, distance_pairs)

    # Phase detection.
    phase_result = detect_throw_phases(coords_arr, fps)
    phase_detected = phase_result.release_frame is not None
    phase_score = 1.0 if phase_detected else 0.0
    phase_issues = []
    if not phase_detected:
        phase_issues.append("Throw phases not detected (no release frame).")

    smooth_score, smooth_issues = _temporal_smoothness(coords_arr)

    # Anatomical plausibility across frames.
    plausible_flags = [check_anatomical_plausibility(f) for f in coords_arr]
    plausible_ratio = float(np.mean(plausible_flags))
    anatomy_score = plausible_ratio
    anatomy_issues = []
    if plausible_ratio < 0.9:
        bad_frames = [i for i, ok in enumerate(plausible_flags) if not ok]
        if bad_frames:
            anatomy_issues.append(f"Anatomical plausibility failed in frames {bad_frames[:5]}{'...' if len(bad_frames)>5 else ''}.")

    issues = presence_issues + stability_issues + phase_issues + smooth_issues + anatomy_issues

    quality_score = (
        0.30 * presence_score
        + 0.30 * anatomy_score
        + 0.30 * phase_score
        + 0.10 * smooth_score
    ) * 100.0

    is_valid = quality_score >= 80.0 and presence_score >= VALID_FRAME_THRESHOLD and phase_detected

    if issues:
        logger.warning("Pose quality issues: %s", "; ".join(issues))

    return {
        "quality_score": round(float(quality_score), 2),
        "is_valid": bool(is_valid),
        "issues": issues,
        # Ensure this payload is JSON serializable (webapp persists it alongside pose JSON).
        "phase_result": asdict(phase_result),
    }


def plot_quality_issues(mean_jumps: np.ndarray, issue_frames: List[int]) -> plt.Figure:
    """Visualize temporal jumps and highlight problematic frames."""
    fig, ax = plt.subplots()
    ax.plot(mean_jumps, label="mean joint jump")
    if issue_frames:
        ax.scatter(issue_frames, mean_jumps[issue_frames], color="red", label="issues")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Displacement (norm)")
    ax.set_title("Temporal consistency")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig
