import importlib
import os
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("mediapipe")


def _reload_config_with_env(monkeypatch, **env_vars):
    import javelin_tracker.biomechanics.config as config

    for key, value in env_vars.items():
        monkeypatch.setenv(key, str(value))
    reloaded = importlib.reload(config)
    return reloaded


def test_config_env_overrides(monkeypatch):
    # Override confidence threshold and smoothing window via env, then reload.
    config = _reload_config_with_env(
        monkeypatch,
        JAVELIN_CONFIDENCE_THRESHOLD="0.9",
        JAVELIN_FRAME_SMOOTHING_WINDOW="7",
    )
    assert config.CONFIDENCE_THRESHOLD == 0.9
    assert config.FRAME_SMOOTHING_WINDOW == 7
    # Ensure MediaPipe indices validation runs without warning on default enum.
    config.validate_mediapipe_indices()

    # Clean up by reloading with env cleared.
    monkeypatch.delenv("JAVELIN_CONFIDENCE_THRESHOLD", raising=False)
    monkeypatch.delenv("JAVELIN_FRAME_SMOOTHING_WINDOW", raising=False)
    importlib.reload(config)


def test_key_joints_match_mediapipe_enums():
    import mediapipe as mp
    from javelin_tracker.biomechanics.config import JAVELIN_KEY_JOINTS

    if not hasattr(mp, "solutions") or not getattr(mp, "solutions", None) or not hasattr(mp.solutions, "holistic"):
        pytest.skip("MediaPipe solutions API is unavailable in this environment.")

    pose_enum = mp.solutions.holistic.PoseLandmark
    assert JAVELIN_KEY_JOINTS["shoulders"] == (
        pose_enum.LEFT_SHOULDER.value,
        pose_enum.RIGHT_SHOULDER.value,
    )
    assert JAVELIN_KEY_JOINTS["elbows"] == (
        pose_enum.LEFT_ELBOW.value,
        pose_enum.RIGHT_ELBOW.value,
    )
    assert JAVELIN_KEY_JOINTS["wrists"] == (
        pose_enum.LEFT_WRIST.value,
        pose_enum.RIGHT_WRIST.value,
    )
    assert JAVELIN_KEY_JOINTS["hips"] == (
        pose_enum.LEFT_HIP.value,
        pose_enum.RIGHT_HIP.value,
    )
    assert JAVELIN_KEY_JOINTS["knees"] == (
        pose_enum.LEFT_KNEE.value,
        pose_enum.RIGHT_KNEE.value,
    )


def test_pose_detector_process_frame():
    from javelin_tracker.biomechanics.pose_estimation import PoseDetector

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with PoseDetector() as detector:
        result1 = detector.process_frame(dummy_frame)
        result2 = detector.process_frame(dummy_frame)

    assert result1["frame_idx"] == 0
    assert result2["frame_idx"] == 1
    assert len(result1["landmarks"]) == 33
    assert all(len(lm) == 4 for lm in result1["landmarks"])
    assert set(result1["hands_data"].keys()) == {"left", "right"}
    assert isinstance(result1["pose_confidence_avg"], float)


def _make_temp_video(tmp_path: Path, num_frames: int = 5) -> Path:
    path = tmp_path / "sample.avi"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fps = 30
    writer = cv2.VideoWriter(str(path), fourcc, fps, (64, 64))
    for _ in range(num_frames):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_video_metadata_and_extraction(tmp_path):
    from javelin_tracker.biomechanics.utils import extract_frames, get_video_metadata, validate_video_readable

    video_path = _make_temp_video(tmp_path)

    validation = validate_video_readable(video_path)
    assert validation["readable"] is True
    assert validation["width"] == 64
    assert validation["height"] == 64

    metadata = get_video_metadata(video_path)
    assert metadata["width"] == 64
    assert metadata["height"] == 64
    assert metadata["total_frames"] >= 5

    frames = list(extract_frames(video_path, fps=30))
    assert len(frames) == 5
    assert frames[0][0] == 0
    assert isinstance(frames[0][1], float)
    assert frames[0][2].shape == (64, 64, 3)

    frames_rgb_resized = list(extract_frames(video_path, output_format="RGB", resize_to=(32, 32)))
    assert frames_rgb_resized[0][2].shape == (32, 32, 3)


def test_validate_video_readable_errors_on_corrupted(tmp_path):
    from javelin_tracker.biomechanics.utils import validate_video_readable

    corrupted = tmp_path / "corrupted.avi"
    corrupted.write_text("not a video file", encoding="utf-8")

    with pytest.raises((ValueError, RuntimeError)):
        validate_video_readable(corrupted)


def test_smooth_landmarks_and_trajectory():
    from javelin_tracker.biomechanics.utils import smooth_landmarks, smooth_trajectory

    n_frames = 4
    # Create a simple ramp for landmarks (n_frames, 33, 3)
    base = np.linspace(0, 1, n_frames)[:, None, None]
    landmarks = np.repeat(base, 33, axis=1)
    landmarks = np.repeat(landmarks, 3, axis=2)

    smoothed = smooth_landmarks(landmarks, window_length=5, polyorder=2)
    assert smoothed.shape == landmarks.shape

    # Confidence weighting should preserve low-confidence frames.
    confidence = np.zeros((n_frames, 33))
    smoothed_conf = smooth_landmarks(landmarks, window_length=5, polyorder=2, confidence=confidence)
    assert np.allclose(smoothed_conf, landmarks)

    # Trajectory smoothing shape check.
    traj = np.stack([np.arange(n_frames), np.arange(n_frames)], axis=1).astype(float)
    traj_smoothed = smooth_trajectory(traj, window_length=3, polyorder=1)
    assert traj_smoothed.shape == traj.shape


def test_fill_low_confidence_landmarks_interpolates():
    from javelin_tracker.biomechanics.utils.filtering import fill_low_confidence_landmarks

    n_frames = 5
    coords = np.zeros((n_frames, 33, 3), dtype=float)
    conf = np.ones((n_frames, 33), dtype=float)

    # Joint 0 drops out briefly at frame 2.
    coords[:, 0, 0] = [0.0, 1.0, 0.0, 3.0, 4.0]
    conf[:, 0] = [1.0, 1.0, 0.0, 1.0, 1.0]

    filled = fill_low_confidence_landmarks(coords, conf, confidence_threshold=0.5)
    assert filled.shape == coords.shape
    assert filled[2, 0, 0] == pytest.approx(2.0)


def test_bone_length_outlier_mask_flags_distal_joint():
    from javelin_tracker.biomechanics.utils.filtering import bone_length_outlier_mask

    n_frames = 12
    coords = np.zeros((n_frames, 33, 3), dtype=float)
    conf = np.ones((n_frames, 33), dtype=float)

    # Right shoulder (12) at origin; right elbow (14) mostly at distance ~1.
    coords[:, 12, :] = (0.0, 0.0, 0.0)
    coords[:, 14, :] = (1.0, 0.0, 0.0)
    coords[5, 14, :] = (0.1, 0.0, 0.0)  # implausibly short upper arm for one frame

    mask = bone_length_outlier_mask(coords, conf, confidence_threshold=0.5, min_samples=5)
    assert mask.shape == (n_frames, 33)
    assert bool(mask[5, 14]) is True


def test_joint_displacement_outlier_mask_flags_snap():
    from javelin_tracker.biomechanics.utils.filtering import joint_displacement_outlier_mask

    n_frames = 20
    coords = np.zeros((n_frames, 33, 3), dtype=float)
    conf = np.ones((n_frames, 33), dtype=float)

    # Joint 16 (right wrist) moves smoothly, then snaps far away for one frame.
    coords[:, 16, 0] = np.linspace(0.2, 0.3, n_frames)
    coords[:, 16, 1] = 0.5
    coords[10, 16, :] = (0.95, 0.05, 0.0)

    mask = joint_displacement_outlier_mask(
        coords,
        conf,
        confidence_threshold=0.5,
        joint_indices=(16,),
        absolute_threshold=0.10,
        min_samples=8,
    )
    assert mask.shape == (n_frames, 33)
    assert bool(mask[10, 16]) is True


def test_kalman_smooth_landmarks_rejects_large_jump():
    from javelin_tracker.biomechanics.utils.filtering import kalman_smooth_landmarks

    n_frames = 20
    coords = np.zeros((n_frames, 33, 3), dtype=float)
    conf = np.ones((n_frames, 33), dtype=float)

    # Joint 0 is steady, but has one extreme outlier.
    coords[:, 0, 0] = 0.0
    coords[10, 0, 0] = 10.0

    smoothed = kalman_smooth_landmarks(coords, conf, fps=30.0, confidence_threshold=0.5)
    assert smoothed.shape == coords.shape
    # Outlier should be strongly damped (gated).
    assert smoothed[10, 0, 0] < 2.0


def test_pose_detector_thread_safety():
    from javelin_tracker.biomechanics.pose_estimation import PoseDetector

    dummy_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    results = []

    def worker(det: PoseDetector, idx: int) -> None:
        res = det.process_frame(dummy_frame)
        results.append((idx, res["frame_idx"]))

    with PoseDetector() as detector:
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            for i in range(4):
                executor.submit(worker, detector, i)

    frame_indices = sorted(frame_idx for _, frame_idx in results)
    assert frame_indices == [0, 1, 2, 3]
