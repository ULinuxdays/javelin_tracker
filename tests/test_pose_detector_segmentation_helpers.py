import numpy as np
import pytest


def _blank_landmarks() -> list[tuple[float, float, float, float]]:
    return [(0.0, 0.0, 0.0, 0.0) for _ in range(33)]


def test_filter_landmarks_to_person_mask_reduces_outside_conf(monkeypatch):
    from javelin_tracker.biomechanics.pose_estimation.pose_detector import PoseDetector

    monkeypatch.setenv("THROWS_TRACKER_POSE_SEGMENTATION_THRESHOLD", "0.5")
    monkeypatch.setenv("THROWS_TRACKER_POSE_SEGMENTATION_DILATE_PX", "0")

    mask = np.zeros((10, 10), dtype=np.float32)
    mask[:, :5] = 1.0  # left half is the athlete
    binary = PoseDetector._binary_person_mask(mask)

    landmarks = _blank_landmarks()
    landmarks[0] = (0.2, 0.5, 0.0, 1.0)  # inside
    landmarks[1] = (0.8, 0.5, 0.0, 1.0)  # outside

    filtered = PoseDetector._filter_landmarks_to_person_mask(landmarks, binary, outside_confidence=0.05)
    assert filtered[0][3] == pytest.approx(1.0)
    assert filtered[1][3] == pytest.approx(0.05)


def test_bbox_from_person_mask_maps_to_full_frame(monkeypatch):
    from javelin_tracker.biomechanics.pose_estimation.pose_detector import PoseDetector

    monkeypatch.setenv("THROWS_TRACKER_POSE_SEGMENTATION_THRESHOLD", "0.5")
    monkeypatch.setenv("THROWS_TRACKER_POSE_SEGMENTATION_DILATE_PX", "0")
    monkeypatch.setenv("THROWS_TRACKER_POSE_SEGMENTATION_MIN_PIXELS", "1")

    mask = np.zeros((10, 10), dtype=np.float32)
    mask[2:6, 3:8] = 1.0
    binary = PoseDetector._binary_person_mask(mask)

    bbox = PoseDetector._bbox_from_person_mask(binary, roi_used=None, full_width=100, full_height=200)
    assert bbox == (30, 40, 80, 120)


def test_bbox_from_person_mask_maps_with_roi(monkeypatch):
    from javelin_tracker.biomechanics.pose_estimation.pose_detector import PoseDetector

    monkeypatch.setenv("THROWS_TRACKER_POSE_SEGMENTATION_THRESHOLD", "0.5")
    monkeypatch.setenv("THROWS_TRACKER_POSE_SEGMENTATION_DILATE_PX", "0")
    monkeypatch.setenv("THROWS_TRACKER_POSE_SEGMENTATION_MIN_PIXELS", "1")

    mask = np.zeros((10, 10), dtype=np.float32)
    mask[2:6, 3:8] = 1.0
    binary = PoseDetector._binary_person_mask(mask)

    roi = (50, 100, 150, 300)  # crop size 100x200
    bbox = PoseDetector._bbox_from_person_mask(binary, roi_used=roi, full_width=200, full_height=400)
    # x: 50 + 0.3*100=80, 50 + 0.8*100=130
    # y: 100 + 0.2*200=140, 100 + 0.6*200=220
    assert bbox == (80, 140, 130, 220)


def test_extract_segmentation_masks_skips_channel_mismatch(monkeypatch):
    mp = pytest.importorskip("mediapipe")
    from javelin_tracker.biomechanics.pose_estimation.pose_detector import PoseDetector

    class DummyMask:
        image_format = mp.ImageFormat.VEC32F1
        channels = 4

        def numpy_view(self):  # pragma: no cover
            raise AssertionError("numpy_view should not be called for mismatched masks")

    class DummyResults:
        segmentation_masks = [DummyMask()]

    masks = PoseDetector._extract_segmentation_masks_tasks(DummyResults())
    assert masks == []
