import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")


def _blank_landmarks() -> list[tuple[float, float, float, float]]:
    return [(0.0, 0.0, 0.0, 0.0) for _ in range(33)]


def test_optical_flow_refiner_replaces_pose_jump():
    from javelin_tracker.biomechanics.pose_estimation.optical_flow_refiner import OpticalFlowLandmarkRefiner

    width = 100
    height = 100

    frame0 = np.zeros((height, width, 3), dtype=np.uint8)
    frame1 = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.circle(frame0, (30, 30), 3, (255, 255, 255), -1)
    cv2.circle(frame1, (35, 30), 3, (255, 255, 255), -1)

    prev_landmarks = _blank_landmarks()
    prev_landmarks[16] = (0.30, 0.30, 0.0, 1.0)

    # Simulate a bad pose estimate that "snaps" far away.
    current_landmarks = _blank_landmarks()
    # In the real pipeline, segmentation/bone-length gating often down-weights
    # snapped landmarks before optical flow refinement runs.
    current_landmarks[16] = (0.80, 0.80, 0.0, 0.3)

    refiner = OpticalFlowLandmarkRefiner(enabled=True, confidence_threshold=0.5, replace_distance_fraction=0.05)
    _ = refiner.refine(frame0, prev_landmarks)
    refined = refiner.refine(frame1, current_landmarks)

    x, y, _z, conf = refined[16]
    assert conf >= 0.5
    assert x == pytest.approx(0.35, abs=0.06)
    assert y == pytest.approx(0.30, abs=0.06)


def test_correct_symmetric_swaps_prefers_continuity():
    from javelin_tracker.biomechanics.pose_estimation.optical_flow_refiner import correct_symmetric_swaps

    width = 100
    height = 100
    prev = _blank_landmarks()
    cur = _blank_landmarks()

    # Previous: left wrist on the left, right wrist on the right.
    prev[15] = (0.2, 0.5, 0.0, 1.0)
    prev[16] = (0.8, 0.5, 0.0, 1.0)

    # Current: swapped (a known failure mode during occlusions).
    cur[15] = (0.8, 0.5, 0.0, 1.0)
    cur[16] = (0.2, 0.5, 0.0, 1.0)

    swaps = correct_symmetric_swaps(cur, prev, width=width, height=height, pairs=((15, 16),), min_improvement_px=5.0)
    assert swaps == 1
    assert cur[15][0] == pytest.approx(0.2)
    assert cur[16][0] == pytest.approx(0.8)
