import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from flask import Flask

pytest.importorskip("cv2")
pytest.importorskip("mediapipe")


def test_pose_pipeline_happy_path(monkeypatch, tmp_path):
    from javelin_tracker.biomechanics.pose_estimation import pipeline

    video = tmp_path / "vid.mp4"
    video.write_bytes(b"dummy")

    monkeypatch.setattr(pipeline, "validate_video_readable", lambda p: {"total_frames": 2, "width": 1, "height": 1})
    monkeypatch.setattr(pipeline, "get_video_metadata", lambda p: {"fps": 30, "total_frames": 2, "width": 1, "height": 1})

    def fake_extract(path, fps):
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        yield 0, 0.0, frame
        yield 1, 33.3, frame

    monkeypatch.setattr(pipeline, "extract_frames", fake_extract)

    class FakeDetector:
        def __init__(self):
            self.count = 0

        def process_frame(self, frame, *, timestamp_ms=None):
            idx = self.count
            self.count += 1
            landmarks = [(0.0, 0.0, 0.0, 1.0) for _ in range(33)]
            return {"frame_idx": idx, "timestamp": 0.0, "landmarks": landmarks, "hands_data": {"left": [], "right": []}, "pose_confidence_avg": 1.0}

        def close(self):
            pass

    monkeypatch.setattr(pipeline, "PoseDetector", FakeDetector)
    monkeypatch.setattr(pipeline, "smooth_landmarks", lambda coords_array, **_: coords_array)

    p = pipeline.PosePipeline()
    result = p.process_video(video, "vid1", tmp_path)
    assert result["status"] == "success"
    output = Path(result["output_path"])
    assert output.exists()
    payload = json.loads(output.read_text())
    assert payload["video_id"] == "vid1"
    assert len(payload["frames"]) == 2


def test_pose_pipeline_reports_progress_callback(monkeypatch, tmp_path):
    from javelin_tracker.biomechanics.pose_estimation import pipeline

    video = tmp_path / "vid.mp4"
    video.write_bytes(b"dummy")

    monkeypatch.setattr(pipeline, "validate_video_readable", lambda p: {"total_frames": 2, "width": 1, "height": 1})
    monkeypatch.setattr(pipeline, "get_video_metadata", lambda p: {"fps": 30, "total_frames": 2, "width": 1, "height": 1})

    def fake_extract(path, fps):
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        yield 0, 0.0, frame
        yield 1, 33.3, frame

    monkeypatch.setattr(pipeline, "extract_frames", fake_extract)

    class FakeDetector:
        def process_frame(self, frame, *, timestamp_ms=None):
            landmarks = [(0.0, 0.0, 0.0, 1.0) for _ in range(33)]
            return {
                "frame_idx": 0,
                "timestamp": 0.0,
                "landmarks": landmarks,
                "hands_data": {"left": [], "right": []},
                "pose_confidence_avg": 1.0,
            }

        def close(self):
            pass

    monkeypatch.setattr(pipeline, "PoseDetector", FakeDetector)
    monkeypatch.setattr(pipeline, "smooth_landmarks", lambda coords_array, **_: coords_array)

    calls = []

    def progress_cb(done, total):
        calls.append((int(done), int(total)))

    p = pipeline.PosePipeline()
    result = p.process_video(video, "vid1", tmp_path, progress_callback=progress_cb)
    assert result["status"] == "success"
    assert calls[-1] == (2, 2)


def test_pose_pipeline_invalid_video(monkeypatch, tmp_path):
    from javelin_tracker.biomechanics.pose_estimation import pipeline

    video = tmp_path / "vid_bad.mp4"
    video.write_bytes(b"dummy")

    def raise_validation(p):
        raise ValueError("invalid video")

    monkeypatch.setattr(pipeline, "validate_video_readable", raise_validation)
    p = pipeline.PosePipeline()
    result = p.process_video(video, "vidbad", tmp_path)
    assert result["status"] == "error"


def test_phase_detection_shapes_and_indices():
    from javelin_tracker.biomechanics.metrics import detect_throw_phases

    n_frames = 10
    coords = np.zeros((n_frames, 33, 3), dtype=float)
    for i in range(n_frames):
        coords[i, :, 0] = i  # simple forward motion
    res = detect_throw_phases(coords, fps=30.0)
    assert res.release_frame is not None
    assert 0 <= res.release_frame < n_frames


def test_quality_validation_scoring():
    from javelin_tracker.biomechanics.utils import validate_pose_quality

    frames = []
    for i in range(5):
        landmarks = [(float(i), 0.0, 0.0, 1.0) for _ in range(33)]
        frames.append({"landmarks": landmarks})
    video_metadata = {"fps": 30.0}
    result = validate_pose_quality(frames, video_metadata)
    assert result["quality_score"] >= 50  # should be non-zero
    assert isinstance(result["issues"], list)


def test_flask_upload_endpoint_creates_job(monkeypatch, tmp_path):
    from javelin_tracker.webapp import routes_biomechanics as rb

    if hasattr(rb, "EXECUTOR"):
        try:
            rb.EXECUTOR.shutdown(wait=False)
        except Exception:
            pass

    rb.BIOMECH_DIR = tmp_path / "biomech"
    rb.VIDEO_DIR = rb.BIOMECH_DIR / "videos"
    rb.DB_PATH = rb.BIOMECH_DIR / "jobs.db"

    class DummyExecutor:
        def submit(self, fn, *args, **kwargs):
            fn(*args, **kwargs)

    rb.EXECUTOR = DummyExecutor()
    rb._init_db()

    def fake_process(job_id, session_id, path):
        rb._update_job(job_id, "complete", 100.0, session_id=session_id)

    monkeypatch.setattr(rb, "_process_video_job", fake_process)

    app = Flask(__name__)
    rb.register_biomechanics_api(app)
    client = app.test_client()

    data = {
        "file": (BytesIO(b"data"), "video.mp4"),
    }
    resp = client.post(f"/api/sessions/abc/upload-video", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["status"] == "processing"
    job = rb._get_job(payload["video_id"])
    assert job is not None


def test_flask_progress_endpoint_returns_status(monkeypatch, tmp_path):
    from javelin_tracker.webapp import routes_biomechanics as rb

    if hasattr(rb, "EXECUTOR"):
        try:
            rb.EXECUTOR.shutdown(wait=False)
        except Exception:
            pass

    rb.BIOMECH_DIR = tmp_path / "biomech2"
    rb.VIDEO_DIR = rb.BIOMECH_DIR / "videos"
    rb.DB_PATH = rb.BIOMECH_DIR / "jobs.db"

    class DummyExecutor:
        def submit(self, fn, *args, **kwargs):
            fn(*args, **kwargs)

    rb.EXECUTOR = DummyExecutor()
    rb._init_db()

    app = Flask(__name__)
    rb.register_biomechanics_api(app)
    client = app.test_client()

    job_id = "job123"
    rb._set_job(job_id, "sess1", "processing", 50.0)

    resp = client.get(f"/api/sessions/sess1/biomechanics/progress?video_id={job_id}")
    assert resp.status_code == 202
    data = resp.get_json()
    assert data["percent_complete"] == 50.0
    assert data["status"] == "processing"
