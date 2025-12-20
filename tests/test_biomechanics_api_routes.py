from __future__ import annotations

import json

from flask import Flask, g

from javelin_tracker import storage


def _make_app_with_user(user_id: str = "user1") -> Flask:
    app = Flask(__name__)

    @app.before_request
    def _inject_user() -> None:
        g.user = {"id": user_id, "role": "head_coach", "name": "Test Coach"}

    return app


def test_get_session_biomechanics_complete_returns_cached(monkeypatch, tmp_path):
    from javelin_tracker.webapp import routes_biomechanics as rb

    monkeypatch.setattr(rb, "BASE_DIR", tmp_path)
    monkeypatch.setattr(rb, "BIOMECH_DIR", tmp_path / "biomech")
    monkeypatch.setattr(rb, "VIDEO_DIR", rb.BIOMECH_DIR / "videos")
    monkeypatch.setattr(rb, "DB_PATH", rb.BIOMECH_DIR / "jobs.db")
    rb._init_db()

    user_id = "user1"
    session_id = "sess1"
    sessions_file = rb.BASE_DIR / "data" / "webapp" / "userspace" / user_id / "sessions.json"
    sessions_file.parent.mkdir(parents=True, exist_ok=True)

    metrics_dir = tmp_path / "results" / session_id
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "metrics.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    (metrics_dir / "comparison_report.json").write_text(json.dumps({"overall_score": 80}), encoding="utf-8")
    (metrics_dir / "feedback.json").write_text(json.dumps([{"metric_name": "x", "feedback_text": "y"}]), encoding="utf-8")

    sessions_file.write_text(
        json.dumps(
            [
                {
                    "id": session_id,
                    "date": "2024-01-01",
                    "best": 60.0,
                    "throws": [60.0],
                    "duration_minutes": 0.0,
                    "load": 0.0,
                    "athlete": "alpha",
                    "event": "javelin",
                    "schema_version": 2,
                    "video_id": "job1",
                    "biomechanics_analysis_id": "job1",
                    "biomechanics_status": "complete",
                    "biomechanics_result_path": str(metrics_dir / "metrics.json"),
                }
            ]
        ),
        encoding="utf-8",
    )

    app = _make_app_with_user(user_id)
    rb.register_biomechanics_api(app)
    client = app.test_client()

    resp = client.get(f"/api/sessions/{session_id}/biomechanics")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert isinstance(payload, dict)
    assert payload["metrics"]["ok"] is True
    assert payload["comparison"]["overall_score"] == 80
    assert payload["feedback"][0]["metric_name"] == "x"


def test_get_session_biomechanics_processing_returns_202(monkeypatch, tmp_path):
    from javelin_tracker.webapp import routes_biomechanics as rb

    monkeypatch.setattr(rb, "BASE_DIR", tmp_path)
    monkeypatch.setattr(rb, "BIOMECH_DIR", tmp_path / "biomech")
    monkeypatch.setattr(rb, "VIDEO_DIR", rb.BIOMECH_DIR / "videos")
    monkeypatch.setattr(rb, "DB_PATH", rb.BIOMECH_DIR / "jobs.db")
    rb._init_db()

    user_id = "user1"
    session_id = "sess1"
    job_id = "job123"
    sessions_file = rb.BASE_DIR / "data" / "webapp" / "userspace" / user_id / "sessions.json"
    sessions_file.parent.mkdir(parents=True, exist_ok=True)
    sessions_file.write_text(
        json.dumps(
            [
                {
                    "id": session_id,
                    "date": "2024-01-01",
                    "best": 60.0,
                    "throws": [60.0],
                    "duration_minutes": 0.0,
                    "load": 0.0,
                    "athlete": "alpha",
                    "event": "javelin",
                    "schema_version": 2,
                    "video_id": job_id,
                    "biomechanics_analysis_id": job_id,
                    "biomechanics_status": "pending",
                }
            ]
        ),
        encoding="utf-8",
    )

    rb._set_job(job_id, session_id, "processing", 50.0)

    app = _make_app_with_user(user_id)
    rb.register_biomechanics_api(app)
    client = app.test_client()

    resp = client.get(f"/api/sessions/{session_id}/biomechanics")
    assert resp.status_code == 202
    payload = resp.get_json()
    assert payload["status"] == "processing"
    assert payload["percent_complete"] == 50.0


def test_get_session_biomechanics_not_analyzed_returns_404(monkeypatch, tmp_path):
    from javelin_tracker.webapp import routes_biomechanics as rb

    monkeypatch.setattr(rb, "BASE_DIR", tmp_path)
    monkeypatch.setattr(rb, "BIOMECH_DIR", tmp_path / "biomech")
    monkeypatch.setattr(rb, "VIDEO_DIR", rb.BIOMECH_DIR / "videos")
    monkeypatch.setattr(rb, "DB_PATH", rb.BIOMECH_DIR / "jobs.db")
    rb._init_db()

    user_id = "user1"
    session_id = "sess1"
    sessions_file = rb.BASE_DIR / "data" / "webapp" / "userspace" / user_id / "sessions.json"
    sessions_file.parent.mkdir(parents=True, exist_ok=True)
    sessions_file.write_text(
        json.dumps(
            [
                {
                    "id": session_id,
                    "date": "2024-01-01",
                    "best": 60.0,
                    "throws": [60.0],
                    "duration_minutes": 0.0,
                    "load": 0.0,
                    "athlete": "alpha",
                    "event": "javelin",
                    "schema_version": 2,
                    "biomechanics_status": None,
                }
            ]
        ),
        encoding="utf-8",
    )

    app = _make_app_with_user(user_id)
    rb.register_biomechanics_api(app)
    client = app.test_client()

    resp = client.get(f"/api/sessions/{session_id}/biomechanics")
    assert resp.status_code == 404


def test_progress_endpoint_can_infer_job_id_from_session(monkeypatch, tmp_path):
    from javelin_tracker.webapp import routes_biomechanics as rb

    monkeypatch.setattr(rb, "BASE_DIR", tmp_path)
    monkeypatch.setattr(rb, "BIOMECH_DIR", tmp_path / "biomech")
    monkeypatch.setattr(rb, "VIDEO_DIR", rb.BIOMECH_DIR / "videos")
    monkeypatch.setattr(rb, "DB_PATH", rb.BIOMECH_DIR / "jobs.db")
    rb._init_db()

    user_id = "user1"
    session_id = "sess1"
    job_id = "job123"
    sessions_file = rb.BASE_DIR / "data" / "webapp" / "userspace" / user_id / "sessions.json"
    sessions_file.parent.mkdir(parents=True, exist_ok=True)
    sessions_file.write_text(
        json.dumps(
            [
                {
                    "id": session_id,
                    "date": "2024-01-01",
                    "best": 60.0,
                    "throws": [60.0],
                    "duration_minutes": 0.0,
                    "load": 0.0,
                    "athlete": "alpha",
                    "event": "javelin",
                    "schema_version": 2,
                    "video_id": job_id,
                    "biomechanics_analysis_id": job_id,
                    "biomechanics_status": "pending",
                }
            ]
        ),
        encoding="utf-8",
    )

    rb._set_job(job_id, session_id, "processing", 50.0)

    app = _make_app_with_user(user_id)
    rb.register_biomechanics_api(app)
    client = app.test_client()

    resp = client.get(f"/api/sessions/{session_id}/biomechanics/progress")
    assert resp.status_code == 202
    payload = resp.get_json()
    assert payload["status"] == "processing"
    assert payload["percent_complete"] == 50.0


def test_rerun_endpoint_deletes_cached_results_and_requeues(monkeypatch, tmp_path):
    from javelin_tracker.webapp import routes_biomechanics as rb

    class DummyExecutor:
        def submit(self, fn, *args, **kwargs):
            return None

    monkeypatch.setattr(rb, "BASE_DIR", tmp_path)
    monkeypatch.setattr(rb, "BIOMECH_DIR", tmp_path / "biomech")
    monkeypatch.setattr(rb, "VIDEO_DIR", rb.BIOMECH_DIR / "videos")
    monkeypatch.setattr(rb, "DB_PATH", rb.BIOMECH_DIR / "jobs.db")
    monkeypatch.setattr(rb, "EXECUTOR", DummyExecutor())
    rb._init_db()

    user_id = "user1"
    session_id = "sess1"
    sessions_file = rb.BASE_DIR / "data" / "webapp" / "userspace" / user_id / "sessions.json"
    sessions_file.parent.mkdir(parents=True, exist_ok=True)

    metrics_path = rb.BIOMECH_DIR / "sessions" / session_id / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps({"ok": True}), encoding="utf-8")

    raw_video = rb.VIDEO_DIR / session_id / "raw.mp4"
    raw_video.parent.mkdir(parents=True, exist_ok=True)
    raw_video.write_bytes(b"dummy")

    sessions_file.write_text(
        json.dumps(
            [
                {
                    "id": session_id,
                    "date": "2024-01-01",
                    "best": 60.0,
                    "throws": [60.0],
                    "duration_minutes": 0.0,
                    "load": 0.0,
                    "athlete": "alpha",
                    "event": "javelin",
                    "schema_version": 2,
                    "video_id": "job1",
                    "biomechanics_analysis_id": "job1",
                    "biomechanics_status": "complete",
                    "biomechanics_result_path": str(metrics_path),
                }
            ]
        ),
        encoding="utf-8",
    )

    app = _make_app_with_user(user_id)
    rb.register_biomechanics_api(app)
    client = app.test_client()

    resp = client.post(f"/api/sessions/{session_id}/biomechanics/rerun", json={})
    assert resp.status_code == 202
    payload = resp.get_json()
    assert payload["session_id"] == session_id
    assert payload["video_id"]
    assert not metrics_path.parent.exists()

    reloaded = json.loads(sessions_file.read_text(encoding="utf-8"))
    assert reloaded[0]["biomechanics_status"] == "pending"
    assert reloaded[0]["video_id"] == payload["video_id"]


def test_elite_reference_endpoint_returns_reference(monkeypatch, tmp_path):
    from javelin_tracker.webapp import routes_biomechanics as rb

    monkeypatch.setattr(rb, "BASE_DIR", tmp_path)
    monkeypatch.setattr(rb, "BIOMECH_DIR", tmp_path / "biomech")
    monkeypatch.setattr(rb, "VIDEO_DIR", rb.BIOMECH_DIR / "videos")
    monkeypatch.setattr(rb, "DB_PATH", rb.BIOMECH_DIR / "jobs.db")
    rb._init_db()

    reference_path = rb.BASE_DIR / "data" / "biomechanics" / "elite_database" / "reference_profile_overall.json"
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    reference_path.write_text(json.dumps({"metrics": {"foo": {"mean": 1.0}}}), encoding="utf-8")

    app = _make_app_with_user()
    rb.register_biomechanics_api(app)
    client = app.test_client()

    resp = client.get("/api/elite-db/reference/overall")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["metrics"]["foo"]["mean"] == 1.0


def test_set_athlete_style_endpoint_updates_database(monkeypatch, tmp_path):
    from javelin_tracker.webapp import routes_biomechanics as rb

    monkeypatch.setenv("THROWS_TRACKER_DB_FILE", str(tmp_path / "db.sqlite"))

    monkeypatch.setattr(rb, "BASE_DIR", tmp_path)
    monkeypatch.setattr(rb, "BIOMECH_DIR", tmp_path / "biomech")
    monkeypatch.setattr(rb, "VIDEO_DIR", rb.BIOMECH_DIR / "videos")
    monkeypatch.setattr(rb, "DB_PATH", rb.BIOMECH_DIR / "jobs.db")
    rb._init_db()

    with storage.open_database() as conn:
        conn.execute("INSERT INTO Athletes (name) VALUES (?)", ("Alice Smith",))
        conn.commit()

    app = _make_app_with_user("user1")
    rb.register_biomechanics_api(app)
    client = app.test_client()

    resp = client.post("/api/elite-db/set-style/Alice%20Smith", json={"throwing_style": "finnish"})
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["athlete"] == "Alice Smith"
    assert payload["throwing_style"] == "finnish"

    with storage.open_database(readonly=True) as conn:
        row = conn.execute("SELECT throwing_style FROM Athletes WHERE name = ?", ("Alice Smith",)).fetchone()
        assert row is not None
        assert row[0] == "finnish"


def test_frontend_viewer_jsx_is_served(monkeypatch, tmp_path):
    from javelin_tracker.webapp import routes_biomechanics as rb

    monkeypatch.setattr(rb, "BASE_DIR", tmp_path)
    monkeypatch.setattr(rb, "BIOMECH_DIR", tmp_path / "biomech")
    monkeypatch.setattr(rb, "VIDEO_DIR", rb.BIOMECH_DIR / "videos")
    monkeypatch.setattr(rb, "DB_PATH", rb.BIOMECH_DIR / "jobs.db")
    rb._init_db()

    app = _make_app_with_user("user1")
    rb.register_biomechanics_api(app)
    client = app.test_client()

    resp = client.get("/frontend/BiomechanicsViewer.jsx")
    assert resp.status_code == 200
    assert b"BiomechanicsViewer" in resp.data


def test_biomechanics_video_endpoint_serves_raw_video(monkeypatch, tmp_path):
    from javelin_tracker.webapp import routes_biomechanics as rb

    monkeypatch.setattr(rb, "BASE_DIR", tmp_path)
    monkeypatch.setattr(rb, "BIOMECH_DIR", tmp_path / "biomech")
    monkeypatch.setattr(rb, "VIDEO_DIR", rb.BIOMECH_DIR / "videos")
    monkeypatch.setattr(rb, "DB_PATH", rb.BIOMECH_DIR / "jobs.db")
    rb._init_db()

    session_id = "sess1"
    raw_video = rb.VIDEO_DIR / session_id / "raw.mp4"
    raw_video.parent.mkdir(parents=True, exist_ok=True)
    raw_video.write_bytes(b"dummy")

    app = _make_app_with_user("user1")
    rb.register_biomechanics_api(app)
    client = app.test_client()

    resp = client.get(f"/api/sessions/{session_id}/biomechanics/video")
    assert resp.status_code in (200, 206)
    assert resp.mimetype == "video/mp4"


def test_biomechanics_pose_endpoint_returns_slim_payload(monkeypatch, tmp_path):
    from javelin_tracker.webapp import routes_biomechanics as rb

    monkeypatch.setattr(rb, "BASE_DIR", tmp_path)
    monkeypatch.setattr(rb, "BIOMECH_DIR", tmp_path / "biomech")
    monkeypatch.setattr(rb, "VIDEO_DIR", rb.BIOMECH_DIR / "videos")
    monkeypatch.setattr(rb, "DB_PATH", rb.BIOMECH_DIR / "jobs.db")
    rb._init_db()

    user_id = "user1"
    session_id = "sess1"
    job_id = "job123"
    sessions_file = rb.BASE_DIR / "data" / "webapp" / "userspace" / user_id / "sessions.json"
    sessions_file.parent.mkdir(parents=True, exist_ok=True)
    sessions_file.write_text(
        json.dumps(
            [
                {
                    "id": session_id,
                    "date": "2024-01-01",
                    "best": 60.0,
                    "throws": [60.0],
                    "duration_minutes": 0.0,
                    "load": 0.0,
                    "athlete": "alpha",
                    "event": "javelin",
                    "schema_version": 2,
                    "video_id": job_id,
                    "biomechanics_analysis_id": job_id,
                    "biomechanics_status": "complete",
                }
            ]
        ),
        encoding="utf-8",
    )

    pose_path = rb.VIDEO_DIR / job_id / "pose_data.json"
    pose_path.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for idx in range(5):
        frames.append(
            {
                "frame_idx": idx,
                "timestamp_ms": float(idx * 20.0),
                "landmarks": [(0.1, 0.2, 0.0, 1.0) for _ in range(33)],
                "valid": True,
            }
        )
    pose_path.write_text(
        json.dumps(
            {
                "video_id": job_id,
                "video_metadata": {"fps": 50.0, "width": 100, "height": 100},
                "frames": frames,
            }
        ),
        encoding="utf-8",
    )

    app = _make_app_with_user(user_id)
    rb.register_biomechanics_api(app)
    client = app.test_client()

    resp = client.get(f"/api/sessions/{session_id}/biomechanics/pose?stride=2")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["stride"] == 2
    assert payload["video_id"] == job_id
    assert isinstance(payload["frames"], list)
    assert len(payload["frames"]) == 3
    assert "landmarks" in payload["frames"][0]
