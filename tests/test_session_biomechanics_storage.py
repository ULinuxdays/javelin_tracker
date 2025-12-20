from __future__ import annotations

import json

from javelin_tracker import storage


def test_session_migration_adds_biomechanics_fields(monkeypatch, tmp_path):
    sessions_file = tmp_path / "sessions.json"
    monkeypatch.setenv("JAVELIN_TRACKER_SESSIONS_FILE", str(sessions_file))

    legacy_rows = [
        {
            "date": "2024-01-01",
            "best": 60.0,
            "throws": [59.5, 60.0],
            "rpe": 5,
            "duration_minutes": 30,
            "athlete": "alpha",
        }
    ]
    sessions_file.write_text(json.dumps(legacy_rows), encoding="utf-8")

    sessions = storage.load_sessions()
    assert sessions
    migrated = sessions[0]
    for key in (
        "video_id",
        "biomechanics_analysis_id",
        "biomechanics_status",
        "biomechanics_timestamp",
        "biomechanics_result_path",
    ):
        assert key in migrated
        assert migrated[key] is None


def test_update_session_by_id_persists_patch(tmp_path):
    sessions_file = tmp_path / "sessions.json"
    sessions_file.write_text(
        json.dumps(
            [
                {
                    "id": "sess1",
                    "date": "2024-01-01",
                    "best": 60.0,
                    "throws": [60.0],
                    "duration_minutes": 0.0,
                    "load": 0.0,
                    "athlete": "alpha",
                    "event": "javelin",
                    "schema_version": 2,
                }
            ]
        ),
        encoding="utf-8",
    )

    updated = storage.update_session_by_id(
        "sess1",
        {"biomechanics_status": "pending", "video_id": "vid123"},
        sessions_file=sessions_file,
    )
    assert updated is not None
    assert updated["biomechanics_status"] == "pending"
    assert updated["video_id"] == "vid123"

    reloaded = storage.get_session_by_id("sess1", sessions_file=sessions_file)
    assert reloaded is not None
    assert reloaded["biomechanics_status"] == "pending"
    assert reloaded["video_id"] == "vid123"


def test_get_session_biomechanics_loads_artifacts(tmp_path):
    sessions_file = tmp_path / "sessions.json"
    metrics_dir = tmp_path / "biomech" / "sessions" / "sess1"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "metrics.json"
    report_path = metrics_dir / "comparison_report.json"
    feedback_path = metrics_dir / "feedback.json"

    metrics_path.write_text(json.dumps({"throw_metrics": {"release_metrics": {"wrist_speed": 12.3}}}), encoding="utf-8")
    report_path.write_text(json.dumps({"overall_score": 85, "top_issues": []}), encoding="utf-8")
    feedback_path.write_text(json.dumps([{"metric_name": "wrist_speed", "feedback_text": "Keep the hand fast."}]), encoding="utf-8")

    sessions_file.write_text(
        json.dumps(
            [
                {
                    "id": "sess1",
                    "date": "2024-01-01",
                    "best": 60.0,
                    "throws": [60.0],
                    "duration_minutes": 0.0,
                    "load": 0.0,
                    "athlete": "alpha",
                    "event": "javelin",
                    "schema_version": 2,
                    "biomechanics_status": "complete",
                    "biomechanics_result_path": str(metrics_path),
                }
            ]
        ),
        encoding="utf-8",
    )

    payload = storage.get_session_biomechanics("sess1", sessions_file=sessions_file)
    assert payload is not None
    assert payload["biomechanics_status"] == "complete"
    assert isinstance(payload.get("metrics"), dict)
    assert payload["metrics"]["throw_metrics"]["release_metrics"]["wrist_speed"] == 12.3
    assert payload["comparison_report"]["overall_score"] == 85
    assert payload["feedback"][0]["metric_name"] == "wrist_speed"

