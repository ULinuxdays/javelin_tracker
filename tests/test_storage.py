from __future__ import annotations

import json

import pytest

from javelin_tracker.storage import load_sessions


def test_load_sessions_migrates_legacy_records(monkeypatch, tmp_path):
    sessions_file = tmp_path / "legacy_sessions.json"
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

    sessions = load_sessions()
    assert sessions, "Expected migration to load legacy rows"
    migrated = sessions[0]
    assert migrated["event"] == "javelin"
    assert migrated["schema_version"] >= 2
    assert migrated["load"] == pytest.approx(150.0)
