from __future__ import annotations

from javelin_tracker.storage import (
    get_daily_routine,
    log_strength_workout,
    log_throw_distance,
    open_database,
    record_workout_results,
)


def _create_athlete(conn, name: str) -> int:
    cursor = conn.execute(
        """
        INSERT INTO Athletes (name, height_cm, weight_kg, bmi)
        VALUES (?, ?, ?, ?)
        """,
        (name, 185.0, 90.0, 90.0 / (1.85**2)),
    )
    conn.commit()
    return cursor.lastrowid


def test_generate_workout_adjusts_after_improvement(tmp_path, monkeypatch):
    monkeypatch.setenv("THROWS_TRACKER_DATA_DIR", str(tmp_path))
    with open_database() as conn:
        athlete_id = _create_athlete(conn, "alpha")

    # Baseline logs (moderate loads/distances).
    log_strength_workout(athlete_id, "2024-01-01", "back squat", 120.0, 5)
    log_throw_distance(athlete_id, "2024-01-01", "javelin", 60.0)

    baseline_routine = get_daily_routine(athlete_id)
    squat_entry = next(
        item for item in baseline_routine if item["name"] == "back squat"
    )
    baseline_weight = squat_entry["target_weight_kg"]

    # Athlete improves (heavier lift, longer throw) and logs actual results.
    improved_results = [
        {
            "athlete_id": athlete_id,
            "name": "back squat",
            "actual_weight_kg": 140.0,
            "actual_reps": 5,
            "date": "2024-01-08",
        }
    ]
    record_workout_results(improved_results)
    log_throw_distance(athlete_id, "2024-01-08", "javelin", 65.0)

    updated_routine = get_daily_routine(athlete_id)
    updated_squat = next(
        item for item in updated_routine if item["name"] == "back squat"
    )

    assert updated_squat["target_weight_kg"] > baseline_weight, "Expected higher squat load after improvement."
    weighted_drill = [item for item in updated_routine if item["name"] == "Weighted Ball Throws"]
    assert weighted_drill, "Expected weighted throw drill to be recommended after improvement."
