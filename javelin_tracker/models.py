from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional

CURRENT_SCHEMA_VERSION = 2
DEFAULT_EVENT = "javelin"
STRENGTH_EXERCISES = {
    "back squat",
    "front squat",
    "overhead squat",
    "deadlift",
    "romanian deadlift",
    "bench press",
    "incline bench press",
    "push press",
    "snatch",
    "clean",
    "clean and jerk",
    "jerk",
    "pull-up",
    "chin-up",
    "row",
    "split squat",
    "hip thrust",
    "lunge",
    "press",
}

__all__ = [
    "parse_iso_date",
    "coerce_number",
    "clamp_rpe",
    "validate_distance",
    "parse_throws",
    "parse_tags",
    "validate_duration",
    "calculate_bmi",
    "calculate_session_load",
    "is_strength_exercise",
    "Session",
    "Athlete",
    "WorkoutExercise",
    "WorkoutRoutine",
    "ValidationError",
]


class ValidationError(ValueError):
    """Raised when user-supplied data cannot be normalised safely."""


def parse_iso_date(value: Any, *, field: str = "date") -> date:
    """
    Parse user-supplied ISO-8601 dates.

    Accepts `datetime.date`, `datetime.datetime`, or strings. Raises `ValidationError`
    with a friendlier message if the payload cannot be parsed.
    """
    if isinstance(value, date):
        return value if not isinstance(value, datetime) else value.date()

    if not isinstance(value, str):
        raise ValidationError(
            f"{field} must be provided as YYYY-MM-DD text; received {value!r}."
        )

    candidate = value.strip()
    if not candidate:
        raise ValidationError(f"{field} cannot be empty.")

    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ValidationError(
            f"{field} must be a valid ISO date (YYYY-MM-DD); received {candidate!r}."
        ) from exc

    return parsed.date()


def coerce_number(
    value: Any,
    *,
    field: str = "value",
    minimum: float | None = None,
    maximum: float | None = None,
    allow_empty: bool = False,
    allow_float: bool = True,
) -> float:
    """
    Convert arbitrary input into a float with guardrails.

    The `minimum` and `maximum` bounds (inclusive) trigger a ValidationError when
    breached. When `allow_float` is False, the coerced number must be whole.
    """
    if value is None:
        if allow_empty:
            return float("nan")
        raise ValidationError(f"{field} is required.")

    if isinstance(value, bool):
        raise ValidationError(f"{field} must be a number; received {value!r}.")

    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            if allow_empty:
                return float("nan")
            raise ValidationError(f"{field} is required.")
        try:
            number = float(stripped)
        except ValueError as exc:
            raise ValidationError(f"{field} must be a number; received {value!r}.") from exc
    else:
        raise ValidationError(f"{field} must be a number; received {value!r}.")

    if not allow_float and number != round(number):
        raise ValidationError(f"{field} must be an integer; received {value!r}.")

    if minimum is not None and number < minimum:
        raise ValidationError(f"{field} must be >= {minimum}; received {number}.")

    if maximum is not None and number > maximum:
        raise ValidationError(f"{field} must be <= {maximum}; received {number}.")

    return number


def clamp_rpe(value: Any, *, field: str = "rpe") -> int:
    """
    Coerce the Rate of Perceived Exertion into an integer between 1 and 10.

    Values outside the bounds are gently clamped to keep datasets consistent.
    """
    coerced = coerce_number(value, field=field, allow_float=False)

    lower, upper = 1, 10
    if coerced < lower:
        return lower
    if coerced > upper:
        return upper
    return int(coerced)


def validate_distance(value: Any, *, field: str = "distance") -> float:
    """
    Guard against negative throw distances while keeping the coercion logic shared.
    """
    distance = coerce_number(value, field=field, minimum=0.0)
    return distance


def parse_throws(payload: Any, *, field: str = "throws") -> list[str]:
    """
    Normalise a comma-separated list of throws, trimming whitespace quietly.

    Empty entries are removed and the remainder is returned as a list.
    """
    if payload is None:
        return []

    if isinstance(payload, (list, tuple)):
        return [str(item).strip() for item in payload if str(item).strip()]

    if not isinstance(payload, str):
        raise ValidationError(
            f"{field} must be provided as comma-separated text; received {payload!r}."
        )

    entries = [chunk.strip() for chunk in payload.split(",")]
    return [entry for entry in entries if entry]


def parse_tags(payload: Any, *, field: str = "tags") -> list[str]:
    """Normalise comma- or space-separated tags."""
    if payload is None:
        return []

    if isinstance(payload, str):
        tokens = [token.strip() for token in payload.split(",")]
    elif isinstance(payload, (list, tuple, set)):
        tokens = [str(token).strip() for token in payload]
    else:
        raise ValidationError(
            f"{field} must be a comma-separated list or sequence; received {payload!r}."
        )

    return [token for token in tokens if token]


def validate_duration(value: Any, *, field: str = "duration_minutes") -> float:
    """Ensure duration is a non-negative floating-point number."""
    return coerce_number(value, field=field, minimum=0.0)


def calculate_bmi(height_cm: float, weight_kg: float) -> float:
    """Compute Body Mass Index (kg/m^2) from centimetres and kilograms."""
    height_m = height_cm / 100.0
    if height_m <= 0:
        raise ValidationError("height_cm must be positive to compute BMI.")
    return weight_kg / (height_m**2)


def calculate_session_load(rpe: Any | None, duration_minutes: float) -> float:
    """Compute the classic session load (RPE × duration) or zero when not applicable."""
    if rpe is None:
        return 0.0
    try:
        rpe_value = float(coerce_number(rpe, field="rpe"))
    except ValidationError:
        return 0.0
    if duration_minutes <= 0:
        return 0.0
    return rpe_value * float(duration_minutes)


@dataclass
class Session:
    """Lightweight record capturing a single training session."""

    date: date
    best: float
    athlete: str
    event: str
    team: Optional[str] = None
    throws: List[float] = field(default_factory=list)
    rpe: Optional[int] = None
    duration_minutes: float = 0.0
    notes: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    implement_weight_kg: Optional[float] = None
    technique: Optional[str] = None
    fouls: Optional[int] = None
    schema_version: int = CURRENT_SCHEMA_VERSION
    video_id: Optional[str] = None
    biomechanics_analysis_id: Optional[str] = None
    biomechanics_status: Optional[str] = None
    biomechanics_timestamp: Optional[str] = None
    biomechanics_result_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Make the session JSON serialisable."""
        payload: Dict[str, Any] = {
            "date": self.date.isoformat(),
            "best": self.best,
            "athlete": self.athlete,
            "event": self.event,
            "throws": self.throws,
            "duration_minutes": self.duration_minutes,
            "load": calculate_session_load(self.rpe, self.duration_minutes),
            "schema_version": self.schema_version,
            "video_id": self.video_id,
            "biomechanics_analysis_id": self.biomechanics_analysis_id,
            "biomechanics_status": self.biomechanics_status,
            "biomechanics_timestamp": self.biomechanics_timestamp,
            "biomechanics_result_path": self.biomechanics_result_path,
        }
        if self.team:
            payload["team"] = self.team
        if self.rpe is not None:
            payload["rpe"] = self.rpe
        if self.notes:
            payload["notes"] = self.notes
        if self.tags:
            payload["tags"] = self.tags
        if self.implement_weight_kg is not None:
            payload["implement_weight_kg"] = self.implement_weight_kg
        if self.technique:
            payload["technique"] = self.technique
        if self.fouls is not None:
            payload["fouls"] = self.fouls
        return payload


@dataclass
class Athlete:
    """Structured representation of a thrower with strength and distance history."""

    name: str
    height_cm: float
    weight_kg: float
    strength_benchmarks: dict[str, float] = field(default_factory=dict)
    throw_distances: dict[str, list[float]] = field(default_factory=dict)
    notes: Optional[str] = None

    @property
    def bmi(self) -> float:
        """Body Mass Index derived from height/weight (kg/m^2)."""
        try:
            return calculate_bmi(self.height_cm, self.weight_kg)
        except ValidationError:
            return float("nan")


@dataclass
class WorkoutExercise:
    """A single exercise prescription inside a workout routine."""

    name: str
    sets: int
    reps: int
    load: Optional[float] = None  # kilograms or %1RM scaled values
    is_strength: bool | None = None

    def __post_init__(self) -> None:
        if self.is_strength is None:
            self.is_strength = is_strength_exercise(self.name)


@dataclass
class WorkoutRoutine:
    """Collection of exercises, sets, and reps for a training session."""

    title: str
    focus: Optional[str] = None
    exercises: list[WorkoutExercise] = field(default_factory=list)
    notes: Optional[str] = None

    def add_exercise(self, exercise: WorkoutExercise) -> None:
        self.exercises.append(exercise)


def is_strength_exercise(name: str) -> bool:
    """
    Determine whether an exercise should be tracked as a strength lift.

    Returns True for heavy compound movements that should be logged, False for
    accessories such as cardio, mobility, or recovery work.
    """
    normalized = (name or "").strip().lower()
    if not normalized:
        return False
    if normalized in STRENGTH_EXERCISES:
        return True
    keywords = ("squat", "deadlift", "press", "pull", "snatch", "clean", "jerk", "row")
    return any(token in normalized for token in keywords)
