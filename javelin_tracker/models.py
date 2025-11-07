from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional

__all__ = [
    "parse_iso_date",
    "coerce_number",
    "clamp_rpe",
    "validate_distance",
    "parse_throws",
    "parse_tags",
    "validate_duration",
    "Session",
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


@dataclass(slots=True)
class Session:
    """Lightweight record capturing a single training session."""

    date: date
    best: float
    athlete: str
    team: Optional[str] = None
    throws: List[float] = field(default_factory=list)
    rpe: Optional[int] = None
    duration_minutes: float = 0.0
    notes: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Make the session JSON serialisable."""
        payload: Dict[str, Any] = {
            "date": self.date.isoformat(),
            "best": self.best,
            "athlete": self.athlete,
            "throws": self.throws,
        }
        if self.team:
            payload["team"] = self.team
        if self.rpe is not None:
            payload["rpe"] = self.rpe
        payload["duration_minutes"] = self.duration_minutes
        if self.notes:
            payload["notes"] = self.notes
        if self.tags:
            payload["tags"] = self.tags
        return payload
