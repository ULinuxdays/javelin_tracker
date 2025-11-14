from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from datetime import date
from statistics import mean, median
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    Session,
    ValidationError,
    parse_iso_date,
    parse_tags,
    parse_throws,
    validate_distance,
)

# Python 3.9-compatible runtime type alias (avoid PEP 604 union at runtime)
from typing import Union as _Union
SessionInput = _Union[Mapping[str, Any], Session]
SessionRecord = Dict[str, Any]
Group = Dict[str, List[SessionRecord]]


def personal_best(sessions: Iterable[SessionInput]) -> Optional[Tuple[date, float]]:
    """Return the best distance and its date from an iterable of sessions."""
    best_pair: Optional[Tuple[date, float]] = None
    for session in sessions:
        record = _normalise_session(session)
        if best_pair is None or record["best"] > best_pair[1]:
            best_pair = (record["date"], record["best"])
    return best_pair


def group_by_week(sessions: Iterable[SessionInput]) -> Group:
    """
    Group sessions by ISO week (YYYY-Www).
    """
    grouped: Group = {}
    for session in sessions:
        record = _normalise_session(session)
        iso_year, iso_week, _ = record["date"].isocalendar()
        key = f"{iso_year}-W{iso_week:02d}"
        grouped.setdefault(key, []).append(record)
    return grouped


def group_by_month(sessions: Iterable[SessionInput]) -> Group:
    """
    Group sessions by calendar month (YYYY-MM).
    """
    grouped: Group = {}
    for session in sessions:
        record = _normalise_session(session)
        key = record["date"].strftime("%Y-%m")
        grouped.setdefault(key, []).append(record)
    return grouped


def mean_best_by_group(groups: Mapping[str, Sequence[SessionRecord]]) -> Dict[str, float]:
    """
    Compute the average best distance per group.
    """
    return {
        key: mean(record["best"] for record in records)
        for key, records in groups.items()
        if records
    }


def median_best_by_group(groups: Mapping[str, Sequence[SessionRecord]]) -> Dict[str, float]:
    """
    Compute the median best distance per group.
    """
    return {
        key: median(record["best"] for record in records)
        for key, records in groups.items()
        if records
    }


def total_throws_by_group(groups: Mapping[str, Sequence[SessionRecord]]) -> Dict[str, int]:
    """
    Count the total throws recorded in each group.
    """
    return {
        key: sum(len(record["throws"]) for record in records)
        for key, records in groups.items()
        if records
    }


def _normalise_session(session: SessionInput) -> SessionRecord:
    """
    Convert various session representations into a consistent dict.
    """
    if isinstance(session, Session):
        raw = asdict(session)
        raw["date"] = session.date.isoformat()
    elif isinstance(session, Mapping):
        raw = dict(session)
    else:
        raise TypeError(f"Unsupported session type: {type(session)!r}")

    try:
        session_date = parse_iso_date(raw.get("date"), field="date")
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc

    try:
        best = validate_distance(raw.get("best"), field="best")
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc

    throws_raw = raw.get("throws", [])
    if isinstance(throws_raw, str):
        throw_values = parse_throws(throws_raw, field="throws")
    else:
        if not isinstance(throws_raw, Iterable):
            raise ValueError("throws must be a sequence or comma-separated string.")
        throw_values = list(throws_raw)

    throws: List[float] = []
    for index, item in enumerate(throw_values, start=1):
        try:
            throws.append(validate_distance(item, field=f"throws[{index}]"))
        except ValidationError as exc:
            raise ValueError(str(exc)) from exc

    try:
        tags = parse_tags(raw.get("tags", []), field="tags")
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc

    normalised: SessionRecord = {
        "date": session_date,
        "best": best,
        "throws": throws,
        "notes": raw.get("notes"),
        "tags": tags,
    }

    return normalised
