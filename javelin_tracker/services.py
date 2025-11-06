from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

from .analysis import (
    group_by_month,
    group_by_week,
    mean_best_by_group,
    median_best_by_group,
    personal_best,
    total_throws_by_group,
)
from .models import (
    Session,
    ValidationError,
    clamp_rpe,
    parse_iso_date,
    parse_tags,
    parse_throws,
    validate_distance,
)


@dataclass(frozen=True)
class LogResult:
    """Structured outcome of parsing log arguments."""

    session: Session
    throws: list[float]

    @property
    def confirmation(self) -> str:
        return f"Logged {self.session.date.isoformat()} best {self.session.best:.1f} m."

    @property
    def verbose_tokens(self) -> list[str]:
        tokens: list[str] = []
        if self.throws:
            tokens.append(
                "throws=" + ", ".join(f"{value:.1f}" for value in self.throws)
            )
        if self.session.tags:
            tokens.append("tags=" + ", ".join(self.session.tags))
        if self.session.rpe is not None:
            tokens.append(f"rpe={self.session.rpe}")
        if self.session.notes:
            tokens.append("notes recorded")
        return tokens


@dataclass(frozen=True)
class SummaryRow:
    marker: str
    label: str
    sessions: int
    average_best: float | None
    median_best: float | None
    throw_volume: int


@dataclass(frozen=True)
class SummaryReport:
    rows: list[SummaryRow]
    total_sessions: int
    total_throws: int
    date_range: tuple[date, date] | None
    personal_best: tuple[date, float] | None
    pb_groups: list[str]


def build_session_from_inputs(
    *,
    date_text: str | None,
    best: float | None,
    throws: str | None,
    rpe: int | None,
    notes: str | None,
    tags: str | None,
) -> LogResult:
    """Convert CLI inputs into a validated session payload."""
    session_date = parse_iso_date(date_text, field="date") if date_text else date.today()
    throws_tokens = parse_throws(throws, field="throws")
    throws_distances = _normalise_throw_distances(throws_tokens)

    best_distance = (
        validate_distance(best, field="best")
        if best is not None
        else _resolve_best_from_throws(throws_distances)
    )

    rpe_value = clamp_rpe(rpe, field="rpe") if rpe is not None else None
    tag_list = parse_tags(tags, field="tags")
    note_value = notes.strip() if notes else None

    session = Session(
        date=session_date,
        best=best_distance,
        throws=throws_distances,
        rpe=rpe_value,
        notes=note_value,
        tags=tag_list,
    )
    return LogResult(session=session, throws=throws_distances)


def build_summary_report(
    sessions: Sequence[Mapping[str, Any]],
    *,
    group: Literal["week", "month"],
) -> SummaryReport:
    """Produce aggregated statistics for summary displays."""
    groups = _group_sessions(sessions, group=group)
    records = [record for bucket in groups.values() for record in bucket]
    pb = personal_best(sessions)

    rows = [
        _build_summary_row(label, entries, pb)
        for label, entries in sorted(groups.items())
    ]

    total_sessions = len(records)
    total_throws = sum(len(record["throws"]) for record in records)
    date_range = None
    if records:
        sorted_dates = sorted(record["date"] for record in records)
        date_range = (sorted_dates[0], sorted_dates[-1])
    pb_groups = _locate_pb_groups(groups, pb[0]) if pb else []

    return SummaryReport(
        rows=rows,
        total_sessions=total_sessions,
        total_throws=total_throws,
        date_range=date_range,
        personal_best=pb,
        pb_groups=pb_groups,
    )


def render_summary_table(report: SummaryReport) -> str:
    """Render a fixed-width table for summary rows."""
    headers = ("marker", "group", "sessions", "avg_best", "median_best", "volume")
    rows = [
        {
            "marker": row.marker,
            "group": row.label,
            "sessions": str(row.sessions),
            "avg_best": f"{row.average_best:.1f}" if row.average_best is not None else "n/a",
            "median_best": f"{row.median_best:.1f}"
            if row.median_best is not None
            else "n/a",
            "volume": str(row.throw_volume),
        }
        for row in report.rows
    ]
    widths = {key: len(key) for key in headers}
    for row in rows:
        for key in headers:
            widths[key] = max(widths[key], len(row[key]))

    def _format_line(values: Mapping[str, str]) -> str:
        return "  ".join(values[key].rjust(widths[key]) for key in headers)

    header_line = "  ".join(key.upper().rjust(widths[key]) for key in headers)
    body = "\n".join(_format_line(row) for row in rows)
    return "\n".join(filter(None, [header_line, body]))


def generate_plots(
    sessions: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path,
) -> list[Path]:
    """Create distance line plot and weekly volume bar chart."""
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised via CLI
        raise RuntimeError("matplotlib is required to generate plots.") from exc

    weekly = group_by_week(sessions)
    records = [record for bucket in weekly.values() for record in bucket]
    if not records:
        raise ValueError("No sessions available to plot.")

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    distance_path = _create_distance_plot(records, output_dir, timestamp, plt)
    volume_path = _create_volume_plot(weekly, output_dir, timestamp, plt)
    return [distance_path, volume_path]


def build_export_rows(
    sessions: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Normalise sessions for CSV export."""
    rows: list[dict[str, str]] = []
    for session in sessions:
        session_date = parse_iso_date(session.get("date"), field="date").isoformat()
        best_distance = validate_distance(session.get("best"), field="best")

        rpe_value = session.get("rpe")
        rpe_norm = clamp_rpe(rpe_value, field="rpe") if rpe_value is not None else ""

        throws_tokens = session.get("throws", [])
        throws_list = (
            parse_throws(throws_tokens, field="throws")
            if isinstance(throws_tokens, str)
            else list(throws_tokens)
        )
        distances = _normalise_throw_distances(throws_list)

        tags = parse_tags(session.get("tags", []), field="tags")

        rows.append(
            {
                "date": session_date,
                "best": f"{best_distance:.1f}",
                "rpe": str(rpe_norm) if rpe_norm != "" else "",
                "notes": session.get("notes", "") or "",
                "tags": ", ".join(tags),
                "throws_json": json.dumps(distances),
            }
        )
    return rows


def load_seed_payload(source: Path) -> list[dict[str, Any]]:
    """Load seed sessions from disk."""
    if not source.exists():
        raise FileNotFoundError(f"Seed file not found: {source}")

    raw_text = source.read_text(encoding="utf-8")
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError("Seed file is not valid JSON.") from exc

    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ValueError("Seed data must be a JSON list of session objects.")
    return payload


def _create_distance_plot(
    records: Sequence[Mapping[str, Any]],
    output_dir: Path,
    timestamp: str,
    plt: Any,
) -> Path:
    ordered = sorted(records, key=lambda item: item["date"])
    dates = [record["date"] for record in ordered]
    bests = [record["best"] for record in ordered]

    line_path = output_dir / f"best_distance_{timestamp}.png"
    fig, ax = plt.subplots()
    ax.plot(dates, bests, marker="o", linewidth=2)
    ax.set_title("Best Distance Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Best Distance (m)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(line_path, dpi=150)
    plt.close(fig)
    return line_path


def _create_volume_plot(
    weekly: Mapping[str, Sequence[Mapping[str, Any]]],
    output_dir: Path,
    timestamp: str,
    plt: Any,
) -> Path:
    volume_map = total_throws_by_group(weekly)
    labels: list[str]
    volumes: list[int]
    if volume_map:
        volume_items = sorted(volume_map.items())
        labels = [item[0] for item in volume_items]
        volumes = [item[1] for item in volume_items]
    else:
        labels, volumes = [], []

    bar_path = output_dir / f"weekly_volume_{timestamp}.png"
    fig, ax = plt.subplots()
    ax.bar(labels, volumes, color="#4C72B0")
    ax.set_title("Weekly Throw Volume")
    ax.set_xlabel("ISO Week")
    ax.set_ylabel("Throws Recorded")
    if labels:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(bar_path, dpi=150)
    plt.close(fig)
    return bar_path


def _normalise_throw_distances(tokens: Iterable[Any]) -> list[float]:
    distances: list[float] = []
    for index, token in enumerate(tokens, start=1):
        distances.append(validate_distance(token, field=f"throws[{index}]"))
    return distances


def _resolve_best_from_throws(throws: Sequence[float]) -> float:
    if not throws:
        raise ValidationError("Provide --best or at least one value via --throws.")
    return max(throws)


def _group_sessions(
    sessions: Sequence[Mapping[str, Any]],
    *,
    group: Literal["week", "month"],
) -> dict[str, list[Mapping[str, Any]]]:
    if group == "week":
        return group_by_week(sessions)
    if group == "month":
        return group_by_month(sessions)
    raise ValueError("Grouping must be 'week' or 'month'.")


def _build_summary_row(
    label: str,
    entries: Sequence[Mapping[str, Any]],
    pb: tuple[date, float] | None,
) -> SummaryRow:
    marker = "★ PB" if pb and any(entry["date"] == pb[0] for entry in entries) else ""
    mean_map = mean_best_by_group({label: entries})
    median_map = median_best_by_group({label: entries})
    volume_map = total_throws_by_group({label: entries})

    return SummaryRow(
        marker=marker,
        label=label,
        sessions=len(entries),
        average_best=mean_map.get(label),
        median_best=median_map.get(label),
        throw_volume=volume_map.get(label, 0),
    )


def _locate_pb_groups(
    groups: Mapping[str, Sequence[Mapping[str, Any]]],
    pb_date: date,
) -> list[str]:
    return [
        label
        for label, entries in groups.items()
        if any(entry["date"] == pb_date for entry in entries)
    ]
