from __future__ import annotations

import json
import os
import platform
from datetime import date, datetime, timezone
from pathlib import Path
from functools import lru_cache
from importlib import metadata
from typing import Any, Mapping, Optional, Sequence

import typer

from .constants import DEFAULT_ATHLETE_PLACEHOLDER
from .models import ValidationError, parse_iso_date
from .reports import generate_weekly_report
from .services import (
    LogResult,
    SummaryReport,
    build_export_dataframe,
    build_session_from_inputs,
    build_summary_report,
    generate_plots,
    load_seed_payload,
    render_summary_table,
)
from .storage import append_session, load_sessions, save_sessions

app = typer.Typer(help="Capture, review, and visualise javelin training sessions.")


def _fail(message: str, *, code: int = 1) -> None:
    typer.secho(message, fg=typer.colors.RED, err=True)
    raise typer.Exit(code=code)


def _echo_summary_recap(report: SummaryReport) -> None:
    start, end = (None, None) if report.date_range is None else report.date_range
    range_text = f"{start.isoformat()} – {end.isoformat()}" if start and end else "n/a"
    load_text = f"{report.total_load:.1f} AU"
    avg_rpe_text = f"{report.average_rpe:.1f}" if report.average_rpe is not None else "n/a"
    athletes = sorted({row.athlete for row in report.rows}) or ["n/a"]
    athlete_text = ", ".join(athletes)
    prefix = (
        f"Totals ({len(athletes)} athlete{'s' if len(athletes) != 1 else ''}: {athlete_text}): "
        f"{report.total_sessions} sessions, {report.total_throws} throws, "
        f"load {load_text}, avg RPE {avg_rpe_text} ({range_text})."
    )
    typer.echo(prefix)


@app.command()
def log(
    date: Optional[str] = typer.Option(
        None,
        "--date",
        "-d",
        help="Session date in YYYY-MM-DD format (defaults to today).",
    ),
    best: Optional[float] = typer.Option(
        None,
        "--best",
        "-b",
        help="Best throw distance in metres.",
    ),
    throws: Optional[str] = typer.Option(
        None,
        "--throws",
        "-t",
        help="Comma-separated throw distances (e.g. '61.2, 60.8, 62.4').",
    ),
    rpe: Optional[int] = typer.Option(
        None,
        "--rpe",
        help="Rate of perceived exertion on a 1-10 scale.",
    ),
    notes: Optional[str] = typer.Option(
        None,
        "--notes",
        help="Free-form notes about the session.",
    ),
    tags: Optional[str] = typer.Option(
        None,
        "--tags",
        help="Comma-separated tags (e.g. 'run-up, release-angle').",
    ),
    duration_minutes: Optional[float] = typer.Option(
        None,
        "--duration-minutes",
        "-m",
        help="Session duration in minutes (used for workload calculations).",
    ),
    athlete: Optional[str] = typer.Option(
        None,
        "--athlete",
        "-a",
        help="Athlete identifier (defaults to env/JAVELIN_TRACKER_DEFAULT_ATHLETE).",
    ),
    team: Optional[str] = typer.Option(
        None,
        "--team",
        help="Team or squad tag stored with the session.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show extra detail about the stored session.",
    ),
) -> None:
    """
    Log a new training session.

    Usage:
        python -m javelin_tracker log --throws "61.2, 60.8, 62.4" --rpe 7
    """
    athlete_scope = _resolve_role_athlete_scope(_clean_identifier(athlete)) or _default_athlete()
    team_scope = _clean_identifier(team)

    try:
        result: LogResult = build_session_from_inputs(
            date_text=date,
            best=best,
            throws=throws,
            rpe=rpe,
            notes=notes,
            tags=tags,
            duration_minutes=duration_minutes,
            athlete=athlete_scope,
            team=team_scope,
        )
    except ValidationError as exc:
        raise typer.BadParameter(str(exc)) from exc

    try:
        append_session(result.session.to_dict())
    except ValueError as exc:
        _fail(f"Could not store session: {exc}")

    typer.echo(result.confirmation)
    if verbose and result.verbose_tokens:
        typer.echo(" • " + "; ".join(result.verbose_tokens))


@app.command()
def summary(
    by: str = typer.Option(
        "week",
        "--by",
        "-b",
        case_sensitive=False,
        help="Group sessions by 'week' or 'month'.",
    ),
    athlete: Optional[str] = typer.Option(
        None,
        "--athlete",
        "-a",
        help="Filter sessions to a specific athlete (defaults to role/env scope).",
    ),
    team: Optional[str] = typer.Option(
        None,
        "--team",
        help="Filter sessions to a specific team/roster.",
    ),
    since: Optional[str] = typer.Option(
        None,
        "--since",
        help="Only include sessions on or after this date (YYYY-MM-DD).",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show additional details about the computed statistics.",
    ),
) -> None:
    """
    Summarise best distances and workload metrics.

    Usage:
        python -m javelin_tracker summary --by month
    """
    try:
        sessions = load_sessions()
    except ValueError as exc:
        _fail(f"Could not read session log: {exc}")

    requested_athlete = _clean_identifier(athlete)
    athlete_scope = _clean_identifier(_resolve_role_athlete_scope(requested_athlete))
    team_scope = _clean_identifier(team)
    since_date = _parse_since_option(since)
    scoped_sessions = _filter_sessions(
        sessions,
        athlete=athlete_scope,
        team=team_scope,
        since=since_date,
    )

    if not scoped_sessions:
        typer.echo("No sessions matched the provided filters.")
        raise typer.Exit(code=0)

    filters = _build_filter_metadata(athlete_scope, team_scope, since_date)
    if filters:
        typer.echo("Filters: " + ", ".join(f"{key}={value}" for key, value in filters.items() if value))

    try:
        report = build_summary_report(scoped_sessions, group=by, filters=filters)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_name="by") from exc

    typer.echo(render_summary_table(report))
    _echo_summary_recap(report)
    if report.personal_bests:
        for athlete_name, (pb_date, pb_value) in sorted(report.personal_bests.items()):
            typer.echo(f"PB [{athlete_name}] {pb_value:.1f} m on {pb_date.isoformat()}.")
    if verbose and report.pb_groups:
        display = []
        for entry in sorted(report.pb_groups):
            athlete_name, _, label = entry.partition("|")
            display.append(f"{athlete_name.strip() or 'n/a'} → {label}")
        typer.echo("PB flagged in group(s): " + ", ".join(display))
    if report.high_risk_dates:
        for athlete_name, dates in sorted(report.high_risk_dates.items()):
            typer.secho(
                f"High-risk workload ratios detected for {athlete_name}: "
                + ", ".join(day.isoformat() for day in dates),
                fg=typer.colors.YELLOW,
            )


@app.command()
def plot(
    athlete: Optional[str] = typer.Option(
        None,
        "--athlete",
        "-a",
        help="Filter sessions to a specific athlete (defaults to role/env scope).",
    ),
    team: Optional[str] = typer.Option(
        None,
        "--team",
        help="Filter sessions to a specific team/roster.",
    ),
    since: Optional[str] = typer.Option(
        None,
        "--since",
        help="Only include sessions on or after this date (YYYY-MM-DD).",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show extra context about generated plots.",
    ),
) -> None:
    """
    Create distance and workload plots.

    Usage:
        python -m javelin_tracker plot
    """
    try:
        sessions = load_sessions()
    except ValueError as exc:
        _fail(f"Could not read session log: {exc}")

    requested_athlete = _clean_identifier(athlete)
    athlete_scope = _clean_identifier(_resolve_role_athlete_scope(requested_athlete))
    team_scope = _clean_identifier(team)
    since_date = _parse_since_option(since)
    scoped_sessions = _filter_sessions(
        sessions,
        athlete=athlete_scope,
        team=team_scope,
        since=since_date,
    )

    if not scoped_sessions:
        typer.echo("No sessions matched the provided filters.")
        raise typer.Exit(code=0)

    try:
        paths = generate_plots(scoped_sessions, output_dir=Path("data/plots"))
    except RuntimeError as exc:
        _fail(str(exc))
    except ValueError as exc:
        _fail(str(exc), code=0)

    for path in paths:
        typer.echo(f"Saved plot to {path}")
    if verbose:
        first = min(parse_iso_date(item["date"]) for item in scoped_sessions)
        last = max(parse_iso_date(item["date"]) for item in scoped_sessions)
        typer.echo(f"Processed {len(sessions)} sessions spanning {first} – {last}.")


@app.command()
def export(
    to: Path = typer.Option(
        Path("export"),
        "--to",
        "-t",
        help="Destination directory or base file name (default: export/).",
    ),
    athlete: Optional[str] = typer.Option(
        None,
        "--athlete",
        "-a",
        help="Filter sessions to a specific athlete (defaults to role/env scope).",
    ),
    team: Optional[str] = typer.Option(
        None,
        "--team",
        help="Filter sessions to a specific team/roster.",
    ),
    since: Optional[str] = typer.Option(
        None,
        "--since",
        help="Only include sessions on or after this date (YYYY-MM-DD).",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show extra details about exported records.",
    ),
) -> None:
    """
    Export sessions for spreadsheets or further analysis.

    Usage:
        python -m javelin_tracker export --to export/sessions
    """
    try:
        sessions = load_sessions()
    except ValueError as exc:
        _fail(f"Could not read session log: {exc}")

    requested_athlete = _clean_identifier(athlete)
    athlete_scope = _clean_identifier(_resolve_role_athlete_scope(requested_athlete))
    team_scope = _clean_identifier(team)
    since_date = _parse_since_option(since)
    scoped_sessions = _filter_sessions(
        sessions,
        athlete=athlete_scope,
        team=team_scope,
        since=since_date,
    )

    if not scoped_sessions:
        typer.echo("No sessions matched the provided filters.")
        raise typer.Exit(code=0)

    filters = _build_filter_metadata(athlete_scope, team_scope, since_date)
    export_timestamp = datetime.now(timezone.utc)
    df = build_export_dataframe(scoped_sessions)
    if df.empty:
        typer.echo("No valid sessions available for export.")
        raise typer.Exit(code=0)

    df = df.copy()
    app_version = _app_version()
    iso_timestamp = export_timestamp.isoformat()
    df["app_version"] = app_version
    df["exported_at"] = iso_timestamp

    paths = _resolve_export_paths(to)
    paths["csv"].parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(paths["csv"], index=False)
    df.to_parquet(paths["parquet"], index=False)

    json_ready = df.copy()
    json_ready["date"] = json_ready["date"].astype(str)
    json_payload = json.dumps(json_ready.to_dict(orient="records"), indent=2) + "\n"
    paths["json"].write_text(json_payload, encoding="utf-8")

    metadata_payload = _build_export_metadata(
        row_count=len(df),
        columns=list(df.columns),
        exported_at=iso_timestamp,
        version=app_version,
        filters=filters,
    )
    paths["metadata"].write_text(json.dumps(metadata_payload, indent=2) + "\n", encoding="utf-8")

    typer.echo(f"Exported {len(df)} sessions to:")
    typer.echo(f" • CSV: {paths['csv']}")
    typer.echo(f" • Parquet: {paths['parquet']}")
    typer.echo(f" • JSON: {paths['json']}")
    typer.echo(f" • Metadata: {paths['metadata']}")
    if verbose:
        typer.echo("Columns: " + ", ".join(df.columns))


@app.command("weekly-report")
def weekly_report(
    athlete: Optional[str] = typer.Option(
        None,
        "--athlete",
        "-a",
        help="Athlete identifier to include in the PDF (defaults to role/env scope).",
    ),
    team: Optional[str] = typer.Option(
        None,
        "--team",
        help="Filter sessions to this team when generating the report.",
    ),
    week_ending: Optional[str] = typer.Option(
        None,
        "--week-ending",
        "-w",
        help="Week-ending date (YYYY-MM-DD). Defaults to the most recent Sunday.",
    ),
    output_dir: Path = typer.Option(
        Path("reports"),
        "--output-dir",
        "-o",
        help="Destination directory for the generated PDF.",
    ),
    team_name: Optional[str] = typer.Option(
        None,
        "--team-name",
        help="Override the team name printed on the PDF.",
    ),
    school_name: Optional[str] = typer.Option(
        None,
        "--school-name",
        help="Optional school/program name for the PDF header.",
    ),
) -> None:
    """
    Generate a PDF report summarising the previous week's workload.
    """
    try:
        sessions = load_sessions()
    except ValueError as exc:
        _fail(f"Could not read session log: {exc}")

    requested_athlete = _clean_identifier(athlete)
    athlete_scope = _clean_identifier(_resolve_role_athlete_scope(requested_athlete))
    if not athlete_scope:
        raise typer.BadParameter("Athlete is required for weekly reports.", param_name="athlete")

    team_scope = _clean_identifier(team)
    scoped_sessions = _filter_sessions(
        sessions,
        athlete=athlete_scope,
        team=team_scope,
        since=None,
    )

    if not scoped_sessions:
        typer.echo("No sessions matched the provided filters.")
        raise typer.Exit(code=0)

    week_end = parse_iso_date(week_ending, field="week-ending") if week_ending else None

    try:
        pdf_path = generate_weekly_report(
            scoped_sessions,
            athlete=athlete_scope,
            week_ending=week_end,
            output_dir=output_dir,
            team_name=team_name or team_scope,
            school_name=school_name,
        )
    except ValueError as exc:
        _fail(str(exc), code=0)
    except RuntimeError as exc:
        _fail(str(exc))

    typer.echo(f"Weekly report saved to {pdf_path}")


@app.command()
def seed(
    source: Path = typer.Option(
        Path("data/sample_sessions.json"),
        "--source",
        "-s",
        help="Seed data source file (default: data/sample_sessions.json).",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Overwrite existing sessions without prompting.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show details about the seeded data.",
    ),
    allow_production: bool = typer.Option(
        False,
        "--allow-production",
        help="Permit seeding when JAVELIN_TRACKER_ENV=production.",
    ),
) -> None:
    """
    Populate the session log with sample throws.
    """
    source_path = source.expanduser()
    try:
        payload = load_seed_payload(source_path)
    except (FileNotFoundError, ValueError) as exc:
        _fail(str(exc))

    env_name = os.getenv("JAVELIN_TRACKER_ENV", "development").lower()
    if env_name == "production" and not allow_production:
        _fail(
            "Seeding is disabled when JAVELIN_TRACKER_ENV=production. "
            "Pass --allow-production to override intentionally.",
            code=3,
        )

    destination = Path("data/sessions.json")
    if destination.exists() and not force:
        _fail(
            f"{destination} already exists. Re-run with --force to overwrite.",
            code=2,
        )

    try:
        save_sessions(payload)
    except ValueError as exc:
        _fail(f"Could not seed sessions: {exc}")

    typer.echo(f"Seeded {len(payload)} sessions to {destination}")
    if verbose:
        dates = [session.get("date", "?") for session in payload]
        typer.echo(f" • Dates: {', '.join(dates)}")


def _clean_identifier(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    candidate = value.strip()
    return candidate or None


def _parse_since_option(value: Optional[str]) -> date | None:
    if value is None:
        return None
    try:
        return parse_iso_date(value, field="since")
    except ValidationError as exc:
        raise typer.BadParameter(str(exc), param_name="since") from exc


def _resolve_role_athlete_scope(requested: Optional[str]) -> Optional[str]:
    if requested:
        return requested
    role = os.getenv("JAVELIN_TRACKER_ROLE", "").lower()
    if role == "athlete":
        return _clean_identifier(os.getenv("JAVELIN_TRACKER_DEFAULT_ATHLETE"))
    return requested


def _default_athlete() -> str:
    return _clean_identifier(os.getenv("JAVELIN_TRACKER_DEFAULT_ATHLETE")) or DEFAULT_ATHLETE_PLACEHOLDER


def _filter_sessions(
    sessions: Sequence[Mapping[str, Any]],
    *,
    athlete: Optional[str],
    team: Optional[str],
    since: date | None,
) -> list[Mapping[str, Any]]:
    results: list[Mapping[str, Any]] = []
    athlete_lower = athlete.lower() if athlete else None
    team_lower = team.lower() if team else None
    for session in sessions:
        record = dict(session)
        record_athlete = _clean_identifier(record.get("athlete")) or DEFAULT_ATHLETE_PLACEHOLDER
        if athlete_lower and record_athlete.lower() != athlete_lower:
            continue
        record_team = _clean_identifier(record.get("team"))
        if team_lower and (record_team or "").lower() != team_lower:
            continue
        if since:
            session_date = parse_iso_date(record.get("date"), field="date")
            if session_date < since:
                continue
        results.append(record)
    return results


def _build_filter_metadata(
    athlete: Optional[str],
    team: Optional[str],
    since: date | None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if athlete:
        metadata["athlete"] = athlete
    if team:
        metadata["team"] = team
    if since:
        metadata["since"] = since.isoformat()
    return metadata


def _resolve_export_paths(target: Path) -> dict[str, Path]:
    target = target.expanduser()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    if target.suffix in {".csv", ".parquet"}:
        base = target.with_suffix("")
        directory = base.parent
        stem = base.name
    elif target.suffix:
        directory = target.parent
        stem = target.stem
    else:
        directory = target
        stem = f"sessions_{timestamp}"

    return {
        "csv": directory / f"{stem}.csv",
        "parquet": directory / f"{stem}.parquet",
        "json": directory / f"{stem}.json",
        "metadata": directory / f"{stem}_metadata.json",
    }


@lru_cache(maxsize=1)
def _app_version() -> str:
    try:
        return metadata.version("javelin-tracker")
    except metadata.PackageNotFoundError:  # pragma: no cover - local dev fallback
        return "0.0.0"


def _build_export_metadata(
    *,
    row_count: int,
    columns: list[str],
    exported_at: str,
    version: str,
    filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "application": "javelin-tracker",
        "version": version,
        "exported_at": exported_at,
        "rows": row_count,
        "columns": columns,
        "data_formats": ["csv", "parquet", "json"],
        "environment": {
            "python_version": platform.python_version(),
            "javelin_tracker_data_dir": os.getenv("JAVELIN_TRACKER_DATA_DIR"),
            "javelin_tracker_sessions_file": os.getenv("JAVELIN_TRACKER_SESSIONS_FILE"),
        },
    }
    if filters:
        payload["filters"] = filters
    return payload


def main() -> None:
    app()


if __name__ == "__main__":
    main()
