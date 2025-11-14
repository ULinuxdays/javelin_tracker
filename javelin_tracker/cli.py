from __future__ import annotations

import csv
import json
import platform
from datetime import date, datetime, timezone
from pathlib import Path
from functools import lru_cache
from importlib import metadata
from typing import Any, Mapping, Optional, Sequence

import typer

from .constants import DEFAULT_ATHLETE_PLACEHOLDER
from .config import as_dict as config_as_dict, get_config
from .env import get_env
from .models import (
    CURRENT_SCHEMA_VERSION,
    DEFAULT_EVENT,
    ValidationError,
    calculate_session_load,
    parse_iso_date,
)
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
from .storage import (
    append_session,
    load_sessions,
    save_sessions,
    get_daily_routine,
    print_workout_routine,
    predict_trends,
    open_database,
    log_strength_workout,
)

app = typer.Typer(help="Capture, review, and visualise multi-event throwing sessions.")


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
        help="Athlete identifier (defaults to env/THROWS_TRACKER_DEFAULT_ATHLETE; "
        "legacy JAVELIN_TRACKER_DEFAULT_ATHLETE is still accepted).",
    ),
    team: Optional[str] = typer.Option(
        None,
        "--team",
        help="Team or squad tag stored with the session.",
    ),
    event: str = typer.Option(
        ...,
        "--event",
        "-e",
        help="Throwing event (e.g. javelin, discus, shot, hammer).",
    ),
    implement_weight_kg: Optional[float] = typer.Option(
        None,
        "--implement-weight-kg",
        help="Implement weight used for the session (kilograms).",
    ),
    technique: Optional[str] = typer.Option(
        None,
        "--technique",
        help="Technique notes (e.g. '3-step', 'spin').",
    ),
    fouls: Optional[int] = typer.Option(
        None,
        "--fouls",
        help="Number of fouls recorded in the session.",
    ),
    interactive: bool = typer.Option(
        False,
        "--interactive",
        "-i",
        help="Prompt for missing values interactively.",
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

    Examples:
        python -m javelin_tracker log --event javelin --throws "61.2, 60.8, 62.4" --rpe 7
        python -m javelin_tracker log --event discus --best 58.3 --fouls 1 --technique "full"
    """
    if interactive:
        # Ask for inputs that aren't provided
        allowed = ", ".join(get_config().allowed_events)
        event = event or typer.prompt(f"Event [{allowed}]", default=get_config().allowed_events[0])
        date = date or typer.prompt("Date (YYYY-MM-DD)", default=datetime.now().date().isoformat())
        if not throws and best is None:
            throws = typer.prompt("Throws (comma-separated, optional)", default="") or None
            if not throws:
                best_text = typer.prompt("Best distance (m, optional)", default="")
                best = float(best_text) if best_text.strip() else None
        if rpe is None:
            rpe_text = typer.prompt("RPE 1-10 (optional)", default="")
            rpe = int(rpe_text) if rpe_text.strip() else None
        if duration_minutes is None:
            dur_text = typer.prompt("Duration minutes (optional)", default="")
            duration_minutes = float(dur_text) if dur_text.strip() else None
        athlete = athlete or _default_athlete()
        team = team or typer.prompt("Team (optional)", default="") or None
        tags = tags or typer.prompt("Tags (comma-separated, optional)", default="") or None
        notes = notes or typer.prompt("Notes (optional)", default="") or None

    athlete_scope = _resolve_role_athlete_scope(_clean_identifier(athlete)) or _default_athlete()
    team_scope = _clean_identifier(team)

    event_choice = _validate_event_choice(event)

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
            event=event_choice,
            implement_weight_kg=implement_weight_kg,
            technique=technique,
            fouls=fouls,
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
    event: list[str] = typer.Option(
        [],
        "--event",
        "-e",
        help="Filter sessions to one or more events (repeatable).",
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

    Examples:
        python -m javelin_tracker summary --event javelin --by week
        python -m javelin_tracker summary --event discus --event shot --by month
    """
    try:
        sessions = load_sessions()
    except ValueError as exc:
        _fail(f"Could not read session log: {exc}")

    requested_athlete = _clean_identifier(athlete)
    athlete_scope = _clean_identifier(_resolve_role_athlete_scope(requested_athlete))
    team_scope = _clean_identifier(team)
    since_date = _parse_since_option(since)
    event_filters = _normalise_event_filters(event)
    scoped_sessions = _filter_sessions(
        sessions,
        athlete=athlete_scope,
        team=team_scope,
        events=event_filters,
        since=since_date,
    )

    if not scoped_sessions:
        typer.echo("No sessions matched the provided filters.")
        raise typer.Exit(code=0)

    filters = _build_filter_metadata(athlete_scope, team_scope, since_date, event_filters)
    if filters:
        typer.echo("Filters: " + ", ".join(f"{key}={value}" for key, value in filters.items() if value))

    try:
        report = build_summary_report(
            scoped_sessions,
            group=by,
            filters=filters,
            athlete=athlete_scope,
            team=team_scope,
            events=event_filters,
            since=since_date,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_name="by") from exc

    typer.echo(render_summary_table(report))
    _echo_summary_recap(report)
    if report.personal_bests:
        for (athlete_name, event_name), (pb_date, pb_value) in sorted(report.personal_bests.items()):
            typer.echo(
                f"PB [{athlete_name} / {_display_event_label(event_name)}] "
                f"{pb_value:.1f} m on {pb_date.isoformat()}."
            )
    if verbose and report.pb_groups:
        display = []
        for athlete_name, event_name, label in sorted(report.pb_groups):
            display.append(f"{athlete_name.strip() or 'n/a'} ({_display_event_label(event_name)}) → {label}")
        typer.echo("PB flagged in group(s): " + ", ".join(display))
    if report.high_risk_dates:
        for (athlete_name, event_name), dates in sorted(report.high_risk_dates.items()):
            typer.secho(
                f"High-risk workload ratios detected for {athlete_name} [{_display_event_label(event_name)}]: "
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
    event: list[str] = typer.Option(
        [],
        "--event",
        "-e",
        help="Filter to one or more events before plotting.",
    ),
    predict_metric: Optional[str] = typer.Option(
        None,
        "--predict-metric",
        help="Optional metric to forecast and include as an extra plot (e.g., throw_distance).",
    ),
    predict_days: int = typer.Option(
        14,
        "--predict-days",
        help="Days ahead for forecast if --predict-metric is provided.",
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

    Examples:
        python -m javelin_tracker plot --event javelin
        python -m javelin_tracker plot --event discus --event shot
    """
    try:
        sessions = load_sessions()
    except ValueError as exc:
        _fail(f"Could not read session log: {exc}")

    requested_athlete = _clean_identifier(athlete)
    athlete_scope = _clean_identifier(_resolve_role_athlete_scope(requested_athlete))
    team_scope = _clean_identifier(team)
    since_date = _parse_since_option(since)
    event_filters = _normalise_event_filters(event)
    scoped_sessions = _filter_sessions(
        sessions,
        athlete=athlete_scope,
        team=team_scope,
        events=event_filters,
        since=since_date,
    )

    if not scoped_sessions:
        typer.echo("No sessions matched the provided filters.")
        raise typer.Exit(code=0)

    forecast = None
    forecast_title = None
    athlete_id_for_forecast: Optional[int] = None
    if predict_metric:
        # Resolve athlete ID if possible
        name_for_lookup = athlete_scope or _default_athlete()
        with open_database(readonly=True) as conn:
            row = conn.execute("SELECT id FROM Athletes WHERE name = ? LIMIT 1", (name_for_lookup or "",)).fetchone()
            if row:
                athlete_id_for_forecast = int(row[0])
        if athlete_id_for_forecast:
            try:
                fc = predict_trends(athlete_id_for_forecast, predict_metric, days_ahead=predict_days)
                forecast = fc.get("forecasts")
                forecast_title = f"Forecast: {predict_metric} ({fc.get('model')}, RMSE={fc.get('confidence')})"
            except Exception as exc:  # pragma: no cover
                typer.secho(f"Could not compute forecast: {exc}", fg=typer.colors.YELLOW)

    try:
        paths = generate_plots(
            scoped_sessions,
            output_dir=Path("data/plots"),
            athlete=athlete_scope,
            team=team_scope,
            events=event_filters,
            since=since_date,
            forecast=forecast,
            forecast_title=predict_metric if forecast_title is None else forecast_title,
        )
    except RuntimeError as exc:
        _fail(str(exc))
    except ValueError as exc:
        _fail(str(exc), code=0)

    for path in paths:
        typer.echo(f"Saved plot to {path}")
    if verbose:
        first = min(parse_iso_date(item["date"]) for item in scoped_sessions)
        last = max(parse_iso_date(item["date"]) for item in scoped_sessions)
        typer.echo(f"Processed {len(scoped_sessions)} sessions spanning {first} – {last}.")
        if event_filters:
            typer.echo("Events: " + ", ".join(_display_event_label(evt) for evt in event_filters))


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
    event: list[str] = typer.Option(
        [],
        "--event",
        "-e",
        help="Filter to one or more events before exporting.",
    ),
    predict_metric: Optional[str] = typer.Option(
        None,
        "--predict-metric",
        help="Optional metric to forecast and include alongside exports.",
    ),
    predict_days: int = typer.Option(
        14,
        "--predict-days",
        help="Days ahead for forecast if --predict-metric is provided.",
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

    Examples:
        python -m javelin_tracker export --event javelin --to export/jav_sessions
        python -m javelin_tracker export --event discus --event shot --athlete esha
    """
    try:
        sessions = load_sessions()
    except ValueError as exc:
        _fail(f"Could not read session log: {exc}")

    requested_athlete = _clean_identifier(athlete)
    athlete_scope = _clean_identifier(_resolve_role_athlete_scope(requested_athlete))
    team_scope = _clean_identifier(team)
    since_date = _parse_since_option(since)
    event_filters = _normalise_event_filters(event)
    scoped_sessions = _filter_sessions(
        sessions,
        athlete=athlete_scope,
        team=team_scope,
        events=event_filters,
        since=since_date,
    )

    if not scoped_sessions:
        typer.echo("No sessions matched the provided filters.")
        raise typer.Exit(code=0)

    filters = _build_filter_metadata(athlete_scope, team_scope, since_date, event_filters)
    export_timestamp = datetime.now(timezone.utc)
    df = build_export_dataframe(scoped_sessions)
    if df.empty:
        typer.echo("No valid sessions available for export.")
        raise typer.Exit(code=0)

    df = df.copy()
    app_version = _app_version()
    iso_timestamp = export_timestamp.isoformat()
    df["app_version"] = app_version
    df["generated_at"] = iso_timestamp

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
        generated_at=iso_timestamp,
        version=app_version,
        filters=filters,
    )

    # Optionally include forecast artifacts
    if predict_metric:
        name_for_lookup = athlete_scope or _default_athlete()
        athlete_id_for_forecast: Optional[int] = None
        with open_database(readonly=True) as conn:
            row = conn.execute("SELECT id FROM Athletes WHERE name = ? LIMIT 1", (name_for_lookup or "",)).fetchone()
            if row:
                athlete_id_for_forecast = int(row[0])
        if athlete_id_for_forecast is not None:
            try:
                fc = predict_trends(athlete_id_for_forecast, predict_metric, days_ahead=predict_days)
                forecasts = fc.get("forecasts") or []
                if forecasts:
                    base_csv = paths["csv"]
                    f_csv = base_csv.with_name(base_csv.stem + f"_forecast_{predict_metric}.csv")
                    f_json = base_csv.with_name(base_csv.stem + f"_forecast_{predict_metric}.json")
                    with f_csv.open("w", encoding="utf-8") as fh:
                        fh.write("date,prediction\n")
                        for d, v in forecasts:
                            fh.write(f"{d},{v}\n")
                    f_json.write_text(json.dumps(fc, indent=2) + "\n", encoding="utf-8")
                    metadata_payload.setdefault("forecasts", []).append(
                        {
                            "metric": predict_metric,
                            "model": fc.get("model"),
                            "confidence_rmse": fc.get("confidence"),
                            "trend": fc.get("trend"),
                            "files": {"csv": str(f_csv), "json": str(f_json)},
                        }
                    )
            except Exception as exc:  # pragma: no cover
                typer.secho(f"Forecast export failed: {exc}", fg=typer.colors.YELLOW)
    paths["metadata"].write_text(json.dumps(metadata_payload, indent=2) + "\n", encoding="utf-8")

    typer.echo(f"Exported {len(df)} sessions to:")
    typer.echo(f" • CSV: {paths['csv']}")
    typer.echo(f" • Parquet: {paths['parquet']}")
    typer.echo(f" • JSON: {paths['json']}")
    typer.echo(f" • Metadata: {paths['metadata']}")
    if verbose:
        typer.echo("Columns: " + ", ".join(df.columns))


@app.command("import")
def import_sessions_cli(
    source: Path = typer.Option(
        Path("import/sessions.csv"),
        "--source",
        "-s",
        help="Source file containing sessions (CSV or JSON).",
    ),
    fmt: str = typer.Option(
        "auto",
        "--format",
        "-f",
        case_sensitive=False,
        help="Source format: auto, csv, or json.",
    ),
    event: Optional[str] = typer.Option(
        None,
        "--event",
        "-e",
        help="Default event applied when rows omit the event column.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate the file without writing to storage.",
    ),
) -> None:
    """
    Import session logs from CSV or JSON files.

    Examples:
        python -m javelin_tracker import --source data/archive.json
        python -m javelin_tracker import --source uploads/discus.csv --event discus
    """
    source_path = source.expanduser()
    if not source_path.exists():
        _fail(f"Import source not found: {source_path}")

    try:
        format_choice = _resolve_import_format(source_path, fmt)
    except ValueError as exc:
        _fail(str(exc))
    default_event = _validate_event_choice(event) if event else None

    try:
        records = _load_import_records(source_path, format_choice)
    except ValueError as exc:
        _fail(str(exc))

    if not records:
        typer.echo("No rows found to import.")
        raise typer.Exit(code=0)

    normalised = [_normalise_import_record(record, default_event) for record in records]
    if dry_run:
        typer.echo(f"Validated {len(normalised)} sessions from {source_path} (dry-run).")
        raise typer.Exit(code=0)

    existing = load_sessions()
    existing.extend(normalised)
    save_sessions(existing)
    typer.echo(f"Imported {len(normalised)} sessions from {source_path} (total now {len(existing)}).")


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
    event: list[str] = typer.Option(
        [],
        "--event",
        "-e",
        help="Limit the report to one or more events (repeatable).",
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
    Generate PDF reports summarising the previous week's workload.

    Examples:
        python -m javelin_tracker weekly-report --athlete esha --event javelin
        python -m javelin_tracker weekly-report --athlete esha --event discus --event shot
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
    event_filters = _normalise_event_filters(event)
    scoped_sessions = _filter_sessions(
        sessions,
        athlete=athlete_scope,
        team=team_scope,
        events=event_filters,
        since=None,
    )

    if not scoped_sessions:
        typer.echo("No sessions matched the provided filters.")
        raise typer.Exit(code=0)

    week_end = parse_iso_date(week_ending, field="week-ending") if week_ending else None

    try:
        pdf_paths = generate_weekly_report(
            scoped_sessions,
            athlete=athlete_scope,
            events=event_filters or None,
            week_ending=week_end,
            output_dir=output_dir,
            team_name=team_name or team_scope,
            school_name=school_name,
        )
    except ValueError as exc:
        _fail(str(exc), code=0)
    except RuntimeError as exc:
        _fail(str(exc))

    for path in pdf_paths:
        typer.echo(f"Weekly report saved to {path}")


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
        help="Permit seeding when THROWS_TRACKER_ENV=production "
        "(legacy JAVELIN_TRACKER_ENV).",
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

    env_name = (get_env("ENV") or "development").lower()
    if env_name == "production" and not allow_production:
        _fail(
            "Seeding is disabled when THROWS_TRACKER_ENV=production "
            "(legacy JAVELIN_TRACKER_ENV). "
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


@app.command("config")
def config_show() -> None:
    """
    Show the effective configuration (allowed events, ACWR thresholds).
    """
    config = config_as_dict()
    typer.echo(f"Config source: {config.get('source')}")
    typer.echo("Allowed events: " + ", ".join(config.get("allowed_events", [])))
    thresholds = config.get("acwr_thresholds", {})
    sweet_min = thresholds.get("sweet_min")
    sweet_max = thresholds.get("sweet_max")
    high = thresholds.get("high")
    typer.echo(
        "ACWR thresholds: "
        f"sweet={sweet_min}–{sweet_max}, high>{high}"
    )


@app.command("daily")
def daily_routine(
    athlete_id: int = typer.Option(
        ...,
        "--athlete-id",
        "-i",
        help="Athlete ID to generate the routine for (as stored in the database).",
    )
) -> None:
    """
    Generate and display today's personalised workout routine.
    """
    routine = get_daily_routine(athlete_id)
    if not routine:
        typer.echo("No routine generated (missing profile data?).")
        raise typer.Exit(code=1)
    typer.echo(f"Daily Routine for athlete #{athlete_id}")
    formatted = print_workout_routine(routine)
    typer.echo("\n" + formatted)


@app.command("predict")
@app.command("forecast")
def predict_metric(
    athlete_id: Optional[int] = typer.Option(
        None,
        "--athlete-id",
        "-i",
        help="Athlete ID for forecasting (resolves from --athlete or env if omitted).",
    ),
    athlete: Optional[str] = typer.Option(
        None,
        "--athlete",
        "-a",
        help="Athlete name to forecast for (looked up in DB; falls back to env default).",
    ),
    metric: str = typer.Option(
        ...,
        "--metric",
        "-m",
        help="Metric to forecast (throw_distance, bench_1rm, squat_1rm, session_load).",
    ),
    days_ahead: int = typer.Option(
        14,
        "--days-ahead",
        "-d",
        help="Number of days to forecast into the future.",
    ),
    fmt: str = typer.Option(
        "table",
        "--format",
        "-f",
        case_sensitive=False,
        help="Output format: table, json, or csv.",
    ),
    to: Optional[Path] = typer.Option(
        None,
        "--to",
        "-o",
        help="Optional file path for JSON/CSV output (printed to stdout if omitted).",
    ),
) -> None:
    """
    Forecast future performance for a given metric.
    """
    # Resolve athlete ID if not explicitly provided
    resolved_id = athlete_id
    if resolved_id is None:
        candidate_name = _clean_identifier(athlete) or _default_athlete()
        if candidate_name:
            resolved_id = _lookup_athlete_id(candidate_name)
        if resolved_id is None:
            _fail("Provide --athlete-id or --athlete (or set THROWS_TRACKER_DEFAULT_ATHLETE).")

    forecast = predict_trends(int(resolved_id), metric, days_ahead=days_ahead)

    output_format = (fmt or "table").strip().lower()
    if output_format not in {"table", "json", "csv"}:
        raise typer.BadParameter("--format must be table, json, or csv.", param_name="format")

    if output_format == "table":
        if not forecast.get("forecasts"):
            typer.echo("No forecast available (insufficient data).")
            raise typer.Exit(code=1)
        typer.echo(
            f"Metric: {metric} | Model: {forecast['model']} | Trend: {forecast.get('trend')} "
            f"| RMSE: {forecast.get('confidence')}"
        )
        typer.echo("Date        | Prediction")
        typer.echo("-" * 30)
        for date_str, value in forecast["forecasts"]:
            typer.echo(f"{date_str} | {value:.2f}")
        return

    # JSON/CSV – build metadata
    payload = {
        "athlete_id": int(resolved_id),
        "athlete": _clean_identifier(athlete) or _default_athlete(),
        "metric": metric,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "app_version": _app_version(),
        **forecast,
    }

    if output_format == "json":
        text = json.dumps(payload, indent=2) + "\n"
        if to:
            to_path = to.expanduser()
            to_path.parent.mkdir(parents=True, exist_ok=True)
            to_path.write_text(text, encoding="utf-8")
            typer.echo(f"Saved forecast JSON to {to_path}")
        else:
            typer.echo(text)
        return

    # CSV output
    rows = forecast.get("forecasts") or []
    if to is None:
        base = Path("export")
        base.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        to = base / f"forecast_{metric}_{stamp}.csv"
    to_path = to.expanduser()
    to_path.parent.mkdir(parents=True, exist_ok=True)
    with to_path.open("w", encoding="utf-8") as handle:
        handle.write("date,prediction\n")
        for d, v in rows:
            handle.write(f"{d},{v}\n")
    typer.echo(f"Saved forecast CSV to {to_path}")


@app.command("strength-log")
def strength_log(
    date: Optional[str] = typer.Option(None, "--date", "-d", help="Date (YYYY-MM-DD). Defaults to today."),
    exercise: Optional[str] = typer.Option(None, "--exercise", "-e", help="Exercise name (e.g., back squat)."),
    weight_kg: Optional[float] = typer.Option(None, "--weight-kg", "-w", help="Load in kilograms."),
    reps: Optional[int] = typer.Option(None, "--reps", "-r", help="Repetitions performed."),
    athlete_id: Optional[int] = typer.Option(None, "--athlete-id", help="Athlete ID in database."),
    athlete: Optional[str] = typer.Option(None, "--athlete", "-a", help="Athlete name (will be created if missing)."),
    notes: Optional[str] = typer.Option(None, "--notes", help="Optional notes or RPE."),
    interactive: bool = typer.Option(True, "--interactive/--no-interactive", help="Prompt for missing values interactively."),
) -> None:
    """
    Log a single strength set (weight x reps) with optional interactive prompts.
    """
    if interactive:
        if not exercise:
            exercise = typer.prompt("Exercise", default="back squat")
        if weight_kg is None:
            weight_kg = float(typer.prompt("Weight (kg)", default="60"))
        if reps is None:
            reps = int(typer.prompt("Reps", default="8"))
        if not date:
            date = typer.prompt("Date (YYYY-MM-DD)", default=datetime.now().date().isoformat())
        if not athlete_id and not athlete:
            athlete = typer.prompt("Athlete name", default=_default_athlete())

    if athlete_id is None and athlete:
        athlete_id = _get_or_create_athlete(athlete)
    if athlete_id is None:
        _fail("Provide --athlete-id or --athlete.")

    try:
        row_id = log_strength_workout(int(athlete_id), date or datetime.now().date().isoformat(), exercise or "exercise", float(weight_kg), int(reps), notes=notes)
    except Exception as exc:
        _fail(f"Could not log strength set: {exc}")
    typer.echo(f"Logged strength set id={row_id} for athlete #{athlete_id} ({exercise} {weight_kg} x {reps}).")


def _get_or_create_athlete(name: str) -> int:
    clean = (name or "").strip()
    if not clean:
        raise typer.BadParameter("Athlete name is required.")
    with open_database() as conn:
        row = conn.execute("SELECT id FROM Athletes WHERE name = ? LIMIT 1", (clean,)).fetchone()
        if row:
            return int(row[0])
        conn.execute(
            "INSERT INTO Athletes (name) VALUES (?)",
            (clean,),
        )
        conn.commit()
        new_id = conn.execute("SELECT id FROM Athletes WHERE name = ?", (clean,)).fetchone()[0]
        return int(new_id)

def _lookup_athlete_id(name: str) -> Optional[int]:
    clean = (name or "").strip()
    if not clean:
        return None
    with open_database(readonly=True) as conn:
        row = conn.execute("SELECT id FROM Athletes WHERE name = ? LIMIT 1", (clean,)).fetchone()
        if row:
            return int(row[0])
    return None

def _validate_event_choice(value: str | None) -> str:
    candidate = (value or "").strip().lower()
    if not candidate:
        raise typer.BadParameter("Event is required.", param_name="event")
    return _warn_if_custom_event(candidate)


def _normalise_event_filters(values: Sequence[str]) -> list[str]:
    filters: list[str] = []
    for value in values:
        candidate = value.strip().lower()
        if not candidate:
            continue
        filters.append(_warn_if_custom_event(candidate))
    return filters


def _warn_if_custom_event(candidate: str) -> str:
    allowed = {event.lower() for event in get_config().allowed_events}
    if allowed and candidate not in allowed:
        typer.secho(
            f"Warning: event '{candidate}' is not in the configured allowlist {sorted(allowed)}.",
            fg=typer.colors.YELLOW,
        )
    return candidate


def _display_event_label(value: str | None) -> str:
    text = (value or DEFAULT_EVENT).replace("_", " ").strip() or DEFAULT_EVENT
    return text.title()


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
    role = (get_env("ROLE") or "").lower()
    if role == "athlete":
        return _clean_identifier(get_env("DEFAULT_ATHLETE"))
    return requested


def _default_athlete() -> str:
    return _clean_identifier(get_env("DEFAULT_ATHLETE")) or DEFAULT_ATHLETE_PLACEHOLDER


def _filter_sessions(
    sessions: Sequence[Mapping[str, Any]],
    *,
    athlete: Optional[str],
    team: Optional[str],
    events: Sequence[str] | None,
    since: date | None,
) -> list[Mapping[str, Any]]:
    results: list[Mapping[str, Any]] = []
    athlete_lower = athlete.lower() if athlete else None
    team_lower = team.lower() if team else None
    event_set = {event.lower() for event in events or [] if event}
    for session in sessions:
        record = dict(session)
        record_athlete = _clean_identifier(record.get("athlete")) or DEFAULT_ATHLETE_PLACEHOLDER
        if athlete_lower and record_athlete.lower() != athlete_lower:
            continue
        record_team = _clean_identifier(record.get("team"))
        if team_lower and (record_team or "").lower() != team_lower:
            continue
        if event_set:
            record_event = str(record.get("event") or DEFAULT_EVENT).strip().lower() or DEFAULT_EVENT
            if record_event not in event_set:
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
    events: Sequence[str] | None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if athlete:
        metadata["athlete"] = athlete
    if team:
        metadata["team"] = team
    if since:
        metadata["since"] = since.isoformat()
    if events:
        metadata["event"] = ", ".join(events)
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
        return metadata.version("throws-tracker")
    except metadata.PackageNotFoundError:  # pragma: no cover - local dev fallback
        return "0.0.0"


def _build_export_metadata(
    *,
    row_count: int,
    columns: list[str],
    generated_at: str,
    version: str,
    filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "application": "throws-tracker",
        "version": version,
        "generated_at": generated_at,
        "rows": row_count,
        "columns": columns,
        "data_formats": ["csv", "parquet", "json"],
        "environment": {
            "python_version": platform.python_version(),
            "throws_tracker_data_dir": get_env("DATA_DIR"),
            "throws_tracker_sessions_file": get_env("SESSIONS_FILE"),
        },
    }
    if filters:
        payload["filters"] = filters
    return payload


def _resolve_import_format(source: Path, fmt: str) -> str:
    choice = fmt.lower()
    if choice not in {"auto", "csv", "json"}:
        raise typer.BadParameter("Format must be auto, csv, or json.", param_name="format")
    if choice == "auto":
        suffix = source.suffix.lower()
        if suffix == ".csv":
            return "csv"
        if suffix == ".json":
            return "json"
        raise ValueError("Could not infer file format. Specify --format explicitly.")
    return choice


def _load_import_records(source: Path, fmt: str) -> list[dict[str, Any]]:
    if fmt == "json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("JSON import must contain a list of session objects.")
        return [dict(item) for item in payload if isinstance(item, Mapping)]
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(dict(row))
    return rows


def _normalise_import_record(record: Mapping[str, Any], default_event: str | None) -> dict[str, Any]:
    normalised = dict(record)
    normalised["event"] = _resolve_event_default(normalised.get("event"), default_event)
    normalised["duration_minutes"] = _coerce_float(normalised.get("duration_minutes"))
    normalised["load"] = calculate_session_load(normalised.get("rpe"), normalised["duration_minutes"])
    schema_value = normalised.get("schema_version")
    try:
        normalised["schema_version"] = (
            int(schema_value) if schema_value not in (None, "") else CURRENT_SCHEMA_VERSION
        )
    except (TypeError, ValueError):
        normalised["schema_version"] = CURRENT_SCHEMA_VERSION
    return normalised


def _resolve_event_default(value: Any, fallback: str | None) -> str:
    text = str(value).strip().lower() if value is not None else ""
    if not text:
        text = fallback or DEFAULT_EVENT
    return _warn_if_custom_event(text)


def _coerce_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def main() -> None:
    app()


if __name__ == "__main__":
    main()
