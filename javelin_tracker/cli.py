from __future__ import annotations

import csv
import json
import logging
import platform
import re
import shutil
from datetime import date, datetime, timezone
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import typer

from . import biomechanics
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
from javelin_tracker.biomechanics.database.elite_reference import (
    compute_elite_reference_profile,
    compute_style_profiles,
    get_reference_value,
)
from javelin_tracker.biomechanics.elite_database.init_elite_database import REQUIRED_COLUMNS
from javelin_tracker.biomechanics.database.validation import (
    exclude_throw,
    load_exclusions,
    recompute_profiles_after_exclusions,
    validate_elite_profiles,
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
biomechanics_app = typer.Typer(help="Biomechanics utilities for pose/video workflows.")
elite_db_app = typer.Typer(help="Elite database management (processing, profiles, QC).")


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


_ELITE_DB_QUALITY_COLUMN = "quality_score"
_ELITE_DB_SLUG_RE = re.compile(r"[^A-Za-z0-9_]+")


def _elite_db_slug(value: str) -> str:
    text = (value or "").strip().replace(" ", "_")
    text = _ELITE_DB_SLUG_RE.sub("_", text)
    return text.strip("_") or "unknown"


def _elite_db_throw_id_from_row(row: Mapping[str, str]) -> str:
    thrower = _elite_db_slug(row.get("thrower_name", ""))
    throw_number = _elite_db_slug(row.get("throw_number", ""))
    return f"{thrower}_{throw_number}".strip("_")


def _elite_db_parse_throw_parts(value: str) -> tuple[str | None, str | None]:
    """Best-effort parse of `{thrower}_{throw_number}` strings."""
    text = (value or "").strip()
    match = re.match(r"(.+)_([0-9]+)$", text)
    if not match:
        return None, None
    thrower_part = match.group(1).replace("_", " ").strip()
    throw_number = match.group(2).strip()
    return (thrower_part or None), (throw_number or None)


def _elite_db_configure_logger(log_path: Path) -> logging.Logger:
    log = logging.getLogger("javelin_tracker.elite_db")
    log.setLevel(logging.INFO)
    log.propagate = False

    for handler in list(log.handlers):
        log.removeHandler(handler)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(handler)
    return log


def _elite_db_get_pose_pipeline() -> object:
    try:
        from javelin_tracker.biomechanics.pose_estimation.pipeline import PosePipeline
    except Exception as exc:  # pragma: no cover - optional heavy deps
        _fail(
            "PosePipeline dependencies are missing. Install `opencv-python` and `mediapipe` to process videos.",
            code=2,
        )
        raise typer.Exit(code=2) from exc
    return PosePipeline()


def _elite_db_get_validate_pose_quality():
    try:
        from javelin_tracker.biomechanics.utils.validation import validate_pose_quality
    except Exception as exc:  # pragma: no cover - optional heavy deps
        _fail(
            "Pose quality validation dependencies are missing. Install `opencv-python` and `mediapipe` to validate.",
            code=2,
        )
        raise typer.Exit(code=2) from exc
    return validate_pose_quality


def _elite_db_load_metadata(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    return fieldnames, rows


def _elite_db_stable_fieldnames(existing: list[str], rows: list[dict[str, str]]) -> list[str]:
    fieldnames: list[str] = []
    for col in REQUIRED_COLUMNS:
        fieldnames.append(col)
    if _ELITE_DB_QUALITY_COLUMN not in fieldnames:
        fieldnames.append(_ELITE_DB_QUALITY_COLUMN)

    for col in existing:
        if col not in fieldnames:
            fieldnames.append(col)

    all_keys: set[str] = set()
    for row in rows:
        all_keys.update(row.keys())
    for col in sorted(all_keys):
        if col not in fieldnames:
            fieldnames.append(col)
    return fieldnames


def _elite_db_save_metadata(csv_path: Path, existing_fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    fieldnames = _elite_db_stable_fieldnames(existing_fieldnames, rows)
    for row in rows:
        for col in fieldnames:
            row.setdefault(col, "")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@elite_db_app.command("process")
def elite_process(
    video_path: str = typer.Argument(..., help="Path to the elite video (.mp4)."),
    thrower_name: Optional[str] = typer.Option(None, "--thrower-name", help="Thrower full name (for metadata/output naming)."),
    throw_number: Optional[str] = typer.Option(None, "--throw-number", help="Throw number (for metadata/output naming)."),
    video_id: Optional[str] = typer.Option(None, "--video-id", help="Override output stem (default derived from filename)."),
    poses_dir: str = typer.Option("data/biomechanics/elite_database/poses", help="Elite poses output directory."),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", help="(Deprecated) Alias for --poses-dir."),
    metadata_csv: str = typer.Option("data/biomechanics/elite_database/elite_metadata.csv", help="Elite metadata CSV."),
    log_path: str = typer.Option("data/biomechanics/elite_database/elite_database_qc.log", help="QC/action log file."),
    register_metadata: bool = typer.Option(
        True, "--register-metadata/--no-register-metadata", help="Update elite_metadata.csv with processing results."
    ),
    copy_to_videos: bool = typer.Option(
        False,
        "--copy-to-videos/--no-copy-to-videos",
        help="Copy the provided video into the elite videos folder when registering metadata.",
    ),
    force_reprocess: bool = typer.Option(False, "--force-reprocess", help="Reprocess even if output already exists."),
    throwing_style: Optional[str] = typer.Option(None, "--throwing-style", help="Set style when registering a new row."),
    distance_m: Optional[float] = typer.Option(None, "--distance-m", help="Set distance when registering a new row."),
    date_recorded: Optional[str] = typer.Option(None, "--date-recorded", help="Set date_recorded when registering a new row (YYYY-MM-DD)."),
    source_url: Optional[str] = typer.Option(None, "--source-url", help="Set source_url when registering a new row."),
    notes: Optional[str] = typer.Option(None, "--notes", help="Set notes when registering a new row."),
) -> None:
    """Process a single elite video through the biomechanics pipeline."""
    vid_path = Path(video_path)
    if not vid_path.exists():
        _fail(f"Video not found: {vid_path}")

    if output_dir and poses_dir == "data/biomechanics/elite_database/poses":
        poses_dir = output_dir
        typer.secho("--output-dir is deprecated; use --poses-dir instead.", fg=typer.colors.YELLOW, err=True)

    parsed_name, parsed_number = _elite_db_parse_throw_parts(vid_path.stem)
    thrower_name = thrower_name or parsed_name
    throw_number = throw_number or parsed_number

    if video_id:
        vid_id = _elite_db_slug(video_id)
    elif thrower_name and throw_number:
        vid_id = f"{_elite_db_slug(thrower_name)}_{_elite_db_slug(throw_number)}".strip("_")
    else:
        vid_id = _elite_db_slug(vid_path.stem)
        inferred_name, inferred_number = _elite_db_parse_throw_parts(vid_id)
        thrower_name = thrower_name or inferred_name
        throw_number = throw_number or inferred_number

    poses_root = Path(poses_dir)
    poses_root.mkdir(parents=True, exist_ok=True)
    pose_output = poses_root / f"{vid_id}.json"

    log = _elite_db_configure_logger(Path(log_path))
    log.info("CLI elite-db process start video=%s video_id=%s", vid_path, vid_id)

    if pose_output.exists() and not force_reprocess:
        typer.echo(f"Pose already exists at {pose_output}; use --force-reprocess to overwrite.")
        log.info("Skipping %s (already exists)", pose_output)
        return

    pipeline = _elite_db_get_pose_pipeline()
    validate_pose_quality = _elite_db_get_validate_pose_quality()

    try:
        result = pipeline.process_video(vid_path, vid_id, str(poses_root))  # type: ignore[attr-defined]
    except Exception as exc:
        log.exception("Pipeline exception video=%s: %s", vid_path, exc)
        _fail(f"Processing failed: {exc}")

    if result.get("status") != "success":
        error = result.get("error") or "unknown error"
        log.error("Processing failed video=%s status=%s error=%s", vid_path, result.get("status"), error)
        _fail(f"Processing failed: {error}")

    try:
        payload = json.loads(Path(result["output_path"]).read_text(encoding="utf-8"))
    except Exception as exc:
        log.exception("Failed reading pipeline output %s: %s", result.get("output_path"), exc)
        _fail(f"Failed to read pipeline output: {exc}")

    try:
        quality = validate_pose_quality(payload.get("frames", []), payload.get("video_metadata", {}) or {})
    except Exception as exc:
        log.exception("Quality validation failed video=%s: %s", vid_path, exc)
        _fail(f"Quality validation failed: {exc}")

    issues = quality.get("issues")
    if isinstance(issues, list):
        for issue in issues:
            if issue:
                log.warning("QC issue [%s] %s", vid_id, issue)

    payload["quality"] = quality
    pose_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    quality_score = quality.get("quality_score")
    is_valid = bool(quality.get("is_valid"))
    if not is_valid:
        typer.secho(
            f"Processed {vid_path} -> {pose_output} (quality {quality_score}; flagged for review)",
            fg=typer.colors.YELLOW,
        )
        log.warning("Processed flagged for review video_id=%s quality_score=%s", vid_id, quality_score)
    else:
        typer.secho(
            f"Processed {vid_path} -> {pose_output} (quality {quality_score})",
            fg=typer.colors.GREEN,
        )
        log.info("Processed complete video_id=%s quality_score=%s", vid_id, quality_score)

    if not register_metadata:
        return

    csv_path = Path(metadata_csv)
    if not csv_path.exists():
        typer.secho(
            f"Metadata CSV not found ({csv_path}); run the initializer and re-run with --register-metadata.",
            fg=typer.colors.YELLOW,
        )
        log.warning("Metadata CSV missing; cannot register results: %s", csv_path)
        return

    base_dir = csv_path.parent
    videos_dir = base_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    abs_video = vid_path.resolve()
    relative_video_path: Optional[str] = None
    try:
        rel_to_videos = abs_video.relative_to(videos_dir.resolve())
        relative_video_path = str(Path("videos") / rel_to_videos).replace("\\", "/")
    except Exception:
        if copy_to_videos:
            dest = videos_dir / f"{vid_id}.mp4"
            if not dest.exists():
                shutil.copy2(abs_video, dest)
                log.info("Copied video to %s", dest)
            relative_video_path = str(Path("videos") / dest.name).replace("\\", "/")

    if relative_video_path is None:
        typer.secho(
            "Video is not under the elite videos/ folder, so metadata cannot store a portable relative path. "
            "Move/copy it under videos/ or re-run with --copy-to-videos.",
            fg=typer.colors.YELLOW,
        )
        log.warning("Cannot register metadata for %s: video not under videos/", abs_video)
        return

    existing_fieldnames, rows = _elite_db_load_metadata(csv_path)
    if not all(col in existing_fieldnames for col in REQUIRED_COLUMNS):
        _fail(
            f"Invalid elite metadata CSV header; expected {REQUIRED_COLUMNS}. Found={existing_fieldnames}.",
        )

    target_row: Optional[dict[str, str]] = None
    for row in rows:
        if _elite_db_throw_id_from_row(row) == vid_id:
            target_row = row
            break

    if target_row is None:
        if not (thrower_name and throw_number):
            typer.secho(
                "Could not infer thrower_name/throw_number for metadata row creation. "
                "Pass --thrower-name and --throw-number or rename the file to {thrower}_{number}.mp4.",
                fg=typer.colors.YELLOW,
            )
            log.warning("Cannot create metadata row for %s (missing thrower_name/throw_number)", vid_id)
            return
        target_row = {col: "" for col in REQUIRED_COLUMNS}
        target_row["thrower_name"] = thrower_name
        target_row["throw_number"] = str(throw_number)
        target_row["distance_m"] = "" if distance_m is None else f"{float(distance_m):.2f}"
        target_row["throwing_style"] = (throwing_style or "").strip()
        target_row["video_path"] = relative_video_path
        target_row["fps"] = ""
        target_row["date_recorded"] = (date_recorded or "").strip()
        target_row["source_url"] = (source_url or "").strip()
        target_row["notes"] = (notes or "").strip()
        target_row["processed_status"] = "pending"
        rows.append(target_row)
        log.info("Appended metadata row for %s", vid_id)

    if not target_row.get("video_path"):
        target_row["video_path"] = relative_video_path
    target_row["processed_status"] = "complete"
    target_row[_ELITE_DB_QUALITY_COLUMN] = "" if quality_score is None else str(quality_score)
    fps_val = payload.get("video_metadata", {}).get("fps") if isinstance(payload.get("video_metadata"), dict) else None
    if fps_val and not target_row.get("fps"):
        target_row["fps"] = str(fps_val)

    if throwing_style and not target_row.get("throwing_style"):
        target_row["throwing_style"] = throwing_style
    if distance_m is not None and not target_row.get("distance_m"):
        target_row["distance_m"] = f"{float(distance_m):.2f}"
    if date_recorded and not target_row.get("date_recorded"):
        target_row["date_recorded"] = date_recorded
    if source_url and not target_row.get("source_url"):
        target_row["source_url"] = source_url
    if notes and not target_row.get("notes"):
        target_row["notes"] = notes

    _elite_db_save_metadata(csv_path, existing_fieldnames, rows)
    typer.echo(f"Updated metadata row for {vid_id} in {csv_path}")
    log.info("Updated metadata row for %s", vid_id)


@elite_db_app.command("recompute")
def elite_recompute(
    poses_dir: str = typer.Option("data/biomechanics/elite_database/poses", help="Directory with pose JSONs."),
    metadata_csv: str = typer.Option("data/biomechanics/elite_database/elite_metadata.csv", help="Metadata CSV."),
    output_dir: str = typer.Option(
        "data/biomechanics/elite_database",
        help="Where to write reference profiles (overall + per-style).",
    ),
    exclusions_path: str = typer.Option(
        "data/biomechanics/elite_database/exclusions.json",
        help="Exclusions file used to omit coach-flagged throws.",
    ),
    log_path: str = typer.Option("data/biomechanics/elite_database/elite_database_qc.log", help="QC/action log file."),
) -> None:
    """Recompute overall and style reference profiles (respecting exclusions)."""
    log = _elite_db_configure_logger(Path(log_path))
    log.info("CLI elite-db recompute start poses_dir=%s metadata_csv=%s", poses_dir, metadata_csv)

    profiles = recompute_profiles_after_exclusions(
        poses_dir=Path(poses_dir),
        metadata_csv=Path(metadata_csv),
        output_dir=Path(output_dir),
        exclusions_path=Path(exclusions_path),
        log_path=Path(log_path),
    )
    overall_samples = int(profiles.get("overall", {}).get("n_samples", 0))
    styles = sorted([style for style in profiles.keys() if style != "overall"])
    styles_text = ", ".join(styles) if styles else "none"
    typer.echo(f"Recomputed profiles: overall={overall_samples} samples; styles={styles_text}")
    log.info("CLI elite-db recompute done overall=%s styles=%s", overall_samples, styles_text)


@elite_db_app.command("list-styles")
def elite_list_styles(
    metadata_csv: str = typer.Option("data/biomechanics/elite_database/elite_metadata.csv", help="Metadata CSV."),
    poses_dir: str = typer.Option("data/biomechanics/elite_database/poses", help="Poses directory."),
    exclusions_path: str = typer.Option(
        "data/biomechanics/elite_database/exclusions.json",
        help="Exclusions file used to omit coach-flagged throws.",
    ),
    only_processed: bool = typer.Option(True, "--only-processed/--all", help="Count only processed throws."),
    log_path: str = typer.Option("data/biomechanics/elite_database/elite_database_qc.log", help="QC/action log file."),
) -> None:
    """List available styles and sample counts."""
    log = _elite_db_configure_logger(Path(log_path))
    log.info("CLI elite-db list-styles start metadata_csv=%s poses_dir=%s", metadata_csv, poses_dir)

    excluded_ids = set(load_exclusions(Path(exclusions_path)).keys())
    counts: dict[str, int] = {}
    total = 0

    with Path(metadata_csv).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            status = (row.get("processed_status") or "").strip().lower()
            if only_processed and status != "complete":
                continue
            throw_id = _elite_db_throw_id_from_row(row)
            if throw_id in excluded_ids:
                continue
            pose_path = Path(poses_dir) / f"{throw_id}.json"
            if only_processed and not pose_path.exists():
                continue

            style = (row.get("throwing_style") or "unknown").strip().lower() or "unknown"
            counts[style] = counts.get(style, 0) + 1
            total += 1

    for style in sorted(counts):
        typer.echo(f"{style}: {counts[style]}")
    typer.echo(f"overall: {total}")
    log.info("CLI elite-db list-styles done total=%s styles=%s", total, ",".join(sorted(counts.keys())))


@elite_db_app.command("export")
def elite_export(
    reference_path: str = typer.Option(
        "data/biomechanics/elite_database/reference_profile_overall.json", help="Reference profile to export."
    ),
    output_path: str = typer.Argument(..., help="Where to save the exported JSON."),
    log_path: str = typer.Option("data/biomechanics/elite_database/elite_database_qc.log", help="QC/action log file."),
) -> None:
    """Export a reference profile JSON to a custom location."""
    log = _elite_db_configure_logger(Path(log_path))
    log.info("CLI elite-db export start reference=%s output=%s", reference_path, output_path)
    ref = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    Path(output_path).write_text(json.dumps(ref, indent=2), encoding="utf-8")
    typer.echo(f"Exported profile to {output_path}")
    log.info("CLI elite-db export done output=%s", output_path)


@elite_db_app.command("validate")
def elite_validate(
    reference_path: str = typer.Option(
        "data/biomechanics/elite_database/reference_profile_overall.json", help="Reference profile JSON."
    ),
    metadata_csv: str = typer.Option("data/biomechanics/elite_database/elite_metadata.csv", help="Metadata CSV."),
    poses_dir: str = typer.Option("data/biomechanics/elite_database/poses", help="Poses directory."),
    exclusions_path: str = typer.Option(
        "data/biomechanics/elite_database/exclusions.json",
        help="Exclusions file used to omit coach-flagged throws.",
    ),
    log_path: str = typer.Option("data/biomechanics/elite_database/elite_database_qc.log", help="QC/action log file."),
) -> None:
    """Run QC checks on reference profiles."""
    log = _elite_db_configure_logger(Path(log_path))
    log.info("CLI elite-db validate start reference=%s metadata_csv=%s poses_dir=%s", reference_path, metadata_csv, poses_dir)
    report = validate_elite_profiles(
        Path(reference_path),
        None,
        poses_dir=Path(poses_dir),
        metadata_csv=Path(metadata_csv),
        exclusions_path=Path(exclusions_path),
        log_path=Path(log_path),
    )
    typer.echo(json.dumps(report, indent=2))
    log.info(
        "CLI elite-db validate done outliers=%s low_confidence=%s",
        len(report.get("outliers", [])),
        len(report.get("low_confidence_metrics", [])),
    )


@elite_db_app.command("exclude")
def elite_exclude(
    throw_id: str = typer.Argument(...),
    reason: str = typer.Argument(...),
    exclusions_path: str = typer.Option(
        "data/biomechanics/elite_database/exclusions.json",
        help="Where to persist excluded throw ids.",
    ),
    log_path: str = typer.Option("data/biomechanics/elite_database/elite_database_qc.log", help="QC/action log file."),
    recompute: bool = typer.Option(
        False,
        "--recompute",
        help="Recompute profiles after exclusion (writes updated reference JSONs).",
    ),
    poses_dir: str = typer.Option("data/biomechanics/elite_database/poses", help="Poses directory (for recompute)."),
    metadata_csv: str = typer.Option("data/biomechanics/elite_database/elite_metadata.csv", help="Metadata CSV (for recompute)."),
    output_dir: str = typer.Option(
        "data/biomechanics/elite_database",
        help="Where to write reference profiles when --recompute is set.",
    ),
) -> None:
    """Exclude a throw from reference calculations."""
    log = _elite_db_configure_logger(Path(log_path))
    log.info("CLI elite-db exclude start throw_id=%s", throw_id)
    confirm = typer.confirm(f"Exclude throw {throw_id} for reason: {reason}?")
    if not confirm:
        typer.echo("Aborted.")
        raise typer.Exit(code=1)
    exclude_throw(throw_id, reason, exclusions_path=Path(exclusions_path), log_path=Path(log_path))
    typer.echo(f"Excluded {throw_id}")
    log.info("CLI elite-db exclude done throw_id=%s", throw_id)
    if recompute:
        recompute_profiles_after_exclusions(
            poses_dir=Path(poses_dir),
            metadata_csv=Path(metadata_csv),
            output_dir=Path(output_dir),
            exclusions_path=Path(exclusions_path),
            log_path=Path(log_path),
        )
        typer.echo("Recomputed profiles after exclusion.")


@biomechanics_app.command("info")
def biomechanics_info(show_config: bool = typer.Option(False, "--config", help="Print biomechanics config.")) -> None:
    """
    Display available biomechanics capabilities and optional config details.

    Example:
        javelin biomechanics info --config
    """
    typer.echo("Biomechanics module available: pose detection, video utilities, config.")
    typer.echo("Key imports: PoseDetector, extract_frames, get_video_metadata, validate_video_readable.")
    if show_config:
        biomechanics.print_config()


app.add_typer(biomechanics_app, name="biomechanics", help="Biomechanics tools (pose, video, config).")
app.add_typer(elite_db_app, name="elite-db", help="Elite database tools (processing, profiles, QC).")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
