from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import typer

from .models import ValidationError, parse_iso_date
from .services import (
    LogResult,
    SummaryReport,
    build_export_rows,
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
    range_text = (
        f"{start.isoformat()} – {end.isoformat()}" if start and end else "n/a"
    )
    pb = report.personal_best
    if pb:
        typer.echo(
            f"Totals: {report.total_sessions} sessions, {report.total_throws} throws, "
            f"PB {pb[1]:.1f} m on {pb[0].isoformat()} ({range_text})."
        )
    else:
        typer.echo(
            f"Totals: {report.total_sessions} sessions, {report.total_throws} throws ({range_text})."
        )


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
    try:
        result: LogResult = build_session_from_inputs(
            date_text=date,
            best=best,
            throws=throws,
            rpe=rpe,
            notes=notes,
            tags=tags,
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
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show additional details about the computed statistics.",
    ),
) -> None:
    """
    Summarise best distances and throw volume.

    Usage:
        python -m javelin_tracker summary --by month
    """
    try:
        sessions = load_sessions()
    except ValueError as exc:
        _fail(f"Could not read session log: {exc}")

    if not sessions:
        typer.echo("No sessions logged yet.")
        raise typer.Exit(code=0)

    try:
        report = build_summary_report(sessions, group=by.lower())
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_name="by") from exc

    typer.echo(render_summary_table(report))
    _echo_summary_recap(report)
    if verbose and report.personal_best:
        typer.echo("PB flagged in group(s): " + ", ".join(report.pb_groups))


@app.command()
def plot(
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show extra context about generated plots.",
    ),
) -> None:
    """
    Create distance and volume plots.

    Usage:
        python -m javelin_tracker plot
    """
    try:
        sessions = load_sessions()
    except ValueError as exc:
        _fail(f"Could not read session log: {exc}")

    if not sessions:
        typer.echo("No sessions logged yet.")
        raise typer.Exit(code=0)

    try:
        paths = generate_plots(sessions, output_dir=Path("data/plots"))
    except RuntimeError as exc:
        _fail(str(exc))
    except ValueError as exc:
        _fail(str(exc), code=0)

    for path in paths:
        typer.echo(f"Saved plot to {path}")
    if verbose:
        first = min(parse_iso_date(item["date"]) for item in sessions)
        last = max(parse_iso_date(item["date"]) for item in sessions)
        typer.echo(f"Processed {len(sessions)} sessions spanning {first} – {last}.")


@app.command()
def export(
    to: Path = typer.Option(
        Path("data/sessions.csv"),
        "--to",
        "-t",
        help="Destination CSV file (default: data/sessions.csv).",
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
        python -m javelin_tracker export --to data/sessions.csv
    """
    try:
        sessions = load_sessions()
    except ValueError as exc:
        _fail(f"Could not read session log: {exc}")

    if not sessions:
        typer.echo("No sessions logged yet.")
        raise typer.Exit(code=0)

    rows = build_export_rows(sessions)
    if not rows:
        typer.echo("No valid sessions available for export.")
        raise typer.Exit(code=0)

    destination = to.expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["date", "best", "rpe", "notes", "tags", "throws_json"]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    typer.echo(f"Exported {len(rows)} sessions to {destination}")
    if verbose:
        typer.echo("Columns: date, best, rpe, notes, tags, throws_json")


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
) -> None:
    """
    Populate the session log with sample throws.

    Usage:
        python -m javelin_tracker seed --force
    """
    source_path = source.expanduser()
    try:
        payload = load_seed_payload(source_path)
    except (FileNotFoundError, ValueError) as exc:
        _fail(str(exc))

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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
