# Throws Tracker By Uday Thakran

Command-line toolkit for capturing throwing sessions, analysing workload, and generating coaching/research reports across javelin, discus, shot, hammer (plus custom events).

[![CI](https://github.com/uday/Javelin/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/uday/Javelin/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/throws-tracker.svg?label=PyPI&logo=pypi)](https://pypi.org/project/throws-tracker/)
[![Python](https://img.shields.io/pypi/pyversions/throws-tracker.svg?logo=python)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-yellow.svg)](CHANGELOG.md)

Note: The CLI binary is available as both `javelin` and `throws`. The package was formerly published as `javelin-tracker` and is now `throws-tracker`.

## Highlights

- Multi-event tracking with configurable allowlist (defaults: javelin, discus, shot, hammer)
- Session-RPE workload plus ACWR (rolling and EWMA) and PB flags
- Multi-athlete workflows with role-based scoping (coach vs athlete)
- Exports with provenance (CSV, Parquet, JSON + metadata)
- Plots and weekly PDF reports for athletes/events
- Import/migration for legacy logs; deterministic demo datasets
- Optional SQLite-backed athlete profile, strength logs, and metric forecasting

## Installation

Choose the workflow that matches your environment. See `docs/INSTALL.md` for platform notes and troubleshooting.

### PyPI / pipx

```bash
pip install throws-tracker   # formerly published as javelin-tracker
# or
pipx install throws-tracker
```

### From source (developer setup)

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e ".[dev]"
```

Set `THROWS_TRACKER_DATA_DIR` (legacy `JAVELIN_TRACKER_DATA_DIR`) if you want the CLI to store sessions outside the default `data/` directory (e.g. `export THROWS_TRACKER_DATA_DIR=~/.local/share/throws`).

## Quick Start

```bash
# Log sessions
javelin log --event javelin --throws "61.4, 60.8, 62.1" --rpe 6 --duration-minutes 55
javelin log --event discus --best 55.3 --fouls 1

# Summaries, plots, and exports
javelin summary --event shot --by week
javelin plot --event javelin --event discus
javelin export --event javelin --to export/sessions

# Weekly PDF reports
javelin weekly-report --athlete esha --event discus --week-ending 2024-06-09
```

Show the active config (allowed events, ACWR thresholds) any time with `javelin config`.

Bootstrap a sandbox dataset with:
```bash
python -m javelin_tracker seed --source demo/demo_sessions.json --force
```

## CLI overview

- `log` – Record a session (best, per-throw distances, RPE, duration, notes, tags, event, athlete/team)
- `summary` – Aggregate by week/month, PB flags, workload totals and risk highlights
- `plot` – Generate distance/volume/workload plots per athlete/event
- `weekly-report` – Produce per-event weekly PDF with tables and plots
- `export` – Write CSV/Parquet/JSON + metadata for reproducible sharing
- `import` – Ingest CSV/JSON from legacy systems, auto-migrating schema
- `config` – Display effective configuration (allowed events, ACWR thresholds)
- `seed` – Populate `data/sessions.json` with deterministic demo data
- `daily` – Show a personalised daily workout routine (DB-driven)
- `predict` / `forecast` – Forecast a metric for an athlete or ID. Supports table, JSON, or CSV output via `--format` and `--to`.
- `strength-log` – Log a strength set (exercise, kg, reps) to the DB

Run `javelin --help` or `javelin <command> --help` for full options.

Forecast examples:

```bash
# Resolve athlete from name or env; show table
javelin forecast --athlete esha --metric throw_distance --days-ahead 10

# JSON to stdout (machine-readable)
javelin forecast --athlete esha --metric throw_distance --format json

# CSV to file
javelin forecast --athlete esha --metric throw_distance --format csv --to export/esha_throw_fc.csv
```

## Multi-athlete workflows

- Attach an athlete identifier whenever you log: `javelin log --event javelin --athlete esha --team varsity --throws ...`.
- Filter summaries, plots, exports, reports, and imports with `--athlete`, `--team`, `--since`, and `--event`.
  Example: `javelin summary --athlete esha --event discus --since 2024-01-01`.
- Set `THROWS_TRACKER_ROLE=athlete` plus `THROWS_TRACKER_DEFAULT_ATHLETE=esha` (legacy `JAVELIN_…` names still work) to scope the CLI to one athlete. Leave unset (coach mode) to aggregate across the roster.

## Configuration

The CLI reads environment variables using the `THROWS_TRACKER_` prefix (legacy `JAVELIN_TRACKER_` is still accepted):

- `THROWS_TRACKER_DATA_DIR` – Base directory for app data (default: `<repo>/data`)
- `THROWS_TRACKER_SESSIONS_FILE` – Override the exact JSON file path
- `THROWS_TRACKER_CONFIG` – Path to a TOML config with allowed events and ACWR thresholds
- `THROWS_TRACKER_ROLE` – `coach` (default) or `athlete`
- `THROWS_TRACKER_DEFAULT_ATHLETE` – Default athlete name/id in athlete mode
- `THROWS_TRACKER_ENV` – `production` enables safeguards (e.g., blocking `seed` without `--allow-production`)
- `THROWS_TRACKER_DB_FILE` – SQLite database file for athlete profile/log features

Inspect the effective config with `javelin config`. See `javelin_tracker/config.py` and `docs/INSTALL.md` for details.

## Data export & reproducibility

- `javelin export --to export/sessions` emits CSV, Parquet, JSON, and a metadata JSON (version, timestamp, schema, applied filters).
- Each export row includes `event`, per-event fields (implement weight, technique, fouls when available), `app_version`, and `generated_at`.
- All date/times use ISO 8601 strings; loads and distances retain original precision.
- Import legacy logs with `javelin import --source legacy.csv --event javelin`; missing `event` values are backfilled and every row is tagged with the current `schema_version`.
- The schema is documented in `docs/schema.md`; methodology and formulas live in `docs/methods.md`.

## Demo sandbox & synthetic data

- `scripts/generate_demo_data.py` creates deterministic demo logs (`demo/demo_sessions.{json,csv}`).
- `javelin seed --source demo/demo_sessions.json --force` copies the synthetic data into `data/sessions.json`.
- The Docker entrypoint accepts `THROWS_TRACKER_BOOTSTRAP_DEMO=1` (or legacy `JAVELIN_TRACKER_BOOTSTRAP_DEMO=1`) to seed the container automatically.

More detailed instructions live in `docs/demo-sandbox.md`.

## Docker

Build and run the ready-to-use image:

```bash
docker build -t throws-tracker .
docker run --rm -it \
  -e THROWS_TRACKER_BOOTSTRAP_DEMO=1 \
  -v "$(pwd)/data:/data" \
  throws-tracker summary
```

The container runs as a non-root `javelin` user and writes sessions to `/data` (mounted as a volume by default).

## Development

- `pytest` runs unit tests and the CLI smoke test (`tests/test_smoke_cli.py`).
- `ruff check .` enforces linting and formatting guidelines.
- GitHub Actions (`.github/workflows/ci.yml`) runs lint + tests on Python 3.9, 3.11, and 3.12 for every push or pull request.

To produce an sdist/wheel and publish:

```bash
rm -rf dist
python -m build
twine upload dist/*
```

Release, container, and citation notes live in `docs/RELEASING.md`.

## Documentation & citation

- `docs/methods.md` – citable methods note covering metrics (session‑RPE, ACWR, EWMA) with references.
- `docs/INSTALL.md`, `docs/demo-sandbox.md`, and `docs/RELEASING.md` – platform-specific guides.
- `docs/privacy.md` – FERPA-safe defaults, role-based usage, anonymised demo datasets, and production safeguards.
- `CHANGELOG.md` – every public version and algorithmic adjustments.
- Cite the software via `CITATION.cff` (rendered automatically by GitHub/Zenodo once a DOI is minted).

## Directory layout

- `javelin_tracker/` – CLI commands, metrics, report generation, storage, and DB helpers
- `tests/` – Pytest suite (unit + smoke CLI coverage)
- `demo/` – Synthetic sample data for sandboxing and demos
- `scripts/` – Utility scripts (demo data generator, Docker entrypoint)
- `docs/` – Methodology, installation, compliance, and sandbox guides

Refer to `docs/` for deeper dives into workload calculations, deployment environments, and demo data flows.
