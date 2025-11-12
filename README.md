# Throws Tracker

Command-line utilities for logging sessions, analysing workloads, and generating reports for throwing events (javelin, discus, shot, hammer, or custom).

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

### Developer notes

- Activate the virtual environment: `source .venv/bin/activate`.
- Extend `javelin_tracker/cli.py` with new command definitions as features grow.
- Add persistent storage helpers in `javelin_tracker/storage.py` and additional analysis routines (e.g. `analysis.py`, `metrics.py`) as needed.

## Quick Start

```bash
javelin log --event javelin --throws "61.4, 60.8, 62.1" --rpe 6 --duration-minutes 55
javelin log --event discus --best 55.3 --fouls 1
javelin summary --event shot --by week
javelin plot --event javelin --event discus
javelin export --event javelin --to export/sessions
javelin weekly-report --athlete esha --event discus --week-ending 2024-06-09
```

Show the active config (allowed events, ACWR thresholds) any time with `javelin config`.

Use `python -m javelin_tracker seed --source demo/demo_sessions.json --force` to bootstrap a dataset for experimentation.

## Multi-athlete workflows

- Attach an athlete identifier whenever you log: `javelin log --event javelin --athlete esha --team varsity --throws ...`.
- Filter summaries, plots, exports, reports, and imports with `--athlete`, `--team`, `--since`, and `--event`. Example: `javelin summary --athlete esha --event discus --since 2024-01-01`.
- Set `THROWS_TRACKER_ROLE=athlete` plus `THROWS_TRACKER_DEFAULT_ATHLETE=esha` (legacy env vars with the `JAVELIN_` prefix still work) to force the CLI to view only one athlete (useful for self-service portals). Leave the role unset (coach mode) to aggregate across the roster.

## Data export & reproducibility

- `javelin export --to export/sessions` now emits CSV, Parquet, JSON, and a metadata JSON (version, timestamp, schema, applied filters). Each record contains `event`, optional per-event fields (implement weight, technique, fouls), `app_version`, and `generated_at`.
- `storage.export_csv("path.csv")` remains available for scripting contexts.
- All date/times use ISO&nbsp;8601 strings; loads and distances retain their original floating-point precision.
- Bring legacy logs into the new schema with `javelin import --source legacy.csv --event javelin`; missing `event` values are backfilled automatically and every row is tagged with the current `schema_version`.

## Demo sandbox & synthetic data

- `scripts/generate_demo_data.py` creates deterministic demo logs (`demo/demo_sessions.{json,csv}`).
- `javelin seed --source demo/demo_sessions.json --force` copies the synthetic data into `data/sessions.json`.
- The Docker entrypoint accepts `THROWS_TRACKER_BOOTSTRAP_DEMO=1` (or the legacy `JAVELIN_TRACKER_BOOTSTRAP_DEMO=1`) to seed the container automatically (see below).

More detailed instructions live in `docs/demo-sandbox.md`.

## Docker image

Build and run the ready-to-use image:

```bash
docker build -t throws-tracker .
docker run --rm -it \
  -e THROWS_TRACKER_BOOTSTRAP_DEMO=1 \
  -v "$(pwd)/data:/data" \
  throws-tracker summary
```

The container runs as a non-root `javelin` user and writes sessions to `/data` (mounted as a volume by default).

## Testing, linting & CI

- `pytest` runs the unit suite plus a CLI smoke test (`tests/test_smoke_cli.py`).
- `ruff check .` enforces linting and formatting guidelines.
- GitHub Actions (`.github/workflows/ci.yml`) runs lint + tests on Python 3.9, 3.11, and 3.12 for every push/pull request.

## Packaging & releases

The project ships as a standard PEP 621 package (`pyproject.toml`) with console entry point `javelin`. Use the following to produce an sdist/wheel and publish:

```bash
rm -rf dist
python -m build
twine upload dist/*
```

Detailed release, container, compliance, and Zenodo DOI notes live in `docs/RELEASING.md`.

## Documentation & citation

- `docs/methods.md` is the citable methods note covering all metrics (session-RPE, ACWR, EWMA) with literature references.
- `docs/INSTALL.md`, `docs/demo-sandbox.md`, and `docs/RELEASING.md` provide platform-specific setup guides.
- `docs/privacy.md` outlines FERPA-safe defaults, role-based usage, anonymised demo datasets, and production safeguards (`THROWS_TRACKER_ENV`, `THROWS_TRACKER_ROLE`; legacy names still function).
- `CHANGELOG.md` chronicles every public version so algorithm adjustments are discoverable.
- Cite the software via `CITATION.cff` (rendered automatically by GitHub/Zenodo once a DOI is minted).

## Directory layout

- `javelin_tracker/` – CLI commands, metrics, report generation, and storage helpers
- `tests/` – Pytest suite (unit tests + smoke CLI coverage)
- `demo/` – Synthetic sample data for sandboxing and demos
- `scripts/` – Utility scripts (demo data generator, Docker entrypoint)
- `docs/` – Methodology, installation, compliance, and sandbox guides

Refer to `docs/` for deeper dives into workload calculations, deployment environments, and demo data flows.
