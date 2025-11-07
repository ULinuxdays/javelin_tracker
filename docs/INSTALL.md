## Installation Guide

These instructions cover the most common deployment targets. The CLI supports Python 3.9–3.12 on macOS, Linux, and Windows (WSL recommended).

### 1. Prerequisites

- Python ≥ 3.9 with `pip`
- Optional: `pipx` for isolated CLI installs
- System libraries for `matplotlib`/`reportlab` when compiling from source (Linux: `sudo apt install build-essential libfreetype6-dev libpng-dev libjpeg-dev zlib1g-dev`)

### 2. Stable install from PyPI/pipx

```bash
# global environment (virtual environments still encouraged)
python -m pip install --upgrade pip
pip install javelin-tracker

# isolated CLI using pipx
pipx install javelin-tracker
```

After installation, confirm the binary is available:

```bash
javelin --help
```

### 3. Editable developer install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e ".[dev]"
```

- `requirements.txt` locks minimum versions for quick reproduction on research machines.
- `.[dev]` pulls in Ruff, Pytest, Build, and Twine for linting, testing, and publishing.

### 4. Data directory & configuration

- Default storage lives in `<repo>/data/sessions.json`. This path is ignored by git.
- Override the location via `export JAVELIN_TRACKER_DATA_DIR=/secure/storage/path`.
- Provide `JAVELIN_TRACKER_SESSIONS_FILE` if you prefer a custom filename.
- Set `JAVELIN_TRACKER_ROLE=athlete` (plus `JAVELIN_TRACKER_DEFAULT_ATHLETE=<id>`) on shared workstations so the CLI automatically scopes to a single athlete. Leave unset (coach mode) to view every athlete.
- Use `JAVELIN_TRACKER_ENV=production` on servers to enable safeguards such as seed-command blocking (requires `--allow-production` overrides).

### 5. Smoke test

```bash
pytest tests/test_smoke_cli.py -q
```

This end-to-end test logs a fake session, summarises it, and exercises the export pipeline to ensure dependencies were compiled correctly.
