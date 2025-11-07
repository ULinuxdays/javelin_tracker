## Demo Sandbox & Synthetic Data

Use the bundled assets to trial the tracker without touching real athlete information.

### 1. Generate demo data

```bash
python scripts/generate_demo_data.py \
  --days 35 \
  --json demo/demo_sessions.json \
  --csv demo/demo_sessions.csv
```

- Deterministic seed (`--seed`) makes automated demos repeatable.
- Adjust `--start-date` (YYYY-MM-DD) to align with the current competitive season.

### 2. Load demo data into the app

```bash
export JAVELIN_TRACKER_DATA_DIR=$(pwd)/data/demo
python -m javelin_tracker seed --source demo/demo_sessions.json --force --verbose
javelin summary --by week
```

The `seed` command copies the synthetic JSON into the configured data directory. Because `data/` is ignored by git, seeding is safe on shared machines. Run `javelin export --to demo/exports` afterwards to capture CSV/Parquet/JSON plus metadata files for analysts who need the exact numeric inputs used in demonstrations. Demo records include anonymised athlete IDs and short hashes so you can share outputs without exposing real names.

### 3. Dockerised sandbox

The Docker image seeds the demo payload automatically when `JAVELIN_TRACKER_BOOTSTRAP_DEMO=1`.

```bash
docker build -t javelin-tracker .
docker run --rm -it \
  -e JAVELIN_TRACKER_BOOTSTRAP_DEMO=1 \
  javelin-tracker summary
```

Mount a host directory if you want to persist the seeded sessions:

```bash
docker run --rm -it \
  -v "$(pwd)/demo-data:/data" \
  -e JAVELIN_TRACKER_BOOTSTRAP_DEMO=1 \
  javelin-tracker summary
```

### 4. Resetting the sandbox

- Delete the `data/` directory or point `JAVELIN_TRACKER_DATA_DIR` to a new location.
- Re-run the generator/seed steps as needed.
- Add the CLI smoke test (`pytest tests/test_smoke_cli.py`) to any sandbox validation job to make sure core flows—log, summary, export—keep working.
