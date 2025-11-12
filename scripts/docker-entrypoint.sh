#!/usr/bin/env bash
set -euo pipefail

BOOTSTRAP_DEMO="${THROWS_TRACKER_BOOTSTRAP_DEMO:-${JAVELIN_TRACKER_BOOTSTRAP_DEMO:-0}}"
DATA_DIR="${THROWS_TRACKER_DATA_DIR:-${JAVELIN_TRACKER_DATA_DIR:-/data}}"

if [[ "${BOOTSTRAP_DEMO}" != "0" ]]; then
  echo "[entrypoint] Seeding demo data into ${DATA_DIR}"
  python -m javelin_tracker seed --source /app/demo/demo_sessions.json --force --verbose || true
fi

exec "$@"
