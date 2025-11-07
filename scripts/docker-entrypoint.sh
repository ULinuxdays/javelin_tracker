#!/usr/bin/env bash
set -euo pipefail

if [[ "${JAVELIN_TRACKER_BOOTSTRAP_DEMO:-0}" != "0" ]]; then
  echo "[entrypoint] Seeding demo data into $JAVELIN_TRACKER_DATA_DIR"
  python -m javelin_tracker seed --source /app/demo/demo_sessions.json --force --verbose || true
fi

exec "$@"
