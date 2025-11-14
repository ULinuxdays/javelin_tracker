## Data Protection & Privacy Checklist

This project manages training information for identifiable student-athletes. Use the following safeguards whenever you deploy Throws Tracker in academic or athletic environments (legacy Javelin Tracker installations remain compatible).

### 1. Data Classification & Storage

- Treat *all* session data (dates, notes, best throws, workload metrics) as education records protected under FERPA/GDPR equivalents.
- Configure secure storage using `THROWS_TRACKER_DATA_DIR` or `THROWS_TRACKER_SESSIONS_FILE` (legacy env names with the `JAVELIN_` prefix still work) so logs live on approved drives with access controls.
- Keep repositories (`git`) free of raw data. The default `data/` directory is ignored; do not change this setting when working with live athletes.

### 2. Roles & Least-Privilege Access

- Set `THROWS_TRACKER_ROLE=athlete` on personal devices to force CLI commands to scope automatically to the user’s own athlete ID. Pair this with `THROWS_TRACKER_DEFAULT_ATHLETE` to lock the CLI to a single profile.
- Keep the default `coach` behaviour for analysts who need multi-athlete rollups; ensure they understand the privacy obligations for cross-athlete comparisons.

### 3. Production Safeguards

- Use `THROWS_TRACKER_ENV=production` on managed servers. The CLI blocks seed/demo commands unless `--allow-production` is passed explicitly, reducing the risk of overwriting live data.
- Audit cron jobs or CI runners to confirm they do **not** run destructive commands (seed, delete, etc.) without the `--allow-production` flag.

### 4. Anonymisation & Demo Data

- Synthetic datasets (`demo/demo_sessions.*`) now include anonymised athlete identifiers plus a hashed `athlete_hash` column. Use these when sharing tutorials, screenshots, or reproducing analyses.
- If you need to export data for research, rely on `javelin export --to ... --athlete <id>` and strip personally identifiable metadata (names, notes) before circulation.

### 5. Retention & Deletion

- Establish institutional retention windows (e.g., delete logs after the competitive season) and document them. Build periodic jobs that archive/export data before removal.
- When athletes graduate or leave the program, delete their session records or replace identifiers with pseudonyms.

### 6. Secure Transport & Backups

- Use encrypted channels (SSH/HTTPS/VPN) when copying `sessions.json` or export files off the host machine.
- Encrypt backups at rest and restrict restore rights to approved staff. Keep a minimal audit log of who exported data and when.

### 7. Transparency & Consent

- Reference this document in onboarding materials for athletes/coaches. Make clear which data points are collected, how they are used, and who can access them.
- Before introducing new analytics (e.g., risk scoring), log the change in `CHANGELOG.md` and include a short rationale so downstream researchers can cite the algorithm history.
