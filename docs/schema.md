# Session Schema Reference

This file documents the persistent session schema written to `data/sessions.json`, exported CSV/Parquet/JSON files, and any downstream seed/import artifacts. Use `schema_version` to detect breaking changes.

## Current version

- `schema_version`: 2  
- `allowed_events`: configurable (defaults to `["javelin", "discus", "shot", "hammer"]`)

## Fields

| Field | Type | Notes |
| --- | --- | --- |
| `date` | string (ISO `YYYY-MM-DD`) | Session date, timezone-naïve UTC. |
| `athlete` | string | Identifier; defaults to `unassigned` when omitted. |
| `team` | string \| null | Optional roster tag. |
| `event` | string | Required for all new rows. Legacy rows migrate to `"javelin"`. |
| `best` | float | Best throw distance (metres). |
| `throws` | list\[float] | Ordered distances for the session. |
| `throws_count` | integer | Derived column used in exports/reports. |
| `rpe` | integer \| null | Rate of Perceived Exertion (1–10). |
| `duration_minutes` | float | Non-negative. Defaults to `0.0` if omitted. |
| `load` | float | Calculated as `rpe × duration_minutes`. Backfilled during migration/import. |
| `implement_weight_kg` | float \| null | Optional per-event metadata. |
| `technique` | string \| null | Optional descriptor (e.g. `"3-step"`, `"spin"`). |
| `fouls` | integer \| null | Optional count of fouls. |
| `tags` | list\[string] | Normalised/sanitised labels. |
| `notes` | string \| null | Free-form text. |
| `app_version` | string | Added during export for provenance. |
| `generated_at` | string | ISO&nbsp;8601 timestamp when exports were produced. |

## Migration helpers

- `javelin_tracker.storage.load_sessions()` transparently migrates legacy files (adds `event="javelin"`, recomputes `load`, adds `schema_version=2`).
- `javelin import` reads CSV/JSON from older systems, applies the same migration rules, and accepts `--event` to supply a default when rows are missing an explicit event.

Keep this document in sync with `docs/methods.md` and the CLI help output whenever the schema evolves.
