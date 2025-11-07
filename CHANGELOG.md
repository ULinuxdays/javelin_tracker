# Changelog

All notable changes to this project will be documented here. The format roughly follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); dates use ISO 8601.

## [0.1.0] - 2025-02-14

### Added
- Initial public CLI with session logging, workload analytics, ACWR computations, and weekly PDF reporting.
- Deterministic demo data generator plus Docker workflows for sandboxing and reproducible installs.
- GitHub Actions pipeline (lint + tests across Python 3.9/3.11/3.12), smoke tests, and Ruff enforcement.
- PyPI-ready metadata, MIT license, and comprehensive installation/releasing guides with Zenodo/CITATION instructions.
- Multi-format exports (CSV, Parquet, JSON) with embedded metadata for reproducible research.
- Multi-athlete logging/filtering via `--athlete/--team/--since`, role-aware CLI defaults, and privacy safeguards (`docs/privacy.md`).
- Weekly PDF reports now include team/school branding, athlete-specific filenames, and color-coded daily risk tables.
- Demo datasets populate anonymised athlete IDs plus hashed identifiers for FERPA-safe sharing.

### Changed
- Storage paths now respect `JAVELIN_TRACKER_DATA_DIR`/`JAVELIN_TRACKER_SESSIONS_FILE` for secure deployments.
- Documentation reorganized into `docs/` (methods, installation, demo sandbox, releasing).

### Fixed
- Improved validation around session persistence and export serialization.

[0.1.0]: https://github.com/uday/Javelin/releases/tag/v0.1.0
