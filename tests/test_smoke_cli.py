from __future__ import annotations

import json

from typer.testing import CliRunner

from javelin_tracker.cli import app


def test_cli_smoke(tmp_path, monkeypatch):
    runner = CliRunner()
    data_dir = tmp_path / "sessions"
    monkeypatch.setenv("JAVELIN_TRACKER_DATA_DIR", str(data_dir))

    log_result = runner.invoke(
        app,
        [
            "log",
            "--date",
            "2024-05-01",
            "--best",
            "65.5",
            "--throws",
            "65.5,64.8,66.1",
            "--rpe",
            "6",
            "--duration-minutes",
            "55",
            "--notes",
            "e2e smoke",
            "--tags",
            "smoke,ci",
            "--athlete",
            "esha",
            "--team",
            "Jaguars",
            "--verbose",
        ],
    )
    assert log_result.exit_code == 0, log_result.stdout
    assert "[esha]" in log_result.stdout

    summary_result = runner.invoke(app, ["summary", "--athlete", "esha"])
    assert summary_result.exit_code == 0, summary_result.stdout
    assert "Filters: athlete=esha" in summary_result.stdout
    assert "Totals" in summary_result.stdout

    export_dir = tmp_path / "exports"
    export_result = runner.invoke(app, ["export", "--to", str(export_dir), "--athlete", "esha"])
    assert export_result.exit_code == 0, export_result.stdout

    csv_files = list(export_dir.glob("*.csv"))
    parquet_files = list(export_dir.glob("*.parquet"))
    assert csv_files, "Expected at least one CSV export"
    assert parquet_files, "Expected at least one Parquet export"

    raw_json = [path for path in export_dir.glob("*.json") if not path.name.endswith("_metadata.json")]
    metadata_files = list(export_dir.glob("*_metadata.json"))
    assert raw_json, "Expected JSON export for reproducibility"
    assert metadata_files, "Expected metadata JSON to describe the export"

    export_payload = json.loads(raw_json[0].read_text(encoding="utf-8"))
    assert export_payload and "app_version" in export_payload[0]
    assert export_payload[0]["exported_at"]

    metadata_payload = json.loads(metadata_files[0].read_text(encoding="utf-8"))
    assert metadata_payload["application"] == "javelin-tracker"
    assert metadata_payload["rows"] >= 1
    assert metadata_payload["filters"]["athlete"] == "esha"
