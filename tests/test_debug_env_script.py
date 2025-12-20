from __future__ import annotations

from pathlib import Path


def test_debug_env_freeze_filters_editables_and_sorts(tmp_path: Path) -> None:
    import scripts.debug_env as de

    lines = [
        "numpy==2.2.6",
        "# comment",
        "   ",
        "-e /Users/example/project",
        "requests==2.32.5",
        "NumPy==2.2.6",  # duplicate with different casing
        "localpkg @ file:///tmp/localpkg",
    ]
    filtered = de._filter_freeze_lines(lines)
    assert "numpy==2.2.6" in filtered
    assert "requests==2.32.5" in filtered
    assert not any(line.startswith("-e ") for line in filtered)
    assert not any("file://" in line for line in filtered)
    assert len([l for l in filtered if l.lower().startswith("numpy==")]) == 1
    assert filtered == sorted(filtered, key=lambda s: s.lower())


def test_debug_env_parse_lock_reads_versions(tmp_path: Path) -> None:
    import scripts.debug_env as de

    lock = tmp_path / "requirements.lock"
    lock.write_text(
        "\n".join(
            [
                "# header",
                "numpy==2.2.6",
                "opencv-python==4.12.0.88",
                "",
            ]
        ),
        encoding="utf-8",
    )
    parsed = de._parse_lock(lock)
    assert parsed["numpy"] == "2.2.6"
    assert parsed["opencv-python"] == "4.12.0.88"

