from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_biomech_repro_register_writes_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Import as a module so we can monkeypatch paths and helpers without spawning subprocesses.
    import scripts.biomech_repro as repro

    monkeypatch.setattr(repro, "DEBUG_ROOT", tmp_path / "debug_assets")
    monkeypatch.setattr(repro, "VIDEOS_DIR", repro.DEBUG_ROOT / "videos")
    monkeypatch.setattr(repro, "META_DIR", repro.DEBUG_ROOT / "metadata")
    monkeypatch.setattr(repro, "RUNS_DIR", repro.DEBUG_ROOT / "runs")
    repro._ensure_dirs()

    src = tmp_path / "src.mp4"
    src.write_bytes(b"video-bytes")

    monkeypatch.setattr(repro, "_video_metadata", lambda _p: {"width": 1920, "height": 1080, "fps": 60.0, "total_frames": 600, "duration_seconds": 10.0})
    monkeypatch.setattr(repro, "_estimate_vfr", lambda _p, **_kw: {"checked": True, "vfr_suspected": False})
    monkeypatch.setattr(repro, "_sha256", lambda _p: "deadbeef")

    args = repro.build_parser().parse_args(
        [
            "register",
            "--clip-type",
            "good",
            "--source",
            str(src),
            "--view",
            "side",
            "--camera",
            "static",
            "--date",
            "20250101",
            "--version",
            "1",
        ]
    )
    args.func(args)

    vids = list((repro.VIDEOS_DIR).glob("*.mp4"))
    assert len(vids) == 1
    assert vids[0].name.startswith("debug_good_v1_20250101_1080p_60fps_side_static")

    meta_files = list((repro.META_DIR).glob("*.json"))
    assert len(meta_files) == 1
    meta = json.loads(meta_files[0].read_text(encoding="utf-8"))
    assert meta["sha256"] == "deadbeef"
    assert meta["clip_type"] == "good"

