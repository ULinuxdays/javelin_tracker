#!/usr/bin/env python3
"""Capture/freeze/verify the biomechanics runtime environment.

Commands:
  capture  -> write a JSON snapshot (OS/Python/CPU/GPU + key package versions)
  freeze   -> generate a fully pinned requirements lock from the current env
  verify   -> check current env matches the lock + sanity-import pose deps
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    except Exception:
        return None
    out = out.strip()
    return out or None


def _cmd_exists(name: str) -> bool:
    return shutil.which(name) is not None


def _system_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
    }

    # OS-specific details (best-effort; avoid PII such as usernames).
    if info["system"] == "Darwin":
        info["os"] = {"sw_vers": _run(["sw_vers"])}
        if _cmd_exists("system_profiler"):
            info["hardware"] = {
                "sp_hardware": _run(["system_profiler", "SPHardwareDataType"]),
                "sp_displays": _run(["system_profiler", "SPDisplaysDataType"]),
            }
    elif info["system"] == "Linux":
        info["os"] = {"uname": _run(["uname", "-a"])}
        if _cmd_exists("lscpu"):
            info.setdefault("hardware", {})["lscpu"] = _run(["lscpu"])
        if _cmd_exists("nvidia-smi"):
            info.setdefault("hardware", {})["nvidia_smi"] = _run(["nvidia-smi"])
    elif info["system"] == "Windows":
        info["os"] = {"ver": _run(["cmd", "/c", "ver"])}
        if _cmd_exists("wmic"):
            info.setdefault("hardware", {})["cpu"] = _run(["wmic", "cpu", "get", "name"])
            info.setdefault("hardware", {})["gpu"] = _run(["wmic", "path", "win32_videocontroller", "get", "name"])

    return info


def _pip_json_list() -> list[dict[str, Any]]:
    # pip list is more structured than freeze and avoids local editable paths.
    out = _run([sys.executable, "-m", "pip", "list", "--format=json"])
    if not out:
        return []
    try:
        data = json.loads(out)
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    out_list: list[dict[str, Any]] = []
    for row in data:
        if not isinstance(row, Mapping):
            continue
        name = str(row.get("name") or "")
        version = str(row.get("version") or "")
        if name and version:
            out_list.append({"name": name, "version": version})
    return out_list


def _package_versions(packages: Iterable[str]) -> dict[str, str | None]:
    installed = {row["name"].lower(): str(row["version"]) for row in _pip_json_list() if "name" in row and "version" in row}
    out: dict[str, str | None] = {}
    for name in packages:
        out[name] = installed.get(name.lower())
    return out


def cmd_capture(args: argparse.Namespace) -> None:
    path = Path(args.out).expanduser() if args.out else (REPO_ROOT / "debug_assets" / "metadata" / "debug_env_snapshot.json")
    path.parent.mkdir(parents=True, exist_ok=True)

    key_pkgs = [
        "mediapipe",
        "opencv-python",
        "numpy",
        "scipy",
        "pandas",
        "Flask",
    ]
    payload = {
        "captured_at": _now_iso(),
        "system": _system_info(),
        "pip": {
            "version": _run([sys.executable, "-m", "pip", "--version"]),
            "packages_key": _package_versions(key_pkgs),
        },
        "app_env": {
            # Only include vars that affect tracking; omit secrets.
            k: os.environ.get(k)
            for k in (
                "THROWS_TRACKER_POSE_LANDMARKER_MODEL_VARIANT",
                "THROWS_TRACKER_POSE_TIME_SCALE",
                "THROWS_TRACKER_POSE_ASSUMED_FPS",
                "THROWS_TRACKER_POSE_ROI",
                "THROWS_TRACKER_POSE_OPTICAL_FLOW",
                "THROWS_TRACKER_POSE_SEGMENTATION",
            )
            if os.environ.get(k) is not None
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote: {path}")


def _filter_freeze_lines(lines: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        # Drop editable installs and local file references (repo itself will be installed separately).
        if line.startswith("-e "):
            continue
        if " @ file://" in line or " @ " in line and "://" in line:
            continue
        # Normalize name casing for dedupe: package==version
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(line)
    return sorted(out, key=lambda s: s.lower())


def cmd_freeze(args: argparse.Namespace) -> None:
    out_path = Path(args.out).expanduser() if args.out else (REPO_ROOT / "requirements.lock")

    freeze = _run([sys.executable, "-m", "pip", "freeze"])
    if not freeze:
        raise SystemExit("Unable to run `pip freeze`.")
    lines = freeze.splitlines()
    pinned = _filter_freeze_lines(lines)

    header = [
        f"# requirements.lock (generated) - {_now_iso()}",
        f"# platform: {platform.platform()}",
        f"# python: {platform.python_version()} ({platform.python_implementation()})",
        "#",
        "# Install:",
        "#   python -m venv .venv && . .venv/bin/activate",
        "#   python -m pip install --upgrade pip",
        "#   pip install -r requirements.lock",
        "#   pip install -e .",
        "",
    ]
    out_path.write_text("\n".join(header + pinned) + "\n", encoding="utf-8")
    print(f"Wrote: {out_path} ({len(pinned)} pinned packages)")


def _parse_lock(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "==" not in line:
            continue
        name, version = line.split("==", 1)
        name = name.strip().lower()
        version = version.strip()
        if name and version:
            out[name] = version
    return out


def _installed_versions() -> dict[str, str]:
    return {row["name"].lower(): str(row["version"]) for row in _pip_json_list() if row.get("name") and row.get("version")}


def _sanity_imports() -> list[str]:
    errors: list[str] = []
    try:
        import cv2  # noqa: F401
    except Exception as exc:
        errors.append(f"import cv2 failed: {exc}")
    try:
        import mediapipe  # noqa: F401
    except Exception as exc:
        errors.append(f"import mediapipe failed: {exc}")
    try:
        from javelin_tracker.biomechanics.pose_estimation import PosePipeline  # noqa: F401
    except Exception as exc:
        errors.append(f"import PosePipeline failed: {exc}")
    return errors


def cmd_verify(args: argparse.Namespace) -> None:
    lock_path = Path(args.lock).expanduser()
    if not lock_path.exists():
        raise SystemExit(f"Lock not found: {lock_path}")

    expected = _parse_lock(lock_path)
    installed = _installed_versions()

    missing = sorted([name for name in expected if name not in installed])
    mismatched = sorted(
        [
            (name, expected[name], installed.get(name))
            for name in expected
            if name in installed and installed.get(name) != expected[name]
        ],
        key=lambda t: t[0],
    )

    if missing:
        print("Missing packages:")
        for name in missing[:50]:
            print(f"- {name}=={expected[name]}")
        if len(missing) > 50:
            print(f"... and {len(missing) - 50} more")

    if mismatched:
        print("Version mismatches:")
        for name, exp, got in mismatched[:80]:
            print(f"- {name}: expected {exp}, installed {got}")
        if len(mismatched) > 80:
            print(f"... and {len(mismatched) - 80} more")

    sanity_errors = _sanity_imports()
    if sanity_errors:
        print("Sanity import errors:")
        for err in sanity_errors:
            print(f"- {err}")

    if missing or mismatched or sanity_errors:
        raise SystemExit(2)

    print("OK: environment matches lock and core imports work.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="debug_env", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    cap = sub.add_parser("capture", help="Write a JSON environment snapshot.")
    cap.add_argument("--out", default=None, help="Output JSON path.")
    cap.set_defaults(func=cmd_capture)

    frz = sub.add_parser("freeze", help="Generate requirements.lock from current env.")
    frz.add_argument("--out", default=None, help="Output lock path (default: requirements.lock).")
    frz.set_defaults(func=cmd_freeze)

    ver = sub.add_parser("verify", help="Verify env matches a lock file and imports work.")
    ver.add_argument("--lock", default="requirements.lock", help="Lock file path.")
    ver.set_defaults(func=cmd_verify)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

