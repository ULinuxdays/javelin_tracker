from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterable, List

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _data_dir() -> Path:
    override = os.getenv("JAVELIN_TRACKER_DATA_DIR")
    base = Path(override).expanduser() if override else DEFAULT_DATA_DIR
    base.mkdir(parents=True, exist_ok=True)
    return base


def _sessions_file() -> Path:
    override = os.getenv("JAVELIN_TRACKER_SESSIONS_FILE")
    if override:
        target = Path(override).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        return target
    return _data_dir() / "sessions.json"


def load_sessions() -> List[Any]:
    sessions_file = _sessions_file()
    if not sessions_file.exists():
        sessions_file.write_text("[]\n", encoding="utf-8")
        return []

    raw = sessions_file.read_text(encoding="utf-8").strip() or "[]"
    try:
        sessions = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse {sessions_file}: {exc}") from exc

    if not isinstance(sessions, list):
        raise ValueError(f"{sessions_file} must contain a JSON list")
    return sessions


def save_sessions(sessions: Iterable[Any]) -> None:
    sessions_file = _sessions_file()
    sessions_file.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(list(sessions), indent=2, sort_keys=True) + "\n"

    with NamedTemporaryFile(
        "w", dir=sessions_file.parent, delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(payload)
        temp_path = Path(tmp.name)
    temp_path.replace(sessions_file)


def append_session(session: Any) -> List[Any]:
    sessions = load_sessions()
    sessions.append(session)
    save_sessions(sessions)
    return sessions


def export_csv(path: Path | str) -> Path:
    sessions = load_sessions()
    flattened = [_flatten(record) for record in sessions]
    fieldnames = sorted({key for row in flattened for key in row.keys()})

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flattened)
    return target


def _flatten(payload: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(payload, dict):
        items = {}
        for key, value in payload.items():
            full_key = f"{prefix}.{key}" if prefix else str(key)
            items.update(_flatten(value, full_key))
        return items

    if isinstance(payload, list):
        if all(isinstance(item, dict) for item in payload):
            items = {}
            for index, value in enumerate(payload):
                full_key = f"{prefix}[{index}]"
                items.update(_flatten(value, full_key))
            return items
        return {prefix: ";".join(map(str, payload)) if prefix else list(payload)}

    return {prefix: payload} if prefix else {"value": payload}
