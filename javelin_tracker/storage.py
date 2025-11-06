from __future__ import annotations

import csv
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterable, List

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SESSIONS_FILE = DATA_DIR / "sessions.json"


def load_sessions() -> List[Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not SESSIONS_FILE.exists():
        SESSIONS_FILE.write_text("[]\n", encoding="utf-8")
        return []

    raw = SESSIONS_FILE.read_text(encoding="utf-8").strip() or "[]"
    try:
        sessions = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse {SESSIONS_FILE}: {exc}") from exc

    if not isinstance(sessions, list):
        raise ValueError(f"{SESSIONS_FILE} must contain a JSON list")
    return sessions


def save_sessions(sessions: Iterable[Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(list(sessions), indent=2, sort_keys=True) + "\n"

    with NamedTemporaryFile("w", dir=DATA_DIR, delete=False, encoding="utf-8") as tmp:
        tmp.write(payload)
        temp_path = Path(tmp.name)
    temp_path.replace(SESSIONS_FILE)


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
