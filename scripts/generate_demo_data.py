from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = ROOT / "demo" / "demo_sessions.json"
DEFAULT_CSV = ROOT / "demo" / "demo_sessions.csv"
DEFAULT_ATHLETES = ["athlete-001", "athlete-002", "athlete-003"]
DEFAULT_TEAMS = ["demo-team-alpha", "demo-team-bravo"]


def _build_sessions(days: int, start: date, seed: int, athletes: Sequence[str], teams: Sequence[str]) -> list[dict[str, object]]:
    rng = random.Random(seed)
    sessions: list[dict[str, object]] = []
    tags = ["run-up", "release-angle", "lift", "tempo"]

    for offset in range(days):
        session_day = start + timedelta(days=offset)
        fatigue_factor = 1 + 0.01 * offset
        best = round(58 + rng.uniform(-1.5, 2.0) * fatigue_factor, 1)
        throws = [round(best + rng.uniform(-0.8, 0.6), 1) for _ in range(rng.randint(5, 8))]
        rpe = rng.randint(4, 8)
        duration = round(rng.uniform(35, 80), 1)
        tag_subset = sorted({rng.choice(tags) for _ in range(2)})
        athlete = rng.choice(list(athletes))
        team = rng.choice(list(teams)) if teams else None
        hashed_id = hashlib.sha256(f"{athlete}:{seed}:{offset}".encode("utf-8")).hexdigest()[:12]
        note = rng.choice(
            [
                "Focus on rhythm through the runway.",
                "Strength block with reduced run-up.",
                "High winds, emphasised grip position.",
                "Active recovery with mobility.",
            ]
        )
        sessions.append(
            {
                "date": session_day.isoformat(),
                "best": best,
                "throws": throws,
                "rpe": rpe,
                "duration_minutes": duration,
                "notes": note,
                "tags": tag_subset,
                "athlete": athlete,
                "team": team,
                "athlete_hash": hashed_id,
            }
        )
    return sessions


def _write_json(path: Path, sessions: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(sessions), indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, sessions: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "date",
        "best",
        "throws",
        "rpe",
        "duration_minutes",
        "notes",
        "tags",
        "athlete",
        "team",
        "athlete_hash",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for session in sessions:
            writer.writerow(
                {
                    "date": session["date"],
                    "best": session["best"],
                    "throws": ";".join(map(str, session["throws"])),
                    "rpe": session["rpe"],
                    "duration_minutes": session["duration_minutes"],
                    "notes": session["notes"],
                    "tags": ",".join(session["tags"]),
                    "athlete": session.get("athlete"),
                    "team": session.get("team"),
                    "athlete_hash": session.get("athlete_hash"),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic demo sessions.")
    parser.add_argument("--days", type=int, default=28, help="Number of sequential days to generate.")
    parser.add_argument(
        "--start-date",
        type=date.fromisoformat,
        default=(date.today() - timedelta(days=27)).isoformat(),
        help="Start date (YYYY-MM-DD). Defaults to 27 days before today.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON, help="Destination .json file.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Destination .csv file.")
    parser.add_argument(
        "--athletes",
        nargs="+",
        default=DEFAULT_ATHLETES,
        help="Space-separated anonymised athlete identifiers (default: %(default)s).",
    )
    parser.add_argument(
        "--teams",
        nargs="+",
        default=DEFAULT_TEAMS,
        help="Space-separated team labels (default: %(default)s).",
    )
    args = parser.parse_args()

    if isinstance(args.start_date, str):
        start = date.fromisoformat(args.start_date)
    else:
        start = args.start_date

    sessions = _build_sessions(days=args.days, start=start, seed=args.seed, athletes=args.athletes, teams=args.teams)
    _write_json(args.json, sessions)
    _write_csv(args.csv, sessions)

    print(f"Wrote {len(sessions)} demo sessions to {args.json} and {args.csv}")


if __name__ == "__main__":
    main()
