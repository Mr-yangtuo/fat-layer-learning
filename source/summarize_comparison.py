#!/usr/bin/env python3
"""Write a compact summary for a completed two-split experiment."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def read_json(path: Path):
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--task-name", required=True)
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    payload = {
        "task_name": args.task_name,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "train100": {
            "split": read_json(root / "train100" / "split_summary.json"),
            "best": read_json(root / "train100" / "best_metrics.json"),
        },
        "train90": {
            "split": read_json(root / "train90" / "split_summary.json"),
            "best": read_json(root / "train90" / "best_metrics.json"),
        },
    }
    with (root / "comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
