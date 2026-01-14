from __future__ import annotations

import json
from typing import Callable, Dict, Iterable, List

REQUIRED_KEYS = ("t", "x", "y", "action_id")


def parse_events_jsonl(path: str, warn_fn: Callable[[str], None]) -> List[Dict]:
    events: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError as exc:
                warn_fn(f"events.jsonl line {line_no}: invalid json ({exc})")
                continue

            missing = [k for k in REQUIRED_KEYS if k not in item]
            if missing:
                warn_fn(f"events.jsonl line {line_no}: missing keys {missing}")
                continue

            try:
                item["t"] = float(item["t"])
                item["x"] = float(item["x"])
                item["y"] = float(item["y"])
                item["action_id"] = str(item["action_id"])
            except (TypeError, ValueError):
                warn_fn(f"events.jsonl line {line_no}: invalid value types")
                continue

            events.append(item)

    return events
