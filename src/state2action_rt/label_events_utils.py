from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def append_event(path: str | Path, event: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=True) + "\n")


def count_events(path: str | Path) -> int:
    path = Path(path)
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def undo_last_event(path: str | Path) -> Optional[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8").splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return None
    last_raw = lines.pop()
    text = "\n".join(lines)
    if text:
        text += "\n"
    path.write_text(text, encoding="utf-8")
    try:
        return json.loads(last_raw)
    except json.JSONDecodeError:
        return None


def apply_roi_offset(
    x: float, y: float, roi: Optional[Tuple[int, int, int, int]]
) -> Tuple[float, float]:
    if roi is None:
        return x, y
    x1, y1, _, _ = roi
    return x + x1, y + y1
