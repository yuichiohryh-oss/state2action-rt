from __future__ import annotations

from typing import Tuple


def xy_to_grid_id(
    x: float,
    y: float,
    roi: Tuple[int, int, int, int],
    gw: int = 6,
    gh: int = 9,
) -> int:
    x1, y1, x2, y2 = roi
    roi_w = max(1.0, float(x2 - x1))
    roi_h = max(1.0, float(y2 - y1))

    cx = min(max(x, x1), x2 - 1)
    cy = min(max(y, y1), y2 - 1)

    cell_w = roi_w / gw
    cell_h = roi_h / gh

    col = int((cx - x1) / cell_w)
    row = int((cy - y1) / cell_h)

    col = min(max(col, 0), gw - 1)
    row = min(max(row, 0), gh - 1)

    return row * gw + col


def grid_id_to_cell_rect(
    grid_id: int,
    roi: Tuple[int, int, int, int],
    gw: int = 6,
    gh: int = 9,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = roi
    roi_w = max(1.0, float(x2 - x1))
    roi_h = max(1.0, float(y2 - y1))
    cell_w = roi_w / gw
    cell_h = roi_h / gh

    grid_id = min(max(grid_id, 0), gw * gh - 1)
    row = grid_id // gw
    col = grid_id % gw

    cx1 = int(round(x1 + col * cell_w))
    cy1 = int(round(y1 + row * cell_h))
    cx2 = int(round(x1 + (col + 1) * cell_w))
    cy2 = int(round(y1 + (row + 1) * cell_h))

    return (cx1, cy1, cx2, cy2)
