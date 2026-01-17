from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np

ElixirRoiPixels = Tuple[int | None, int | None, int | None, int | None]

# Approximate default for scrcpy 330x752; adjust via tools/inspect_elixir.py.
DEFAULT_ELIXIR_ROI_330x752 = (40, 712, 320, 745)


def compute_elixir_roi_rect(
    frame_bgr: np.ndarray,
    elixir_roi_pixels: ElixirRoiPixels | None = None,
) -> Tuple[int, int, int, int]:
    h, w = frame_bgr.shape[:2]
    if elixir_roi_pixels is None:
        elixir_roi_pixels = (None, None, None, None)
    x1_px, x2_px, y1_px, y2_px = elixir_roi_pixels
    if any(value is not None for value in elixir_roi_pixels):
        if not all(value is not None for value in elixir_roi_pixels):
            raise ValueError("elixir ROI override requires x1/x2/y1/y2")
        x1 = int(x1_px)
        x2 = int(x2_px)
        y1 = int(y1_px)
        y2 = int(y2_px)
    elif w == 330 and h == 752:
        x1, y1, x2, y2 = DEFAULT_ELIXIR_ROI_330x752
    else:
        base_w, base_h = 330.0, 752.0
        x1 = int(round(w * (DEFAULT_ELIXIR_ROI_330x752[0] / base_w)))
        y1 = int(round(h * (DEFAULT_ELIXIR_ROI_330x752[1] / base_h)))
        x2 = int(round(w * (DEFAULT_ELIXIR_ROI_330x752[2] / base_w)))
        y2 = int(round(h * (DEFAULT_ELIXIR_ROI_330x752[3] / base_h)))

    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("elixir ROI is empty")
    return x1, y1, x2, y2


def compute_elixir_metrics(
    roi_bgr: np.ndarray,
    purple_h_min: int = 120,
    purple_h_max: int = 170,
    purple_s_min: int = 60,
    purple_v_min: int = 40,
    col_fill_ratio_th: float = 0.35,
) -> Dict[str, float | int]:
    if roi_bgr.size == 0:
        raise ValueError("elixir ROI image is empty")
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    purple_mask = (
        (h >= purple_h_min)
        & (h <= purple_h_max)
        & (s >= purple_s_min)
        & (v >= purple_v_min)
    )
    purple_ratio = float(purple_mask.mean())
    col_ratios = purple_mask.mean(axis=0)
    filled_mask = col_ratios >= col_fill_ratio_th
    filled_cols = int(filled_mask.sum())
    width = int(roi_bgr.shape[1])
    fill_ratio = float(filled_cols / width) if width > 0 else 0.0
    holes_ratio = 1.0
    if filled_cols > 0:
        first = int(np.argmax(filled_mask))
        last = int(width - 1 - np.argmax(filled_mask[::-1]))
        span = max(1, last - first + 1)
        holes = span - filled_cols
        holes_ratio = float(holes / span)
    mean_s = float(s.mean())
    return {
        "purple_ratio": purple_ratio,
        "holes_ratio": holes_ratio,
        "filled_cols": filled_cols,
        "fill_ratio": fill_ratio,
        "mean_s": mean_s,
    }


def estimate_elixir_from_frame(
    frame_bgr: np.ndarray,
    elixir_roi_pixels: ElixirRoiPixels | None = None,
    purple_h_min: int = 120,
    purple_h_max: int = 170,
    purple_s_min: int = 60,
    purple_v_min: int = 40,
    col_fill_ratio_th: float = 0.35,
    min_purple_ratio: float = 0.01,
    max_holes_ratio: float = 0.6,
    allow_empty: bool = False,
    empty_purple_ratio_max: float = 0.002,
    empty_mean_s_max: float = 80.0,
) -> Dict[str, object]:
    x1, y1, x2, y2 = compute_elixir_roi_rect(frame_bgr, elixir_roi_pixels=elixir_roi_pixels)
    roi_bgr = frame_bgr[y1:y2, x1:x2]
    metrics = compute_elixir_metrics(
        roi_bgr,
        purple_h_min=purple_h_min,
        purple_h_max=purple_h_max,
        purple_s_min=purple_s_min,
        purple_v_min=purple_v_min,
        col_fill_ratio_th=col_fill_ratio_th,
    )
    purple_ratio = float(metrics["purple_ratio"])
    holes_ratio = float(metrics["holes_ratio"])
    filled_cols = int(metrics["filled_cols"])
    mean_s = float(metrics["mean_s"])

    stable = True
    elixir = -1
    elixir_frac = -1.0

    if filled_cols == 0:
        if allow_empty and purple_ratio <= empty_purple_ratio_max and mean_s <= empty_mean_s_max:
            elixir = 0
            elixir_frac = 0.0
        else:
            stable = False
    else:
        if purple_ratio < min_purple_ratio or holes_ratio > max_holes_ratio:
            stable = False
        else:
            elixir_frac = float(metrics["fill_ratio"]) * 10.0
            elixir_frac = max(0.0, min(10.0, elixir_frac))
            elixir = int(round(elixir_frac))
            elixir = max(0, min(10, elixir))

    if not stable:
        elixir = -1
        elixir_frac = -1.0

    return {
        **metrics,
        "elixir": elixir,
        "elixir_frac": elixir_frac,
        "roi": (x1, y1, x2, y2),
        "roi_bgr": roi_bgr,
    }
