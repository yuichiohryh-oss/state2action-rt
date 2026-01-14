from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import warnings

import cv2
import numpy as np


@dataclass(frozen=True)
class RoiConfig:
    y1: int = 70
    y2_mode: str = "auto"  # auto | fixed
    y2_fixed: int | None = None
    x1: int = 0
    x2: int | None = None


def detect_bottom_ui_y(frame_bgr: np.ndarray, fallback_ratio: float = 0.85) -> int:
    """
    Detect the start of the bottom UI bar by row-wise intensity change.
    Returns y2 (exclusive) for the playable area.
    """
    h, w = frame_bgr.shape[:2]
    if h < 2:
        return int(h * fallback_ratio)

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    row_means = gray.mean(axis=1)
    diffs = np.abs(row_means[1:] - row_means[:-1])

    mean = float(diffs.mean()) if diffs.size else 0.0
    std = float(diffs.std()) if diffs.size else 0.0
    threshold = max(8.0, mean + 2.0 * std)

    search_start = h - 2
    search_end = int(h * 0.4)
    for y in range(search_start, search_end, -1):
        if diffs[y - 1] > threshold:
            return y

    return int(h * fallback_ratio)


def detect_roi(frame_bgr: np.ndarray, config: RoiConfig) -> Tuple[int, int, int, int]:
    h, w = frame_bgr.shape[:2]
    x1 = max(0, config.x1)
    x2 = w if config.x2 is None else min(w, config.x2)
    y1 = max(0, config.y1)
    if config.y2_mode == "fixed":
        y2 = config.y2_fixed if config.y2_fixed is not None else int(h * 0.85)
    else:
        y2 = detect_bottom_ui_y(frame_bgr)

        roi_h = y2 - y1
        if roi_h < int(h * 0.4) or roi_h > int(h * 0.95):
            if config.y2_fixed is not None:
                y2 = config.y2_fixed
            else:
                y2 = int(h * 0.85)
            warnings.warn("auto ROI detection out of range; using fallback y2", stacklevel=2)

    y2 = min(h, max(y1 + 1, y2))
    return (x1, y1, x2, y2)


def make_state_image(frame_bgr: np.ndarray, roi: Tuple[int, int, int, int], size: int = 256) -> np.ndarray:
    x1, y1, x2, y2 = roi
    cropped = frame_bgr[y1:y2, x1:x2]
    if cropped.size == 0:
        raise ValueError("ROI crop is empty")
    return cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
