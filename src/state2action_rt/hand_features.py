from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np


def compute_hand_roi(
    frame_bgr: np.ndarray,
    y1_ratio: float = 0.90,
    y2_ratio: float = 0.97,
    x_margin_ratio: float = 0.03,
) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    y1 = int(h * y1_ratio)
    y2 = int(h * y2_ratio)
    x1 = int(w * 0.0)
    x2 = int(w * 1.0)
    x_margin = int(w * x_margin_ratio)
    x1 = max(0, x1 + x_margin)
    x2 = min(w, x2 - x_margin)
    y1 = max(0, y1)
    y2 = min(h, y2)
    if y2 <= y1 or x2 <= x1:
        raise ValueError("hand ROI is empty")
    return frame_bgr[y1:y2, x1:x2]


def split_hand_slots(hand_bgr: np.ndarray, n_slots: int = 4) -> List[np.ndarray]:
    if n_slots < 1:
        raise ValueError("n_slots must be >= 1")
    h, w = hand_bgr.shape[:2]
    if w < 1 or h < 1:
        raise ValueError("hand image is empty")
    slots: List[np.ndarray] = []
    for i in range(n_slots):
        x1 = int(round(i * w / n_slots))
        x2 = int(round((i + 1) * w / n_slots))
        if x2 <= x1:
            x2 = min(w, x1 + 1)
        slots.append(hand_bgr[:, x1:x2])
    return slots


def mean_saturation(slot_bgr: np.ndarray) -> float:
    if slot_bgr.size == 0:
        raise ValueError("slot image is empty")
    hsv = cv2.cvtColor(slot_bgr, cv2.COLOR_BGR2HSV)
    mean_s = float(hsv[:, :, 1].mean())
    return mean_s


def hand_available_from_frame(
    frame_bgr: np.ndarray,
    s_th: float = 30.0,
    y1_ratio: float = 0.90,
    y2_ratio: float = 0.97,
    x_margin_ratio: float = 0.03,
    n_slots: int = 4,
) -> Tuple[List[float], List[int], np.ndarray, List[np.ndarray]]:
    hand_bgr = compute_hand_roi(
        frame_bgr,
        y1_ratio=y1_ratio,
        y2_ratio=y2_ratio,
        x_margin_ratio=x_margin_ratio,
    )
    slots = split_hand_slots(hand_bgr, n_slots=n_slots)
    mean_s_list = [mean_saturation(slot) for slot in slots]
    avail_list = [1 if mean_s > s_th else 0 for mean_s in mean_s_list]
    return mean_s_list, avail_list, hand_bgr, slots
