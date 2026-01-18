from __future__ import annotations

from typing import List, Sequence, Tuple, TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from .hand_cards import HandCardTemplate

HandRoiPixels = Tuple[int | None, int | None, int | None, int | None]


def _normalize_template_size(template_size: int | Tuple[int, int]) -> Tuple[int, int]:
    if isinstance(template_size, int):
        return (template_size, template_size)
    if len(template_size) != 2:
        raise ValueError("template_size must be int or (w, h)")
    return (int(template_size[0]), int(template_size[1]))


def compute_hand_roi_rect(
    frame_bgr: np.ndarray,
    y1_ratio: float = 0.90,
    y2_ratio: float = 0.97,
    x_margin_ratio: float = 0.03,
    hand_roi_pixels: HandRoiPixels | None = None,
) -> Tuple[int, int, int, int]:
    h, w = frame_bgr.shape[:2]
    if hand_roi_pixels is None:
        hand_roi_pixels = (None, None, None, None)
    x1_px, x2_px, y1_px, y2_px = hand_roi_pixels
    if any(value is not None for value in hand_roi_pixels):
        if not all(value is not None for value in hand_roi_pixels):
            raise ValueError("hand ROI override requires x1/x2/y1/y2")
        x1 = int(x1_px)
        x2 = int(x2_px)
        y1 = int(y1_px)
        y2 = int(y2_px)
    elif w == 330 and h == 752:
        x1, x2, y1, y2 = 75, 318, 630, 680
    else:
        x_margin = int(w * x_margin_ratio)
        x1 = x_margin
        x2 = w - x_margin
        y1 = int(h * y1_ratio)
        y2 = int(h * y2_ratio)
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("hand ROI is empty")
    return x1, y1, x2, y2


def compute_hand_roi(
    frame_bgr: np.ndarray,
    y1_ratio: float = 0.90,
    y2_ratio: float = 0.97,
    x_margin_ratio: float = 0.03,
    hand_roi_pixels: HandRoiPixels | None = None,
) -> np.ndarray:
    x1, y1, x2, y2 = compute_hand_roi_rect(
        frame_bgr,
        y1_ratio=y1_ratio,
        y2_ratio=y2_ratio,
        x_margin_ratio=x_margin_ratio,
        hand_roi_pixels=hand_roi_pixels,
    )
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


def crop_match_roi(
    slot_bgr: np.ndarray,
    side_cut_ratio: float = 0.18,
    bottom_cut_ratio: float = 0.45,
) -> np.ndarray:
    h, w = slot_bgr.shape[:2]
    if h < 1 or w < 1:
        return slot_bgr
    x_margin = int(round(w * side_cut_ratio))
    y_cut = int(round(h * bottom_cut_ratio))
    x1 = max(0, x_margin)
    x2 = min(w, w - x_margin)
    y2 = min(h, h - y_cut)
    if x2 <= x1 or y2 <= 0:
        return slot_bgr
    match_roi = slot_bgr[:y2, x1:x2]
    if match_roi.size == 0:
        return slot_bgr
    return match_roi


def mean_saturation(slot_bgr: np.ndarray) -> float:
    if slot_bgr.size == 0:
        raise ValueError("slot image is empty")
    hsv = cv2.cvtColor(slot_bgr, cv2.COLOR_BGR2HSV)
    mean_s = float(hsv[:, :, 1].mean())
    return mean_s


def load_hand_templates(
    templates_dir: str,
    template_size: int | Tuple[int, int],
) -> List["HandCardTemplate"]:
    from .hand_cards import load_hand_card_templates

    size = _normalize_template_size(template_size)
    return load_hand_card_templates(templates_dir, template_size=size)


def _prepare_slot_gray(slot_bgr: np.ndarray, template_size: Tuple[int, int]) -> np.ndarray:
    if slot_bgr.size == 0:
        raise ValueError("slot image is empty")
    slot_gray = cv2.cvtColor(slot_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(slot_gray, template_size, interpolation=cv2.INTER_AREA)


def _match_slot_to_templates(
    slot_bgr: np.ndarray,
    templates: Sequence["HandCardTemplate"],
    template_size: Tuple[int, int],
    min_score: float,
) -> Tuple[int, float]:
    if not templates:
        return -1, -1.0
    slot_gray = _prepare_slot_gray(slot_bgr, template_size)
    best_score = float("-inf")
    best_id = -1
    for template in templates:
        score = cv2.matchTemplate(
            slot_gray, template.image_gray, cv2.TM_CCOEFF_NORMED
        )[0][0]
        if score > best_score:
            best_score = score
            best_id = template.card_id
    if best_score < min_score:
        return -1, float(best_score)
    return best_id, float(best_score)


def hand_state_from_frame(
    frame_bgr: np.ndarray,
    templates: Sequence["HandCardTemplate"] | None,
    s_th: float,
    min_score: float,
    y1_ratio: float = 0.90,
    y2_ratio: float = 0.97,
    x_margin_ratio: float = 0.03,
    n_slots: int = 4,
    template_size: int | Tuple[int, int] = 64,
    hand_roi_pixels: HandRoiPixels | None = None,
    side_cut_ratio: float = 0.10,
    bottom_cut_ratio: float = 0.25,
) -> dict:
    if templates is None:
        templates = []
    size = _normalize_template_size(template_size)
    x1, y1, x2, y2 = compute_hand_roi_rect(
        frame_bgr,
        y1_ratio=y1_ratio,
        y2_ratio=y2_ratio,
        x_margin_ratio=x_margin_ratio,
        hand_roi_pixels=hand_roi_pixels,
    )
    hand_roi = frame_bgr[y1:y2, x1:x2]
    slot_rois = split_hand_slots(hand_roi, n_slots=n_slots)
    match_rois: List[np.ndarray] = []
    mean_s_list: List[float] = []
    available: List[int] = []
    card_ids: List[int] = []
    scores: List[float] = []
    for slot in slot_rois:
        match_roi = crop_match_roi(
            slot, side_cut_ratio=side_cut_ratio, bottom_cut_ratio=bottom_cut_ratio
        )
        match_rois.append(match_roi)
        mean_s = mean_saturation(match_roi)
        mean_s_list.append(mean_s)
        is_available = mean_s > s_th
        available.append(1 if is_available else 0)
        if is_available:
            card_id, score = _match_slot_to_templates(
                match_roi,
                templates,
                template_size=size,
                min_score=min_score,
            )
        else:
            card_id, score = -1, -1.0
        card_ids.append(card_id)
        scores.append(score)
    return {
        "hand_roi": hand_roi,
        "slot_rois": slot_rois,
        "match_rois": match_rois,
        "mean_s": mean_s_list,
        "available": available,
        "card_ids": card_ids,
        "scores": scores,
    }


def hand_available_from_frame(
    frame_bgr: np.ndarray,
    s_th: float = 30.0,
    y1_ratio: float = 0.90,
    y2_ratio: float = 0.97,
    x_margin_ratio: float = 0.03,
    n_slots: int = 4,
    hand_roi_pixels: HandRoiPixels | None = None,
) -> Tuple[List[float], List[int], np.ndarray, List[np.ndarray]]:
    state = hand_state_from_frame(
        frame_bgr,
        templates=[],
        s_th=s_th,
        min_score=0.0,
        y1_ratio=y1_ratio,
        y2_ratio=y2_ratio,
        x_margin_ratio=x_margin_ratio,
        n_slots=n_slots,
        hand_roi_pixels=hand_roi_pixels,
    )
    return state["mean_s"], state["available"], state["hand_roi"], state["slot_rois"]
