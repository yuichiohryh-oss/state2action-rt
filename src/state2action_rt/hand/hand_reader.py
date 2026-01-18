from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from ..hand_cards import HandCardTemplate
from ..hand_features import HandRoiPixels, compute_hand_roi_rect, hand_state_from_frame


@dataclass(frozen=True)
class HandDetectConfig:
    s_th: float = 60.0
    card_min_score: float = 0.65
    y1_ratio: float = 0.90
    y2_ratio: float = 0.97
    x_margin_ratio: float = 0.03
    n_slots: int = 4
    template_size: int | Tuple[int, int] = 64
    hand_roi_pixels: HandRoiPixels | None = None
    templates: Sequence[HandCardTemplate] | None = None
    side_cut_ratio: float = 0.10
    bottom_cut_ratio: float = 0.25


def resolve_hand_roi_pixels(hand_roi_pixels: HandRoiPixels | None) -> HandRoiPixels | None:
    if hand_roi_pixels is None:
        return None
    if all(value is None for value in hand_roi_pixels):
        return None
    if not all(value is not None for value in hand_roi_pixels):
        raise ValueError("hand ROI override requires x1/x2/y1/y2")
    return hand_roi_pixels


def detect_hand_state(
    frame_bgr: np.ndarray, cfg: HandDetectConfig
) -> Tuple[dict, Tuple[int, int, int, int]]:
    roi_pixels = resolve_hand_roi_pixels(cfg.hand_roi_pixels)
    roi_px = compute_hand_roi_rect(
        frame_bgr,
        y1_ratio=cfg.y1_ratio,
        y2_ratio=cfg.y2_ratio,
        x_margin_ratio=cfg.x_margin_ratio,
        hand_roi_pixels=roi_pixels,
    )
    state = hand_state_from_frame(
        frame_bgr,
        templates=list(cfg.templates or []),
        s_th=cfg.s_th,
        min_score=cfg.card_min_score,
        y1_ratio=cfg.y1_ratio,
        y2_ratio=cfg.y2_ratio,
        x_margin_ratio=cfg.x_margin_ratio,
        n_slots=cfg.n_slots,
        template_size=cfg.template_size,
        hand_roi_pixels=roi_pixels,
        side_cut_ratio=cfg.side_cut_ratio,
        bottom_cut_ratio=cfg.bottom_cut_ratio,
    )
    return state, roi_px


def summarize_hand_state(
    state: dict, cfg: HandDetectConfig
) -> Tuple[List[int], List[int], List[float]]:
    available = [int(v) for v in state["available"]]
    card_ids = [int(v) for v in state["card_ids"]]
    scores = [float(v) for v in state["scores"]]

    if not cfg.templates:
        card_ids = [-1 for _ in range(cfg.n_slots)]
        scores = [-1.0 for _ in range(cfg.n_slots)]
        return available, card_ids, scores

    for i, is_available in enumerate(available):
        if not is_available:
            scores[i] = -1.0
        if not is_available or scores[i] < cfg.card_min_score:
            card_ids[i] = -1

    return available, card_ids, scores


def detect_hand(
    frame_bgr: np.ndarray, cfg: HandDetectConfig
) -> Tuple[List[int], List[int], List[float], Tuple[int, int, int, int]]:
    state, roi_px = detect_hand_state(frame_bgr, cfg)
    available, card_ids, scores = summarize_hand_state(state, cfg)
    return available, card_ids, scores, roi_px
