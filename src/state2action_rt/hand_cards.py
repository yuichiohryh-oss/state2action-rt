from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np

from .hand_features import hand_state_from_frame

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


@dataclass(frozen=True)
class HandCardTemplate:
    card_id: int
    image_gray: np.ndarray


def parse_card_id(text: str) -> Optional[int]:
    match = re.search(r"\d+", text)
    if not match:
        return None
    return int(match.group(0))


def load_hand_card_templates(
    templates_dir: str,
    template_size: Tuple[int, int] = (64, 64),
) -> List[HandCardTemplate]:
    if not os.path.isdir(templates_dir):
        return []
    templates: List[HandCardTemplate] = []
    for name in sorted(os.listdir(templates_dir)):
        _, ext = os.path.splitext(name)
        if ext.lower() not in IMAGE_EXTS:
            continue
        card_id = parse_card_id(name)
        if card_id is None:
            continue
        path = os.path.join(templates_dir, name)
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        if template_size:
            image = cv2.resize(image, template_size, interpolation=cv2.INTER_AREA)
        templates.append(HandCardTemplate(card_id=card_id, image_gray=image))
    return templates


def _prepare_slot_gray(slot_bgr: np.ndarray, template_size: Tuple[int, int]) -> np.ndarray:
    if slot_bgr.size == 0:
        raise ValueError("slot image is empty")
    slot_gray = cv2.cvtColor(slot_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(slot_gray, template_size, interpolation=cv2.INTER_AREA)


def match_hand_cards(
    slots_bgr: Iterable[np.ndarray],
    templates: List[HandCardTemplate],
    min_score: float,
    template_size: Tuple[int, int] = (64, 64),
) -> List[int]:
    card_ids: List[int] = []
    if not templates:
        return [-1 for _ in slots_bgr]
    for slot in slots_bgr:
        slot_gray = _prepare_slot_gray(slot, template_size)
        best_score = float("-inf")
        best_id = -1
        for template in templates:
            score = cv2.matchTemplate(slot_gray, template.image_gray, cv2.TM_CCOEFF_NORMED)[0][0]
            if score > best_score:
                best_score = score
                best_id = template.card_id
        if best_score < min_score:
            card_ids.append(-1)
        else:
            card_ids.append(best_id)
    return card_ids


def infer_hand_card_ids_from_frame(
    frame_bgr: np.ndarray,
    templates: List[HandCardTemplate],
    min_score: float,
    y1_ratio: float = 0.90,
    y2_ratio: float = 0.97,
    x_margin_ratio: float = 0.03,
    n_slots: int = 4,
    template_size: Tuple[int, int] = (64, 64),
    s_th: float = 30.0,
    hand_roi_pixels: Tuple[int | None, int | None, int | None, int | None] | None = None,
) -> List[int]:
    state = hand_state_from_frame(
        frame_bgr,
        templates=templates,
        s_th=s_th,
        min_score=min_score,
        y1_ratio=y1_ratio,
        y2_ratio=y2_ratio,
        x_margin_ratio=x_margin_ratio,
        n_slots=n_slots,
        template_size=template_size,
        hand_roi_pixels=hand_roi_pixels,
    )
    return [int(card_id) for card_id in state["card_ids"]]
