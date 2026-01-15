import cv2
import numpy as np

from state2action_rt.hand_cards import HandCardTemplate, match_hand_cards


def test_match_hand_cards_matches_template() -> None:
    template = np.zeros((64, 64), dtype=np.uint8)
    cv2.rectangle(template, (16, 16), (48, 48), 255, -1)
    slot_bgr = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
    templates = [HandCardTemplate(card_id=3, image_gray=template)]

    card_ids = match_hand_cards([slot_bgr], templates, min_score=0.5, template_size=(64, 64))

    assert card_ids == [3]


def test_match_hand_cards_handles_empty_templates() -> None:
    slot_bgr = np.zeros((32, 32, 3), dtype=np.uint8)

    card_ids = match_hand_cards([slot_bgr, slot_bgr], [], min_score=0.5, template_size=(64, 64))

    assert card_ids == [-1, -1]
