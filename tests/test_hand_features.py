import numpy as np

from state2action_rt.hand_features import (
    compute_hand_roi_rect,
    compute_hand_roi,
    hand_state_from_frame,
    hand_available_from_frame,
    mean_saturation,
    split_hand_slots,
)


def test_compute_hand_roi_shape():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    hand = compute_hand_roi(frame)
    assert hand.shape[0] == 7
    assert hand.shape[1] == 188


def test_split_hand_slots_count():
    hand = np.zeros((10, 40, 3), dtype=np.uint8)
    slots = split_hand_slots(hand, n_slots=4)
    assert len(slots) == 4
    assert all(slot.shape[1] == 10 for slot in slots)


def test_mean_saturation_difference():
    gray = np.full((8, 8, 3), 128, dtype=np.uint8)
    red = np.zeros((8, 8, 3), dtype=np.uint8)
    red[:, :, 2] = 255
    mean_gray = mean_saturation(gray)
    mean_red = mean_saturation(red)
    assert mean_red > mean_gray


def test_hand_available_from_frame_binary_slots():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    y1 = int(100 * 0.90)
    y2 = int(100 * 1.00)
    slot_w = 100 // 4
    frame[y1:y2, 0:slot_w] = (0, 0, 255)
    frame[y1:y2, slot_w : 2 * slot_w] = (0, 0, 255)
    frame[y1:y2, 2 * slot_w : 3 * slot_w] = (128, 128, 128)
    frame[y1:y2, 3 * slot_w : 4 * slot_w] = (128, 128, 128)

    _, avail_list, _, _ = hand_available_from_frame(
        frame,
        s_th=30.0,
        y1_ratio=0.90,
        y2_ratio=1.0,
        x_margin_ratio=0.0,
        n_slots=4,
    )

    assert avail_list == [1, 1, 0, 0]


def test_compute_hand_roi_rect_fixed_scrcpy():
    frame = np.zeros((752, 330, 3), dtype=np.uint8)
    x1, y1, x2, y2 = compute_hand_roi_rect(frame)
    assert (x1, y1, x2, y2) == (75, 630, 318, 680)


def test_hand_state_from_frame_empty_templates():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    y1 = int(100 * 0.90)
    y2 = int(100 * 0.97)
    frame[y1:y2, :] = (0, 0, 255)

    state = hand_state_from_frame(
        frame,
        templates=[],
        s_th=30.0,
        min_score=0.6,
        y1_ratio=0.90,
        y2_ratio=0.97,
        x_margin_ratio=0.0,
        n_slots=4,
    )

    assert state["available"] == [1, 1, 1, 1]
    assert state["card_ids"] == [-1, -1, -1, -1]
