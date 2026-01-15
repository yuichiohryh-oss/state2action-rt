import numpy as np

from state2action_rt.hand_features import compute_hand_roi, mean_saturation, split_hand_slots


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
