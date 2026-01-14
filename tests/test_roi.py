import numpy as np

from state2action_rt.roi import RoiConfig, detect_roi


def test_detect_roi_bottom_bar():
    h, w = 480, 640
    frame = np.full((h, w, 3), 120, dtype=np.uint8)
    frame[420:, :, :] = (255, 0, 0)

    roi = detect_roi(frame, RoiConfig(y1=70, y2_mode="auto"))
    assert 417 <= roi[3] <= 423
