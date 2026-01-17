import numpy as np

from state2action_rt.elixir_features import (
    DEFAULT_ELIXIR_ROI_330x752,
    compute_elixir_roi_rect,
    estimate_elixir_from_frame,
)


def make_frame(width: int = 330, height: int = 752) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def draw_elixir_fill(frame: np.ndarray, fill_ratio: float) -> None:
    x1, y1, x2, y2 = compute_elixir_roi_rect(frame)
    width = x2 - x1
    fill_width = int(round(width * fill_ratio))
    if fill_width <= 0:
        return
    frame[y1:y2, x1 : x1 + fill_width] = (255, 0, 255)


def test_compute_elixir_roi_rect_fixed_scrcpy() -> None:
    frame = make_frame()
    assert compute_elixir_roi_rect(frame) == DEFAULT_ELIXIR_ROI_330x752


def test_estimate_elixir_half_fill() -> None:
    frame = make_frame()
    draw_elixir_fill(frame, 0.5)
    metrics = estimate_elixir_from_frame(frame)
    assert metrics["elixir"] == 5
    assert metrics["elixir_frac"] == 5.0


def test_estimate_elixir_full_fill() -> None:
    frame = make_frame()
    draw_elixir_fill(frame, 1.0)
    metrics = estimate_elixir_from_frame(frame)
    assert metrics["elixir"] == 10
    assert metrics["elixir_frac"] == 10.0


def test_estimate_elixir_empty_allow() -> None:
    frame = make_frame()
    metrics = estimate_elixir_from_frame(frame, allow_empty=True)
    assert metrics["elixir"] == 0
    assert metrics["elixir_frac"] == 0.0
