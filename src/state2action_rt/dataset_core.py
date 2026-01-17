from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import cv2

from .elixir_features import ElixirRoiPixels, estimate_elixir_from_frame
from .frame_source import FrameSource
from .grid import xy_to_grid_id
from .hand_cards import HandCardTemplate
from .hand_features import HandRoiPixels, hand_available_from_frame, hand_state_from_frame, load_hand_templates
from .roi import RoiConfig, detect_roi, make_state_image


@dataclass
class GridConfig:
    gw: int = 6
    gh: int = 9


def fetch_frame_with_fallback(
    frame_source: FrameSource,
    index: int,
    max_offset: int = 2,
) -> Optional[np.ndarray]:
    offsets = [0]
    for i in range(1, max_offset + 1):
        offsets.extend([-i, i])
    for offset in offsets:
        frame = frame_source.get_frame(index + offset)
        if frame is not None:
            return frame
    return None


def build_dataset(
    events: Iterable[Dict],
    frame_source: FrameSource,
    out_dir: str,
    roi_config: RoiConfig,
    grid_config: GridConfig,
    lead_sec: float,
    warn_fn: Callable[[str], None],
    hand_s_th: float = 30.0,
    hand_y1_ratio: float = 0.90,
    hand_y2_ratio: float = 0.97,
    hand_x_margin_ratio: float = 0.03,
    hand_templates_dir: str | None = None,
    hand_card_min_score: float = 0.6,
    hand_template_size: Tuple[int, int] = (64, 64),
    hand_roi_pixels: HandRoiPixels | None = None,
    elixir_roi_pixels: ElixirRoiPixels | None = None,
    elixir_purple_h_min: int = 120,
    elixir_purple_h_max: int = 170,
    elixir_purple_s_min: int = 60,
    elixir_purple_v_min: int = 40,
    elixir_col_fill_ratio_th: float = 0.35,
    elixir_min_purple_ratio: float = 0.01,
    elixir_max_holes_ratio: float = 0.6,
    elixir_allow_empty: bool = False,
    elixir_empty_purple_ratio_max: float = 0.002,
    elixir_empty_mean_s_max: float = 80.0,
) -> List[Dict]:
    os.makedirs(out_dir, exist_ok=True)
    frames_dir = os.path.join(out_dir, "state_frames")
    os.makedirs(frames_dir, exist_ok=True)

    templates: List[HandCardTemplate] = []
    if hand_templates_dir:
        templates = load_hand_templates(hand_templates_dir, hand_template_size)
        if not templates:
            warn_fn(f"no hand templates loaded from {hand_templates_dir}")

    records: List[Dict] = []
    for idx, event in enumerate(events):
        t_action = float(event["t"])
        t_state = t_action - lead_sec
        if t_state < 0:
            warn_fn(f"event {idx}: t_state < 0, skipping")
            continue

        frame_index = int(round(t_state * frame_source.fps))
        frame = fetch_frame_with_fallback(frame_source, frame_index)
        if frame is None:
            warn_fn(f"event {idx}: failed to load frame near {frame_index}")
            continue

        roi = detect_roi(frame, roi_config)
        event_x = float(event["x"])
        event_y = float(event["y"])
        grid_id = xy_to_grid_id(event_x, event_y, roi, grid_config.gw, grid_config.gh)
        roi_w = max(1.0, float(roi[2] - roi[0]))
        roi_h = max(1.0, float(roi[3] - roi[1]))
        x_rel = (event_x - roi[0]) / roi_w
        y_rel = (event_y - roi[1]) / roi_h
        x_rel = min(max(x_rel, 0.0), 1.0)
        y_rel = min(max(y_rel, 0.0), 1.0)
        state_img = make_state_image(frame, roi)

        state_rel = os.path.join("state_frames", f"{idx:06d}.png")
        state_abs = os.path.join(out_dir, state_rel)
        success = cv2.imwrite(state_abs, state_img)
        if not success:
            warn_fn(f"event {idx}: failed to write state frame")
            continue

        record = {
            "idx": idx,
            "t_action": t_action,
            "t_state": t_state,
            "action_id": event["action_id"],
            "grid_id": grid_id,
            "x": event_x,
            "y": event_y,
            "x_rel": x_rel,
            "y_rel": y_rel,
            "roi": [int(v) for v in roi],
            "state_path": state_rel.replace("\\", "/"),
            "meta": {
                "gw": grid_config.gw,
                "gh": grid_config.gh,
                "lead_sec": lead_sec,
                "fps_effective": float(frame_source.fps),
            },
        }
        record = with_hand_features(
            record,
            frame,
            templates=templates,
            s_th=hand_s_th,
            min_score=hand_card_min_score,
            y1_ratio=hand_y1_ratio,
            y2_ratio=hand_y2_ratio,
            x_margin_ratio=hand_x_margin_ratio,
            n_slots=4,
            template_size=hand_template_size,
            hand_roi_pixels=hand_roi_pixels,
            warn_fn=warn_fn,
        )
        record = with_elixir_features(
            record,
            frame,
            elixir_roi_pixels=elixir_roi_pixels,
            purple_h_min=elixir_purple_h_min,
            purple_h_max=elixir_purple_h_max,
            purple_s_min=elixir_purple_s_min,
            purple_v_min=elixir_purple_v_min,
            col_fill_ratio_th=elixir_col_fill_ratio_th,
            min_purple_ratio=elixir_min_purple_ratio,
            max_holes_ratio=elixir_max_holes_ratio,
            allow_empty=elixir_allow_empty,
            empty_purple_ratio_max=elixir_empty_purple_ratio_max,
            empty_mean_s_max=elixir_empty_mean_s_max,
            warn_fn=warn_fn,
        )
        records.append(record)

    dataset_path = os.path.join(out_dir, "dataset.jsonl")
    with open(dataset_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    return records


def with_hand_available(
    record: Dict,
    frame_bgr: np.ndarray,
    s_th: float = 30.0,
    y1_ratio: float = 0.90,
    y2_ratio: float = 0.97,
    x_margin_ratio: float = 0.03,
) -> Dict:
    _, avail_list, _, _ = hand_available_from_frame(
        frame_bgr,
        s_th=s_th,
        y1_ratio=y1_ratio,
        y2_ratio=y2_ratio,
        x_margin_ratio=x_margin_ratio,
        n_slots=4,
    )
    return {**record, "hand_available": [int(v) for v in avail_list]}


def with_hand_features(
    record: Dict,
    frame_bgr: np.ndarray,
    templates: List[HandCardTemplate],
    min_score: float,
    s_th: float,
    y1_ratio: float,
    y2_ratio: float,
    x_margin_ratio: float,
    n_slots: int,
    template_size: Tuple[int, int],
    hand_roi_pixels: HandRoiPixels | None,
    warn_fn: Callable[[str], None],
) -> Dict:
    try:
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
    except ValueError as exc:
        warn_fn(f"hand_features failed: {exc}")
        return {
            **record,
            "hand_available": [1 for _ in range(n_slots)],
            "hand_card_ids": [-1 for _ in range(n_slots)],
            "hand_scores": [0.0 for _ in range(n_slots)],
        }

    available = [int(v) for v in state["available"]]
    if not templates:
        card_ids = [-1 for _ in range(n_slots)]
        scores = [0.0 for _ in range(n_slots)]
    else:
        card_ids = [int(v) for v in state["card_ids"]]
        scores = [float(v) for v in state["scores"]]
        for i, is_available in enumerate(available):
            if not is_available:
                card_ids[i] = -1
                scores[i] = 0.0

    return {
        **record,
        "hand_available": available,
        "hand_card_ids": card_ids,
        "hand_scores": scores,
    }


def with_elixir_features(
    record: Dict,
    frame_bgr: np.ndarray,
    elixir_roi_pixels: ElixirRoiPixels | None,
    purple_h_min: int,
    purple_h_max: int,
    purple_s_min: int,
    purple_v_min: int,
    col_fill_ratio_th: float,
    min_purple_ratio: float,
    max_holes_ratio: float,
    allow_empty: bool,
    empty_purple_ratio_max: float,
    empty_mean_s_max: float,
    warn_fn: Callable[[str], None],
) -> Dict:
    try:
        metrics = estimate_elixir_from_frame(
            frame_bgr,
            elixir_roi_pixels=elixir_roi_pixels,
            purple_h_min=purple_h_min,
            purple_h_max=purple_h_max,
            purple_s_min=purple_s_min,
            purple_v_min=purple_v_min,
            col_fill_ratio_th=col_fill_ratio_th,
            min_purple_ratio=min_purple_ratio,
            max_holes_ratio=max_holes_ratio,
            allow_empty=allow_empty,
            empty_purple_ratio_max=empty_purple_ratio_max,
            empty_mean_s_max=empty_mean_s_max,
        )
    except ValueError as exc:
        warn_fn(f"elixir estimation failed: {exc}")
        return {**record, "elixir": -1, "elixir_frac": -1.0}

    return {
        **record,
        "elixir": int(metrics["elixir"]),
        "elixir_frac": float(metrics["elixir_frac"]),
    }
