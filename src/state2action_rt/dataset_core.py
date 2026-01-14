from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

import cv2

from .frame_source import FrameSource
from .grid import xy_to_grid_id
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
) -> List[Dict]:
    os.makedirs(out_dir, exist_ok=True)
    frames_dir = os.path.join(out_dir, "state_frames")
    os.makedirs(frames_dir, exist_ok=True)

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
        grid_id = xy_to_grid_id(event["x"], event["y"], roi, grid_config.gw, grid_config.gh)
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
            "roi": [int(v) for v in roi],
            "state_path": state_rel.replace("\\", "/"),
            "meta": {
                "gw": grid_config.gw,
                "gh": grid_config.gh,
                "lead_sec": lead_sec,
            },
        }
        records.append(record)

    dataset_path = os.path.join(out_dir, "dataset.jsonl")
    with open(dataset_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    return records
