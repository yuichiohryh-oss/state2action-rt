import json

import numpy as np

from state2action_rt.dataset_core import GridConfig, build_dataset
from state2action_rt.frame_source import FrameSource
from state2action_rt.roi import RoiConfig


def test_build_dataset_single_record(tmp_path):
    frame = np.full((100, 100, 3), 200, dtype=np.uint8)

    def get_frame(_index: int):
        return frame

    frame_source = FrameSource(fps=10.0, get_frame=get_frame)
    events = [{"t": 1.0, "x": 75.0, "y": 75.0, "action_id": "a1"}]

    out_dir = tmp_path / "out"
    records = build_dataset(
        events,
        frame_source,
        str(out_dir),
        RoiConfig(y1=0, y2_mode="fixed", y2_fixed=100, x1=0, x2=100),
        GridConfig(gw=2, gh=2),
        0.5,
        lambda _msg: None,
    )

    assert len(records) == 1
    assert records[0]["grid_id"] == 3
    assert records[0]["x_rel"] == 0.75
    assert records[0]["y_rel"] == 0.75

    dataset_path = out_dir / "dataset.jsonl"
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]
    assert len(data) == 1
    assert data[0]["x"] == 75.0
    assert data[0]["y"] == 75.0
    assert data[0]["meta"]["fps_effective"] == 10.0
    assert (out_dir / data[0]["state_path"]).exists()
