import json
import subprocess
import sys

import cv2
import numpy as np


def write_dummy_image(path: str) -> None:
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    ok = cv2.imwrite(path, img)
    assert ok


def test_augment_noop_min_gap(tmp_path) -> None:
    data_dir = tmp_path / "data"
    frames_dir = data_dir / "state_frames"
    frames_dir.mkdir(parents=True)

    write_dummy_image(str(frames_dir / "000000.png"))
    write_dummy_image(str(frames_dir / "000001.png"))

    records = [
        {
            "idx": 0,
            "t_action": 1.0,
            "t_state": 0.5,
            "action_id": "a_action",
            "grid_id": 3,
            "state_path": "state_frames/000000.png",
            "roi": [0, 0, 10, 10],
            "meta": {"gw": 6, "gh": 9},
        },
        {
            "idx": 1,
            "t_action": 10.0,
            "t_state": 9.0,
            "action_id": "b_action",
            "grid_id": 4,
            "state_path": "state_frames/000001.png",
            "roi": [0, 0, 10, 10],
            "meta": {"gw": 6, "gh": 9},
        },
    ]
    dataset_path = data_dir / "dataset.jsonl"
    with open(dataset_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    out_dir = tmp_path / "noop_out"
    result = subprocess.run(
        [
            sys.executable,
            "tools/augment_noop.py",
            "--data-dir",
            str(data_dir),
            "--out-dir",
            str(out_dir),
            "--noop-per-action",
            "1",
            "--min-gap-sec",
            "0.6",
            "--span-sec",
            "0.0",
            "--seed",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "created=1" in result.stdout

    output_path = out_dir / "dataset_with_noop.jsonl"
    assert output_path.exists()
    with open(output_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    noop_records = [record for record in data if record.get("action_id") == "__NOOP__"]
    assert len(noop_records) == 1
    noop = noop_records[0]
    assert noop["t_state"] == 9.0
    assert noop["grid_id"] == -1
    assert noop["x"] is None
    assert noop["y"] is None
