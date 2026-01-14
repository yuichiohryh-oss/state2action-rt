import json
import sys

import cv2
import numpy as np

from tools import train_policy


def write_dummy_image(path: str, value: int) -> None:
    img = np.full((256, 256, 3), value, dtype=np.uint8)
    ok = cv2.imwrite(path, img)
    assert ok


def test_train_saves_last_checkpoint(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    frames_dir = data_dir / "state_frames"
    frames_dir.mkdir(parents=True)

    write_dummy_image(str(frames_dir / "000000.png"), 10)
    write_dummy_image(str(frames_dir / "000001.png"), 20)

    records = [
        {
            "idx": 0,
            "action_id": "action_1",
            "grid_id": 0,
            "state_path": "state_frames/000000.png",
            "meta": {"gw": 6, "gh": 9},
        },
        {
            "idx": 1,
            "action_id": "action_2",
            "grid_id": 1,
            "state_path": "state_frames/000001.png",
            "meta": {"gw": 6, "gh": 9},
        },
    ]
    dataset_path = data_dir / "dataset.jsonl"
    data_dir.mkdir(parents=True, exist_ok=True)
    with open(dataset_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    out_dir = tmp_path / "out"
    args = [
        "train_policy.py",
        "--data-dir",
        str(data_dir),
        "--out-dir",
        str(out_dir),
        "--epochs",
        "1",
        "--batch-size",
        "1",
        "--lr",
        "1e-3",
        "--seed",
        "7",
        "--val-ratio",
        "0.6",
        "--gw",
        "6",
        "--gh",
        "9",
        "--device",
        "cpu",
    ]
    monkeypatch.setattr(sys, "argv", args)

    rc = train_policy.main()

    assert rc == 0
    assert (out_dir / "checkpoints" / "last.pt").exists()
