import json
import os

import cv2
import numpy as np
import torch

from state2action_rt.learning.dataset import ActionVocab, StateActionDataset, load_or_create_vocab, load_records


def write_dummy_image(path: str, value: int) -> None:
    img = np.full((256, 256, 3), value, dtype=np.uint8)
    ok = cv2.imwrite(path, img)
    assert ok


def test_dataset_loads_and_builds_vocab(tmp_path) -> None:
    data_dir = tmp_path / "out"
    frames_dir = data_dir / "state_frames"
    frames_dir.mkdir(parents=True)

    write_dummy_image(str(frames_dir / "000000.png"), 10)
    write_dummy_image(str(frames_dir / "000001.png"), 20)

    records = [
        {
            "idx": 0,
            "action_id": "b_action",
            "grid_id": 3,
            "state_path": "state_frames/000000.png",
            "meta": {"gw": 6, "gh": 9},
        },
        {
            "idx": 1,
            "action_id": "a_action",
            "grid_id": 4,
            "state_path": "state_frames/000001.png",
            "meta": {"gw": 6, "gh": 9},
        },
    ]
    dataset_path = data_dir / "dataset.jsonl"
    data_dir.mkdir(parents=True, exist_ok=True)
    with open(dataset_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    loaded = load_records(str(data_dir))
    vocab = load_or_create_vocab(str(data_dir), loaded)
    assert (data_dir / "vocab.json").exists()

    dataset = StateActionDataset(str(data_dir), loaded, vocab)
    image, card_label, grid_label = dataset[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 256, 256)
    assert image.dtype == torch.float32
    assert grid_label == 3
    assert vocab.id_to_action == ["a_action", "b_action"]
    assert card_label == vocab.action_to_id["b_action"]

    reload_vocab = ActionVocab.load(os.path.join(str(data_dir), "vocab.json"))
    assert reload_vocab.id_to_action == vocab.id_to_action


def test_dataset_two_frame_concat_and_fallback(tmp_path) -> None:
    data_dir = tmp_path / "out"
    frames_dir = data_dir / "state_frames"
    frames_dir.mkdir(parents=True)

    write_dummy_image(str(frames_dir / "000000.png"), 10)
    write_dummy_image(str(frames_dir / "000001.png"), 20)

    records = [
        {
            "idx": 0,
            "action_id": "b_action",
            "grid_id": 3,
            "state_path": "state_frames/000000.png",
            "meta": {"gw": 6, "gh": 9, "fps_effective": 10.0},
        },
        {
            "idx": 1,
            "action_id": "a_action",
            "grid_id": 4,
            "state_path": "state_frames/000001.png",
            "meta": {"gw": 6, "gh": 9, "fps_effective": 10.0},
        },
    ]
    dataset_path = data_dir / "dataset.jsonl"
    data_dir.mkdir(parents=True, exist_ok=True)
    with open(dataset_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    loaded = load_records(str(data_dir))
    vocab = load_or_create_vocab(str(data_dir), loaded)
    dataset = StateActionDataset(str(data_dir), loaded, vocab, two_frame=True, delta_frames=1)

    image, _, _ = dataset[1]
    assert image.shape == (6, 256, 256)
    assert abs(float(image[0, 0, 0]) - (20.0 / 255.0)) < 1e-6
    assert abs(float(image[3, 0, 0]) - (10.0 / 255.0)) < 1e-6

    image_fallback, _, _ = dataset[0]
    assert torch.allclose(image_fallback[:3], image_fallback[3:])


def test_dataset_two_frame_diff_channels(tmp_path) -> None:
    data_dir = tmp_path / "out"
    frames_dir = data_dir / "state_frames"
    frames_dir.mkdir(parents=True)

    write_dummy_image(str(frames_dir / "000000.png"), 10)
    write_dummy_image(str(frames_dir / "000001.png"), 20)

    records = [
        {
            "idx": 0,
            "action_id": "b_action",
            "grid_id": 3,
            "state_path": "state_frames/000000.png",
            "meta": {"gw": 6, "gh": 9, "fps_effective": 10.0},
        },
        {
            "idx": 1,
            "action_id": "a_action",
            "grid_id": 4,
            "state_path": "state_frames/000001.png",
            "meta": {"gw": 6, "gh": 9, "fps_effective": 10.0},
        },
    ]
    dataset_path = data_dir / "dataset.jsonl"
    data_dir.mkdir(parents=True, exist_ok=True)
    with open(dataset_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    loaded = load_records(str(data_dir))
    vocab = load_or_create_vocab(str(data_dir), loaded)
    dataset = StateActionDataset(
        str(data_dir),
        loaded,
        vocab,
        two_frame=True,
        diff_channels=True,
        delta_frames=1,
    )

    image, _, _ = dataset[1]
    assert image.shape == (9, 256, 256)
    assert abs(float(image[0, 0, 0]) - (20.0 / 255.0)) < 1e-6
    assert abs(float(image[3, 0, 0]) - (10.0 / 255.0)) < 1e-6
    assert abs(float(image[6, 0, 0]) - (10.0 / 255.0)) < 1e-6

    image_fallback, _, _ = dataset[0]
    assert torch.allclose(image_fallback[6:], torch.zeros_like(image_fallback[6:]))
