from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import cv2
import numpy as np
import torch


@dataclass(frozen=True)
class ActionVocab:
    id_to_action: List[str]

    @property
    def action_to_id(self) -> dict[str, int]:
        return {action_id: idx for idx, action_id in enumerate(self.id_to_action)}

    @classmethod
    def build(cls, records: Iterable[dict]) -> "ActionVocab":
        action_ids = sorted({str(record["action_id"]) for record in records})
        return cls(action_ids)

    @classmethod
    def load(cls, path: str) -> "ActionVocab":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        id_to_action = [str(v) for v in payload.get("id_to_action", [])]
        if not id_to_action:
            raise ValueError("vocab.json missing id_to_action")
        return cls(id_to_action)

    def save(self, path: str) -> None:
        payload = {"id_to_action": self.id_to_action}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)


def load_records_from_path(dataset_path: str) -> List[dict]:
    records: List[dict] = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def load_records(data_dir: str) -> List[dict]:
    dataset_path = os.path.join(data_dir, "dataset.jsonl")
    return load_records_from_path(dataset_path)


def infer_grid_shape(records: Iterable[dict]) -> Tuple[int, int] | None:
    shapes = set()
    for record in records:
        meta = record.get("meta", {})
        if "gw" in meta and "gh" in meta:
            shapes.add((int(meta["gw"]), int(meta["gh"])))
    if not shapes:
        return None
    if len(shapes) > 1:
        raise ValueError("dataset contains mixed grid shapes")
    return next(iter(shapes))


def load_or_create_vocab(data_dir: str, records: Iterable[dict]) -> ActionVocab:
    vocab_path = os.path.join(data_dir, "vocab.json")
    if os.path.exists(vocab_path):
        return ActionVocab.load(vocab_path)
    vocab = ActionVocab.build(records)
    vocab.save(vocab_path)
    return vocab


def split_records(records: List[dict], val_ratio: float, seed: int) -> Tuple[List[dict], List[dict]]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0.0, 1.0)")
    indices = list(range(len(records)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_count = int(round(len(records) * val_ratio))
    val_indices = set(indices[:val_count])
    train_records = [record for i, record in enumerate(records) if i not in val_indices]
    val_records = [record for i, record in enumerate(records) if i in val_indices]
    return train_records, val_records


def load_state_image_tensor(data_dir: str, state_path: str) -> torch.Tensor:
    img_path = os.path.join(data_dir, state_path)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"failed to load image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[0] != 256 or img.shape[1] != 256:
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).contiguous()
    return tensor


def resolve_delta_frames(record: dict, delta_frames: int, delta_sec: float | None) -> int:
    if delta_sec is None:
        return delta_frames
    meta = record.get("meta", {})
    fps_effective = meta.get("fps_effective")
    if fps_effective is None:
        return delta_frames
    try:
        fps_value = float(fps_effective)
    except (TypeError, ValueError):
        return delta_frames
    return max(0, int(round(delta_sec * fps_value)))


def resolve_past_state_path(data_dir: str, state_path: str, delta_frames: int) -> str:
    if delta_frames <= 0:
        return state_path
    directory, filename = os.path.split(state_path)
    stem, ext = os.path.splitext(filename)
    if not stem.isdigit():
        return state_path
    past_index = int(stem) - delta_frames
    if past_index < 0:
        return state_path
    past_name = f"{past_index:0{len(stem)}d}{ext}"
    past_path = os.path.join(directory, past_name) if directory else past_name
    abs_path = os.path.join(data_dir, past_path)
    if not os.path.exists(abs_path):
        return state_path
    return past_path


def load_state_pair_tensor(data_dir: str, state_path: str, delta_frames: int) -> torch.Tensor:
    current = load_state_image_tensor(data_dir, state_path)
    past_path = resolve_past_state_path(data_dir, state_path, delta_frames)
    past = load_state_image_tensor(data_dir, past_path)
    return torch.cat([current, past], dim=0)


def load_state_pair_with_diff_tensor(data_dir: str, state_path: str, delta_frames: int) -> torch.Tensor:
    current = load_state_image_tensor(data_dir, state_path)
    past_path = resolve_past_state_path(data_dir, state_path, delta_frames)
    past = load_state_image_tensor(data_dir, past_path)
    diff = current - past
    return torch.cat([current, past, diff], dim=0)


class StateActionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str,
        records: List[dict],
        vocab: ActionVocab,
        two_frame: bool = False,
        diff_channels: bool = False,
        delta_frames: int = 1,
        delta_sec: float | None = None,
    ) -> None:
        if delta_frames < 0:
            raise ValueError("delta_frames must be non-negative")
        if diff_channels and not two_frame:
            raise ValueError("diff_channels requires two_frame=True")
        self.data_dir = data_dir
        self.records = records
        self.vocab = vocab
        self.two_frame = two_frame
        self.diff_channels = diff_channels
        self.delta_frames = delta_frames
        self.delta_sec = delta_sec

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        record = self.records[idx]
        state_path = record["state_path"]
        if self.two_frame:
            delta_frames = resolve_delta_frames(record, self.delta_frames, self.delta_sec)
            if self.diff_channels:
                image = load_state_pair_with_diff_tensor(self.data_dir, state_path, delta_frames)
            else:
                image = load_state_pair_tensor(self.data_dir, state_path, delta_frames)
        else:
            image = load_state_image_tensor(self.data_dir, state_path)
        action_id = str(record["action_id"])
        card_label = self.vocab.action_to_id[action_id]
        grid_label = int(record["grid_id"])
        return image, card_label, grid_label


def load_record_by_idx(data_dir: str, idx: int) -> dict | None:
    dataset_path = os.path.join(data_dir, "dataset.jsonl")
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if int(record.get("idx", -1)) == idx:
                return record
    return None
