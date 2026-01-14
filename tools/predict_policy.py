import argparse
import os
import sys
from typing import List, Tuple

import cv2
import torch

from state2action_rt.grid import grid_id_to_cell_rect
from state2action_rt.learning.dataset import (
    ActionVocab,
    infer_grid_shape,
    load_record_by_idx,
    load_state_pair_tensor,
    load_state_image_tensor,
    resolve_delta_frames,
)
from state2action_rt.learning.model import PolicyNet


def warn(msg: str) -> None:
    print(f"[warn] {msg}", file=sys.stderr)


def load_checkpoint(path: str, device: torch.device) -> dict:
    return torch.load(path, map_location=device)


def topk_probs(probs: torch.Tensor, k: int) -> Tuple[List[int], List[float]]:
    values, indices = torch.topk(probs, k)
    return indices.tolist(), values.tolist()


def render_overlay(
    data_dir: str,
    record: dict,
    grid_id: int,
    gw: int,
    gh: int,
    out_path: str,
) -> None:
    state_path = os.path.join(data_dir, record["state_path"])
    state_img = cv2.imread(state_path, cv2.IMREAD_COLOR)
    if state_img is None:
        raise FileNotFoundError(f"failed to load state image: {state_path}")
    roi = record["roi"]
    roi_w = max(1.0, float(roi[2] - roi[0]))
    roi_h = max(1.0, float(roi[3] - roi[1]))
    scale_x = state_img.shape[1] / roi_w
    scale_y = state_img.shape[0] / roi_h

    cell = grid_id_to_cell_rect(grid_id, tuple(roi), gw, gh)
    sx1 = int(round((cell[0] - roi[0]) * scale_x))
    sy1 = int(round((cell[1] - roi[1]) * scale_y))
    sx2 = int(round((cell[2] - roi[0]) * scale_x))
    sy2 = int(round((cell[3] - roi[1]) * scale_y))

    cv2.rectangle(state_img, (sx1, sy1), (sx2, sy2), (0, 255, 0), 2)
    ok = cv2.imwrite(out_path, state_img)
    if not ok:
        raise RuntimeError("failed to write overlay image")


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict top-k action candidates for a dataset sample.")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt")
    parser.add_argument("--data-dir", required=True, help="Dataset directory containing dataset.jsonl")
    parser.add_argument("--idx", type=int, required=True, help="Record idx from dataset.jsonl")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--render-overlay", action="store_true", help="Write top-1 grid overlay to out.png")
    parser.add_argument("--overlay-out", default="out.png", help="Overlay output path")
    parser.add_argument("--two-frame", action="store_true", default=None, help="Force two-frame input")
    parser.add_argument("--delta-frames", type=int, default=None, help="Frame offset for past frame")
    parser.add_argument("--delta-sec", type=float, default=None, help="Optional seconds offset for past frame")
    parser.add_argument("--in-channels", type=int, default=None, help="Override model input channels")
    args = parser.parse_args()

    device = torch.device("cpu")
    checkpoint = load_checkpoint(args.checkpoint, device)
    checkpoint_vocab = checkpoint.get("id_to_action")
    if checkpoint_vocab:
        vocab = ActionVocab([str(v) for v in checkpoint_vocab])
        vocab_path = os.path.join(args.data_dir, "vocab.json")
        if os.path.exists(vocab_path):
            disk_vocab = ActionVocab.load(vocab_path)
            if disk_vocab.id_to_action != vocab.id_to_action:
                warn("vocab.json does not match checkpoint vocab")
    else:
        vocab = ActionVocab.load(os.path.join(args.data_dir, "vocab.json"))
    record = load_record_by_idx(args.data_dir, args.idx)
    if record is None:
        warn("record not found")
        return 1

    inferred_shape = infer_grid_shape([record])
    gw, gh = inferred_shape if inferred_shape is not None else (6, 9)
    num_grids = gw * gh
    if len(vocab.id_to_action) != int(checkpoint.get("num_actions", len(vocab.id_to_action))):
        warn("vocab size does not match checkpoint")
        return 1

    ckpt_two_frame = bool(checkpoint.get("two_frame", False))
    ckpt_delta_frames = int(checkpoint.get("delta_frames", 1))
    ckpt_in_channels = checkpoint.get("in_channels")
    if ckpt_in_channels is None:
        ckpt_in_channels = 6 if ckpt_two_frame else 3
    else:
        ckpt_in_channels = int(ckpt_in_channels)

    two_frame = ckpt_two_frame if args.two_frame is None else args.two_frame
    delta_frames = ckpt_delta_frames if args.delta_frames is None else args.delta_frames
    in_channels = ckpt_in_channels if args.in_channels is None else args.in_channels
    if not two_frame and args.in_channels is None:
        in_channels = 3

    model = PolicyNet(num_actions=len(vocab.id_to_action), num_grids=num_grids, in_channels=in_channels)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    if two_frame:
        delta_frames = resolve_delta_frames(record, delta_frames, args.delta_sec)
        image = load_state_pair_tensor(args.data_dir, record["state_path"], delta_frames).unsqueeze(0)
    else:
        image = load_state_image_tensor(args.data_dir, record["state_path"]).unsqueeze(0)
    with torch.no_grad():
        card_logits, grid_logits = model(image)
        card_probs = torch.softmax(card_logits, dim=1).squeeze(0)
        grid_probs = torch.softmax(grid_logits, dim=1).squeeze(0)

    k_card = min(args.topk, card_probs.numel())
    k_grid = min(args.topk, grid_probs.numel())
    card_ids, card_vals = topk_probs(card_probs, k_card)
    grid_ids, grid_vals = topk_probs(grid_probs, k_grid)

    candidates = []
    for c_id, c_prob in zip(card_ids, card_vals):
        action_id = vocab.id_to_action[c_id]
        if action_id == "__NOOP__":
            candidates.append((c_prob, c_id, c_prob, -1, 0.0))
            continue
        for g_id, g_prob in zip(grid_ids, grid_vals):
            candidates.append((c_prob * g_prob, c_id, c_prob, g_id, g_prob))
    candidates.sort(key=lambda x: x[0], reverse=True)
    top_candidates = candidates[: args.topk]

    for rank, (score, card_id, card_prob, grid_id, grid_prob) in enumerate(top_candidates, start=1):
        action_id = vocab.id_to_action[card_id]
        if grid_id >= 0:
            gx = grid_id % gw
            gy = grid_id // gw
            grid_info = f"grid_id={grid_id} (gx={gx},gy={gy}) grid_prob={grid_prob:.4f}"
        else:
            grid_info = "grid_id=-1 grid_prob=0.0000"
        print(
            f"{rank:02d} action_id={action_id} card_prob={card_prob:.4f} "
            f"{grid_info} "
            f"combined_score={score:.6f}"
        )

    if args.render_overlay and top_candidates:
        top_grid = top_candidates[0][3]
        if top_grid >= 0:
            render_overlay(args.data_dir, record, top_grid, gw, gh, args.overlay_out)
            print(f"overlay_saved={args.overlay_out}")
        else:
            print("overlay_skipped=NOOP")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
