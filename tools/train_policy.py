import argparse
import json
import os
import random
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from state2action_rt.learning.dataset import (
    ActionVocab,
    StateActionDataset,
    infer_grid_shape,
    load_or_create_vocab,
    load_records,
    load_records_from_path,
    split_records,
)
from state2action_rt.learning.model import PolicyNet


def warn(msg: str) -> None:
    print(f"[warn] {msg}", file=sys.stderr)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, ks: List[int]) -> Dict[int, float]:
    maxk = min(max(ks), logits.size(1))
    _, pred = logits.topk(maxk, dim=1)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    results: Dict[int, float] = {}
    batch_size = targets.size(0)
    for k in ks:
        use_k = min(k, maxk)
        correct_k = correct[:use_k].reshape(-1).float().sum().item()
        results[k] = correct_k / batch_size
    return results


def masked_topk_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ks: List[int],
    mask: torch.Tensor,
) -> Tuple[Dict[int, float], int]:
    valid = mask.nonzero(as_tuple=False).squeeze(1)
    if valid.numel() == 0:
        return {k: 0.0 for k in ks}, 0
    return topk_accuracy(logits[valid], targets[valid], ks), int(valid.numel())


def evaluate(
    model: PolicyNet,
    loader: DataLoader,
    criterion: nn.Module,
    grid_criterion: nn.Module,
    device: torch.device,
    noop_id: int | None,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    card_top1 = 0.0
    card_top3 = 0.0
    grid_top1 = 0.0
    grid_top3 = 0.0
    grid_total = 0
    noop_correct = 0.0
    noop_total = 0.0
    total_samples = 0
    with torch.no_grad():
        for images, card_labels, grid_labels, hand_available in loader:
            images = images.to(device)
            card_labels = card_labels.to(device)
            grid_labels = grid_labels.to(device)
            hand_available = hand_available.to(device)
            card_logits, grid_logits = model(images, hand_available=hand_available)
            loss = criterion(card_logits, card_labels) + grid_criterion(grid_logits, grid_labels)
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            card_acc = topk_accuracy(card_logits, card_labels, [1, 3])
            grid_mask = grid_labels != -1
            grid_acc, grid_count = masked_topk_accuracy(grid_logits, grid_labels, [1, 3], grid_mask)
            card_top1 += card_acc[1] * batch_size
            card_top3 += card_acc[3] * batch_size
            grid_top1 += grid_acc[1] * grid_count
            grid_top3 += grid_acc[3] * grid_count
            grid_total += grid_count

            if noop_id is not None:
                noop_mask = card_labels == noop_id
                noop_total += noop_mask.sum().item()
                if noop_mask.any():
                    preds = torch.argmax(card_logits, dim=1)
                    noop_correct += (preds[noop_mask] == noop_id).sum().item()

    if total_samples == 0:
        return {
            "loss": 0.0,
            "card_top1": 0.0,
            "card_top3": 0.0,
            "grid_top1": 0.0,
            "grid_top3": 0.0,
            "noop_acc": 0.0,
        }
    if grid_total == 0:
        grid_top1_avg = 0.0
        grid_top3_avg = 0.0
    else:
        grid_top1_avg = grid_top1 / grid_total
        grid_top3_avg = grid_top3 / grid_total
    return {
        "loss": total_loss / total_samples,
        "card_top1": card_top1 / total_samples,
        "card_top3": card_top3 / total_samples,
        "grid_top1": grid_top1_avg,
        "grid_top3": grid_top3_avg,
        "noop_acc": (noop_correct / noop_total) if noop_total > 0 else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a lightweight policy model from dataset.jsonl.")
    parser.add_argument("--data-dir", required=True, help="Dataset directory containing dataset.jsonl")
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Optional dataset jsonl path (default: data-dir/dataset.jsonl)",
    )
    parser.add_argument("--out-dir", required=True, help="Output directory for checkpoints and metrics")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--two-frame", action="store_true", help="Concatenate current/past frames as input")
    parser.add_argument(
        "--diff-channels",
        action="store_true",
        help="Append diff channels (current - past); requires --two-frame",
    )
    parser.add_argument("--delta-frames", type=int, default=1, help="Frame offset for past frame")
    parser.add_argument("--delta-sec", type=float, default=None, help="Optional seconds offset for past frame")
    parser.add_argument("--gw", type=int, required=True, help="Grid width")
    parser.add_argument("--gh", type=int, required=True, help="Grid height")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Training device",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    if args.diff_channels and not args.two_frame:
        raise ValueError("--diff-channels requires --two-frame")

    dataset_path = args.dataset_path or os.path.join(args.data_dir, "dataset.jsonl")
    records = load_records_from_path(dataset_path)
    if not records:
        warn("dataset.jsonl is empty")
        return 1

    inferred_shape = infer_grid_shape(records)
    if inferred_shape is not None and inferred_shape != (args.gw, args.gh):
        raise ValueError("grid shape mismatch between dataset and arguments")

    vocab = load_or_create_vocab(args.data_dir, records)
    num_actions = len(vocab.id_to_action)
    num_grids = args.gw * args.gh

    train_records, val_records = split_records(records, args.val_ratio, args.seed)
    train_dataset = StateActionDataset(
        args.data_dir,
        train_records,
        vocab,
        two_frame=args.two_frame,
        diff_channels=args.diff_channels,
        delta_frames=args.delta_frames,
        delta_sec=args.delta_sec,
    )
    val_dataset = StateActionDataset(
        args.data_dir,
        val_records,
        vocab,
        two_frame=args.two_frame,
        diff_channels=args.diff_channels,
        delta_frames=args.delta_frames,
        delta_sec=args.delta_sec,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = select_device(args.device)
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cuda"
        print(f"using device: {device} ({device_name})")
    else:
        print(f"using device: {device}")
    if args.diff_channels:
        in_channels = 9
    elif args.two_frame:
        in_channels = 6
    else:
        in_channels = 3
    model = PolicyNet(num_actions=num_actions, num_grids=num_grids, in_channels=in_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    grid_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    noop_id = vocab.action_to_id.get("__NOOP__")

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val = float("inf")
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        total_samples = 0
        for images, card_labels, grid_labels, hand_available in tqdm(train_loader, desc=f"epoch {epoch}"):
            images = images.to(device)
            card_labels = card_labels.to(device)
            grid_labels = grid_labels.to(device)
            hand_available = hand_available.to(device)
            optimizer.zero_grad()
            card_logits, grid_logits = model(images, hand_available=hand_available)
            loss = criterion(card_logits, card_labels) + grid_criterion(grid_logits, grid_labels)
            loss.backward()
            optimizer.step()
            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

        train_loss = running_loss / max(1, total_samples)
        val_metrics = evaluate(model, val_loader, criterion, grid_criterion, device, noop_id)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_card_top1": val_metrics["card_top1"],
                "val_card_top3": val_metrics["card_top3"],
                "val_grid_top1": val_metrics["grid_top1"],
                "val_grid_top3": val_metrics["grid_top3"],
                "val_noop_acc": val_metrics["noop_acc"],
            }
        )

        ckpt_payload = {
            "model_state": model.state_dict(),
            "num_actions": num_actions,
            "num_grids": num_grids,
            "gw": args.gw,
            "gh": args.gh,
            "embedding_dim": model.embedding_dim,
            "id_to_action": vocab.id_to_action,
            "two_frame": args.two_frame,
            "diff_channels": args.diff_channels,
            "delta_frames": args.delta_frames,
            "in_channels": in_channels,
        }
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            ckpt_path = os.path.join(ckpt_dir, "best.pt")
            torch.save(ckpt_payload, ckpt_path)

        last_path = os.path.join(ckpt_dir, "last.pt")
        torch.save(ckpt_payload, last_path)

    metrics_path = os.path.join(args.out_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=True, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
