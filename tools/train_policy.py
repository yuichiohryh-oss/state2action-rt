import argparse
import json
import os
import random
import sys
from typing import Dict, List

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


def evaluate(
    model: PolicyNet,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    card_top1 = 0.0
    card_top3 = 0.0
    grid_top1 = 0.0
    grid_top3 = 0.0
    total_samples = 0
    with torch.no_grad():
        for images, card_labels, grid_labels in loader:
            images = images.to(device)
            card_labels = card_labels.to(device)
            grid_labels = grid_labels.to(device)
            card_logits, grid_logits = model(images)
            loss = criterion(card_logits, card_labels) + criterion(grid_logits, grid_labels)
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            card_acc = topk_accuracy(card_logits, card_labels, [1, 3])
            grid_acc = topk_accuracy(grid_logits, grid_labels, [1, 3])
            card_top1 += card_acc[1] * batch_size
            card_top3 += card_acc[3] * batch_size
            grid_top1 += grid_acc[1] * batch_size
            grid_top3 += grid_acc[3] * batch_size

    if total_samples == 0:
        return {
            "loss": 0.0,
            "card_top1": 0.0,
            "card_top3": 0.0,
            "grid_top1": 0.0,
            "grid_top3": 0.0,
        }
    return {
        "loss": total_loss / total_samples,
        "card_top1": card_top1 / total_samples,
        "card_top3": card_top3 / total_samples,
        "grid_top1": grid_top1 / total_samples,
        "grid_top3": grid_top3 / total_samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a lightweight policy model from dataset.jsonl.")
    parser.add_argument("--data-dir", required=True, help="Dataset directory containing dataset.jsonl")
    parser.add_argument("--out-dir", required=True, help="Output directory for checkpoints and metrics")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
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

    records = load_records(args.data_dir)
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
    train_dataset = StateActionDataset(args.data_dir, train_records, vocab)
    val_dataset = StateActionDataset(args.data_dir, val_records, vocab)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = select_device(args.device)
    model = PolicyNet(num_actions=num_actions, num_grids=num_grids).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val = float("inf")
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        total_samples = 0
        for images, card_labels, grid_labels in tqdm(train_loader, desc=f"epoch {epoch}"):
            images = images.to(device)
            card_labels = card_labels.to(device)
            grid_labels = grid_labels.to(device)
            optimizer.zero_grad()
            card_logits, grid_logits = model(images)
            loss = criterion(card_logits, card_labels) + criterion(grid_logits, grid_labels)
            loss.backward()
            optimizer.step()
            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

        train_loss = running_loss / max(1, total_samples)
        val_metrics = evaluate(model, val_loader, criterion, device)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_card_top1": val_metrics["card_top1"],
                "val_card_top3": val_metrics["card_top3"],
                "val_grid_top1": val_metrics["grid_top1"],
                "val_grid_top3": val_metrics["grid_top3"],
            }
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            ckpt_path = os.path.join(ckpt_dir, "best.pt")
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "num_actions": num_actions,
                    "num_grids": num_grids,
                    "gw": args.gw,
                    "gh": args.gh,
                    "embedding_dim": model.embedding_dim,
                },
                ckpt_path,
            )

    metrics_path = os.path.join(args.out_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=True, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
