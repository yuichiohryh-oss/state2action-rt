import argparse
import math
import os
from pathlib import Path
import sys
from typing import List, Tuple

import cv2
import torch

from state2action_rt.grid import grid_id_to_cell_rect
from state2action_rt.hand_cards import (
    infer_hand_card_ids_from_frame,
    load_hand_card_templates,
    parse_card_id,
)
from state2action_rt.hand_features import hand_available_from_frame
from state2action_rt.learning.dataset import (
    ActionVocab,
    infer_grid_shape,
    load_record_by_idx,
    load_state_pair_tensor,
    load_state_pair_with_diff_tensor,
    load_state_image_tensor,
    resolve_delta_frames,
)
from state2action_rt.learning.model import PolicyNet


def warn(msg: str) -> None:
    print(f"[warn] {msg}", file=sys.stderr)


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def load_checkpoint(path: str, device: torch.device) -> dict:
    return torch.load(path, map_location=device)


def topk_probs(
    probs: torch.Tensor,
    k: int,
    allowed_indices: List[int] | None = None,
) -> Tuple[List[int], List[float]]:
    if k <= 0:
        return [], []
    if allowed_indices is None:
        values, indices = torch.topk(probs, k)
        return indices.tolist(), values.tolist()
    if not allowed_indices:
        return [], []
    allowed = torch.tensor(allowed_indices, device=probs.device, dtype=torch.long)
    allowed_probs = probs.index_select(0, allowed)
    k = min(k, int(allowed_probs.numel()))
    if k <= 0:
        return [], []
    values, indices = torch.topk(allowed_probs, k)
    resolved = [allowed_indices[i] for i in indices.tolist()]
    return resolved, values.tolist()


def normalize_hand_available(value: object, n_slots: int = 4) -> List[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != n_slots:
        return None
    normalized: List[int] = []
    for item in value:
        try:
            ivalue = int(item)
        except (TypeError, ValueError):
            return None
        normalized.append(1 if ivalue != 0 else 0)
    return normalized


def normalize_hand_card_ids(value: object, n_slots: int = 4) -> List[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != n_slots:
        return None
    normalized: List[int] = []
    for item in value:
        try:
            normalized.append(int(item))
        except (TypeError, ValueError):
            return None
    return normalized


def normalize_elixir(value: object) -> int | None:
    try:
        ivalue = int(value)
    except (TypeError, ValueError):
        return None
    if ivalue == -1:
        return -1
    if 0 <= ivalue <= 10:
        return ivalue
    return None


def normalize_elixir_frac(value: object) -> float | None:
    try:
        fvalue = float(value)
    except (TypeError, ValueError):
        return None
    if fvalue == -1.0:
        return -1.0
    if 0.0 <= fvalue <= 10.0:
        return fvalue
    return None


def build_allowed_card_ids(
    hand_available: List[int] | None,
    hand_card_ids: List[int] | None,
) -> set[int]:
    if not hand_available or not hand_card_ids:
        return set()
    allowed: set[int] = set()
    for slot_id, card_id in enumerate(hand_card_ids):
        if slot_id < len(hand_available) and bool(hand_available[slot_id]) and card_id != -1:
            allowed.add(int(card_id))
    return allowed


def build_valid_action_indices(
    allowed_card_ids: set[int],
    card_id_to_action_idx_map: dict[int, int],
    noop_idx: int | None,
    skill_idx: int | None,
) -> set[int]:
    valid: set[int] = set()
    if noop_idx is not None:
        valid.add(noop_idx)
    if skill_idx is not None:
        valid.add(skill_idx)
    for card_id in allowed_card_ids:
        action_idx = card_id_to_action_idx_map.get(card_id)
        if action_idx is not None:
            valid.add(action_idx)
    return valid


def find_slot_index(card_id: int | None, hand_card_ids: List[int] | None) -> int:
    if card_id is None or not hand_card_ids:
        return -1
    try:
        return hand_card_ids.index(card_id)
    except ValueError:
        return -1


def format_mask_topk(
    label: str,
    card_probs: torch.Tensor,
    vocab: ActionVocab,
    topk: int,
    hand_card_ids: List[int] | None,
    allowed_indices: List[int] | None = None,
) -> List[str]:
    k = min(topk, int(card_probs.numel()))
    card_ids, card_vals = topk_probs(card_probs, k, allowed_indices=allowed_indices)
    lines: List[str] = []
    for rank, (action_idx, score) in enumerate(zip(card_ids, card_vals), start=1):
        action_id = vocab.id_to_action[action_idx]
        card_id = parse_card_id(str(action_id))
        slot_index = find_slot_index(card_id, hand_card_ids)
        card_label = card_id if card_id is not None else -1
        lines.append(
            f"{label} {rank:02d} action_id={action_id} card_id={card_label} "
            f"score={score:.6f} slot_index={slot_index}"
        )
    return lines


def combine_score(card_prob: float, grid_prob: float, mode: str) -> float:
    if mode == "mul":
        return card_prob * grid_prob
    if mode == "add":
        return card_prob + grid_prob
    if mode == "logadd":
        eps = 1e-8
        return math.log(card_prob + eps) + math.log(grid_prob + eps)
    raise ValueError(f"unknown score mode: {mode}")


def apply_noop_penalty(
    logits: torch.Tensor,
    hand_available: torch.Tensor | List[int],
    noop_idx: int | None,
    penalty: float,
) -> torch.Tensor:
    if noop_idx is None or penalty <= 0:
        return logits
    if isinstance(hand_available, torch.Tensor):
        any_available = bool(torch.any(hand_available != 0).item())
    else:
        any_available = any(bool(v) for v in hand_available)
    if not any_available:
        return logits
    adjusted = logits.clone()
    if adjusted.dim() == 1:
        adjusted[noop_idx] -= penalty
    elif adjusted.dim() == 2:
        adjusted[:, noop_idx] -= penalty
    else:
        raise ValueError("logits must be 1D or 2D")
    return adjusted


def build_card_id_to_action_idx_map(vocab: ActionVocab, max_card_id: int | None = None) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for idx, action_id in enumerate(vocab.id_to_action):
        card_id = parse_card_id(str(action_id))
        if card_id is None:
            continue
        if max_card_id is not None and card_id >= max_card_id:
            continue
        mapping[card_id] = idx
    return mapping


def apply_action_mask(
    logits: torch.Tensor,
    allowed_card_ids: set[int],
    noop_idx: int | None,
    skill_idx: int | None,
    card_id_to_action_idx_map: dict[int, int],
    mask_value: float = -1e9,
) -> torch.Tensor:
    if not card_id_to_action_idx_map:
        return logits
    exempt_indices = {idx for idx in (noop_idx, skill_idx) if idx is not None}
    masked = logits.clone()
    for card_id, action_idx in card_id_to_action_idx_map.items():
        if action_idx in exempt_indices:
            continue
        if card_id in allowed_card_ids:
            continue
        if masked.dim() == 1:
            masked[action_idx] = mask_value
        elif masked.dim() == 2:
            masked[:, action_idx] = mask_value
        else:
            raise ValueError("logits must be 1D or 2D")
    return masked


def build_candidates(
    card_probs: torch.Tensor,
    grid_probs: torch.Tensor,
    vocab: ActionVocab,
    topk: int,
    score_mode: str,
    exclude_noop: bool,
    valid_action_indices: set[int] | None = None,
) -> List[Tuple[float, int, float, int, float]]:
    allowed_card_indices: List[int] | None = None
    if valid_action_indices is not None:
        allowed_card_indices = [
            idx for idx in sorted(valid_action_indices) if 0 <= idx < int(card_probs.numel())
        ]
    k_card = min(
        topk,
        len(allowed_card_indices) if allowed_card_indices is not None else int(card_probs.numel()),
    )
    k_grid = min(topk, grid_probs.numel())
    card_ids, card_vals = topk_probs(card_probs, k_card, allowed_indices=allowed_card_indices)
    grid_ids, grid_vals = topk_probs(grid_probs, k_grid)

    candidates = []
    for c_id, c_prob in zip(card_ids, card_vals):
        action_id = vocab.id_to_action[c_id]
        if action_id == "__NOOP__":
            if exclude_noop:
                continue
            candidates.append((c_prob, c_id, c_prob, -1, 0.0))
            continue
        for g_id, g_prob in zip(grid_ids, grid_vals):
            candidates.append((combine_score(c_prob, g_prob, score_mode), c_id, c_prob, g_id, g_prob))
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[:topk]


def ensure_candidates_with_noop(
    candidates: List[Tuple[float, int, float, int, float]],
    card_probs: torch.Tensor,
    noop_idx: int | None,
    disable_hand_card_mask: bool,
    warn_fn=warn,
) -> List[Tuple[float, int, float, int, float]]:
    if candidates or disable_hand_card_mask:
        return candidates
    if noop_idx is None:
        if warn_fn is not None:
            warn_fn("hand mask removed all candidates and __NOOP__ is missing")
        return candidates
    noop_prob = float(card_probs[noop_idx].item())
    return [(noop_prob, noop_idx, noop_prob, -1, 0.0)]


def render_overlay(
    data_dir: str,
    record: dict,
    grid_id: int,
    gw: int,
    gh: int,
    out_path: str,
) -> None:
    out_path_value = out_path
    if out_path:
        out_path_obj = Path(out_path)
        out_path_obj.parent.mkdir(parents=True, exist_ok=True)
        out_path_value = str(out_path_obj)
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
    ok = cv2.imwrite(out_path_value, state_img)
    if not ok:
        raise RuntimeError("failed to write overlay image")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict top-k action candidates for a dataset sample.")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt")
    parser.add_argument("--data-dir", required=True, help="Dataset directory containing dataset.jsonl")
    parser.add_argument("--idx", type=int, required=True, help="Record idx from dataset.jsonl")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--render-overlay", action="store_true", help="Write top-1 grid overlay to out.png")
    parser.add_argument("--overlay-out", default="out.png", help="Overlay output path")
    parser.add_argument("--exclude-noop", action="store_true", help="Exclude __NOOP__ from ranking")
    parser.add_argument("--hand-s-th", type=float, default=30.0, help="Hand mean-S threshold")
    parser.add_argument("--hand-y1-ratio", type=float, default=0.90, help="Hand ROI y1 ratio")
    parser.add_argument("--hand-y2-ratio", type=float, default=0.97, help="Hand ROI y2 ratio")
    parser.add_argument("--hand-x-margin-ratio", type=float, default=0.03, help="Hand ROI x margin ratio")
    parser.add_argument(
        "--hand-templates-dir",
        default=os.path.join("templates", "hand_cards"),
        help="Directory containing hand card templates",
    )
    parser.add_argument("--hand-card-min-score", type=float, default=0.6, help="Min template score for card id")
    parser.add_argument(
        "--disable-hand-card-mask",
        action="store_true",
        help="Disable hand card action masking (debug)",
    )
    parser.add_argument(
        "--debug-hand-mask",
        action="store_true",
        help="Print pre/post hand mask top-k debug output",
    )
    parser.add_argument("--noop-penalty", type=float, default=1.5, help="NOOP logit penalty")
    parser.add_argument(
        "--score-mode",
        choices=["mul", "add", "logadd"],
        default="mul",
        help="Combined score mode (default: mul)",
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Optional dataset jsonl path (default: data-dir/dataset.jsonl)",
    )
    parser.add_argument("--two-frame", action="store_true", default=None, help="Force two-frame input")
    parser.add_argument(
        "--diff-channels",
        action="store_true",
        default=None,
        help="Force diff channels (current - past); requires --two-frame",
    )
    parser.add_argument("--delta-frames", type=int, default=None, help="Frame offset for past frame")
    parser.add_argument("--delta-sec", type=float, default=None, help="Optional seconds offset for past frame")
    parser.add_argument("--in-channels", type=int, default=None, help="Override model input channels")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Inference device",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    device = select_device(args.device)
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cuda"
        print(f"using device: {device} ({device_name})")
    else:
        print(f"using device: {device}")
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
    record = load_record_by_idx(args.data_dir, args.idx, dataset_path=args.dataset_path)
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
    ckpt_diff_channels = bool(checkpoint.get("diff_channels", False))
    ckpt_delta_frames = int(checkpoint.get("delta_frames", 1))
    ckpt_in_channels = checkpoint.get("in_channels")
    if ckpt_in_channels is None:
        if ckpt_diff_channels:
            ckpt_in_channels = 9
        elif ckpt_two_frame:
            ckpt_in_channels = 6
        else:
            ckpt_in_channels = 3
    else:
        ckpt_in_channels = int(ckpt_in_channels)

    two_frame = ckpt_two_frame if args.two_frame is None else args.two_frame
    diff_channels = ckpt_diff_channels if args.diff_channels is None else args.diff_channels
    delta_frames = ckpt_delta_frames if args.delta_frames is None else args.delta_frames
    if diff_channels and not two_frame:
        raise ValueError("--diff-channels requires --two-frame")
    if args.in_channels is None:
        if args.two_frame is None and args.diff_channels is None:
            in_channels = ckpt_in_channels
        else:
            if diff_channels:
                in_channels = 9
            elif two_frame:
                in_channels = 6
            else:
                in_channels = 3
    else:
        in_channels = args.in_channels

    model = PolicyNet(num_actions=len(vocab.id_to_action), num_grids=num_grids, in_channels=in_channels)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    state_path = os.path.join(args.data_dir, record["state_path"])
    hand_available = normalize_hand_available(record.get("hand_available"))
    hand_card_ids = normalize_hand_card_ids(record.get("hand_card_ids"))
    elixir = normalize_elixir(record.get("elixir"))
    elixir_frac = normalize_elixir_frac(record.get("elixir_frac"))
    hand_available_source = "record" if hand_available is not None else None
    hand_card_ids_source = "record" if hand_card_ids is not None else None

    frame_bgr = None
    if hand_available is None or hand_card_ids is None:
        frame_bgr = cv2.imread(state_path, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            warn(f"failed to load frame for hand features: {state_path}")
        else:
            if hand_available is None:
                try:
                    _, hand_available, _, _ = hand_available_from_frame(
                        frame_bgr,
                        s_th=args.hand_s_th,
                        y1_ratio=args.hand_y1_ratio,
                        y2_ratio=args.hand_y2_ratio,
                        x_margin_ratio=args.hand_x_margin_ratio,
                        n_slots=4,
                    )
                    hand_available_source = "frame"
                except ValueError as exc:
                    warn(f"hand_available failed: {exc}")
            if hand_card_ids is None:
                templates = load_hand_card_templates(args.hand_templates_dir)
                if not templates:
                    hand_card_ids = [-1, -1, -1, -1]
                    hand_card_ids_source = "default"
                else:
                    try:
                        hand_card_ids = infer_hand_card_ids_from_frame(
                            frame_bgr,
                            templates,
                            min_score=args.hand_card_min_score,
                            s_th=args.hand_s_th,
                            y1_ratio=args.hand_y1_ratio,
                            y2_ratio=args.hand_y2_ratio,
                            x_margin_ratio=args.hand_x_margin_ratio,
                            n_slots=4,
                        )
                        hand_card_ids_source = "frame"
                    except ValueError as exc:
                        warn(f"hand_card_ids failed: {exc}")
                        hand_card_ids = [-1, -1, -1, -1]
                        hand_card_ids_source = "default"

    if hand_available is None:
        hand_available = [0, 0, 0, 0]
        hand_available_source = "default"
    if hand_card_ids is None:
        hand_card_ids = [-1, -1, -1, -1]
        hand_card_ids_source = "default"
    if elixir is None:
        elixir = -1
        warn("elixir missing; using -1")
    if elixir_frac is None:
        elixir_frac = -1.0
    if hand_available_source == "default":
        warn("hand_available missing; using zeros")
    hand_mask_auto_skip = False
    if not args.disable_hand_card_mask:
        if all(cid == -1 for cid in hand_card_ids) or all(a == 0 for a in hand_available):
            warn("hand unknown/empty; skipping hand mask (safe)")
            hand_mask_auto_skip = True
    if (
        hand_card_ids_source == "default"
        and not args.disable_hand_card_mask
        and not hand_mask_auto_skip
    ):
        warn("hand_card_ids missing; card mask will exclude card actions")
    print(f"elixir={elixir} elixir_frac={elixir_frac:.2f}")
    hand_tensor = torch.tensor(hand_available, dtype=torch.float32, device=device).unsqueeze(0)

    if two_frame:
        delta_frames = resolve_delta_frames(record, delta_frames, args.delta_sec)
        if diff_channels:
            image = load_state_pair_with_diff_tensor(
                args.data_dir, record["state_path"], delta_frames
            ).unsqueeze(0)
        else:
            image = load_state_pair_tensor(args.data_dir, record["state_path"], delta_frames).unsqueeze(0)
    else:
        image = load_state_image_tensor(args.data_dir, record["state_path"]).unsqueeze(0)
    image = image.to(device)
    noop_idx = vocab.action_to_id.get("__NOOP__")
    skill_idx = vocab.action_to_id.get("SKILL_RIGHT")
    pre_mask_probs = None
    valid_action_indices: set[int] | None = None
    with torch.no_grad():
        card_logits, grid_logits = model(image, hand_tensor)
        card_logits = apply_noop_penalty(card_logits, hand_available, noop_idx, args.noop_penalty)
        if args.debug_hand_mask:
            pre_mask_probs = torch.softmax(card_logits, dim=1).squeeze(0)
        if not args.disable_hand_card_mask and not hand_mask_auto_skip:
            allowed_card_ids = build_allowed_card_ids(hand_available, hand_card_ids)
            card_id_to_action_idx = build_card_id_to_action_idx_map(vocab)
            card_logits = apply_action_mask(
                card_logits,
                allowed_card_ids,
                noop_idx,
                skill_idx,
                card_id_to_action_idx,
            )
            valid_action_indices = build_valid_action_indices(
                allowed_card_ids,
                card_id_to_action_idx,
                noop_idx,
                skill_idx,
            )
        card_probs = torch.softmax(card_logits, dim=1).squeeze(0)
        grid_probs = torch.softmax(grid_logits, dim=1).squeeze(0)

    if args.debug_hand_mask and pre_mask_probs is not None:
        for line in format_mask_topk(
            "mask_pre_topk",
            pre_mask_probs,
            vocab,
            args.topk,
            hand_card_ids,
        ):
            print(line)
        allowed_indices = None
        if valid_action_indices is not None:
            allowed_indices = sorted(valid_action_indices)
        for line in format_mask_topk(
            "mask_post_topk",
            card_probs,
            vocab,
            args.topk,
            hand_card_ids,
            allowed_indices=allowed_indices,
        ):
            print(line)

    top_candidates = build_candidates(
        card_probs,
        grid_probs,
        vocab,
        args.topk,
        args.score_mode,
        args.exclude_noop,
        valid_action_indices=valid_action_indices,
    )
    top_candidates = ensure_candidates_with_noop(
        top_candidates,
        card_probs,
        noop_idx,
        args.disable_hand_card_mask or hand_mask_auto_skip,
    )

    for rank, (score, card_id, card_prob, grid_id, grid_prob) in enumerate(top_candidates, start=1):
        action_id = vocab.id_to_action[card_id]
        action_card_id = parse_card_id(str(action_id))
        slot_index = find_slot_index(action_card_id, hand_card_ids)
        if grid_id >= 0:
            gx = grid_id % gw
            gy = grid_id // gw
            grid_info = f"grid_id={grid_id} (gx={gx},gy={gy}) grid_prob={grid_prob:.4f}"
        else:
            grid_info = "grid_id=-1 grid_prob=0.0000"
        print(
            f"{rank:02d} action_id={action_id} card_prob={card_prob:.4f} "
            f"{grid_info} "
            f"slot_index={slot_index} combined_score={score:.6f}"
        )

    if args.render_overlay:
        if top_candidates:
            top_grid = top_candidates[0][3]
            if top_grid >= 0:
                render_overlay(args.data_dir, record, top_grid, gw, gh, args.overlay_out)
                print(f"overlay_saved={args.overlay_out}")
            else:
                print("overlay_skipped=NOOP")
        elif args.exclude_noop:
            print("overlay_skipped=NOOP")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
