import argparse
import json
import os
import sys

import cv2

from state2action_rt.grid import grid_id_to_cell_rect


def warn(msg: str) -> None:
    print(f"[warn] {msg}", file=sys.stderr)


def load_record(dataset_path: str, idx: int) -> dict | None:
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if int(record.get("idx", -1)) == idx:
                return record
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a debug overlay for a dataset sample.")
    parser.add_argument("--out-dir", required=True, help="Dataset output directory")
    parser.add_argument("--idx", type=int, required=True, help="Record index to inspect")
    parser.add_argument("--out", default=None, help="Output image path")
    args = parser.parse_args()

    dataset_path = os.path.join(args.out_dir, "dataset.jsonl")
    record = load_record(dataset_path, args.idx)
    if record is None:
        warn("record not found")
        return 1

    state_path = os.path.join(args.out_dir, record["state_path"])
    state_img = cv2.imread(state_path, cv2.IMREAD_COLOR)
    if state_img is None:
        warn("failed to load state image")
        return 1

    roi = record["roi"]
    grid_id = int(record["grid_id"])
    meta = record.get("meta", {})
    gw = int(meta.get("gw", 6))
    gh = int(meta.get("gh", 9))

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
    cx = int(round((sx1 + sx2) / 2))
    cy = int(round((sy1 + sy2) / 2))
    cv2.circle(state_img, (cx, cy), 4, (0, 0, 255), -1)

    out_path = args.out
    if out_path is None:
        out_path = os.path.join(args.out_dir, f"inspect_{args.idx:06d}.png")
    ok = cv2.imwrite(out_path, state_img)
    if not ok:
        warn("failed to write output")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
