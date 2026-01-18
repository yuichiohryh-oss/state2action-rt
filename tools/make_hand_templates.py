import argparse
import os
import cv2

from state2action_rt.hand_features import crop_match_roi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot-png", required=True, help="path to frame_xxxx_slotY.png")
    ap.add_argument("--card-id", required=True, type=int, help="0..7")
    ap.add_argument("--out-dir", default=r"templates\hand_cards")
    ap.add_argument("--side-cut", type=float, default=0.18)
    ap.add_argument("--bottom-cut", type=float, default=0.45)
    ap.add_argument("--size", type=int, default=64)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    img = cv2.imread(args.slot_png)
    if img is None:
        raise SystemExit(f"failed to read: {args.slot_png}")

    match = crop_match_roi(img, side_cut_ratio=args.side_cut, bottom_cut_ratio=args.bottom_cut)
    gray = cv2.cvtColor(match, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (args.size, args.size), interpolation=cv2.INTER_AREA)

    out_path = os.path.join(args.out_dir, f"{args.card_id}.png")
    cv2.imwrite(out_path, gray)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
