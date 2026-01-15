import argparse
import os
import sys

import cv2

from state2action_rt.frame_source import VideoFrameSource
from state2action_rt.hand_features import hand_available_from_frame


def warn(msg: str) -> None:
    print(f"[warn] {msg}", file=sys.stderr)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect hand slots by HSV saturation over video frames."
    )
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--out-dir", default="out/hand_debug", help="Output directory")
    parser.add_argument(
        "--stride",
        type=positive_int,
        default=15,
        help="Process every N frames",
    )
    parser.add_argument("--start-sec", type=float, default=0.0, help="Start time in seconds")
    parser.add_argument(
        "--end-sec",
        type=float,
        default=-1.0,
        help="End time in seconds (-1 for end)",
    )
    parser.add_argument(
        "--max-frames",
        type=positive_int,
        default=200,
        help="Maximum number of processed frames",
    )
    parser.add_argument(
        "--x-margin-ratio",
        type=float,
        default=0.03,
        help="Horizontal margin ratio to trim both sides",
    )
    parser.add_argument("--s-th", type=float, default=30.0, help="Saturation threshold")
    parser.add_argument(
        "--save-full",
        action="store_true",
        help="Save full frame images in addition to hand ROI/slots",
    )
    parser.add_argument(
        "--write-csv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write summary.csv to out-dir",
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    try:
        src = VideoFrameSource(args.video)
    except Exception as exc:
        warn(f"failed to open video source: {exc}")
        return 1

    summary_path = os.path.join(args.out_dir, "summary.csv")
    csv_file = None
    try:
        if args.write_csv:
            is_new = not os.path.exists(summary_path)
            csv_file = open(summary_path, "a", encoding="utf-8")
            if is_new:
                csv_file.write(
                    "t_sec,frame_idx,mean_s0,mean_s1,mean_s2,mean_s3,avail0,avail1,avail2,avail3\n"
                )

        fps = src.fps
        start_idx = max(0, int(args.start_sec * fps))
        end_idx = None if args.end_sec < 0 else int(args.end_sec * fps)
        if end_idx is not None and end_idx < start_idx:
            warn("end-sec is earlier than start-sec")
            return 1

        processed = 0
        frame_idx = start_idx
        while True:
            if processed >= args.max_frames:
                break
            if end_idx is not None and frame_idx > end_idx:
                break

            frame = src.get_frame(frame_idx)
            if frame is None:
                break

            mean_s_list, avail_list, hand_bgr, slots = hand_available_from_frame(
                frame,
                s_th=args.s_th,
                x_margin_ratio=args.x_margin_ratio,
                n_slots=4,
            )

            hand_path = os.path.join(args.out_dir, f"frame_{frame_idx:06d}_hand.png")
            cv2.imwrite(hand_path, hand_bgr)
            for i, slot in enumerate(slots):
                slot_path = os.path.join(args.out_dir, f"frame_{frame_idx:06d}_slot{i}.png")
                cv2.imwrite(slot_path, slot)
            if args.save_full:
                full_path = os.path.join(args.out_dir, f"frame_{frame_idx:06d}_full.png")
                cv2.imwrite(full_path, frame)

            t_sec = frame_idx / fps
            mean_s_str = ", ".join(f"{val:.2f}" for val in mean_s_list)
            avail_str = ", ".join(str(val) for val in avail_list)
            line = f"{t_sec:.3f}, {frame_idx}, {mean_s_str}, {avail_str}"
            print(line)
            if csv_file is not None:
                csv_line = line.replace(", ", ",") + "\n"
                csv_file.write(csv_line)

            processed += 1
            frame_idx += args.stride
    finally:
        if csv_file is not None:
            csv_file.close()
        src.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
