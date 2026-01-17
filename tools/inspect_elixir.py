import argparse
import csv
import os
import sys

import cv2

from state2action_rt.elixir_features import estimate_elixir_from_frame
from state2action_rt.frame_source import VideoFrameSource


def warn(msg: str) -> None:
    print(f"[warn] {msg}", file=sys.stderr)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect elixir bar estimation from video frames.")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--out-dir", default="out/elixir_debug", help="Output directory")
    parser.add_argument("--stride", type=positive_int, default=15, help="Process every N frames")
    parser.add_argument("--start-sec", type=float, default=0.0, help="Start time in seconds")
    parser.add_argument("--end-sec", type=float, default=-1.0, help="End time in seconds (-1 for end)")
    parser.add_argument(
        "--max-frames",
        type=positive_int,
        default=200,
        help="Maximum number of processed frames",
    )
    parser.add_argument("--elixir-roi-x1", type=int, default=None, help="Elixir ROI x1 in pixels")
    parser.add_argument("--elixir-roi-x2", type=int, default=None, help="Elixir ROI x2 in pixels")
    parser.add_argument("--elixir-roi-y1", type=int, default=None, help="Elixir ROI y1 in pixels")
    parser.add_argument("--elixir-roi-y2", type=int, default=None, help="Elixir ROI y2 in pixels")
    parser.add_argument("--purple-h-min", type=int, default=120, help="Purple hue min (HSV)")
    parser.add_argument("--purple-h-max", type=int, default=170, help="Purple hue max (HSV)")
    parser.add_argument("--purple-s-min", type=int, default=60, help="Purple saturation min (HSV)")
    parser.add_argument("--purple-v-min", type=int, default=40, help="Purple value min (HSV)")
    parser.add_argument(
        "--col-fill-ratio-th",
        type=float,
        default=0.35,
        help="Column purple ratio threshold",
    )
    parser.add_argument(
        "--min-purple-ratio",
        type=float,
        default=0.01,
        help="Minimum purple ratio for stability",
    )
    parser.add_argument(
        "--max-holes-ratio",
        type=float,
        default=0.6,
        help="Maximum holes ratio for stability",
    )
    parser.add_argument(
        "--allow-empty",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow empty elixir bar detection",
    )
    parser.add_argument(
        "--empty-purple-ratio-max",
        type=float,
        default=0.002,
        help="Max purple ratio to treat as empty",
    )
    parser.add_argument(
        "--empty-mean-s-max",
        type=float,
        default=80.0,
        help="Max mean saturation to treat as empty",
    )
    parser.add_argument(
        "--save-full",
        action="store_true",
        help="Save full frame images in addition to ROI/overlay",
    )
    parser.add_argument(
        "--debug-roi-overlay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save full-frame ROI overlay",
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
            csv_file = open(summary_path, "a", encoding="utf-8", newline="")
            csv_writer = csv.writer(csv_file)
            if is_new:
                csv_writer.writerow(
                    [
                        "frame_idx",
                        "time_sec",
                        "elixir",
                        "elixir_frac",
                        "purple_ratio",
                        "holes_ratio",
                        "filled_cols",
                        "fill_ratio",
                    ]
                )
        else:
            csv_writer = None

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

            elixir_roi_pixels = (
                args.elixir_roi_x1,
                args.elixir_roi_x2,
                args.elixir_roi_y1,
                args.elixir_roi_y2,
            )
            try:
                metrics = estimate_elixir_from_frame(
                    frame,
                    elixir_roi_pixels=elixir_roi_pixels,
                    purple_h_min=args.purple_h_min,
                    purple_h_max=args.purple_h_max,
                    purple_s_min=args.purple_s_min,
                    purple_v_min=args.purple_v_min,
                    col_fill_ratio_th=args.col_fill_ratio_th,
                    min_purple_ratio=args.min_purple_ratio,
                    max_holes_ratio=args.max_holes_ratio,
                    allow_empty=args.allow_empty,
                    empty_purple_ratio_max=args.empty_purple_ratio_max,
                    empty_mean_s_max=args.empty_mean_s_max,
                )
            except ValueError as exc:
                warn(f"elixir estimation failed: {exc}")
                return 1

            x1, y1, x2, y2 = metrics["roi"]
            roi_bgr = metrics["roi_bgr"]
            elixir = int(metrics["elixir"])
            elixir_frac = float(metrics["elixir_frac"])
            purple_ratio = float(metrics["purple_ratio"])
            holes_ratio = float(metrics["holes_ratio"])
            filled_cols = int(metrics["filled_cols"])
            fill_ratio = float(metrics["fill_ratio"])

            roi_path = os.path.join(args.out_dir, f"frame_{frame_idx:06d}_elixir_roi.png")
            cv2.imwrite(roi_path, roi_bgr)

            full_overlay = frame.copy() if args.debug_roi_overlay else None
            if full_overlay is not None:
                cv2.rectangle(full_overlay, (x1, y1), (x2 - 1, y2 - 1), (200, 0, 200), 2)
                overlay_path = os.path.join(
                    args.out_dir, f"frame_{frame_idx:06d}_full_roi_overlay.png"
                )
                cv2.imwrite(overlay_path, full_overlay)

            if args.save_full:
                full_path = os.path.join(args.out_dir, f"frame_{frame_idx:06d}_full.png")
                cv2.imwrite(full_path, frame)

            t_sec = frame_idx / fps
            print(
                f"{frame_idx} t={t_sec:.3f} elixir={elixir} elixir_frac={elixir_frac:.2f} "
                f"purple_ratio={purple_ratio:.4f} holes_ratio={holes_ratio:.3f} "
                f"filled_cols={filled_cols} fill_ratio={fill_ratio:.3f}"
            )
            if csv_writer is not None:
                csv_writer.writerow(
                    [
                        frame_idx,
                        f"{t_sec:.3f}",
                        elixir,
                        f"{elixir_frac:.3f}",
                        f"{purple_ratio:.6f}",
                        f"{holes_ratio:.6f}",
                        filled_cols,
                        f"{fill_ratio:.6f}",
                    ]
                )

            processed += 1
            frame_idx += args.stride
    finally:
        if csv_file is not None:
            csv_file.close()
        src.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
