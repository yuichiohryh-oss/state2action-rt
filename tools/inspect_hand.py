"""Inspect hand slots by HSV saturation and template matching.

Updates:
- Expose hand ROI y1/y2 ratios via CLI.
- Dump full-frame overlays, slot ROIs, and match ROIs for debugging.
- Pixel ROI overrides ratios/margins when all four coordinates are provided.
"""

import argparse
import csv
import os
import sys

import cv2
import numpy as np

from state2action_rt.frame_source import VideoFrameSource
from state2action_rt.hand.hand_reader import (
    HandDetectConfig,
    detect_hand_state,
    resolve_hand_roi_pixels,
    summarize_hand_state,
)
from state2action_rt.hand_features import load_hand_templates


def warn(msg: str) -> None:
    print(f"[warn] {msg}", file=sys.stderr)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def iter_slot_rects(hand_bgr: np.ndarray, n_slots: int) -> list[tuple[int, int, int, int]]:
    h, w = hand_bgr.shape[:2]
    rects: list[tuple[int, int, int, int]] = []
    for i in range(n_slots):
        x1 = int(round(i * w / n_slots))
        x2 = int(round((i + 1) * w / n_slots))
        if x2 <= x1:
            x2 = min(w, x1 + 1)
        rects.append((x1, 0, x2, h))
    return rects


def draw_text(
    img: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]
) -> None:
    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        color,
        1,
        cv2.LINE_AA,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect hand slots by HSV saturation and template matching.",
        epilog=(
            "Example (720x1604 screenshot): --hand-roi-x1 150 --hand-roi-x2 710 "
            "--hand-roi-y1 1360 --hand-roi-y2 1604"
        ),
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
    parser.add_argument(
        "--hand-y1-ratio",
        type=float,
        default=0.90,
        help="Hand ROI y1 ratio (ignored when pixel override is set)",
    )
    parser.add_argument(
        "--hand-y2-ratio",
        type=float,
        default=0.97,
        help="Hand ROI y2 ratio (ignored when pixel override is set)",
    )
    parser.add_argument("--s-th", type=float, default=30.0, help="Saturation threshold")
    parser.add_argument(
        "--hand-templates-dir",
        default="templates/hand_cards",
        help="Directory with hand card templates",
    )
    parser.add_argument(
        "--hand-card-min-score",
        type=float,
        default=0.60,
        help="Minimum score to accept template match",
    )
    parser.add_argument(
        "--hand-template-size",
        type=positive_int,
        default=64,
        help="Resize templates and slots to square size",
    )
    parser.add_argument(
        "--save-full",
        action="store_true",
        help="Save full frame images in addition to hand ROI/slots",
    )
    parser.add_argument(
        "--debug-roi-overlay",
        action="store_true",
        help="Save full-frame ROI overlay with hand/slot boxes",
    )
    parser.add_argument(
        "--hand-roi-x1",
        type=int,
        default=None,
        help="Hand ROI x1 in pixels (overrides ratios when all four are set)",
    )
    parser.add_argument(
        "--hand-roi-x2",
        type=int,
        default=None,
        help="Hand ROI x2 in pixels (overrides ratios when all four are set)",
    )
    parser.add_argument(
        "--hand-roi-y1",
        type=int,
        default=None,
        help="Hand ROI y1 in pixels (overrides ratios when all four are set)",
    )
    parser.add_argument(
        "--hand-roi-y2",
        type=int,
        default=None,
        help="Hand ROI y2 in pixels (overrides ratios when all four are set)",
    )
    parser.add_argument(
        "--write-csv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write summary.csv to out-dir",
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    templates = load_hand_templates(args.hand_templates_dir, args.hand_template_size)
    if not templates:
        warn(f"no hand templates loaded from {args.hand_templates_dir}")

    hand_roi_pixels = (
        args.hand_roi_x1,
        args.hand_roi_x2,
        args.hand_roi_y1,
        args.hand_roi_y2,
    )
    try:
        resolved_hand_roi_pixels = resolve_hand_roi_pixels(hand_roi_pixels)
    except ValueError as exc:
        warn(f"invalid hand ROI: {exc}")
        return 1

    hand_cfg = HandDetectConfig(
        s_th=args.s_th,
        card_min_score=args.hand_card_min_score,
        y1_ratio=args.hand_y1_ratio,
        y2_ratio=args.hand_y2_ratio,
        x_margin_ratio=args.x_margin_ratio,
        n_slots=4,
        template_size=args.hand_template_size,
        hand_roi_pixels=resolved_hand_roi_pixels,
        templates=templates,
    )

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
                    ["frame_idx", "time_sec", "slot", "mean_s", "available", "card_id", "score"]
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

            try:
                state, (hand_x1, hand_y1, hand_x2, hand_y2) = detect_hand_state(
                    frame, hand_cfg
                )
            except ValueError as exc:
                warn(f"invalid hand ROI: {exc}")
                return 1
            hand_bgr = state["hand_roi"]
            slots = state["slot_rois"]
            match_rois = state["match_rois"]
            mean_s_list = state["mean_s"]
            available_list, card_ids, scores = summarize_hand_state(state, hand_cfg)

            prefix = os.path.join(args.out_dir, f"frame_{frame_idx:06d}")
            hand_path = f"{prefix}_hand.png"
            cv2.imwrite(hand_path, hand_bgr)
            overlay = hand_bgr.copy()
            full_overlay = frame.copy() if args.debug_roi_overlay else None
            rects = iter_slot_rects(hand_bgr, n_slots=4)
            if full_overlay is not None:
                cv2.rectangle(
                    full_overlay,
                    (hand_x1, hand_y1),
                    (hand_x2 - 1, hand_y2 - 1),
                    (0, 200, 200),
                    2,
                )
            for i, slot in enumerate(slots):
                slot_path = f"{prefix}_slot{i}.png"
                cv2.imwrite(slot_path, slot)
                match_roi = match_rois[i]
                match_path = f"{prefix}_match{i}.png"
                cv2.imwrite(match_path, match_roi)
                mean_s = mean_s_list[i]
                available = bool(available_list[i])
                card_id = card_ids[i]
                score = scores[i]

                x1, y1, x2, y2 = rects[i]
                color = (0, 200, 0) if available else (40, 40, 200)
                cv2.rectangle(overlay, (x1, y1), (x2 - 1, y2 - 1), color, 1)
                if full_overlay is not None:
                    fx1 = hand_x1 + x1
                    fy1 = hand_y1 + y1
                    fx2 = hand_x1 + x2
                    fy2 = hand_y1 + y2
                    cv2.rectangle(full_overlay, (fx1, fy1), (fx2 - 1, fy2 - 1), color, 1)
                base_x = x1 + 3
                base_y = y1 + 12
                line_h = 12
                draw_text(overlay, f"slot {i}", base_x, base_y, color)
                draw_text(
                    overlay,
                    f"available {int(available)}",
                    base_x,
                    base_y + line_h,
                    color,
                )
                draw_text(overlay, f"card_id {card_id}", base_x, base_y + line_h * 2, color)
                draw_text(overlay, f"score {score:.2f}", base_x, base_y + line_h * 3, color)
                draw_text(overlay, f"mean_s {mean_s:.2f}", base_x, base_y + line_h * 4, color)

                t_sec = frame_idx / fps
                print(
                    f"{frame_idx} t={t_sec:.3f} slot={i} mean_s={mean_s:.2f} "
                    f"avail={int(available)} card_id={card_id} score={score:.3f}"
                )
                if csv_writer is not None:
                    csv_writer.writerow(
                        [
                            frame_idx,
                            f"{t_sec:.3f}",
                            i,
                            f"{mean_s:.3f}",
                            int(available),
                            card_id,
                            f"{score:.4f}",
                        ]
                    )
            if args.save_full:
                full_path = f"{prefix}_full.png"
                cv2.imwrite(full_path, frame)
            if full_overlay is not None:
                full_overlay_path = f"{prefix}_full_overlay.png"
                cv2.imwrite(full_overlay_path, full_overlay)

            overlay_path = f"{prefix}_hand_overlay.png"
            cv2.imwrite(overlay_path, overlay)

            processed += 1
            frame_idx += args.stride
    finally:
        if csv_file is not None:
            csv_file.close()
        src.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
