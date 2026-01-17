import argparse
import os
import sys

from state2action_rt.dataset_core import GridConfig, build_dataset
from state2action_rt.events import parse_events_jsonl
from state2action_rt.frame_source import (
    FrameSource,
    ImageDirFrameSource,
    VideoFrameSource,
    list_image_files,
)
from state2action_rt.roi import RoiConfig


def warn(msg: str) -> None:
    print(f"[warn] {msg}", file=sys.stderr)


def build_frame_source(video_path: str, fps: float | None) -> FrameSource:
    if os.path.isdir(video_path):
        exts = {".png", ".jpg", ".jpeg", ".bmp"}
        files = list_image_files(video_path, exts)
        if not files:
            raise ValueError("no images found in directory")
        if fps is None:
            raise ValueError("image directory input requires --fps")
        src = ImageDirFrameSource(files, fps)
        return FrameSource(fps=src.fps, get_frame=src.get_frame)

    src = VideoFrameSource(video_path, fallback_fps=fps)
    return FrameSource(fps=src.fps, get_frame=src.get_frame, close=src.close)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a state/action dataset from video and events.")
    parser.add_argument("--video", required=True, help="Path to video file or image directory")
    parser.add_argument("--events", required=True, help="Path to events.jsonl")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--lead-sec", type=float, default=0.8, help="State lead time in seconds")
    parser.add_argument("--gw", type=positive_int, default=6, help="Grid width (>=1)")
    parser.add_argument("--gh", type=positive_int, default=9, help="Grid height (>=1)")
    parser.add_argument("--roi-y1", type=int, default=70, help="ROI top offset")
    parser.add_argument(
        "--roi-y2-mode",
        choices=["auto", "fixed"],
        default="auto",
        help="ROI bottom detection mode",
    )
    parser.add_argument("--roi-y2-fixed", type=int, default=None, help="Fixed ROI bottom (y2)")
    parser.add_argument("--roi-x1", type=int, default=0, help="ROI left offset")
    parser.add_argument("--roi-x2", type=int, default=None, help="ROI right bound")
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="FPS override/fallback (required for image directory input)",
    )
    parser.add_argument("--hand-s-th", type=int, default=30, help="Hand HSV S threshold")
    parser.add_argument(
        "--hand-y1-ratio",
        type=float,
        default=0.90,
        help="Hand ROI top ratio (relative to frame height)",
    )
    parser.add_argument(
        "--hand-y2-ratio",
        type=float,
        default=0.97,
        help="Hand ROI bottom ratio (relative to frame height)",
    )
    parser.add_argument(
        "--hand-x-margin-ratio",
        type=float,
        default=0.03,
        help="Hand ROI horizontal margin ratio (relative to frame width)",
    )
    parser.add_argument(
        "--hand-templates-dir",
        default=None,
        help="Optional directory of hand card templates (enables hand_card_ids)",
    )
    parser.add_argument("--hand-card-min-score", type=float, default=0.6, help="Min template score for card id")
    parser.add_argument("--hand-roi-x1", type=int, default=None, help="Hand ROI x1 in pixels")
    parser.add_argument("--hand-roi-x2", type=int, default=None, help="Hand ROI x2 in pixels")
    parser.add_argument("--hand-roi-y1", type=int, default=None, help="Hand ROI y1 in pixels")
    parser.add_argument("--hand-roi-y2", type=int, default=None, help="Hand ROI y2 in pixels")
    parser.add_argument("--elixir-roi-x1", type=int, default=None, help="Elixir ROI x1 in pixels")
    parser.add_argument("--elixir-roi-x2", type=int, default=None, help="Elixir ROI x2 in pixels")
    parser.add_argument("--elixir-roi-y1", type=int, default=None, help="Elixir ROI y1 in pixels")
    parser.add_argument("--elixir-roi-y2", type=int, default=None, help="Elixir ROI y2 in pixels")
    parser.add_argument(
        "--elixir-purple-h-min",
        type=int,
        default=120,
        help="Elixir purple hue min (HSV)",
    )
    parser.add_argument(
        "--elixir-purple-h-max",
        type=int,
        default=170,
        help="Elixir purple hue max (HSV)",
    )
    parser.add_argument(
        "--elixir-purple-s-min",
        type=int,
        default=60,
        help="Elixir purple saturation min (HSV)",
    )
    parser.add_argument(
        "--elixir-purple-v-min",
        type=int,
        default=40,
        help="Elixir purple value min (HSV)",
    )
    parser.add_argument(
        "--elixir-col-fill-ratio-th",
        type=float,
        default=0.35,
        help="Elixir column purple ratio threshold",
    )
    parser.add_argument(
        "--elixir-min-purple-ratio",
        type=float,
        default=0.01,
        help="Elixir minimum purple ratio",
    )
    parser.add_argument(
        "--elixir-max-holes-ratio",
        type=float,
        default=0.6,
        help="Elixir max holes ratio before unstable",
    )
    parser.add_argument(
        "--elixir-allow-empty",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow empty elixir bar detection",
    )
    parser.add_argument(
        "--elixir-empty-purple-ratio-max",
        type=float,
        default=0.002,
        help="Elixir max purple ratio to treat as empty",
    )
    parser.add_argument(
        "--elixir-empty-mean-s-max",
        type=float,
        default=80.0,
        help="Elixir max mean saturation to treat as empty",
    )

    args = parser.parse_args()

    events = parse_events_jsonl(args.events, warn)
    if not events:
        warn("no valid events found")

    try:
        frame_source = build_frame_source(args.video, args.fps)
    except Exception as exc:
        warn(f"failed to open video source: {exc}")
        return 1

    roi_config = RoiConfig(
        y1=args.roi_y1,
        y2_mode=args.roi_y2_mode,
        y2_fixed=args.roi_y2_fixed,
        x1=args.roi_x1,
        x2=args.roi_x2,
    )
    grid_config = GridConfig(gw=args.gw, gh=args.gh)

    try:
        build_dataset(
            events=events,
            frame_source=frame_source,
            out_dir=args.out_dir,
            roi_config=roi_config,
            grid_config=grid_config,
            lead_sec=args.lead_sec,
            warn_fn=warn,
            hand_s_th=float(args.hand_s_th),
            hand_y1_ratio=args.hand_y1_ratio,
            hand_y2_ratio=args.hand_y2_ratio,
            hand_x_margin_ratio=args.hand_x_margin_ratio,
            hand_templates_dir=args.hand_templates_dir,
            hand_card_min_score=args.hand_card_min_score,
            hand_roi_pixels=(
                args.hand_roi_x1,
                args.hand_roi_x2,
                args.hand_roi_y1,
                args.hand_roi_y2,
            ),
            elixir_roi_pixels=(
                args.elixir_roi_x1,
                args.elixir_roi_x2,
                args.elixir_roi_y1,
                args.elixir_roi_y2,
            ),
            elixir_purple_h_min=args.elixir_purple_h_min,
            elixir_purple_h_max=args.elixir_purple_h_max,
            elixir_purple_s_min=args.elixir_purple_s_min,
            elixir_purple_v_min=args.elixir_purple_v_min,
            elixir_col_fill_ratio_th=args.elixir_col_fill_ratio_th,
            elixir_min_purple_ratio=args.elixir_min_purple_ratio,
            elixir_max_holes_ratio=args.elixir_max_holes_ratio,
            elixir_allow_empty=args.elixir_allow_empty,
            elixir_empty_purple_ratio_max=args.elixir_empty_purple_ratio_max,
            elixir_empty_mean_s_max=args.elixir_empty_mean_s_max,
        )
    finally:
        if frame_source.close:
            frame_source.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
