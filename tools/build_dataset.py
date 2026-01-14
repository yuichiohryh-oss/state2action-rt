import argparse
import os
import sys
from pathlib import Path

from state2action_rt.dataset_core import GridConfig, build_dataset
from state2action_rt.events import parse_events_jsonl
from state2action_rt.frame_source import FrameSource, ImageDirFrameSource, VideoFrameSource
from state2action_rt.roi import RoiConfig


def warn(msg: str) -> None:
    print(f"[warn] {msg}", file=sys.stderr)


def build_frame_source(video_path: str, fps: float | None) -> FrameSource:
    if os.path.isdir(video_path):
        exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
        files = []
        for ext in exts:
            files.extend([str(p) for p in sorted(Path(video_path).glob(ext))])
        if not files:
            raise ValueError("no images found in directory")
        if fps is None:
            warn("--fps not set for image directory, using 30.0")
            fps = 30.0
        src = ImageDirFrameSource(files, fps)
        return FrameSource(fps=src.fps, get_frame=src.get_frame)

    fallback_fps = fps if fps is not None else 30.0
    src = VideoFrameSource(video_path, fallback_fps=fallback_fps)
    return FrameSource(fps=src.fps, get_frame=src.get_frame, close=src.close)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a state/action dataset from video and events.")
    parser.add_argument("--video", required=True, help="Path to video file or image directory")
    parser.add_argument("--events", required=True, help="Path to events.jsonl")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--lead-sec", type=float, default=0.8, help="State lead time in seconds")
    parser.add_argument("--gw", type=int, default=6, help="Grid width")
    parser.add_argument("--gh", type=int, default=9, help="Grid height")
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
    parser.add_argument("--fps", type=float, default=None, help="FPS fallback (required for image dir)")

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
            events,
            frame_source,
            args.out_dir,
            roi_config,
            grid_config,
            args.lead_sec,
            warn,
        )
    finally:
        if frame_source.close:
            frame_source.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
