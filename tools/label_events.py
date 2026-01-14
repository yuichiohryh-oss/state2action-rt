import argparse
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

import cv2

from state2action_rt.label_events_utils import (
    append_event,
    apply_roi_offset,
    count_events,
    undo_last_event,
)
from state2action_rt.roi import RoiConfig, detect_roi


def warn(msg: str) -> None:
    print(f"[warn] {msg}", file=sys.stderr)


def info(msg: str) -> None:
    print(f"[info] {msg}")


def beep() -> None:
    print("\a", end="", flush=True)


def print_help() -> None:
    info("Keys: Space play/pause | A/D -1s/+1s | J/L -0.1s/+0.1s | 0-9 save action")
    info("      N save __NOOP__ | U undo last | H help | Q/ESC quit")
    info("Mouse: Left click to select point")


def resolve_fps(cap: cv2.VideoCapture, fallback_fps: Optional[float]) -> float:
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0.0 or fps != fps:
        if fallback_fps is None:
            raise ValueError("fps unavailable; provide --fps")
        return float(fallback_fps)
    return fps


def format_overlay(
    t_sec: float, frame_idx: int, paused: bool, action_hint: str
) -> Tuple[str, str]:
    state = "paused" if paused else "playing"
    hint = action_hint if action_hint else "-"
    return (f"t={t_sec:.2f}s frame={frame_idx} {state}", f"action_hint={hint}")


def draw_text(image, lines: Tuple[str, str]) -> None:
    for i, line in enumerate(lines):
        y = 20 + i * 20
        cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


def seek_and_read(
    cap: cv2.VideoCapture, t_sec: float
) -> Tuple[bool, Optional[Tuple[float, int]], Optional[Any]]:
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t_sec) * 1000.0)
    ok, frame = cap.read()
    if not ok:
        return False, None, None
    t_now = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
    frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
    if frame_idx < 0:
        frame_idx = 0
    return True, (t_now, frame_idx), frame


def read_next(cap: cv2.VideoCapture) -> Tuple[bool, Optional[Tuple[float, int]], Optional[Any]]:
    ok, frame = cap.read()
    if not ok:
        return False, None, None
    t_now = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
    frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
    if frame_idx < 0:
        frame_idx = 0
    return True, (t_now, frame_idx), frame


def main() -> int:
    parser = argparse.ArgumentParser(description="Label events by clicking on a video frame.")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--out", required=True, help="Output events.jsonl path")
    parser.add_argument("--start-sec", type=float, default=0.0, help="Minimum start time in seconds")
    parser.add_argument("--seek-sec", type=float, default=None, help="Seek to time in seconds on startup")
    parser.add_argument("--fps", type=float, default=None, help="FPS fallback if video does not report it")
    parser.add_argument("--roi-y1", type=int, default=None, help="ROI top offset")
    parser.add_argument("--roi-y2-mode", choices=["auto", "fixed"], default=None, help="ROI bottom mode")
    parser.add_argument("--roi-y2-fixed", type=int, default=None, help="Fixed ROI bottom (y2)")
    parser.add_argument("--roi-x1", type=int, default=None, help="ROI left offset")
    parser.add_argument("--roi-x2", type=int, default=None, help="ROI right bound")
    args = parser.parse_args()

    video_path = Path(args.video)
    out_path = Path(args.out)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        warn(f"failed to open video: {video_path}")
        return 1

    try:
        fps = resolve_fps(cap, args.fps)
    except ValueError as exc:
        warn(str(exc))
        return 1

    min_t = max(0.0, float(args.start_sec))
    if args.seek_sec is not None:
        start_t = max(min_t, float(args.seek_sec))
    else:
        start_t = min_t
    ok, frame_state, frame = seek_and_read(cap, start_t)
    if not ok or frame_state is None or frame is None:
        warn("failed to read initial frame")
        return 1

    current_t, frame_idx = frame_state
    paused = False
    action_hint = ""

    roi_active = any(
        value is not None
        for value in (
            args.roi_y1,
            args.roi_y2_mode,
            args.roi_y2_fixed,
            args.roi_x1,
            args.roi_x2,
        )
    )
    roi: Optional[Tuple[int, int, int, int]] = None
    if roi_active:
        y2_mode = args.roi_y2_mode
        if y2_mode is None:
            y2_mode = "fixed" if args.roi_y2_fixed is not None else "auto"
        roi_config = RoiConfig(
            y1=args.roi_y1 if args.roi_y1 is not None else 0,
            y2_mode=y2_mode,
            y2_fixed=args.roi_y2_fixed,
            x1=args.roi_x1 if args.roi_x1 is not None else 0,
            x2=args.roi_x2,
        )
        roi = detect_roi(frame, roi_config)

    loaded = count_events(out_path)
    info(f"loaded {loaded} events from {out_path}")
    print_help()

    state = {"last_click_display": None, "last_click_full": None}

    def on_mouse(event, x, y, flags, userdata) -> None:
        del flags, userdata
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        state["last_click_display"] = (x, y)
        full_x, full_y = apply_roi_offset(float(x), float(y), roi)
        state["last_click_full"] = (full_x, full_y)

    window_name = "label_events"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        if not paused:
            ok, frame_state, next_frame = read_next(cap)
            if ok and frame_state is not None and next_frame is not None:
                current_t, frame_idx = frame_state
                frame = next_frame
            else:
                paused = True
                warn("reached end of video")

        display = frame
        if roi is not None:
            x1, y1, x2, y2 = roi
            display = frame[y1:y2, x1:x2]

        overlay = display.copy()
        draw_text(overlay, format_overlay(current_t, frame_idx, paused, action_hint))
        if state["last_click_display"] is not None:
            cx, cy = state["last_click_display"]
            cv2.drawMarker(overlay, (int(cx), int(cy)), (0, 0, 255), cv2.MARKER_CROSS, 12, 2)

        cv2.imshow(window_name, overlay)
        delay = max(1, int(1000.0 / fps)) if not paused else 30
        key = cv2.waitKey(delay) & 0xFF

        if key == 255:
            continue
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = not paused
            continue
        if key in (ord("h"), ord("H")):
            print_help()
            continue
        if key in (ord("a"), ord("A")):
            target = max(min_t, current_t - 1.0)
        elif key in (ord("d"), ord("D")):
            target = max(min_t, current_t + 1.0)
        elif key in (ord("j"), ord("J")):
            target = max(min_t, current_t - 0.1)
        elif key in (ord("l"), ord("L")):
            target = max(min_t, current_t + 0.1)
        else:
            target = None

        if target is not None:
            ok, frame_state, next_frame = seek_and_read(cap, target)
            if ok and frame_state is not None and next_frame is not None:
                current_t, frame_idx = frame_state
                frame = next_frame
            else:
                warn("seek failed")
            continue

        if ord("0") <= key <= ord("9"):
            if state["last_click_full"] is None:
                warn("click required before saving action")
                beep()
                continue
            action_id = f"action_{chr(key)}"
            action_hint = action_id
            x, y = state["last_click_full"]
            event = {
                "t": float(current_t),
                "x": float(x),
                "y": float(y),
                "action_id": action_id,
                "frame_idx": int(frame_idx),
                "fps_effective": float(fps),
            }
            append_event(out_path, event)
            info(f"saved {action_id} at t={current_t:.2f}")
            continue

        if key in (ord("n"), ord("N")):
            action_hint = "__NOOP__"
            event = {
                "t": float(current_t),
                "x": -1.0,
                "y": -1.0,
                "action_id": "__NOOP__",
                "frame_idx": int(frame_idx),
                "fps_effective": float(fps),
            }
            append_event(out_path, event)
            info("saved __NOOP__")
            continue

        if key in (ord("u"), ord("U")):
            removed = undo_last_event(out_path)
            if removed is None:
                warn("no event to undo")
            else:
                info("undid last event")
            continue

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
