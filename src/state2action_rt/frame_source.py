from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import cv2
import numpy as np


@dataclass
class FrameSource:
    fps: float
    get_frame: Callable[[int], Optional[np.ndarray]]
    close: Callable[[], None] | None = None


class VideoFrameSource:
    def __init__(self, path: str, fallback_fps: float | None = None):
        self.cap = cv2.VideoCapture(path)
        fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        if fps <= 1e-3:
            if fallback_fps is None:
                self.cap.release()
                raise ValueError("video input requires --fps when fps is unavailable")
            fps = fallback_fps
        self.fps = fps

    def get_frame(self, index: int) -> Optional[np.ndarray]:
        if index < 0:
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame

    def close(self) -> None:
        self.cap.release()


class ImageDirFrameSource:
    def __init__(self, file_paths: list[str], fps: float):
        self.file_paths = file_paths
        self.fps = fps

    def get_frame(self, index: int) -> Optional[np.ndarray]:
        if index < 0 or index >= len(self.file_paths):
            return None
        return cv2.imread(self.file_paths[index], cv2.IMREAD_COLOR)


def list_image_files(dir_path: str, exts: Iterable[str]) -> list[str]:
    root = Path(dir_path)
    files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort(key=lambda p: p.name)
    return [str(p) for p in files]
