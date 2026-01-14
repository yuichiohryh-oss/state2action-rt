from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np


@dataclass
class FrameSource:
    fps: float
    get_frame: Callable[[int], Optional[np.ndarray]]
    close: Callable[[], None] | None = None


class VideoFrameSource:
    def __init__(self, path: str, fallback_fps: float = 30.0):
        self.cap = cv2.VideoCapture(path)
        fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        if fps <= 1e-3:
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
