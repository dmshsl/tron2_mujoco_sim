"""Window video capture for --selftest: grabs the whole main window at a fixed rate, encodes on a thread."""
from __future__ import annotations

import queue
import threading
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QWidget


class WindowRecorder:
    def __init__(self, window: QWidget, path: Path, fps: int = 30) -> None:
        self.window, self.path, self.fps = window, Path(path), fps
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.frames: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=240)
        self.size: tuple[int, int] | None = None
        self.count = self.dropped = 0
        self.timer = QTimer(window, timeout=self._grab)
        self.thread = threading.Thread(target=self._encode, daemon=True)
        self.snapshots: list[tuple[str, Path]] = []

    def start(self) -> None:
        self.thread.start()
        self.timer.start(int(1000 / self.fps))

    def _image(self) -> np.ndarray:
        image = self.window.grab().toImage().convertToFormat(QImage.Format.Format_RGB888)
        width, height = image.width(), image.height()
        array = np.frombuffer(image.constBits(), dtype=np.uint8, count=image.bytesPerLine() * height)
        return array.reshape(height, image.bytesPerLine())[:, :3 * width].reshape(height, width, 3).copy()

    def _grab(self) -> None:
        frame = self._image()
        if self.size is None:
            self.size = (frame.shape[1] // 2 * 2, frame.shape[0] // 2 * 2)
        width, height = self.size
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        h, w = min(height, frame.shape[0]), min(width, frame.shape[1])
        canvas[:h, :w] = frame[:h, :w]
        try:
            self.frames.put_nowait(canvas)
            self.count += 1
        except queue.Full:
            self.dropped += 1

    def snapshot(self, name: str) -> Path:
        path = self.path.with_name(f"{self.path.stem}_{name}.png")
        imageio.imwrite(path, self._image())
        self.snapshots.append((name, path))
        return path

    def _encode(self) -> None:
        with imageio.get_writer(str(self.path), fps=self.fps, codec="libx264", quality=7, macro_block_size=1,
                                ffmpeg_params=["-pix_fmt", "yuv420p"]) as writer:
            while (frame := self.frames.get()) is not None:
                writer.append_data(frame)

    def stop(self) -> None:
        self.timer.stop()
        self.frames.put(None)
        self.thread.join(timeout=60)
