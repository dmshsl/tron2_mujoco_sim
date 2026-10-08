"""Live D455 depth rendering thread + file IPC feeding the real deploy perception
pipeline (tron2_deploy.perception.odometry + height_map) for sim2sim's
`--height_scan perception` mode.

Wire format: two fixed-size camera records (front then rear), each little-endian
float64 wall-monotonic capture time, float64 simulation capture time (diagnostic
only; the DepthFrame timestamp handed to HeightMap is the wall time, matching
the SDK loopback's own `time.monotonic()` clock used by SdkState.timestamp),
then 480*848 little-endian uint16 millimetre depth samples (row-major, matching
DepthFrame's expected (480, 848) shape). Millimetre 0 means "no valid return",
mirroring a real sensor's invalid-pixel convention. Both cameras share one
forward-kinematics snapshot, as hardware-synced D455s would.

Rendering runs on its own thread with its own 30 Hz-target cadence and never
blocks physics: it free-runs against its own target period and reports the
achieved rate rather than catching up (see PerceptionRenderer.achieved_fps).
"""
from __future__ import annotations

import fcntl
import os
import struct
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Final

import mujoco
import numpy as np
from numpy.typing import NDArray

from tron2_sim.rendering import OffscreenRenderer

if TYPE_CHECKING:
    from tron2_deploy.height_scan_provider import PerceptionHeightScanProvider
    from tron2_deploy.perception.height_map import DepthFrame
    from tron2_deploy.policy_contract import DeploymentContract

DEFAULT_PATH: Final = Path("/tmp/tron2-height-scan-perception.bin")
FRAME_SHAPE: Final = (480, 848)
FRAME_PIXELS: Final = FRAME_SHAPE[0] * FRAME_SHAPE[1]
HEADER: Final = struct.Struct("<dd")
RECORD_BYTES: Final = HEADER.size + FRAME_PIXELS * 2
CAMERAS: Final = ("front", "rear")
_MUJOCO_CAMERA_NAME: Final = {"front": "d455_front", "rear": "d455_rear"}


class SharedPose:
    """Thread-safe qpos/time snapshot handoff; the lock is held only for a memcpy."""

    def __init__(self, nq: int) -> None:
        self._lock = threading.Lock()
        self._qpos = np.zeros(nq, dtype=np.float64)
        self._time = 0.0

    def publish(self, qpos: NDArray[np.float64], sim_time: float) -> None:
        with self._lock:
            self._qpos[:] = qpos
            self._time = sim_time

    def snapshot(self) -> tuple[NDArray[np.float64], float]:
        with self._lock:
            return self._qpos.copy(), self._time


def _depth_to_mm(depth: NDArray[np.float32]) -> NDArray[np.uint16]:
    """Clamp to a real sensor's representable band; invalid/out-of-range -> 0."""
    finite = np.isfinite(depth) & (depth >= 0.) & (depth < 65.)
    return np.where(finite, depth * 1000., 0.).astype(np.uint16)


def publish_frames(path: Path, wall_t: float, sim_t: float,
                    front: NDArray[np.float32], rear: NDArray[np.float32]) -> None:
    if front.shape != FRAME_SHAPE or rear.shape != FRAME_SHAPE:
        raise ValueError(f"depth frames must be {FRAME_SHAPE}")
    header = HEADER.pack(wall_t, sim_t)
    fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "r+b", buffering=0) as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(header + np.ascontiguousarray(_depth_to_mm(front)).tobytes())
        stream.write(header + np.ascontiguousarray(_depth_to_mm(rear)).tobytes())
        stream.truncate()


def read_frames(path: Path) -> tuple[DepthFrame, DepthFrame]:
    from tron2_deploy.perception.height_map import DepthFrame

    with path.open("rb") as stream:
        fcntl.flock(stream, fcntl.LOCK_SH)
        data = stream.read()
    if len(data) != 2 * RECORD_BYTES:
        raise ValueError(f"malformed perception frame file: {len(data)} bytes, expected {2 * RECORD_BYTES}")
    frames = []
    for camera, offset in zip(CAMERAS, (0, RECORD_BYTES), strict=True):
        wall_t, _sim_t = HEADER.unpack_from(data, offset)
        depth = np.frombuffer(data, dtype="<u2", count=FRAME_PIXELS,
                              offset=offset + HEADER.size).reshape(FRAME_SHAPE).copy()
        frames.append(DepthFrame(camera, depth, wall_t))
    return tuple(frames)


class PerceptionRenderer:
    """Owns a private MjData + OffscreenRenderer; free-runs at its own cadence.

    `warm_up()` performs one synchronous render before the thread starts so the
    one-time EGL/shader cold-start cost (observed ~135ms) lands during setup,
    never as a stall inside the real-time loop.
    """

    def __init__(self, model: mujoco.MjModel, shared_pose: SharedPose, scan_path: Path,
                target_fps: float = 30.0) -> None:
        self.model = model
        self.shared_pose = shared_pose
        self.scan_path = scan_path
        self.period = 1.0 / target_fps
        self.rendered = 0
        self.skipped_ticks = 0
        self._stop = threading.Event()
        self._started_wall: float | None = None
        self._thread = threading.Thread(target=self._run, name="perception-renderer", daemon=True)

    def warm_up(self) -> None:
        """Render one frame synchronously (cold EGL/shader cost) before starting."""
        data = mujoco.MjData(self.model)
        qpos, _ = self.shared_pose.snapshot()
        data.qpos[:] = qpos
        mujoco.mj_forward(self.model, data)
        with OffscreenRenderer(self.model) as renderer:
            for camera in _MUJOCO_CAMERA_NAME.values():
                renderer.depth(data, camera)

    def start(self) -> None:
        self._started_wall = time.monotonic()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)

    @property
    def achieved_fps(self) -> float:
        if self._started_wall is None:
            return 0.0
        elapsed = time.monotonic() - self._started_wall
        return self.rendered / elapsed if elapsed > 0 else 0.0

    def _run(self) -> None:
        data = mujoco.MjData(self.model)
        with OffscreenRenderer(self.model) as renderer:
            next_tick = time.monotonic()
            while not self._stop.is_set():
                qpos, sim_time = self.shared_pose.snapshot()
                data.qpos[:] = qpos
                mujoco.mj_forward(self.model, data)
                front = renderer.depth(data, _MUJOCO_CAMERA_NAME["front"])
                rear = renderer.depth(data, _MUJOCO_CAMERA_NAME["rear"])
                publish_frames(self.scan_path, time.monotonic(), sim_time, front, rear)
                self.rendered += 1
                next_tick += self.period
                remaining = next_tick - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
                else:
                    self.skipped_ticks += 1
                    next_tick = time.monotonic()


def _frame_source(path: Path, query_timestamp: float) -> tuple[DepthFrame, ...]:
    """Drop a frame the renderer timestamped microseconds after this query.

    Both processes read the same `time.monotonic()` clock, but the renderer
    thread and the controller's query sample independently, so a frame can
    occasionally land a fraction of a millisecond in this query's future.
    HeightMap.update() correctly rejects such a frame rather than fabricating
    an ordering; deferring it to the next (slightly later) query is correct
    and lossless, not a workaround for bad data.
    """
    return tuple(frame for frame in read_frames(path) if frame.timestamp <= query_timestamp)


def create_provider(contract: DeploymentContract, mode: str) -> PerceptionHeightScanProvider:
    from tron2_deploy.height_scan_provider import PerceptionHeightScanProvider

    if mode != "perception" or contract.policy != "base":
        raise ValueError("Perception factory requires mode=perception and the Base contract")
    path = Path(os.environ.get("TRON2_PERCEPTION_SCAN_PATH", str(DEFAULT_PATH)))
    return PerceptionHeightScanProvider(contract, lambda timestamp: _frame_source(path, timestamp))
