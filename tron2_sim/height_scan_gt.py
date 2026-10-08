"""Terrain-only GT and a NumPy-only deployment client for a locked local scan file.

Wire format: little-endian float64 monotonic wall timestamp, float64 simulation
time, then 231 float32 values (x fastest). Writers/readers use flock; stale or
missing data raises rather than silently supplying a valid-looking flat scan.
"""
from __future__ import annotations

import fcntl
import math
import os
import struct
import time
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import mujoco

DEFAULT_PATH: Final = Path("/tmp/tron2-height-scan-gt.bin")
PACKET: Final = struct.Struct("<dd231f")


class ScanState(Protocol):
    timestamp: float


class ScanContract(Protocol):
    policy: str


class HeightScanProvider:
    """Single physics-thread ray caster; validity is retained separately from -1."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model = model
        self.data = data
        self.base_id = model.body("base_Link").id
        x, y = np.meshgrid(np.linspace(-1, 1, 21), np.linspace(-.5, .5, 11), indexing="xy")
        self.offsets = np.column_stack((x.ravel(), y.ravel()))
        self.valid = np.zeros(231, dtype=np.bool_)

    def get_scan(self, *, flg_static: bool = True) -> NDArray[np.float32]:
        import mujoco

        base = self.data.xpos[self.base_id]
        rotation = self.data.xmat[self.base_id].reshape(3, 3)
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        c, s = math.cos(yaw), math.sin(yaw)
        xy = self.offsets @ np.array([[c, s], [-s, c]]) + base[:2]
        mask = np.array([0, 0, 0, 0, 0, 1], dtype=np.uint8)
        direction = np.array([0., 0., -1.])
        geom_id = np.array([-1], dtype=np.int32)
        values = np.full(231, -1., dtype=np.float32)
        self.valid[:] = False
        for index, point in enumerate(xy):
            start = np.array([point[0], point[1], base[2] + 20.])
            distance = mujoco.mj_ray(self.model, self.data, start, direction,
                                     mask, flg_static, -1, geom_id)
            if distance >= 0:
                self.valid[index] = True
                values[index] = np.clip(base[2] - (start[2] - distance) - .8, -1, 1)
        return values


def publish_scan(path: Path, sim_time: float, scan: NDArray[np.float32]) -> None:
    if scan.shape != (231,) or not np.isfinite(scan).all():
        raise ValueError("GT scan must be 231 finite float values")
    fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "r+b", buffering=0) as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(PACKET.pack(time.monotonic(), sim_time, *scan))
        stream.truncate()


class LocalHeightScanProvider:
    """Importable without MuJoCo; get_scan()/call returns a fresh (231,) float32 array."""

    def __init__(self, path: Path | None = None, max_age: float = .5) -> None:
        self.path = path or Path(os.environ.get("TRON2_GT_SCAN_PATH", str(DEFAULT_PATH)))
        self.max_age = max_age
        self.sim_time = 0.

    def get_scan(self, state: ScanState | None = None) -> NDArray[np.float32]:
        with self.path.open("rb") as stream:
            fcntl.flock(stream, fcntl.LOCK_SH)
            stamp, sim_time, *values = PACKET.unpack(stream.read())
        if time.monotonic() - stamp > self.max_age:
            raise TimeoutError(f"Stale GT scan: {self.path}")
        if state is not None and abs(state.timestamp - stamp) > self.max_age:
            raise TimeoutError("GT scan and SDK state's monotonic timestamps are too far apart")
        self.sim_time = sim_time
        return np.array(values, dtype=np.float32)

    def __call__(self) -> NDArray[np.float32]:
        return self.get_scan()


def create_provider(contract: ScanContract, mode: str) -> LocalHeightScanProvider:
    if mode != "gt" or contract.policy not in ("base", "base_blind"):
        raise ValueError("Simulator GT factory requires mode=gt and a Base deployment contract")
    return LocalHeightScanProvider()
