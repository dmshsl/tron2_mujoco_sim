"""Sim -> GUI state block: one seqlocked record in a /dev/shm file, written by the sim server at ~60 Hz."""
from __future__ import annotations

import mmap
import os
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# sim_state
BOOT, SETTLING, HEAP, ACTIVE = 0, 1, 2, 3
# controller
NONE, STARTING, RUNNING, EXITED = 0, 1, 2, 3

_HEAD = struct.Struct("<QddiiiiidddiI")  # seq wall sim_t sim_state controller exit seed slow rtf vx wz spawn_id nq
_HEAD_SIZE = 128


@dataclass
class SimState:
    wall: float = 0.
    sim_time: float = 0.
    sim_state: int = BOOT
    controller: int = NONE
    exit_code: int = 0
    seed: int = -1
    slow: int = 0
    rtf: float = 0.
    vx: float = 0.
    wz: float = 0.
    spawn_id: int = 0
    qpos: np.ndarray | None = None


class StateBlock:
    def __init__(self, path: Path, nq: int = 0, create: bool = False) -> None:
        self.path = Path(path)
        if create:
            self.nq = nq
            size = _HEAD_SIZE + 8 * nq
            fd = os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
            os.ftruncate(fd, size)
        else:
            fd = os.open(self.path, os.O_RDWR | os.O_NOFOLLOW)
            size = os.fstat(fd).st_size
            self.nq = (size - _HEAD_SIZE) // 8
        try:
            self.map = mmap.mmap(fd, size)
        finally:
            os.close(fd)
        self.qpos = np.ndarray((self.nq,), dtype=np.float64, buffer=self.map, offset=_HEAD_SIZE)

    def write(self, state: SimState) -> None:
        odd = (struct.unpack_from("<Q", self.map, 0)[0] + 1) | 1
        struct.pack_into("<Q", self.map, 0, odd)
        _HEAD.pack_into(self.map, 0, odd, state.wall, state.sim_time, state.sim_state, state.controller,
                        state.exit_code, state.seed, state.slow, state.rtf, state.vx, state.wz, state.spawn_id,
                        self.nq)
        if state.qpos is not None:
            self.qpos[:] = state.qpos
        struct.pack_into("<Q", self.map, 0, odd + 1)

    def read(self) -> SimState | None:
        for _ in range(50):
            values = _HEAD.unpack_from(self.map, 0)
            qpos = self.qpos.copy()
            if values[0] % 2 == 0 and struct.unpack_from("<Q", self.map, 0)[0] == values[0]:
                return SimState(*values[1:12], qpos=qpos)
        return None

    def close(self) -> None:
        del self.qpos
        self.map.close()
