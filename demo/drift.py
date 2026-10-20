"""Base drift per stand-up phase for the demo batches: xy displacement and heading change of base_Link between
phase boundaries (spawn = the settled heap at [기립], route = seated route P0-P3 + HOLD, learned = stand-up policy
until the hand-off, base = the first BASE_SECONDS of Base with a zero command)."""
from __future__ import annotations

import math

import numpy as np

BASE_SECONDS = 3.
# controller standup_phase -> segment that starts there
SEGMENT_START = {0: "route", 5: "learned", 6: "base"}


def heading(qpos: np.ndarray) -> float:
    w, x, y, z = (float(v) for v in qpos[3:7])
    return math.atan2(2. * (w * z + x * y), 1. - 2. * (y * y + z * z))


def pose(qpos: np.ndarray) -> tuple[float, float, float]:
    return float(qpos[0]), float(qpos[1]), heading(qpos)


def delta(a: tuple[float, float, float], b: tuple[float, float, float]) -> dict[str, float]:
    """xy displacement (m), its components in the start heading frame (forward, left) and heading change (deg)."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    c, s = math.cos(a[2]), math.sin(a[2])
    turn = (b[2] - a[2] + math.pi) % (2 * math.pi) - math.pi
    return {"xy": round(math.hypot(dx, dy), 4), "fwd": round(c * dx + s * dy, 4), "left": round(-s * dx + c * dy, 4),
            "yaw_deg": round(math.degrees(turn), 2)}


class DriftTracker:
    """Feed (phase, qpos) samples; keeps the pose at the start of every segment."""

    def __init__(self, qpos: np.ndarray) -> None:
        self.marks: dict[str, tuple[float, float, float]] = {"spawn": pose(qpos)}
        self.track: list[tuple[float, float, float]] = []

    def sample(self, phase: int, qpos: np.ndarray, elapsed: float) -> None:
        segment = SEGMENT_START.get(phase)
        if segment is not None and segment not in self.marks:
            self.marks[segment] = pose(qpos)
        self.marks.setdefault(f"phase{phase}", pose(qpos))
        if phase >= 0:
            self.track.append(pose(qpos)[:2] + (elapsed,))

    def end_base(self, qpos: np.ndarray) -> None:
        self.marks.setdefault("base_end", pose(qpos))

    def summary(self) -> dict[str, dict[str, float]]:
        order = [k for k in ("spawn", "learned", "base", "base_end") if k in self.marks]
        out = {}
        names = {("spawn", "learned"): "route", ("learned", "base"): "learned", ("base", "base_end"): "base",
                 ("spawn", "base"): "spawn_to_handoff"}
        for a, b in [*zip(order, order[1:]), ("spawn", "base")]:
            if a in self.marks and b in self.marks and (a, b) in names:
                out[names[(a, b)]] = delta(self.marks[a], self.marks[b])
        phases = sorted(int(k[5:]) for k in self.marks if k.startswith("phase"))
        out["phases"] = {f"{a}-{b}": delta(self.marks[f"phase{a}"], self.marks[f"phase{b}"])
                         for a, b in zip(phases, phases[1:])}
        return out
