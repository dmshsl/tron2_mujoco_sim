"""Tidy limp heaps for the demo: pose 1 (both legs laid out sideways, the way a person sets the robot
down neatly) plus a uniform random perturbation per joint, then a limp settle on the floor.

Pose 1 (left leg, deg) = (0, 169.8, -90, -142.05), the right leg mirrored (pitch, -roll, -yaw, knee);
it is in the roll family, the family whose end pose A' the learned stand-up was trained on.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import mujoco
import numpy as np
import numpy.typing as npt

COMMON = Path(__file__).resolve().parents[2] / "tron2_common"
POSE1_LEFT_DEG = (0., 169.8, -90., -142.05)
SPREAD_DEG = 35.  # +- per joint (the brief: 30-40 deg)
SETTLE_S = 1.
LIMP_DAMPING = .3  # N m s/rad, as tron2_sim.start_conditions.seated_start


def pose1() -> npt.NDArray[np.float64]:
    pitch, roll, yaw, knee = POSE1_LEFT_DEG
    return np.radians([pitch, roll, yaw, knee, pitch, -roll, -yaw, knee])


def tidy_pose(seed: int, spread_deg: float = SPREAD_DEG) -> npt.NDArray[np.float64]:
    """Pose 1 + U(-spread, spread) per joint, clamped into the 3 deg joint margin; draws whose links
    reach into the payload frame (clearance < 0.03 m) are redrawn, as the Isaac seated draw does."""
    if str(COMMON) not in sys.path:
        sys.path.insert(0, str(COMMON))
    from tron2_common.kinematics import LOWER, UPPER, payload_clearance

    rng = np.random.default_rng(seed)
    base = pose1()
    for _ in range(1000):
        q = np.clip(base + rng.uniform(-1., 1., 8) * math.radians(spread_deg), LOWER, UPPER)
        if payload_clearance(q) >= .03:
            return q
    raise ValueError(f"no tidy heap without payload contact for seed {seed}")


def place(model: mujoco.MjModel, data: mujoco.MjData, q: npt.NDArray[np.float64]) -> None:
    """Upright base at the FK support height (lowest body/wheel point 1 mm above the floor), legs at q,
    everything at rest. Uses tron2_sim.start_conditions.seated_start with no settle."""
    sim = Path(__file__).resolve().parents[1]
    if str(sim) not in sys.path:
        sys.path.insert(0, str(sim))
    from tron2_sim.start_conditions import seated_start

    seated_start(model, data, 0, settle_s=0., pose=q)


class LimpSettle:
    """Joint damping only (no stiffness, wheels free), applied by the sim while nothing commands the robot."""

    def __init__(self, model: mujoco.MjModel) -> None:
        if str(COMMON) not in sys.path:
            sys.path.insert(0, str(COMMON))
        from tron2_common.kinematics import NAMES

        self.dofs = [int(model.joint(name).dofadr[0]) for name in NAMES]
        self.actuators = [int(model.actuator(name.removesuffix("_Joint")).id) for name in NAMES]

    def apply(self, data: mujoco.MjData) -> None:
        data.ctrl[:] = 0.
        data.ctrl[self.actuators] = -LIMP_DAMPING * data.qvel[self.dofs]
