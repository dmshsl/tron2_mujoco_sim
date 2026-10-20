"""Tidy limp heaps for the demo: pose 1 (both legs laid out sideways, the way a person sets the robot
down neatly) plus a uniform random perturbation per joint, then a limp settle on the floor.

Pose 1 (left leg, deg) = (0, 169.8, -90, -142.05), the right leg mirrored (pitch, -roll, -yaw, knee);
it is in the roll family, the family whose end pose A' the learned stand-up was trained on.

"Tidy" is checked on the settled heap (draw_tidy_heap): the limp legs topple in pitch while they settle, and
a person who lays the robot down neatly leaves it sitting on its base with both legs on the roll side:
  * both legs reach the roll family (A') with the thigh kept raised (tron2_common.leg_family), and
  * the CoM is over the base face, at most TIDY_COM_BEHIND behind its edge (the seated route rejects a CoM
    more than 1 cm behind the base face while a leg is airborne as a predicted tip-over), and
  * the seated route's first planner attempt finds a route (plannable; standup-inplace).
Measured over 200 draws (seeds 0-199): 186 plan; the 14 rejects are 11 "CoM > 1 cm behind the base face"
heaps (both thighs toppled backwards), 2 no-route and 1 over the 40 s budget; 2 more plan to a mixed family.
With the check: 180 kept, 1 reject (no route), 2 mixed -> excluded by the family rule.
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
TIDY_COM_BEHIND = .01  # m, CoM margin over the base face may be down to -this
MAX_ATTEMPTS = 60
SETTLE_S = 1.
LIMP_DAMPING = .3  # N m s/rad, as tron2_sim.start_conditions.seated_start


def pose1() -> npt.NDArray[np.float64]:
    pitch, roll, yaw, knee = POSE1_LEFT_DEG
    return np.radians([pitch, roll, yaw, knee, pitch, -roll, -yaw, knee])


def tidy_pose(seed: int, spread_deg: float = SPREAD_DEG, attempt: int = 0) -> npt.NDArray[np.float64]:
    """Pose 1 + U(-spread, spread) per joint, clamped into the 3 deg joint margin; draws whose links
    reach into the payload frame (clearance < 0.03 m) are redrawn, as the Isaac seated draw does."""
    if str(COMMON) not in sys.path:
        sys.path.insert(0, str(COMMON))
    from tron2_common.kinematics import LOWER, UPPER, payload_clearance

    rng = np.random.default_rng(seed if attempt == 0 else [seed, attempt])
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


def settle_offline(model: mujoco.MjModel, q: npt.NDArray[np.float64],
                   seconds: float = SETTLE_S) -> npt.NDArray[np.float64]:
    """The heap the real-time settle will produce: place, LimpSettle for `seconds`, settled eight joints."""
    if str(COMMON) not in sys.path:
        sys.path.insert(0, str(COMMON))
    from tron2_common.kinematics import NAMES

    data = mujoco.MjData(model)
    place(model, data, q)
    limp = LimpSettle(model)
    for _ in range(round(seconds / model.opt.timestep)):
        limp.apply(data)
        mujoco.mj_step(model, data)
    return np.array([data.qpos[model.joint(name).qposadr[0]] for name in NAMES])


def tidy_check(q: npt.NDArray[np.float64]) -> str | None:
    """None if the settled heap is tidy, else the reason."""
    if str(COMMON) not in sys.path:
        sys.path.insert(0, str(COMMON))
    from tron2_common.kinematics import LOWER, UPPER
    from tron2_common.route_geometry import leg_state, support_margin
    from tron2_common.seated_route import leg_families

    clamped = np.clip(q, LOWER, UPPER)
    families = leg_families(clamped, fallback=True)
    if set(families.values()) != {"roll"}:
        return f"families {families}"
    margin = support_margin((leg_state(clamped[:4], "L"), leg_state(clamped[4:], "R")))
    if margin < -TIDY_COM_BEHIND:
        return f"CoM {-margin:.3f} m behind the base face"
    if not plannable(clamped):
        return "no strict seated route (first planner attempt)"
    return None


def plannable(q: npt.NDArray[np.float64]) -> bool:
    """The seated route's first attempt (strict support limits, RRT seed 0, either leg order) finds a route: the
    plan at [기립] then succeeds on that same first attempt (~1 s) instead of exhausting its seeds and fallbacks on a
    heap without a route (seed 1024 of the demo batch: 34 s, then "no route")."""
    from tron2_common import seated_route
    from tron2_common.safety import UnsafeReference

    try:
        q0c, _ = seated_route.admit(q)
        families = seated_route.leg_families(q0c, fallback=True)
    except UnsafeReference:
        return False
    return any(seated_route._stages(q0c, first, seated_route.SEEDS[0], families, seated_route.STRICT) is not None
               for first in seated_route._orders(q0c, families))


_MODEL: list[mujoco.MjModel] = []


def draw_tidy_heap(seed: int, model_path: str) -> tuple[npt.NDArray[np.float64], int, list[str]]:
    """(initial pose, attempt, reasons of the rejected attempts) of the first tidy draw for `seed`."""
    if not _MODEL:
        _MODEL.append(mujoco.MjModel.from_xml_path(model_path))
    rejected = []
    for attempt in range(MAX_ATTEMPTS):
        q = tidy_pose(seed, attempt=attempt)
        reason = tidy_check(settle_offline(_MODEL[0], q))
        if reason is None:
            return q, attempt, rejected
        rejected.append(reason)
    raise ValueError(f"no tidy heap for seed {seed} in {MAX_ATTEMPTS} attempts: {rejected[-3:]}")
