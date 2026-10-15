"""Harness-only grounded/seated initialization and a unilateral vertical gantry support."""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import TextIO

import mujoco
import numpy as np
import numpy.typing as npt
import yaml

COMMON = Path(__file__).resolve().parents[2] / "tron2_common"


def wheel_bottoms(model: mujoco.MjModel, data: mujoco.MjData) -> npt.NDArray[np.float64]:
    points = []
    for gid in range(model.ngeom):
        if not model.body(model.geom_bodyid[gid]).name.startswith("wheel_") or not model.geom_contype[gid]:
            continue
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_CYLINDER:
            raise ValueError("Ground initialization requires the compiled WF wheel cylinders")
        axis_z = data.geom_xmat[gid].reshape(3, 3)[2, 2]
        radius, half_length = model.geom_size[gid, :2]
        extent = radius * math.sqrt(max(0., 1. - axis_z**2)) + half_length * abs(axis_z)
        point = data.geom_xpos[gid].copy()
        point[2] -= extent
        points.append(point)
    if len(points) != 2:
        raise ValueError("Expected exactly two colliding wheel cylinders")
    return np.asarray(points, dtype=np.float64)


def ground_start(model: mujoco.MjModel, data: mujoco.MjData, contract_path: Path) -> None:
    joints = yaml.safe_load(contract_path.read_text())["joints"]
    data.qpos[:7] = [0., 0., model.qpos0[2], 1., 0., 0., 0.]
    for name, value in zip(joints["sdk_names"], joints["default_position"], strict=True):
        data.qpos[model.joint(name).qposadr[0]] = float(value)
    data.qvel[:] = 0.
    data.ctrl[:] = 0.
    data.xfrc_applied[:] = 0.
    mujoco.mj_forward(model, data)
    _move_to_flat_border(model, data)
    bottoms = wheel_bottoms(model, data)
    elevations = []
    mask = np.array([0, 0, 0, 0, 0, 1], dtype=np.uint8)
    for point in bottoms:
        origin = point.copy()
        origin[2] += 2. * model.stat.extent
        distance = mujoco.mj_ray(model, data, origin, np.array([0., 0., -1.]), mask, True, -1,
                                 np.zeros(1, dtype=np.int32))
        if distance < 0:
            raise ValueError("No terrain below a start wheel")
        elevations.append(origin[2] - distance)
    if np.ptp(elevations) > 1e-6:
        raise ValueError("Start wheels must be on the same flat pad")
    data.qpos[2] += float(np.max(np.asarray(elevations) - bottoms[:, 2]))
    mujoco.mj_forward(model, data)
    terrain_contacts = [data.contact[i].dist for i in range(data.ncon)
                        if any(model.geom_group[g] == 5 for g in data.contact[i].geom)]
    # Hfield contact skins can penetrate even when analytic cylinder bottoms touch.
    data.qpos[2] += max(0., -min(terrain_contacts, default=0.) - .001)
    mujoco.mj_forward(model, data)


def _move_to_flat_border(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    field = model.hfield("terrain")
    count = int(field.nrow[0] * field.ncol[0])
    offset = int(field.adr[0])
    if np.ptp(model.hfield_data[offset:offset + count]) > 0:
        robot_geoms = (model.geom_contype != 0) & (model.geom_group != 5)
        forward_extent = float(np.max(data.geom_xpos[robot_geoms, 0] + model.geom_rbound[robot_geoms]))
        cell_width = 2. * field.size[0] / (int(field.ncol[0]) - 1)
        data.qpos[0] = data.geom("terrain").xpos[0] - field.size[0] - forward_extent - cell_width
        mujoco.mj_forward(model, data)


def _terrain_height(model: mujoco.MjModel, data: mujoco.MjData, point: npt.NDArray[np.float64]) -> float:
    origin = np.array([point[0], point[1], point[2] + 2. * model.stat.extent])
    mask = np.array([0, 0, 0, 0, 0, 1], dtype=np.uint8)
    distance = mujoco.mj_ray(model, data, origin, np.array([0., 0., -1.]), mask, True, -1,
                             np.zeros(1, dtype=np.int32))
    if distance < 0:
        raise ValueError("No terrain below the seated start")
    return float(origin[2] - distance)


def seated_draw(seed: int, draw: str = "full") -> npt.NDArray[np.float64]:
    """"full": the Isaac StandUp rung draw - uniform inside the reference's 3 degree
    limits, rejected while any leg/wheel is below the base-bottom plane or the payload
    gap < 30 mm. "admissible": DESIGN.md's K3-neighbourhood distribution (the one the
    reference's own 500-start test uses), reported separately from "full" by design."""
    sys.path.insert(0, str(COMMON))
    from tron2_common import standup_reference as reference
    from tron2_common.kinematics import BOTTOM, LOWER, UPPER, leg_floor, payload_clearance, wheel_floors

    rng = np.random.default_rng(seed)
    if draw == "admissible":
        span = np.array([.002, .001, .001, .06] * 2)
        return reference.k3_plant() + rng.uniform(-span, span) + np.array([0., 0., 0., .07] * 2)
    if draw != "full":
        raise ValueError(f"Unknown seated draw {draw!r}")
    for _ in range(100000):
        q = rng.uniform(LOWER, UPPER)
        if leg_floor(q) + BOTTOM < 0 or np.min(wheel_floors(q)) + BOTTOM < 0:
            continue
        if payload_clearance(q) >= .03:
            return q
    raise ValueError("No nonpenetrating random seated draw within 100000 attempts")


def seated_start(model: mujoco.MjModel, data: mujoco.MjData, seed: int, *, draw: str = "full",
                 settle_s: float = 1., damping: float = .3) -> dict[str, float]:
    """Seat the upright base on its bottom with random in-limit legs, then settle limp
    (position stiffness 0, damping 0.3 N*m*s/rad) for settle_s; wheels are free.
    settle_s=0 is the diagnostic "held" start: the pose is not a passive equilibrium."""
    sys.path.insert(0, str(COMMON))
    from tron2_common.kinematics import BOTTOM, NAMES

    if not np.isfinite(settle_s) or settle_s < 0 or damping < 0:
        raise ValueError("settle time and damping must be finite and nonnegative")
    q = seated_draw(seed, draw)
    data.qpos[:7] = [0., 0., model.qpos0[2], 1., 0., 0., 0.]
    for joint in range(model.njnt):
        if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_HINGE:
            data.qpos[model.jnt_qposadr[joint]] = 0.
    for name, value in zip(NAMES, q, strict=True):
        data.qpos[model.joint(name).qposadr[0]] = float(value)
    data.qvel[:] = 0.
    data.ctrl[:] = 0.
    data.xfrc_applied[:] = 0.
    mujoco.mj_forward(model, data)
    _move_to_flat_border(model, data)
    data.qpos[2] = _terrain_height(model, data, data.qpos[:3]) + BOTTOM + .001
    mujoco.mj_forward(model, data)
    dofs = [int(model.joint(name).dofadr[0]) for name in NAMES]
    actuators = [int(model.actuator(name.removesuffix("_Joint")).id) for name in NAMES]
    steps = round(settle_s / model.opt.timestep)
    for _ in range(steps):
        data.ctrl[:] = 0.
        data.ctrl[actuators] = -damping * data.qvel[dofs]
        mujoco.mj_step(model, data)
    data.ctrl[:] = 0.
    data.time = 0.
    mujoco.mj_forward(model, data)
    tilt = math.acos(max(-1., min(1., 1. - 2. * (data.qpos[4] ** 2 + data.qpos[5] ** 2))))
    return {"seed": float(seed), "settle_steps": float(steps), "base_z": float(data.qpos[2]),
            "tilt": tilt, "residual_joint_speed": float(np.max(np.abs(data.qvel[dofs])))}


class ContactLabels:
    """Per-step contact categories for the stand-up logs: terrain contacts by body (base_Link
    bottom as "base", payload geoms as "payload:<geom>") and robot self contacts as "self:a/b"."""

    def __init__(self, model: mujoco.MjModel, gap: float = .001) -> None:
        self.gap = gap
        self.terrain = model.geom_group == 5
        self.label = []
        for gid in range(model.ngeom):
            body = model.body(model.geom_bodyid[gid]).name
            name = model.geom(gid).name
            short = body.removesuffix("_Link")
            self.label.append(f"payload:{name.removesuffix('_collision')}" if body == "base_Link" and name
                              else "base" if body == "base_Link" else short)

    def __call__(self, data: mujoco.MjData) -> str:
        labels = set()
        for contact in data.contact[:data.ncon]:
            if contact.dist > self.gap:
                continue
            a, b = int(contact.geom1), int(contact.geom2)
            if self.terrain[a] and self.terrain[b]:
                continue
            if self.terrain[a] or self.terrain[b]:
                labels.add(self.label[b if self.terrain[a] else a])
            else:
                labels.add("self:" + "/".join(sorted((self.label[a], self.label[b]))))
        return ";".join(sorted(labels))


class PolicyProgress:
    """Consume complete controller CSV records, including zero-valued first actions."""

    def __init__(self, source: TextIO) -> None:
        self.source = source
        self.offset = 0
        self.partial = ""
        self.header: list[str] = []
        self.controller_t = -1.
        self.policy_t = -1.
        self.first_action_t: float | None = None

    def update(self) -> None:
        self.source.seek(self.offset)
        chunk = self.source.read()
        self.offset = self.source.tell()
        lines = (self.partial + chunk).split("\n")
        self.partial = lines.pop()
        for values in csv.reader(lines):
            if not self.header:
                self.header = values
                continue
            row = dict(zip(self.header, values, strict=True))
            self.controller_t, self.policy_t = float(row["t"]), float(row["policy_t"])
            actions = [float(value) for key, value in row.items() if key.startswith("action_")]
            if not actions or not all(math.isfinite(value) for value in actions):
                raise ValueError("Invalid action record in controller CSV")
            if self.policy_t > 0 and self.first_action_t is None:
                self.first_action_t = self.controller_t


class GantrySupport:
    """5 Hz critically damped vertical strap; releases once and never pushes down."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.body = model.body("base_Link").id
        self.height = float(data.qpos[2])
        mass = float(model.body_mass.sum())
        omega = 2. * math.pi * 5.
        self.stiffness = mass * omega**2
        self.damping = 2. * mass * omega
        self.weight = mass * abs(float(model.opt.gravity[2]))
        self.released = False

    def apply(self, data: mujoco.MjData, policy_started: bool) -> float:
        self.released = self.released or policy_started
        force = 0. if self.released else max(
            0., self.weight + self.stiffness * (self.height - data.qpos[2]) - self.damping * data.qvel[2])
        data.xfrc_applied[self.body, 2] = force
        return float(force)
