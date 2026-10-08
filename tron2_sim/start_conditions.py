"""Harness-only grounded initialization and a unilateral vertical gantry support."""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import TextIO

import mujoco
import numpy as np
import numpy.typing as npt
import yaml


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
    field = model.hfield("terrain")
    count = int(field.nrow[0] * field.ncol[0])
    offset = int(field.adr[0])
    if np.ptp(model.hfield_data[offset:offset + count]) > 0:
        robot_geoms = (model.geom_contype != 0) & (model.geom_group != 5)
        forward_extent = float(np.max(data.geom_xpos[robot_geoms, 0] + model.geom_rbound[robot_geoms]))
        cell_width = 2. * field.size[0] / (int(field.ncol[0]) - 1)
        data.qpos[0] = data.geom("terrain").xpos[0] - field.size[0] - forward_extent - cell_width
        mujoco.mj_forward(model, data)
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
