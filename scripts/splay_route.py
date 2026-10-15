"""Scripted (no RL) splay -> kneel -> stand route test in MuJoCo with deploy-like PD.

S0 balance the seated splay (CoM x on the wheel-contact line), S1 fold the legs down until the
knee housings land, S2 turn the kneeling legs from lateral to forward (halfcrouch_front),
S3 extend along the balanced manifold with a wheel-pitch feedback. Logs every 10 ms.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull

from tron2_sim.payload_model import materialize
from tron2_sim.terrain import load_terrain

LEG = ["proximal_pitch_L_Joint", "proximal_roll_L_Joint", "proximal_yaw_L_Joint", "knee_L_Joint",
       "proximal_pitch_R_Joint", "proximal_roll_R_Joint", "proximal_yaw_R_Joint", "knee_R_Joint"]
WHEELS = ["wheel_L_Joint", "wheel_R_Joint"]
KP = np.array([159.67, 159.67, 53.22, 159.67] * 2)
KD = np.array([10.16, 10.16, 3.39, 10.16] * 2)
TAU = np.array([140., 140., 40., 140.] * 2)
WHEEL_KD, WHEEL_TAU = .6, 20.
VCAP = .6  # rad/s cosine peak; leaves margin for tracking catch-up under the 1 rad/s route limit
POSE1 = [0., 169.8, -90., -142.05]
KNEEL = [55., 10., 0., -150.]
# balanced extension manifold (standup-redesign.md): base h, hip, knee (deg)
MANIFOLD = [(.225, 108, -149.1), (.246, 104, -145.6), (.289, 96, -138.5), (.338, 88, -130.),
            (.387, 80, -121.2), (.437, 72, -111.9), (.490, 64, -101.1)]
PLANNED = {"S0": {"body", "wheel"}, "S1": {"body", "wheel", "knee"}, "S2": {"body", "wheel", "knee"},
           "S3": {"wheel", "knee"}, "hold": {"body", "wheel", "knee"}}


def mirror(left: list[float]) -> list[float]:
    p, r, y, k = left
    return [p, -r, -y, k]


class Robot:
    def __init__(self) -> None:
        # Flat floor as an analytic plane: the flat hfield's convex-vs-hfield contacts flicker
        # (ncon 8 -> 0 for several 1 ms steps while seated), which makes the support chatter.
        path = materialize("flat")
        xml = path.read_text()
        xml = re.sub(r'(<geom name="(?:terrain|west|east|south|north)"[^>]*?)contype="1" conaffinity="15"',
                     r'\1contype="0" conaffinity="0"', xml)
        xml = xml.replace("<worldbody>", '<worldbody><geom name="floor" type="plane" size="30 30 0.1" group="5" '
                          'contype="1" conaffinity="15" rgba="0.45 0.52 0.40 1"/>', 1)
        floor_xml = path.with_name("flat_plane_floor.xml")
        floor_xml.write_text(xml)
        self.m = mujoco.MjModel.from_xml_path(str(floor_xml))
        load_terrain(self.m, "flat")
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.qadr = np.array([m.joint(n).qposadr[0] for n in LEG])
        self.dadr = np.array([m.joint(n).dofadr[0] for n in LEG])
        self.wq = np.array([m.joint(n).qposadr[0] for n in WHEELS])
        self.wd = np.array([m.joint(n).dofadr[0] for n in WHEELS])
        self.act = np.array([m.actuator(n.removesuffix("_Joint")).id for n in LEG])
        self.wact = np.array([m.actuator(n.removesuffix("_Joint")).id for n in WHEELS])
        self.base = m.body("base_Link").id
        self.geoms = [g for g in range(m.ngeom) if m.geom_contype[g] and m.geom_bodyid[g] >= 2]
        self.terrain = {g for g in range(m.ngeom) if m.geom_group[g] == 5}
        self.hulls: dict[int, np.ndarray] = {}
        self.wheel_geom = {s: next(g for g in self.geoms if m.body(m.geom_bodyid[g]).name == f"wheel_{s}_Link") for s in "LR"}
        self.wheel_r = float(m.geom_size[self.wheel_geom["L"], 0])

    # ---------------- kinematics ----------------
    def set_pose(self, q8: list[float], z: float, pitch: float = 0.) -> None:
        d = self.d
        d.qpos[:] = 0.
        d.qpos[2] = z
        quat = np.zeros(4)
        mujoco.mju_euler2Quat(quat, np.array([0., pitch, 0.]), "xyz")
        d.qpos[3:7] = quat
        d.qpos[self.qadr] = np.radians(q8)
        d.qvel[:] = 0.
        mujoco.mj_kinematics(self.m, d)
        mujoco.mj_comPos(self.m, d)

    def hull(self, mesh: int) -> np.ndarray:
        if mesh not in self.hulls:
            a, n = self.m.mesh_vertadr[mesh], self.m.mesh_vertnum[mesh]
            v = self.m.mesh_vert[a:a + n]
            self.hulls[mesh] = v[ConvexHull(v).vertices].copy()
        return self.hulls[mesh]

    def geom_minz(self, g: int) -> float:
        m, d = self.m, self.d
        rot, pos = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            return float((self.hull(m.geom_dataid[g]) @ rot.T + pos)[:, 2].min())
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER:
            r, h = m.geom_size[g, :2]
            az = rot[2, 2]
            return float(pos[2] - r * math.sqrt(max(0., 1. - az * az)) - h * abs(az))
        corners = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]) * m.geom_size[g]
        return float((corners @ rot.T + pos)[:, 2].min())

    def body_minz(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for g in self.geoms:
            b = self.m.body(self.m.geom_bodyid[g]).name
            out[b] = min(out.get(b, 1e9), self.geom_minz(g))
        return out

    def grounded(self, q8: list[float], pitch: float = 0.) -> float:
        """Base z putting the lowest robot point on the floor (static placement)."""
        self.set_pose(q8, .5, pitch)
        z = .5 - min(self.body_minz().values())
        self.set_pose(q8, z, pitch)
        return z

    def on_wheels(self, left: list[float]) -> float:
        q = left + mirror(left)
        self.set_pose(q, .5)
        z = .5 - self.geom_minz(self.wheel_geom["L"])
        self.set_pose(q, z)
        return z

    def com(self) -> np.ndarray:
        return self.d.subtree_com[self.base].copy()

    def wheel_line_offset(self) -> float:
        """CoM x minus the x of the line joining the two wheel contact centres, at CoM y (static FK)."""
        d = self.d
        pl, pr = d.geom_xpos[self.wheel_geom["L"]], d.geom_xpos[self.wheel_geom["R"]]
        c = self.com()
        t = 0. if abs(pl[1] - pr[1]) < 1e-9 else (c[1] - pr[1]) / (pl[1] - pr[1])
        return float(c[0] - (pr[0] + t * (pl[0] - pr[0])))

    def axle_z(self) -> float:
        return float(self.d.geom_xmat[self.wheel_geom["L"]].reshape(3, 3)[2, 2])


# ---------------- waypoint construction ----------------
def solve_s0(rb: Robot) -> tuple[list[float], dict]:
    """Seated: keep base bottom + wheels on the floor, put CoM x on the wheel line, level axle."""
    base_z = rb.grounded(POSE1 + mirror(POSE1))
    before = rb.wheel_line_offset()
    wz0 = rb.geom_minz(rb.wheel_geom["L"])

    def residual(x: np.ndarray) -> list[float]:
        left = [x[0], x[1], x[2], POSE1[3]]
        rb.set_pose(left + mirror(left), base_z)
        return [rb.wheel_line_offset() * 100, rb.axle_z() * 10, (rb.geom_minz(rb.wheel_geom["L"]) - wz0) * 100]

    sol = least_squares(residual, [0., POSE1[1], POSE1[2] - 10.])
    left = [float(v) for v in sol.x] + [POSE1[3]]
    rb.set_pose(left + mirror(left), base_z)
    # pitch-only / roll-only cannot do it: report the best each alone achieves
    alone = {}
    for name, idx in (("pitch_only", 0), ("roll_only", 1)):
        def res1(x: np.ndarray, idx: int = idx) -> list[float]:
            l2 = list(POSE1)
            l2[idx] = POSE1[idx] + x[0]
            rb.set_pose(l2 + mirror(l2), base_z)
            return [rb.wheel_line_offset() * 100]
        s1 = least_squares(res1, [0.], bounds=([-20.], [20.]))
        alone[name] = {"best_shift_deg": float(s1.x[0]), "offset_m": float(s1.fun[0] / 100)}
    rb.set_pose(left + mirror(left), base_z)
    return left, {"offset_before_m": before, "offset_after_m": rb.wheel_line_offset(), "base_z": base_z,
                  "axle_z": rb.axle_z(), "single_joint_limits(+-20deg)": alone}


def rotate_legs(rb: Robot, left: list[float], alpha_deg: float, guess: list[float]) -> list[float]:
    """Joints whose yaw-link orientation equals Rz(alpha) * orientation(left) (knee unchanged)."""
    yb = rb.m.body("proximal_yaw_L_Link").id
    rb.set_pose(left + mirror(left), .5)
    c, s = math.cos(math.radians(alpha_deg)), math.sin(math.radians(alpha_deg))
    target = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]]) @ rb.d.xmat[yb].reshape(3, 3)

    def residual(x: np.ndarray) -> np.ndarray:
        l2 = [x[0], x[1], x[2], left[3]]
        rb.set_pose(l2 + mirror(l2), .5)
        return (rb.d.xmat[yb].reshape(3, 3) - target).ravel()

    sol = least_squares(residual, guess, max_nfev=400)
    return [float(v) for v in sol.x] + [left[3]]


def s2_path(rb: Robot, alpha_start: float, step: float = 10.) -> list[list[float]]:
    """Kneel rotated by alpha about vertical (alpha -90 = legs lateral) -> halfcrouch_front."""
    path, guess = [], KNEEL[:3]
    alphas = np.arange(0., alpha_start - 1e-6, -step).tolist()
    if abs(alphas[-1] - alpha_start) > 1e-6:
        alphas.append(alpha_start)
    for a in alphas:
        left = rotate_legs(rb, KNEEL, a, guess)
        guess = left[:3]
        path.append(left)
    return path[::-1]


def balanced_roll_path(rb: Robot, s0: list[float], r_end: float, k_end: float, step: float = 5.) -> list[list[float]]:
    """Wheel-only static balance: for each roll, hip pitch+yaw putting the CoM over the wheels, axle level."""
    out, guess = [], [s0[0], s0[2]]
    rolls = np.arange(s0[1], r_end - 1e-6, -step)
    for r in rolls:
        k = s0[3] + (k_end - s0[3]) * (s0[1] - r) / (s0[1] - r_end)

        def residual(x: np.ndarray, r: float = r, k: float = k) -> list[float]:
            rb.on_wheels([x[0], r, x[1], k])
            return [rb.wheel_line_offset() * 100, rb.axle_z() * 10]

        sol = least_squares(residual, guess, max_nfev=200)
        if np.abs(sol.fun).max() > 1e-3:
            break
        guess = list(sol.x)
        out.append([float(sol.x[0]), float(r), float(sol.x[1]), float(k)])
    return out


# ---------------- trajectory ----------------
@dataclass
class Segment:
    name: str
    waypoints: list[list[float]]  # 8-joint deg; the first is the segment start
    hold: float = .3
    wheel: str = "damp"  # damp | balance
    times: list[float] = field(default_factory=list)
    correction: np.ndarray | None = None  # per-1ms-step additive target offset (rad), from ILC

    def build(self) -> float:
        self.times = [0.]
        for a, b in zip(self.waypoints, self.waypoints[1:]):
            delta = float(np.abs(np.radians(b) - np.radians(a)).max())
            self.times.append(self.times[-1] + max(.05, math.pi / 2 * delta / VCAP))
        return self.times[-1] + self.hold

    def target(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        q = np.radians(self.waypoints)
        if t >= self.times[-1]:
            return q[-1], np.zeros(8)
        i = int(np.searchsorted(self.times, t, side="right") - 1)
        t0, t1 = self.times[i], self.times[i + 1]
        u = (t - t0) / (t1 - t0)
        s = .5 - .5 * math.cos(math.pi * u)
        ds = .5 * math.pi * math.sin(math.pi * u) / (t1 - t0)
        return q[i] + s * (q[i + 1] - q[i]), ds * (q[i + 1] - q[i])


# ---------------- simulation ----------------
class Runner:
    def __init__(self, rb: Robot, render: bool, views: tuple[str, ...] = ("side", "front")) -> None:
        self.rb = rb
        self.rows: list[dict] = []
        self.frames: list[np.ndarray] = []
        self.keyframes: dict[str, np.ndarray] = {}
        self.render = render
        if render:
            self.renderer = mujoco.Renderer(rb.m, height=480, width=640)
            self.opt = mujoco.MjvOption()
            self.opt.geomgroup[:] = [0, 1, 0, 0, 0, 1]
        self.label = ""
        self.wheel_cmd = np.zeros(2)

    def start(self, q8: list[float], pitch: float = 0., settle: float = .5) -> None:
        rb = self.rb
        z = rb.grounded(q8, pitch) + .001
        rb.set_pose(q8, z, pitch)
        mujoco.mj_forward(rb.m, rb.d)
        rb.d.time = 0.
        seg = Segment("hold", [q8, q8], hold=settle)
        self.run(seg, log=False)

    def contacts(self) -> tuple[set[str], set[str]]:
        rb, m, d = self.rb, self.rb.m, self.rb.d
        cats, raw = set(), set()
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if c.dist > .002:
                continue
            if (g1 in rb.terrain) == (g2 in rb.terrain):
                if g1 not in rb.terrain:
                    a, b = sorted((m.body(m.geom_bodyid[g1]).name, m.body(m.geom_bodyid[g2]).name))
                    cats.add(f"self:{a}/{b}")
                continue
            g = g2 if g1 in rb.terrain else g1
            name = m.body(m.geom_bodyid[g]).name
            raw.add(name)
            if name == "base_Link":
                cats.add("body" if m.geom(g).name == "" else "payload:" + m.geom(g).name)
            elif name.startswith("wheel"):
                cats.add("wheel")
            elif name.startswith("knee"):
                side = name.split("_")[1]
                knee = d.xpos[m.body(name).id]
                wheel = d.xpos[m.body(f"wheel_{side}_Link").id]
                cats.add("knee" if np.linalg.norm(c.pos - knee) < np.linalg.norm(c.pos - wheel) else "shank_hub")
            else:
                cats.add("other:" + name.replace("_Link", ""))
        return cats, raw

    def wheel_contacts(self) -> tuple[list[np.ndarray], list[float]]:
        rb, m, d = self.rb, self.rb.m, self.rb.d
        pts, slips = [], []
        for s in "LR":
            g = rb.wheel_geom[s]
            best = None
            for i in range(d.ncon):
                c = d.contact[i]
                if g in (c.geom1, c.geom2) and (c.geom1 in rb.terrain or c.geom2 in rb.terrain) and c.dist < .002:
                    best = c.pos.copy() if best is None else best
            if best is None:
                continue
            jac = np.zeros((3, m.nv))
            mujoco.mj_jac(m, d, jac, None, best, m.geom_bodyid[g])
            v = jac @ d.qvel
            pts.append(best)
            slips.append(float(np.linalg.norm(v[:2])))
        return pts, slips

    def wheel_feedback(self) -> np.ndarray:
        """S3: sagittal balance. Wheel velocity target from base pitch + CoM-over-wheel offset."""
        d, rb = self.rb.d, self.rb
        quat = d.qpos[3:7]
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, quat)
        rot = rot.reshape(3, 3)
        pitch = math.atan2(-rot[2, 0], rot[2, 2])
        pitch_rate = float((rot @ d.qvel[3:6])[1])
        c = rb.com()
        wx = np.mean([d.geom_xpos[rb.wheel_geom[s]][0] for s in "LR"])
        heading = rot[:, 0]
        off = float((c[0] - wx) * heading[0] + (c[1] - np.mean([d.geom_xpos[rb.wheel_geom[s]][1] for s in "LR"])) * heading[1])
        com_v = float(np.dot(d.qvel[:2], heading[:2]))
        v_cmd = 6. * off + 1.2 * com_v + 1.5 * (pitch - self.pitch_ref) + .25 * pitch_rate
        omega = v_cmd / rb.wheel_r
        w = d.qvel[rb.wd]
        return np.clip(3. * (np.array([omega, omega]) - w), -WHEEL_TAU, WHEEL_TAU)

    def run(self, seg: Segment, log: bool = True, stop_tilt: float = 1.2) -> dict:
        rb, m, d = self.rb, self.rb.m, self.rb.d
        total = seg.build()
        steps = int(round(total / m.opt.timestep))
        start_t = d.time
        tau_peak = np.zeros(10)
        stats = {"name": seg.name, "duration_s": total, "h_start": float(d.qpos[2]), "max_tilt": 0.,
                 "contacts": set(), "unplanned": set(), "max_slip": 0., "max_offset_abs_wheel_only": 0.}
        self.pitch_ref = 0.
        render_every = int(round(1 / 30 / m.opt.timestep))
        self.err = np.zeros((steps, 8))
        self.tilt_trace = np.zeros(steps)
        for i in range(steps):
            t = d.time - start_t
            q_ref, dq_ref = seg.target(t)
            if seg.correction is not None:
                q_ref = q_ref + seg.correction[min(i, len(seg.correction) - 1)]
            q, dq = d.qpos[rb.qadr], d.qvel[rb.dadr]
            tau = np.clip(KP * (q_ref - q) + KD * (dq_ref - dq), -TAU, TAU)
            if seg.wheel == "balance":
                tw = self.wheel_feedback()
            else:
                tw = np.clip(-WHEEL_KD * d.qvel[rb.wd], -WHEEL_TAU, WHEEL_TAU)
            d.ctrl[:] = 0.
            d.ctrl[rb.act] = tau
            d.ctrl[rb.wact] = tw
            self.err[i] = seg.target(t)[0] - q
            self.tilt_trace[i] = math.acos(max(-1., min(1., 1. - 2. * (d.qpos[4] ** 2 + d.qpos[5] ** 2))))
            mujoco.mj_step(m, d)
            tau_peak = np.maximum(tau_peak, np.abs(np.r_[tau, tw]))
            if log and i % 10 == 0:
                self.log_row(seg.name, t, tau, tw, stats)
            if self.render and log and i % render_every == 0:
                self.frames.append(self.frame(f"{seg.name}  t={d.time:5.2f}s  z={d.qpos[2]:.3f}  {self.label}"))
            if log and stats["max_tilt"] > stop_tilt:
                stats["aborted_at_s"] = float(t)
                self.err, self.tilt_trace = self.err[:i + 1], self.tilt_trace[:i + 1]
                break
        stats["h_end"] = float(d.qpos[2])
        stats["tau_peak"] = {n.replace("_Joint", ""): float(v) for n, v in zip(LEG + WHEELS, tau_peak)}
        stats["contacts"] = sorted(stats["contacts"])
        stats["unplanned"] = sorted(stats["unplanned"])
        stats["q_end_deg"] = np.degrees(d.qpos[rb.qadr]).round(2).tolist()
        stats["q_ref_end_deg"] = np.degrees(seg.target(1e9)[0]).round(2).tolist()
        return stats

    def log_row(self, seg: str, t: float, tau: np.ndarray, tw: np.ndarray, stats: dict) -> None:
        rb, d = self.rb, self.rb.d
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, d.qpos[3:7])
        rot = rot.reshape(3, 3)
        roll = math.atan2(rot[2, 1], rot[2, 2])
        pitch = math.atan2(-rot[2, 0], math.hypot(rot[2, 1], rot[2, 2]))
        tilt = math.acos(max(-1., min(1., rot[2, 2])))
        cats, raw = self.contacts()
        pts, slips = self.wheel_contacts()
        c = rb.com()
        off = float("nan")
        lateral = float("nan")
        if len(pts) == 2:
            a, b = pts
            u = (a - b)[:2]
            n = np.linalg.norm(u)
            if n > 1e-6:
                u /= n
                rel = c[:2] - b[:2]
                off = float(rel[0] * u[1] - rel[1] * u[0])  # signed distance, + = CoM ahead (for L-R line)
                lateral = float(rel @ u)
        row = {"seg": seg, "t": round(float(d.time), 3), "base_z": float(d.qpos[2]), "roll": roll, "pitch": pitch,
               "tilt": tilt, "com_x": float(c[0]), "com_y": float(c[1]), "com_z": float(c[2]),
               "com_off_wheel_line": off, "com_along_line": lateral, "contacts": "|".join(sorted(cats)),
               "slip_L": slips[0] if len(slips) > 0 else 0., "slip_R": slips[1] if len(slips) > 1 else 0.}
        for n, v in zip(LEG + WHEELS, np.r_[tau, tw]):
            row["tau_" + n.replace("_Joint", "")] = float(v)
        for n, v in zip(LEG, d.qpos[rb.qadr]):
            row["q_" + n.replace("_Joint", "")] = float(np.degrees(v))
        self.rows.append(row)
        stats["max_tilt"] = max(stats["max_tilt"], tilt)
        if tilt < .3 and stats.get("tilt_exceeded_s") is None:
            stats["h_max_upright"] = max(stats.get("h_max_upright", 0.), float(d.qpos[2]))
        elif tilt >= .3 and stats.get("tilt_exceeded_s") is None:
            stats["tilt_exceeded_s"] = float(t)
            if self.render:
                self.keyframes[f"{seg}_tilt0.3"] = self.frame(f"{seg}: tilt reaches 0.3 rad, z={d.qpos[2]:.3f}")
        stats["contacts"] |= cats
        planned = PLANNED.get(seg, set())
        stats["unplanned"] |= {x for x in cats if x not in planned}
        stats["max_slip"] = max([stats["max_slip"]] + slips)
        if cats <= {"wheel"} and not math.isnan(off):
            stats["max_offset_abs_wheel_only"] = max(stats["max_offset_abs_wheel_only"], abs(off))

    def frame(self, text: str) -> np.ndarray:
        import cv2
        rb, d = self.rb, self.rb.d
        views = []
        for azimuth in (90., 180.):  # side (looking +y... from -y), front (from +x)
            cam = mujoco.MjvCamera()
            cam.lookat[:] = d.xpos[rb.base] * [1, 1, 0] + [0, 0, .18]
            cam.distance, cam.azimuth, cam.elevation = 1.45, azimuth, -18.
            self.renderer.update_scene(d, camera=cam, scene_option=self.opt)
            views.append(self.renderer.render().copy())
        img = np.concatenate(views, axis=1)
        img = cv2.putText(np.ascontiguousarray(img), text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
        img = cv2.putText(img, "SIDE (from -y)", (10, 470), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 0), 1)
        img = cv2.putText(img, "FRONT (from +x)", (650, 470), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 0), 1)
        return img


def s3_waypoints(start: list[float]) -> list[list[float]]:
    """Transfer: hip 55 -> 104 (knee -145.6) then up the manifold to 0.49 m."""
    out = [start]
    for _, hip, knee in MANIFOLD[1:]:
        left = [hip, 10., 0., knee]
        out.append(left + mirror(left))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", choices=["literal", "balanced", "left_first"], default="literal",
                    help="S1 variant: literal = user's planar roll fold (knee -142->-150, S0 pitch/yaw kept); "
                         "balanced = per-roll hip pitch+yaw that keeps CoM over the wheels while it exists, "
                         "then straight to the S1-end kneel; left_first = literal, left leg then right leg")
    ap.add_argument("--out", type=Path, default=Path("/home/eunwon/tron2_ws/artifacts/basecampaign/standup"))
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--ilc", type=int, default=0, help="Offline ILC passes on the S0/S1 joint targets")
    ap.add_argument("--s1_end_alpha", type=float, default=-50.,
                    help="S1-end kneel = halfcrouch_front rotated by alpha deg about vertical (-90 = lateral)")
    args = ap.parse_args()
    rb = Robot()
    report: dict = {"variant": args.variant}

    s0, s0_info = solve_s0(rb)
    report["S0_solution"] = s0_info
    s2 = s2_path(rb, args.s1_end_alpha)
    s1_end = s2[0]
    rb.set_pose(s1_end + mirror(s1_end), rb.grounded(s1_end + mirror(s1_end)))
    report["S1_end_static"] = {"base_z": float(rb.d.qpos[2]), "com_x": float(rb.com()[0]),
                               "wheel_line_offset": rb.wheel_line_offset()}
    if args.variant == "balanced":
        bal = balanced_roll_path(rb, s0, 55., -150.)
        report["balanced_path_last_roll"] = bal[-1][1] if bal else None
        s1_wps = [s0] + bal[1:] + [s1_end]
        s1_wps = [w + mirror(w) for w in s1_wps]
    elif args.variant == "left_first":
        mid = [s1_end + mirror(s0)]
        s1_wps = [s0 + mirror(s0)] + mid + [s1_end + mirror(s1_end)]
    else:
        lit = [[s0[0], r, s0[2], s0[3] + (-150. - s0[3]) * (s0[1] - r) / (s0[1] - 55.)] for r in np.linspace(s0[1], 55., 8)]
        s1_wps = [w + mirror(w) for w in lit] + [s1_end + mirror(s1_end)]

    # ---- run A: seated splay -> S0 -> S1, with optional offline ILC on the joint targets ----
    corrections: dict[str, np.ndarray | None] = {"S0": None, "S1": None}
    ilc_log = []
    for it in range(args.ilc + 1):
        final = it == args.ilc
        runner = Runner(rb, args.video and final)
        results = []
        runner.label = f"run A ({args.variant}{', ILC %d' % it if args.ilc else ''})"
        runner.start(POSE1 + mirror(POSE1))
        segs = [Segment("S0", [POSE1 + mirror(POSE1), s0 + mirror(s0)], hold=.5, correction=corrections["S0"]),
                Segment("S1", s1_wps, hold=1., correction=corrections["S1"])]
        for seg in segs:
            results.append(runner.run(seg))
            if runner.render:
                runner.keyframes[seg.name] = runner.frame(f"{seg.name} end (run A)")
            # ILC: add the measured tracking error while the body is still upright (tilt < 0.1)
            err = runner.err.copy()
            bad = np.nonzero(runner.tilt_trace > .1)[0]
            if len(bad):
                err[bad[0]:] = err[bad[0] - 1] if bad[0] > 0 else 0.
            n = int(round(seg.build() / rb.m.opt.timestep))
            full = np.zeros((n, 8))
            full[:len(err)] = err
            full[len(err):] = err[-1]
            corrections[seg.name] = full if corrections[seg.name] is None else corrections[seg.name] + full
            if results[-1].get("aborted_at_s") is not None:
                break
        ilc_log.append({r["name"]: (round(r["max_tilt"], 3), r.get("aborted_at_s"), round(r["h_end"], 3)) for r in results})
    report["ilc"] = ilc_log
    rows_a = list(runner.rows)
    if "S1" not in runner.keyframes and runner.render:
        runner.keyframes["S1"] = runner.frame("S1 (aborted)")
    # ---- run B: S1-end kneel (teleported, settled) -> S2 -> S3 ----
    runner.rows = []
    runner.label = "run B (start: S1-end kneel, teleported)"
    runner.start(s1_end + mirror(s1_end), settle=1.)
    results.append(runner.run(Segment("S2", [w + mirror(w) for w in s2], hold=1.)))
    runner.keyframes["S2"] = runner.frame("S2 end") if args.video else None
    results.append(runner.run(Segment("S3", s3_waypoints(list(np.degrees(rb.d.qpos[rb.qadr]))), hold=2., wheel="balance")))
    runner.keyframes["S3"] = runner.frame("S3 end") if args.video else None
    rows_b = runner.rows

    report["segments"] = results
    report["waypoints_deg"] = {"S0": s0 + mirror(s0), "S1_end": s1_end + mirror(s1_end),
                               "S2_end": s2[-1] + mirror(s2[-1]), "S1": s1_wps, "S2": [w + mirror(w) for w in s2]}
    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"splay_route_{args.variant}"
    import csv
    for name, rows in (("A", rows_a), ("B", rows_b)):
        with open(args.out / f"{tag}_{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    (args.out / f"{tag}.json").write_text(json.dumps(report, indent=1, default=float))
    if args.video:
        import cv2
        h, w = runner.frames[0].shape[:2]
        vw = cv2.VideoWriter(str(args.out / f"{tag}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
        for fr in runner.frames:
            vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
        vw.release()
        for k, fr in runner.keyframes.items():
            cv2.imwrite(str(args.out / f"{tag}_{k}.png"), cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    for r in results:
        print(json.dumps({k: v for k, v in r.items() if k not in ("q_end_deg", "q_ref_end_deg")}, default=float))


if __name__ == "__main__":
    main()
