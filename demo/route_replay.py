"""Offline (faster than real time) replay of the seated route on a demo tidy heap: the demo physics model, the heap
placed and settled limp exactly as heap.settle_offline, then the deploy route law (tron2_deploy standup_mode:
Route.command with the gravity-sag compensation, 1 rad/s target rate limit, gain ramp over P0, torque pre-clip
at the stand-up fraction, wheel brake) at 500 Hz on the 1 kHz MuJoCo loop. Reports the base drift (xy, heading)
per route phase. A development tool for the in-place route (standup-inplace); the demo batch is the real check.

  uv run python demo/route_replay.py --seeds 1000-1029 [--no-brake] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

DEMO = Path(__file__).resolve().parent
SIM = DEMO.parent
ROOT = SIM.parent
for path in (ROOT / "tron2_common", SIM, DEMO, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import heap  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from drift import delta, pose  # noqa: E402
from model import physics_xml  # noqa: E402

from tron2_common.kinematics import NAMES  # noqa: E402

CONTRACT = ROOT / "tron2_deploy/controllers/model/WF_TRON2A_BASE/contract.yaml"
WHEELS = ("wheel_L_Joint", "wheel_R_Joint")


def replay(model: mujoco.MjModel, seed: int, brake: tuple[float, float] | None, contract: dict,
           extra_s: float = .5) -> dict:
    from tron2_common import seated_route

    q_init, attempt, _ = heap.draw_tidy_heap(seed, str(physics_xml()))
    data = mujoco.MjData(model)
    heap.place(model, data, q_init)
    limp = heap.LimpSettle(model)
    for _ in range(round(heap.SETTLE_S / model.opt.timestep)):
        limp.apply(data)
        mujoco.mj_step(model, data)
    joints = [model.joint(n) for n in NAMES]
    qadr = [int(j.qposadr[0]) for j in joints]
    dadr = [int(j.dofadr[0]) for j in joints]
    act = [int(model.actuator(n.removesuffix("_Joint")).id) for n in NAMES]
    wq = [int(model.joint(n).qposadr[0]) for n in WHEELS]
    wd = [int(model.joint(n).dofadr[0]) for n in WHEELS]
    wa = [int(model.actuator(n.removesuffix("_Joint")).id) for n in WHEELS]
    q0 = data.qpos[qadr].copy()
    started = time.perf_counter()
    route = seated_route.plan(q0)
    plan_s = time.perf_counter() - started
    sdk = contract["joints"]["sdk_names"]
    index = [sdk.index(n) for n in NAMES]
    kp_all, kd_all = np.array(contract["control"]["kp"]), np.array(contract["control"]["kd"])
    kp, kd = kp_all[index], kd_all[index]
    limit = np.array(contract["control"]["torque_limit"])[index] * contract["standup"]["torque_fraction"]
    wheel_index = [sdk.index(n) for n in WHEELS]
    wlimit = np.array(contract["control"]["torque_limit"])[wheel_index] * contract["standup"]["torque_fraction"]
    initial = contract["control"]["initial_gain_fraction"]
    speed = contract["standup"]["speed_limit"]
    hold = data.qpos[wq].copy()
    previous = q0.copy()
    marks = {"start": pose(data.qpos)}
    phase_seen = set()
    dt = model.opt.timestep
    end = route.duration + 1. + extra_s
    t = 0.
    desired = q0.copy()
    tilt_max = 0.
    wheel_travel = np.zeros(2)
    while t < end:
        step = int(round(t / dt))
        if step % 2 == 0:  # 500 Hz controller
            target = route.command(t, kp)
            previous = previous + np.clip(target - previous, -speed * 2 * dt, speed * 2 * dt)
            desired = previous
            phase = route.phase(t)
            if phase not in phase_seen:
                phase_seen.add(phase)
                marks[f"phase{phase}"] = pose(data.qpos)
        gain = initial + (1. - initial) * min(t / float(route.ends[0]), 1.)
        tau = gain * (kp * (desired - data.qpos[qadr]) - kd * data.qvel[dadr])
        data.ctrl[:] = 0.
        data.ctrl[act] = np.clip(tau, -limit, limit)
        if brake is not None:
            wtau = gain * (brake[0] * (hold - data.qpos[wq]) - brake[1] * data.qvel[wd])
        else:
            wtau = -gain * .6 * data.qvel[wd]
        data.ctrl[wa] = np.clip(wtau, -wlimit, wlimit)
        mujoco.mj_step(model, data)
        wheel_travel += np.abs(data.qvel[wd]) * dt * .1
        tilt_max = max(tilt_max, math.acos(max(-1., min(1., 1. - 2. * (data.qpos[4] ** 2 + data.qpos[5] ** 2)))))
        t += dt
    marks["end"] = pose(data.qpos)
    order = [k for k in ("start", *(f"phase{p}" for p in range(5)), "end") if k in marks]
    phases = {f"{a}->{b}": delta(marks[a], marks[b]) for a, b in zip(order, order[1:])}
    return {"seed": seed, "attempt": attempt, "plan_s": round(plan_s, 3), "duration": round(route.duration, 3),
            "ends": route.ends.round(3).tolist(), "order": route.info["order"],
            "route": delta(marks["start"], marks["end"]),
            "phases": phases, "tilt_max": round(tilt_max, 3), "wheel_travel_m": wheel_travel.round(4).tolist(),
            "final_error_deg": float(np.degrees(np.max(np.abs(data.qpos[qadr] - route.knots[-1]))))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", default="1000-1009")
    parser.add_argument("--no-brake", action="store_true")
    parser.add_argument("--brake", default="30,2", help="wheel brake kp,kd")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    first, _, last = args.seeds.partition("-")
    seeds = range(int(first), int(last or first) + 1)
    model = mujoco.MjModel.from_xml_path(str(physics_xml()))
    contract = yaml.safe_load(CONTRACT.read_text())
    brake = None if args.no_brake else tuple(float(v) for v in args.brake.split(","))
    from tron2_common import seated_route
    seated_route.warm_up()
    rows = []
    for seed in seeds:
        try:
            row = replay(model, seed, brake, contract)
        except Exception as error:  # noqa: BLE001
            row = {"seed": seed, "error": str(error)[:200]}
        rows.append(row)
        print(json.dumps(row), flush=True)
    ok = [r for r in rows if "route" in r]
    for key in ("xy", "yaw_deg"):
        values = np.abs([r["route"][key] for r in ok])
        print(f"ROUTE {key}: median {np.median(values):.4f} p90 {np.percentile(values, 90):.4f} max {values.max():.4f}")
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
