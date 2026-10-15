"""Batch + metrics for the scripted seated route (limp heap -> pose A', then hold) in MuJoCo.

run:     sim2sim.py --start seated (limp 1.0 s settle) + run_policy.py --standup scripted
         --standup-stop-at A, one process per seed, headless unless --video-seeds. On a shared host,
         --max-load waits for the 1-min load average to drop before each seed and --retries re-runs
         a seed whose only failure is an SDK timing latch (skew / stale / output watchdog: a host
         stall, never a route rejection or a physics failure); every earlier attempt is kept under
         host_stall_attempts/ and analyze reports raw first-attempt numbers next to the final ones.
analyze: per seed acceptance, success (HOLD window = first 2 s after P3: at A' within 5 deg per
         joint, base and both wheels on the ground, tilt < 0.2 rad, no damping latch before the window
         ends, AND max tilt < 0.2 rad throughout the route), max tilt while moving, peak torques,
         contacts by phase, durations, and the planner's ROUTE_INFO (order, pre-step, resting-leg
         support margin). A latch later in the long hold (e.g. the SDK stale-data watchdog on this
         shared machine) is reported separately.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM.parent / "tron2_common"))
from tron2_common.kinematics import BOTTOM
from tron2_common.seated_route import A_PRIME

POSITIONS = [0, 1, 2, 3, 5, 6, 7, 8]  # sdk order: L pitch/roll/yaw/knee/wheel, R ...
PHASES = ("P0", "P1", "P2", "P3", "HOLD")
HIP = {"proximal_pitch_L", "proximal_roll_L", "proximal_pitch_R", "proximal_roll_R"}
SETTLED = {"base", "wheel_L", "wheel_R"} | HIP


TIMING_LATCHES = ("timestamp skew", "stale beyond watchdog", "output watchdog tripped")


def host_stall_latch(log: Path) -> str | None:
    """The SDK timing latch that cut a seed short, or None: route rejections and physics failures
    are never retried, nor is a latch that fired only after the 2 s success window at A'."""
    text = log.read_text(errors="replace")
    latch = next((line.split("DAMPING_LATCH: ")[1] for line in text.splitlines() if "DAMPING_LATCH" in line), None)
    if not latch or not any(kind in latch for kind in TIMING_LATCHES) or "rejected" in latch:
        return None
    csv_path = log.with_suffix(".csv")
    if csv_path.exists():
        control = _rows(csv_path)
        hold = [float(r["t"]) for r in control if int(float(r["standup_phase"])) >= 4]
        if hold and float(control[-1]["t"]) >= hold[0] + 2.:
            return None  # reached A' and held through the success window; the late latch is reported, not retried
    return latch


def wait_for_load(max_load: float | None, limit_s: float = 900.) -> None:
    if max_load is None:
        return
    started = time.monotonic()
    while os.getloadavg()[0] > max_load and time.monotonic() - started < limit_s:
        print(f"waiting: load {os.getloadavg()[0]:.1f} > {max_load}", flush=True)
        time.sleep(15.)


def run(args: argparse.Namespace) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        name = f"seed{seed}"
        command = [sys.executable, "scripts/sim2sim.py", "--variant", "wf_payload", "--terrain", "flat",
                   "--policy", "base_blind", "--height_scan", "gt", "--start", "seated", "--seed", str(seed),
                   "--settle", "1.0", "--standup", "scripted", "--standup-stop-at", "A",
                   "--schedule", str(SIM / "tests/hold_schedule.yaml"), "--timeout", str(args.timeout),
                   "--csv", str(args.out / f"{name}.csv")]
        if seed in args.video_seeds:
            command += ["--video", str(args.out / f"limp_to_A_seed{seed}.mp4"), "--video-distance", "1.7"]
        for attempt in range(args.retries + 1):
            wait_for_load(args.max_load)
            with (args.out / f"{name}.log").open("w") as log:
                code = subprocess.run(command, cwd=SIM, stdout=log, stderr=subprocess.STDOUT,
                                      env={**os.environ, "MUJOCO_GL": "egl"}, check=False).returncode
            stall = host_stall_latch(args.out / f"{name}.log")
            print(f"{name} exit={code} attempt={attempt + 1} load={os.getloadavg()[0]:.1f}"
                  + (f" host_stall_latch={stall!r}" if stall else ""), flush=True)
            if stall is None or attempt == args.retries:
                break
            kept = args.out / "host_stall_attempts"
            kept.mkdir(exist_ok=True)
            for suffix in (".log", ".csv", ".sim.csv"):
                source = args.out / f"{name}{suffix}"
                if source.exists():
                    source.rename(kept / f"{name}_attempt{attempt + 1}{suffix}")


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open() as stream:
        return list(csv.DictReader(stream))


def tilt(row: dict[str, str]) -> float:
    qx, qy = float(row["qx"]), float(row["qy"])
    return math.acos(max(-1., min(1., 1. - 2. * (qx * qx + qy * qy))))


def analyze_seed(out: Path, seed: int) -> dict[str, object]:
    log = (out / f"seed{seed}.log").read_text(errors="replace")
    latch = next((line.split("DAMPING_LATCH: ")[1] for line in log.splitlines() if "DAMPING_LATCH" in line), None)
    route = next((json.loads(line.split("ROUTE_INFO ", 1)[1]) for line in log.splitlines()
                  if line.startswith("ROUTE_INFO ")), None)
    control, sim = _rows(out / f"seed{seed}.csv"), _rows(out / f"seed{seed}.sim.csv")
    phase = np.array([int(float(row["standup_phase"])) for row in control])
    times = np.array([float(row["t"]) for row in control])
    accepted = bool(np.any(phase >= 1)) and not (latch and "rejected" in latch)
    attempts = sorted((out / "host_stall_attempts").glob(f"seed{seed}_attempt*.log")) if (out / "host_stall_attempts").is_dir() else []
    result: dict[str, object] = {"seed": seed, "accepted": accepted, "latch": latch, "route": route,
                                 "host_stall_retries": len(attempts),
                                 "first_attempt_latch": host_stall_latch(attempts[0]) if attempts else None,
                                 "order": None if route is None else route.get("order"),
                                 "prestep": None if route is None else route.get("prestep"),
                                 "support_first_leg": None if route is None else route.get("support_first_leg")}
    starts = {PHASES[p]: float(times[np.argmax(phase >= p)]) for p in range(5) if np.any(phase >= p)}
    result["phase_start_s"] = starts
    damped = [float(row["t"]) for row in control if all(float(row[f"target_q_{i}"]) == 0. for i in POSITIONS)]
    latch_t = damped[0] if damped else (float(times[-1]) if latch else None)
    result["latch_t"] = latch_t
    if latch_t is not None:  # the sim keeps stepping after the controller latches; drop that tail
        cut = next((i for i, row in enumerate(sim) if float(row["controller_t"]) >= latch_t - .15), len(sim))
        sim = sim[:cut]
    sim_t = np.array([float(row["controller_t"]) for row in sim])
    sim_phase = phase[np.clip(np.searchsorted(times, sim_t), 0, len(phase) - 1)]
    sim_phase[sim_t < 0] = 0
    tilts = np.array([tilt(row) for row in sim])
    taus = np.abs(np.array([[float(row[k]) for k in row if k.startswith("tau_")] for row in sim]))
    names = [k.removeprefix("tau_") for k in sim[0] if k.startswith("tau_")]
    moving = sim_phase >= 1
    contacts: dict[str, dict[str, float]] = {}
    for p in range(5):
        rows = [sim[i]["contacts"] for i in np.nonzero(sim_phase == p)[0]]
        counts: dict[str, int] = {}
        for labels in rows:
            for label in filter(None, labels.split(";")):
                counts[label] = counts.get(label, 0) + 1
        contacts[PHASES[p]] = {k: round(v / max(len(rows), 1), 3) for k, v in sorted(counts.items())}
    unplanned = sorted({k for p, c in contacts.items() for k in c
                        if k.startswith(("payload:", "self:")) or (p in ("P3", "HOLD") and k not in SETTLED)})
    result |= {"tilt_max_moving": float(tilts[moving].max()) if moving.any() else float(tilts.max()),
               "tilt_start": float(tilts[0]), "contacts": contacts, "unplanned": unplanned,
               "peak_tau": dict(zip(names, np.round(taus[moving].max(0) if moving.any() else taus.max(0), 1).tolist(),
                                    strict=True))}
    hold = phase == 4
    if accepted and hold.any():
        end_t = float(starts["HOLD"]) + 2.
        q = np.array([[float(row[f"q_{i}"]) for i in POSITIONS] for row, t in zip(control, times, strict=True)
                      if end_t - .5 <= t <= end_t])
        dq = np.array([[float(row[f"dq_{i}"]) for i in POSITIONS] for row in control])
        ground = float(sim[0]["base_z"]) - BOTTOM
        last = [row for row, t in zip(sim, sim_t, strict=True) if end_t - .5 <= t <= end_t] or sim[-50:]
        error = float(np.degrees(np.max(np.abs(q - A_PRIME))))
        base_gap = max(float(r["base_z"]) - BOTTOM - ground for r in last)
        wheel_gap = max(max(float(r["wheel_bottom_z_L"]), float(r["wheel_bottom_z_R"])) - ground for r in last)
        end_tilt = max(tilt(r) for r in last)
        wheel_contact = {s: float(np.mean([f"wheel_{s}" in r["contacts"] for r in last])) for s in "LR"}
        result |= {"route_s": float(starts["HOLD"] - starts["P1"]), "enable_to_A_s": float(starts["HOLD"]),
                   "plan_wait_s": float(starts["P1"] - 1.), "end_error_deg": error, "end_base_gap_m": base_gap,
                   "end_wheel_gap_m": wheel_gap, "end_tilt": end_tilt, "end_wheel_contact": wheel_contact,
                   "peak_joint_speed": float(np.abs(dq[phase >= 1]).max()),
                   "latched_in_window": result["latch_t"] is not None and result["latch_t"] <= end_t,
                   "success": bool(error < 5. and base_gap < .005 and wheel_gap < .005 and end_tilt < .2
                                   and float(result["tilt_max_moving"]) < .2
                                   and (result["latch_t"] is None or result["latch_t"] > end_t)
                                   and float(times[-1]) >= end_t)}
    else:
        result["success"] = False
    return result


def analyze(args: argparse.Namespace) -> None:
    results = [analyze_seed(args.out, seed) for seed in args.seeds]
    accepted = [r for r in results if r["accepted"]]
    retried = [r["seed"] for r in results if r["host_stall_retries"]]
    summary = {"n": len(results), "accepted": len(accepted), "success": sum(bool(r["success"]) for r in results),
               "host_stall_retried_seeds": retried,
               "raw_first_attempt": {"accepted": sum(bool(r["accepted"]) for r in results if not r["host_stall_retries"]),
                                     "success": sum(bool(r["success"]) for r in results if not r["host_stall_retries"])},
               "tilt_max": max(float(r["tilt_max_moving"]) for r in results),
               "tilt_over_0.2_runs": [r["seed"] for r in results if float(r["tilt_max_moving"]) >= .2],
               "unplanned_runs": [r["seed"] for r in results if r["unplanned"]],
               "presteps": {r["seed"]: r["prestep"] for r in results if r["prestep"] not in (None, "none")},
               "support_first_leg_min": min((float(r["support_first_leg"]) for r in results
                                             if r["support_first_leg"] is not None), default=None),
               "late_latches": {r["seed"]: r["latch"] for r in results if r["accepted"] and r["latch"]},
               "rejections": {r["seed"]: r["latch"] for r in results if not r["accepted"]}}
    for key in ("route_s", "enable_to_A_s", "plan_wait_s", "end_error_deg", "peak_joint_speed"):
        values = [float(r[key]) for r in results if key in r]
        if values:
            summary[key] = {"min": min(values), "median": float(np.median(values)), "max": max(values)}
    peaks = [r["peak_tau"] for r in accepted]
    summary["peak_tau_max"] = {k: max(p[k] for p in peaks) for k in peaks[0]} if peaks else {}
    (args.out / "metrics.json").write_text(json.dumps({"summary": summary, "runs": results}, indent=1))
    print(json.dumps(summary, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "analyze"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(50)))
    parser.add_argument("--video-seeds", type=int, nargs="*", default=[])
    parser.add_argument("--timeout", type=float, default=50.)
    parser.add_argument("--retries", type=int, default=0, help="re-runs per seed after an SDK timing latch (host stall)")
    parser.add_argument("--max-load", type=float, default=None, help="wait for the 1-min load average below this")
    args = parser.parse_args()
    run(args) if args.mode == "run" else analyze(args)


if __name__ == "__main__":
    main()
