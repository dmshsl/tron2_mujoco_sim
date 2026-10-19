"""MuJoCo learned stand-up chain: limp heap -> seated route to A'/A'_p -> learned stand-up -> handoff
-> Base/BaseBlind drives a short schedule; one run, one JSON summary.

  MUJOCO_GL=egl uv run --extra sdk --with pydantic --with onnxruntime python scripts/standup_chain.py \
      --policy base_blind --seed 10 --out <dir> [--video] [--standup-model <dir>]

Runs scripts/sim2sim.py --start seated --standup learned (flat wf_payload, gt scan for Base) with
the schedule stand 3 s, vx 0.3 m/s for 5 s, stop (times after the post-handoff 3 s preparation),
captures its output, and classifies the run from the controller/sim CSVs and the controller log:
  stood_and_drove  success hold reached, handoff, no latch, upright at the end
  safe_failure     the controller latched (damping) - stand-up timeout, tilt limit, start check ... -
                   and the harness ran to the end with finite state and both processes exiting 0
  crash            anything else (non-zero exit, traceback, missing CSV, non-finite state)
plumbing_ok = the learned stand-up policy ran (phase 5 reached) and the outcome is not a crash.
Also reported: the per-leg joint family at the policy start (roll = A', pitch = A'_p; the seated route decides it) and
the max |wheel speed| from the controller CSV during the stand-up and overall (deploy latch 40 x 1.2 = 48 rad/s).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from pathlib import Path

SIM = Path(__file__).resolve().parents[1]
SCHEDULE = SIM / "tests/standup_learned_schedule.yaml"
PATTERNS = {name: re.compile(pattern) for name, pattern in {
    "policy_start": r"STANDUP_POLICY_START t=([\d.]+)",
    "success": r"STANDUP_SUCCESS t=([\d.]+) policy_s=([\d.]+) height_est=([\d.]+)",
    "handoff": r"HANDOFF policy=(\w+) t=([\d.]+)",
    "latch": r"DAMPING_LATCH: (.*); restart required",
    "seated": r"^SEATED (.*)",
    "result": r"^RESULT: (\w+)",
}.items()}


def tilt(qw: float, qx: float, qy: float, qz: float) -> float:
    return math.acos(max(-1., min(1., 1. - 2. * (qx * qx + qy * qy))))


def leg_family(pitch: float, roll: float, yaw: float) -> str:
    """Coarse per-leg joint family (tron2_rl/standup/catch.leg_family): standard, roll (A'), pitch (A'_p), other."""
    if abs(roll) <= .3 and abs(yaw) <= .5 and pitch > -.5:
        return "standard"
    if abs(roll) > 1.5:
        return "roll"
    return "pitch" if pitch < -2. else "other"


def wheel_and_family(joints: list, policy_t: float | None, handoff_t: float | None) -> dict:
    """Start family per leg at the first policy step; max |wheel dq| during the stand-up (policy start to hand-off)."""
    if not joints:
        return {}
    out: dict = {"wheel_speed_max": round(max(max(w) for _, _, w in joints), 3)}
    if policy_t is not None:
        start = next((q for t, q, _ in joints if t >= policy_t), None)
        if start is not None:
            out["start_family"] = [leg_family(*start[0:3]), leg_family(*start[5:8])]
        during = [max(w) for t, _, w in joints if t >= policy_t and (handoff_t is None or t < handoff_t)]
        out["standup_wheel_speed_max"] = round(max(during), 3) if during else None
    return out


def summarize(log: str, controller_csv: Path, sim_csv: Path, returncode: int) -> dict:
    found: dict[str, list] = {name: [] for name in PATTERNS}
    for line in log.splitlines():
        for name, pattern in PATTERNS.items():
            match = pattern.search(line)
            if match:
                found[name].append(match.groups())
    summary: dict = {"returncode": returncode, "traceback": "Traceback (most recent call last)" in log,
                     "latch": found["latch"][0][0] if found["latch"] else None,
                     "harness_result": found["result"][-1][0] if found["result"] else None,
                     "seated": found["seated"][0][0] if found["seated"] else None}
    if not controller_csv.is_file() or not sim_csv.is_file():
        summary.update(outcome="crash", plumbing_ok=False, reason="missing CSV")
        return summary
    phases: list[tuple[float, int]] = []
    joints: list[tuple[float, list[float], list[float]]] = []  # (t, q SDK order, |wheel dq| L R)
    with controller_csv.open() as stream:
        for row in csv.DictReader(stream):
            phases.append((float(row["t"]), int(float(row["standup_phase"]))))
            if "q_0" in row and "dq_4" in row:
                joints.append((float(row["t"]), [float(row[f"q_{i}"]) for i in range(10)],
                               [abs(float(row["dq_4"])), abs(float(row["dq_9"]))]))
    first = {}
    for t, phase in phases:
        first.setdefault(phase, t)
    summary["phases_reached"] = sorted(first)
    summary["phase_first_t"] = {str(k): round(v, 3) for k, v in sorted(first.items())}
    policy_t = first.get(5)
    handoff_t = first.get(6)
    summary |= wheel_and_family(joints, policy_t, handoff_t)
    rows = []
    finite = True
    with sim_csv.open() as stream:
        for row in csv.DictReader(stream):
            values = [float(row[key]) for key in ("sim_time", "base_x", "base_y", "base_z", "qw", "qx", "qy", "qz",
                                                  "controller_t", "policy_t")]
            finite &= all(math.isfinite(v) for v in values)
            rows.append(values)
    if not rows:
        summary.update(outcome="crash", plumbing_ok=False, reason="empty sim CSV")
        return summary
    tilts = [tilt(*r[4:8]) for r in rows]
    during = [(r, a) for r, a in zip(rows, tilts, strict=True) if policy_t is not None and r[8] >= policy_t
              and (handoff_t is None or r[8] < handoff_t)]
    summary.update({
        "sim_s": rows[-1][0], "finite": finite,
        "policy_start_controller_t": policy_t, "handoff_controller_t": handoff_t,
        "standup_success": bool(found["success"]),
        "standup_seconds": float(found["success"][0][1]) if found["success"] else None,
        "standup_base_z_max": max((r[3] for r, _ in during), default=None),
        "standup_tilt_max": max((a for _, a in during), default=None),
        "tilt_max": max(tilts), "final_tilt": tilts[-1], "final_base_z": rows[-1][3],
    })
    if handoff_t is not None:
        drive = [r for r in rows if 3. <= r[9] <= 8.]
        summary["drive_x_m"] = drive[-1][1] - drive[0][1] if len(drive) > 1 else None
        summary["post_handoff_tilt_max"] = max((a for r, a in zip(rows, tilts, strict=True)
                                              if r[8] >= handoff_t), default=None)
    clean = returncode == 0 and not summary["traceback"] and finite and summary["harness_result"] == "PASS"
    if clean and handoff_t is not None and summary["latch"] is None and tilts[-1] < .5:
        outcome = "stood_and_drove"
    elif clean and summary["latch"] is not None:
        outcome = "safe_failure"
    else:
        outcome = "crash"
    summary["outcome"] = outcome
    summary["plumbing_ok"] = outcome != "crash" and policy_t is not None
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", choices=("base_blind", "base"), required=True)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--standup-model", type=Path)
    parser.add_argument("--timeout", type=float, default=60.)
    parser.add_argument("--start", choices=("seated", "a_prime"), default="seated",
                        help="seated: limp heap + seated route (default); a_prime: held at A'/A' (+-3 deg), no limp heap")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"chain_{args.policy}_seed{args.seed}" if args.start == "seated" else f"chain_{args.policy}_aprime_seed{args.seed}"
    controller_csv = args.out / f"{stem}.csv"
    command = [sys.executable, str(SIM / "scripts/sim2sim.py"), "--variant", "wf_payload", "--terrain", "flat",
               "--policy", args.policy, "--height_scan", "gt", "--start", args.start, "--seed", str(args.seed),
               "--settle", "1.0", "--standup", "learned", "--schedule", str(SCHEDULE),
               "--timeout", str(args.timeout), "--csv", str(controller_csv)]
    if args.video:
        command += ["--video", str(args.out / f"{stem}.mp4")]
    if args.standup_model is not None:
        command += ["--standup-model", str(args.standup_model.resolve())]
    # Warm the controller's imports (onnxruntime, scipy, the planner) once: on a cold uv environment
    # they took longer than sim2sim's 15 s first-command deadline (measured once, 2026-10-19).
    warm = [sys.executable, str(SIM.parent / "tron2_deploy/run_policy.py"), "--policy", args.policy, "--dry-run",
            "--standup", "learned"]
    if args.standup_model is not None:
        warm += ["--standup-model", str(args.standup_model.resolve())]
    subprocess.run(warm, cwd=SIM, capture_output=True, text=True, check=False, timeout=300)
    print("RUN " + " ".join(command), flush=True)
    process = subprocess.run(command, cwd=SIM, capture_output=True, text=True, check=False,
                             timeout=args.timeout * 4 + 300)
    log = process.stdout + process.stderr
    (args.out / f"{stem}.log").write_text(log)
    summary = {"policy": args.policy, "seed": args.seed, "start": args.start, "command": command,
               "video": str(args.out / f"{stem}.mp4") if args.video else None,
               **summarize(log, controller_csv, controller_csv.with_suffix(".sim.csv"), process.returncode)}
    (args.out / f"{stem}.json").write_text(json.dumps(summary, indent=2))
    print("CHAIN " + json.dumps({k: v for k, v in summary.items() if k != "command"}), flush=True)
    return 0 if summary["plumbing_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
