"""Headless demo batch: the demo's own sim server + controller (demo/sim_server.py, run_policy --live-commands),
driven without the window. Per run: [소환] tidy heap (seed) -> [기립] -> hand-off -> BASE_SECONDS of Base at a
zero command -> optional scripted drive (the GUI batch's DRIVE_10S as velocity steps). Records the plan wait
(press -> ROUTE_INFO, plus the planner's own compute time), route duration, phase times and the base drift per
phase (demo/drift.py).

  cd tron2_sim && uv run --extra sdk --extra demo python demo/headless_batch.py --runs 30 --seed 1000 \
      --out <dir> [--drive]
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
DEMO = Path(__file__).resolve().parent
SIM = DEMO.parent
for path in (SIM.parent, SIM, DEMO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np  # noqa: E402
import shared  # noqa: E402
from app import SimLink, tilt_of  # noqa: E402
from automation import foreign_sdk_processes  # noqa: E402
from drift import BASE_SECONDS, DriftTracker  # noqa: E402
from tron2_deploy.live_commands import MODE_IDLE, MODE_RUN, STATE_IDLE, STATE_LATCHED, Command  # noqa: E402

# (seconds after the drive starts, vx, wz): the GUI batch's DRIVE_10S key presses as their accumulated commands
DRIVE = ((0., .5, 0.), (1.5, 1., 0.), (4., 1., .25), (5., 1., .5), (6.5, 1., .25), (6.6, 1., 0.), (7.5, .5, 0.),
         (8.5, 0., 0.), (9.5, 0., 0.))


class Timeout(Exception):
    pass


class Driver:
    def __init__(self, policy: str, standup_model: Path | None = None) -> None:
        self.link = SimLink(policy, standup_model)
        self.mode, self.token, self.vx, self.wz = MODE_IDLE, 0, 0., 0.
        self.sim: shared.SimState | None = None
        self.status = self.link.live.read_status()
        self.route_info: dict | None = None
        self.route_at: float | None = None
        self.events: list[dict] = []
        self.server_exit = None
        self.last_write = 0.

    def pump(self) -> None:
        now = time.monotonic()
        if now - self.last_write >= .05:
            self.last_write = now
            self.link.live.write_command(Command(now, self.mode, self.token, self.vx, 0., self.wz, 1., 1.))
        while True:
            try:
                event = self.link.events.get_nowait()
            except queue.Empty:
                break
            kind = event.get("event")
            if kind == "controller":
                line = event.get("line", "")
                if line.startswith("ROUTE_INFO") and self.route_info is None:
                    self.route_at = time.monotonic()
                    try:
                        self.route_info = json.loads(line.split(" ", 1)[1])
                    except (IndexError, ValueError):
                        self.route_info = {}
            elif kind == "controller_started":
                self.link.controller_pid = event.get("pid")
            elif kind == "server_exit":
                self.server_exit = event.get("code")
            if kind in ("spawned", "settled", "controller_restart", "sim_slow", "warning", "error"):
                self.events.append(event)
        sim = self.link.state()
        if sim is not None:
            self.sim = sim
        try:
            self.status = self.link.live.read_status()
        except TimeoutError:
            pass

    def wait(self, predicate, timeout: float, step: float = .005) -> None:
        deadline = time.monotonic() + timeout
        while True:
            self.pump()
            if predicate():
                return
            if time.monotonic() > deadline or self.server_exit is not None:
                raise Timeout()
            time.sleep(step)

    def idle(self) -> bool:
        s = self.sim
        return (s is not None and s.sim_state == shared.ACTIVE and self.status.state == STATE_IDLE
                and time.monotonic() - self.status.stamp < .5 and s.controller == shared.RUNNING)

    def spawn(self, seed: int) -> None:
        self.mode, self.vx, self.wz = MODE_IDLE, 0., 0.
        self.pump()
        before = self.sim.spawn_id if self.sim is not None else 0
        self.link.send(cmd="spawn", seed=seed)
        self.wait(lambda: self.sim is not None and self.sim.spawn_id != before, 30.)
        self.wait(self.idle, 60.)

    def cycle(self, index: int, seed: int, drive: bool) -> dict:
        record: dict = {"run": index, "seed": seed, "started": time.strftime("%H:%M:%S")}
        foreign = foreign_sdk_processes()
        try:
            self.spawn(seed)
        except Timeout:
            return record | {"outcome": "fail_spawn"}
        spawned = next((e for e in reversed(self.events) if e.get("event") == "spawned"), {})
        record["spawn"] = {k: spawned.get(k) for k in ("attempt", "rejected", "q_deg")}
        until = time.monotonic() + .5  # the GUI batch presses [기립] 0.5 s after idle
        self.wait(lambda: time.monotonic() >= until, 5.)
        self.route_info, self.route_at = None, None
        tracker = DriftTracker(self.sim.qpos)
        self.token += 1
        self.mode = MODE_RUN
        pressed = time.monotonic()
        self.last_write = 0.
        phase_t: dict[int, float] = {}
        handoff = None
        deadline = pressed + 120.
        tilt_max = 0.
        while time.monotonic() < deadline:
            self.pump()
            now = time.monotonic()
            if self.status.state == STATE_LATCHED or self.server_exit is not None:
                break
            phase = self.status.phase if self.status.state != STATE_IDLE else -1
            if phase >= 0:
                phase_t.setdefault(phase, round(now - pressed, 3))
                tracker.sample(phase, self.sim.qpos, now - pressed)
                tilt_max = max(tilt_max, tilt_of(self.sim.qpos))
            if phase == 6:
                handoff = now
                break
            time.sleep(.004)
        record.update(phase_t=phase_t, latch=self.status.reason if self.status.state == STATE_LATCHED else None,
                      a_prime_s=phase_t.get(4))
        if self.route_info is not None:
            record["plan_wait_s"] = round(self.route_at - pressed, 3)
            record["route"] = {k: self.route_info.get(k) for k in ("duration_s", "plan_s", "families", "order",
                                                                    "contact_fallback", "source")}
        if handoff is None:
            record["drift"] = tracker.summary()
            return record | {"outcome": "fail_standup", "foreign": foreign}
        record["handoff_s"] = round(handoff - pressed, 3)
        if 5 in phase_t:
            record["route_s"] = round(phase_t[5] - phase_t.get(1, phase_t[5]), 3)
        end = handoff + BASE_SECONDS
        while time.monotonic() < end and self.status.state != STATE_LATCHED:
            self.pump()
            time.sleep(.005)
        tracker.end_base(self.sim.qpos)
        record["drift"] = tracker.summary()
        record["standup_tilt_max"] = round(tilt_max, 3)
        ok = self.status.state != STATE_LATCHED
        if drive and ok:
            start = time.monotonic()
            positions, tilts = [], []
            pending = list(DRIVE)
            while time.monotonic() - start < DRIVE[-1][0] + 1.5 and self.status.state != STATE_LATCHED:
                elapsed = time.monotonic() - start
                while pending and pending[0][0] <= elapsed:
                    _, self.vx, self.wz = pending.pop(0)
                self.pump()
                positions.append(self.sim.qpos[:2].copy())
                tilts.append(tilt_of(self.sim.qpos))
                time.sleep(.05)
            self.vx = self.wz = 0.
            path = float(np.sum(np.linalg.norm(np.diff(np.array(positions), axis=0), axis=1)))
            record["drive"] = {"path_m": round(path, 3), "tilt_max": round(max(tilts), 3),
                               "final_tilt": round(tilts[-1], 3), "final_z": round(float(self.sim.qpos[2]), 3)}
            ok = (self.status.state != STATE_LATCHED and tilts[-1] < .5 and self.sim.qpos[2] > .3 and path > 1.)
        foreign += foreign_sdk_processes()
        record["latch"] = self.status.reason if self.status.state == STATE_LATCHED else None
        record["foreign"] = foreign
        record["outcome"] = ("stood_and_drove" if drive else "stood") if ok else "fail_drive"
        return record

    def close(self) -> None:
        self.link.close()


def percentile(values: list[float], q: float) -> float | None:
    return round(float(np.percentile(values, q)), 4) if values else None


def summarize(results: list[dict]) -> dict:
    def collect(path):
        out = []
        for r in results:
            value = r
            for key in path:
                value = value.get(key) if isinstance(value, dict) else None
            if isinstance(value, (int, float)):
                out.append(abs(float(value)))
        return out

    summary: dict = {"runs": len(results), "outcomes": {}}
    for r in results:
        summary["outcomes"][r.get("outcome")] = summary["outcomes"].get(r.get("outcome"), 0) + 1
    for name, path in {"plan_wait_s": ("plan_wait_s",), "plan_compute_s": ("route", "plan_s"), "a_prime_s": ("a_prime_s",),
                       "route_duration_s": ("route", "duration_s"), "handoff_s": ("handoff_s",),
                       **{f"{seg}_{m}": ("drift", seg, m) for seg in ("route", "learned", "spawn_to_handoff", "base")
                          for m in ("xy", "yaw_deg", "left", "fwd")}}.items():
        values = collect(path)
        summary[name] = {"n": len(values), "median": percentile(values, 50), "p90": percentile(values, 90),
                         "p95": percentile(values, 95), "max": round(max(values), 4) if values else None}
    signed = [r["drift"]["spawn_to_handoff"]["left"] for r in results if "spawn_to_handoff" in r.get("drift", {})]
    summary["spawn_to_handoff_left_signed_mean"] = round(float(np.mean(signed)), 4) if signed else None
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--policy", choices=("base", "base_blind"), default="base")
    parser.add_argument("--drive", action="store_true", help="10 s scripted drive after the Base settle")
    parser.add_argument("--standup-model", type=Path, default=None, help="stand-up model dir (default: installed)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    foreign = foreign_sdk_processes()
    if foreign:
        print("REFUSED foreign limxsdk users: " + "; ".join(foreign), flush=True)
        return 2
    driver = Driver(args.policy, args.standup_model.resolve() if args.standup_model else None)
    results: list[dict] = []
    try:
        driver.wait(driver.idle, 180.)
        for index in range(args.runs):
            try:
                record = driver.cycle(index, args.seed + index, args.drive)
            except Timeout:
                record = {"run": index, "seed": args.seed + index, "outcome": "fail_timeout"}
            results.append(record)
            print("RUN " + json.dumps(record, ensure_ascii=False), flush=True)
            (args.out / "runs.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results))
            (args.out / "summary.json").write_text(json.dumps(summarize(results), indent=2, ensure_ascii=False))
            if driver.server_exit is not None:
                break
    finally:
        driver.close()
    print("SUMMARY " + json.dumps(summarize(results), ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
