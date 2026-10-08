"""Loopback-only scripted MuJoCo/deploy runner; --controller-command is a smoke-test seam."""
from __future__ import annotations

import argparse
import csv
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))

import imageio.v2 as imageio
import mujoco
import numpy as np

from tron2_sim.core import SimCore
from tron2_sim.rendering import OffscreenRenderer
from tron2_sim.variants.wf_payload import build


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["wf_payload"], required=True)
    parser.add_argument("--terrain", choices=["flat", "slope", "stairs", "rubble"], required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--timeout", type=float, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--height_scan", choices=["gt"], default="gt")
    parser.add_argument("--force_action_scale", type=float,
                        help="Negative control only: replace position action scales; default off")
    parser.add_argument("--controller-command",
                        help="Explicit command without a shell; receives TRON2_GT_SCAN_PATH")
    parser.add_argument("--controller-python", default=sys.executable)
    args = parser.parse_args()
    if args.timeout <= 0 or args.fps <= 0 or not args.schedule.is_file():
        parser.error("positive timeout/fps and an existing schedule are required")
    if args.force_action_scale is not None and (
            not np.isfinite(args.force_action_scale) or args.force_action_scale <= 0 or args.controller_command):
        parser.error("force_action_scale must be positive/finite and requires the policy controller")
    os.environ["ROBOT_IP"] = "127.0.0.1"
    scan_path = Path(f"/tmp/tron2-gt-{os.getpid()}.bin")
    os.environ["TRON2_GT_SCAN_PATH"] = str(scan_path)
    for path in (args.csv, args.video):
        path.parent.mkdir(parents=True, exist_ok=True)
    command = shlex.split(args.controller_command) if args.controller_command else [
        args.controller_python, str(SIM.parent / "tron2_deploy/run_policy.py"),
        "--policy", args.policy, "--robot_ip", "127.0.0.1", "--autostart",
        "--schedule", str(args.schedule.resolve()), "--timeout", str(args.timeout),
        "--csv", str(args.csv.resolve()), "--height_scan", args.height_scan]
    if args.force_action_scale is not None:
        command[1:2] = [str(SIM / "scripts/debug_policy.py"), str(args.force_action_scale)]
    core = SimCore(build(args.terrain, scan_path), str(SIM), headless=True)
    if len(core.modules) != 1:
        raise RuntimeError("Terrain/GT module failed to attach")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SIM) + os.pathsep + env.get("PYTHONPATH", "")
    env["TRON2_HEIGHT_SCAN_FACTORY"] = "tron2_sim.height_scan_gt:create_provider"
    try:
        with subprocess.Popen(command, cwd=SIM, env=env, start_new_session=True) as controller:
            try:
                with OffscreenRenderer(core.model) as renderer, imageio.get_writer(
                    str(args.video), fps=args.fps, codec="libx264", macro_block_size=1
                ) as video, args.csv.with_suffix(".sim.csv").open("w") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["sim_time", "base_x", "base_y", "base_z", "qw", "qx", "qy", "qz",
                                     "vx", "vy", "vz", "wz", "wall_time", "controller_alive"])
                    # Publish an unchanged measured pose until the real SDK command arrives.
                    deadline = time.monotonic() + 15.
                    while not any(core.channels[0].cmd["kp"]):
                        code = controller.poll()
                        if code is not None:
                            raise RuntimeError(f"Controller exited before first SDK command: {code}")
                        if time.monotonic() > deadline:
                            raise TimeoutError("Controller did not deliver an SDK command")
                        for channel in core.channels:
                            channel.read_state(core.data, False)
                            channel.publish(core.data)
                        core.modules[0].next_scan = 0.
                        core.modules[0].on_publish(core)
                        time.sleep(.01)
                    start = time.monotonic()
                    frames = round(args.timeout * args.fps)
                    poses = []
                    failure = None
                    for frame in range(frames):
                        target = (frame + 1) / args.fps
                        while core.data.time + 1e-9 < target:
                            code = controller.poll()
                            if failure is None and code is not None and (code != 0 or frame < frames - 2):
                                failure = f"Controller exited early: {code}, frame {frame}/{frames}"
                                print(f"FAIL {failure}; remaining rollout is unpowered observation", flush=True)
                                channel = core.channels[0]
                                channel.cmd = {key: [0.] * channel.n for key in ("q", "dq", "tau", "kp", "kd")}
                            core._tick()
                            writer.writerow([core.data.time, *core.data.qpos[:7], *core.data.qvel[:3],
                                             core.data.qvel[5], time.monotonic() - start, int(failure is None)])
                            remaining = start + float(core.data.time) - time.monotonic()
                            if remaining > 0:
                                time.sleep(remaining)
                        poses.append(core.data.qpos.copy())
                        if not np.isfinite(core.data.qpos).all():
                            raise FloatingPointError("Non-finite MuJoCo state")
                    print(f"SIM2SIM frames={len(poses)} sim_s={core.data.time:.3f} "
                          f"wall_s={time.monotonic()-start:.3f}", flush=True)
                    render_data = mujoco.MjData(core.model)
                    for pose in poses:
                        render_data.qpos[:] = pose
                        mujoco.mj_forward(core.model, render_data)
                        video.append_data(renderer.rgb(render_data))
                    if failure is not None:
                        raise RuntimeError(failure)
            finally:
                try:
                    os.killpg(controller.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    controller.wait(timeout=5.)
                except subprocess.TimeoutExpired:
                    controller.kill()
                    controller.wait()
                try:
                    os.killpg(controller.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
    finally:
        scan_path.unlink(missing_ok=True)
    print("RESULT: PASS (transport/render harness only; policy gates belong to task 12)", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
