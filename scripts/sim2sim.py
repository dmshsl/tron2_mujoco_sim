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
from contextlib import ExitStack
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))

import imageio.v2 as imageio
import mujoco
import numpy as np

from tron2_sim.core import SimCore
from tron2_sim.height_scan_perception import PerceptionRenderer, SharedPose
from tron2_sim.rendering import OffscreenRenderer
from tron2_sim.start_conditions import GantrySupport, PolicyProgress, ground_start, wheel_bottoms
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
    parser.add_argument("--height_scan", choices=["gt", "perception"], default="gt")
    parser.add_argument("--render_fps", type=float, default=30.0,
                        help="Target rate for the live perception depth renderer (--height_scan perception)")
    parser.add_argument("--start", choices=["ground", "gantry"],
                        help="Default: gantry for a real --policy run (the real-robot-standard start - "
                             "a crane strap holds the robot until the stand-up policy exists, task 23 "
                             "ladder step 1), ground for --controller-command smoke tests")
    parser.add_argument("--force_action_scale", type=float,
                        help="Negative control only: replace position action scales; default off")
    parser.add_argument("--controller-command",
                        help="Explicit command without a shell; receives TRON2_GT_SCAN_PATH")
    parser.add_argument("--controller-python", default=sys.executable)
    args = parser.parse_args()
    if args.start is None:
        args.start = "ground" if args.controller_command else "gantry"
    if args.timeout <= 0 or args.fps <= 0 or not args.schedule.is_file():
        parser.error("positive timeout/fps and an existing schedule are required")
    if args.start == "gantry" and args.controller_command:
        parser.error("gantry release requires the policy controller CSV")
    if args.force_action_scale is not None and (
            not np.isfinite(args.force_action_scale) or args.force_action_scale <= 0 or args.controller_command):
        parser.error("force_action_scale must be positive/finite and requires the policy controller")
    if args.render_fps <= 0:
        parser.error("render_fps must be positive")
    os.environ["ROBOT_IP"] = "127.0.0.1"
    scan_path = Path(f"/tmp/tron2-gt-{os.getpid()}.bin")
    os.environ["TRON2_GT_SCAN_PATH"] = str(scan_path)
    perception_scan_path = Path(f"/tmp/tron2-perception-{os.getpid()}.bin")
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
    pose_policy = args.policy if not args.controller_command else "base_blind"
    contract_path = SIM.parent / f"tron2_deploy/controllers/model/WF_TRON2A_{pose_policy.upper()}/contract.yaml"
    ground_start(core.model, core.data, contract_path)
    support = GantrySupport(core.model, core.data) if args.start == "gantry" else None
    print(f"START mode={args.start} xyz={core.data.qpos[:3].tolist()} "
          f"wheel_bottom_z={wheel_bottoms(core.model, core.data)[:, 2].tolist()} "
          f"min_contact_dist={min((core.data.contact[i].dist for i in range(core.data.ncon)), default=0.)}",
          flush=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SIM) + os.pathsep + env.get("PYTHONPATH", "")
    perception = args.height_scan == "perception"
    if perception:
        env["TRON2_HEIGHT_SCAN_FACTORY"] = "tron2_sim.height_scan_perception:create_provider"
        env["TRON2_PERCEPTION_SCAN_PATH"] = str(perception_scan_path)
    else:
        env["TRON2_HEIGHT_SCAN_FACTORY"] = "tron2_sim.height_scan_gt:create_provider"
    render_thread = None
    if perception:
        shared_pose = SharedPose(core.model.nq)
        shared_pose.publish(core.data.qpos, core.data.time)
        render_thread = PerceptionRenderer(core.model, shared_pose, perception_scan_path, args.render_fps)
        render_thread.warm_up()  # absorb the one-time EGL/shader cold-start cost before the real-time loop
        render_thread.start()
    try:
        with subprocess.Popen(command, cwd=SIM, env=env, start_new_session=True) as controller:
            if perception:
                try:
                    cpus = sorted(os.sched_getaffinity(0))
                    if len(cpus) >= 2:
                        half = len(cpus) // 2
                        os.sched_setaffinity(0, set(cpus[:half]))
                        os.sched_setaffinity(controller.pid, set(cpus[half:]))
                        print(f"CPU_AFFINITY main_and_renderer={cpus[:half]} controller={cpus[half:]}", flush=True)
                    else:
                        print(f"CPU_AFFINITY skipped: only {len(cpus)} CPU(s) available", flush=True)
                except (AttributeError, OSError) as error:
                    print(f"CPU_AFFINITY skipped: {error}", flush=True)
            try:
                with ExitStack() as resources, imageio.get_writer(
                    str(args.video), fps=args.fps, codec="libx264", macro_block_size=1
                ) as video, args.csv.with_suffix(".sim.csv").open("w") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["sim_time", "base_x", "base_y", "base_z", "qw", "qx", "qy", "qz",
                                     "vx", "vy", "vz", "wz", "wall_time", "controller_alive",
                                     "controller_t", "policy_t", "gantry_force", "gantry_released"])
                    gt_writer, gt_stream, next_gt_log = None, None, 0.
                    if perception:
                        gt_stream = resources.enter_context(args.csv.with_suffix(".gtscan.csv").open("w", newline=""))
                        gt_writer = csv.writer(gt_stream)
                        gt_writer.writerow(["sim_time", "policy_t"] + [f"gt_scan_{i}" for i in range(231)])
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
                        if perception:
                            shared_pose.publish(core.data.qpos, core.data.time)
                        time.sleep(.01)
                    start = time.monotonic()
                    progress = None if args.controller_command else PolicyProgress(
                        resources.enter_context(args.csv.open()))
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
                            if progress is not None:
                                was_started = progress.first_action_t is not None
                                progress.update()
                                if not was_started and progress.first_action_t is not None:
                                    print(f"POLICY_FIRST_ACTION controller_t={progress.first_action_t:.9f} "
                                          f"observed_sim_t={core.data.time:.9f}", flush=True)
                            policy_started = progress is not None and progress.first_action_t is not None
                            force = 0. if support is None else support.apply(
                                core.data, policy_started or failure is not None)
                            core._tick()
                            if perception:
                                shared_pose.publish(core.data.qpos, core.data.time)
                                if gt_writer is not None and core.data.time + 1e-9 >= next_gt_log:
                                    gt_writer.writerow([core.data.time,
                                                       -1. if progress is None else progress.policy_t,
                                                       *core.modules[0].provider.get_scan()])
                                    next_gt_log = float(core.data.time) + .02
                            writer.writerow([core.data.time, *core.data.qpos[:7], *core.data.qvel[:3],
                                             core.data.qvel[5], time.monotonic() - start, int(failure is None),
                                             -1. if progress is None else progress.controller_t,
                                             -1. if progress is None else progress.policy_t,
                                             force, int(support is not None and support.released)])
                            remaining = start + float(core.data.time) - time.monotonic()
                            if remaining > 0:
                                time.sleep(remaining)
                        poses.append(core.data.qpos.copy())
                        if not np.isfinite(core.data.qpos).all():
                            raise FloatingPointError("Non-finite MuJoCo state")
                    print(f"SIM2SIM frames={len(poses)} sim_s={core.data.time:.3f} "
                          f"wall_s={time.monotonic()-start:.3f}", flush=True)
                    if render_thread is not None:
                        render_thread.stop()
                        print(f"PERCEPTION_RENDER_RATE achieved_hz={render_thread.achieved_fps:.2f} "
                              f"target_hz={args.render_fps:.2f} frames={render_thread.rendered} "
                              f"skipped_ticks={render_thread.skipped_ticks}", flush=True)
                    render_data = mujoco.MjData(core.model)
                    with OffscreenRenderer(core.model) as renderer:
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
        if render_thread is not None:
            render_thread.stop()  # idempotent if already stopped above; guards early-failure exits
        scan_path.unlink(missing_ok=True)
        perception_scan_path.unlink(missing_ok=True)
    print("RESULT: PASS (transport/render harness only; policy gates belong to task 12)", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
