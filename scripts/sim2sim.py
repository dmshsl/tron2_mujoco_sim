"""Loopback-only scripted MuJoCo/deploy runner; --controller-command is a smoke-test seam."""
from __future__ import annotations

import argparse
import csv
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from contextlib import ExitStack, nullcontext
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw" if "--viewer" in sys.argv else "egl")
SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))

import imageio.v2 as imageio
import mujoco
import mujoco.viewer
import numpy as np

from tron2_sim.core import SimCore
from tron2_sim.height_scan_perception import PerceptionRenderer, SharedPose
from tron2_sim.rendering import OffscreenRenderer
from tron2_sim.start_conditions import (
    ContactLabels,
    ContactSlip,
    GantrySupport,
    PolicyProgress,
    ground_start,
    a_prime_pose,
    seated_start,
    wheel_bottoms,
)
from tron2_sim.variants.wf_payload import build


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["wf_payload"], required=True)
    parser.add_argument("--terrain", choices=["flat", "slope", "stairs", "rubble"], required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--timeout", type=float, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--video", type=Path, help="MP4 output; omit for a headless run without rendering")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--video-distance", type=float, default=2.5, help="Follow-camera distance (m)")
    parser.add_argument("--viewer", action="store_true", help="Show the live passive viewer (use MUJOCO_GL=glfw)")
    parser.add_argument("--height_scan", choices=["gt", "perception"], default="gt")
    parser.add_argument("--render_fps", type=float, default=30.0,
                        help="Target rate for the live perception depth renderer (--height_scan perception)")
    parser.add_argument("--start", choices=["ground", "gantry", "seated", "a_prime"],
                        help="Default: gantry for a real --policy run (the real-robot-standard start - "
                             "a crane strap holds the robot until the stand-up policy exists, task 23 "
                             "ladder step 1), ground for --controller-command smoke tests; seated = "
                             "random in-limit legs, 1 s limp settle on the base bottom (task 22, "
                             "pair with --standup scripted or learned); a_prime = both legs at pose A' "
                             "(+-3 deg, --seed), base at the FK support height, held (no limp settle) - the "
                             "learned stand-up's A'/A' training start without the limp heap")
    parser.add_argument("--seed", type=int, default=0, help="Seated start joint draw seed")
    parser.add_argument("--draw", choices=["full", "admissible"], default="full",
                        help="Seated joint draw: full-limit uniform (default) or DESIGN.md's K3 neighbourhood")
    parser.add_argument("--settle", type=float, default=1.0,
                        help="Seated limp settle seconds (0 = held start diagnostic, not a passive equilibrium)")
    parser.add_argument("--standup", choices=["none", "scripted", "learned"], default="none",
                        help="Forwarded to run_policy.py --standup (learned: seated route to A'/A'_p, the "
                             "stand-up policy, success hold, handoff to --policy)")
    parser.add_argument("--standup-model", type=Path,
                        help="Forwarded to run_policy.py --standup-model (stand-up policy.onnx + contract.yaml)")
    parser.add_argument("--standup-stop-at", choices=["handoff", "A"], default="handoff",
                        help="Forwarded to run_policy.py --standup-stop-at (A: seated route P0-P3 then hold)")
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
    if args.start in ("seated", "a_prime") and (args.controller_command or args.standup not in ("scripted", "learned")):
        parser.error("seated start requires the policy controller with --standup scripted or learned")
    if args.standup_model is not None and args.standup != "learned":
        parser.error("--standup-model requires --standup learned")
    if args.force_action_scale is not None and (
            not np.isfinite(args.force_action_scale) or args.force_action_scale <= 0 or args.controller_command):
        parser.error("force_action_scale must be positive/finite and requires the policy controller")
    if args.render_fps <= 0 or not np.isfinite(args.settle) or args.settle < 0:
        parser.error("render_fps must be positive and settle nonnegative")
    os.environ["ROBOT_IP"] = "127.0.0.1"
    # Every file touched inside the real-time loops lives in RAM (/dev/shm) while the run lasts: the
    # scan files are rewritten 50x per second, the controller streams its CSV at 500 Hz and this
    # process both logs at 1 kHz and tails that CSV. On a busy disk a single blocked read()/write()
    # on the shared ext4 inode froze the sim and the controller together for 250-320 ms, which is
    # exactly the SDK stale-data / output-watchdog latch (measured: SIM_LOOP_SLOW poll_ms~250 on
    # the CSV tail, PROCESS_STALL with zero run-queue wait and zero page faults). The logs are
    # moved to the requested paths when the run ends.
    scratch = Path("/dev/shm") if Path("/dev/shm").is_dir() else Path("/tmp")
    scan_path = scratch / f"tron2-gt-{os.getpid()}.bin"
    os.environ["TRON2_GT_SCAN_PATH"] = str(scan_path)
    perception_scan_path = scratch / f"tron2-perception-{os.getpid()}.bin"
    if args.standup_stop_at != "handoff" and args.standup != "scripted":
        parser.error("--standup-stop-at requires --standup scripted")
    for path in (args.csv, args.video):
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
    work = scratch / f"tron2-run-{os.getpid()}"
    work.mkdir(parents=True, exist_ok=True)
    live = {"controller.csv": args.csv, "sim.csv": args.csv.with_suffix(".sim.csv"),
            "gtscan.csv": args.csv.with_suffix(".gtscan.csv")}
    controller_csv = work / "controller.csv"
    command = shlex.split(args.controller_command) if args.controller_command else [
        args.controller_python, str(SIM.parent / "tron2_deploy/run_policy.py"),
        "--policy", args.policy, "--robot_ip", "127.0.0.1", "--autostart",
        "--schedule", str(args.schedule.resolve()), "--timeout", str(args.timeout),
        "--csv", str(controller_csv), "--height_scan", args.height_scan, "--standup", args.standup,
        "--standup-stop-at", args.standup_stop_at]
    if args.standup_model is not None and not args.controller_command:
        command += ["--standup-model", str(args.standup_model.resolve())]
    if args.force_action_scale is not None:
        command[1:2] = [str(SIM / "scripts/debug_policy.py"), str(args.force_action_scale)]
    core = SimCore(build(args.terrain, scan_path), str(SIM), headless=True)
    if len(core.modules) != 1:
        raise RuntimeError("Terrain/GT module failed to attach")
    pose_policy = args.policy if not args.controller_command else "base_blind"
    contract_path = SIM.parent / f"tron2_deploy/controllers/model/WF_TRON2A_{pose_policy.upper()}/contract.yaml"
    if args.start == "seated":
        seated = seated_start(core.model, core.data, args.seed, draw=args.draw, settle_s=args.settle)
        print(f"SEATED {seated}", flush=True)
    elif args.start == "a_prime":
        seated = seated_start(core.model, core.data, args.seed, settle_s=0., pose=a_prime_pose(args.seed, 3.))
        print(f"SEATED {seated}", flush=True)
    else:
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
    pinned: list[int] = []
    if perception:
        try:
            pinned = sorted(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            pinned = []
        if len(pinned) >= 2:  # the controller's seated-route planner process belongs on the main cores
            env["TRON2_PLANNER_CPUS"] = ",".join(str(cpu) for cpu in pinned[:len(pinned) // 2])
    render_thread = None
    if perception:
        shared_pose = SharedPose(core.model.nq)
        shared_pose.publish(core.data.qpos, core.data.time)
        render_thread = PerceptionRenderer(core.model, shared_pose, perception_scan_path, args.render_fps)
        render_thread.warm_up()  # absorb the one-time EGL/shader cold-start cost before the real-time loop
        render_thread.start()
    try:
        viewer_data = mujoco.MjData(core.model) if args.viewer else core.data
        with (mujoco.viewer.launch_passive(core.model, viewer_data) if args.viewer else nullcontext()) as viewer, \
                subprocess.Popen(command, cwd=SIM, env=env, start_new_session=True) as controller:
            if perception:
                try:
                    cpus = pinned
                    if len(cpus) >= 2:
                        half = len(cpus) // 2
                        os.sched_setaffinity(0, set(cpus[:half]))
                        os.sched_setaffinity(controller.pid, set(cpus[half:]))
                        print(f"CPU_AFFINITY main_and_renderer_and_planner={cpus[:half]} controller={cpus[half:]}",
                              flush=True)
                    else:
                        print(f"CPU_AFFINITY skipped: only {len(cpus)} CPU(s) available", flush=True)
                except (AttributeError, OSError) as error:
                    print(f"CPU_AFFINITY skipped: {error}", flush=True)
            try:
                with ExitStack() as resources, (imageio.get_writer(
                    str(args.video), fps=args.fps, codec="libx264", macro_block_size=1
                ) if args.video is not None else nullcontext()) as video, \
                        (work / "sim.csv").open("w", buffering=1 << 22) as stream:
                    writer = csv.writer(stream)
                    labels = ContactLabels(core.model)
                    slip = ContactSlip(core.model)
                    writer.writerow(["sim_time", "base_x", "base_y", "base_z", "qw", "qx", "qy", "qz",
                                     "vx", "vy", "vz", "wz", "wall_time", "controller_alive",
                                     "controller_t", "policy_t", "gantry_force", "gantry_released", "contacts",
                                     *[f"tau_{core.model.actuator(i).name}" for i in range(core.model.nu)],
                                     "wheel_bottom_z_L", "wheel_bottom_z_R", "slip"])
                    gt_writer, gt_stream, next_gt_log = None, None, 0.
                    if perception:
                        gt_stream = resources.enter_context((work / "gtscan.csv").open("w", newline=""))
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
                        resources.enter_context(controller_csv.open()))
                    frames = round(args.timeout * args.fps)
                    poses = []
                    completed = 0
                    failure = None
                    for frame in range(frames):
                        if viewer is not None and not viewer.is_running():
                            break
                        target = (frame + 1) / args.fps
                        while core.data.time + 1e-9 < target:
                            loop_started = time.monotonic()
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
                            tick_started = time.monotonic()
                            core._tick()
                            tick_done = time.monotonic()
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
                                             force, int(support is not None and support.released),
                                             labels(core.data), *core.data.actuator_force,
                                             *wheel_bottoms(core.model, core.data)[:, 2], slip(core.data)])
                            logged = time.monotonic()
                            remaining = start + float(core.data.time) - logged
                            if remaining > 0:
                                time.sleep(remaining)
                            if time.monotonic() - loop_started > .05:  # a host stall; explains SDK latches
                                print(f"SIM_LOOP_SLOW sim_t={core.data.time:.3f} "
                                      f"poll_ms={1e3 * (tick_started - loop_started):.1f} "
                                      f"tick_ms={1e3 * (tick_done - tick_started):.1f} "
                                      f"log_ms={1e3 * (logged - tick_done):.1f} "
                                      f"sleep_ms={1e3 * (time.monotonic() - logged):.1f} "
                                      f"planned_sleep_ms={1e3 * max(remaining, 0.):.1f}", flush=True)
                        completed += 1
                        if video is not None:
                            poses.append(core.data.qpos.copy())
                        if viewer is not None:
                            with viewer.lock():
                                viewer_data.qpos[:] = core.data.qpos
                                mujoco.mj_forward(core.model, viewer_data)
                                viewer.cam.lookat[:] = core.data.qpos[:3]
                                viewer.cam.distance = 3.0
                            viewer.sync()
                        if not np.isfinite(core.data.qpos).all():
                            raise FloatingPointError("Non-finite MuJoCo state")
                    print(f"SIM2SIM frames={completed} sim_s={core.data.time:.3f} "
                          f"wall_s={time.monotonic()-start:.3f}", flush=True)
                    if render_thread is not None:
                        render_thread.stop()
                        print(f"PERCEPTION_RENDER_RATE achieved_hz={render_thread.achieved_fps:.2f} "
                              f"target_hz={args.render_fps:.2f} frames={render_thread.rendered} "
                              f"skipped_ticks={render_thread.skipped_ticks}", flush=True)
                    if video is not None:
                        render_data = mujoco.MjData(core.model)
                        with OffscreenRenderer(core.model) as renderer:
                            renderer.camera.distance = args.video_distance
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
        for name, destination in live.items():
            if (work / name).exists():
                shutil.move(str(work / name), str(destination))
        shutil.rmtree(work, ignore_errors=True)
    print("RESULT: PASS (transport/render harness only; policy gates belong to task 12)", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
