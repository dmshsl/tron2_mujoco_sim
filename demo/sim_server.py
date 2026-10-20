"""Demo physics process: real-time MuJoCo (wf_payload, plane floor) on the limxsdk loopback, plus the
lifecycle of the real deploy controller (tron2_deploy/run_policy.py --standup learned, Base, GT height scan,
--live-commands) for the GUI in demo/app.py.

The GUI starts this process with pipes: JSON lines on stdin are requests ({"cmd": "spawn", "seed": n},
{"cmd": "reset"}, {"cmd": "quit"}), JSON lines on stdout are events (the controller's own output is forwarded
as {"event": "controller", "line": ...}). The robot state for rendering goes through demo/shared.py.
Physics never waits for the GUI: rendering, planning and the controller live in other processes.

Spawn/reset: the GUI first puts the controller in live idle (limp). Once the controller reports idle the
heap is placed (demo/heap.py) and settles limp for 1 s in real time; the same controller process is kept for
the next stand-up. A controller that latched, exited or does not acknowledge idle within SPAWN_ACK_S is
stopped (process group SIGTERM -> SIGKILL, so its planner worker goes too) and a fresh one starts in live
idle after the settle. Physics keeps running (and publishing) throughout: the SDK watchdog is 0.25 s.
Whenever no controller commands the joints (none running, or it publishes zero gains) the sim applies the
limp joint damping of the heap settle. "suspend" (another limxsdk loopback user appeared) stops the controller
and all SDK traffic until "resume".
"""
from __future__ import annotations

import argparse
import ctypes
import json
import multiprocessing
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
DEMO = Path(__file__).resolve().parent
SIM = DEMO.parent
for path in (SIM.parent, SIM, DEMO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import heap  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import shared  # noqa: E402
from model import physics_xml  # noqa: E402
from tron2_deploy.live_commands import STATE_IDLE, LiveChannel  # noqa: E402

from tron2_sim.core import SimCore  # noqa: E402
from tron2_sim.spec import IMU_STD, LEG_WF, ChannelSpec, RobotSpec  # noqa: E402
from tron2_sim.variants.wf_payload import PayloadTerrain  # noqa: E402

RUN_POLICY = SIM.parent / "tron2_deploy/run_policy.py"
PR_SET_PDEATHSIG = 1
SPAWN_ACK_S = 1.


# Events leave through a bounded queue and one writer thread: the physics loop and the controller's output
# reader must never block on a full stdout pipe (a GUI that stops reading for a second would otherwise freeze
# physics and the controller with it - the SDK watchdog is 0.25 s). Overflow drops events, counted.
_EVENTS: queue.Queue[str] = queue.Queue(maxsize=20000)
_DROPPED = [0]


def _writer() -> None:
    while (line := _EVENTS.get()) is not None:
        try:
            sys.stdout.write(line)
            if _EVENTS.empty():
                sys.stdout.flush()
        except (BrokenPipeError, OSError):
            return


_WRITER = threading.Thread(target=_writer, name="events", daemon=True)
_WRITER.start()


def emit(event: str, **fields) -> None:
    line = json.dumps({"event": event, "t": round(time.monotonic(), 3), **fields}) + "\n"
    try:
        _EVENTS.put_nowait(line)
    except queue.Full:
        _DROPPED[0] += 1


def flush_events(timeout: float = 2.) -> None:
    _EVENTS.put(None)
    _WRITER.join(timeout)


def die_with_parent() -> None:
    """Child-side: get SIGTERM when the parent dies (no orphans if the GUI or this server is killed)."""
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
    except OSError:
        pass


def _heap_worker_init(cpus: list[int]) -> None:
    """Tidy-heap worker: low priority, off the physics/controller cores, dies with the server."""
    die_with_parent()
    os.nice(5)
    if cpus:
        try:
            os.sched_setaffinity(0, set(cpus))
        except OSError:
            pass


def cpu_plan() -> dict[str, list[int]]:
    """Controller on the last 3 allowed CPUs, physics on the 2 before; everything else (GUI, planner) elsewhere."""
    try:
        cpus = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return {}
    if len(cpus) < 8:
        return {}
    return {"controller": cpus[-3:], "sim": cpus[-5:-3], "rest": cpus[:-5]}


class Controller:
    """One run_policy.py process in its own session (process group)."""

    def __init__(self, live_path: Path, scan_path: Path, cpus: dict[str, list[int]], policy: str) -> None:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(SIM) + os.pathsep + env.get("PYTHONPATH", "")
        env["ROBOT_IP"] = "127.0.0.1"
        env["TRON2_HEIGHT_SCAN_FACTORY"] = "tron2_sim.height_scan_gt:create_provider"
        env["TRON2_GT_SCAN_PATH"] = str(scan_path)
        if cpus.get("rest"):
            env["TRON2_PLANNER_CPUS"] = ",".join(str(cpu) for cpu in cpus["rest"])
        command = [sys.executable, "-u", str(RUN_POLICY), "--policy", policy, "--robot_ip", "127.0.0.1",
                   "--height_scan", "gt", "--standup", "learned", "--live-commands", str(live_path),
                   "--timeout", "86400"]
        self.process = subprocess.Popen(command, cwd=SIM, env=env, start_new_session=True, preexec_fn=die_with_parent,
                                        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, bufsize=1)
        if cpus.get("controller"):
            try:
                os.sched_setaffinity(self.process.pid, set(cpus["controller"]))
            except OSError as error:
                emit("warning", message=f"controller affinity: {error}")
        self.reader = threading.Thread(target=self._forward, daemon=True)
        self.reader.start()
        emit("controller_started", pid=self.process.pid, command=command)

    def _forward(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            line = line.rstrip()
            if line and not line.startswith(("  ", "- ")):  # skip the contract YAML dump
                emit("controller", line=line[:2000])

    def poll(self) -> int | None:
        return self.process.poll()

    def stop(self) -> int | None:
        for sig, wait in ((signal.SIGTERM, 2.), (signal.SIGKILL, 2.)):
            try:
                os.killpg(self.process.pid, sig)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=wait)
                break
            except subprocess.TimeoutExpired:
                continue
        try:
            os.killpg(self.process.pid, signal.SIGKILL)  # the planner worker shares the group
        except ProcessLookupError:
            pass
        return self.process.returncode


class Server:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.cpus = cpu_plan()
        if self.cpus.get("sim"):
            os.sched_setaffinity(0, set(self.cpus["sim"]))
        scratch = Path("/dev/shm") if Path("/dev/shm").is_dir() else Path("/tmp")
        self.scan_path = scratch / f"tron2-demo-gt-{os.getpid()}.bin"
        spec = RobotSpec(robot_type="WF_TRON2A_PAYLOAD", base="WF", family_dir="tron2a",
                         model_candidates=[str(physics_xml())],
                         channels=[ChannelSpec("main", LEG_WF, sdk_role="main", imu=IMU_STD)],
                         modules=[PayloadTerrain("flat", self.scan_path)], sdk_robot="Tron2", cam=(3., -20.))
        self.core = SimCore(spec, str(SIM), headless=True)
        if len(self.core.modules) != 1:
            raise RuntimeError("terrain/GT module failed to attach")
        self.model, self.data = self.core.model, self.core.data
        self.channel = self.core.channels[0]
        self.limp = heap.LimpSettle(self.model)
        self.block = shared.StateBlock(args.state, self.model.nq, create=True)
        self.requests: queue.Queue[dict] = queue.Queue()
        self.controller: Controller | None = None
        self.sim_state = shared.BOOT
        self.settle_until = 0.
        self.seed = -1
        self.spawn_id = 0
        self.pending: tuple[int | None, float, Future | None] | None = None
        self.model_path = self.core.model_path
        # The tidy-heap check (offline limp settle + family/CoM test, 0.1-1 s) runs in its own process so the
        # physics loop keeps its real-time publishing; warmed up now (leg-family grid caches).
        self.heaps = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"),
                                         initializer=_heap_worker_init, initargs=(self.cpus.get("rest", []),))
        self.heaps.submit(heap.draw_tidy_heap, 0, self.model_path)
        self.live = LiveChannel(args.live)
        self.exit_code = 0
        self.slow = 0
        self.base = self.model.body("base_Link").id
        self.running = True
        threading.Thread(target=self._read_stdin, daemon=True).start()
        emit("ready", nq=self.model.nq, cpus=self.cpus, scan=str(self.scan_path), model=self.core.model_path)

    def _read_stdin(self) -> None:
        for line in sys.stdin:
            try:
                self.requests.put(json.loads(line))
            except json.JSONDecodeError:
                emit("warning", message=f"bad request {line!r}")
        self.requests.put({"cmd": "quit"})  # the GUI went away

    # ---------------------------------------------------------------- lifecycle
    def stop_controller(self) -> None:
        if self.controller is not None:
            code = self.controller.stop()
            emit("controller_stopped", code=code)
            self.controller = None
        self.channel.cmd = {key: [0.] * self.channel.n for key in ("q", "dq", "tau", "kp", "kd")}

    def request_spawn(self, seed: int | None) -> None:
        future = None if seed is None else self.heaps.submit(heap.draw_tidy_heap, seed, self.model_path)
        self.pending = (seed, time.monotonic() + SPAWN_ACK_S, future)

    def try_spawn(self) -> None:
        """Place the pending heap once the controller is limp (live idle acknowledged), else restart it."""
        assert self.pending is not None
        seed, deadline, future = self.pending
        if future is not None and not future.done():
            return
        now = time.monotonic()
        status = self.live.read_status()
        idle = (self.controller is not None and self.controller.poll() is None and status.state == STATE_IDLE
                and now - status.stamp < .2)
        if not idle and now < deadline and self.controller is not None and self.controller.poll() is None:
            return
        if not idle:
            emit("controller_restart", reason=f"state={status.state} reason={status.reason!r}")
            self.stop_controller()
        self.pending = None
        q, info = heap.pose1(), {}
        if future is not None:
            try:
                q, attempt, rejected = future.result()
                info = {"attempt": attempt, "rejected": rejected}
            except Exception as error:  # never lose the spawn: report and use the raw draw
                q, info = heap.tidy_pose(seed), {"tidy_error": str(error)}
        self.spawn(seed, q, info)

    def spawn(self, seed: int | None, q=None, info: dict | None = None) -> None:
        q = heap.pose1() if q is None else q
        now = float(self.data.time)
        heap.place(self.model, self.data, q)
        self.data.time = now  # keep the sim clock monotonic for the GT scan and logs
        self.core.modules[0].next_scan = 0.
        mujoco.mj_forward(self.model, self.data)
        self.seed = -1 if seed is None else seed
        self.spawn_id += 1
        self.sim_state = shared.SETTLING
        self.settle_until = now + heap.SETTLE_S
        emit("spawned", seed=self.seed, spawn_id=self.spawn_id, q_deg=np.degrees(q).round(2).tolist(), **(info or {}))

    def start_controller(self) -> None:
        self.controller = Controller(self.args.live, self.scan_path, self.cpus, self.args.policy)
        self.exit_code = 0

    def handle(self, request: dict) -> None:
        command = request.get("cmd")
        if command == "spawn":
            self.request_spawn(int(request.get("seed", np.random.SeedSequence().entropy % 1_000_000)))
        elif command == "reset":
            self.request_spawn(None)
        elif command == "suspend":  # another limxsdk loopback user appeared: no controller, no SDK traffic
            self.pending = None
            self.stop_controller()
            self.sim_state = shared.SUSPENDED
            emit("suspended")
        elif command == "resume":
            if self.sim_state == shared.SUSPENDED:
                emit("resumed")
                self.spawn(None)
        elif command == "quit":
            self.running = False
        else:
            emit("warning", message=f"unknown request {request!r}")

    # ---------------------------------------------------------------- physics
    def tick(self) -> None:
        cmd = self.channel.cmd
        commanded = (self.sim_state == shared.ACTIVE and self.controller is not None
                     and (any(cmd["kp"]) or any(cmd["kd"])))
        if commanded:
            self.channel.compute_ctrl(self.data)
        else:
            self.limp.apply(self.data)
        mujoco.mj_step(self.model, self.data)
        self.channel.read_state(self.data, False)
        self.channel.publish(self.data)
        self.core.modules[0].on_publish(self.core)

    def velocities(self) -> tuple[float, float]:
        rotation = self.data.xmat[self.base].reshape(3, 3)
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        vx = float(np.cos(yaw) * self.data.qvel[0] + np.sin(yaw) * self.data.qvel[1])
        wz = float((rotation @ self.data.qvel[3:6])[2])
        return vx, wz

    def publish_gui(self, rtf: float) -> None:
        vx, wz = self.velocities()
        controller = shared.NONE
        if self.controller is not None:
            controller = shared.RUNNING if self.controller.poll() is None else shared.EXITED
        elif self.exit_code:
            controller = shared.EXITED
        self.block.write(shared.SimState(time.monotonic(), float(self.data.time), self.sim_state, controller,
                                         self.exit_code, self.seed, self.slow, rtf, vx, wz, self.spawn_id,
                                         self.data.qpos))

    def run(self) -> None:
        next_gui = next_poll = 0.
        rtf_wall, rtf_sim, rtf = time.monotonic(), float(self.data.time), 1.
        self.spawn(None)
        anchor_wall, anchor_sim = time.monotonic(), float(self.data.time)
        while self.running:
            now = time.monotonic()
            if now >= next_poll:
                next_poll = now + .01
                try:
                    while True:
                        self.handle(self.requests.get_nowait())
                        anchor_wall, anchor_sim = time.monotonic(), float(self.data.time)
                except queue.Empty:
                    pass
                if self.pending is not None:
                    self.try_spawn()
                    anchor_wall, anchor_sim = time.monotonic(), float(self.data.time)
                elif self.sim_state == shared.SETTLING and self.data.time >= self.settle_until:
                    self.sim_state = shared.HEAP
                    if self.controller is None:
                        self.start_controller()
                    self.sim_state = shared.ACTIVE
                    emit("settled", seed=self.seed, base_z=float(self.data.qpos[2]))
                    anchor_wall, anchor_sim = time.monotonic(), float(self.data.time)
                if self.controller is not None and (code := self.controller.poll()) is not None:
                    emit("controller_exited", code=code)
                    self.exit_code = code if code else -999
                    self.controller = None
                    self.channel.cmd = {key: [0.] * self.channel.n for key in ("q", "dq", "tau", "kp", "kd")}
            if self.sim_state == shared.SUSPENDED:
                self.publish_gui(0.)
                time.sleep(.01)
                anchor_wall, anchor_sim = time.monotonic(), float(self.data.time)
                continue
            ahead = anchor_sim + (time.monotonic() - anchor_wall) - float(self.data.time)
            if ahead < 0:
                time.sleep(min(-ahead, .002))
                continue
            if ahead > .05:  # a host stall: drop the backlog instead of fast-forwarding
                self.slow += 1
                emit("sim_slow", behind_ms=round(1e3 * ahead, 1), sim_t=round(float(self.data.time), 3))
                anchor_wall, anchor_sim = time.monotonic(), float(self.data.time)
            self.tick()
            if self.data.time >= next_gui:
                next_gui = float(self.data.time) + 1. / 60.
                wall = time.monotonic()
                if wall - rtf_wall >= .5:
                    rtf = (float(self.data.time) - rtf_sim) / (wall - rtf_wall)
                    rtf_wall, rtf_sim = wall, float(self.data.time)
                self.publish_gui(rtf)
            if not np.isfinite(self.data.qpos).all():
                emit("error", message="non-finite MuJoCo state; resetting")
                self.spawn(None)

    def close(self) -> None:
        self.stop_controller()
        workers = list(getattr(self.heaps, "_processes", {}).values())
        self.heaps.shutdown(wait=False, cancel_futures=True)
        for worker in workers:
            worker.terminate()
        self.scan_path.unlink(missing_ok=True)
        self.block.close()
        self.live.close()
        for path in (self.args.state, self.args.live):  # also when the GUI died without cleaning up
            Path(path).unlink(missing_ok=True)
        emit("closed", dropped_events=_DROPPED[0])
        flush_events()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", type=Path, required=True, help="live command file (created by the GUI)")
    parser.add_argument("--state", type=Path, required=True, help="state block file to create")
    parser.add_argument("--policy", choices=("base", "base_blind"), default="base")
    args = parser.parse_args()
    def interrupt(*_: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    server = Server(args)
    try:
        server.run()
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)  # the SDK's native threads would keep a normal interpreter exit waiting
