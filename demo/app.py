"""TRON2 Stand-up & Drive - presentation demo on the real deploy stack.

One window: the 3D view (MuJoCo wf_payload, plane floor) and a docked control panel. The physics and the
deploy controller (tron2_deploy/run_policy.py: learned stand-up -> hand-off -> Base, GT height scan) run in
their own processes (demo/sim_server.py); this process only renders, takes input and writes live commands
(tron2_deploy/live_commands.py). Log: ~/tron2_demo.log.

  uv run --extra sdk --extra demo python demo/app.py [--selftest] [--batch N] [--fullscreen]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
DEMO = Path(__file__).resolve().parent
SIM = DEMO.parent
for path in (SIM.parent, SIM, DEMO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import limits as limits_module  # noqa: E402
import numpy as np  # noqa: E402
import shared  # noqa: E402
from panel import ControlPanel  # noqa: E402
from PySide6.QtCore import QEvent, QObject, Qt, QTimer  # noqa: E402
from PySide6.QtGui import QFont, QFontDatabase, QIcon, QKeyEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QMainWindow  # noqa: E402
from tron2_deploy.live_commands import (  # noqa: E402
    MODE_IDLE,
    MODE_RUN,
    STATE_IDLE,
    STATE_LATCHED,
    STATE_RUNNING,
    Command,
    LiveChannel,
)
from viewport import Viewport  # noqa: E402

LOG_PATH = Path.home() / "tron2_demo.log"
ICON = DEMO / "assets/tron2_icon.png"
TITLE = "TRON2 Stand-up & Drive"
HANDOFF_LABEL_S = 2.
log = logging.getLogger("tron2_demo")

PHASES = {0: "경로 계획 중...", 1: "다리 정리", 2: "다리 정리", 3: "다리 정리", 4: "초기자세 A'", 5: "기립"}
COLORS = {"wait": "#8a93a6", "work": "#f28c28", "ok": "#3ccf7a", "bad": "#ff4d4f", "heap": "#5aa9ff"}


def tilt_of(qpos: np.ndarray) -> float:
    return float(np.arccos(np.clip(1. - 2. * (qpos[4] ** 2 + qpos[5] ** 2), -1., 1.)))


class SimLink:
    """The sim server process, its event stream, the state block and the live command file."""

    def __init__(self, policy: str) -> None:
        scratch = Path("/dev/shm") if Path("/dev/shm").is_dir() else Path("/tmp")
        self.live_path = scratch / f"tron2-demo-live-{os.getpid()}.bin"
        self.state_path = scratch / f"tron2-demo-state-{os.getpid()}.bin"
        self.live = LiveChannel(self.live_path, create=True)
        self.events: queue.Queue[dict] = queue.Queue()
        self.block: shared.StateBlock | None = None
        command = [sys.executable, "-u", str(DEMO / "sim_server.py"), "--live", str(self.live_path),
                   "--state", str(self.state_path), "--policy", policy]
        from sim_server import die_with_parent

        self.process = subprocess.Popen(command, cwd=SIM, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True, bufsize=1, preexec_fn=die_with_parent)
        threading.Thread(target=self._read, daemon=True).start()
        log.info("sim server pid %d: %s", self.process.pid, " ".join(command))

    def _read(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            line = line.rstrip()
            try:
                self.events.put(json.loads(line))
            except json.JSONDecodeError:
                self.events.put({"event": "text", "line": line})
        self.events.put({"event": "server_exit", "code": self.process.wait()})

    def send(self, **request) -> None:
        if self.process.poll() is None and self.process.stdin is not None:
            try:
                self.process.stdin.write(json.dumps(request) + "\n")
                self.process.stdin.flush()
            except (BrokenPipeError, OSError) as error:
                log.warning("sim server request failed: %s", error)

    def state(self) -> shared.SimState | None:
        if self.block is None:
            if not self.state_path.exists() or self.state_path.stat().st_size <= 128:
                return None
            self.block = shared.StateBlock(self.state_path)
        return self.block.read()

    def close(self) -> None:
        self.live.write_command(Command(time.monotonic(), MODE_IDLE, 0))
        self.send(cmd="quit")
        try:
            if self.process.stdin is not None:
                self.process.stdin.close()
        except OSError:
            pass
        try:
            self.process.wait(timeout=8.)
        except subprocess.TimeoutExpired:
            log.warning("sim server did not quit; terminating")
            self.process.terminate()
            try:
                self.process.wait(timeout=4.)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.block is not None:
            self.block.close()
        self.live.close()
        for path in (self.live_path, self.state_path):
            path.unlink(missing_ok=True)


class KeyFilter(QObject):
    """Application-wide arrow/space handling, whichever widget has the focus."""

    KEYS = (Qt.Key.Key_Up, Qt.Key.Key_Down, Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Space)

    def __init__(self, window: MainWindow) -> None:
        super().__init__(window)
        self.window = window

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if event.type() in (QEvent.Type.KeyPress, QEvent.Type.ShortcutOverride) and isinstance(event, QKeyEvent) \
                and event.key() in self.KEYS:
            if event.type() == QEvent.Type.KeyPress and not event.isAutoRepeat():
                self.window.drive_key(event.key())
            event.accept()
            return True
        return False


class MainWindow(QMainWindow):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.args = args
        self.setWindowTitle(TITLE)
        if ICON.exists():
            self.setWindowIcon(QIcon(str(ICON)))
        self.limits = limits_module.load(args.limits)
        log.info("command limits: %s", self.limits)
        self.cap_vx, self.cap_wz = self.limits.cap_vx, self.limits.cap_wz
        self.vx = self.wz = 0.
        self.mode, self.token = MODE_IDLE, 0
        self.link = SimLink(args.policy)
        self.viewport = Viewport(self)
        self.setCentralWidget(self.viewport)
        self.panel = ControlPanel(self, self.limits)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.panel.dock)
        self.sim: shared.SimState | None = None
        self.status = self.link.live.read_status()
        self.spawn_pending: int | None = None  # spawn_id at the time of the request
        self.planned = False
        self.handoff_seen: float | None = None
        self.vel_filter = np.zeros(2)
        self.ui_phase, self.ui_color, self.ui_detail = "시뮬레이터 준비 중", COLORS["wait"], ""
        self.run: dict | None = None
        self.server_exit: int | None = None
        self.last_latch = ""
        self.closing = False
        self.timer = QTimer(self, timeout=self.tick)
        self.timer.start(16)
        self.command_timer = QTimer(self, timeout=self.write_command)
        self.command_timer.start(50)
        self.key_filter = KeyFilter(self)
        QApplication.instance().installEventFilter(self.key_filter)

    # ------------------------------------------------------------------ actions
    def spawn(self, seed: int | None = None) -> None:
        self.stop_driving()
        self.mode = MODE_IDLE
        self.write_command()
        self.spawn_pending = self.sim.spawn_id if self.sim is not None else 0
        self.planned, self.handoff_seen, self.run = False, None, None
        if seed is None:
            seed = int(np.random.SeedSequence().entropy % 1_000_000)
        log.info("SPAWN seed=%d", seed)
        self.link.send(cmd="spawn", seed=seed)

    def stand_up(self) -> None:
        if not self.ready_idle():
            return
        self.token += 1
        self.mode = MODE_RUN
        self.planned, self.handoff_seen = False, None
        self.run = {"seed": self.sim.seed if self.sim else None, "pressed": time.monotonic(), "phase_t": {},
                    "route": None, "latch": None, "handoff": None, "success": None}
        log.info("STANDUP token=%d seed=%s", self.token, self.run["seed"])
        self.write_command()

    def reset(self) -> None:
        self.stop_driving()
        self.mode = MODE_IDLE
        self.write_command()
        self.spawn_pending = self.sim.spawn_id if self.sim is not None else 0
        self.planned, self.handoff_seen, self.run = False, None, None
        self.viewport.reset_camera()
        log.info("RESET")
        self.link.send(cmd="reset")

    def set_drive_mode(self, enabled: bool) -> None:
        if not enabled:
            self.stop_driving(keep_mode=True)
        log.info("DRIVE_MODE %s", enabled)

    def stop_driving(self, keep_mode: bool = False) -> None:
        self.vx = self.wz = 0.
        if not keep_mode:
            self.panel.drive.setChecked(False)

    def set_caps(self, cap_vx: float, cap_wz: float) -> None:
        self.cap_vx, self.cap_wz = cap_vx, cap_wz
        self.vx = float(np.clip(self.vx, -cap_vx, cap_vx))
        self.wz = float(np.clip(self.wz, -cap_wz, cap_wz))

    def can_drive(self) -> bool:
        return (self.panel.drive.isChecked() and self.status.state == STATE_RUNNING and self.status.phase == 6
                and self.mode == MODE_RUN and not self.stale())

    def drive_key(self, key: Qt.Key | int) -> None:
        key = Qt.Key(key)
        if key == Qt.Key.Key_Space:
            self.vx = self.wz = 0.
            log.info("KEY stop")
            return
        if not self.can_drive():
            return
        step = {Qt.Key.Key_Up: (self.limits.vx_step, 0.), Qt.Key.Key_Down: (-self.limits.vx_step, 0.),
                Qt.Key.Key_Left: (0., self.limits.wz_step), Qt.Key.Key_Right: (0., -self.limits.wz_step)}[key]
        self.vx = float(np.clip(round(self.vx + step[0], 6), -self.cap_vx, self.cap_vx))
        self.wz = float(np.clip(round(self.wz + step[1], 6), -self.cap_wz, self.cap_wz))
        log.info("KEY %s -> vx=%.2f wz=%.2f", Qt.Key(key).name, self.vx, self.wz)

    def write_command(self) -> None:
        driving = self.can_drive()
        self.link.live.write_command(Command(time.monotonic(), self.mode, self.token,
                                             self.vx if driving else 0., 0., self.wz if driving else 0.,
                                             self.cap_vx, self.cap_wz))

    # ------------------------------------------------------------------ state
    def stale(self) -> bool:
        return time.monotonic() - self.status.stamp > .5

    def ready_idle(self) -> bool:
        return (self.sim is not None and self.sim.sim_state == shared.ACTIVE and self.spawn_pending is None
                and self.status.state == STATE_IDLE and not self.stale() and self.sim.controller == shared.RUNNING)

    def standing(self) -> bool:
        return self.ui_phase == "서 있음"

    def handle_event(self, event: dict) -> None:
        kind = event.get("event")
        if kind == "controller":
            line = event.get("line", "")
            log.info("CTRL %s", line)
            if line.startswith("ROUTE_INFO"):
                self.planned = True
                if self.run is not None:
                    try:
                        info = json.loads(line.split(" ", 1)[1])
                        self.run["route"] = {k: info.get(k) for k in ("families", "duration_s", "order",
                                                                      "contact_fallback")}
                    except (IndexError, ValueError):
                        pass
            elif "DAMPING_LATCH" in line and self.run is not None and self.run["latch"] is None:
                self.run["latch"] = line.split("DAMPING_LATCH:", 1)[1].strip()
            return
        if kind == "server_exit":
            self.server_exit = event.get("code")
        log.info("SIM %s", json.dumps(event, ensure_ascii=False)[:1500])

    def update_phase(self) -> None:
        sim, status, now = self.sim, self.status, time.monotonic()
        detail = ""
        if self.server_exit is not None:
            phase, color, detail = "시뮬레이터 종료됨", COLORS["bad"], f"exit {self.server_exit} (로그 확인)"
        elif sim is not None and sim.sim_state == shared.SUSPENDED:
            phase, color, detail = "일시 정지", COLORS["wait"], "다른 limxsdk 시뮬레이터가 실행 중 (포트 공유)"
        elif sim is None or sim.sim_state == shared.BOOT:
            phase, color = "시뮬레이터 준비 중", COLORS["wait"]
        elif self.spawn_pending is not None or sim.sim_state == shared.SETTLING:
            phase, color = "소환 중 (힘 빠진 채 내려놓기)", COLORS["heap"]
        elif sim.controller == shared.EXITED:
            phase, color, detail = "제어기 종료", COLORS["bad"], "[소환] 또는 [리셋]으로 다시 시작"
        elif sim.controller != shared.RUNNING or self.stale() or status.state not in (STATE_IDLE, STATE_RUNNING,
                                                                                         STATE_LATCHED):
            phase, color = "제어기 준비 중", COLORS["wait"]
        elif status.state == STATE_LATCHED:
            phase, color, detail = "안전 정지", COLORS["bad"], status.reason
        elif status.state == STATE_IDLE or self.mode != MODE_RUN:
            phase, color, detail = "널브러진 상태", COLORS["heap"], "[기립]을 누르세요"
        elif status.phase == 6:
            self.handoff_seen = self.handoff_seen or now
            if now - self.handoff_seen < HANDOFF_LABEL_S:
                phase, color = "균형 인계", COLORS["work"]
            else:
                phase, color = "서 있음", COLORS["ok"]
                detail = "주행 모드: 방향키 / 화살표 패드" if self.panel.drive.isChecked() else "[주행 모드]로 운전"
        elif status.phase == 0 and self.planned:
            phase, color = "다리 정리", COLORS["work"]
        else:
            phase, color = PHASES.get(status.phase, f"phase {status.phase}"), COLORS["work"]
        if self.run is not None:
            self.run["phase_t"].setdefault(phase, round(now - self.run["pressed"], 2))
            if phase == "균형 인계" and self.run["handoff"] is None:
                self.run["handoff"] = round(now - self.run["pressed"], 2)
        if status.state == STATE_LATCHED and status.reason != self.last_latch:
            self.last_latch = status.reason
            log.warning("LATCH %s", status.reason)
        if phase != self.ui_phase:
            log.info("PHASE %s %s", phase, detail)
        self.ui_phase, self.ui_color, self.ui_detail = phase, color, detail

    def tick(self) -> None:
        if self.closing:
            return
        while True:
            try:
                self.handle_event(self.link.events.get_nowait())
            except queue.Empty:
                break
        sim = self.link.state()
        if sim is not None:
            if self.spawn_pending is not None and sim.spawn_id != self.spawn_pending:
                self.spawn_pending = None
            self.sim = sim
            self.vel_filter += .15 * (np.array([sim.vx, sim.wz]) - self.vel_filter)
        status = self.link.live.read_status()
        if (status.phase, status.state) != (self.status.phase, self.status.state):
            log.info("STATUS phase=%d state=%d reason=%r", status.phase, status.state, status.reason)
        self.status = status
        self.update_phase()
        driving = self.can_drive()
        self.viewport.hud = {"phase": self.ui_phase, "color": self.ui_color, "detail": self.ui_detail,
                             "speed": (self.vx if driving else 0., self.vel_filter[0], self.wz if driving else 0.,
                                       self.vel_filter[1]) if self.status.phase == 6 else None}
        self.viewport.update_pose(None if sim is None else sim.qpos)
        self.panel.refresh(self)

    # ------------------------------------------------------------------ shutdown
    def closeEvent(self, event) -> None:
        if not self.closing:
            self.closing = True
            log.info("closing")
            self.timer.stop()
            self.command_timer.stop()
            self.link.close()
            self.viewport.close_renderer()
            log.info("closed cleanly")
        event.accept()


def setup_logging() -> None:
    handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    if sys.stderr.isatty():  # the desktop launcher appends stderr to the same log file
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        root.addHandler(stream)
    log.info("=" * 20 + " %s start pid %d argv %s", TITLE, os.getpid(), sys.argv)


def pick_font(app: QApplication) -> None:
    families = set(QFontDatabase.families())
    for name in ("Noto Sans CJK KR", "NanumSquare_ac", "NanumSquare", "Noto Sans KR", "NanumGothic"):
        if name in families:
            font = QFont(name)
            font.setPointSize(11)
            app.setFont(font)
            return


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", choices=("base", "base_blind"), default="base",
                        help="locomotion policy after the hand-off (default Base with the GT height scan)")
    parser.add_argument("--limits", type=Path, default=limits_module.DEFAULT_PATH)
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--selftest", action="store_true", help="press the buttons/keys and record the window")
    parser.add_argument("--video", type=Path,
                        default=Path("/home/eunwon/tron2_ws/artifacts/basecampaign/demo/demo_selftest.mp4"))
    parser.add_argument("--batch", type=int, default=0, help="N x (spawn -> stand up -> 10 s scripted drive)")
    parser.add_argument("--out", type=Path, default=Path("/home/eunwon/tron2_ws/artifacts/basecampaign/demo"))
    parser.add_argument("--seed", type=int, help="first spawn seed for --selftest/--batch (default random)")
    args = parser.parse_args()
    setup_logging()
    try:
        cpus = sorted(os.sched_getaffinity(0))
        if len(cpus) >= 8:
            os.sched_setaffinity(0, set(cpus[:-5]))  # sim_server/controller take the last five
    except (AttributeError, OSError):
        pass
    app = QApplication(sys.argv[:1])
    app.setApplicationName(TITLE)
    app.setDesktopFileName("tron2-demo")
    pick_font(app)
    window = MainWindow(args)
    window.resize(1600, 960)
    if args.fullscreen:
        window.showFullScreen()
    else:
        window.show()
    signal.signal(signal.SIGINT, lambda *_: window.close())
    signal.signal(signal.SIGTERM, lambda *_: window.close())
    keepalive = QTimer(interval=200, timeout=lambda: None)  # lets Python run signal handlers
    keepalive.start()
    script = None
    if args.selftest or args.batch:
        from automation import Automation

        script = Automation(window, args)
        script.start()
    code = app.exec()
    if script is not None:
        code = script.exit_code
    log.info("exit %d", code)
    return code


if __name__ == "__main__":
    status = main()
    logging.shutdown()
    sys.stdout.flush()
    os._exit(status)
