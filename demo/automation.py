"""--selftest / --batch: drive the real window through its own buttons and (posted) key events.

The script is a generator: it yields a number (seconds to wait) or a (predicate, timeout_s) pair; a 50 ms
timer advances it. Buttons are clicked with QPushButton.click() and keys go through the application event
filter exactly like a keyboard press.
"""
from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Generator
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication

from app import MainWindow, tilt_of
from tron2_deploy.live_commands import STATE_LATCHED

log = logging.getLogger("tron2_demo.auto")
UP, DOWN, LEFT, RIGHT, SPACE = Qt.Key.Key_Up, Qt.Key.Key_Down, Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Space
# The 10 s reliability drive (seconds after drive mode is on): forward 0.5 -> 1.0, turn left 0.25 -> 0.5,
# straight, slow down 0.5 -> 0, stop.
DRIVE_10S = ((0., UP), (1.5, UP), (4., LEFT), (5., LEFT), (6.5, RIGHT), (6.6, RIGHT), (7.5, DOWN), (8.5, DOWN),
             (9.5, SPACE))
# The longer selftest drive: forward, curve left, curve right, back up, stop.
DRIVE_SELFTEST = ((0., UP), (1.5, UP), (4., LEFT), (4.2, LEFT), (7., RIGHT), (7.2, RIGHT), (7.4, RIGHT),
                  (7.6, RIGHT), (10.5, LEFT), (10.7, LEFT), (12., DOWN), (12.5, DOWN), (14., DOWN),
                  (17., UP), (18., SPACE))
Wait = float | tuple[Callable[[], bool], float]


class Timeout(Exception):
    pass


class Automation:
    def __init__(self, window: MainWindow, args) -> None:
        self.window, self.args = window, args
        self.exit_code = 1
        self.recorder = None
        self.generator = self.selftest() if args.selftest else self.batch(args.batch)
        self.wait: tuple[Callable[[], bool], float, float] | None = None
        self.timer = QTimer(window, timeout=self.step)
        self.rng = np.random.default_rng(args.seed)

    def start(self) -> None:
        self.timer.start(50)

    # ------------------------------------------------------------------ driver
    def step(self) -> None:
        try:
            if self.wait is not None:
                predicate, deadline, _ = self.wait
                if not predicate():
                    if time.monotonic() > deadline:
                        self.wait = None
                        self.generator.throw(Timeout())
                    return
                self.wait = None
            request = next(self.generator)
            if isinstance(request, (int, float)):
                until = time.monotonic() + float(request)
                self.wait = (lambda: time.monotonic() >= until, until + 1., 0.)
            else:
                predicate, timeout = request
                self.wait = (predicate, time.monotonic() + timeout, 0.)
        except StopIteration:
            self.finish()
        except Exception:
            log.exception("automation failed")
            self.finish()

    def finish(self) -> None:
        self.timer.stop()
        if self.recorder is not None:
            self.recorder.stop()
            log.info("video %s frames=%d dropped=%d", self.recorder.path, self.recorder.count, self.recorder.dropped)
        QTimer.singleShot(500, self.window.close)

    def press(self, key: Qt.Key) -> None:
        for kind in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
            QApplication.postEvent(self.window, QKeyEvent(kind, key, Qt.KeyboardModifier.NoModifier))

    def seed(self, index: int) -> int:
        return (self.args.seed + index) if self.args.seed is not None else int(self.rng.integers(1_000_000))

    # ------------------------------------------------------------------ pieces
    def wait_idle(self, timeout: float) -> Generator[Wait, None, None]:
        yield (self.window.ready_idle, timeout)

    def stand(self, timeout: float = 150.) -> Generator[Wait, None, bool]:
        w = self.window
        w.panel.stand.click()
        latched = lambda: w.status.state == STATE_LATCHED  # noqa: E731
        try:
            yield (lambda: w.standing() or latched() or w.server_exit is not None, timeout)
        except Timeout:
            return False
        return w.standing()

    def drive(self, sequence, snapshots: dict[float, str] | None = None) -> Generator[Wait, None, dict]:
        w = self.window
        w.panel.drive.click()
        yield .3
        start = time.monotonic()
        positions, tilts, track = [], [], []
        pending = list(sequence)
        shots = sorted((snapshots or {}).items())
        end = sequence[-1][0] + 1.5
        while time.monotonic() - start < end:
            elapsed = time.monotonic() - start
            while pending and pending[0][0] <= elapsed:
                self.press(pending.pop(0)[1])
            while shots and shots[0][0] <= elapsed and self.recorder is not None:
                self.recorder.snapshot(shots.pop(0)[1])
            if w.sim is not None:
                positions.append(w.sim.qpos[:2].copy())
                tilts.append(tilt_of(w.sim.qpos))
                track.append((round(elapsed, 2), w.vx if w.can_drive() else 0., w.sim.vx, w.wz if w.can_drive() else 0.,
                              w.sim.wz))
            if w.status.state == STATE_LATCHED:
                break
            yield .05
        path = float(np.sum(np.linalg.norm(np.diff(np.array(positions), axis=0), axis=1))) if len(positions) > 1 else 0.
        cmd = np.array([t[1] for t in track]) if track else np.zeros(1)
        act = np.array([t[2] for t in track]) if track else np.zeros(1)
        return {"path_m": round(path, 3), "tilt_max": round(max(tilts, default=0.), 3),
                "final_tilt": round(tilts[-1], 3) if tilts else None,
                "final_z": round(float(w.sim.qpos[2]), 3) if w.sim is not None else None,
                "vx_cmd_max": round(float(cmd.max()), 3), "vx_act_max": round(float(act.max()), 3),
                "latched": w.status.state == STATE_LATCHED, "track_tail": track[-5:]}

    # ------------------------------------------------------------------ scripts
    def batch(self, count: int) -> Generator[Wait, None, None]:
        w = self.window
        out = Path(self.args.out)
        out.mkdir(parents=True, exist_ok=True)
        results_path = out / "reliability.jsonl"
        results = []
        yield from self.wait_idle(120.)
        for index in range(count):
            seed = self.seed(index)
            record = {"run": index, "seed": seed, "started": time.strftime("%H:%M:%S")}
            w.spawn(seed)
            try:
                yield (lambda: w.spawn_pending is None, 15.)
                yield from self.wait_idle(60.)
            except Timeout:
                record.update(outcome="fail_spawn", reason="controller not idle after spawn")
                results.append(record)
                self._save(results_path, results)
                continue
            yield .5
            stood = yield from self.stand()
            run = w.run or {}
            record.update(phase_t=run.get("phase_t"), route=run.get("route"), latch=w.status.reason or None)
            if not stood:
                record.update(outcome="fail_standup", phase=w.ui_phase, reason=w.status.reason or w.ui_phase)
                results.append(record)
                self._save(results_path, results)
                log.info("BATCH run %d seed %d: %s", index, seed, record["outcome"])
                continue
            yield 1.
            drive = yield from self.drive(DRIVE_10S)
            ok = not drive["latched"] and drive["final_tilt"] is not None and drive["final_tilt"] < .5 \
                and drive["final_z"] > .3 and drive["path_m"] > 1.
            record.update(drive=drive, outcome="stood_and_drove" if ok else "fail_drive",
                          latch=w.status.reason or None)
            results.append(record)
            self._save(results_path, results)
            log.info("BATCH run %d seed %d: %s path %.2f m", index, seed, record["outcome"], drive["path_m"])
        self.exit_code = 0

    def _save(self, path: Path, results: list) -> None:
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results))
        good = sum(r.get("outcome") == "stood_and_drove" for r in results)
        (path.parent / "reliability_summary.json").write_text(json.dumps(
            {"runs": len(results), "stood_and_drove": good,
             "outcomes": {o: sum(r.get("outcome") == o for r in results) for o in {r.get("outcome") for r in results}}},
            indent=2))

    def selftest(self) -> Generator[Wait, None, None]:
        from recorder import WindowRecorder

        w = self.window
        yield from self.wait_idle(120.)
        yield 1.
        self.recorder = WindowRecorder(w, self.args.video)
        self.recorder.start()
        yield 3.
        self.recorder.snapshot("01_pose1_idle")
        w.spawn(self.seed(0))
        yield (lambda: w.spawn_pending is None, 15.)
        yield from self.wait_idle(60.)
        yield 2.
        self.recorder.snapshot("02_heap")
        w.panel.stand.click()
        shots = {"경로 계획 중...": "03_planning", "다리 정리": "04_legs", "초기자세 A'": "05_a_prime",
                 "기립": "06_standup", "균형 인계": "07_handoff"}
        deadline = time.monotonic() + 150.
        while not w.standing() and time.monotonic() < deadline and w.status.state != STATE_LATCHED:
            if w.ui_phase in shots:
                self.recorder.snapshot(shots.pop(w.ui_phase))
            yield .1
        if not w.standing():
            log.error("selftest: did not stand (%s %s)", w.ui_phase, w.status.reason)
            self.recorder.snapshot("99_failure")
            yield 3.
            return
        yield 2.
        self.recorder.snapshot("08_standing")
        result = yield from self.drive(DRIVE_SELFTEST, {3.: "09_driving", 9.: "10_turning", 15.: "11_reverse"})
        yield 2.
        self.recorder.snapshot("12_stopped")
        log.info("selftest drive %s", result)
        w.panel.reset.click()
        yield 3.5
        self.recorder.snapshot("13_reset")
        self.exit_code = 0 if not result["latched"] else 1
