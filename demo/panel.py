"""Docked control panel: action buttons, phase, arrow pad, commanded vs actual speed, software cap."""
from __future__ import annotations

import time
from typing import TYPE_CHECKING

import shared
from limits import Limits
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QDockWidget,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from app import MainWindow

STYLE = """
QWidget#panel { background: #1b1e24; color: #e8eaee; }
QLabel { color: #e8eaee; }
QLabel#title { font-size: 20px; font-weight: 800; color: #ffffff; }
QLabel#subtitle { font-size: 12px; color: #8a93a6; }
QLabel#section { font-size: 12px; font-weight: 700; color: #8a93a6; letter-spacing: 1px; margin-top: 6px; }
QLabel#phase { font-size: 24px; font-weight: 800; padding: 10px 12px; border-radius: 10px; background: #262a33; }
QLabel#detail { font-size: 12px; color: #aab2c0; }
QLabel#latch { background: #4a1517; color: #ffd6d6; border: 1px solid #ff4d4f; border-radius: 8px; padding: 8px;
               font-size: 13px; }
QLabel#small { font-size: 11px; color: #7d8696; }
QPushButton { background: #2c313c; color: #f2f4f7; border: 1px solid #3a4150; border-radius: 10px;
              padding: 12px 10px; font-size: 16px; font-weight: 700; }
QPushButton:hover { background: #353b48; }
QPushButton:pressed { background: #23272f; }
QPushButton:disabled { color: #5d6575; background: #23262d; border-color: #2b2f37; }
QPushButton#primary { background: #f28c28; border-color: #f7a24d; color: #1b1e24; }
QPushButton#primary:disabled { background: #4a3b2b; color: #8a7a66; border-color: #4a3b2b; }
QPushButton#drive:checked { background: #1f6f45; border-color: #3ccf7a; }
QToolButton { background: #2c313c; color: #ffffff; border: 1px solid #3a4150; border-radius: 12px;
              font-size: 26px; font-weight: 800; min-width: 64px; min-height: 56px; }
QToolButton:hover { background: #3a4150; }
QToolButton:pressed { background: #f28c28; color: #1b1e24; }
QToolButton:disabled { color: #4d5462; background: #22252c; border-color: #2a2e36; }
QSlider::groove:horizontal { height: 6px; background: #2c313c; border-radius: 3px; }
QSlider::handle:horizontal { background: #f28c28; width: 16px; margin: -6px 0; border-radius: 8px; }
QSlider::sub-page:horizontal { background: #8a5a2a; border-radius: 3px; }
"""


class SpeedBar(QWidget):
    """Commanded (orange marker) vs actual (filled bar) on a symmetric scale."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(18)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.command = self.actual = 0.
        self.scale = 1.
        self.cap = 1.

    def set_values(self, command: float, actual: float, cap: float, scale: float) -> None:
        self.command, self.actual, self.cap, self.scale = command, actual, cap, max(scale, 1e-3)
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        mid = w / 2
        painter.fillRect(QRectF(0, h * .3, w, h * .4), QColor("#2c313c"))
        x = lambda v: mid + max(-1., min(1., v / self.scale)) * (w / 2 - 2)  # noqa: E731
        painter.fillRect(QRectF(min(mid, x(self.actual)), h * .3, abs(x(self.actual) - mid), h * .4), QColor("#5aa9ff"))
        painter.fillRect(QRectF(x(-self.cap) - 1, 0, 2, h), QColor("#6b7383"))
        painter.fillRect(QRectF(x(self.cap) - 1, 0, 2, h), QColor("#6b7383"))
        painter.fillRect(QRectF(mid - .5, h * .2, 1, h * .6), QColor("#8a93a6"))
        painter.setBrush(QColor("#f28c28"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(QRectF(x(self.command) - 3, 0, 6, h), 2, 2)
        painter.end()


class ControlPanel:
    def __init__(self, window: MainWindow, limits: Limits) -> None:
        self.window = window
        self.limits = limits
        self.dock = QDockWidget("제어 패널", window)
        self.dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        self.dock.setTitleBarWidget(QWidget())
        root = QWidget()
        root.setObjectName("panel")
        root.setStyleSheet(STYLE)
        root.setMinimumWidth(360)
        root.setMaximumWidth(420)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(18, 18, 18, 14)
        layout.setSpacing(8)
        title = QLabel("TRON2 Stand-up & Drive")
        title.setObjectName("title")
        subtitle = QLabel("MuJoCo · 학습 기립 정책 → Base 정책 · 실제 배포 제어기")
        subtitle.setObjectName("subtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        self.phase = QLabel("시뮬레이터 준비 중")
        self.phase.setObjectName("phase")
        self.phase.setWordWrap(True)
        self.detail = QLabel("")
        self.detail.setObjectName("detail")
        self.detail.setWordWrap(True)
        layout.addWidget(self.phase)
        layout.addWidget(self.detail)

        self.spawn = self._button("소환 (널브러진 상태)", window.spawn)
        self.stand = self._button("기립", window.stand_up, "primary")
        self.drive = self._button("주행 모드", None, "drive")
        self.drive.setCheckable(True)
        self.drive.toggled.connect(window.set_drive_mode)
        self.reset = self._button("리셋", window.reset)
        grid = QGridLayout()
        grid.setSpacing(8)
        grid.addWidget(self.spawn, 0, 0, 1, 2)
        grid.addWidget(self.stand, 1, 0, 1, 2)
        grid.addWidget(self.drive, 2, 0)
        grid.addWidget(self.reset, 2, 1)
        layout.addLayout(grid)

        layout.addWidget(self._section("주행 (방향키 · Space 정지)"))
        pad = QGridLayout()
        pad.setSpacing(6)
        self.pad = {}
        for key, text, row, col in ((Qt.Key.Key_Up, "▲", 0, 1), (Qt.Key.Key_Left, "◀", 1, 0),
                                    (Qt.Key.Key_Space, "■", 1, 1), (Qt.Key.Key_Right, "▶", 1, 2),
                                    (Qt.Key.Key_Down, "▼", 2, 1)):
            button = QToolButton()
            button.setText(text)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.clicked.connect(lambda _=False, k=key: window.drive_key(k))
            pad.addWidget(button, row, col, Qt.AlignmentFlag.AlignCenter)
            self.pad[key] = button
        pad_box = QHBoxLayout()
        pad_box.addStretch()
        pad_box.addLayout(pad)
        pad_box.addStretch()
        layout.addLayout(pad_box)
        hint = QLabel(f"▲/▼ 전후진 ±{limits.vx_step:g} m/s · ◀/▶ 좌/우회전 ±{limits.wz_step:g} rad/s "
                      "(누를 때마다 누적)")
        hint.setObjectName("small")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        layout.addWidget(self._section("명령 vs 실제"))
        self.vx_label, self.vx_bar = QLabel(), SpeedBar()
        self.wz_label, self.wz_bar = QLabel(), SpeedBar()
        for label, bar in ((self.vx_label, self.vx_bar), (self.wz_label, self.wz_bar)):
            layout.addWidget(label)
            layout.addWidget(bar)
        legend = QLabel("주황 표식 = 명령, 파란 막대 = 실제, 회색 선 = 상한 · "
                        "정책 = 배포 제어기의 명령 정형(상한·가속 제한) 후 값")
        legend.setWordWrap(True)
        legend.setObjectName("small")
        layout.addWidget(legend)

        layout.addWidget(self._section("소프트웨어 상한"))
        self.cap_label = QLabel()
        layout.addWidget(self.cap_label)
        self.cap_vx = self._slider(limits.cap_vx, limits.max_vx)
        self.cap_wz = self._slider(limits.cap_wz, limits.max_wz)
        for name, slider in (("vx", self.cap_vx), ("wz", self.cap_wz)):
            row = QHBoxLayout()
            label = QLabel(name)
            label.setFixedWidth(24)
            row.addWidget(label)
            row.addWidget(slider)
            layout.addLayout(row)
        self.latch = QLabel("")
        self.latch.setObjectName("latch")
        self.latch.setWordWrap(True)
        self.latch.hide()
        layout.addWidget(self.latch)
        layout.addStretch()
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #2c313c;")
        layout.addWidget(line)
        self.status = QLabel("")
        self.status.setObjectName("small")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.dock.setWidget(root)

    def _section(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("section")
        return label

    def _button(self, text: str, slot, name: str = "") -> QPushButton:
        button = QPushButton(text)
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        if name:
            button.setObjectName(name)
        if slot is not None:
            button.clicked.connect(lambda _=False: slot())
        return button

    def _slider(self, value: float, maximum: float) -> QSlider:
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        slider.setRange(10, max(10, round(maximum * 100)))
        slider.setSingleStep(5)
        slider.setPageStep(25)
        slider.setValue(round(value * 100))
        slider.valueChanged.connect(
            lambda _: self.window.set_caps(self.cap_vx.value() / 100, self.cap_wz.value() / 100))
        return slider

    def refresh(self, window: MainWindow) -> None:
        self.phase.setText(window.ui_phase)
        self.phase.setStyleSheet(f"color: {window.ui_color};")
        self.detail.setText(window.ui_detail if window.ui_phase != "안전 정지" else "")
        sim_ready = window.sim is not None and window.sim.sim_state != shared.BOOT and window.server_exit is None
        self.spawn.setEnabled(sim_ready and window.spawn_pending is None)
        self.stand.setEnabled(window.ready_idle())
        self.reset.setEnabled(sim_ready)
        standing = window.status.phase == 6 and window.mode == 1 and window.ui_phase in ("균형 인계", "서 있음")
        self.drive.setEnabled(standing)
        if not standing and self.drive.isChecked():
            self.drive.setChecked(False)
        driving = window.can_drive()
        for key, button in self.pad.items():
            button.setEnabled(driving or key == Qt.Key.Key_Space)
        cmd_vx, cmd_wz = (window.vx, window.wz) if driving else (0., 0.)
        act_vx, act_wz = window.vel_filter
        shaped_vx, shaped_wz = (window.status.vx, window.status.wz) if window.status.phase == 6 else (0., 0.)
        self.vx_label.setText(f"전진 vx  명령 {cmd_vx:+.2f} → 정책 {shaped_vx:+.2f} · 실제 {act_vx:+.2f} m/s")
        self.wz_label.setText(f"회전 wz  명령 {cmd_wz:+.2f} → 정책 {shaped_wz:+.2f} · 실제 {act_wz:+.2f} rad/s")
        self.vx_bar.set_values(cmd_vx, act_vx, window.cap_vx, self.limits.max_vx * 1.15)
        self.wz_bar.set_values(cmd_wz, act_wz, window.cap_wz, self.limits.max_wz * 1.15)
        self.cap_label.setText(f"현재 상한: vx ±{window.cap_vx:.2f} m/s, wz ±{window.cap_wz:.2f} rad/s "
                               f"(학습 최대 {self.limits.max_vx:g} / {self.limits.max_wz:g})")
        if window.ui_phase == "안전 정지" and window.status.reason:
            self.latch.setText(f"안전 래치: {window.status.reason}\n[소환] 또는 [리셋]으로 다시 시작합니다.")
            self.latch.show()
        else:
            self.latch.hide()
        sim = window.sim
        if sim is not None:
            age = time.monotonic() - window.status.stamp
            controller = {0: "없음", 1: "시작 중", 2: "실행 중", 3: "종료"}.get(sim.controller, "?")
            self.status.setText(f"시드 {sim.seed if sim.seed >= 0 else 'pose 1'} · 실시간 배율 {sim.rtf:.2f} · "
                                f"지연 경고 {sim.slow} · 제어기 {controller}"
                                f"{'' if age < .5 else ' (응답 없음)'} · 정책 {window.args.policy}")
