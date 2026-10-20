"""3D view: off-screen MuJoCo (EGL) render of the latest sim pose into a Qt widget, follow camera, mouse
orbit/zoom, and an on-screen HUD (phase, commanded vs actual speed)."""
from __future__ import annotations

import math
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from model import view_xml

DEFAULT_CAMERA = dict(distance=2.6, azimuth=145., elevation=-18.)


class Viewport(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(640, 400)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.model = mujoco.MjModel.from_xml_path(str(view_xml()))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.option = mujoco.MjvOption()
        self.option.geomgroup[:] = [0, 1, 1, 0, 0, 0]
        self.option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
        self.camera = mujoco.MjvCamera()
        self.reset_camera()
        self.base = self.model.body("base_Link").id
        self.renderer: mujoco.Renderer | None = None
        self.render_size = (0, 0)
        self.image: QImage | None = None
        self.follow = True
        self.drag: tuple[Qt.MouseButton, QPointF] | None = None
        self.hud: dict = {}
        self.max_size = (int(self.model.vis.global_.offwidth), int(self.model.vis.global_.offheight))
        self.resize_timer = QTimer(self, singleShot=True, interval=150, timeout=self._resize_renderer)

    # ------------------------------------------------------------------ camera
    def reset_camera(self) -> None:
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.distance = DEFAULT_CAMERA["distance"]
        self.camera.azimuth = DEFAULT_CAMERA["azimuth"]
        self.camera.elevation = DEFAULT_CAMERA["elevation"]
        self.camera.lookat[:] = self.data.qpos[:3] + [0, 0, .15]
        self.follow = True

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self.drag = (event.button(), event.position())

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self.drag = None

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        self.reset_camera()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self.drag is None:
            return
        button, last = self.drag
        delta = event.position() - last
        self.drag = (button, event.position())
        if button == Qt.MouseButton.LeftButton:
            self.camera.azimuth -= .3 * delta.x()
            self.camera.elevation = float(np.clip(self.camera.elevation - .3 * delta.y(), -89., -2.))
        elif button == Qt.MouseButton.RightButton:
            self.camera.distance = float(np.clip(self.camera.distance * math.exp(.005 * delta.y()), .6, 15.))
        elif button == Qt.MouseButton.MiddleButton:  # pan (stops following until double-click)
            self.follow = False
            yaw = math.radians(self.camera.azimuth)
            scale = .002 * self.camera.distance
            right = np.array([math.sin(yaw), -math.cos(yaw), 0.])
            forward = np.array([math.cos(yaw), math.sin(yaw), 0.])
            self.camera.lookat[:] += scale * (-delta.x() * right + delta.y() * forward)

    def wheelEvent(self, event) -> None:  # noqa: N802
        steps = event.angleDelta().y() / 120.
        self.camera.distance = float(np.clip(self.camera.distance * .9 ** steps, .6, 15.))

    # ------------------------------------------------------------------ rendering
    def resizeEvent(self, event) -> None:  # noqa: N802
        self.resize_timer.start()
        super().resizeEvent(event)

    def _resize_renderer(self) -> None:
        ratio = self.devicePixelRatioF()
        width = min(int(self.width() * ratio) // 2 * 2, self.max_size[0])
        height = min(int(self.height() * ratio) // 2 * 2, self.max_size[1])
        if (width, height) == self.render_size or width < 64 or height < 64:
            return
        if self.renderer is not None:
            self.renderer.close()
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.render_size = (width, height)

    def update_pose(self, qpos: np.ndarray | None) -> None:
        if qpos is not None and len(qpos) == self.model.nq and np.isfinite(qpos).all():
            self.data.qpos[:] = qpos
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)  # subtree CoM: the key light tracks the robot
            mujoco.mj_camlight(self.model, self.data)
        if self.follow:
            target = self.data.xpos[self.base] + [0., 0., .12]
            self.camera.lookat[:] += .12 * (target - self.camera.lookat)
        if self.renderer is None:
            self._resize_renderer()
            if self.renderer is None:
                return
        self.renderer.update_scene(self.data, self.camera, self.option)
        pixels = np.ascontiguousarray(self.renderer.render())
        height, width, _ = pixels.shape
        self.image = QImage(pixels.data, width, height, 3 * width, QImage.Format.Format_RGB888).copy()
        self.update()

    # ------------------------------------------------------------------ HUD
    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        if self.image is not None:
            painter.drawImage(QRectF(self.rect()), self.image)
        else:
            painter.fillRect(self.rect(), QColor(40, 44, 52))
        self._draw_hud(painter)
        painter.end()

    def _pill(self, painter: QPainter, x: float, y: float, text: str, color: QColor, size: int) -> float:
        font = QFont(self.font())
        font.setPixelSize(size)
        font.setBold(True)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        width = metrics.horizontalAdvance(text) + size * 1.4
        height = size * 1.9
        path = QPainterPath()
        path.addRoundedRect(QRectF(x, y, width, height), height / 2, height / 2)
        painter.fillPath(path, QColor(20, 22, 28, 190))
        painter.setPen(QPen(color, 2))
        painter.drawPath(path)
        painter.setPen(QColor(245, 246, 248))
        painter.drawText(QRectF(x, y, width, height), Qt.AlignmentFlag.AlignCenter, text)
        return height

    def _draw_hud(self, painter: QPainter) -> None:
        hud = self.hud
        if not hud:
            return
        scale = max(.8, min(1.6, self.height() / 800.))
        y = 18 * scale
        y += self._pill(painter, 18 * scale, y, hud.get("phase", ""), QColor(hud.get("color", "#f28c28")),
                        int(26 * scale)) + 10 * scale
        if hud.get("detail"):
            self._pill(painter, 18 * scale, y, hud["detail"], QColor("#8a93a6"), int(15 * scale))
        if hud.get("speed") is not None:
            cmd_vx, act_vx, cmd_wz, act_wz = hud["speed"]
            font = QFont(self.font())
            font.setPixelSize(int(17 * scale))
            painter.setFont(font)
            lines = [f"vx  명령 {cmd_vx:+.2f}  실제 {act_vx:+.2f} m/s", f"wz  명령 {cmd_wz:+.2f}  실제 {act_wz:+.2f} rad/s"]
            metrics = painter.fontMetrics()
            width = max(metrics.horizontalAdvance(line) for line in lines) + 28 * scale
            height = len(lines) * metrics.height() + 18 * scale
            box = QRectF(self.width() - width - 18 * scale, self.height() - height - 18 * scale, width, height)
            path = QPainterPath()
            path.addRoundedRect(box, 10 * scale, 10 * scale)
            painter.fillPath(path, QColor(20, 22, 28, 190))
            painter.setPen(QColor(235, 238, 242))
            for index, line in enumerate(lines):
                painter.drawText(QPointF(box.x() + 14 * scale, box.y() + 9 * scale + metrics.ascent()
                                         + index * metrics.height()), line)
        painter.setPen(QColor(255, 255, 255, 150))
        font = QFont(self.font())
        font.setPixelSize(int(12 * scale))
        painter.setFont(font)
        painter.drawText(QPointF(18 * scale, self.height() - 14 * scale),
                         "마우스: 왼쪽 드래그 회전 · 휠/오른쪽 드래그 줌 · 가운데 드래그 이동 · 더블클릭 카메라 초기화")

    def close_renderer(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
