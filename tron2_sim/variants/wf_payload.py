from __future__ import annotations

import os
from pathlib import Path

import mujoco

from ..height_scan_gt import DEFAULT_PATH, HeightScanProvider, publish_scan
from ..payload_model import materialize
from ..spec import IMU_STD, LEG_WF, ChannelSpec, RobotSpec
from ..terrain import load_terrain


class PayloadTerrain:
    name = "payload_terrain"
    slider_control = False

    def __init__(self, terrain: str, scan_path: Path) -> None:
        self.terrain = terrain
        self.scan_path = scan_path
        self.next_scan = 0.

    def attach(self, core) -> bool:
        elevation = load_terrain(core.model, self.terrain)
        core.data.qpos[2] += elevation
        mujoco.mj_forward(core.model, core.data)
        self.provider = HeightScanProvider(core.model, core.data)
        self.on_publish(core)
        return True

    def owned_actuators(self) -> list[int]:
        return []

    def on_control(self, core, dt: float) -> None:
        pass

    def on_manual(self, core) -> None:
        pass

    def on_reset(self, core) -> None:
        self.next_scan = 0.

    def on_render(self, core, data) -> None:
        pass

    def on_publish(self, core) -> None:
        if core.data.time + 1e-9 >= self.next_scan:
            publish_scan(self.scan_path, float(core.data.time), self.provider.get_scan())
            self.next_scan = float(core.data.time) + .02


def build(terrain: str = "flat", scan_path: Path = DEFAULT_PATH) -> RobotSpec:
    if os.environ.get("ROBOT_IP", "127.0.0.1") != "127.0.0.1":
        raise ValueError("wf_payload is restricted to ROBOT_IP=127.0.0.1")
    model_path = materialize(terrain)
    return RobotSpec(
        robot_type="WF_TRON2A_PAYLOAD", base="WF", family_dir="tron2a",
        model_candidates=[str(model_path)],
        channels=[ChannelSpec("main", LEG_WF, sdk_role="main", imu=IMU_STD)],
        modules=[PayloadTerrain(terrain, scan_path)], sdk_robot="Tron2", cam=(3., -20.))
