"""EGL RGB/depth capture. Depth is optical-axis distance in metres, not ray range."""
from __future__ import annotations

import mujoco
import numpy as np
from numpy.typing import NDArray


class OffscreenRenderer:
    def __init__(self, model: mujoco.MjModel) -> None:
        self.renderer = mujoco.Renderer(model, height=480, width=848)
        self.camera = mujoco.MjvCamera()
        self.camera.distance = 2.5
        self.camera.azimuth = 130
        self.camera.elevation = -20
        self.base_id = model.body("base_Link").id
        self.option = mujoco.MjvOption()
        self.option.geomgroup[:] = [0, 1, 0, 0, 0, 1]

    def rgb(self, data: mujoco.MjData, camera: str | None = None) -> NDArray[np.uint8]:
        self.renderer.disable_depth_rendering()
        self.camera.lookat[:] = data.xpos[self.base_id] + [0, 0, .1]
        self.renderer.update_scene(data, camera=camera or self.camera, scene_option=self.option)
        return self.renderer.render().copy()

    def depth(self, data: mujoco.MjData, camera: str = "d455_front") -> NDArray[np.float32]:
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(data, camera=camera, scene_option=self.option)
        return self.renderer.render().copy()

    def __enter__(self) -> OffscreenRenderer:
        return self

    def __exit__(self, *exc) -> None:
        self.renderer.close()
