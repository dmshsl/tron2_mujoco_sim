"""Generate the desktop icon: the standing robot rendered from the demo view model on a rounded badge."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
DEMO = Path(__file__).resolve().parent
SIM = DEMO.parent
for path in (SIM, DEMO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from model import physics_xml, view_xml  # noqa: E402
from PIL import Image, ImageDraw, ImageFilter  # noqa: E402

from tron2_sim.start_conditions import ground_start  # noqa: E402

CONTRACT = SIM.parent / "tron2_deploy/controllers/model/WF_TRON2A_BASE/contract.yaml"


def render_robot(size: int = 768) -> Image.Image:
    physics = mujoco.MjModel.from_xml_path(str(physics_xml()))
    pose = mujoco.MjData(physics)
    ground_start(physics, pose, CONTRACT)
    model = mujoco.MjModel.from_xml_path(str(view_xml()))
    data = mujoco.MjData(model)
    data.qpos[:] = pose.qpos
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, size, size)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = data.xpos[model.body("base_Link").id] + [0., 0., -.12]
    camera.distance, camera.azimuth, camera.elevation = 1.55, 140., -12.
    option = mujoco.MjvOption()
    option.geomgroup[:] = [0, 1, 0, 0, 0, 0]
    renderer.update_scene(data, camera, option)
    rgb = renderer.render().copy()
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera, option)
    depth = renderer.render().copy()
    renderer.close()
    alpha = (depth < .98 * depth.max()).astype(np.uint8) * 255
    image = Image.fromarray(np.dstack([rgb, alpha]), "RGBA")
    return image.crop(image.getbbox())


def badge(robot: Image.Image, size: int = 256) -> Image.Image:
    scale = 4
    big = size * scale
    canvas = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    gradient = Image.new("RGBA", (big, big))
    top, bottom = np.array([44, 49, 60]), np.array([20, 22, 28])
    rows = np.linspace(0., 1., big)[:, None, None]
    gradient = Image.fromarray(np.broadcast_to((top + (bottom - top) * rows).astype(np.uint8), (big, big, 3)).copy())
    mask = Image.new("L", (big, big), 0)
    ImageDraw.Draw(mask).rounded_rectangle((8, 8, big - 8, big - 8), radius=big // 5, fill=255)
    canvas.paste(gradient, (0, 0), mask)
    ring = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    ImageDraw.Draw(ring).ellipse((big * .14, big * .62, big * .86, big * .9), outline=(242, 140, 40, 255),
                                 width=big // 40)
    canvas.alpha_composite(ring.filter(ImageFilter.GaussianBlur(2)))
    height = int(big * .74)
    width = int(robot.width * height / robot.height)
    if width > big * .8:
        width = int(big * .8)
        height = int(robot.height * width / robot.width)
    robot = robot.resize((width, height), Image.LANCZOS)
    canvas.alpha_composite(robot, ((big - width) // 2, int(big * .09)))
    return canvas.resize((size, size), Image.LANCZOS)


def main() -> None:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEMO / "assets/tron2_icon.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    badge(render_robot()).save(target)
    print(target)


if __name__ == "__main__":
    main()
