"""Generate the payload MJCF from RL URDF; vendor XML supplies actuator settings only."""
from __future__ import annotations

import copy
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

SIM = Path(__file__).resolve().parents[1]
REPO = SIM.parent
sys.path.insert(0, str(REPO / "tron2_rl/scripts/rsl_rl"))
sys.path.insert(0, str(SIM))
from build_payload_usd import ASSET, compound, payload_parts

from tron2_sim.spec import LEG_WF


def required(parent: ET.Element, path: str) -> ET.Element:
    element = parent.find(path)
    if element is None:
        raise KeyError(path)
    return element


def numbers(values) -> str:
    return " ".join(f"{float(value):.17g}" for value in values)


def pose(element: ET.Element) -> dict[str, str]:
    origin = element.find("origin")
    if origin is None:
        return {"pos": "0 0 0", "quat": "1 0 0 0"}
    rotation = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" "))
    return {"pos": origin.get("xyz", "0 0 0"), "quat": numbers(rotation.as_quat()[[3, 0, 1, 2]])}


def build() -> Path:
    urdf = ET.parse(ASSET / "urdf/robot.urdf").getroot()
    vendor = ET.parse(SIM / "robot-description/tron2a/WF_TRON2A/xml/robot.xml").getroot()
    root = ET.Element("mujoco", model="wf_payload_rl")
    ET.SubElement(root, "compiler", angle="radian", inertiafromgeom="false", fusestatic="false")
    ET.SubElement(root, "option", timestep="0.001")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="848", offheight="480")
    root.append(copy.deepcopy(required(vendor, "default")))
    assets = ET.SubElement(root, "asset")
    for mesh in sorted((ASSET / "meshes").glob("*.STL")):
        ET.SubElement(assets, "mesh", name=mesh.stem, file=str(mesh.relative_to(REPO)))
    ET.SubElement(assets, "hfield", name="terrain", nrow="61", ncol="61", size="3 3 1 0.1")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0 -2 5", dir="0 0 -1", directional="true")
    terrain_body = ET.SubElement(world, "body", name="terrain_frame")
    ET.SubElement(terrain_body, "geom", name="terrain", type="hfield", hfield="terrain", group="5",
                  contype="1", conaffinity="15", rgba="0.45 0.52 0.40 1")
    with np.load(SIM / "tron2_sim/assets/terrains/flat.npz") as terrain_export:
        border = float(terrain_export["border_width"])
    for name, size, pos in (
        ("west", (border / 2, 3 + border), (-3 - border / 2, 0)),
        ("east", (border / 2, 3 + border), (3 + border / 2, 0)),
        ("south", (3, border / 2), (0, -3 - border / 2)),
        ("north", (3, border / 2), (0, 3 + border / 2)),
    ):
        ET.SubElement(assets, "hfield", name=name, nrow="2", ncol="2", size=numbers([*size, .001, .1]))
        ET.SubElement(world, "geom", name=name, type="hfield", hfield=name, group="5",
                      pos=numbers([*pos, 0]), contype="1", conaffinity="15", rgba="0.45 0.52 0.40 1")
    links = {link.attrib["name"]: link for link in urdf.findall("link")}
    joints = urdf.findall("joint")
    mass, com, tensor = compound()

    def body(link_name: str, parent: ET.Element, joint: ET.Element | None) -> None:
        link = links[link_name]
        attrs = pose(joint) if joint is not None else {"pos": "0 0 0.60"}
        node = ET.SubElement(parent, "body", name=link_name, **attrs)
        inertial = required(link, "inertial")
        inertia = required(inertial, "inertia").attrib
        full = np.array([[float(inertia["i" + "".join(sorted(a + b))]) for b in "xyz"] for a in "xyz"])
        ipose = pose(inertial)
        q = np.fromstring(ipose["quat"], sep=" ")
        rot = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
        full = rot @ full @ rot.T
        m = required(inertial, "mass").attrib["value"]
        if link_name == "base_Link":
            m, ipose["pos"], full = str(mass), numbers(com), tensor
            ET.SubElement(node, "freejoint", name="root")
        ET.SubElement(node, "inertial", pos=ipose["pos"], mass=m,
                      fullinertia=numbers(full[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]))
        if joint is not None and joint.attrib["type"] != "fixed":
            name = joint.attrib["name"]
            settings = required(vendor, f".//joint[@name='{name}']")
            ET.SubElement(node, "joint", name=name, type="hinge", axis=required(joint, "axis").attrib["xyz"],
                          **{k: settings.attrib[k] for k in ("range", "actuatorfrcrange", "armature")})
        for kind in ("visual", "collision"):
            for shape in link.findall(kind):
                geom = required(shape, "geometry")
                attrs = pose(shape)
                mesh = geom.find("mesh")
                box = geom.find("box")
                cylinder = geom.find("cylinder")
                if mesh is not None:
                    attrs.update(type="mesh", mesh=Path(mesh.attrib["filename"]).stem)
                if box is not None:
                    attrs.update(type="box", size=numbers(np.fromstring(box.attrib["size"], sep=" ") / 2))
                if cylinder is not None:
                    size = f"{cylinder.attrib['radius']} {float(cylinder.attrib['length']) / 2}"
                    attrs.update(type="cylinder", size=size)
                ET.SubElement(node, "geom", **attrs, **{"class": kind})
        if link_name == "limx_imu":
            ET.SubElement(node, "site", name="base_imu", size="0.002")
        if link_name == "base_Link":
            for part in payload_parts():
                quat = Rotation.from_matrix(part.rotation).as_quat()[[3, 0, 1, 2]]
                for kind in ("visual", "collision"):
                    ET.SubElement(node, "geom", name=f"{part.name}_{kind}", type="box", pos=numbers(part.position),
                                  quat=numbers(quat), size=numbers(part.size / 2), **{"class": kind},
                                  rgba="0.15 0.15 0.17 1" if "body" in part.name else "0.8 0.82 0.85 1")
                if part.name.startswith("cam_"):
                    # D455 looks along -Y with +Z up; MuJoCo uses -Z view and +Y up.
                    camera_rot = part.rotation @ np.array([[-1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])
                    camera_q = Rotation.from_matrix(camera_rot).as_quat()[[3, 0, 1, 2]]
                    focal = [424 / np.tan(np.deg2rad(87 / 2)), 240 / np.tan(np.deg2rad(58 / 2))]
                    ET.SubElement(node, "camera", name=part.name.replace("cam_", "d455_").replace("_body", ""),
                                  pos=numbers(part.position), quat=numbers(camera_q), resolution="848 480",
                                  sensorsize="848 480", focalpixel=numbers(focal))
        for child in joints:
            if required(child, "parent").attrib["link"] == link_name:
                body(required(child, "child").attrib["link"], node, child)

    body("base_Link", world, None)
    root.append(copy.deepcopy(required(vendor, "actuator")))
    root.append(copy.deepcopy(required(vendor, "sensor")))
    keyframes = ET.SubElement(root, "keyframe")
    q = [0, 0, .60, 1, 0, 0, 0] + [0.9, 0, 0, -1.8, 0] * 2
    ET.SubElement(keyframes, "key", name="default_pose", qpos=numbers(q))
    output = SIM / "tron2_sim/assets/wf_payload/model.xml"
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root)
    ET.ElementTree(root).write(output, encoding="unicode")
    from tron2_sim.payload_model import materialize
    target = materialize()
    print(f"BUILT {target}; compound={mass}; SDK={LEG_WF}")
    return target


if __name__ == "__main__":
    build()
