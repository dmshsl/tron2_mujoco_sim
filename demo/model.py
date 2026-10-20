"""Demo models: the wf_payload flat (plane-floor) model with a large floor, and a render copy of it.

physics_xml(): tron2_sim's wf_payload model compiled for the "flat" terrain exactly as
tron2_sim.payload_model.materialize does (analytic plane collision, the hfield kept for the GT scan),
but with the plane enlarged to DEMO_FLOOR_HALF metres per side and the border hfields' collision off, so
the robot can drive for minutes without reaching the 3 m edge of the experiment floor.
view_xml(): the same model (identical qpos layout) with presentation visuals only: sky, a grid floor,
shadows, robot colours and an off-screen buffer large enough for the window.
Both are written atomically under tron2_sim/assets/wf_payload/generated/ (never flat.xml, which the
experiment scripts own).
"""
from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

SIM = Path(__file__).resolve().parents[1]
if str(SIM) not in sys.path:
    sys.path.insert(0, str(SIM))

from tron2_sim.payload_model import ASSETS  # noqa: E402
from tron2_sim.terrain import FLAT_PLANE, terrain_xml  # noqa: E402

DEMO_FLOOR_HALF = 60.
BORDERS = ("west", "east", "south", "north")
GENERATED = ASSETS / "wf_payload/generated"

# Presentation colours per visual mesh (RGBA) and the materials behind them.
MESH_STYLE = {
    "base_Link": "shell", "proximal_pitch": "joint", "proximal_roll": "limb", "proximal_yaw": "limb",
    "knee": "limb", "wheel": "tyre",
}
MATERIALS = {
    "shell": dict(rgba="0.93 0.94 0.96 1", specular="0.6", shininess="0.6", reflectance="0.05"),
    "limb": dict(rgba="0.42 0.45 0.50 1", specular="0.5", shininess="0.5"),
    "joint": dict(rgba="0.20 0.22 0.25 1", specular="0.4", shininess="0.4"),
    "tyre": dict(rgba="0.07 0.07 0.08 1", specular="0.1", shininess="0.1"),
    "accent": dict(rgba="0.95 0.45 0.10 1", specular="0.5", shininess="0.5"),
    "frame": dict(rgba="0.75 0.78 0.82 1", specular="0.7", shininess="0.8"),
    "device": dict(rgba="0.10 0.10 0.12 1", specular="0.5", shininess="0.6"),
}


def _base_tree() -> ET.Element:
    root = ET.parse(ASSETS / "wf_payload/model.xml").getroot()
    terrain_xml(root, "flat")
    for mesh in root.findall("asset/mesh"):
        mesh.set("file", str(SIM.parent / mesh.attrib["file"]))
    plane = root.find(f"worldbody/geom[@name='{FLAT_PLANE}']")
    if plane is None:
        raise KeyError(FLAT_PLANE)
    plane.set("size", f"{DEMO_FLOOR_HALF} {DEMO_FLOOR_HALF} 0.1")
    for name in BORDERS:
        border = root.find(f"worldbody/geom[@name='{name}']")
        if border is not None:
            border.set("contype", "0")
            border.set("conaffinity", "0")
    return root


def _write(root: ET.Element, name: str) -> Path:
    GENERATED.mkdir(parents=True, exist_ok=True)
    target = GENERATED / name
    scratch = target.with_suffix(f".{os.getpid()}.tmp")
    ET.ElementTree(root).write(scratch, encoding="unicode")
    os.replace(scratch, target)
    return target


def physics_xml() -> Path:
    return _write(_base_tree(), "demo_flat.xml")


def view_xml(width: int = 2560, height: int = 1600) -> Path:
    root = _base_tree()
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    for child in list(visual):
        visual.remove(child)
    ET.SubElement(visual, "global", offwidth=str(width), offheight=str(height), fovy="45")
    ET.SubElement(visual, "quality", shadowsize="8192", offsamples="8")
    ET.SubElement(visual, "headlight", ambient="0.28 0.28 0.29", diffuse="0.22 0.22 0.22", specular="0.05 0.05 0.05")
    ET.SubElement(visual, "map", znear="0.01", zfar="120", haze="0.3", shadowclip="1.6")
    ET.SubElement(visual, "rgba", haze="0.88 0.90 0.93 1")
    asset = root.find("asset")
    assert asset is not None
    ET.SubElement(asset, "texture", type="skybox", builtin="gradient", rgb1="0.93 0.95 0.98",
                  rgb2="0.55 0.62 0.72", width="512", height="3072")
    ET.SubElement(asset, "texture", name="demo_grid", type="2d", builtin="checker", rgb1="0.86 0.86 0.85",
                  rgb2="0.82 0.82 0.81", mark="edge", markrgb="0.62 0.63 0.65", width="512", height="512")
    ET.SubElement(asset, "material", name="demo_floor", texture="demo_grid", texrepeat="2 2", texuniform="true",
                  reflectance="0", specular="0.1", shininess="0.1")
    for name, attributes in MATERIALS.items():
        ET.SubElement(asset, "material", name=f"demo_{name}", **attributes)
    worldbody = root.find("worldbody")
    assert worldbody is not None
    for light in worldbody.findall("light"):
        worldbody.remove(light)
    ET.SubElement(worldbody, "light", name="demo_sun", mode="trackcom", directional="false", castshadow="true",
                  pos="1.2 -1.6 4.5", dir="-0.25 0.35 -1", cutoff="55", exponent="2", attenuation="1 0 0",
                  diffuse="0.62 0.6 0.58", specular="0.25 0.25 0.25")
    ET.SubElement(worldbody, "light", name="demo_fill", directional="true", castshadow="false",
                  pos="-3 3 4", dir="0.3 -0.4 -0.9", diffuse="0.42 0.43 0.45", specular="0 0 0")
    plane = worldbody.find(f"geom[@name='{FLAT_PLANE}']")
    assert plane is not None
    plane.set("material", "demo_floor")
    plane.set("rgba", "0.62 0.62 0.62 1")  # the default geom rgba (0.5 0.6 0.7 0.5) would override the material
    plane.set("group", "2")
    for geom in worldbody.iter("geom"):
        name = geom.get("name", "")
        if geom.get("type") == "hfield":
            geom.set("rgba", "0 0 0 0")
            geom.set("group", "4")
            continue
        if geom.get("class") != "visual":
            continue
        mesh = geom.get("mesh", "")
        style = next((s for prefix, s in MESH_STYLE.items() if mesh.startswith(prefix)), None)
        if name.startswith("rail_") or name.startswith("leg_"):
            style = "frame"
        elif name.startswith("thor") or name.startswith("cam_"):
            style = "device"
        if style is not None:
            geom.set("material", f"demo_{style}")
            geom.set("rgba", MATERIALS[style]["rgba"])  # the visual class sets an rgba, which overrides a material
    return _write(root, "demo_view.xml")


if __name__ == "__main__":
    print(physics_xml(), view_xml())
