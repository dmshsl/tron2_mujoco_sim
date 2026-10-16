from __future__ import annotations

import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .payload_model import ASSETS

TERRAINS = ("flat", "slope", "stairs", "rubble")


FLAT_PLANE = "flat_plane"


def terrain_xml(root: ET.Element, name: str) -> None:
    """Compile terrain dimensions/offsets first so MuJoCo's static BVH is correct.

    The flat terrain collides through an analytic plane at its height (the height field stays
    for the GT height scan, collision off): MuJoCo's hfield-vs-mesh/box collision gives the
    seated base ~0.7 contacts per 1 ms step with its normal force zero in 80 % of the steps and
    delivered as 15 kN spikes, so the body bounces, creeps 10 mm/s under a few N m of leg torque
    and a loaded wheel spins in place at 0.5 rad/s; on the plane the same base has 24 steady
    contacts, 0.0 mm/s creep and no spin (measured, tests/test_start_conditions.py)."""
    if name not in TERRAINS:
        raise ValueError(f"Unknown v22 terrain: {name}")
    with np.load(ASSETS / "terrains" / f"{name}.npz", allow_pickle=False) as export:
        heights = export["heights"] * float(export["vertical_scale"])
        spacing = float(export["horizontal_scale"])
    field = root.find("asset/hfield[@name='terrain']")
    body = root.find("worldbody/body[@name='terrain_frame']")
    if field is None or body is None:
        raise KeyError("terrain asset/frame")
    extent = max(float(np.ptp(heights)), .001)
    size = [(heights.shape[0] - 1) * spacing / 2, (heights.shape[1] - 1) * spacing / 2, extent, .1]
    field.set("size", " ".join(str(value) for value in size))
    body.set("pos", f"0 0 {float(heights.min())}")
    if name == "flat":
        hfield = body.find("geom[@name='terrain']")
        if hfield is None:
            raise KeyError("terrain geom")
        hfield.set("contype", "0")
        hfield.set("conaffinity", "0")
        worldbody = root.find("worldbody")
        if worldbody is None:
            raise KeyError("worldbody")
        ET.SubElement(worldbody, "geom", name=FLAT_PLANE, type="plane", size=f"{size[0]} {size[1]} 0.1",
                      pos=f"0 0 {float(heights.max())}", group="5", contype="1", conaffinity="15",
                      rgba=hfield.attrib.get("rgba", "0.45 0.52 0.40 1"))


def load_terrain(model: mujoco.MjModel, name: str) -> float:
    """Isaac arrays are [x,y]; MuJoCo rows are y, columns x. Return spawn elevation."""
    if name not in TERRAINS:
        raise ValueError(f"Unknown v22 terrain: {name}")
    with np.load(ASSETS / "terrains" / f"{name}.npz", allow_pickle=False) as export:
        heights = export["heights"].astype(np.float64) * float(export["vertical_scale"])
        spacing = float(export["horizontal_scale"])
        spawn_z = float(export["origin_z"])
    field = model.hfield("terrain")
    minimum = float(heights.min())
    extent = max(float(np.ptp(heights)), .001)
    size = [(heights.shape[0] - 1) * spacing / 2, (heights.shape[1] - 1) * spacing / 2, extent, .1]
    if not np.allclose(field.size, size) or model.body("terrain_frame").pos[2] != minimum:
        raise ValueError(f"Compile materialize({name!r}) before loading its heights")
    start = int(model.hfield_adr[field.id])
    model.hfield_data[start:start + heights.size] = ((heights.T - minimum) / extent).ravel()
    return spawn_z
