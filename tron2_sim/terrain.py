from __future__ import annotations

import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .payload_model import ASSETS

TERRAINS = ("flat", "slope", "stairs", "rubble")


def terrain_xml(root: ET.Element, name: str) -> None:
    """Compile terrain dimensions/offsets first so MuJoCo's static BVH is correct."""
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
