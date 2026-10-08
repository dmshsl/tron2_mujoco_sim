"""Materialize the tracked RL-derived model into the generated asset directory."""
import xml.etree.ElementTree as ET
from pathlib import Path

SIM = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).resolve().parent / "assets"


def materialize(terrain: str = "flat") -> Path:
    from .terrain import terrain_xml

    root = ET.parse(ASSETS / "wf_payload.xml").getroot()
    terrain_xml(root, terrain)
    for mesh in root.findall("asset/mesh"):
        mesh.set("file", str(SIM.parent / mesh.attrib["file"]))
    target = SIM / "robot-description/tron2a/WF_TRON2A_PAYLOAD/xml/robot.xml"
    target.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(target, encoding="unicode")
    return target
