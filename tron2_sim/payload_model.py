"""Materialize the tracked RL-derived model into the generated asset directory."""
import xml.etree.ElementTree as ET
from pathlib import Path

SIM = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).resolve().parent / "assets"


def materialize(terrain: str = "flat") -> Path:
    from .terrain import terrain_xml

    root = ET.parse(ASSETS / "wf_payload/model.xml").getroot()
    terrain_xml(root, terrain)
    for mesh in root.findall("asset/mesh"):
        mesh.set("file", str(SIM.parent / mesh.attrib["file"]))
    target = ASSETS / "wf_payload/generated" / f"{terrain}.xml"
    target.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(target, encoding="unicode")
    return target
