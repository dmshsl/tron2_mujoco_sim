"""Drive command limits for the demo, read once at launch from demo_command_limits.json.

Accepted keys (any nesting level; the first match wins):
  vx_step, wz_step                      increments per key press / pad click
  cap_vx | vx_cap, cap_wz | wz_cap      default software cap (also {"cap"|"software_cap"|"default_cap": {"vx", "wz"}})
  max_vx | vx_max, max_wz | wz_max      trained maximum, the slider's upper end (also {"max"|"trained_max"|
                                        "hardware_max"|"limits": {"vx", "wz"}}); defaults to the cap
Missing file or keys fall back to vx_step 0.5, wz_step 0.25, cap 1.0 / 1.0.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PATH = Path("/home/eunwon/tron2_ws/artifacts/basecampaign/demo_command_limits.json")


@dataclass(frozen=True)
class Limits:
    vx_step: float = .5
    wz_step: float = .25
    cap_vx: float = 1.
    cap_wz: float = 1.
    max_vx: float = 1.
    max_wz: float = 1.
    source: str = "defaults"


def _find(tree: object, names: tuple[str, ...], groups: tuple[str, ...], axis: str) -> float | None:
    if isinstance(tree, dict):
        for name in names:
            value = tree.get(name)
            if isinstance(value, (int, float)) and math.isfinite(value) and value > 0:
                return float(abs(value))
            if isinstance(value, (list, tuple)) and len(value) == 2 and all(isinstance(v, (int, float)) for v in value):
                return float(max(abs(value[0]), abs(value[1])))
        for group in groups:
            sub = tree.get(group)
            if isinstance(sub, dict):
                value = sub.get(axis)
                if isinstance(value, (int, float)) and math.isfinite(value) and value > 0:
                    return float(value)
                if isinstance(value, (list, tuple)) and len(value) == 2:
                    return float(max(abs(value[0]), abs(value[1])))
        for value in tree.values():
            found = _find(value, names, groups, axis)
            if found is not None:
                return found
    return None


def load(path: Path = DEFAULT_PATH) -> Limits:
    try:
        tree = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return Limits(source=f"defaults ({path} missing or unreadable)")
    defaults = Limits()
    values = {}
    for axis in ("vx", "wz"):
        values[f"{axis}_step"] = _find(tree, (f"{axis}_step", f"step_{axis}"), ("step", "steps"), axis) or getattr(
            defaults, f"{axis}_step")
        cap = _find(tree, (f"cap_{axis}", f"{axis}_cap", f"{axis}_cap_default", f"default_cap_{axis}",
                           f"software_cap_{axis}"),
                    ("cap", "software_cap", "default_cap", "default_software_cap"), axis)
        maximum = _find(tree, (f"max_{axis}", f"{axis}_max", f"trained_max_{axis}"),
                        ("max", "trained_max", "hardware_max", "limits", "trained"), axis)
        cap = cap or (min(maximum, getattr(defaults, f"cap_{axis}")) if maximum else getattr(defaults, f"cap_{axis}"))
        values[f"cap_{axis}"] = cap
        values[f"max_{axis}"] = max(maximum or cap, cap)
    return Limits(**values, source=str(path))
