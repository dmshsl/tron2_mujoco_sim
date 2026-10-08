"""Run with tron2_rl/run.sh scripts/export_v22_terrains.py --headless (absolute path)."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "tron2_sim/assets/terrains")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)

import numpy as np
from bipedal_locomotion.tasks.locomotion.cfg.WF_TRON2A.terrains_cfg import single_terrain_cfg_v22
from isaaclab.sensors.ray_caster.patterns import GridPatternCfg, grid_pattern
from isaaclab.terrains.height_field import utils


def export() -> None:
    """Capture the actual decorator's bordered int16 field before mesh conversion."""
    args.out.mkdir(parents=True, exist_ok=True)
    starts, directions = grid_pattern(GridPatternCfg(resolution=0.1, size=(2.0, 1.0), ordering="xy"), "cpu")
    np.savez(args.out / "isaac_grid.npz", starts=starts.numpy(), directions=directions.numpy())
    original = utils.convert_height_field_to_mesh
    captured = []

    def capture(heights, horizontal_scale, vertical_scale, slope_threshold=None):
        captured.append(heights.copy())
        return original(heights, horizontal_scale, vertical_scale, slope_threshold)

    for name in ("flat", "slope", "stairs", "rubble"):
        np.random.seed(42)
        cfg = single_terrain_cfg_v22(name, rows=1, cols=1, difficulty=1.0)
        sub = cfg.sub_terrains[name]
        sub.size = cfg.size
        captured.clear()
        utils.convert_height_field_to_mesh = capture
        try:
            if name == "flat":
                heights = np.zeros((61, 61), dtype=np.int16)
                origin_z = 0.0
                landmarks = np.array([[1., 2., 0.], [4., 5., 0.]])
            else:
                sub.horizontal_scale = cfg.horizontal_scale
                sub.vertical_scale = cfg.vertical_scale
                sub.slope_threshold = cfg.slope_threshold
                meshes, origin = sub.function(1.0, sub)
                if len(captured) != 1:
                    raise RuntimeError(f"Expected one height field for {name}, got {len(captured)}")
                heights = captured[0]
                origin_z = float(origin[2])
                landmarks = meshes[0].vertices[[10 * 61 + 20, 40 * 61 + 50]].copy()
        finally:
            utils.convert_height_field_to_mesh = original
        np.savez_compressed(args.out / f"{name}.npz", heights=heights,
                            horizontal_scale=cfg.horizontal_scale, vertical_scale=cfg.vertical_scale,
                            origin_z=origin_z, difficulty=1.0, seed=42, border_width=cfg.border_width,
                            mesh_landmarks=landmarks,
                            source="single_terrain_cfg_v22; Isaac height_field_to_mesh pre-conversion")
        print(f"EXPORTED {name} shape={heights.shape} "
              f"range={heights.min()}..{heights.max()} origin_z={origin_z}", flush=True)


if __name__ == "__main__":
    export()
    print("RESULT: PASS", flush=True)
    sys.stdout.flush()
    os._exit(0)
