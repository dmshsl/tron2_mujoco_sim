"""Demo app pieces that need neither the SDK nor a display: tidy heaps, command limits, models, state block."""
import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM / "demo"))
sys.path.insert(0, str(SIM.parent / "tron2_common"))

import heap  # noqa: E402
import limits  # noqa: E402
import shared  # noqa: E402
from model import DEMO_FLOOR_HALF, physics_xml, view_xml  # noqa: E402
from tron2_common.kinematics import LOWER, UPPER, payload_clearance  # noqa: E402


def test_tidy_heaps_are_pose1_plus_bounded_noise() -> None:
    base = heap.pose1()
    for seed in range(20):
        q = heap.tidy_pose(seed)
        assert np.all(q >= LOWER - 1e-12) and np.all(q <= UPPER + 1e-12)
        assert np.max(np.abs(q - base)) <= math.radians(heap.SPREAD_DEG) + 1e-9
        assert payload_clearance(q) >= .03
        np.testing.assert_array_equal(q, heap.tidy_pose(seed))


def test_limits_schema_and_defaults(tmp_path: Path) -> None:
    missing = limits.load(tmp_path / "none.json")
    assert (missing.vx_step, missing.wz_step, missing.cap_vx, missing.cap_wz) == (.5, .25, 1., 1.)
    path = tmp_path / "limits.json"
    path.write_text(json.dumps({"vx_max": 3.0, "wz_max": 2.0, "vx_cap_default": 1.2, "wz_cap_default": .8,
                                "vx_step": .4, "wz_step": .2}))
    loaded = limits.load(path)
    assert (loaded.vx_step, loaded.wz_step, loaded.cap_vx, loaded.cap_wz, loaded.max_vx, loaded.max_wz) == (
        .4, .2, 1.2, .8, 3., 2.)


def test_demo_models_share_the_state_layout_and_a_large_floor() -> None:
    physics = mujoco.MjModel.from_xml_path(str(physics_xml()))
    view = mujoco.MjModel.from_xml_path(str(view_xml(640, 480)))
    assert (physics.nq, physics.nv, physics.nu) == (view.nq, view.nv, view.nu)
    assert physics.geom("flat_plane").size[0] == DEMO_FLOOR_HALF
    for name in ("west", "east", "south", "north"):
        assert physics.geom(name).contype == 0 and physics.geom(name).conaffinity == 0


def test_state_block_round_trip(tmp_path: Path) -> None:
    writer = shared.StateBlock(tmp_path / "state.bin", 17, create=True)
    qpos = np.arange(17, dtype=np.float64)
    writer.write(shared.SimState(1., 2., shared.ACTIVE, shared.RUNNING, 0, 7, 1, .99, .5, -.25, 3, qpos))
    reader = shared.StateBlock(tmp_path / "state.bin")
    state = reader.read()
    assert state is not None and state.seed == 7 and state.spawn_id == 3 and state.sim_state == shared.ACTIVE
    np.testing.assert_array_equal(state.qpos, qpos)
    reader.close()
    writer.close()
