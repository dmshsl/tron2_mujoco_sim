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


def test_drift_per_phase_in_the_start_heading_frame() -> None:
    import drift

    def qpos(x: float, y: float, yaw: float) -> np.ndarray:
        return np.array([x, y, .02, math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)])

    start = qpos(1., 2., math.pi / 2)  # facing +y: "left" is -x
    d = drift.delta(drift.pose(start), drift.pose(qpos(.97, 2.04, math.pi / 2 + math.radians(4.))))
    assert d["fwd"] == pytest_approx(.04) and d["left"] == pytest_approx(.03) and d["xy"] == pytest_approx(.05)
    assert d["yaw_deg"] == pytest_approx(4.)
    assert drift.delta(drift.pose(qpos(0, 0, 3.1)), drift.pose(qpos(0, 0, -3.1)))["yaw_deg"] == pytest_approx(
        math.degrees(2 * math.pi - 6.2), .01)  # wrapped (rounded to 0.01 deg)
    tracker = drift.DriftTracker(start)
    for phase, pose in ((0, start), (1, start), (5, qpos(1., 2.01, math.pi / 2)), (6, qpos(.99, 2.02, math.pi / 2))):
        tracker.sample(phase, pose, 0.)
    tracker.end_base(qpos(.99, 2.03, math.pi / 2))
    summary = tracker.summary()
    assert summary["route"]["fwd"] == pytest_approx(.01) and summary["learned"]["left"] == pytest_approx(.01)
    assert summary["spawn_to_handoff"]["xy"] == pytest_approx(math.hypot(.01, .02), 1e-3)
    assert summary["base"]["fwd"] == pytest_approx(.01) and set(summary["phases"]) == {"0-1", "1-5", "5-6"}


def pytest_approx(value: float, tol: float = 1e-4):
    import pytest

    return pytest.approx(value, abs=tol)
