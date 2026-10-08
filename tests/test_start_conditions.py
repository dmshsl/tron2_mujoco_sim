import io
from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml

from tron2_sim.payload_model import materialize
from tron2_sim.terrain import load_terrain

CONTRACT = Path(__file__).resolve().parents[2] / "tron2_deploy/controllers/model/WF_TRON2A_BASE_BLIND/contract.yaml"


@pytest.mark.parametrize("terrain", ["flat", "stairs", "rubble"])
def test_ground_start_matches_contract_and_touches_flat_surface(terrain: str) -> None:
    from tron2_sim.start_conditions import ground_start, wheel_bottoms

    model = mujoco.MjModel.from_xml_path(str(materialize(terrain)))
    load_terrain(model, terrain)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qvel[:] = 1.
    ground_start(model, data, CONTRACT)
    expected = yaml.safe_load(CONTRACT.read_text())["joints"]
    for name, value in zip(expected["sdk_names"], expected["default_position"], strict=True):
        assert data.qpos[model.joint(name).qposadr[0]] == pytest.approx(value)
    np.testing.assert_array_equal(data.qvel, 0.)
    assert np.min(wheel_bottoms(model, data)[:, 2]) >= -1e-8
    assert data.ncon > 0
    assert -.001 - 1e-8 <= min(data.contact[i].dist for i in range(data.ncon)) <= 1e-6
    if terrain != "flat":
        assert data.qpos[0] < -model.hfield("terrain").size[0]


def test_ground_height_is_computed_after_wheel_radius_changes() -> None:
    from tron2_sim.start_conditions import ground_start, wheel_bottoms

    model = mujoco.MjModel.from_xml_path(str(materialize()))
    load_terrain(model, "flat")
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    ground_start(model, data, CONTRACT)
    original_z = data.qpos[2]
    for gid in range(model.ngeom):
        if model.body(model.geom_bodyid[gid]).name.startswith("wheel_") and model.geom_contype[gid]:
            model.geom_size[gid, 0] += .025
    ground_start(model, data, CONTRACT)
    assert data.qpos[2] > original_z + .02
    assert np.min(wheel_bottoms(model, data)[:, 2]) >= -1e-8


def test_action_monitor_waits_for_complete_policy_record() -> None:
    from tron2_sim.start_conditions import PolicyProgress

    source = io.StringIO("t,policy_t,action_0\n9,0,0\n9.1,0.1,0")
    progress = PolicyProgress(source)
    progress.update()
    assert progress.first_action_t is None
    source.seek(0, io.SEEK_END)
    source.write("\n")
    progress.update()
    assert progress.first_action_t == 9.1
    assert progress.policy_t == .1


def test_gantry_release_is_latched_to_policy_record_not_elapsed_time() -> None:
    from tron2_sim.start_conditions import GantrySupport, ground_start

    model = mujoco.MjModel.from_xml_path(str(materialize()))
    load_terrain(model, "flat")
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    ground_start(model, data, CONTRACT)
    support = GantrySupport(model, data)
    data.time = 100.
    assert support.apply(data, False) > 0
    data.time = .1
    assert support.apply(data, True) == 0
    assert support.apply(data, False) == 0
    np.testing.assert_array_equal(data.xfrc_applied, 0.)
