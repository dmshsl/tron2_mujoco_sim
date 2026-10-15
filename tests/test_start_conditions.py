import io
import sys
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


@pytest.mark.parametrize("draw", ["full", "admissible"])
def test_seated_start_settles_limp_on_the_base_bottom(draw: str) -> None:
    from tron2_sim.start_conditions import seated_draw, seated_start

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tron2_common"))
    from tron2_common.kinematics import BOTTOM, LOWER, NAMES, UPPER

    model = mujoco.MjModel.from_xml_path(str(materialize()))
    load_terrain(model, "flat")
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    drawn = seated_draw(7, draw)
    assert np.all(drawn >= LOWER) and np.all(drawn <= UPPER)
    np.testing.assert_array_equal(seated_draw(7, draw), drawn)  # seeded, reproducible
    info = seated_start(model, data, 7, draw=draw)
    assert info["settle_steps"] == pytest.approx(1. / model.opt.timestep)
    assert info["tilt"] < .1 and abs(data.qpos[2] - BOTTOM) < .01  # upright, on its bottom
    assert data.time == 0.  # contacts hover at the 1 mm skin margin, so ncon itself is not asserted
    settled = np.array([data.qpos[model.joint(n).qposadr[0]] for n in NAMES])
    hard = np.array([model.jnt_range[model.joint(n).id] for n in NAMES])
    assert np.all(settled >= hard[:, 0] - .02) and np.all(settled <= hard[:, 1] + .02)
    assert np.max(np.abs(settled - drawn)) > .01  # limp legs moved under gravity
    for name in NAMES:
        assert data.ctrl[model.actuator(name.removesuffix("_Joint")).id] == 0.


def test_seated_start_held_variant_keeps_the_drawn_pose() -> None:
    from tron2_sim.start_conditions import seated_draw, seated_start

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tron2_common"))
    from tron2_common.kinematics import NAMES

    model = mujoco.MjModel.from_xml_path(str(materialize()))
    load_terrain(model, "flat")
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    info = seated_start(model, data, 3, draw="admissible", settle_s=0.)
    assert info["settle_steps"] == 0 and info["tilt"] == 0.
    held = np.array([data.qpos[model.joint(n).qposadr[0]] for n in NAMES])
    np.testing.assert_allclose(held, seated_draw(3, "admissible"))
    with pytest.raises(ValueError):
        seated_start(model, data, 3, settle_s=-1.)
    with pytest.raises(ValueError):
        seated_draw(3, "elsewhere")


def test_contact_labels_name_terrain_contacts_by_body() -> None:
    from tron2_sim.start_conditions import ContactLabels, seated_start

    model = mujoco.MjModel.from_xml_path(str(materialize()))
    load_terrain(model, "flat")
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    seated_start(model, data, 2)
    labels = ContactLabels(model, gap=.002)
    for _ in range(20):  # the seated hfield contact set flickers step to step
        mujoco.mj_step(model, data)
        if "base" in labels(data).split(";"):
            break
    seen = set(labels(data).split(";")) - {""}
    assert "base" in seen and all(not label.startswith(("payload:", "self:")) for label in seen)
    assert labels.label[model.geom("cam_rear_body_collision").id] == "payload:cam_rear_body"
