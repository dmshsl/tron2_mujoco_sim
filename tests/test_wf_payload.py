import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import mujoco
import numpy as np
import pytest

from tron2_sim.channels import JointChannel
from tron2_sim.height_scan_gt import HeightScanProvider, LocalHeightScanProvider, create_provider, publish_scan
from tron2_sim.payload_model import ASSETS, materialize
from tron2_sim.rendering import OffscreenRenderer
from tron2_sim.spec import IMU_STD, LEG_WF, ChannelSpec
from tron2_sim.terrain import load_terrain
from tron2_sim.variants.wf_payload import build

ROOT = Path(__file__).resolve().parents[1]


def test_payload_model_exists():
    assert (ROOT / "tron2_sim/assets/wf_payload/model.xml").is_file()


def test_payload_variant_resolves_model_outside_vendor_from_other_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given a working directory unrelated to the checkout.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ROBOT_IP", "127.0.0.1")
    # When the variant materializes its model.
    spec = build()
    path = Path(spec.model_candidates[0])
    # Then the simulator gets a loadable absolute path owned by this repo.
    assert path.is_absolute()
    assert path.is_relative_to(ASSETS / "wf_payload")
    assert mujoco.MjModel.from_xml_path(str(path)).nu == 10


def test_payload_rejects_nonloopback_before_sdk_initialization(monkeypatch):
    monkeypatch.setenv("ROBOT_IP", "192.0.2.1")
    with pytest.raises(ValueError, match="restricted"):
        build()


@pytest.fixture
def scene():
    model = mujoco.MjModel.from_xml_path(str(materialize()))
    load_terrain(model, "flat")
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return model, data


def test_mass_and_full_compound_tensor(scene):
    model, _ = scene
    base = model.body("base_Link")
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, base.iquat)
    tensor = rotation.reshape(3, 3) @ np.diag(base.inertia) @ rotation.reshape(3, 3).T
    assert model.body_mass.sum() == pytest.approx(34.82968, abs=.01)
    assert model.body("limx_imu").mass == pytest.approx(.01)
    np.testing.assert_allclose(base.ipos,
                               [-.0016516760215667327, -.00008056956202764553, .14884619903811447], atol=1e-12)
    np.testing.assert_allclose(tensor, [[.19447513637749922, .000033099055355874374, .006405962022128576],
                                      [.000033099055355874374, .2139082654940388, .00004455253402833443],
                                      [.006405962022128576, .00004455253402833443, .12230891762885787]], atol=1e-7)


def test_sdk_mapping_and_limits_match_vendor(scene):
    model, _ = scene
    vendor = mujoco.MjModel.from_xml_path(str(ROOT / "robot-description/tron2a/WF_TRON2A/xml/robot.xml"))
    channel = JointChannel(ChannelSpec("main", LEG_WF, imu=IMU_STD)).resolve(model)
    assert len(channel.map) == 10
    for name, (_, _, actuator) in zip(LEG_WF, channel.map, strict=True):
        np.testing.assert_array_equal(model.joint(name).range, vendor.joint(name).range)
        assert model.actuator(actuator).name == vendor.actuator(actuator).name
    np.testing.assert_array_equal(model.actuator_ctrlrange, vendor.actuator_ctrlrange)


def test_all_nonbase_inertials_match_rl_urdf(scene):
    model, _ = scene
    source = ROOT.parent / "tron2_rl/exts/bipedal_locomotion/bipedal_locomotion/assets/usd/WF_TRON2A/urdf/robot.urdf"
    for link in ET.parse(source).getroot().findall("link"):
        name = link.attrib["name"]
        if name == "base_Link":
            continue
        body = model.body(name)
        inertial = link.find("inertial")
        assert body.mass == pytest.approx(float(inertial.find("mass").attrib["value"]))
        np.testing.assert_allclose(body.ipos, np.fromstring(inertial.find("origin").attrib["xyz"], sep=" "))
        inertia = inertial.find("inertia").attrib
        expected = np.array([[float(inertia["i" + "".join(sorted(a + b))]) for b in "xyz"] for a in "xyz"])
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, body.iquat)
        actual = rotation.reshape(3, 3) @ np.diag(body.inertia) @ rotation.reshape(3, 3).T
        np.testing.assert_allclose(actual, expected, atol=1e-7)


@pytest.mark.parametrize("base_x,valid", [(5., True), (25., False)])
def test_flat_border_and_outside_misses(scene, base_x, valid):
    model, data = scene
    data.qpos[0] = base_x
    mujoco.mj_forward(model, data)
    provider = HeightScanProvider(model, data)
    scan = provider.get_scan()
    assert np.all(provider.valid == valid)
    np.testing.assert_allclose(scan, data.qpos[2] - .8 if valid else -1., atol=.001)


def test_flat_scan_has_zero_misses_and_analytic_values(scene):
    model, data = scene
    provider = HeightScanProvider(model, data)
    scan = provider.get_scan()
    assert provider.valid.all()
    np.testing.assert_allclose(scan, data.xpos[provider.base_id, 2] - .8, atol=.001)


def test_static_zero_demonstrates_all_miss_trap(scene):
    provider = HeightScanProvider(*scene)
    scan = provider.get_scan(flg_static=False)
    assert not provider.valid.any()
    np.testing.assert_array_equal(scan, -np.ones(231))


@pytest.mark.parametrize("name", ["flat", "slope", "stairs", "rubble"])
def test_hfield_and_x_fastest_scan_match_isaac_export(scene, name):
    model = mujoco.MjModel.from_xml_path(str(materialize(name)))
    load_terrain(model, name)
    data = mujoco.MjData(model)
    bounds = model.geom_aabb[model.geom("terrain").id]
    assert bounds[2] + bounds[5] == pytest.approx(model.hfield("terrain").size[2])
    data.qpos[2] = .8
    mujoco.mj_forward(model, data)
    provider = HeightScanProvider(model, data)
    with np.load(ASSETS / "terrains/isaac_grid.npz") as fixture:
        np.testing.assert_allclose(provider.offsets, fixture["starts"][:, :2], atol=1e-7)
    with np.load(ASSETS / f"terrains/{name}.npz") as fixture:
        heights = fixture["heights"] * float(fixture["vertical_scale"])
        expected = np.clip(-heights[20:41, 25:36].T.ravel(), -1, 1)
    actual = provider.get_scan()
    assert provider.valid.all()
    np.testing.assert_allclose(actual, expected, atol=.001)


def test_scan_yaw_alignment_and_robot_exclusion(scene):
    model = mujoco.MjModel.from_xml_path(str(materialize("slope")))
    load_terrain(model, "slope")
    data = mujoco.MjData(model)
    data.qpos[2] = .8
    data.qpos[3:7] = [np.sqrt(.5), 0, 0, np.sqrt(.5)]
    mujoco.mj_forward(model, data)
    provider = HeightScanProvider(model, data)
    with np.load(ASSETS / "terrains/slope.npz") as fixture:
        heights = fixture["heights"] * float(fixture["vertical_scale"])
    indices = np.rint(np.column_stack((-provider.offsets[:, 1], provider.offsets[:, 0])) * 10 + 30).astype(int)
    np.testing.assert_allclose(provider.get_scan(), np.clip(-heights[indices[:, 0], indices[:, 1]], -1, 1), atol=.001)


def test_local_scan_transport_and_staleness(scene, tmp_path):
    path = tmp_path / "scan.bin"
    expected = HeightScanProvider(*scene).get_scan()
    publish_scan(path, 1.25, expected)
    client = LocalHeightScanProvider(path)
    np.testing.assert_array_equal(client(), expected)
    assert client.sim_time == 1.25
    with pytest.raises(TimeoutError):
        LocalHeightScanProvider(path, max_age=-1).get_scan()


def test_deploy_factory_accepts_state_and_rejects_skew(scene, tmp_path, monkeypatch):
    path = tmp_path / "deploy_scan.bin"
    monkeypatch.setenv("TRON2_GT_SCAN_PATH", str(path))
    expected = HeightScanProvider(*scene).get_scan()
    publish_scan(path, 1., expected)
    provider = create_provider(SimpleNamespace(policy="base"), "gt")
    np.testing.assert_array_equal(provider.get_scan(SimpleNamespace(timestamp=time.monotonic())), expected)
    with pytest.raises(TimeoutError):
        provider.get_scan(SimpleNamespace(timestamp=0.))


def test_slope_landmark_world_coordinates_and_spawn_from_isaac_mesh(tmp_path):
    spec = build("slope", tmp_path / "scan.bin")
    model = mujoco.MjModel.from_xml_path(str(materialize("slope")))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    spec.modules[0].attach(SimpleNamespace(model=model, data=data))
    assert data.qpos[2] == pytest.approx(.6 + .775)
    with np.load(ASSETS / "terrains/slope.npz") as fixture:
        points = fixture["mesh_landmarks"]
    for point in points:
        start = np.array([point[0] - 3, point[1] - 3, 20.])
        distance = mujoco.mj_ray(model, data, start, np.array([0., 0., -1.]),
                                 np.array([0, 0, 0, 0, 0, 1], dtype=np.uint8), True, -1,
                                 np.array([-1], dtype=np.int32))
        assert 20. - distance == pytest.approx(point[2], abs=.001)


def test_renderer_rgb_depth_and_d455_intrinsics(scene):
    model, data = scene
    for name in ("d455_front", "d455_rear"):
        camera = model.camera(name)
        fov = np.rad2deg(2 * np.arctan(model.cam_sensorsize[camera.id] / (2 * model.cam_intrinsic[camera.id, :2])))
        np.testing.assert_allclose(fov, [87, 58], atol=1e-4)
        rotation = data.cam_xmat[camera.id].reshape(3, 3)
        assert rotation[2, 1] > .7
    with OffscreenRenderer(model) as renderer:
        rgb = renderer.rgb(data)
        assert rgb.shape == (480, 848, 3) and rgb.std() > 5
        for camera in ("d455_front", "d455_rear"):
            depth = renderer.depth(data, camera)
            assert depth.shape == (480, 848)
            assert np.isfinite(depth).all() and (depth > 0).all()
            assert ((depth > .52) & (depth < 6)).mean() > .1


def test_two_second_video_reopens_with_exact_frame_count(scene, tmp_path):
    model, data = scene
    video = tmp_path / "two_seconds.mp4"
    fps = 10
    with OffscreenRenderer(model) as renderer, imageio.get_writer(video, fps=fps, macro_block_size=1) as writer:
        for _ in range(2 * fps):
            writer.append_data(renderer.rgb(data))
    with imageio.get_reader(video) as reader:
        assert reader.count_frames() == 2 * fps
        assert reader.get_data(0).shape == (480, 848, 3)


def test_harness_with_sdk_hold_controller(tmp_path):
    video = tmp_path / "hold.mp4"
    command = [sys.executable, "scripts/sim2sim.py", "--variant", "wf_payload", "--terrain", "flat",
               "--policy", "hold", "--schedule", "tests/hold_schedule.yaml", "--timeout", "2", "--fps", "10",
               "--csv", str(tmp_path / "hold.csv"), "--video", str(video),
               "--controller-command", f"{sys.executable} scripts/hold_pose.py --timeout 15"]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=45,
                            env={**os.environ, "MUJOCO_GL": "egl"})
    assert result.returncode == 0, result.stdout + result.stderr
    with imageio.get_reader(video) as reader:
        assert reader.count_frames() == 20
    assert (tmp_path / "hold.sim.csv").is_file()
