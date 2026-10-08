"""Unit + integration tests for the live perception renderer/IPC used by
`sim2sim.py --height_scan perception` (todo 16)."""
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

# tron2_deploy is a namespace package: create_provider()/read_frames() import it
# lazily, exactly as run_policy.py does in production (which inserts this same
# superproject root before importing the factory). Pin it here too, so these
# tests resolve it regardless of which directory invoked pytest.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tron2_sim.height_scan_perception import (
    CAMERAS,
    FRAME_SHAPE,
    RECORD_BYTES,
    PerceptionRenderer,
    SharedPose,
    _depth_to_mm,
    _frame_source,
    create_provider,
    publish_frames,
    read_frames,
)
from tron2_sim.payload_model import materialize


def test_shared_pose_snapshot_is_a_copy():
    # Given a published qpos/time pair.
    pose = SharedPose(nq=5)
    qpos = np.arange(5, dtype=np.float64)
    pose.publish(qpos, 1.25)
    # When snapshotting and mutating the returned array.
    snapshot_qpos, snapshot_time = pose.snapshot()
    snapshot_qpos[0] = 999.
    # Then the internal state (and a fresh snapshot) are unaffected.
    np.testing.assert_array_equal(pose.snapshot()[0], [0., 1., 2., 3., 4.])
    assert snapshot_time == 1.25


@pytest.mark.parametrize("value,expected_mm", [(1.5, 1500), (0.0, 0), (65.0, 0), (-1.0, 0)])
def test_depth_to_mm_clamps_invalid(value, expected_mm):
    # Given a value at/outside the representable band, when converting to mm,
    # then out-of-range maps to the real sensor's 0 (invalid) sentinel.
    depth = np.full((2, 2), value, dtype=np.float32)
    depth[0, 0] = np.nan
    mm = _depth_to_mm(depth)
    assert mm[1, 1] == expected_mm
    assert mm[0, 0] == 0


def test_publish_read_frames_round_trip(tmp_path):
    # Given distinct front/rear depth frames.
    path = tmp_path / "frames.bin"
    front = np.random.default_rng(0).uniform(0.5, 4.0, FRAME_SHAPE).astype(np.float32)
    rear = np.random.default_rng(1).uniform(0.5, 4.0, FRAME_SHAPE).astype(np.float32)
    # When publishing then reading them back.
    publish_frames(path, wall_t=10.0, sim_t=5.0, front=front, rear=rear)
    frames = read_frames(path)
    # Then camera order/shape/timestamp are preserved and depth round-trips to mm precision.
    assert [f.camera for f in frames] == list(CAMERAS)
    for frame, source in zip(frames, (front, rear), strict=True):
        assert frame.depth.shape == FRAME_SHAPE
        assert frame.depth.dtype == np.uint16
        np.testing.assert_allclose(frame.depth.astype(np.float64) / 1000., source, atol=1e-3)
        assert frame.timestamp == 10.0


def test_publish_frames_rejects_wrong_shape(tmp_path):
    path = tmp_path / "bad.bin"
    with pytest.raises(ValueError, match="480, 848"):
        publish_frames(path, 0., 0., np.zeros((10, 10), dtype=np.float32), np.zeros(FRAME_SHAPE, dtype=np.float32))


def test_read_frames_rejects_truncated_file(tmp_path):
    path = tmp_path / "short.bin"
    path.write_bytes(b"\x00" * RECORD_BYTES)  # only one camera's worth of bytes
    with pytest.raises(ValueError, match="malformed"):
        read_frames(path)


def test_frame_source_defers_a_frame_timestamped_after_the_query(tmp_path):
    # Given a frame published with renderer wall time T (same clock as SdkState.timestamp).
    path = tmp_path / "frames.bin"
    depth = np.full(FRAME_SHAPE, 1.0, dtype=np.float32)
    publish_frames(path, wall_t=100.0, sim_t=0., front=depth, rear=depth)
    # When queried from a timestamp strictly before T (the race _frame_source guards against),
    # then the frame is deferred (empty), never raised past to HeightMap.update as a "future" frame.
    assert _frame_source(path, 99.0) == ()
    # When queried at or after T, then both cameras are returned.
    assert len(_frame_source(path, 100.0)) == 2
    assert len(_frame_source(path, 101.0)) == 2


def test_create_provider_rejects_wrong_mode_or_policy():
    with pytest.raises(ValueError, match="mode=perception"):
        create_provider(SimpleNamespace(policy="base"), "gt")
    with pytest.raises(ValueError, match="Base contract"):
        create_provider(SimpleNamespace(policy="base_blind"), "perception")


@pytest.fixture
def payload_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(materialize()))


def test_perception_renderer_warms_up_and_renders_at_target_rate(payload_model, tmp_path):
    # Given a live renderer targeting a modest rate (keeps the test fast).
    pose = SharedPose(payload_model.nq)
    data = mujoco.MjData(payload_model)
    mujoco.mj_resetDataKeyframe(payload_model, data, 0)
    mujoco.mj_forward(payload_model, data)
    pose.publish(data.qpos, data.time)
    scan_path = tmp_path / "live.bin"
    renderer = PerceptionRenderer(payload_model, pose, scan_path, target_fps=20.0)
    # When warming up (absorbing the one-time EGL/shader cold start) then running briefly.
    renderer.warm_up()
    renderer.start()
    time.sleep(0.5)
    renderer.stop()
    # Then frames were produced near the target rate, on its own thread, and are readable.
    assert renderer.rendered >= 5
    assert renderer.achieved_fps > 10.0
    frames = read_frames(scan_path)
    assert len(frames) == 2
    for frame in frames:
        assert frame.depth.shape == FRAME_SHAPE
        assert frame.depth.dtype == np.uint16
