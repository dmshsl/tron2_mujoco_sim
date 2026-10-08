"""The negative-control scale is confined to a simulator-owned process wrapper."""
import csv
import os
import subprocess
import sys
from pathlib import Path

import imageio.v2 as imageio
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tron2_sim/scripts"))


def test_position_scale_override_preserves_wheels_and_source() -> None:
    # Given the real generated contract and its independent wheel scales.
    import debug_policy
    from tron2_deploy.policy_contract import load_contract

    path = ROOT / "tron2_deploy/controllers/model/WF_TRON2A_BASE_BLIND/contract.yaml"
    original = load_contract(path)
    # When the harness-only negative control is requested.
    changed = debug_policy.scaled_contract(original, 0.25)
    # Then only position-action scales change; the source remains immutable.
    for i, name in enumerate(original.joints.action_names):
        expected = original.joints.action_scale[i] if "wheel" in name else 0.25
        assert changed.joints.action_scale[i] == expected
    assert load_contract(path) == original
    assert changed.control == original.control


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_scale_is_rejected(scale: float) -> None:
    import debug_policy
    from tron2_deploy.policy_contract import load_contract

    original = load_contract(ROOT / "tron2_deploy/controllers/model/WF_TRON2A_BASE_BLIND/contract.yaml")
    with pytest.raises(ValueError):
        debug_policy.scaled_contract(original, scale)


def test_controller_exit_keeps_video_but_fails_gate(tmp_path: Path) -> None:
    command = [sys.executable, "scripts/sim2sim.py", "--variant", "wf_payload", "--terrain", "flat",
               "--policy", "hold", "--schedule", "tests/hold_schedule.yaml", "--timeout", "3",
               "--fps", "10", "--csv", str(tmp_path / "failed.csv"), "--video", str(tmp_path / "failed.mp4"),
               "--controller-command", f"{sys.executable} scripts/hold_pose.py --timeout 0.1"]
    result = subprocess.run(command, cwd=ROOT / "tron2_sim", capture_output=True, text=True, timeout=30)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "unpowered observation" in result.stdout
    # An explicit --controller-command smoke test still defaults to an unsupported ground start.
    assert "START mode=ground " in result.stdout
    import imageio.v2 as imageio
    with imageio.get_reader(str(tmp_path / "failed.mp4")) as reader:
        assert reader.count_frames() == 30


def test_default_start_is_gantry_for_a_real_policy_run(tmp_path: Path) -> None:
    # A real --policy run (no --controller-command) is the real-robot-standard
    # start: held by a crane strap until the policy's first action, matching
    # the task-23 bring-up ladder, which never powers on a base policy
    # freestanding. --start is not passed, proving the default resolves this way.
    command = [sys.executable, "scripts/sim2sim.py", "--variant", "wf_payload", "--terrain", "flat",
               "--policy", "base_blind", "--height_scan", "gt", "--schedule", "tests/hold_schedule.yaml",
               "--timeout", "2", "--fps", "10",
               "--csv", str(tmp_path / "policy.csv"), "--video", str(tmp_path / "policy.mp4")]
    result = subprocess.run(command, cwd=ROOT / "tron2_sim", capture_output=True, text=True, timeout=45,
                            env={**os.environ, "MUJOCO_GL": "egl"})
    assert "START mode=gantry " in result.stdout, result.stdout + result.stderr


def test_live_perception_feeds_the_real_deploy_pipeline_without_blocking_physics(tmp_path: Path) -> None:
    # Given the Base policy with a short schedule, when run with
    # --height_scan perception (todo 16), then the live renderer thread
    # produces frames near its 30 Hz target, the real odometry/height_map
    # pipeline runs inside the controller subprocess (scan_* columns appear
    # in its CSV), and the harness still completes and exits cleanly.
    command = [sys.executable, "scripts/sim2sim.py", "--variant", "wf_payload", "--terrain", "flat",
               "--policy", "base", "--height_scan", "perception", "--schedule", "tests/hold_schedule.yaml",
               "--timeout", "5", "--fps", "10",
               "--csv", str(tmp_path / "live.csv"), "--video", str(tmp_path / "live.mp4")]
    result = subprocess.run(command, cwd=ROOT / "tron2_sim", capture_output=True, text=True, timeout=60,
                            env={**os.environ, "MUJOCO_GL": "egl"})
    assert result.returncode == 0, result.stdout + result.stderr
    assert "RESULT: PASS" in result.stdout
    rate_line = next(line for line in result.stdout.splitlines() if line.startswith("PERCEPTION_RENDER_RATE"))
    achieved = float(rate_line.split("achieved_hz=")[1].split()[0])
    assert achieved > 15.0  # well below the 30 Hz target still proves the thread ran, not stalled
    with (tmp_path / "live.csv").open() as stream:
        header = next(csv.reader(stream))
    assert sum(name.startswith("scan_") for name in header) == 231
    with imageio.get_reader(str(tmp_path / "live.mp4")) as reader:
        assert reader.count_frames() == 50
