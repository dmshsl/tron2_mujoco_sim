"""The negative-control scale is confined to a simulator-owned process wrapper."""
import subprocess
import sys
from pathlib import Path

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
    import imageio.v2 as imageio
    with imageio.get_reader(str(tmp_path / "failed.mp4")) as reader:
        assert reader.count_frames() == 30
