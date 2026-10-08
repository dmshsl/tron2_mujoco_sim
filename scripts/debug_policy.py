"""Simulation-only negative control; never changes the on-disk deployment contract."""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tron2_deploy"))

from tron2_deploy.policy_contract import DeploymentContract, load_contract


def scaled_contract(contract: DeploymentContract, scale: float) -> DeploymentContract:
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(scale)
    wheel_actions = {contract.joints.action_to_sdk[i] for i in contract.joints.wheel_indices}
    scales = tuple(old if i in wheel_actions else scale
                   for i, old in enumerate(contract.joints.action_scale))
    joints = contract.joints.model_copy(update={"action_scale": scales})
    return contract.model_copy(update={"joints": joints})


if __name__ == "__main__":
    import run_policy

    scale = float(sys.argv.pop(1))

    def debug_load(path: Path) -> DeploymentContract:
        return scaled_contract(load_contract(path), scale)

    run_policy.load_contract = debug_load
    raise SystemExit(run_policy.main())
