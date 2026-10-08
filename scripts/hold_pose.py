"""SDK-only harness smoke controller, not a balance policy or hardware entry point."""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from limxsdk import datatypes
from limxsdk.robot.Robot import Robot
from limxsdk.robot.RobotType import RobotType

from tron2_sim.spec import LEG_WF


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=5.)
    args = parser.parse_args()
    ready = threading.Event()
    pose: list[float] = []

    def state_callback(state) -> None:
        if not ready.is_set() and len(state.q) == 10:
            pose.extend(state.q)
            ready.set()

    robot = Robot(RobotType.Tron2, False)
    if not robot.init("127.0.0.1"):
        raise RuntimeError("Loopback SDK init failed")
    robot.subscribeRobotState(state_callback)
    if not ready.wait(10.):
        raise TimeoutError("No simulator state")
    command = datatypes.RobotCmd()
    command.mode = [0] * 10
    command.q, command.dq, command.tau = pose, [0.] * 10, [0.] * 10
    command.Kp = [100., 100., 40., 100., 0.] * 2
    command.Kd = [3., 3., 2., 3., 1.] * 2
    command.motor_names = LEG_WF
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        command.stamp = time.time_ns()
        robot.publishRobotCmd(command)
        time.sleep(.002)
    print("HOLD_POSE SDK loopback complete", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
