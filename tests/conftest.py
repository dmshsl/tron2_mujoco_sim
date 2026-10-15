"""Shared test guards for the tron2_sim suite."""
import subprocess

import pytest


def require_sdk_loopback_free() -> None:
    """The limxsdk simulator loopback uses fixed localhost ports: a second simulator or controller on
    the host receives the other pair's packets (measured: a stray first command, segfaults, heap
    corruption in the native SDK). Tests that run the full harness skip instead of colliding with
    e.g. a running seated_route_eval batch; run them when the host has no other sim2sim."""
    probe = subprocess.run(["pgrep", "-f", r"scripts/sim2sim\.py|run_policy\.py|seated_route_eval\.py run"],
                           capture_output=True, text=True, check=False)
    others = [pid for pid in probe.stdout.split() if pid.isdigit()]
    if others:
        pytest.skip(f"another limxsdk simulator/controller is running (pids {', '.join(others)}): "
                    "the SDK loopback ports are shared, run the harness tests alone")
