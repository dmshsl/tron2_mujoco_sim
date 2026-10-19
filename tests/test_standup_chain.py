"""Learned stand-up chain: sim2sim CLI wiring and standup_chain.py outcome classification (no simulator)."""
import csv
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

SIM = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("standup_chain", SIM / "scripts/standup_chain.py")
assert spec is not None and spec.loader is not None
CHAIN = importlib.util.module_from_spec(spec)
sys.modules["standup_chain"] = CHAIN
spec.loader.exec_module(CHAIN)

SIM_COLUMNS = ["sim_time", "base_x", "base_y", "base_z", "qw", "qx", "qy", "qz", "controller_t", "policy_t"]


def write_logs(tmp_path: Path, phases: list[tuple[float, int]], sim_rows: list[list[float]]) -> tuple[Path, Path]:
    controller = tmp_path / "chain.csv"
    with controller.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["t", "policy_t", "standup_phase"])
        for t, phase in phases:
            writer.writerow([t, 0., phase])
    sim = tmp_path / "chain.sim.csv"
    with sim.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(SIM_COLUMNS)
        writer.writerows(sim_rows)
    return controller, sim


def test_schedule_is_stand_drive_stop():
    entries = yaml.safe_load(CHAIN.SCHEDULE.read_text())
    assert entries == [[0.0, 0.0, 0.0, 0.0], [3.0, 0.3, 0.0, 0.0], [8.0, 0.0, 0.0, 0.0]]


def test_latched_standup_is_a_safe_failure(tmp_path):
    phases = [(0., 0), (5., 3), (8., 4), (10., 5), (12., 5)]
    rows = [[t, 0., 0., .02, 1., 0., 0., 0., t, 0.] for t in (0., 5., 10., 11.)]
    rows.append([12., 0., 0., .12, .8, .6, 0., 0., 12., 0.])
    controller, sim = write_logs(tmp_path, phases, rows)
    log = ("STANDUP_POLICY_START t=10.000 q=[]\nDAMPING_LATCH: tilt 1.25 > 1.2 rad; restart required\n"
           "RESULT: PASS (harness)\n")
    summary = CHAIN.summarize(log, controller, sim, 0)
    assert summary["outcome"] == "safe_failure" and summary["plumbing_ok"]
    assert summary["phases_reached"] == [0, 3, 4, 5] and summary["latch"].startswith("tilt")
    assert summary["standup_tilt_max"] == pytest.approx(1.287, abs=1e-3)


def test_success_and_drive_is_stood_and_drove(tmp_path):
    phases = [(0., 0), (10., 5), (13., 6)]
    times = ((10., 0.), (13., 0.), (16., 0.), (19., 3.), (24., 8.))
    rows = [[t, .1 * max(0., p - 3.), 0., .5, 1., 0., 0., 0., t, p] for t, p in times]
    controller, sim = write_logs(tmp_path, phases, rows)
    log = ("STANDUP_POLICY_START t=10.0\nSTANDUP_SUCCESS t=13.000 policy_s=3.000 height_est=0.4800\n"
           "HANDOFF policy=base_blind t=13.000\nRESULT: PASS\n")
    summary = CHAIN.summarize(log, controller, sim, 0)
    assert summary["outcome"] == "stood_and_drove" and summary["standup_seconds"] == 3.
    assert summary["drive_x_m"] == pytest.approx(.5)


def test_nonzero_exit_or_missing_csv_is_a_crash(tmp_path):
    controller, sim = write_logs(tmp_path, [(0., 0), (10., 5)], [[0., 0., 0., .02, 1., 0., 0., 0., 0., 0.]])
    crashed = CHAIN.summarize("DAMPING_LATCH: x; restart required\nRESULT: PASS\n", controller, sim, -11)
    assert crashed["outcome"] == "crash"
    missing = CHAIN.summarize("", tmp_path / "none.csv", sim, 0)
    assert missing["outcome"] == "crash" and not missing["plumbing_ok"]


@pytest.mark.parametrize("arguments,message", [
    (["--start", "seated"], "seated start requires the policy controller with --standup scripted or learned"),
    (["--standup", "scripted", "--standup-model", "/tmp"], "--standup-model requires --standup learned"),
])
def test_sim2sim_cli_rejects_bad_standup_wiring(arguments, message):
    command = [sys.executable, "scripts/sim2sim.py", "--variant", "wf_payload", "--terrain", "flat", "--policy",
               "base_blind", "--schedule", "tests/hold_schedule.yaml", "--timeout", "1", "--csv", "/tmp/unused.csv",
               *arguments]
    result = subprocess.run(command, cwd=SIM, capture_output=True, text=True, timeout=120, check=False)
    assert result.returncode == 2 and message in result.stderr
