"""Drag metrics of scripts/seated_route_eval.py on synthetic logs (no simulator)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

SIM = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("seated_route_eval", SIM / "scripts/seated_route_eval.py")
assert spec is not None and spec.loader is not None
EVAL = importlib.util.module_from_spec(spec)
sys.modules["seated_route_eval"] = EVAL
spec.loader.exec_module(EVAL)


def synthetic(seconds: float = 2., dt: float = .001):
    """Controller rows (L leg targets move during the second half) and sim rows with a knee that
    slides steadily at 10 mm/s, a base that chatters +-20 mm/s with zero mean, and a resting wheel
    that spins in place (material slip 50 mm/s, centre still)."""
    times = np.arange(0., seconds, .002)
    control = []
    for t in times:
        row = {"t": f"{t:.6f}", "standup_phase": "2" if t >= .5 else "1"}
        for i in EVAL.POSITIONS:
            row[f"target_q_{i}"] = f"{(.5 * (t - 1.) if (i < 4 and t >= 1.) else 0.):.6f}"
        control.append(row)
    sim, stamps = [], np.arange(0., seconds, dt)
    for k, t in enumerate(stamps):
        chatter = .02 if k % 2 else -.02
        parts = [f"knee_L={.01:.5f},0.00000,{.01:.5f},0.00000,{.01:.5f},0.00000",
                 f"base={abs(chatter):.5f},0.00000,{chatter:.5f},0.00000,{chatter:.5f},0.00000",
                 "wheel_R=0.05000,0.00000,0.05000,0.00000,0.00000,0.00000"]
        sim.append({"sim_time": f"{t:.6f}", "controller_t": f"{t:.6f}", "slip": ";".join(parts)})
    phase = np.array([int(r["standup_phase"]) for r in control])
    sim_t = stamps
    sim_phase = phase[np.clip(np.searchsorted(times, sim_t), 0, len(phase) - 1)]
    return control, times, sim, sim_t, sim_phase


def test_drag_integrates_steady_sliding_and_rejects_chatter_and_wheel_spin() -> None:
    control, times, sim, sim_t, sim_phase = synthetic()
    drag = EVAL.drag_metrics(control, times, sim, sim_t, sim_phase)
    assert drag is not None
    # the L knee slides 10 mm/s: resting for the first second (P1 0.5 s + P2 0.5 s), moving in the second
    assert abs(drag["drag_m"]["knee_L|P2|moving"] - .01) < 1e-3 and abs(drag["drag_m"]["knee_L|P1|resting"] - .005) < 1e-3
    assert abs(drag["moving_drag_m"] - .01) < 1e-3 and drag["moving_drag_worst"] == "knee_L/moving"
    # the base chatters +-20 mm/s with zero mean: raw slip 40 mm over 2 s, drag ~0
    assert abs(drag["slip_m"]["base|P2|body"] - .03) < 2e-3 and drag["drag_m"].get("base|P2|body", 0.) < 5e-4
    # the resting wheel spins in place: raw slip 50 mm/s, drag 0 (centre velocity)
    assert abs(drag["slip_m"]["wheel_R|P2|resting"] - .075) < 2e-3 and drag["drag_m"].get("wheel_R|P2|resting", 0.) == 0.
    assert abs(drag["resting_drag_m"] - .01) < 1e-3 and drag["resting_drag_worst"] == "knee_L/resting"  # its resting second
    assert drag["drag_m"]["base|P1|body"] + drag["drag_m"]["base|P2|body"] < 5e-4  # chatter cancels


def test_moving_legs_follow_the_target_changes() -> None:
    control, times, *_ = synthetic()
    moving = EVAL.moving_legs(control, times)
    assert not moving["R"].any() and not moving["L"][times < .95].any() and moving["L"][times > 1.05].all()


def test_parse_slip_accepts_the_first_log_format() -> None:
    assert EVAL.parse_slip("wheel_L=0.1,0.2,0.3,0.4")["wheel_L"][0] == .1 and np.isnan(EVAL.parse_slip("a=1,2,3,4")["a"][2])
    assert EVAL.parse_slip("base=1,2,3,4,5,6") == {"base": (1., 2., 3., 4., 5., 6.)}
