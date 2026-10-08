# English | [中文](README_zh-CN.md)

<!--
  SPDX-FileCopyrightText: 2024-2026 LimX Dynamics Technology Co., Ltd.
  SPDX-License-Identifier: Apache-2.0
-->

> **Distribution.** The primary public distribution point for this
> repository is GitHub:
> <https://github.com/limxdynamics/tron2_mujoco_sim>. The internal
> LimX GitLab is a mirror; open issues, PRs, and security reports on
> the GitHub repository.

# tron2-mujoco-sim

MuJoCo simulator for the TRON2 robot family. It bridges the LimX low-level SDK
(`RobotCmd` / `RobotState` carrying `q` / `dq` / `tau` / `Kp` / `Kd`, plus
`ImuData` and gripper messages) to `mujoco.MjData`, so the same controller wire
format that drives a physical robot can be exercised against a simulated one.

A robot variant is declared, not hard-coded: it is a set of joint channels plus
optional modules, assembled in `tron2_sim/variants/`. Every channel of every
supported variant runs over the SDK's `*ForSim` simulator-side API — there is no
second messaging stack.

## License and attribution

Apache License, Version 2.0. See [`LICENSE`](LICENSE); SPDX identifier
`Apache-2.0`.

- [`NOTICE`](NOTICE) — required attribution notice.
- [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) — per-submodule and
  per-dependency provenance.
- [`SECURITY.md`](SECURITY.md) — how to report a vulnerability, and the
  simulation-versus-hardware boundary.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — development workflow, submodule
  procedure, DCO sign-off.
- [`CHANGELOG.md`](CHANGELOG.md) — release notes and items blocked on upstream.

## Scope

**Included:** `simulator.py`, the `tron2_sim/` package (including the shipped
gripper calibration under `tron2_sim/config/`), and documentation. Submodules
are *declared* (pinned by commit) but not vendored:

- `robot-description/` — URDF / MJCF models and meshes.
- `robot-joystick/` — gamepad helper binary used to drive a controller
  by hand (see [§3a Gamepad control](#3a-gamepad-control)).
- `limxsdk-lowlevel/` — LimX low-level SDK with pre-built wheels.

**Excluded by design:** trained control policies (`.onnx`, `.pt`, `.pth`,
`.ckpt`), SDK binaries or wheels committed into this tree, calibration values,
firmware, bag captures, and hard-coded private network addresses. Command
examples use the placeholder `<robot-ip>`; the default endpoint is `127.0.0.1`.

## 1. Dependencies and deployment

| Dependency | Version | Notes |
|---|---|---|
| Python | >= 3.10 | verified on 3.10 |
| `mujoco` | >= 3.2.2 | physics and passive viewer |
| `PyYAML` | >= 6.0 | reads `tron2_sim/config/gripper_config.yaml` |
| `limxsdk` | 4.8+ | architecture-specific wheel inside the `limxsdk-lowlevel` submodule; installed separately, always with `--no-deps` |

Clone first, either way:

```bash
git clone --recurse-submodules https://github.com/limxdynamics/tron2_mujoco_sim.git
cd tron2-mujoco-sim
```

If you already cloned without submodules, run
`git submodule update --init --recursive` before continuing.

### Option A: uv (recommended)

[uv](https://docs.astral.sh/uv/) installs the right Python, resolves everything
from the committed `uv.lock`, and needs no manual virtualenv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # install uv
uv sync --extra sdk                               # physics + SDK

export ROBOT_TYPE=SF_TRON2A
uv run simulator.py                               # with viewer
uv run simulator.py --headless                    # no graphics (CI / remote)
uv run simulator.py --headless --duration 30      # exit after 30 s

uv run simulator.py --headless --duration 30 --no-grasper  # DACH without the 2F gripper
```

`uv sync` requires the `limxsdk-lowlevel` submodule to be checked out, since
that is where the SDK wheel lives. A later plain `uv sync` (without
`--extra sdk`) removes `limxsdk` again.

### Option B: pip

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install "mujoco>=3.2.2" "pyyaml>=6.0"

# SDK wheel matching your architecture. --no-deps is required: the wheel's own
# metadata pulls in onnxruntime/pygame/scipy/pandas and pins numpy<1.26.4.
pip install --no-deps limxsdk-lowlevel/python3/amd64/limxsdk-*.whl     # x86_64
# pip install --no-deps limxsdk-lowlevel/python3/aarch64/limxsdk-*.whl # aarch64

export ROBOT_TYPE=SF_TRON2A
python3 simulator.py                              # with viewer
python3 simulator.py --headless --duration 30
```

### Environment variables

| Variable | Default | Effect |
|---|---|---|
| `ROBOT_TYPE` | *(required)* | Selects the variant, e.g. `SF_TRON2A`. |
| `ROBOT_IP` | `127.0.0.1` | SDK endpoint. |
| `GRIPPER_CONFIG` | `tron2_sim/config/gripper_config.yaml` | Loads a different DACH gripper calibration file. |
| `DACH_GRASPER` | `1` | `0` loads `robot.xml` instead of `robot_grasper.xml`. Same as `--no-grasper`. |

## 2. Supported robot types

Ten types: five base names in a `TRON2A` and a `TRON2B` variant. Joint names and
wire order are identical across the two families, so a controller needs no
changes to talk to either.

| Base | TRON2A | TRON2B | Description |
|---|---|---|---|
| SF | `SF_TRON2A` | `SF_TRON2B` | Sole-foot biped (10 joints) |
| WF | `WF_TRON2A` | `WF_TRON2B` | Wheel-foot biped (10 joints) |
| DA | `DA_TRON2A` | `DA_TRON2B` | Dual arm (14 joints) |
| DACH | `DACH_TRON2A` | `DACH_TRON2B` | Dual arm + 2-DOF head (16 joints), optional 2F gripper |
| DASF | `DASF_TRON2A` | `DASF_TRON2B` | Humanoid composite: SF legs + DACH arms/head (26 joints) |

## 3. Keyboard controls (viewer mode)

- **Space** — pause / resume. Paused means manual mode: the Control sliders on
  the right pose joints directly. (The DACH 2F gripper idles in manual mode and
  tracks neither sliders nor commands.)
- **R** — reset the floating-base pose only; joint angles are kept. Fixed-base
  models print a notice and ignore it.
- **Backspace / Reset button** — full reset (using the selected keyframe, when
  one is selected).
- **Double-click to select, then Ctrl+drag** — apply a perturbation force.

### 3a. Gamepad control

For the controller-side gamepad workflow (SF/WF variants running the
sibling `tron2-rl-deploy-python` deployment stack), initialize the
`robot-joystick` submodule and run the helper binary alongside the
controller:

```bash
./robot-joystick/robot-joystick
```

Default bindings:

- `L1 + Y` — switch to WALK.
- `L1 + X` — switch back to IDLE.
- `R1` — clear velocity commands.

## 4. Communication

**Wire-order contract** (cmd/state index order, identical on TRON2A and TRON2B):
the DACH head is pitch then yaw; the DASF head is **yaw then pitch**. Where a
model's XML declares actuators in a different order, the simulator absorbs it by
resolving joints by name — never by position.

Behavioural notes:

- **Commands on Centaur channels (DASF) should carry joint names**
  (`RobotCmd.motor_names`). The native layer validates them against the
  published state and rejects mismatches with
  `ERROR: Centaur ... RobotCmd does not match the corresponding RobotState`.
  The simulator always publishes state with real joint names.
- DACH 2F gripper calibration lives in `tron2_sim/config/gripper_config.yaml`.
- On DACH grasper models, `grasper_base_{L,R}_Joint_ctrl` are driven by the
  linkage rather than by a channel, so their `ctrl` stays 0 and the simulator
  prints an `INFO: unowned actuators` line at startup. This is expected.

## 5. Architecture

```
simulator.py            entry point: ROBOT_TYPE -> variant registry -> SimCore
tron2_sim/
  spec.py               wire order + variant specs (pure data)
  core.py               physics/render dual-thread, dual-MjData snapshots,
                        manual mode, resets
  channels.py           JointChannel: name-based joint map, MIT control law,
                        state/IMU publishing
  transports.py         SdkBus (limxsdk *ForSim) and the capability guard
  config/               YAML loading + the shipped gripper_config.yaml
  modules/              optional capabilities; dach_grasper (2F linkage gripper)
  variants/             one file per base name; __init__.py is the registry
doc/gif/                demo animations (see the gallery below)
```

Threading: the physics thread owns the physics `MjData`; the render thread owns
`render_data`. UI resets and drag forces are handed over through flags and
buffers, so `MjData` always has exactly one writer.

## 6. TRON2B versus TRON2A

Joint names, ordering and topics are byte-identical, so controller code needs no
changes. But 2B is a hardware revision: **policies and gains tuned for 2A are not
directly transferable.**

| Difference | TRON2A | TRON2B |
|---|---|---|
| Hip pitch/roll and knee torque | ±150 N·m | **±200 N·m** (armature updated to match) |
| Hip yaw, ankle pitch, elbow torque | ±60 / ±70 N·m | **±70 N·m** |
| Wrist and head torque | ±20 N·m | **±15 N·m** |
| `base_Link` mass (SF/WF/DA/DACH) | 12.57 kg | **13.4 kg** (CoM and inertia updated) |
| SF knee / hip-yaw limits | `knee: [-2.618, 0.262]` | **sign-flipped** `knee: [-0.262, 2.618]`, link geometry mirrored |
| DA / DACH base | floating (near-rigid freejoint) | **fixed** (no freejoint; R ignores the reset with a notice) |

## 7. Demo gallery

Recorded from this simulator, one per variant.

| Variant | Demo |
|---|---|
| `SF_TRON2A` | ![SF_TRON2A](doc/gif/SF_TRON2A.gif) |
| `WF_TRON2A` | ![WF_TRON2A](doc/gif/WF_TRON2A.gif) |
| `DACH_TRON2A` | ![DACH_TRON2A](doc/gif/DACH_TRON2A.gif) |
| `DASF_TRON2A` | ![DASF_TRON2A](doc/gif/DASF_TRON2A.gif) |

See [`doc/gif/README.md`](doc/gif/README.md) for the media rules CI enforces.

## 8. FAQ

**`Error: Please set the ROBOT_TYPE ...`** — export `ROBOT_TYPE` first; the
message lists every supported value.

**`uv sync` fails on a missing `limxsdk-*.whl`** — the `limxsdk-lowlevel`
submodule is not checked out. Run `git submodule update --init --recursive`.

**`limxsdk ... does not support the Centaur simulator side`** — the installed
SDK predates the Centaur API, so `DASF_*` cannot run. Update the
`limxsdk-lowlevel` submodule to a release that exposes `RobotType.Centaur` and
the LowerBody/UpperBody `*ForSim` methods.

**`Error: none of ['robot.xml'] exists under ...`** — the `robot-description`
submodule is missing, or that variant's assets are not published yet.

**`No module named limxsdk`** — the environment was created without the extra.
Run `uv sync --extra sdk`. Note that a later plain `uv sync` removes it again.
On a pip install, the wheel was not installed, or it was installed into a
different interpreter than the one running `simulator.py`.

**`python3 -m venv` fails with an `ensurepip` error** — on Debian and Ubuntu the
standard library venv module is packaged separately: `sudo apt install
python3-venv`. Or use option A, which needs no system package.

**The robot loads but never moves** — no controller is publishing, or (on
`DASF_*`) its commands lack `motor_names` and the native layer is rejecting
them.

**Simulator output disappears when piped** — fixed; if you see it on an older
checkout, set `PYTHONUNBUFFERED=1`.

## 9. Cite and support

## RL payload sim2sim (explicit, simulation-only)

In the parent `tron2` checkout, `wf_payload` is the RL-URDF-derived model,
not the vendor WF mass model. It preserves WF channel order, actuator names,
limits, armatures and simulator contact defaults. All endpoints are loopback.

```bash
uv sync --extra sdk
uv run python scripts/build_wf_payload.py
MUJOCO_GL=egl uv run simulator.py --variant wf_payload --terrain flat --headless --duration 2
MUJOCO_GL=egl uv run python -m pytest tests/test_wf_payload.py -q
MUJOCO_GL=egl uv run --extra sdk python scripts/sim2sim.py \
  --variant wf_payload --terrain flat --policy base --height_scan gt \
  --schedule tests/hold_schedule.yaml --timeout 2 --csv /tmp/base.csv --video /tmp/base.mp4
```

The tracked canonical `tron2_sim/assets/wf_payload/model.xml` is generated from the
parent RL URDF and `build_payload_usd.compound()/payload_parts()`, which import
`make_camera_mount_urdf` geometry. At load time it is materialized at
`tron2_sim/assets/wf_payload/generated/<terrain>.xml`, resolving the parent
RL meshes to absolute paths. These runtime outputs are ignored and stay in this
repository, outside the vendor submodule. No SDK or Isaac is needed to run the generator. Limb and IMU
inertials are copied from the RL URDF, the 13.73968 kg compound uses `fullinertia`
about its COM, and all nine payload boxes have separate visual/collision geoms.

Terrain exports are committed under `tron2_sim/assets/terrains/`. To regenerate:

```bash
# From the parent tron2_rl directory; do not bypass the shared lock.
OMNI_KIT_ACCEPT_EULA=YES flock -w 1800 /tmp/isaac.lock timeout 900 \
  ./run.sh ../tron2_sim/scripts/export_v22_terrains.py --headless
```

They contain the actual bordered v22 int16 arrays (seed 42, difficulty 1.0,
0.1 m horizontal / 0.005 m vertical resolution), plus an Isaac `grid_pattern`
fixture. Flat is the zero-height hfield equivalent of Isaac's plane. Array axis
0 is x in Isaac; conversion transposes to MuJoCo y-row/x-column order. The
static terrain body is in geom group 5; scans enable only that group and use
`flg_static=1`, `bodyexclude=-1`. They exclude robot, payload and visual geoms.

### GT provider interface for the deploy entry point

`tron2_sim.height_scan_gt.LocalHeightScanProvider` requires only NumPy and the
Python standard library. Put the parent `tron2_sim/` directory on `sys.path`, then:

```python
from tron2_sim.height_scan_gt import LocalHeightScanProvider
provider = LocalHeightScanProvider()
scan = provider.get_scan()  # or provider(); float32 (231,), already clipped
```

The harness sets `PYTHONPATH`, `TRON2_GT_SCAN_PATH`, and
`TRON2_HEIGHT_SCAN_FACTORY=tron2_sim.height_scan_gt:create_provider` for the
controller child. The factory accepts `(contract, mode)` and its client supports
deployment's `get_scan(state)` interface. `state.timestamp` must use the local
monotonic clock; excessive scan/state skew raises rather than using old geometry.
Without it, the provider uses `/tmp/tron2-height-scan-gt.bin` (the standalone
simulator's default). A frame contains `struct.Struct('<dd231f')`: monotonic
wall timestamp, simulation timestamp, 231 values, under `flock`. It is refreshed
at 50 Hz. The client raises on missing/malformed data or age >0.5 s; callers must
stop safely rather than substitute a zero scan. Values are
`clip(base_z - terrain_z - 0.8, -1, 1)`; ray misses are -1. Grid coordinates are
x=-1..1, y=-0.5..0.5, 0.1 m spacing, yaw-aligned; x varies fastest. The harness
unlinks its per-process file on exit. The GT interface is simulation-only.

The default harness command runs `tron2_deploy/run_policy.py --autostart
--robot_ip 127.0.0.1` with policy/schedule/timeout/csv/height_scan arguments.
Use `--controller-python /path/to/deploy/python` if ONNX dependencies live in a
different environment. For task-11 transport QA only, replace it with
`--controller-command '.venv/bin/python scripts/hold_pose.py --timeout 15'`.
This is a real SDK hold-pose controller, not a balance-policy test. The supplied
CSV goes to the deploy controller; a sibling `*.sim.csv` always records MuJoCo
ground truth. Subprocess failure/early exit fails the harness; an isolated process
group ensures controller descendants are terminated as well. Video frame count
is `round(timeout * fps)`, with simulation time paced against wall time.

`OffscreenRenderer` provides RGB and optical-axis metric depth (848x480) from
`d455_front` and `d455_rear`, using independent 87x58 degree focal lengths.
Third-person RGB is the default video view. EGL must be selected before importing
MuJoCo. MP4 encoding uses explicit imageio/imageio-ffmpeg dependencies.

### Explicit sim-to-real differences for the task-24 audit

- MuJoCo retains vendor torque limits: 150 Nm pitch/roll/knee, 60 yaw, 22 wheel;
  these differ from RL training limits and must not be silently treated as aligned.
- Vendor armatures (0.0685894 / 0.0169834 / 0.0110718), damping 0.01,
  contact friction (1.0, 0.3, 0.3) and 1 ms timestep are retained. Controller gains
  come through the SDK; sensor noise, transport latency and actuator delays are
  not calibrated to hardware. Payload inertia is nominal, without randomization.
- Heightfields preserve sample heights but cannot represent Isaac's horizontal
  vertex shifts used to make steep risers vertical (`slope_threshold=0.75`).
  Stair/rubble edge contacts therefore differ. The obstacle tile spans 6x6 m,
  surrounded by the exported config's 20 m flat border (also group-5 hfields).
  Long drives beyond the central tile no longer exercise the named obstacle.
- Depth is ideal noiseless z-depth, not a D455 sensor model (no stereo holes,
  USB delay or lens distortion). Hardware IMU extrinsics remain unverified;
  this model uses the RL URDF IMU transform.
