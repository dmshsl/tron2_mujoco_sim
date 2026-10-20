# TRON2 Stand-up & Drive 데모

발표용 MuJoCo 데모입니다. 실제 배포 스택을 그대로 씁니다.

- 시뮬레이터: `tron2_sim` wf_payload 모델, 평면 바닥(데모용으로 ±60 m로 넓힘), limxsdk 루프백
- 제어기: `tron2_deploy/run_policy.py` + `PolicyWheelfootController`
  - 학습 기립 정책 `controllers/model/WF_TRON2A_STANDUP` (`--standup learned`)
  - 기립 후 Base 정책 `controllers/model/WF_TRON2A_BASE`
- 앱을 켤 때 설치되어 있는 Base 모델을 그대로 읽습니다. 모델을 바꿔도 앱은 고칠 필요가 없습니다.

## 실행

- 바탕화면의 **TRON2 데모** 아이콘을 더블클릭합니다.
- 터미널에서 실행하려면:
  ```bash
  ~/tron2_ws/src/tron2/tron2_sim/demo/launch.sh            # 창 모드
  ~/tron2_ws/src/tron2/tron2_sim/demo/launch.sh --fullscreen
  ```
- 처음 설치하거나 다시 설치할 때:
  ```bash
  ~/tron2_ws/src/tron2/tron2_sim/demo/install_desktop.sh
  ```
  - uv 환경(`--extra sdk --extra demo`)을 맞춥니다.
  - 시스템에 `libxcb-cursor0`이 없으면 `~/.local/share/tron2-demo/lib`에 풀어 둡니다(Qt 6.5 이상 xcb 플러그인에 필요, root 불필요). `sudo apt install libxcb-cursor0`을 하면 이 단계는 필요 없습니다.
  - 아이콘(`~/.local/share/tron2-demo/tron2_icon.png`)을 만듭니다.
  - `tron2-demo.desktop`을 `xdg-user-dir DESKTOP`(이 PC는 `~/Desktop`)과 `~/.local/share/applications/`에 놓고, 실행 권한과 `metadata::trusted`를 설정합니다.
- 로그: `~/tron2_demo.log` (앱, 시뮬레이터, 제어기 출력이 모두 들어갑니다)
- 다른 limxsdk 시뮬레이터/제어기(`scripts/sim2sim.py`, `run_policy.py`)가 실행 중이면 실행을 거부합니다. limxsdk 루프백 포트가 고정이라 두 쌍이 동시에 돌면 서로의 패킷을 받습니다.

## 화면과 조작

- 왼쪽: 3D 화면(카메라가 로봇을 따라감), 오른쪽: 제어 패널
- 마우스
  - 왼쪽 드래그: 회전
  - 휠 또는 오른쪽 드래그: 줌
  - 가운데 드래그: 이동(추적 해제)
  - 더블클릭: 카메라 초기화(추적 다시 켬)

| 버튼 | 동작 |
|---|---|
| **소환 (널브러진 상태)** | 무작위 "가지런한" 널브러진 자세로 로봇을 놓습니다. 사람이 다리를 옆으로 가지런히 펴서 내려놓은 자세(pose 1)에 관절마다 ±35° 무작위 변화를 준 뒤, 힘을 뺀 채 1 s 동안 바닥에 가라앉힙니다. 가라앉은 뒤 두 다리가 모두 roll family(A')에 닿을 수 있고 CoM이 base 면 위(가장자리 뒤로 최대 1 cm)에 있는 자세만 씁니다. 아니면 같은 시드에서 다시 뽑습니다(`heap.py`). |
| **기립** | 경로 계획 → 다리 정리 → 초기자세 A' → 학습 기립 → Base로 균형 인계 |
| **주행 모드** | 서 있을 때만 켜집니다. 방향키 또는 화면의 화살표 패드로 운전합니다. |
| **리셋** | 제어를 멈추고(힘 빠짐) 로봇을 기본 자세(pose 1)로 되돌립니다. 카메라도 초기화합니다. |

**주행 키** (어느 위젯에 포커스가 있든 동작)

| 키 | 동작 |
|---|---|
| ↑ / ↓ | vx를 vx_step(기본 0.5 m/s)만큼 더하기/빼기 |
| ← / → | wz를 wz_step(기본 0.25 rad/s)만큼 더하기/빼기 (← = +wz = 좌회전) |
| Space | 정지(vx = wz = 0) |

- 명령은 누를 때마다 누적되고, 소프트웨어 상한에서 잘립니다.
- 상한 슬라이더는 학습 최대값까지 올릴 수 있습니다.
- 값은 시작할 때 `~/tron2_ws/artifacts/basecampaign/demo_command_limits.json`에서 읽습니다(`vx_step`, `wz_step`, `vx_cap_default`, `wz_cap_default`, `vx_max`, `wz_max`). 파일이 없으면 0.5 / 0.25 / 1.0 / 1.0을 씁니다.
- 명령은 배포 제어기의 명령 정형(`command_shaping.py`: 학습 범위 clamp, 소프트웨어 상한, 가감속 제한)을 거쳐 정책에 들어갑니다.
  - 상한 슬라이더 값은 `set_command_cap`으로 제어기에 전달됩니다.
  - 패널에는 명령 → 정책 입력(정형 후) · 실제 속도를 보여 줍니다.
  - 아래 막대: 주황 = 명령, 파랑 = 실제, 회색 = 상한

## 단계 표시

| 표시 | 의미 (제어기 `standup_phase`) |
|---|---|
| 소환 중 | 자세 배치와 1 s 무력 정착 |
| 널브러진 상태 | 제어기 대기(SDK 연결, 게인 0). [기립] 가능 |
| 경로 계획 중... | P0: 측정 자세에서 앉은 경로(seated route)를 계획하고 인증합니다. 별도 프로세스에서 돌며, 예산은 40 s입니다. |
| 다리 정리 | P0 유지, P1 첫 다리 들기, P2 공중에서 A_air로, P3 A'로 내리기 |
| 초기자세 A' | HOLD: A'에 정착 |
| 기립 | 학습 기립 정책(phase 5) |
| 균형 인계 | catch 영역에 처음 들어간 순간 Base로 인계(phase 6, 처음 2 s 표시) |
| 서 있음 | Base 정책이 균형을 잡고 있음. [주행 모드] 가능 |
| 안전 정지 | 제어기 안전 래치(사유 표시). [소환]/[리셋]으로 새 제어기를 시작합니다. |

## 구조

- `app.py`: 창(PySide6), 키 입력, live command 쓰기, 단계 표시. 렌더링과 입력만 맡습니다.
- `viewport.py`: EGL 오프스크린 MuJoCo 렌더(발표용 조명/격자 바닥은 `model.py`의 렌더 전용 사본), 추적 카메라, HUD
- `panel.py`: 제어 패널
- `sim_server.py`: 실시간 물리(1 kHz) + limxsdk 상태 발행 + 제어기 프로세스 관리. GUI와 분리된 별도 프로세스입니다.
- `tron2_deploy/live_commands.py`: GUI ↔ `run_policy.py --live-commands` 공유 메모리(속도 명령, 상한, 모드 idle/run, 상태와 래치 사유)
- `heap.py`: 가지런한 널브러진 자세와 무력 정착
- `automation.py`: `--selftest`, `--batch`
- CPU 배치
  - 제어기: 마지막 3개 코어
  - 물리: 그 앞 2개 코어
  - GUI와 경로 계획 프로세스: 나머지 코어(`TRON2_PLANNER_CPUS`)
- 종료
  - 창을 닫으면 제어기 프로세스 그룹(경로 계획 워커 포함), 물리 프로세스, 공유 메모리 파일까지 모두 정리합니다.
  - GUI가 비정상 종료해도 `PR_SET_PDEATHSIG`로 자식 프로세스가 함께 종료됩니다.
- 높이 스캔: `--height_scan gt`를 씁니다.
  - 평면 바닥에서는 GT 스캔이 정확합니다.
  - perception(깊이 렌더 → 높이맵)은 GPU/CPU를 3D 화면과 나눠 쓰게 됩니다. 이 데모에서는 이득이 없어 쓰지 않았습니다.

## 자체 시험

```bash
cd ~/tron2_ws/src/tron2/tron2_sim
demo/launch.sh --selftest            # 버튼/키를 자동으로 누르고 창을 녹화 (60-90 s)
                                     # -> artifacts/basecampaign/demo/demo_selftest.mp4 (+ 단계별 PNG)
demo/launch.sh --batch 30 --seed 1000  # 30 x (소환 -> 기립 -> 10 s 스크립트 주행), reliability.jsonl
```

## 알려진 한계

- 학습 기립 정책은 A'/A' 시작만 학습했습니다.
  - 그래서 경로가 두 다리를 모두 A'(roll family)로 보내도록 우선합니다(`tron2_common.seated_route.PREFERRED_FAMILY`).
  - 다리가 pitch family(A'_p)에만 닿을 수 있는 자세는 혼합 시작이 되고, 실패 가능성이 큽니다(round 4: 0/5).
- 경로 계획기는 일부 널브러진 자세를 거부합니다. 이 경우 제어기가 "안전 정지"로 끝나고 로봇은 움직이지 않습니다.
  - 넘어짐이 예측되는 자세: CoM이 base 면 뒤로 1 cm 넘게 나가는 경우
  - 40 s 계획 예산 초과
  - 이때는 [소환]을 다시 누르면 됩니다.
- MuJoCo 체인은 한 번에 하나만 돌 수 있습니다(limxsdk 루프백 포트 공유).
- 실기 검증이 아닙니다.
  - 시뮬레이션 접촉, 바퀴 armature, IMU는 모두 모델 값입니다.
  - 실제 로봇에서는 이 데모를 쓰지 않습니다(`run_policy.py`는 127.0.0.1만 허용).
