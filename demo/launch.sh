#!/usr/bin/env bash
# Launch the TRON2 Stand-up & Drive demo (desktop icon target). Arguments go to demo/app.py.
set -uo pipefail
DEMO="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
SIM="$(dirname "$DEMO")"
LOG="$HOME/tron2_demo.log"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"  # uv when started from the desktop (no login shell)
LIB="$HOME/.local/share/tron2-demo/lib"  # libxcb-cursor0 for Qt's xcb plugin when the system lacks it
if [ -d "$LIB" ]; then export LD_LIBRARY_PATH="$LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; fi
export MUJOCO_GL=egl
cd "$SIM" || exit 1

fail() {
    echo "$(date '+%F %T') LAUNCH_ERROR $1" >>"$LOG"
    if [ -n "${DISPLAY:-}" ] && command -v zenity >/dev/null; then
        zenity --error --title="TRON2 데모" --width=420 --text="$1" 2>/dev/null
    fi
    exit 1
}

# The limxsdk loopback uses fixed localhost ports: a second simulator/controller pair would cross-talk.
if pgrep -f "tron2_sim/demo/sim_server\.py|tron2_deploy/run_policy\.py|scripts/sim2sim\.py" >/dev/null; then
    fail "다른 TRON2 시뮬레이터/제어기가 이미 실행 중입니다 (limxsdk 포트 공유).\n먼저 종료한 뒤 다시 실행하세요."
fi
command -v uv >/dev/null || fail "uv를 찾을 수 없습니다 (~/.local/bin/uv)."

echo "$(date '+%F %T') LAUNCH $0 $*" >>"$LOG"
uv run --extra sdk --extra demo python demo/app.py "$@" 2>>"$LOG"
code=$?
if [ $code -ne 0 ] && [ $code -ne 130 ]; then
    fail "TRON2 데모가 오류로 종료되었습니다 (exit $code).\n로그: $LOG"
fi
exit $code
