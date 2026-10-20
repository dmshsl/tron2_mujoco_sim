#!/usr/bin/env bash
# Install the "TRON2 데모" desktop launcher for the current user:
#   1. uv environment with the sdk + demo extras
#   2. libxcb-cursor0 (needed by Qt >= 6.5's xcb plugin) unpacked to ~/.local/share/tron2-demo/lib when the
#      system does not have it (no root needed; `sudo apt install libxcb-cursor0` makes this step a no-op)
#   3. robot icon PNG, .desktop file on the desktop (xdg-user-dir DESKTOP) and in ~/.local/share/applications
set -euo pipefail
DEMO="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
SIM="$(dirname "$DEMO")"
SHARE="$HOME/.local/share/tron2-demo"
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$SHARE"

cd "$SIM"
uv sync --extra sdk --extra demo --inexact

if ! ldconfig -p | grep -q "libxcb-cursor.so.0"; then
    if [ ! -e "$SHARE/lib/libxcb-cursor.so.0" ]; then
        work="$(mktemp -d)"
        (cd "$work" && apt-get download libxcb-cursor0 && dpkg -x libxcb-cursor0_*.deb root)
        mkdir -p "$SHARE/lib"
        cp -a "$work"/root/usr/lib/*/libxcb-cursor.so.0* "$SHARE/lib/"
        rm -rf "$work"
    fi
    echo "libxcb-cursor0: $SHARE/lib"
fi

MUJOCO_GL=egl uv run --extra sdk --extra demo python demo/make_icon.py "$SHARE/tron2_icon.png" 2>/dev/null | tail -1
chmod +x "$DEMO/launch.sh"

DESKTOP_DIR="$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")"
mkdir -p "$DESKTOP_DIR" "$HOME/.local/share/applications"
ENTRY="$DESKTOP_DIR/tron2-demo.desktop"
cat >"$ENTRY" <<EOF
[Desktop Entry]
Type=Application
Version=1.0
Name=TRON2 데모
Name[ko]=TRON2 데모
Comment=TRON2 Stand-up & Drive (MuJoCo, 학습 기립 정책 + Base 주행)
Exec=$DEMO/launch.sh
Path=$SIM
Icon=$SHARE/tron2_icon.png
Terminal=false
Categories=Science;Education;
StartupNotify=true
StartupWMClass=tron2-demo
EOF
chmod +x "$ENTRY"
gio set "$ENTRY" metadata::trusted true 2>/dev/null || echo "note: gio could not mark the launcher trusted"
cp "$ENTRY" "$HOME/.local/share/applications/tron2-demo.desktop"
chmod +x "$HOME/.local/share/applications/tron2-demo.desktop"
command -v update-desktop-database >/dev/null && update-desktop-database "$HOME/.local/share/applications" || true
echo "installed: $ENTRY"
