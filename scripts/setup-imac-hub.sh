#!/usr/bin/env bash
# =============================================================================
# Neural Memory Hub Server — iMac Setup Script
# =============================================================================
# Run this on iMac to set up as a sync hub for multi-device brain sync.
#
# Usage:
#   chmod +x setup-imac-hub.sh
#   ./setup-imac-hub.sh
#
# After setup, configure Windows PC to sync:
#   nmem sync-config set --hub-url http://<iMac-IP>:8369 --enabled --auto-sync
#   nmem sync push
# =============================================================================

set -euo pipefail

# ── Config ───────────────────────────────────────────────────────────────────
NM_VERSION="2.29.0"
HUB_PORT=8369
HUB_HOST="0.0.0.0"
LAN_CIDR="192.168.0.0/16"  # Adjust if your LAN uses a different range
LAUNCHD_LABEL="com.neuralmemory.hub"
PLIST_PATH="$HOME/Library/LaunchAgents/${LAUNCHD_LABEL}.plist"

# ── Colors ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Step 1: Check Python ────────────────────────────────────────────────────
info "Checking Python installation..."
if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    error "Python not found. Install with: brew install python@3.12"
fi

PY_VERSION=$($PYTHON --version 2>&1 | awk '{print $2}')
ok "Python $PY_VERSION found at $(which $PYTHON)"

# ── Step 2: Install/Upgrade Neural Memory ────────────────────────────────────
info "Installing neural-memory v${NM_VERSION}..."
$PYTHON -m pip install --upgrade "neural-memory==${NM_VERSION}" 2>&1 | tail -3

# Verify
INSTALLED=$($PYTHON -c "import neural_memory; print(neural_memory.__version__)" 2>/dev/null || echo "FAIL")
if [ "$INSTALLED" != "$NM_VERSION" ]; then
    error "Installation failed. Expected v${NM_VERSION}, got: $INSTALLED"
fi
ok "neural-memory v${INSTALLED} installed"

# ── Step 3: Verify nmem CLI ─────────────────────────────────────────────────
info "Checking nmem CLI..."
NMEM_PATH=$(which nmem 2>/dev/null || echo "")
if [ -z "$NMEM_PATH" ]; then
    # Try common pip bin paths on macOS
    for p in "$HOME/.local/bin/nmem" "$HOME/Library/Python/3.12/bin/nmem" "/opt/homebrew/bin/nmem"; do
        if [ -x "$p" ]; then
            NMEM_PATH="$p"
            break
        fi
    done
fi

if [ -z "$NMEM_PATH" ]; then
    warn "nmem not in PATH. You may need to add ~/.local/bin to PATH:"
    warn '  echo '\''export PATH="$HOME/.local/bin:$PATH"'\'' >> ~/.zshrc && source ~/.zshrc'
    NMEM_PATH="$PYTHON -m neural_memory.cli"
    info "Falling back to: $NMEM_PATH"
else
    ok "nmem CLI at $NMEM_PATH"
fi

# ── Step 4: Show LAN IP ─────────────────────────────────────────────────────
info "Detecting LAN IP..."
LAN_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "unknown")
if [ "$LAN_IP" = "unknown" ]; then
    warn "Could not auto-detect LAN IP. Check manually: ifconfig | grep inet"
else
    ok "iMac LAN IP: $LAN_IP"
fi

# ── Step 5: Create launchd plist (auto-start on boot) ───────────────────────
info "Creating launchd service for Hub Server..."

cat > "$PLIST_PATH" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${LAUNCHD_LABEL}</string>

    <key>ProgramArguments</key>
    <array>
        <string>$(which $PYTHON)</string>
        <string>-m</string>
        <string>neural_memory.cli</string>
        <string>serve</string>
        <string>--host</string>
        <string>${HUB_HOST}</string>
        <string>--port</string>
        <string>${HUB_PORT}</string>
    </array>

    <key>EnvironmentVariables</key>
    <dict>
        <key>NEURAL_MEMORY_TRUSTED_NETWORKS</key>
        <string>${LAN_CIDR}</string>
        <key>PATH</key>
        <string>/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:$HOME/.local/bin</string>
    </dict>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>

    <key>StandardOutPath</key>
    <string>${HOME}/.neural-memory/hub-server.log</string>
    <key>StandardErrorPath</key>
    <string>${HOME}/.neural-memory/hub-server.error.log</string>

    <key>WorkingDirectory</key>
    <string>${HOME}</string>

    <key>ThrottleInterval</key>
    <integer>10</integer>
</dict>
</plist>
PLIST

ok "Launchd plist created at $PLIST_PATH"

# ── Step 6: Load the service ────────────────────────────────────────────────
info "Loading Hub Server service..."

# Unload if already running
launchctl bootout "gui/$(id -u)/${LAUNCHD_LABEL}" 2>/dev/null || true
sleep 1

launchctl bootstrap "gui/$(id -u)" "$PLIST_PATH"
ok "Service loaded"

# ── Step 7: Verify server is running ────────────────────────────────────────
info "Waiting for server to start..."
sleep 3

if curl -sf "http://127.0.0.1:${HUB_PORT}/health" >/dev/null 2>&1; then
    ok "Hub Server is running on port ${HUB_PORT}"
elif curl -sf "http://127.0.0.1:${HUB_PORT}/docs" >/dev/null 2>&1; then
    ok "Hub Server is running on port ${HUB_PORT} (docs accessible)"
else
    warn "Server may still be starting. Check logs:"
    warn "  tail -f ~/.neural-memory/hub-server.log"
    warn "  tail -f ~/.neural-memory/hub-server.error.log"
fi

# ── Step 8: Configure local sync ────────────────────────────────────────────
info "Configuring local sync on iMac..."
$PYTHON -m neural_memory.cli sync-config set \
    --hub-url "http://127.0.0.1:${HUB_PORT}" \
    --enabled \
    --auto-sync 2>/dev/null || warn "Could not auto-configure sync. Do it manually via MCP."

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN} Neural Memory Hub Server — Setup Complete${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  Hub URL:     ${CYAN}http://${LAN_IP}:${HUB_PORT}${NC}"
echo -e "  Dashboard:   ${CYAN}http://${LAN_IP}:${HUB_PORT}/ui${NC}"
echo -e "  API Docs:    ${CYAN}http://${LAN_IP}:${HUB_PORT}/docs${NC}"
echo -e "  Logs:        ~/.neural-memory/hub-server.log"
echo -e "  Service:     launchctl print gui/$(id -u)/${LAUNCHD_LABEL}"
echo ""
echo -e "  ${YELLOW}On Windows PC, run:${NC}"
echo -e "    nmem sync-config set --hub-url http://${LAN_IP}:${HUB_PORT} --enabled --auto-sync"
echo -e "    nmem sync push"
echo ""
echo -e "  ${YELLOW}Management commands:${NC}"
echo -e "    Stop:    launchctl bootout gui/$(id -u)/${LAUNCHD_LABEL}"
echo -e "    Start:   launchctl bootstrap gui/$(id -u) $PLIST_PATH"
echo -e "    Logs:    tail -f ~/.neural-memory/hub-server.log"
echo ""
