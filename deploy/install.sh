#!/usr/bin/env bash
# Install the yield-daemon as a systemd service.
# Usage: sudo ./install.sh            (installs to /opt/yield-daemon, enables service)
#        sudo ./install.sh --no-start (install but do not start)
set -euo pipefail

PREFIX="/opt/yield-daemon"
UNIT="/etc/systemd/system/yield-daemon.service"
START=1
[[ "${1:-}" == "--no-start" ]] && START=0

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $EUID -ne 0 ]]; then
  echo "error: must run as root (use sudo)" >&2
  exit 1
fi

# Resolve artifacts whether run from the extracted tarball (all files next to
# this script) or directly from the source tree (deploy/ + ../target + ../config.toml).
BIN_SRC=""
for c in "${HERE}/yield-daemon" \
         "${HERE}/../target/x86_64-unknown-linux-musl/release/yield-daemon" \
         "${HERE}/../target/release/yield-daemon"; do
  [[ -x "$c" ]] && { BIN_SRC="$c"; break; }
done
[[ -n "${BIN_SRC}" ]] || { echo "error: yield-daemon binary not found — run deploy/package.sh first" >&2; exit 1; }
CFG_SRC="${HERE}/config.toml"; [[ -f "${CFG_SRC}" ]] || CFG_SRC="${HERE}/../config.toml"
UNIT_SRC="${HERE}/yield-daemon.service"

echo "==> Installing yield-daemon to ${PREFIX}  (binary: ${BIN_SRC})"
install -d -m 0755 "${PREFIX}" "${PREFIX}/runtime"
install -m 0755 "${BIN_SRC}"  "${PREFIX}/yield-daemon"
install -m 0644 "${CFG_SRC}"  "${PREFIX}/config.toml"
install -m 0644 "${UNIT_SRC}" "${UNIT}"

echo "==> Reloading systemd"
systemctl daemon-reload
systemctl enable yield-daemon.service

if [[ $START -eq 1 ]]; then
  echo "==> Starting yield-daemon (dry-run mode)"
  systemctl restart yield-daemon.service
  sleep 1
  systemctl --no-pager --lines=10 status yield-daemon.service || true
else
  echo "==> Installed but not started. Start with: systemctl start yield-daemon"
fi

echo "==> Done. Logs: journalctl -u yield-daemon -f"
echo "    Config: ${PREFIX}/config.toml  (edit, then: systemctl restart yield-daemon)"
