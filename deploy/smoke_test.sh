#!/usr/bin/env bash
# End-to-end smoke test for the yield-daemon deployment artifacts.
# Verifies, from source, that each shippable form boots in dry-run, writes the
# metrics file bridge, and serves the Prometheus :9191 endpoint.
#
#   Stage 1: static musl binary
#   Stage 2: release tarball (package.sh) + contents + checksum
#   Stage 3: Docker image (build + run)
#
# Exit 0 = all stages pass. Safe to run in CI.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
TARGET="x86_64-unknown-linux-musl"
PORT="${SMOKE_PORT:-19191}"   # non-default port: never clashes with a live daemon
PASS=0; FAIL=0
ok()  { echo "  ✅ $1"; PASS=$((PASS+1)); }
bad() { echo "  ❌ $1"; FAIL=$((FAIL+1)); }
wait_http() { local url=$1 t=${2:-10}; for _ in $(seq "$t"); do curl -sf -o /dev/null "$url" && return 0; sleep 1; done; return 1; }

command -v musl-gcc >/dev/null || { echo "musl-gcc missing (apt-get install musl-tools)"; exit 2; }
export CC_x86_64_unknown_linux_musl=musl-gcc
export CARGO_TARGET_X86_64_UNKNOWN_LINUX_MUSL_LINKER=musl-gcc

echo "== Stage 1: static musl binary =="
( cd "$ROOT" && cargo build --release --target "$TARGET" ) >/tmp/smoke_build.log 2>&1 \
  && ok "compiles" || { bad "compile (see /tmp/smoke_build.log)"; }
BIN="$ROOT/target/$TARGET/release/yield-daemon"
file "$BIN" 2>/dev/null | grep -q "static" && ok "statically linked" || bad "not static"
WORK="$(mktemp -d)"; cp "$ROOT/config.toml" "$WORK/"
pushd "$WORK" >/dev/null
"$BIN" --config config.toml --dry-run --metrics-port "$PORT" >boot.log 2>&1 &
PID=$!
popd >/dev/null
sleep 3
grep -q "DRY RUN" "$WORK/boot.log" && ok "boots in dry-run" || bad "did not boot"
[ -f "$WORK/runtime/yield_daemon/zk_metrics.json" ] && ok "writes metrics file bridge" || bad "no metrics files"
if wait_http "http://localhost:$PORT/metrics" 8 && curl -s "http://localhost:$PORT/metrics" | grep -q "yield_daemon_"; then
  ok "serves Prometheus :$PORT/metrics"; else bad "no Prometheus endpoint"; fi
kill "$PID" 2>/dev/null; sleep 1; kill -9 "$PID" 2>/dev/null; rm -rf "$WORK"

echo "== Stage 2: release tarball =="
"$HERE/package.sh" >/tmp/smoke_pkg.log 2>&1 && ok "package.sh builds tarball" || bad "package.sh failed"
TARBALL="$(ls "$HERE"/dist/yield-daemon-*-x86_64-linux-musl.tar.gz 2>/dev/null | head -1)"
[ -n "$TARBALL" ] && ok "tarball: $(basename "$TARBALL")" || bad "no tarball produced"
if [ -n "$TARBALL" ]; then
  EX="$(mktemp -d)"; tar xzf "$TARBALL" -C "$EX"; D="$(ls -d "$EX"/yield-daemon-*/)"
  for f in yield-daemon config.toml yield-daemon.service install.sh README.md; do
    [ -e "$D/$f" ] && ok "tarball has $f" || bad "tarball missing $f"
  done
  ( cd "$HERE/dist" && sha256sum -c "$(basename "$TARBALL").sha256" >/dev/null 2>&1 ) && ok "sha256 verifies" || bad "sha256 mismatch"
  rm -rf "$EX"
fi

echo "== Stage 3: Docker image =="
if command -v docker >/dev/null && docker info >/dev/null 2>&1; then
  ( cd "$ROOT" && docker build -q -f deploy/Dockerfile -t yield-daemon:smoke . ) >/tmp/smoke_docker.log 2>&1 \
    && ok "image builds" || bad "image build (see /tmp/smoke_docker.log)"
  docker rm -f yd-smoke >/dev/null 2>&1 || true
  docker run -d --name yd-smoke -p "$PORT:9191" yield-daemon:smoke >/dev/null 2>&1
  sleep 4
  docker logs yd-smoke 2>&1 | grep -q "DRY RUN" && ok "container boots dry-run" || bad "container did not boot"
  wait_http "http://localhost:$PORT/metrics" 8 && ok "container serves :9191/metrics" || bad "container no metrics"
  docker stop yd-smoke >/dev/null 2>&1; docker rm -f yd-smoke >/dev/null 2>&1
else
  echo "  ⚠️  docker unavailable — skipping image stage"
fi

echo
echo "== Smoke test: ${PASS} passed, ${FAIL} failed =="
[ "$FAIL" -eq 0 ]
