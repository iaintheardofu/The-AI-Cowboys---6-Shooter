#!/usr/bin/env bash
# Build the static musl binary and assemble a self-contained release tarball.
# Output: deploy/dist/yield-daemon-<version>-x86_64-linux-musl.tar.gz
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
TARGET="x86_64-unknown-linux-musl"
VERSION="$(grep -m1 '^version' "${ROOT}/Cargo.toml" | sed -E 's/.*"(.*)".*/\1/')"
NAME="yield-daemon-${VERSION}-x86_64-linux-musl"
DIST="${HERE}/dist"
STAGE="${DIST}/${NAME}"

echo "==> Building static binary (${TARGET})"
rustup target add "${TARGET}" >/dev/null 2>&1 || true
export CC_x86_64_unknown_linux_musl="${CC_x86_64_unknown_linux_musl:-musl-gcc}"
export CARGO_TARGET_X86_64_UNKNOWN_LINUX_MUSL_LINKER="${CARGO_TARGET_X86_64_UNKNOWN_LINUX_MUSL_LINKER:-musl-gcc}"
( cd "${ROOT}" && cargo build --release --target "${TARGET}" )

echo "==> Staging ${NAME}"
rm -rf "${STAGE}"; mkdir -p "${STAGE}"
install -m 0755 "${ROOT}/target/${TARGET}/release/yield-daemon" "${STAGE}/yield-daemon"
install -m 0644 "${ROOT}/config.toml"        "${STAGE}/config.toml"
install -m 0644 "${HERE}/yield-daemon.service" "${STAGE}/yield-daemon.service"
install -m 0755 "${HERE}/install.sh"         "${STAGE}/install.sh"
install -m 0644 "${HERE}/README.md"          "${STAGE}/README.md"

echo "==> Compressing"
tar -C "${DIST}" -czf "${DIST}/${NAME}.tar.gz" "${NAME}"
sha256sum "${DIST}/${NAME}.tar.gz" | tee "${DIST}/${NAME}.tar.gz.sha256"
rm -rf "${STAGE}"
echo "==> Done: ${DIST}/${NAME}.tar.gz"
