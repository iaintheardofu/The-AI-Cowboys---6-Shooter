# syntax=docker/dockerfile:1
# Multi-stage build → fully static binary on a scratch runtime image.
# Build from the crate root:  docker build -f deploy/Dockerfile -t yield-daemon:VERSION .

# ---- Builder ----
FROM rust:1.96-slim AS builder
RUN apt-get update && apt-get install -y --no-install-recommends musl-tools \
    && rm -rf /var/lib/apt/lists/* \
    && rustup target add x86_64-unknown-linux-musl
WORKDIR /build
COPY . .
ENV CC_x86_64_unknown_linux_musl=musl-gcc \
    CARGO_TARGET_X86_64_UNKNOWN_LINUX_MUSL_LINKER=musl-gcc
RUN cargo build --release --target x86_64-unknown-linux-musl \
    && cp target/x86_64-unknown-linux-musl/release/yield-daemon /yield-daemon

# ---- Runtime (scratch — binary is fully static) ----
FROM scratch
COPY --from=builder /yield-daemon /yield-daemon
COPY --from=builder /build/config.toml /config.toml
# Prometheus metrics port declared by the daemon CLI (see #2 in deploy README).
EXPOSE 9191
# --dry-run on purpose; override the CMD to go live after configuring config.toml.
ENTRYPOINT ["/yield-daemon", "--config", "/config.toml"]
CMD ["--dry-run"]
