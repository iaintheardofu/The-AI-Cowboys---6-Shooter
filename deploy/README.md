# Yield Daemon — Deployment

Three ways to ship the daemon as a standalone artifact. The binary is a fully
**static musl** build (no glibc, no shared libraries) and runs on any x86_64
Linux. It always starts in `--dry-run` (no real transactions) until you edit
`config.toml` and remove the flag.

## 1. Static binary (simplest)

```bash
./deploy/package.sh        # builds + emits deploy/dist/yield-daemon-<ver>-x86_64-linux-musl.tar.gz
```

The binary is self-contained:

```
$ ldd yield-daemon
        statically linked
$ ./yield-daemon --config config.toml --dry-run
```

## 2. systemd service (tarball)

```bash
tar xzf yield-daemon-<ver>-x86_64-linux-musl.tar.gz
cd yield-daemon-<ver>-x86_64-linux-musl
sudo ./install.sh          # installs to /opt/yield-daemon, enables + starts the unit
journalctl -u yield-daemon -f
```

The unit (`yield-daemon.service`) runs hardened: `DynamicUser`, `ProtectSystem=strict`,
`NoNewPrivileges`, `MemoryMax=2G`. Edit `/opt/yield-daemon/config.toml` then
`systemctl restart yield-daemon`.

## 3. Docker

```bash
# from the crate root (platforms/yield-daemon)
docker build -f deploy/Dockerfile -t yield-daemon:0.1.0 .
docker run --rm yield-daemon:0.1.0            # dry-run
```

Runtime image is `FROM scratch` (just the static binary + config) — a few MB.
Mount a config and persist state:

```bash
docker run -d --name yield-daemon \
  -v $PWD/config.toml:/config.toml:ro \
  -v yield-state:/runtime/yield_daemon \
  yield-daemon:0.1.0
```

## Notes

- **Portability:** release builds intentionally do **not** pin `target-cpu=native`
  (it bakes in the build host's ISA → SIGILL elsewhere). For single-host max
  perf: `RUSTFLAGS="-C target-cpu=native" cargo build --release`.
- **Metrics:** the daemon writes `{zk,mev,ml}_metrics.json` under
  `state_dir` (default `runtime/yield_daemon`) every `metrics_interval_secs`;
  the Python orchestrator polls those. The `--metrics-port 9191` Prometheus
  HTTP endpoint is **not yet served** (declared only) — file-based metrics are
  the working channel today.
- **Going live:** flip `dry_run = false` and configure RPC endpoints + wallets
  in `config.toml`. The orchestrator additionally gates live mode behind a
  Human Gate. Understand the risk model first.
