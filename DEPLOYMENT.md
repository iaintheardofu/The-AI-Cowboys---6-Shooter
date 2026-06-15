# Yield Daemon — Deployment Guide

Production deployment for the yield daemon. The daemon is a single, fully
**static** binary (musl, no glibc, no shared libraries) that runs on any x86_64
Linux host. It starts in `--dry-run` (no real transactions) until you
deliberately configure it for live operation.

---

## External requirements — none

The daemon is **self-contained**:

- **No API keys.** It ships no cloud credentials and needs none to run. In
  dry-run it makes no outbound calls; for live operation you supply your own
  RPC endpoints/wallets in `config.toml` (see [Going live](#going-live)).
- **No external model / GPU.** The daemon does its own compute; it does not call
  any LLM or model server.
- **No runtime dependencies.** `ldd ./yield-daemon` → `statically linked`.

> Note: the *Stallion* coding assistant in this org is the component that runs
> entirely on a **local Ollama** model (no cloud keys) — that requirement is
> Stallion's, not this daemon's. They are separate systems.

---

## Quick start (one command each)

```bash
# Build the static binary + release tarball
./deploy/package.sh

# Install as a hardened systemd service (binary → /opt, state → /var/lib)
sudo ./deploy/install.sh
systemctl status yield-daemon
```

`install.sh` works both from the source tree and from an extracted tarball.

---

## Install methods

### 1. systemd (recommended for servers)

```bash
./deploy/package.sh
sudo ./deploy/install.sh          # add --no-start to install without starting
journalctl -u yield-daemon -f     # logs (structured JSON)
```

Layout: binary + `config.toml` are installed read-only under `/opt/yield-daemon`;
writable state lives in `/var/lib/yield-daemon` (systemd `StateDirectory`, owned
by a transient `DynamicUser`). The unit runs hardened: `DynamicUser`,
`ProtectSystem=strict`, `NoNewPrivileges`, `PrivateTmp`, `MemoryMax=2G`.

Manage:

```bash
systemctl restart yield-daemon    # after editing /opt/yield-daemon/config.toml
systemctl stop yield-daemon
systemctl disable --now yield-daemon
```

### 2. Docker

```bash
docker build -f deploy/Dockerfile -t yield-daemon:0.1.0 .   # from the crate root
docker run -d --name yield-daemon -p 9191:9191 \
  -v $PWD/config.toml:/config.toml:ro \
  -v yield-state:/runtime/yield_daemon \
  yield-daemon:0.1.0
```

Runtime image is `FROM scratch` (static binary + config), ~2.3 MB.

### 3. Bare binary / tarball

```bash
tar xzf deploy/dist/yield-daemon-<ver>-x86_64-linux-musl.tar.gz
cd yield-daemon-<ver>-x86_64-linux-musl
./yield-daemon --config config.toml --dry-run
```

---

## Prometheus metrics

The daemon serves a Prometheus exposition endpoint at `:9191/metrics`
(`--metrics-port` to change). Scrape config:

```yaml
scrape_configs:
  - job_name: yield-daemon
    scrape_interval: 30s
    static_configs:
      - targets: ['HOST:9191']
```

Exposed series (all `counter` except uptime `gauge`):

```
yield_daemon_zk_proofs_generated      yield_daemon_mev_opportunities_detected
yield_daemon_zk_proofs_accepted       yield_daemon_mev_bundles_submitted
yield_daemon_zk_revenue_sat           yield_daemon_mev_revenue_sat
yield_daemon_ml_inferences_served     yield_daemon_ml_training_rounds
yield_daemon_ml_revenue_sat           yield_daemon_total_cycles
yield_daemon_uptime_seconds
```

The daemon also writes `{zk,mev,ml}_metrics.json` to `state_dir` every
`metrics_interval_secs` — that JSON bridge is what the Python orchestrator polls
(the Prometheus endpoint is for external scrapers).

---

## Configuration

All settings live in `config.toml`; every value has a safe default and the
daemon starts in dry-run. Key sections: `[general]` (dry_run, state_dir,
metrics_interval_secs), `[zk]`, `[mev]`, `[ml]`, `[risk]`.

### Going live

1. Set `dry_run = false`.
2. Uncomment and fill RPC endpoints + wallets (`succinct_rpc`, Solana
   `rpc_endpoints`/`jito_block_engine`, `bittensor_endpoint`, etc.).
3. Review `[risk]` (max capital at risk, circuit breaker threshold).

The Python orchestrator additionally gates live-mode activation behind a Human
Gate. **Understand the risk model before disabling dry-run.**

---

## Verify a deployment

```bash
./deploy/smoke_test.sh    # builds + tests static binary, tarball, Docker:
                          # boots dry-run, writes metrics, serves :9191
```

Exit 0 = all stages pass. Safe for CI.

---

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `SIGILL` / illegal instruction on a different host | Binary built with `target-cpu=native`. Rebuild without it (the default), or build on the target ISA. |
| `[Metrics] cannot create state_dir ... Permission denied` | Service's writable dir isn't owned by the run user. Use the shipped unit (`StateDirectory=yield-daemon` → `/var/lib/yield-daemon`); don't point `state_dir` at a root-owned path. |
| `:9191` not reachable | Another process holds the port (`ss -tlnp | grep 9191`), or container started without `-p 9191:9191`. Change with `--metrics-port`. |
| `Config load failed ... using defaults` | `config.toml` didn't parse — the daemon fell back to defaults. Check the WARN line for the offending key. |
| Metrics all zero | Expected in dry-run with no live endpoints — the modules simulate minimal activity. Configure live endpoints for real numbers. |
| Service won't start | `journalctl -u yield-daemon -n 50` for the panic/error; `systemctl cat yield-daemon` to confirm paths. |

---

*Built by AI Cowboys.*
