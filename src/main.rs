//! Yield Daemon — Autonomous Yield Generation Infrastructure
//!
//! Headless daemon that converts compute into decentralized yield across:
//! 1. ZK Prover Networks (Succinct, Gevulot) — cryptographic proof generation
//! 2. MEV Arbitrage / Intent Solving — sub-millisecond on-chain extraction
//! 3. Decentralized ML Subnets (Bittensor) — distributed AI inference

use yield_daemon::{config, zk_prover, mev, ml_subnet, treasury, DaemonState, DaemonMetrics};

use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{info, warn, error};

#[derive(Parser, Debug)]
#[command(name = "yield-daemon", version, about = "Autonomous Yield Generation Daemon")]
struct Cli {
    #[arg(short, long, default_value = "config.toml")]
    config: PathBuf,
    #[arg(long, default_value_t = true)]
    zk: bool,
    #[arg(long, default_value_t = true)]
    mev: bool,
    #[arg(long, default_value_t = true)]
    ml: bool,
    #[arg(long, default_value_t = true)]
    dry_run: bool,
    #[arg(long, default_value_t = 9191)]
    metrics_port: u16,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_target(false)
        .with_timer(tracing_subscriber::fmt::time::uptime())
        .json()
        .init();

    let cli = Cli::parse();
    info!("Yield Daemon v{} starting", env!("CARGO_PKG_VERSION"));
    info!("Mode: {}", if cli.dry_run { "DRY RUN" } else { "LIVE" });

    let cfg = config::DaemonConfig::load(&cli.config).unwrap_or_else(|e| {
        warn!("Config load failed ({}), using defaults", e);
        config::DaemonConfig::default()
    });

    let state = Arc::new(DaemonState {
        config: cfg,
        metrics: DaemonMetrics::new(),
        running: std::sync::atomic::AtomicBool::new(true),
    });

    let shutdown_state = state.clone();
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        info!("Shutdown signal received");
        shutdown_state.running.store(false, std::sync::atomic::Ordering::Relaxed);
    });

    let mut handles = Vec::new();

    if cli.zk {
        let s = state.clone();
        handles.push(tokio::spawn(async move {
            info!("[ZK] Prover module starting");
            if let Err(e) = zk_prover::run(s).await {
                error!("[ZK] Module error: {}", e);
            }
        }));
    }

    if cli.mev {
        let s = state.clone();
        handles.push(tokio::spawn(async move {
            info!("[MEV] Arbitrage module starting");
            if let Err(e) = mev::run(s).await {
                error!("[MEV] Module error: {}", e);
            }
        }));
    }

    if cli.ml {
        let s = state.clone();
        handles.push(tokio::spawn(async move {
            info!("[ML] Subnet miner module starting");
            if let Err(e) = ml_subnet::run(s).await {
                error!("[ML] Module error: {}", e);
            }
        }));
    }

    // ── Treasury / Off-Ramp Pipeline ──────────────────────────────────────
    // The treasury module monitors accumulated profits across all domains,
    // consolidates volatile tokens to stablecoins, and triggers automated
    // fiat withdrawals to the operator's bank account.
    if state.config.treasury.enabled {
        let s = state.clone();
        handles.push(tokio::spawn(async move {
            info!("[Treasury] Off-ramp pipeline starting");
            if let Err(e) = treasury::run(s).await {
                error!("[Treasury] Pipeline error: {}", e);
            }
        }));
    }

    let uptime_state = state.clone();
    tokio::spawn(async move {
        let start = std::time::Instant::now();
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            if !uptime_state.running.load(std::sync::atomic::Ordering::Relaxed) {
                break;
            }
            uptime_state.metrics.uptime_secs
                .store(start.elapsed().as_secs(), std::sync::atomic::Ordering::Relaxed);
        }
    });

    // ── Metrics bridge ───────────────────────────────────────────────────
    // Export atomic counters to per-domain JSON files that the Python
    // off-ramp service polls.
    // Without this the Python side has no telemetry channel and reads zeros.
    let metrics_state = state.clone();
    tokio::spawn(async move {
        use std::sync::atomic::Ordering::Relaxed;
        let state_dir = PathBuf::from(&metrics_state.config.general.state_dir);
        if let Err(e) = std::fs::create_dir_all(&state_dir) {
            warn!("[Metrics] cannot create state_dir {:?}: {}", state_dir, e);
        }
        let interval = metrics_state.config.general.metrics_interval_secs.max(1);
        info!("[Metrics] exporting to {:?} every {}s", state_dir, interval);
        loop {
            let m = &metrics_state.metrics;
            write_metrics_file(&state_dir.join("zk_metrics.json"), &serde_json::json!({
                "proofs_generated": m.zk_proofs_generated.load(Relaxed),
                "proofs_accepted": m.zk_proofs_accepted.load(Relaxed),
                "revenue_sat": m.zk_revenue_sat.load(Relaxed),
            }));
            write_metrics_file(&state_dir.join("mev_metrics.json"), &serde_json::json!({
                "opportunities": m.mev_opportunities_detected.load(Relaxed),
                "bundles_submitted": m.mev_bundles_submitted.load(Relaxed),
                "revenue_sat": m.mev_revenue_sat.load(Relaxed),
            }));
            write_metrics_file(&state_dir.join("ml_metrics.json"), &serde_json::json!({
                "inferences": m.ml_inferences_served.load(Relaxed),
                "training_rounds": m.ml_training_rounds.load(Relaxed),
                "revenue_sat": m.ml_revenue_sat.load(Relaxed),
            }));
            write_metrics_file(&state_dir.join("treasury_metrics.json"), &serde_json::json!({
                "profit_accumulated": m.treasury_profit_accumulated.load(Relaxed),
                "stablecoin_balance": m.treasury_stablecoin_balance.load(Relaxed),
                "fiat_withdrawn_cents": m.treasury_fiat_withdrawn_cents.load(Relaxed),
                "offramp_cycles": m.treasury_offramp_cycles.load(Relaxed),
            }));
            // Sleep in 1s steps so shutdown stays responsive.
            for _ in 0..interval {
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                if !metrics_state.running.load(Relaxed) {
                    return;
                }
            }
        }
    });

    // ── Prometheus metrics endpoint ──────────────────────────────────────
    // Serve DaemonMetrics in Prometheus text format on --metrics-port so
    // external scrapers (Prometheus/Grafana) can read live counters. The
    // The off-ramp service uses the JSON file bridge above, not this endpoint.
    let prom_state = state.clone();
    let prom_port = cli.metrics_port;
    tokio::spawn(async move {
        if let Err(e) = serve_prometheus(prom_state, prom_port).await {
            warn!("[Metrics] Prometheus endpoint stopped: {}", e);
        }
    });

    info!("All modules spawned. Daemon running.");

    for h in handles {
        h.await?;
    }

    info!("Yield Daemon shutdown complete");
    Ok(())
}

/// Serve DaemonMetrics in Prometheus text exposition format on `port`.
/// Minimal HTTP/1.1 over a tokio TcpListener — no extra framework. Answers
/// GET /metrics (and GET /) with the metrics; anything else returns 404.
async fn serve_prometheus(state: Arc<DaemonState>, port: u16) -> anyhow::Result<()> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    let addr = format!("0.0.0.0:{port}");
    let listener = TcpListener::bind(&addr).await?;
    info!("[Metrics] Prometheus endpoint on http://{}/metrics", addr);

    loop {
        let (mut sock, _) = match listener.accept().await {
            Ok(pair) => pair,
            Err(_) => continue,
        };
        let st = state.clone();
        tokio::spawn(async move {
            let mut buf = [0u8; 1024];
            let n = sock.read(&mut buf).await.unwrap_or(0);
            let req = String::from_utf8_lossy(&buf[..n]);
            let is_metrics = req.starts_with("GET /metrics") || req.starts_with("GET / ");
            let (status, ctype, body) = if is_metrics {
                ("200 OK", "text/plain; version=0.0.4", render_prometheus(&st.metrics))
            } else {
                ("404 Not Found", "text/plain", "not found\n".to_string())
            };
            let resp = format!(
                "HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = sock.write_all(resp.as_bytes()).await;
            let _ = sock.shutdown().await;
        });
    }
}

/// Render DaemonMetrics as Prometheus text exposition format (v0.0.4).
fn render_prometheus(m: &DaemonMetrics) -> String {
    use std::fmt::Write;
    use std::sync::atomic::Ordering::Relaxed;
    let mut out = String::with_capacity(2048);
    macro_rules! emit {
        ($name:literal, $help:literal, $typ:literal, $val:expr) => {{
            let _ = write!(out, "# HELP {} {}\n# TYPE {} {}\n{} {}\n", $name, $help, $name, $typ, $name, $val);
        }};
    }
    emit!("yield_daemon_zk_proofs_generated", "ZK proofs generated", "counter", m.zk_proofs_generated.load(Relaxed));
    emit!("yield_daemon_zk_proofs_accepted", "ZK proofs accepted", "counter", m.zk_proofs_accepted.load(Relaxed));
    emit!("yield_daemon_zk_revenue_sat", "ZK revenue in satoshis", "counter", m.zk_revenue_sat.load(Relaxed));
    emit!("yield_daemon_mev_opportunities_detected", "MEV opportunities detected", "counter", m.mev_opportunities_detected.load(Relaxed));
    emit!("yield_daemon_mev_bundles_submitted", "MEV bundles submitted", "counter", m.mev_bundles_submitted.load(Relaxed));
    emit!("yield_daemon_mev_revenue_sat", "MEV revenue in satoshis", "counter", m.mev_revenue_sat.load(Relaxed));
    emit!("yield_daemon_ml_inferences_served", "ML inferences served", "counter", m.ml_inferences_served.load(Relaxed));
    emit!("yield_daemon_ml_training_rounds", "ML background training rounds", "counter", m.ml_training_rounds.load(Relaxed));
    emit!("yield_daemon_ml_revenue_sat", "ML revenue in satoshis", "counter", m.ml_revenue_sat.load(Relaxed));
    emit!("yield_daemon_total_cycles", "Total daemon cycles", "counter", m.total_cycles.load(Relaxed));
    emit!("yield_daemon_uptime_seconds", "Daemon uptime in seconds", "gauge", m.uptime_secs.load(Relaxed));
    emit!("yield_daemon_aste_cycles", "ASTE searcher cycles", "counter", m.aste_cycles.load(Relaxed));
    emit!("yield_daemon_aste_arb_detected", "ASTE arbitrage cycles detected", "counter", m.aste_arb_detected.load(Relaxed));
    emit!("yield_daemon_aste_arb_profitable", "ASTE profitable arbitrages", "counter", m.aste_arb_profitable.load(Relaxed));
    emit!("yield_daemon_aste_latency_ns", "ASTE avg cycle latency nanoseconds", "gauge", m.aste_latency_ns.load(Relaxed));
    emit!("yield_daemon_treasury_profit_accumulated", "Treasury total profit accumulated", "counter", m.treasury_profit_accumulated.load(Relaxed));
    emit!("yield_daemon_treasury_stablecoin_balance", "Treasury stablecoin balance (micro-USD)", "gauge", m.treasury_stablecoin_balance.load(Relaxed));
    emit!("yield_daemon_treasury_fiat_withdrawn_cents", "Treasury fiat withdrawn to bank (cents)", "counter", m.treasury_fiat_withdrawn_cents.load(Relaxed));
    emit!("yield_daemon_treasury_offramp_cycles", "Treasury off-ramp cycles completed", "counter", m.treasury_offramp_cycles.load(Relaxed));
    out
}

/// Atomically write a metrics JSON file (write-tmp + rename) for the Python
/// off-ramp service's poller. Best-effort: a failed write is skipped, not fatal.
fn write_metrics_file(path: &std::path::Path, value: &serde_json::Value) {
    let body = match serde_json::to_string_pretty(value) {
        Ok(s) => s,
        Err(_) => return,
    };
    let tmp = path.with_file_name(format!(
        "{}.tmp",
        path.file_name()
            .map(|f| f.to_string_lossy().into_owned())
            .unwrap_or_default()
    ));
    if std::fs::write(&tmp, body).is_ok() {
        let _ = std::fs::rename(&tmp, path);
    }
}
