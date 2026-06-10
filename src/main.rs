//! Yield Daemon — Autonomous Yield Generation Infrastructure
//!
//! Headless daemon that converts compute into decentralized yield across:
//! 1. ZK Prover Networks (Succinct, Gevulot) — cryptographic proof generation
//! 2. MEV Arbitrage / Intent Solving — sub-millisecond on-chain extraction
//! 3. Decentralized ML Subnets (Bittensor) — distributed AI inference

use yield_daemon::{config, zk_prover, mev, ml_subnet, DaemonState, DaemonMetrics};

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

    info!("All modules spawned. Daemon running.");

    for h in handles {
        h.await?;
    }

    info!("Yield Daemon shutdown complete");
    Ok(())
}
