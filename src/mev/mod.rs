pub mod mempool;
pub mod router;
pub mod amm;
pub mod bundle;
pub mod solver;
pub mod state_shadow;
pub mod graph;
pub mod simd_solver;
pub mod aste;

use crate::DaemonState;
use crate::memory::arena::Arena;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use tracing::{info, warn, error, debug};
use tokio::sync::mpsc;

/// Run the MEV module event loop.
/// Monitors mempool for arbitrage opportunities, computes optimal routes,
/// constructs transaction bundles, and submits via Jito/block engine.
pub async fn run(state: Arc<DaemonState>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let config = &state.config.mev;
    info!("[MEV] Module initialized on {}", config.chain);
    info!("[MEV] Arena size: {}MB", config.arena_size_bytes / (1024 * 1024));
    info!("[MEV] Max latency budget: {}us", config.max_latency_us);
    info!("[MEV] Min profit threshold: {} lamports", config.min_profit_threshold);

    // Pre-allocate arena for zero-alloc hot path
    let arena = Arena::new(config.arena_size_bytes);

    // Build AMM pool registry
    let mut pool_registry = amm::PoolRegistry::new();

    // Mempool transaction channel
    let (tx_sender, mut tx_receiver) = mpsc::channel::<crate::net::p2p::MempoolTransaction>(10_000);

    // Start mempool subscriber
    let subscriber = crate::net::p2p::MempoolSubscriber::new(
        config.ws_endpoints.clone(),
        config.amm_programs.clone(),
        tx_sender,
    );
    subscriber.run().await?;

    // Build route optimizer
    let optimizer = router::RouteOptimizer::new(
        config.min_profit_threshold,
        config.max_latency_us,
    );

    // Build bundle constructor
    let bundler = bundle::BundleConstructor::new(state.config.general.dry_run);

    // ── ASTE Integration ────────────────────────────────────────────
    // The ASTE runs alongside the legacy pipeline. Events are forked:
    // mempool → legacy pipeline + ASTE event channel
    let (aste_tx, aste_rx) = mpsc::channel::<aste::MempoolEvent>(50_000);
    let aste_config = aste::AsteConfig {
        max_hops: 4,
        min_profit: config.min_profit_threshold as u128,
        max_input: 100_000_000_000,
        arena_size: 16 * 1024 * 1024,
        graph_rebuild_interval: 50,
        max_cycles_per_sec: 10_000,
        dry_run: state.config.general.dry_run,
        latency_budget_us: config.max_latency_us,
    };
    let aste_state = state.clone();
    tokio::spawn(async move {
        info!("[MEV] ASTE (Atomic State Transition Engine) starting");
        if let Err(e) = aste::run_aste(aste_state, aste_rx, aste_config).await {
            error!("[MEV] ASTE error: {}", e);
        }
    });

    info!("[MEV] Event loop starting");

    while state.running.load(Ordering::Relaxed) {
        // Poll for mempool transactions (non-blocking)
        match tx_receiver.try_recv() {
            Ok(tx) => {
                let start = std::time::Instant::now();

                // Phase 1: Parse transaction and identify AMM interaction
                if let Some(swap) = amm::parse_swap_instruction(&tx) {
                    debug!("[MEV] Swap detected: {:?}", swap);

                    // Forward to ASTE as a speculative event
                    let _ = aste_tx.try_send(aste::MempoolEvent::PendingSwap {
                        pool_idx: 0, // Resolved by ASTE via pool key lookup
                        amount_in: swap.amount_in,
                        x_to_y: true,
                        tx_signature: [0u8; 64],
                    });

                    // Phase 2: Check all pools for arbitrage opportunity
                    if let Some(route) = optimizer.find_arbitrage(&pool_registry, &swap) {
                        state.metrics.mev_opportunities_detected.fetch_add(1, Ordering::Relaxed);

                        // Phase 3: Compute optimal input amount
                        let optimal = optimizer.optimize_input(&route);

                        // Phase 4: Construct and submit bundle
                        if optimal.expected_profit >= config.min_profit_threshold {
                            let bundle = bundler.construct(&route, &optimal);
                            if state.config.general.dry_run {
                                debug!("[MEV] DRY RUN: would submit bundle for {} profit",
                                    optimal.expected_profit);
                            } else {
                                match bundler.submit(&bundle).await {
                                    Ok(sig) => {
                                        state.metrics.mev_bundles_submitted.fetch_add(1, Ordering::Relaxed);
                                        state.metrics.mev_revenue_sat.fetch_add(
                                            optimal.expected_profit, Ordering::Relaxed);
                                        info!("[MEV] Bundle submitted: {} | profit: {}",
                                            sig, optimal.expected_profit);
                                    }
                                    Err(e) => warn!("[MEV] Bundle submit failed: {}", e),
                                }
                            }
                        }
                    }
                }

                let elapsed = start.elapsed();
                if elapsed.as_micros() > config.max_latency_us as u128 {
                    warn!("[MEV] Latency budget exceeded: {}us", elapsed.as_micros());
                }

                // Reset arena for next cycle
                arena.reset();
            }
            Err(mpsc::error::TryRecvError::Empty) => {
                // No pending transactions — brief yield to avoid busy-wait
                tokio::time::sleep(std::time::Duration::from_millis(1)).await;
            }
            Err(mpsc::error::TryRecvError::Disconnected) => {
                warn!("[MEV] Mempool channel disconnected");
                break;
            }
        }

        // Periodic pool state refresh (every N cycles)
        // In production: subscribe to account changes via geyser
    }

    info!("[MEV] Module shutdown");
    Ok(())
}
