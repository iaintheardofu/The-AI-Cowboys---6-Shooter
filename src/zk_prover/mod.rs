pub mod montgomery;
pub mod ntt;
pub mod msm;
pub mod fields;
pub mod prover;

use crate::DaemonState;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use tracing::{info, warn};

/// Run the ZK Prover module event loop.
/// Polls prover networks for proof requests, evaluates bids,
/// generates proofs using accelerated NTT/MSM, and submits results.
pub async fn run(state: Arc<DaemonState>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    info!("[ZK] Module initialized");
    info!("[ZK] Montgomery IFMA: {}", state.config.zk.montgomery_ifma);
    info!("[ZK] NTT optimization level: {}", state.config.zk.ntt_optimization);
    info!("[ZK] Pippenger window: {}", state.config.zk.pippenger_window);
    info!("[ZK] Max concurrent proofs: {}", state.config.zk.max_concurrent_proofs);

    let mut prover = prover::ZkProverNode::new(&state.config.zk);

    while state.running.load(Ordering::Relaxed) {
        match prover.poll_and_prove().await {
            Ok(result) => {
                if result.proof_generated {
                    state.metrics.zk_proofs_generated.fetch_add(1, Ordering::Relaxed);
                    if result.accepted {
                        state.metrics.zk_proofs_accepted.fetch_add(1, Ordering::Relaxed);
                        state.metrics.zk_revenue_sat.fetch_add(result.reward_sat, Ordering::Relaxed);
                    }
                }
            }
            Err(e) => {
                warn!("[ZK] Cycle error: {}", e);
            }
        }

        // Proof polling interval — adjustable based on network activity
        tokio::time::sleep(std::time::Duration::from_secs(5)).await;
    }

    info!("[ZK] Module shutdown");
    Ok(())
}
