pub mod miner;
pub mod inference;
pub mod training;

use crate::DaemonState;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use tracing::{info, warn, error};

/// Run the ML Subnet miner module.
/// Connects to Bittensor (or compatible) network, serves inference
/// requests from validators, and runs background training loops.
pub async fn run(state: Arc<DaemonState>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let config = &state.config.ml;
    info!("[ML] Module initialized");
    info!("[ML] Subnet UID: {}", config.subnet_uid);
    info!("[ML] Precision: {}", config.precision);
    info!("[ML] Batch size: {}", config.batch_size);
    info!("[ML] Background training: {}", config.background_training);

    let miner = miner::SubnetMiner::new(config);

    // Start background training loop if enabled
    if config.background_training {
        let train_state = state.clone();
        let train_config = config.clone();
        tokio::spawn(async move {
            info!("[ML] Background training loop started");
            let trainer = training::BackgroundTrainer::new(&train_config);
            loop {
                if !train_state.running.load(Ordering::Relaxed) {
                    break;
                }
                match trainer.train_step().await {
                    Ok(loss) => {
                        train_state.metrics.ml_training_rounds.fetch_add(1, Ordering::Relaxed);
                        if loss < 0.01 {
                            info!("[ML] Training converged, loss={:.6}", loss);
                        }
                    }
                    Err(e) => warn!("[ML] Training step error: {}", e),
                }
                tokio::time::sleep(std::time::Duration::from_secs(60)).await;
            }
        });
    }

    // Main inference serving loop
    while state.running.load(Ordering::Relaxed) {
        match miner.serve_request().await {
            Ok(result) => {
                if result.served {
                    state.metrics.ml_inferences_served.fetch_add(1, Ordering::Relaxed);
                    state.metrics.ml_revenue_sat.fetch_add(result.reward, Ordering::Relaxed);
                }
            }
            Err(e) => {
                warn!("[ML] Inference error: {}", e);
            }
        }

        // Polling interval for validator requests
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    }

    info!("[ML] Module shutdown");
    Ok(())
}
