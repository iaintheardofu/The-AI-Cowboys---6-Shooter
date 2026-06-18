pub mod config;
pub mod memory;
pub mod net;
pub mod zk_prover;
pub mod mev;
pub mod ml_subnet;

use std::sync::atomic::AtomicBool;

/// Global daemon state, shared across all domain modules.
pub struct DaemonState {
    pub config: config::DaemonConfig,
    pub metrics: DaemonMetrics,
    pub running: AtomicBool,
}

/// Aggregate metrics across all yield domains.
pub struct DaemonMetrics {
    pub zk_proofs_generated: std::sync::atomic::AtomicU64,
    pub zk_proofs_accepted: std::sync::atomic::AtomicU64,
    pub zk_revenue_sat: std::sync::atomic::AtomicU64,
    pub mev_opportunities_detected: std::sync::atomic::AtomicU64,
    pub mev_bundles_submitted: std::sync::atomic::AtomicU64,
    pub mev_revenue_sat: std::sync::atomic::AtomicU64,
    pub ml_inferences_served: std::sync::atomic::AtomicU64,
    pub ml_training_rounds: std::sync::atomic::AtomicU64,
    pub ml_revenue_sat: std::sync::atomic::AtomicU64,
    pub total_cycles: std::sync::atomic::AtomicU64,
    pub uptime_secs: std::sync::atomic::AtomicU64,
    // ASTE (Atomic State Transition Engine) metrics
    pub aste_cycles: std::sync::atomic::AtomicU64,
    pub aste_arb_detected: std::sync::atomic::AtomicU64,
    pub aste_arb_profitable: std::sync::atomic::AtomicU64,
    pub aste_latency_ns: std::sync::atomic::AtomicU64,
}

impl DaemonMetrics {
    pub fn new() -> Self {
        use std::sync::atomic::AtomicU64;
        Self {
            zk_proofs_generated: AtomicU64::new(0),
            zk_proofs_accepted: AtomicU64::new(0),
            zk_revenue_sat: AtomicU64::new(0),
            mev_opportunities_detected: AtomicU64::new(0),
            mev_bundles_submitted: AtomicU64::new(0),
            mev_revenue_sat: AtomicU64::new(0),
            ml_inferences_served: AtomicU64::new(0),
            ml_training_rounds: AtomicU64::new(0),
            ml_revenue_sat: AtomicU64::new(0),
            total_cycles: AtomicU64::new(0),
            uptime_secs: AtomicU64::new(0),
            aste_cycles: AtomicU64::new(0),
            aste_arb_detected: AtomicU64::new(0),
            aste_arb_profitable: AtomicU64::new(0),
            aste_latency_ns: AtomicU64::new(0),
        }
    }
}
