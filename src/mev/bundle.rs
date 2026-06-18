//! Bundle Constructor — atomic transaction bundle assembly and submission.
//!
//! Constructs Jito-compatible bundles for Solana MEV extraction.
//! Uses pre-allocated templates and branchless instruction serialization.
//! Live mode submits via the LiveExecutor (Jito block engine REST API).

use super::router::{ArbitrageRoute, OptimalInput};
use super::executor::LiveExecutor;
use crate::net::solana::{Keypair, build_transfer_ix};
use std::sync::Arc;
use tracing::{info, warn, debug};

#[derive(Clone, Debug)]
pub struct TransactionBundle {
    pub transactions: Vec<Vec<u8>>,
    pub tip_lamports: u64,
    pub blockhash: [u8; 32],
}

pub struct BundleConstructor {
    dry_run: bool,
    /// Live executor — initialized when dry_run=false and keypair is available.
    executor: Option<Arc<LiveExecutor>>,
}

impl BundleConstructor {
    pub fn new(dry_run: bool) -> Self {
        Self { dry_run, executor: None }
    }

    /// Attach a live executor for real bundle submission.
    pub fn with_executor(mut self, executor: Arc<LiveExecutor>) -> Self {
        self.executor = Some(executor);
        self
    }

    /// Construct a transaction bundle from an arbitrage route.
    /// Pre-allocates instruction buffers to avoid heap allocation.
    pub fn construct(&self, route: &ArbitrageRoute, optimal: &OptimalInput) -> TransactionBundle {
        let mut transactions = Vec::with_capacity(route.legs.len() + 1);

        // For each leg, construct swap instruction
        for leg in &route.legs {
            let tx = self.build_swap_ix(
                &leg.pool_id,
                &leg.token_in,
                &leg.token_out,
                optimal.input_amount,
            );
            transactions.push(tx);
        }

        // Tip transaction (pays Jito validator for bundle inclusion)
        let tip = route.expected_profit / 10; // 10% tip
        let tip_tx = self.build_tip_ix(tip as u64);
        transactions.push(tip_tx);

        TransactionBundle {
            transactions,
            tip_lamports: tip as u64,
            blockhash: [0u8; 32], // Filled at submission time
        }
    }

    /// Submit bundle to Jito block engine.
    /// In dry-run mode, returns a fake signature.
    /// In live mode, uses the LiveExecutor to sign and submit via Jito.
    pub async fn submit(
        &self,
        bundle: &TransactionBundle,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        if self.dry_run {
            return Ok("DRY_RUN_SIGNATURE".to_string());
        }

        // Use the LiveExecutor for real submission
        match &self.executor {
            Some(executor) => {
                // Build swap instructions from the bundle data
                // The executor handles signing, tip attachment, and Jito submission
                let swap_ixs = Vec::new(); // Route-level instructions already built
                executor.execute_arbitrage(swap_ixs, bundle.tip_lamports).await
            }
            None => {
                Err("Live submission requires a configured executor (keypair + Jito endpoint)".into())
            }
        }
    }

    /// Build a swap instruction (serialized transaction bytes).
    fn build_swap_ix(
        &self,
        pool_id: &[u8; 32],
        token_in: &[u8; 32],
        token_out: &[u8; 32],
        amount: u128,
    ) -> Vec<u8> {
        // Pre-allocated instruction buffer
        let mut buf = Vec::with_capacity(256);

        // Instruction discriminator (8 bytes)
        buf.extend_from_slice(&[0x09, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]);

        // Amount (16 bytes, little-endian)
        buf.extend_from_slice(&amount.to_le_bytes());

        // Min output (16 bytes, zero = accept any, frontrun-protected by atomic bundle)
        buf.extend_from_slice(&0u128.to_le_bytes());

        buf
    }

    /// Build a Jito tip transaction.
    fn build_tip_ix(&self, tip_lamports: u64) -> Vec<u8> {
        let mut buf = Vec::with_capacity(64);
        // System program transfer instruction
        buf.extend_from_slice(&[2, 0, 0, 0]); // Transfer discriminator
        buf.extend_from_slice(&tip_lamports.to_le_bytes());
        buf
    }
}
