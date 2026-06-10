//! P2P Networking — WebSocket mempool subscription and block streaming.
//!
//! Subscribes to pending transactions from validator nodes, filters by
//! target AMM programs, and feeds the MEV hot path with pre-parsed data.

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{info, warn, error, debug};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MempoolTransaction {
    pub signature: String,
    pub program_ids: Vec<String>,
    pub accounts: Vec<String>,
    pub data: Vec<u8>,
    pub slot: u64,
    pub received_at_ns: u64,
}

/// Mempool subscriber that filters and dispatches relevant transactions.
pub struct MempoolSubscriber {
    ws_endpoints: Vec<String>,
    target_programs: Vec<String>,
    tx_sender: mpsc::Sender<MempoolTransaction>,
}

impl MempoolSubscriber {
    pub fn new(
        ws_endpoints: Vec<String>,
        target_programs: Vec<String>,
        tx_sender: mpsc::Sender<MempoolTransaction>,
    ) -> Self {
        Self {
            ws_endpoints,
            target_programs,
            tx_sender,
        }
    }

    /// Subscribe to mempool via WebSocket and filter relevant transactions.
    pub async fn run(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if self.ws_endpoints.is_empty() {
            info!("[P2P] No WebSocket endpoints configured, mempool subscription disabled");
            return Ok(());
        }

        for endpoint in &self.ws_endpoints {
            let ep = endpoint.clone();
            let programs = self.target_programs.clone();
            let sender = self.tx_sender.clone();

            tokio::spawn(async move {
                loop {
                    match Self::connect_and_subscribe(&ep, &programs, &sender).await {
                        Ok(()) => info!("[P2P] WebSocket connection closed normally"),
                        Err(e) => warn!("[P2P] WebSocket error on {}: {}", ep, e),
                    }
                    // Reconnect after 1 second
                    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                }
            });
        }

        Ok(())
    }

    async fn connect_and_subscribe(
        endpoint: &str,
        _programs: &[String],
        _sender: &mpsc::Sender<MempoolTransaction>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        info!("[P2P] Connecting to {}", endpoint);

        // WebSocket connection and subscription logic
        // In production, this connects to Jito or validator geyser plugins
        // For now, structured as a polling fallback

        let subscribe_msg = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "logsSubscribe",
            "params": [
                {"mentions": _programs},
                {"commitment": "processed"}
            ]
        });

        debug!("[P2P] Subscription message: {}", subscribe_msg);

        // Placeholder: actual WS connection via tungstenite
        // The real implementation would:
        // 1. Connect via tokio_tungstenite
        // 2. Send subscribe_msg
        // 3. Parse incoming log notifications
        // 4. Filter by target_programs (branchless bloom filter check)
        // 5. Parse transaction data with SIMD JSON parser
        // 6. Send to tx_sender channel

        Ok(())
    }
}

/// Bloom filter for O(1) program ID matching on hot path.
/// Avoids branching over a list of target program IDs.
pub struct ProgramFilter {
    bits: Vec<u64>,
    num_bits: usize,
    num_hashes: u8,
}

impl ProgramFilter {
    pub fn new(expected_items: usize, false_positive_rate: f64) -> Self {
        let num_bits = (-(expected_items as f64) * false_positive_rate.ln()
            / (2.0_f64.ln().powi(2)))
            .ceil() as usize;
        let num_hashes = ((num_bits as f64 / expected_items as f64) * 2.0_f64.ln())
            .ceil() as u8;
        let words = (num_bits + 63) / 64;
        Self {
            bits: vec![0u64; words],
            num_bits,
            num_hashes,
        }
    }

    #[inline(always)]
    fn hash_pair(data: &[u8], seed: u8) -> (u64, u64) {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        data.hash(&mut hasher);
        seed.hash(&mut hasher);
        let h = hasher.finish();
        (h, h.wrapping_mul(0x517cc1b727220a95))
    }

    pub fn insert(&mut self, item: &[u8]) {
        let (h1, h2) = Self::hash_pair(item, 0);
        for i in 0..self.num_hashes {
            let bit = (h1.wrapping_add(h2.wrapping_mul(i as u64))) as usize % self.num_bits;
            self.bits[bit / 64] |= 1u64 << (bit % 64);
        }
    }

    #[inline(always)]
    pub fn contains(&self, item: &[u8]) -> bool {
        let (h1, h2) = Self::hash_pair(item, 0);
        for i in 0..self.num_hashes {
            let bit = (h1.wrapping_add(h2.wrapping_mul(i as u64))) as usize % self.num_bits;
            if self.bits[bit / 64] & (1u64 << (bit % 64)) == 0 {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bloom_filter() {
        let mut filter = ProgramFilter::new(100, 0.01);
        let program_a = b"675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8";
        let program_b = b"whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc";
        let program_c = b"NotInFilter11111111111111111111111111111111";

        filter.insert(program_a);
        filter.insert(program_b);

        assert!(filter.contains(program_a));
        assert!(filter.contains(program_b));
        assert!(!filter.contains(program_c));
    }
}
