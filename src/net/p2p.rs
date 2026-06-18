//! P2P Networking — WebSocket mempool subscription and block streaming.
//!
//! Subscribes to pending transactions from validator nodes, filters by
//! target AMM programs, and feeds the MEV hot path with pre-parsed data.
//! Live mode uses tokio-tungstenite for real WebSocket connections.

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{info, warn, error, debug};
use futures_util::{StreamExt, SinkExt};

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

            // Build a bloom filter for O(1) program matching
            let mut filter = ProgramFilter::new(programs.len().max(10), 0.001);
            for p in &programs {
                filter.insert(p.as_bytes());
            }

            tokio::spawn(async move {
                loop {
                    match Self::connect_and_subscribe(&ep, &programs, &filter, &sender).await {
                        Ok(()) => info!("[P2P] WebSocket connection closed normally on {}", ep),
                        Err(e) => warn!("[P2P] WebSocket error on {}: {}", ep, e),
                    }
                    // Reconnect with backoff
                    tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                    info!("[P2P] Reconnecting to {}", ep);
                }
            });
        }

        Ok(())
    }

    async fn connect_and_subscribe(
        endpoint: &str,
        programs: &[String],
        filter: &ProgramFilter,
        sender: &mpsc::Sender<MempoolTransaction>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        info!("[P2P] Connecting to {}", endpoint);

        let (ws_stream, _response) = tokio_tungstenite::connect_async(endpoint).await?;
        let (mut write, mut read) = ws_stream.split();

        info!("[P2P] Connected to {}", endpoint);

        // Subscribe to log notifications mentioning our target programs
        let subscribe_msg = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "logsSubscribe",
            "params": [
                {"mentions": programs},
                {"commitment": "processed"}
            ]
        });

        write.send(tungstenite::Message::Text(subscribe_msg.to_string())).await?;
        debug!("[P2P] Subscribed to logs on {}", endpoint);

        // Process incoming messages
        while let Some(msg) = read.next().await {
            match msg {
                Ok(tungstenite::Message::Text(text)) => {
                    let now_ns = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_nanos() as u64;

                    // Parse the RPC notification
                    if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) {
                        // Extract log notification data
                        if let Some(params) = json.get("params") {
                            if let Some(result) = params.get("result") {
                                if let Some(value) = result.get("value") {
                                    let signature = value.get("signature")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("")
                                        .to_string();

                                    // Extract logs and check for AMM program mentions
                                    let logs: Vec<String> = value.get("logs")
                                        .and_then(|v| v.as_array())
                                        .map(|arr| arr.iter()
                                            .filter_map(|v| v.as_str().map(String::from))
                                            .collect())
                                        .unwrap_or_default();

                                    // Extract program IDs from invoke logs
                                    let mut program_ids = Vec::new();
                                    for log in &logs {
                                        if log.starts_with("Program ") && log.contains(" invoke") {
                                            if let Some(pid) = log.strip_prefix("Program ") {
                                                if let Some(pid) = pid.split_whitespace().next() {
                                                    if filter.contains(pid.as_bytes()) {
                                                        program_ids.push(pid.to_string());
                                                    }
                                                }
                                            }
                                        }
                                    }

                                    if !program_ids.is_empty() && !signature.is_empty() {
                                        let tx = MempoolTransaction {
                                            signature,
                                            program_ids,
                                            accounts: Vec::new(), // Populated by getTransaction if needed
                                            data: Vec::new(),
                                            slot: value.get("slot")
                                                .and_then(|v| v.as_u64())
                                                .unwrap_or(0),
                                            received_at_ns: now_ns,
                                        };

                                        if sender.try_send(tx).is_err() {
                                            debug!("[P2P] Channel full, dropping transaction");
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                Ok(tungstenite::Message::Ping(data)) => {
                    let _ = write.send(tungstenite::Message::Pong(data)).await;
                }
                Ok(tungstenite::Message::Close(_)) => {
                    info!("[P2P] Server closed connection on {}", endpoint);
                    break;
                }
                Err(e) => {
                    error!("[P2P] WebSocket read error: {}", e);
                    break;
                }
                _ => {}
            }
        }

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
