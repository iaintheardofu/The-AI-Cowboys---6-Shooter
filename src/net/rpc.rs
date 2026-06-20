//! RPC Client — low-latency blockchain RPC with connection pooling.
//!
//! Supports Solana JSON-RPC, Ethereum JSON-RPC, and WebSocket subscriptions.
//! Pre-serializes common requests to avoid allocation on hot path.

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, Clone)]
pub struct RpcConfig {
    pub endpoints: Vec<String>,
    pub ws_endpoints: Vec<String>,
    pub timeout_ms: u64,
    pub max_retries: u32,
}

impl Default for RpcConfig {
    fn default() -> Self {
        Self {
            endpoints: vec!["https://api.mainnet-beta.solana.com".to_string()],
            ws_endpoints: vec![],
            timeout_ms: 5000,
            max_retries: 3,
        }
    }
}

/// Round-robin RPC client with latency tracking per endpoint.
pub struct RpcClient {
    config: RpcConfig,
    client: reqwest::Client,
    request_id: AtomicU64,
    endpoint_idx: AtomicU64,
    /// Latency tracking per endpoint (nanoseconds, exponential moving avg)
    endpoint_latencies: Vec<AtomicU64>,
}

#[derive(Serialize)]
struct JsonRpcRequest<'a> {
    jsonrpc: &'static str,
    id: u64,
    method: &'a str,
    params: serde_json::Value,
}

#[derive(Deserialize, Debug)]
pub struct JsonRpcResponse {
    pub id: u64,
    pub result: Option<serde_json::Value>,
    pub error: Option<JsonRpcError>,
}

#[derive(Deserialize, Debug)]
pub struct JsonRpcError {
    pub code: i64,
    pub message: String,
}

impl RpcClient {
    pub fn new(config: RpcConfig) -> Self {
        let n = config.endpoints.len().max(1);
        let client = reqwest::Client::builder()
            .pool_max_idle_per_host(16)
            .tcp_keepalive(Some(std::time::Duration::from_secs(30)))
            .timeout(std::time::Duration::from_millis(config.timeout_ms))
            .build()
            .expect("Failed to build HTTP client");

        Self {
            endpoint_latencies: (0..n).map(|_| AtomicU64::new(0)).collect(),
            config,
            client,
            request_id: AtomicU64::new(1),
            endpoint_idx: AtomicU64::new(0),
        }
    }

    /// Select the lowest-latency endpoint (or round-robin if no data).
    fn select_endpoint(&self) -> &str {
        if self.config.endpoints.is_empty() {
            return "https://api.mainnet-beta.solana.com";
        }

        // Pick lowest latency endpoint
        let mut best_idx = 0;
        let mut best_lat = u64::MAX;
        for (i, lat) in self.endpoint_latencies.iter().enumerate() {
            let l = lat.load(Ordering::Relaxed);
            if l < best_lat {
                best_lat = l;
                best_idx = i;
            }
        }

        // If all zero (cold start), round-robin
        if best_lat == 0 {
            let idx = self.endpoint_idx.fetch_add(1, Ordering::Relaxed) as usize;
            &self.config.endpoints[idx % self.config.endpoints.len()]
        } else {
            &self.config.endpoints[best_idx]
        }
    }

    /// Send a JSON-RPC request with latency tracking.
    pub async fn call(
        &self,
        method: &str,
        params: serde_json::Value,
    ) -> Result<serde_json::Value, Box<dyn std::error::Error + Send + Sync>> {
        let id = self.request_id.fetch_add(1, Ordering::Relaxed);
        let req = JsonRpcRequest {
            jsonrpc: "2.0",
            id,
            method,
            params,
        };

        let endpoint = self.select_endpoint();
        let start = std::time::Instant::now();

        let resp = self.client
            .post(endpoint)
            .json(&req)
            .send()
            .await?
            .json::<JsonRpcResponse>()
            .await?;

        // Update latency EMA
        let latency_ns = start.elapsed().as_nanos() as u64;
        if let Some(idx) = self.config.endpoints.iter().position(|e| e == endpoint) {
            let prev = self.endpoint_latencies[idx].load(Ordering::Relaxed);
            let ema = if prev == 0 { latency_ns } else { (prev * 7 + latency_ns) / 8 };
            self.endpoint_latencies[idx].store(ema, Ordering::Relaxed);
        }

        if let Some(err) = resp.error {
            return Err(format!("RPC error {}: {}", err.code, err.message).into());
        }

        resp.result.ok_or_else(|| "Empty RPC result".into())
    }

    /// Get Solana slot height (fast health check).
    pub async fn get_slot(&self) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
        let result = self.call("getSlot", serde_json::json!([])).await?;
        result.as_u64().ok_or_else(|| "Invalid slot".into())
    }

    /// Get multiple account infos in a single batch (Solana).
    pub async fn get_multiple_accounts(
        &self,
        pubkeys: &[&str],
    ) -> Result<Vec<Option<serde_json::Value>>, Box<dyn std::error::Error + Send + Sync>> {
        let params = serde_json::json!([
            pubkeys,
            {"encoding": "base64", "commitment": "confirmed"}
        ]);
        let result = self.call("getMultipleAccounts", params).await?;
        let accounts = result.get("value")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().map(|v| {
                if v.is_null() { None } else { Some(v.clone()) }
            }).collect())
            .unwrap_or_default();
        Ok(accounts)
    }
}
