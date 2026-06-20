//! Solana Live Execution — keypair loading, transaction signing, and submission.
//!
//! This module bridges the ASTE engine to the real Solana blockchain.
//! It handles:
//! 1. Loading ed25519 keypairs from JSON files (Solana CLI format)
//! 2. Building and signing Solana transactions
//! 3. Submitting transactions via RPC (sendTransaction)
//! 4. Submitting bundles via Jito block engine REST API
//!
//! No `solana-sdk` dependency — we implement the wire format directly
//! for minimal binary size and maximum control over the hot path.

use ed25519_dalek::{SigningKey, Signer, VerifyingKey};
use std::path::Path;
use tracing::{info, warn};

// ── Ed25519 Keypair ─────────────────────────────────────────────────────────

/// A Solana keypair (ed25519). The first 32 bytes are the secret key,
/// the last 32 bytes are the public key. This matches the Solana CLI
/// JSON keypair format: a 64-element byte array.
#[derive(Clone)]
pub struct Keypair {
    signing_key: SigningKey,
    pub pubkey: [u8; 32],
}

impl Keypair {
    /// Load from a Solana CLI JSON keypair file.
    /// Format: JSON array of 64 u8 values.
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let data = std::fs::read_to_string(path)?;
        let bytes: Vec<u8> = serde_json::from_str(&data)?;
        if bytes.len() != 64 {
            return Err(format!("Keypair must be 64 bytes, got {}", bytes.len()).into());
        }
        let mut secret = [0u8; 32];
        let mut pubkey = [0u8; 32];
        secret.copy_from_slice(&bytes[..32]);
        pubkey.copy_from_slice(&bytes[32..]);

        let signing_key = SigningKey::from_bytes(&secret);

        // Verify the pubkey matches what ed25519-dalek derives
        let derived_pubkey = VerifyingKey::from(&signing_key);
        if derived_pubkey.as_bytes() != &pubkey {
            warn!("[Keypair] Derived pubkey differs from stored pubkey — using derived");
            pubkey.copy_from_slice(derived_pubkey.as_bytes());
        }

        Ok(Self { signing_key, pubkey })
    }

    /// Sign a message using ed25519-dalek (constant-time, production-ready).
    /// Returns 64-byte signature.
    pub fn sign(&self, message: &[u8]) -> [u8; 64] {
        let signature = self.signing_key.sign(message);
        signature.to_bytes()
    }

    /// Get the base58-encoded public key.
    pub fn pubkey_base58(&self) -> String {
        bs58_encode(&self.pubkey)
    }
}

// ── Solana Transaction Format ───────────────────────────────────────────────

/// A compact Solana transaction ready for signing.
pub struct SolanaTransaction {
    /// Recent blockhash (32 bytes)
    pub recent_blockhash: [u8; 32],
    /// Instructions
    pub instructions: Vec<SolanaInstruction>,
    /// Account keys (ordered: signers first, then read-only)
    pub account_keys: Vec<[u8; 32]>,
    /// Number of required signers
    pub num_signers: u8,
    /// Number of read-only signed accounts
    pub num_readonly_signed: u8,
    /// Number of read-only unsigned accounts
    pub num_readonly_unsigned: u8,
}

pub struct SolanaInstruction {
    pub program_id_idx: u8,
    pub account_indices: Vec<u8>,
    pub data: Vec<u8>,
}

impl SolanaTransaction {
    /// Serialize the transaction message (the part that gets signed).
    pub fn serialize_message(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(512);

        // Header: [num_signers, num_readonly_signed, num_readonly_unsigned]
        buf.push(self.num_signers);
        buf.push(self.num_readonly_signed);
        buf.push(self.num_readonly_unsigned);

        // Account keys (compact-u16 length prefix + 32 bytes each)
        compact_u16_encode(self.account_keys.len() as u16, &mut buf);
        for key in &self.account_keys {
            buf.extend_from_slice(key);
        }

        // Recent blockhash
        buf.extend_from_slice(&self.recent_blockhash);

        // Instructions (compact-u16 length prefix)
        compact_u16_encode(self.instructions.len() as u16, &mut buf);
        for ix in &self.instructions {
            buf.push(ix.program_id_idx);
            // Account indices
            compact_u16_encode(ix.account_indices.len() as u16, &mut buf);
            buf.extend_from_slice(&ix.account_indices);
            // Data
            compact_u16_encode(ix.data.len() as u16, &mut buf);
            buf.extend_from_slice(&ix.data);
        }

        buf
    }

    /// Sign and serialize the full transaction (signatures + message).
    pub fn sign_and_serialize(&self, keypair: &Keypair) -> Vec<u8> {
        let message = self.serialize_message();
        let signature = keypair.sign(&message);

        let mut buf = Vec::with_capacity(1 + 64 + message.len());
        // Signature count
        compact_u16_encode(1, &mut buf);
        // Signature
        buf.extend_from_slice(&signature);
        // Message
        buf.extend_from_slice(&message);

        buf
    }
}

/// Build a SOL transfer instruction (System Program Transfer).
pub fn build_transfer_ix(
    from: &[u8; 32],
    to: &[u8; 32],
    lamports: u64,
) -> (Vec<[u8; 32]>, SolanaInstruction) {
    let system_program = [0u8; 32]; // 11111111111111111111111111111111

    let accounts = vec![*from, *to, system_program];

    let mut data = Vec::with_capacity(12);
    // Transfer instruction index = 2
    data.extend_from_slice(&2u32.to_le_bytes());
    data.extend_from_slice(&lamports.to_le_bytes());

    let ix = SolanaInstruction {
        program_id_idx: 2, // system_program is at index 2
        account_indices: vec![0, 1],
        data,
    };

    (accounts, ix)
}

// ── Jito Bundle Submission ──────────────────────────────────────────────────

/// Submit a bundle of signed transactions to the Jito block engine.
pub struct JitoClient {
    http: reqwest::Client,
    block_engine_url: String,
    /// Tip account (one of Jito's tip program PDAs)
    tip_accounts: Vec<String>,
}

impl JitoClient {
    pub fn new(block_engine_url: &str) -> Self {
        // Jito tip accounts (mainnet)
        let tip_accounts = vec![
            "96gYZGLnJYVFmbjzopPSU6QiEV5fGqZNyN9nmNhvrZU5".to_string(),
            "HFqU5x63VTqvQss8hp11i4bPUWm8rSSXMuAs7mEjn5g9".to_string(),
            "Cw8CFyM9FkoMi7K7Crf6HNQqf4uEMzpKw6QNghXLvLkY".to_string(),
            "ADaUMid9yfUytqMBgopwjb2DTLSx5NTed13Bd4bkJsTD".to_string(),
            "DfXygSm4jCyNCybVYYK6DwvWqjKee8pbDmJGcLWNDXjh".to_string(),
            "ADuUkR4vqLUMWXxW9gh6D6L8pMSawimctcNZ5pGwDcEt".to_string(),
            "DttWaMuVvTiduZRnguLF7jNxTgiMBZ1hyAumKUiL2KRL".to_string(),
            "3AVi9Tg9Uo68tJfuvoKvqKNWKkC5wPdSSdeBnizKZ6jT".to_string(),
        ];

        Self {
            http: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(5))
                .build()
                .expect("HTTP client"),
            block_engine_url: block_engine_url.to_string(),
            tip_accounts,
        }
    }

    /// Get a random tip account for bundle inclusion.
    pub fn get_tip_account(&self) -> &str {
        let idx = (std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos() as usize)
            % self.tip_accounts.len();
        &self.tip_accounts[idx]
    }

    /// Submit a bundle of base64-encoded signed transactions.
    pub async fn send_bundle(
        &self,
        transactions: &[Vec<u8>],
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        // Encode transactions as base58
        let encoded_txs: Vec<String> = transactions.iter()
            .map(|tx| bs58_encode(tx))
            .collect();

        let body = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "sendBundle",
            "params": [encoded_txs]
        });

        let url = format!("{}/api/v1/bundles", self.block_engine_url);
        let resp = self.http.post(&url)
            .json(&body)
            .send()
            .await?;

        let status = resp.status();
        let text = resp.text().await?;

        if !status.is_success() {
            return Err(format!("Jito bundle error {}: {}", status, text).into());
        }

        let json: serde_json::Value = serde_json::from_str(&text)?;
        let bundle_id = json["result"]
            .as_str()
            .unwrap_or("unknown")
            .to_string();

        info!("[Jito] Bundle submitted: {}", bundle_id);
        Ok(bundle_id)
    }

    /// Get the bundle status.
    pub async fn get_bundle_status(
        &self,
        bundle_id: &str,
    ) -> Result<serde_json::Value, Box<dyn std::error::Error + Send + Sync>> {
        let body = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getBundleStatuses",
            "params": [[bundle_id]]
        });

        let url = format!("{}/api/v1/bundles", self.block_engine_url);
        let resp: serde_json::Value = self.http.post(&url)
            .json(&body)
            .send()
            .await?
            .json()
            .await?;

        Ok(resp)
    }

    /// Get tip floor (minimum tip for bundle inclusion).
    pub async fn get_tip_floor(&self) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
        let url = format!("{}/api/v1/bundles/tip_floor", self.block_engine_url);
        let resp: serde_json::Value = self.http.get(&url)
            .send()
            .await?
            .json()
            .await?;

        // Returns array of tip percentiles
        let floor = resp.as_array()
            .and_then(|arr| arr.first())
            .and_then(|v| v.get("landed_tips_50th_percentile"))
            .and_then(|v| v.as_f64())
            .unwrap_or(1000.0); // Default 1000 lamports

        // Jito returns tip values already in lamports (u64-compatible).
        // landed_tips_50th_percentile is reported in SOL as a float,
        // so multiply by 1e9 to convert SOL → lamports.
        Ok((floor * 1e9) as u64)
    }
}

// ── RPC Extensions for Live Operation ───────────────────────────────────────

/// Extended RPC calls needed for live transaction submission.
pub struct LiveRpc {
    client: reqwest::Client,
    endpoint: String,
}

impl LiveRpc {
    pub fn new(endpoint: &str) -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(10))
                .build()
                .expect("HTTP client"),
            endpoint: endpoint.to_string(),
        }
    }

    /// Get the RPC endpoint URL.
    pub fn url(&self) -> &str {
        &self.endpoint
    }

    /// Get latest blockhash for transaction signing.
    pub async fn get_latest_blockhash(&self) -> Result<[u8; 32], Box<dyn std::error::Error + Send + Sync>> {
        let body = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getLatestBlockhash",
            "params": [{"commitment": "confirmed"}]
        });

        let resp: serde_json::Value = self.client.post(&self.endpoint)
            .json(&body)
            .send().await?
            .json().await?;

        let blockhash_str = resp["result"]["value"]["blockhash"]
            .as_str()
            .ok_or("No blockhash in response")?;

        let decoded = bs58_decode(blockhash_str)?;
        if decoded.len() != 32 {
            return Err("Invalid blockhash length".into());
        }
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&decoded);
        Ok(hash)
    }

    /// Send a signed transaction (base58-encoded).
    pub async fn send_transaction(
        &self,
        signed_tx: &[u8],
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let encoded = bs58_encode(signed_tx);

        let body = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "sendTransaction",
            "params": [
                encoded,
                {
                    "encoding": "base58",
                    "skipPreflight": true,
                    "maxRetries": 2,
                    "preflightCommitment": "confirmed"
                }
            ]
        });

        let resp: serde_json::Value = self.client.post(&self.endpoint)
            .json(&body)
            .send().await?
            .json().await?;

        if let Some(err) = resp.get("error") {
            return Err(format!("sendTransaction error: {}", err).into());
        }

        let sig = resp["result"]
            .as_str()
            .unwrap_or("unknown")
            .to_string();

        Ok(sig)
    }

    /// Confirm a transaction has landed.
    pub async fn confirm_transaction(
        &self,
        signature: &str,
        timeout_secs: u64,
    ) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(timeout_secs);

        while std::time::Instant::now() < deadline {
            let body = serde_json::json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getSignatureStatuses",
                "params": [[signature], {"searchTransactionHistory": false}]
            });

            let resp: serde_json::Value = self.client.post(&self.endpoint)
                .json(&body)
                .send().await?
                .json().await?;

            if let Some(status) = resp["result"]["value"]
                .as_array()
                .and_then(|a| a.first())
            {
                if !status.is_null() {
                    let err = status.get("err");
                    if err.is_none() || err.unwrap().is_null() {
                        return Ok(true); // Confirmed with no error
                    } else {
                        return Ok(false); // Transaction failed
                    }
                }
            }

            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        }

        Ok(false) // Timed out
    }

    /// Get minimum balance for rent exemption.
    pub async fn get_min_rent(&self, data_len: usize) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
        let body = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getMinimumBalanceForRentExemption",
            "params": [data_len]
        });

        let resp: serde_json::Value = self.client.post(&self.endpoint)
            .json(&body)
            .send().await?
            .json().await?;

        Ok(resp["result"].as_u64().unwrap_or(0))
    }
}

// ── Base58 ──────────────────────────────────────────────────────────────────

const BASE58_ALPHABET: &[u8] = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

/// Encode bytes as base58 (Bitcoin/Solana variant).
pub fn bs58_encode(data: &[u8]) -> String {
    if data.is_empty() {
        return String::new();
    }

    // Count leading zeros
    let leading_zeros = data.iter().take_while(|&&b| b == 0).count();

    // Convert to base58
    let mut digits: Vec<u8> = Vec::new();
    for &byte in data {
        let mut carry = byte as u32;
        for d in digits.iter_mut() {
            carry += (*d as u32) << 8;
            *d = (carry % 58) as u8;
            carry /= 58;
        }
        while carry > 0 {
            digits.push((carry % 58) as u8);
            carry /= 58;
        }
    }

    let mut result = String::with_capacity(leading_zeros + digits.len());
    for _ in 0..leading_zeros {
        result.push('1');
    }
    for &d in digits.iter().rev() {
        result.push(BASE58_ALPHABET[d as usize] as char);
    }

    result
}

/// Decode base58 string to bytes.
pub fn bs58_decode(s: &str) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
    let mut table = [255u8; 128];
    for (i, &c) in BASE58_ALPHABET.iter().enumerate() {
        table[c as usize] = i as u8;
    }

    let leading_ones = s.bytes().take_while(|&b| b == b'1').count();
    let mut digits: Vec<u8> = Vec::new();

    for byte in s.bytes() {
        if byte >= 128 || table[byte as usize] == 255 {
            return Err(format!("Invalid base58 character: {}", byte as char).into());
        }
        let mut carry = table[byte as usize] as u32;
        for d in digits.iter_mut() {
            carry += (*d as u32) * 58;
            *d = (carry & 0xFF) as u8;
            carry >>= 8;
        }
        while carry > 0 {
            digits.push((carry & 0xFF) as u8);
            carry >>= 8;
        }
    }

    let mut result = Vec::with_capacity(leading_ones + digits.len());
    for _ in 0..leading_ones {
        result.push(0);
    }
    result.extend(digits.into_iter().rev());

    Ok(result)
}

/// Solana compact-u16 encoding (variable-length unsigned integer).
fn compact_u16_encode(value: u16, buf: &mut Vec<u8>) {
    let mut val = value;
    loop {
        let mut elem = (val & 0x7f) as u8;
        val >>= 7;
        if val > 0 {
            elem |= 0x80;
        }
        buf.push(elem);
        if val == 0 {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bs58_roundtrip() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32];
        let encoded = bs58_encode(&data);
        let decoded = bs58_decode(&encoded).unwrap();
        assert_eq!(data, decoded);
    }

    #[test]
    fn test_bs58_leading_zeros() {
        let data = vec![0, 0, 0, 1, 2, 3];
        let encoded = bs58_encode(&data);
        assert!(encoded.starts_with("111"));
        let decoded = bs58_decode(&encoded).unwrap();
        assert_eq!(data, decoded);
    }

    #[test]
    fn test_compact_u16() {
        let mut buf = Vec::new();
        compact_u16_encode(0, &mut buf);
        assert_eq!(buf, vec![0]);

        buf.clear();
        compact_u16_encode(127, &mut buf);
        assert_eq!(buf, vec![127]);

        buf.clear();
        compact_u16_encode(128, &mut buf);
        assert_eq!(buf, vec![0x80, 0x01]);
    }

    #[test]
    fn test_transaction_message_serialization() {
        let tx = SolanaTransaction {
            recent_blockhash: [0xAA; 32],
            instructions: vec![],
            account_keys: vec![[0xBB; 32]],
            num_signers: 1,
            num_readonly_signed: 0,
            num_readonly_unsigned: 0,
        };
        let msg = tx.serialize_message();
        // Header (3 bytes) + compact len (1) + 32 bytes key + 32 bytes blockhash + compact len (1)
        assert_eq!(msg.len(), 3 + 1 + 32 + 32 + 1);
        assert_eq!(msg[0], 1); // num_signers
    }

    #[test]
    fn test_build_transfer() {
        let from = [1u8; 32];
        let to = [2u8; 32];
        let (accounts, ix) = build_transfer_ix(&from, &to, 1_000_000);
        assert_eq!(accounts.len(), 3);
        assert_eq!(ix.data.len(), 12); // 4 bytes discriminator + 8 bytes lamports
    }

    #[test]
    fn test_jito_client_tip_account() {
        let client = JitoClient::new("https://mainnet.block-engine.jito.wtf");
        let tip = client.get_tip_account();
        assert!(!tip.is_empty());
    }
}
