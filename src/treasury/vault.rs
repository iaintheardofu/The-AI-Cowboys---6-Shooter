//! On-Chain Vault — smart contract interaction for profit accumulation.
//!
//! For Solana: direct SOL/SPL token accounts with program-derived addresses.
//! For EVM chains: ProfitVault.sol contract with owner-only withdrawal.
//!
//! The vault provides:
//! - Atomic profit deposits from arbitrage bundles
//! - Balance queries for threshold checks
//! - Batch withdrawal to exchange deposit address

use tracing::{info, warn, debug};

/// Chain-agnostic vault interface.
pub struct OnChainVault {
    chain: ChainType,
    vault_address: String,
    wallet_address: String,
}

#[derive(Debug, Clone)]
pub enum ChainType {
    Solana,
    Ethereum,
    Arbitrum,
    Base,
    Polygon,
}

impl OnChainVault {
    pub fn new(chain: &str, vault_address: String, wallet_address: String) -> Self {
        let chain = match chain.to_lowercase().as_str() {
            "solana" => ChainType::Solana,
            "ethereum" | "eth" => ChainType::Ethereum,
            "arbitrum" | "arb" => ChainType::Arbitrum,
            "base" => ChainType::Base,
            "polygon" | "matic" => ChainType::Polygon,
            _ => ChainType::Solana,
        };
        Self { chain, vault_address, wallet_address }
    }

    /// Get the vault's current balance in the native token (lamports/wei).
    pub async fn get_native_balance(
        &self,
        rpc_url: &str,
    ) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
        let client = reqwest::Client::new();

        match &self.chain {
            ChainType::Solana => {
                let body = serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getBalance",
                    "params": [&self.vault_address]
                });
                let resp: serde_json::Value = client.post(rpc_url)
                    .json(&body)
                    .send().await?
                    .json().await?;
                Ok(resp["result"]["value"].as_u64().unwrap_or(0))
            }
            _ => {
                // EVM: eth_getBalance
                let body = serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "eth_getBalance",
                    "params": [&self.vault_address, "latest"]
                });
                let resp: serde_json::Value = client.post(rpc_url)
                    .json(&body)
                    .send().await?
                    .json().await?;
                let hex_balance = resp["result"].as_str().unwrap_or("0x0");
                let balance = u64::from_str_radix(hex_balance.trim_start_matches("0x"), 16)
                    .unwrap_or(0);
                Ok(balance)
            }
        }
    }

    /// Get SPL token balance (Solana) or ERC20 balance (EVM).
    pub async fn get_token_balance(
        &self,
        rpc_url: &str,
        token_mint: &str,
    ) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
        let client = reqwest::Client::new();

        match &self.chain {
            ChainType::Solana => {
                let body = serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getTokenAccountsByOwner",
                    "params": [
                        &self.vault_address,
                        {"mint": token_mint},
                        {"encoding": "jsonParsed"}
                    ]
                });
                let resp: serde_json::Value = client.post(rpc_url)
                    .json(&body)
                    .send().await?
                    .json().await?;

                let balance = resp["result"]["value"]
                    .as_array()
                    .and_then(|arr| arr.first())
                    .and_then(|acc| acc["account"]["data"]["parsed"]["info"]["tokenAmount"]["amount"].as_str())
                    .and_then(|s| s.parse::<u64>().ok())
                    .unwrap_or(0);

                Ok(balance)
            }
            _ => {
                // EVM: ERC20 balanceOf(address)
                // Function selector: 0x70a08231
                let padded_addr = format!("0x70a08231000000000000000000000000{}", &self.vault_address[2..]);
                let body = serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "eth_call",
                    "params": [{
                        "to": token_mint,
                        "data": padded_addr,
                    }, "latest"]
                });
                let resp: serde_json::Value = client.post(rpc_url)
                    .json(&body)
                    .send().await?
                    .json().await?;
                let hex_balance = resp["result"].as_str().unwrap_or("0x0");
                let balance = u64::from_str_radix(hex_balance.trim_start_matches("0x"), 16)
                    .unwrap_or(0);
                Ok(balance)
            }
        }
    }

    /// Transfer tokens from vault to a destination (exchange deposit address).
    /// For Solana: builds and signs a system transfer or SPL transfer.
    /// For EVM: builds and signs an ERC20 transfer or ETH transfer.
    pub async fn transfer_to_exchange(
        &self,
        _rpc_url: &str,
        _destination: &str,
        _amount: u64,
        _token_mint: Option<&str>,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        // In production: sign and submit transaction
        // This requires the wallet keypair/private key which is handled
        // by the keeper module using the configured credentials.
        info!(
            "[Vault] Transfer {} to exchange (chain: {:?})",
            _amount, self.chain
        );
        Err("Transfer requires keeper module integration".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vault_construction() {
        let vault = OnChainVault::new(
            "solana",
            "11111111111111111111111111111111".to_string(),
            "22222222222222222222222222222222".to_string(),
        );
        assert!(matches!(vault.chain, ChainType::Solana));
    }

    #[test]
    fn test_vault_evm_chains() {
        for chain in &["ethereum", "arbitrum", "base", "polygon"] {
            let vault = OnChainVault::new(
                chain,
                "0x0000000000000000000000000000000000000001".to_string(),
                "0x0000000000000000000000000000000000000002".to_string(),
            );
            assert!(!matches!(vault.chain, ChainType::Solana));
        }
    }
}
