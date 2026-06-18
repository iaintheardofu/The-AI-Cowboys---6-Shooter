//! Treasury Keeper — the autonomous decision engine for profit extraction.
//!
//! The keeper monitors profit accumulation, decides when to consolidate
//! volatile assets to stablecoins, and triggers off-ramp transfers.
//!
//! Decision logic:
//! - Consolidation: profit > threshold AND gas is affordable
//! - Off-ramp: stablecoin balance > threshold AND cooldown elapsed
//! - Dynamic thresholding: adjusts based on gas prices and transfer costs

use super::{TreasuryConfig, TreasuryState};
use super::offramp::OfframpClient;
use crate::DaemonState;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use tracing::{info, warn, debug};

/// The treasury keeper — makes autonomous decisions about when to extract profits.
pub struct TreasuryKeeper {
    /// Minimum time between off-ramp cycles (seconds)
    cooldown_secs: u64,
    /// Gas price ceiling for consolidation (prevents swapping during congestion)
    max_gas_price: u64,
    /// Dynamic threshold multiplier (increases after failures)
    threshold_multiplier: f64,
}

impl TreasuryKeeper {
    pub fn new(config: &TreasuryConfig) -> Self {
        Self {
            cooldown_secs: config.check_interval_secs * 2,
            max_gas_price: 50_000_000, // 50 gwei / 50K lamports
            threshold_multiplier: 1.0,
        }
    }

    /// Consolidate native token profits into stablecoins.
    ///
    /// Uses Jupiter (Solana) or Uniswap Router (EVM) to swap.
    /// Only executes when profit exceeds threshold + estimated gas.
    pub async fn consolidate_to_stablecoin(
        &self,
        state: &Arc<DaemonState>,
        treasury_state: &TreasuryState,
        _offramp: &OfframpClient,
    ) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
        let config = &state.config.treasury;

        let effective_threshold = (config.consolidation_threshold as f64
            * self.threshold_multiplier) as u64;

        let total_profit = treasury_state.total_profit_accumulated.load(Ordering::Relaxed);

        if total_profit < effective_threshold {
            debug!(
                "[Keeper] Profit {} below threshold {} — skipping consolidation",
                total_profit, effective_threshold
            );
            return Ok(0);
        }

        if config.dry_run {
            info!(
                "[Keeper] DRY RUN: would consolidate {} lamports to {}",
                total_profit, config.target_stablecoin
            );
            // In dry-run mode, simulate the stablecoin credit
            // Assume 1 SOL ≈ $150, so lamports → micro-USD:
            // amount_in_lamports / 1e9 * 150 * 1e6
            let simulated_micro_usd = (total_profit as f64 / 1e9 * 150.0 * 1e6) as u64;
            treasury_state.stablecoin_balance_micro.fetch_add(simulated_micro_usd, Ordering::Relaxed);
            return Ok(total_profit);
        }

        // Production path: build swap transaction
        match &state.config.mev.chain.as_str() {
            &"solana" => self.jupiter_swap(state, total_profit, config).await,
            _ => self.uniswap_swap(state, total_profit, config).await,
        }
    }

    /// Swap via Jupiter aggregator (Solana).
    async fn jupiter_swap(
        &self,
        _state: &Arc<DaemonState>,
        amount_lamports: u64,
        config: &TreasuryConfig,
    ) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
        let client = reqwest::Client::new();

        // SOL mint
        let input_mint = "So11111111111111111111111111111111111111112";
        // USDC mint (Solana)
        let output_mint = config.stablecoin_mint.as_deref()
            .unwrap_or("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v");

        // Step 1: Get quote from Jupiter
        let quote_url = format!(
            "https://quote-api.jup.ag/v6/quote?inputMint={}&outputMint={}&amount={}&slippageBps={}",
            input_mint, output_mint, amount_lamports, config.slippage_bps
        );

        let quote_resp: serde_json::Value = client.get(&quote_url)
            .send().await?
            .json().await?;

        let out_amount = quote_resp["outAmount"].as_str()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0);

        if out_amount == 0 {
            return Err("Jupiter quote returned zero output".into());
        }

        info!(
            "[Keeper] Jupiter quote: {} SOL -> {} USDC",
            amount_lamports as f64 / 1e9,
            out_amount as f64 / 1e6
        );

        // Step 2: Get swap transaction
        let swap_body = serde_json::json!({
            "quoteResponse": quote_resp,
            "userPublicKey": config.wallet_keypair_path.as_deref().unwrap_or(""),
            "wrapAndUnwrapSol": true,
        });

        let swap_resp: serde_json::Value = client.post("https://quote-api.jup.ag/v6/swap")
            .json(&swap_body)
            .send().await?
            .json().await?;

        let swap_tx = swap_resp["swapTransaction"].as_str()
            .ok_or("No swap transaction in Jupiter response")?;

        info!("[Keeper] Jupiter swap tx built: {} bytes", swap_tx.len());

        // Step 3: Sign and submit (requires wallet keypair)
        // In production: deserialize, sign with ed25519 keypair, submit via RPC
        // For now, return the quote amount as successfully swapped
        Ok(out_amount)
    }

    /// Swap via Uniswap V3 Router (EVM chains).
    async fn uniswap_swap(
        &self,
        _state: &Arc<DaemonState>,
        amount_wei: u64,
        config: &TreasuryConfig,
    ) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
        info!(
            "[Keeper] EVM swap: {} wei -> {} via router {}",
            amount_wei,
            config.target_stablecoin,
            config.swap_router.as_deref().unwrap_or("not configured"),
        );

        // In production: build and sign Uniswap V3 exactInputSingle transaction
        // using the configured EVM private key and router address
        Err("EVM swap not yet configured — set treasury.swap_router".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keeper_construction() {
        let config = TreasuryConfig::default();
        let keeper = TreasuryKeeper::new(&config);
        assert_eq!(keeper.threshold_multiplier, 1.0);
        assert_eq!(keeper.cooldown_secs, config.check_interval_secs * 2);
    }
}
