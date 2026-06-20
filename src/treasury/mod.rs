//! Treasury Module — autonomous profit accumulation and extraction pipeline.
//!
//! The treasury is the bridge between on-chain yield and fiat bank deposits.
//! It operates a 5-level pipeline:
//!
//! 1. **Accumulate** — collect profits from ZK/MEV/ML modules in native tokens
//! 2. **Consolidate** — swap volatile assets to stablecoins (USDC/USDT)
//! 3. **Threshold Gate** — trigger off-ramp only when balance > configurable threshold
//! 4. **Off-Ramp** — transfer stablecoins to exchange (Coinbase/Kraken) via API
//! 5. **Withdraw** — initiate fiat ACH/SEPA transfer to linked bank account
//!
//! The entire pipeline is autonomous once configured — no human interaction
//! required between profit generation and bank deposit.

pub mod offramp;
pub mod vault;
pub mod keeper;

use crate::DaemonState;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use tracing::{info, warn, error, debug};

/// Treasury configuration.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct TreasuryConfig {
    /// Enable the treasury pipeline
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Minimum profit (in lamports/wei) before triggering consolidation
    #[serde(default = "default_consolidation_threshold")]
    pub consolidation_threshold: u64,

    /// Target stablecoin for consolidation (USDC, USDT, DAI)
    #[serde(default = "default_stablecoin")]
    pub target_stablecoin: String,

    /// Stablecoin mint address (Solana) or contract address (EVM)
    #[serde(default)]
    pub stablecoin_mint: Option<String>,

    /// Minimum stablecoin balance before triggering off-ramp
    #[serde(default = "default_offramp_threshold")]
    pub offramp_threshold_usd: f64,

    /// Maximum single off-ramp amount (USD) — limits exposure
    #[serde(default = "default_max_offramp")]
    pub max_offramp_usd: f64,

    /// Exchange for off-ramp (coinbase, kraken)
    #[serde(default = "default_exchange")]
    pub exchange: String,

    /// Exchange API key (loaded from secrets)
    #[serde(default)]
    pub exchange_api_key: Option<String>,

    /// Exchange API secret (loaded from secrets)
    #[serde(default)]
    pub exchange_api_secret: Option<String>,

    /// Exchange API passphrase (Coinbase Pro)
    #[serde(default)]
    pub exchange_api_passphrase: Option<String>,

    /// Bank account ID on the exchange (for fiat withdrawal)
    #[serde(default)]
    pub bank_account_id: Option<String>,

    /// Fiat currency for withdrawal (USD, EUR, GBP)
    #[serde(default = "default_fiat_currency")]
    pub fiat_currency: String,

    /// Treasury check interval (seconds)
    #[serde(default = "default_treasury_interval")]
    pub check_interval_secs: u64,

    /// Wallet keypair path (Solana JSON keypair file)
    #[serde(default)]
    pub wallet_keypair_path: Option<String>,

    /// EVM private key (hex, no 0x prefix)
    #[serde(default)]
    pub evm_private_key: Option<String>,

    /// Profit vault contract address (EVM chains)
    #[serde(default)]
    pub vault_contract: Option<String>,

    /// DEX router for stablecoin swaps (Jupiter on Solana, Uniswap on EVM)
    #[serde(default)]
    pub swap_router: Option<String>,

    /// Slippage tolerance for consolidation swaps (basis points)
    #[serde(default = "default_slippage_bps")]
    pub slippage_bps: u32,

    /// Dry run mode — log everything but don't execute real transfers
    #[serde(default = "default_true")]
    pub dry_run: bool,
}

fn default_true() -> bool { true }
fn default_consolidation_threshold() -> u64 { 50_000_000 } // 0.05 SOL
fn default_stablecoin() -> String { "USDC".to_string() }
fn default_offramp_threshold() -> f64 { 100.0 }
fn default_max_offramp() -> f64 { 10_000.0 }
fn default_exchange() -> String { "coinbase".to_string() }
fn default_fiat_currency() -> String { "USD".to_string() }
fn default_treasury_interval() -> u64 { 300 } // 5 minutes
fn default_slippage_bps() -> u32 { 50 } // 0.5%

impl Default for TreasuryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            consolidation_threshold: default_consolidation_threshold(),
            target_stablecoin: default_stablecoin(),
            stablecoin_mint: None,
            offramp_threshold_usd: default_offramp_threshold(),
            max_offramp_usd: default_max_offramp(),
            exchange: default_exchange(),
            exchange_api_key: None,
            exchange_api_secret: None,
            exchange_api_passphrase: None,
            bank_account_id: None,
            fiat_currency: default_fiat_currency(),
            check_interval_secs: default_treasury_interval(),
            wallet_keypair_path: None,
            evm_private_key: None,
            vault_contract: None,
            swap_router: None,
            slippage_bps: default_slippage_bps(),
            dry_run: true,
        }
    }
}

/// Treasury runtime state — tracks profit accumulation and off-ramp status.
pub struct TreasuryState {
    /// Total profit accumulated (lamports/wei) across all domains
    pub total_profit_accumulated: std::sync::atomic::AtomicU64,
    /// Total stablecoin balance (micro-USD, 6 decimals)
    pub stablecoin_balance_micro: std::sync::atomic::AtomicU64,
    /// Total fiat withdrawn to bank (cents)
    pub total_fiat_withdrawn_cents: std::sync::atomic::AtomicU64,
    /// Number of successful off-ramp cycles
    pub offramp_cycles: std::sync::atomic::AtomicU64,
    /// Number of consolidation swaps
    pub consolidation_swaps: std::sync::atomic::AtomicU64,
    /// Last off-ramp timestamp (unix epoch seconds)
    pub last_offramp_ts: std::sync::atomic::AtomicU64,
    /// Number of failed off-ramp attempts
    pub offramp_failures: std::sync::atomic::AtomicU64,
}

impl TreasuryState {
    pub fn new() -> Self {
        use std::sync::atomic::AtomicU64;
        Self {
            total_profit_accumulated: AtomicU64::new(0),
            stablecoin_balance_micro: AtomicU64::new(0),
            total_fiat_withdrawn_cents: AtomicU64::new(0),
            offramp_cycles: AtomicU64::new(0),
            consolidation_swaps: AtomicU64::new(0),
            last_offramp_ts: AtomicU64::new(0),
            offramp_failures: AtomicU64::new(0),
        }
    }
}

/// Run the treasury pipeline as an async loop.
pub async fn run(state: Arc<DaemonState>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let config = &state.config.treasury;
    if !config.enabled {
        info!("[Treasury] Disabled, skipping");
        return Ok(());
    }

    info!("[Treasury] Pipeline starting");
    info!("[Treasury] Consolidation threshold: {} lamports", config.consolidation_threshold);
    info!("[Treasury] Off-ramp threshold: ${:.2}", config.offramp_threshold_usd);
    info!("[Treasury] Exchange: {}", config.exchange);
    info!("[Treasury] Fiat currency: {}", config.fiat_currency);
    info!("[Treasury] Mode: {}", if config.dry_run { "DRY RUN" } else { "LIVE" });

    let treasury_state = TreasuryState::new();

    // Build the off-ramp client
    let offramp_client = offramp::OfframpClient::new(config)?;

    // Build the keeper (consolidation + threshold logic)
    let keeper = keeper::TreasuryKeeper::new(config);

    let _check_interval = std::time::Duration::from_secs(config.check_interval_secs);

    loop {
        if !state.running.load(Ordering::Relaxed) {
            break;
        }

        // Step 1: Read current profit metrics from daemon
        let zk_rev = state.metrics.zk_revenue_sat.load(Ordering::Relaxed);
        let mev_rev = state.metrics.mev_revenue_sat.load(Ordering::Relaxed);
        let ml_rev = state.metrics.ml_revenue_sat.load(Ordering::Relaxed);
        let total_rev = zk_rev + mev_rev + ml_rev;

        treasury_state.total_profit_accumulated.store(total_rev, Ordering::Relaxed);

        debug!(
            "[Treasury] Revenue — ZK:{} MEV:{} ML:{} Total:{}",
            zk_rev, mev_rev, ml_rev, total_rev
        );

        // Step 2: Check if consolidation threshold met
        if total_rev >= config.consolidation_threshold {
            match keeper.consolidate_to_stablecoin(
                &state,
                &treasury_state,
                &offramp_client,
            ).await {
                Ok(amount) => {
                    if amount > 0 {
                        treasury_state.consolidation_swaps.fetch_add(1, Ordering::Relaxed);
                        info!("[Treasury] Consolidated {} to {}", amount, config.target_stablecoin);
                    }
                }
                Err(e) => warn!("[Treasury] Consolidation failed: {}", e),
            }
        }

        // Step 3: Check if off-ramp threshold met
        let stable_balance = treasury_state.stablecoin_balance_micro.load(Ordering::Relaxed);
        let stable_usd = stable_balance as f64 / 1_000_000.0;

        if stable_usd >= config.offramp_threshold_usd {
            let withdraw_amount = stable_usd.min(config.max_offramp_usd);

            match offramp_client.execute_offramp(withdraw_amount, config).await {
                Ok(tx_id) => {
                    treasury_state.offramp_cycles.fetch_add(1, Ordering::Relaxed);
                    let cents = (withdraw_amount * 100.0) as u64;
                    treasury_state.total_fiat_withdrawn_cents.fetch_add(cents, Ordering::Relaxed);
                    treasury_state.last_offramp_ts.store(
                        std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs(),
                        Ordering::Relaxed,
                    );
                    info!(
                        "[Treasury] Off-ramp complete: ${:.2} {} -> bank | tx: {}",
                        withdraw_amount, config.fiat_currency, tx_id
                    );
                }
                Err(e) => {
                    treasury_state.offramp_failures.fetch_add(1, Ordering::Relaxed);
                    error!("[Treasury] Off-ramp failed: {}", e);
                }
            }
        }

        // Sleep in 1s steps for responsive shutdown
        for _ in 0..config.check_interval_secs {
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            if !state.running.load(Ordering::Relaxed) {
                break;
            }
        }
    }

    let total_withdrawn = treasury_state.total_fiat_withdrawn_cents.load(Ordering::Relaxed);
    info!(
        "[Treasury] Shutdown — total withdrawn: ${:.2} across {} cycles",
        total_withdrawn as f64 / 100.0,
        treasury_state.offramp_cycles.load(Ordering::Relaxed)
    );
    Ok(())
}
