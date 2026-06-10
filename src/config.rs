use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct DaemonConfig {
    #[serde(default)]
    pub general: GeneralConfig,
    #[serde(default)]
    pub zk: ZkConfig,
    #[serde(default)]
    pub mev: MevConfig,
    #[serde(default)]
    pub ml: MlConfig,
    #[serde(default)]
    pub risk: RiskConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GeneralConfig {
    /// Dry run mode — no real transactions
    #[serde(default = "default_true")]
    pub dry_run: bool,
    /// State directory for persistence
    #[serde(default = "default_state_dir")]
    pub state_dir: String,
    /// Log level
    #[serde(default = "default_log_level")]
    pub log_level: String,
    /// Metrics export interval (seconds)
    #[serde(default = "default_metrics_interval")]
    pub metrics_interval_secs: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ZkConfig {
    /// Enable ZK prover
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Succinct Network RPC endpoint
    #[serde(default)]
    pub succinct_rpc: Option<String>,
    /// Gevulot endpoint
    #[serde(default)]
    pub gevulot_rpc: Option<String>,
    /// Max concurrent proof jobs
    #[serde(default = "default_zk_concurrency")]
    pub max_concurrent_proofs: usize,
    /// Min bid multiplier (1.0 = break-even)
    #[serde(default = "default_min_bid")]
    pub min_bid_multiplier: f64,
    /// NTT optimization level: 0=scalar, 1=AVX2, 2=AVX-512
    #[serde(default)]
    pub ntt_optimization: u8,
    /// Montgomery domain: use 52-bit limbs for IFMA
    #[serde(default = "default_true")]
    pub montgomery_ifma: bool,
    /// Pippenger window size for MSM
    #[serde(default = "default_pippenger_window")]
    pub pippenger_window: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MevConfig {
    /// Enable MEV extraction
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Blockchain target: solana, ethereum, arbitrum, base
    #[serde(default = "default_chain")]
    pub chain: String,
    /// RPC endpoints (multiple for redundancy)
    #[serde(default)]
    pub rpc_endpoints: Vec<String>,
    /// WebSocket endpoints for mempool streaming
    #[serde(default)]
    pub ws_endpoints: Vec<String>,
    /// Jito block engine URL (Solana MEV)
    #[serde(default)]
    pub jito_block_engine: Option<String>,
    /// Max latency budget (microseconds) for hot path
    #[serde(default = "default_max_latency_us")]
    pub max_latency_us: u64,
    /// AMM programs to monitor
    #[serde(default = "default_amm_programs")]
    pub amm_programs: Vec<String>,
    /// Min profit threshold (lamports/wei)
    #[serde(default = "default_min_profit")]
    pub min_profit_threshold: u64,
    /// Arena allocator pre-allocation (bytes)
    #[serde(default = "default_arena_size")]
    pub arena_size_bytes: usize,
    /// Enable kernel-bypass networking (AF_XDP)
    #[serde(default)]
    pub kernel_bypass: bool,
    /// Intent solver mode (CoW Protocol)
    #[serde(default)]
    pub intent_solver: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MlConfig {
    /// Enable ML subnet mining
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Bittensor network endpoint
    #[serde(default)]
    pub bittensor_endpoint: Option<String>,
    /// Subnet UID to mine
    #[serde(default = "default_subnet_uid")]
    pub subnet_uid: u16,
    /// Wallet name
    #[serde(default = "default_wallet")]
    pub wallet_name: String,
    /// Hotkey name
    #[serde(default = "default_hotkey")]
    pub hotkey_name: String,
    /// Model path for inference
    #[serde(default)]
    pub model_path: Option<String>,
    /// GPU device IDs
    #[serde(default)]
    pub gpu_devices: Vec<u32>,
    /// Batch size for inference
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    /// Enable background training loop
    #[serde(default = "default_true")]
    pub background_training: bool,
    /// Training data directory
    #[serde(default)]
    pub training_data_dir: Option<String>,
    /// FP precision: fp32, fp16, fp8, int4
    #[serde(default = "default_precision")]
    pub precision: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RiskConfig {
    /// Max total capital at risk (fraction)
    #[serde(default = "default_max_capital_risk")]
    pub max_capital_at_risk: f64,
    /// Max loss per cycle (fraction of total capital)
    #[serde(default = "default_max_cycle_loss")]
    pub max_cycle_loss: f64,
    /// Circuit breaker: halt after N consecutive losses
    #[serde(default = "default_circuit_breaker")]
    pub circuit_breaker_threshold: u32,
    /// Daily P&L floor (absolute, in base currency)
    #[serde(default)]
    pub daily_loss_limit: Option<f64>,
    /// Slashing protection: max stake at risk for ZK proofs
    #[serde(default = "default_max_stake")]
    pub max_stake_fraction: f64,
}

// Default value functions
fn default_true() -> bool { true }
fn default_state_dir() -> String { "runtime/yield_daemon".to_string() }
fn default_log_level() -> String { "info".to_string() }
fn default_metrics_interval() -> u64 { 30 }
fn default_zk_concurrency() -> usize { 4 }
fn default_min_bid() -> f64 { 1.15 }
fn default_pippenger_window() -> usize { 15 }
fn default_chain() -> String { "solana".to_string() }
fn default_max_latency_us() -> u64 { 200_000 } // 200ms
fn default_min_profit() -> u64 { 10_000 } // 10K lamports
fn default_arena_size() -> usize { 64 * 1024 * 1024 } // 64MB
fn default_subnet_uid() -> u16 { 1 }
fn default_wallet() -> String { "default".to_string() }
fn default_hotkey() -> String { "default".to_string() }
fn default_batch_size() -> usize { 32 }
fn default_precision() -> String { "fp16".to_string() }
fn default_max_capital_risk() -> f64 { 0.05 }
fn default_max_cycle_loss() -> f64 { 0.01 }
fn default_circuit_breaker() -> u32 { 10 }
fn default_max_stake() -> f64 { 0.10 }

fn default_amm_programs() -> Vec<String> {
    vec![
        "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8".to_string(), // Raydium
        "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc".to_string(),  // Orca Whirlpool
        "LBUZKhRxPF3XUpBCjp4YzTKgLccjZhTSDM9YuVaPwxo".to_string(),  // Meteora
    ]
}

impl Default for GeneralConfig {
    fn default() -> Self {
        Self {
            dry_run: true,
            state_dir: default_state_dir(),
            log_level: default_log_level(),
            metrics_interval_secs: default_metrics_interval(),
        }
    }
}

impl Default for ZkConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            succinct_rpc: None,
            gevulot_rpc: None,
            max_concurrent_proofs: default_zk_concurrency(),
            min_bid_multiplier: default_min_bid(),
            ntt_optimization: 0,
            montgomery_ifma: true,
            pippenger_window: default_pippenger_window(),
        }
    }
}

impl Default for MevConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            chain: default_chain(),
            rpc_endpoints: vec![],
            ws_endpoints: vec![],
            jito_block_engine: None,
            max_latency_us: default_max_latency_us(),
            amm_programs: default_amm_programs(),
            min_profit_threshold: default_min_profit(),
            arena_size_bytes: default_arena_size(),
            kernel_bypass: false,
            intent_solver: false,
        }
    }
}

impl Default for MlConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            bittensor_endpoint: None,
            subnet_uid: default_subnet_uid(),
            wallet_name: default_wallet(),
            hotkey_name: default_hotkey(),
            model_path: None,
            gpu_devices: vec![],
            batch_size: default_batch_size(),
            background_training: true,
            training_data_dir: None,
            precision: default_precision(),
        }
    }
}

impl Default for RiskConfig {
    fn default() -> Self {
        Self {
            max_capital_at_risk: default_max_capital_risk(),
            max_cycle_loss: default_max_cycle_loss(),
            circuit_breaker_threshold: default_circuit_breaker(),
            daily_loss_limit: None,
            max_stake_fraction: default_max_stake(),
        }
    }
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            general: GeneralConfig::default(),
            zk: ZkConfig::default(),
            mev: MevConfig::default(),
            ml: MlConfig::default(),
            risk: RiskConfig::default(),
        }
    }
}

impl DaemonConfig {
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = toml::from_str(&content)?;
        Ok(config)
    }
}
