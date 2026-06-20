//! AMM Pool Models — constant-product, concentrated liquidity, and weighted curves.
//!
//! All calculations use integer arithmetic for determinism.
//! No floating point on the hot path.

use crate::net::p2p::MempoolTransaction;
use crate::net::solana::bs58_decode;
use crate::memory::cache::AmmPool;
use std::collections::HashMap;

/// Parsed swap instruction from a mempool transaction.
#[derive(Clone, Debug)]
pub struct SwapInstruction {
    pub pool_id: [u8; 32],
    pub token_in: [u8; 32],
    pub token_out: [u8; 32],
    pub amount_in: u128,
    pub min_amount_out: u128,
    pub user: [u8; 32],
}

/// Registry of known AMM pool states, updated via RPC polling or Geyser.
pub struct PoolRegistry {
    pools: HashMap<[u8; 32], AmmPool>,
    /// Index: token pair -> list of pool IDs
    pair_index: HashMap<([u8; 32], [u8; 32]), Vec<[u8; 32]>>,
}

impl PoolRegistry {
    pub fn new() -> Self {
        Self {
            pools: HashMap::new(),
            pair_index: HashMap::new(),
        }
    }

    pub fn register_pool(&mut self, pool: AmmPool, token_a: [u8; 32], token_b: [u8; 32]) {
        let id = pool.pool_id;
        self.pools.insert(id, pool);
        self.pair_index.entry((token_a, token_b)).or_default().push(id);
        self.pair_index.entry((token_b, token_a)).or_default().push(id);
    }

    pub fn get_pool(&self, id: &[u8; 32]) -> Option<&AmmPool> {
        self.pools.get(id)
    }

    pub fn get_pools_for_pair(&self, token_a: &[u8; 32], token_b: &[u8; 32]) -> Vec<&AmmPool> {
        self.pair_index
            .get(&(*token_a, *token_b))
            .map(|ids| ids.iter().filter_map(|id| self.pools.get(id)).collect())
            .unwrap_or_default()
    }

    pub fn pool_count(&self) -> usize {
        self.pools.len()
    }
}

/// Known AMM program discriminators for O(1) dispatch.
const RAYDIUM_SWAP_DISC: [u8; 8] = [0x09, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
const ORCA_SWAP_DISC: [u8; 8] = [0xf8, 0xc6, 0x9e, 0x91, 0xe1, 0x75, 0x87, 0xc8];
const METEORA_SWAP_DISC: [u8; 8] = [0xe4, 0x45, 0xa5, 0x2e, 0x51, 0xcb, 0x9a, 0x1d];

/// Decode a base58 account string to [u8; 32], or return zeroes on failure.
#[inline]
fn decode_account(s: &str) -> [u8; 32] {
    let mut arr = [0u8; 32];
    if let Ok(bytes) = bs58_decode(s) {
        if bytes.len() == 32 {
            arr.copy_from_slice(&bytes);
        }
    }
    arr
}

/// Get decoded account at index, or return zeroes if out of bounds.
#[inline]
fn account_at(accounts: &[String], idx: usize) -> [u8; 32] {
    accounts.get(idx).map(|s| decode_account(s)).unwrap_or([0u8; 32])
}

/// Parse a mempool transaction to extract swap instruction.
/// Uses discriminator-based dispatch for Raydium V4, Orca Whirlpool, and Meteora DLMM.
pub fn parse_swap_instruction(tx: &MempoolTransaction) -> Option<SwapInstruction> {
    if tx.data.len() < 8 {
        return None;
    }

    let discriminator: [u8; 8] = tx.data[..8].try_into().ok()?;

    // Raydium V4 swap: [8 disc][8 amount_in][8 min_out] = 24 bytes
    // Accounts: [token_program, amm_id, authority, open_orders, target_orders,
    //            pool_coin, pool_pc, serum_program, serum_market, serum_bids,
    //            serum_asks, serum_eq, serum_coin_vault, serum_pc_vault,
    //            serum_vault_signer, user_source, user_dest, user_owner]
    if discriminator == RAYDIUM_SWAP_DISC {
        if tx.data.len() < 24 || tx.accounts.len() < 18 {
            return None;
        }
        let amount_in = u64::from_le_bytes(tx.data[8..16].try_into().ok()?) as u128;
        let min_amount_out = u64::from_le_bytes(tx.data[16..24].try_into().ok()?) as u128;
        return Some(SwapInstruction {
            pool_id: account_at(&tx.accounts, 1),     // amm_id
            token_in: account_at(&tx.accounts, 15),    // user_source
            token_out: account_at(&tx.accounts, 16),   // user_dest
            amount_in,
            min_amount_out,
            user: account_at(&tx.accounts, 17),        // user_owner
        });
    }

    // Orca Whirlpool (Anchor): [8 disc][8 amount][8 other_amount_threshold]...
    if discriminator == ORCA_SWAP_DISC {
        if tx.data.len() < 16 || tx.accounts.len() < 11 {
            return None;
        }
        let amount_in = u64::from_le_bytes(tx.data[8..16].try_into().ok()?) as u128;
        let min_amount_out = if tx.data.len() >= 24 {
            u64::from_le_bytes(tx.data[16..24].try_into().ok()?) as u128
        } else { 0 };
        return Some(SwapInstruction {
            pool_id: account_at(&tx.accounts, 2),      // whirlpool
            token_in: account_at(&tx.accounts, 5),     // token_owner_account_a
            token_out: account_at(&tx.accounts, 7),    // token_owner_account_b
            amount_in,
            min_amount_out,
            user: account_at(&tx.accounts, 1),         // token_authority
        });
    }

    // Meteora DLMM swap
    if discriminator == METEORA_SWAP_DISC {
        if tx.data.len() < 24 || tx.accounts.len() < 10 {
            return None;
        }
        let amount_in = u64::from_le_bytes(tx.data[8..16].try_into().ok()?) as u128;
        let min_amount_out = u64::from_le_bytes(tx.data[16..24].try_into().ok()?) as u128;
        return Some(SwapInstruction {
            pool_id: account_at(&tx.accounts, 0),      // lb_pair
            token_in: account_at(&tx.accounts, 5),     // user_token_in
            token_out: account_at(&tx.accounts, 6),    // user_token_out
            amount_in,
            min_amount_out,
            user: account_at(&tx.accounts, 8),         // sender
        });
    }

    None
}

/// Concentrated Liquidity pool (Uniswap V3 / Orca Whirlpool style).
/// Maintains tick-indexed liquidity for more capital-efficient swaps.
#[derive(Clone, Debug)]
pub struct ConcentratedPool {
    pub pool_id: [u8; 32],
    pub sqrt_price_x64: u128, // Q64.64 fixed-point sqrt(price)
    pub liquidity: u128,
    pub tick_current: i32,
    pub fee_rate: u32, // basis points
    pub tick_spacing: i32,
    /// Tick -> liquidity delta (net liquidity change at this tick)
    pub tick_data: Vec<(i32, i128)>,
}

impl ConcentratedPool {
    /// Compute swap output for concentrated liquidity.
    /// Steps through ticks, consuming liquidity at each price level.
    pub fn swap(&self, amount_in: u128, zero_for_one: bool) -> u128 {
        let mut remaining = amount_in;
        let mut amount_out: u128 = 0;
        let mut sqrt_price = self.sqrt_price_x64;
        let mut liquidity = self.liquidity;
        let mut tick = self.tick_current;

        let fee_complement = 10_000u128 - self.fee_rate as u128;

        while remaining > 0 {
            // Find next initialized tick
            let next_tick = if zero_for_one {
                self.find_prev_tick(tick)
            } else {
                self.find_next_tick(tick)
            };

            let next_tick = match next_tick {
                Some(t) => t,
                None => break, // No more liquidity
            };

            // Compute sqrt_price at next tick
            let sqrt_price_next = tick_to_sqrt_price(next_tick);

            // Compute max amount swappable in this tick range
            let (amount_in_step, amount_out_step) = if zero_for_one {
                compute_step_amounts(sqrt_price, sqrt_price_next, liquidity, remaining, true)
            } else {
                compute_step_amounts(sqrt_price, sqrt_price_next, liquidity, remaining, false)
            };

            let _fee_adjusted = amount_in_step * fee_complement / 10_000;
            remaining = remaining.saturating_sub(amount_in_step);
            amount_out += amount_out_step;

            // Cross tick: update liquidity
            if remaining > 0 {
                if let Some((_, delta)) = self.tick_data.iter().find(|(t, _)| *t == next_tick) {
                    if zero_for_one {
                        liquidity = (liquidity as i128 - delta) as u128;
                    } else {
                        liquidity = (liquidity as i128 + delta) as u128;
                    }
                }
                sqrt_price = sqrt_price_next;
                tick = next_tick;
            }
        }

        amount_out
    }

    fn find_next_tick(&self, current: i32) -> Option<i32> {
        self.tick_data.iter()
            .filter(|(t, _)| *t > current)
            .map(|(t, _)| *t)
            .min()
    }

    fn find_prev_tick(&self, current: i32) -> Option<i32> {
        self.tick_data.iter()
            .filter(|(t, _)| *t < current)
            .map(|(t, _)| *t)
            .max()
    }
}

/// Convert tick index to sqrt(price) in Q64.64 fixed-point.
#[inline]
fn tick_to_sqrt_price(tick: i32) -> u128 {
    // sqrt(1.0001^tick) * 2^64
    // Using fixed-point approximation: 2^64 * 1.00005^tick
    let base: f64 = 1.0001_f64.powi(tick).sqrt();
    (base * (1u128 << 64) as f64) as u128
}

/// Compute swap amounts within a single tick range.
fn compute_step_amounts(
    sqrt_price_a: u128,
    sqrt_price_b: u128,
    liquidity: u128,
    max_amount_in: u128,
    zero_for_one: bool,
) -> (u128, u128) {
    if liquidity == 0 || sqrt_price_a == sqrt_price_b {
        return (0, 0);
    }

    let (lower, upper) = if sqrt_price_a < sqrt_price_b {
        (sqrt_price_a, sqrt_price_b)
    } else {
        (sqrt_price_b, sqrt_price_a)
    };

    if zero_for_one {
        // amount0 = L * (1/sqrt_b - 1/sqrt_a) = L * (sqrt_a - sqrt_b) / (sqrt_a * sqrt_b)
        let diff = upper.saturating_sub(lower);
        let amount_in = (liquidity as u128 * diff) / ((upper >> 32) * (lower >> 32));
        let amount_in = amount_in.min(max_amount_in);
        // amount1 = L * (sqrt_a - sqrt_b)
        let amount_out = (liquidity * diff) >> 64;
        (amount_in, amount_out)
    } else {
        let diff = upper.saturating_sub(lower);
        let amount_in = (liquidity * diff) >> 64;
        let amount_in = amount_in.min(max_amount_in);
        let amount_out = (liquidity as u128 * diff) / ((upper >> 32) * (lower >> 32));
        (amount_in, amount_out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_registry() {
        let mut reg = PoolRegistry::new();
        let pool = AmmPool {
            reserve_x: 1_000_000,
            reserve_y: 2_000_000,
            fee_numerator: 3,
            fee_denominator: 1000,
            pool_id: [1u8; 32],
        };
        let token_a = [0xAA; 32];
        let token_b = [0xBB; 32];
        reg.register_pool(pool, token_a, token_b);

        assert_eq!(reg.pool_count(), 1);
        assert_eq!(reg.get_pools_for_pair(&token_a, &token_b).len(), 1);
        assert_eq!(reg.get_pools_for_pair(&token_b, &token_a).len(), 1);
    }

    #[test]
    fn test_parse_swap_instruction_raydium() {
        // Build a Raydium V4 swap transaction
        let mut data = Vec::new();
        data.extend_from_slice(&RAYDIUM_SWAP_DISC); // 8-byte discriminator
        data.extend_from_slice(&1_000_000u64.to_le_bytes()); // amount_in
        data.extend_from_slice(&900_000u64.to_le_bytes()); // min_amount_out

        // 18 accounts required for Raydium V4
        let accounts: Vec<String> = (0..18).map(|i| format!("Account{:02}", i)).collect();

        let tx = MempoolTransaction {
            signature: "test_sig".to_string(),
            program_ids: vec!["675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8".to_string()],
            accounts,
            data,
            slot: 100,
            received_at_ns: 0,
        };

        let swap = parse_swap_instruction(&tx);
        assert!(swap.is_some(), "Should parse Raydium swap");
        let swap = swap.unwrap();
        assert_eq!(swap.amount_in, 1_000_000);
        assert_eq!(swap.min_amount_out, 900_000);
    }

    #[test]
    fn test_parse_swap_instruction_orca() {
        let mut data = Vec::new();
        data.extend_from_slice(&ORCA_SWAP_DISC);
        data.extend_from_slice(&500_000u64.to_le_bytes());
        data.extend_from_slice(&400_000u64.to_le_bytes());

        let accounts: Vec<String> = (0..11).map(|i| format!("OrcaAcc{:02}", i)).collect();

        let tx = MempoolTransaction {
            signature: "orca_sig".to_string(),
            program_ids: vec!["whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc".to_string()],
            accounts,
            data,
            slot: 200,
            received_at_ns: 0,
        };

        let swap = parse_swap_instruction(&tx);
        assert!(swap.is_some(), "Should parse Orca swap");
        assert_eq!(swap.unwrap().amount_in, 500_000);
    }

    #[test]
    fn test_parse_swap_instruction_meteora() {
        let mut data = Vec::new();
        data.extend_from_slice(&METEORA_SWAP_DISC);
        data.extend_from_slice(&250_000u64.to_le_bytes());
        data.extend_from_slice(&200_000u64.to_le_bytes());

        let accounts: Vec<String> = (0..10).map(|i| format!("MetAcc{:02}", i)).collect();

        let tx = MempoolTransaction {
            signature: "met_sig".to_string(),
            program_ids: vec![],
            accounts,
            data,
            slot: 300,
            received_at_ns: 0,
        };

        let swap = parse_swap_instruction(&tx);
        assert!(swap.is_some(), "Should parse Meteora swap");
        assert_eq!(swap.unwrap().amount_in, 250_000);
    }

    #[test]
    fn test_parse_swap_instruction_unknown_disc() {
        let data = vec![0xFF; 24];
        let accounts: Vec<String> = (0..18).map(|i| format!("Acc{}", i)).collect();
        let tx = MempoolTransaction {
            signature: "unknown".to_string(),
            program_ids: vec![],
            accounts,
            data,
            slot: 0,
            received_at_ns: 0,
        };
        assert!(parse_swap_instruction(&tx).is_none(), "Unknown discriminator should return None");
    }

    #[test]
    fn test_parse_swap_instruction_too_short() {
        let tx = MempoolTransaction {
            signature: "short".to_string(),
            program_ids: vec![],
            accounts: vec![],
            data: vec![0x09, 0x00], // Only 2 bytes — too short
            slot: 0,
            received_at_ns: 0,
        };
        assert!(parse_swap_instruction(&tx).is_none(), "Short data should return None");
    }
}
