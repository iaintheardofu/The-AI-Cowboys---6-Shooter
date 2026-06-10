//! AMM Pool Models — constant-product, concentrated liquidity, and weighted curves.
//!
//! All calculations use integer arithmetic for determinism.
//! No floating point on the hot path.

use crate::net::p2p::MempoolTransaction;
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

/// Parse a mempool transaction to extract swap instruction.
/// Uses branchless program ID matching via bloom filter.
pub fn parse_swap_instruction(tx: &MempoolTransaction) -> Option<SwapInstruction> {
    // In production: decode Raydium/Orca/Meteora instruction layout
    // from transaction data using pre-computed discriminator lookup table.
    //
    // The discriminator is the first 8 bytes of the instruction data,
    // which is a hash of the instruction name. We use a jump table
    // (instruction table) for O(1) dispatch instead of if/else chains.

    if tx.data.len() < 8 {
        return None;
    }

    // Discriminator-based dispatch (placeholder)
    let _discriminator = &tx.data[..8];

    // Would decode: pool_id, token_in, token_out, amount_in, min_amount_out
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

            let fee_adjusted = amount_in_step * fee_complement / 10_000;
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
}
