//! State Shadow — cache-aligned local replica of on-chain pool state.
//!
//! Maintains a Structure-of-Arrays (SoA) representation of all monitored
//! AMM pools for SIMD-friendly scanning. Supports speculative state
//! application (simulate pending transactions before block confirmation)
//! and fast invalidation on new block arrival.
//!
//! Memory layout: all reserve arrays are contiguous and 64-byte aligned,
//! so a single cache-line prefetch brings 4 pool reserves (4 × u128 = 64B).
//! The CPU's hardware prefetcher will stream sequential reads at L1 bandwidth.

use crate::memory::cache::CACHE_LINE;
use std::sync::atomic::{AtomicU64, Ordering};

/// Maximum number of pools tracked in the shadow state.
/// Power-of-two for branchless modular index (mask = MAX - 1).
pub const MAX_POOLS: usize = 4096;
const INDEX_MASK: usize = MAX_POOLS - 1;

/// Cache-aligned Structure-of-Arrays pool state.
/// Each field is a contiguous array — SIMD can sweep entire columns.
#[repr(align(64))]
pub struct StateShadow {
    // ── Pool reserves (SoA) ──────────────────────────────────────────
    /// Token X reserve per pool (constant-product: x * y = k)
    pub reserve_x: Box<[u128; MAX_POOLS]>,
    /// Token Y reserve per pool
    pub reserve_y: Box<[u128; MAX_POOLS]>,
    /// Fee numerator per pool (basis points style)
    pub fee_num: Box<[u32; MAX_POOLS]>,
    /// Fee denominator per pool
    pub fee_den: Box<[u32; MAX_POOLS]>,

    // ── Token identity (SoA) ─────────────────────────────────────────
    /// Token X mint address (first 8 bytes as u64 for fast comparison)
    pub token_x_id: Box<[u64; MAX_POOLS]>,
    /// Token Y mint address (first 8 bytes as u64)
    pub token_y_id: Box<[u64; MAX_POOLS]>,
    /// Full pool pubkey (32 bytes each)
    pub pool_keys: Box<[[u8; 32]; MAX_POOLS]>,

    // ── Derived pricing (precomputed for hot path) ───────────────────
    /// Price of Y in terms of X: (reserve_x * PRICE_SCALE) / reserve_y
    /// Fixed-point Q32.32 to avoid floating point.
    pub price_x_per_y: Box<[u64; MAX_POOLS]>,
    /// Inverse: price of X in terms of Y
    pub price_y_per_x: Box<[u64; MAX_POOLS]>,

    // ── Metadata ─────────────────────────────────────────────────────
    /// Slot at which each pool was last updated
    pub last_updated_slot: Box<[u64; MAX_POOLS]>,
    /// Bitfield: 1 = pool is active, 0 = slot is empty
    pub active: Box<[u8; MAX_POOLS]>,
    /// Number of active pools
    pub pool_count: usize,
    /// Current confirmed slot
    pub confirmed_slot: AtomicU64,

    // ── Speculative overlay ──────────────────────────────────────────
    /// Speculative reserve_x (after simulating pending mempool txs)
    pub spec_reserve_x: Box<[u128; MAX_POOLS]>,
    /// Speculative reserve_y
    pub spec_reserve_y: Box<[u128; MAX_POOLS]>,
    /// Number of pending speculative modifications
    pub spec_dirty_count: usize,
    /// Bitmap of which pools have speculative state
    pub spec_dirty: Box<[u8; MAX_POOLS]>,
}

/// Q32.32 fixed-point scale factor for price representation.
pub const PRICE_SCALE: u128 = 1u128 << 32;

impl StateShadow {
    pub fn new() -> Self {
        Self {
            reserve_x: Box::new([0u128; MAX_POOLS]),
            reserve_y: Box::new([0u128; MAX_POOLS]),
            fee_num: Box::new([0u32; MAX_POOLS]),
            fee_den: Box::new([0u32; MAX_POOLS]),
            token_x_id: Box::new([0u64; MAX_POOLS]),
            token_y_id: Box::new([0u64; MAX_POOLS]),
            pool_keys: Box::new([[0u8; 32]; MAX_POOLS]),
            price_x_per_y: Box::new([0u64; MAX_POOLS]),
            price_y_per_x: Box::new([0u64; MAX_POOLS]),
            last_updated_slot: Box::new([0u64; MAX_POOLS]),
            active: Box::new([0u8; MAX_POOLS]),
            pool_count: 0,
            confirmed_slot: AtomicU64::new(0),
            spec_reserve_x: Box::new([0u128; MAX_POOLS]),
            spec_reserve_y: Box::new([0u128; MAX_POOLS]),
            spec_dirty_count: 0,
            spec_dirty: Box::new([0u8; MAX_POOLS]),
        }
    }

    /// Register a new pool. Returns the pool index.
    pub fn register_pool(
        &mut self,
        pool_key: [u8; 32],
        token_x: u64,
        token_y: u64,
        reserve_x: u128,
        reserve_y: u128,
        fee_num: u32,
        fee_den: u32,
        slot: u64,
    ) -> Option<usize> {
        if self.pool_count >= MAX_POOLS {
            return None;
        }
        let idx = self.pool_count;
        self.pool_keys[idx] = pool_key;
        self.token_x_id[idx] = token_x;
        self.token_y_id[idx] = token_y;
        self.reserve_x[idx] = reserve_x;
        self.reserve_y[idx] = reserve_y;
        self.spec_reserve_x[idx] = reserve_x;
        self.spec_reserve_y[idx] = reserve_y;
        self.fee_num[idx] = fee_num;
        self.fee_den[idx] = fee_den;
        self.last_updated_slot[idx] = slot;
        self.active[idx] = 1;
        self.recompute_price(idx);
        self.pool_count += 1;
        Some(idx)
    }

    /// Update pool reserves from a confirmed block.
    /// Clears any speculative state for this pool.
    #[inline]
    pub fn update_reserves(&mut self, idx: usize, reserve_x: u128, reserve_y: u128, slot: u64) {
        debug_assert!(idx < self.pool_count);
        self.reserve_x[idx] = reserve_x;
        self.reserve_y[idx] = reserve_y;
        self.spec_reserve_x[idx] = reserve_x;
        self.spec_reserve_y[idx] = reserve_y;
        self.last_updated_slot[idx] = slot;
        self.spec_dirty[idx] = 0;
        self.recompute_price(idx);
    }

    /// Apply a speculative swap to the shadow state.
    /// Models what the reserves WILL be after a pending mempool tx lands.
    #[inline]
    pub fn apply_speculative_swap(
        &mut self,
        idx: usize,
        amount_in: u128,
        x_to_y: bool,
    ) -> u128 {
        debug_assert!(idx < self.pool_count);
        let (rx, ry) = (self.spec_reserve_x[idx], self.spec_reserve_y[idx]);
        let fee_num = self.fee_num[idx] as u128;
        let fee_den = self.fee_den[idx] as u128;
        let amount_after_fee = amount_in - (amount_in * fee_num / fee_den);

        let (new_rx, new_ry, amount_out) = if x_to_y {
            let new_rx = rx + amount_after_fee;
            let k = rx * ry;
            let new_ry = k / new_rx;
            let out = ry - new_ry;
            (new_rx, new_ry, out)
        } else {
            let new_ry = ry + amount_after_fee;
            let k = rx * ry;
            let new_rx = k / new_ry;
            let out = rx - new_rx;
            (new_rx, new_ry, out)
        };

        self.spec_reserve_x[idx] = new_rx;
        self.spec_reserve_y[idx] = new_ry;
        if self.spec_dirty[idx] == 0 {
            self.spec_dirty[idx] = 1;
            self.spec_dirty_count += 1;
        }
        amount_out
    }

    /// Revert all speculative state back to confirmed reserves.
    pub fn revert_speculative(&mut self) {
        for i in 0..self.pool_count {
            // Branchless: always copy, skip conditional
            self.spec_reserve_x[i] = self.reserve_x[i];
            self.spec_reserve_y[i] = self.reserve_y[i];
            self.spec_dirty[i] = 0;
        }
        self.spec_dirty_count = 0;
    }

    /// On new block confirmation: advance the confirmed slot,
    /// revert speculative state, and prepare for next block.
    pub fn on_new_block(&mut self, slot: u64) {
        self.confirmed_slot.store(slot, Ordering::Release);
        self.revert_speculative();
    }

    /// Simulate a constant-product swap on SPECULATIVE reserves.
    /// Used by the solver to evaluate arb profitability.
    #[inline(always)]
    pub fn sim_swap(&self, idx: usize, amount_in: u128, x_to_y: bool) -> u128 {
        let (rx, ry) = (self.spec_reserve_x[idx], self.spec_reserve_y[idx]);
        if rx == 0 || ry == 0 {
            return 0;
        }
        let fee_num = self.fee_num[idx] as u128;
        let fee_den = self.fee_den[idx] as u128;
        let amount_after_fee = amount_in - (amount_in * fee_num / fee_den);

        if x_to_y {
            let new_rx = rx + amount_after_fee;
            let k = rx * ry;
            ry - k / new_rx
        } else {
            let new_ry = ry + amount_after_fee;
            let k = rx * ry;
            rx - k / new_ry
        }
    }

    /// Branchless lookup: find pool index by pool pubkey.
    /// Linear scan with early-exit (pools are typically < 1000).
    #[inline]
    pub fn find_pool(&self, key: &[u8; 32]) -> Option<usize> {
        for i in 0..self.pool_count {
            if self.pool_keys[i] == *key {
                return Some(i);
            }
        }
        None
    }

    /// Find all pools that trade a given token pair.
    /// Returns pool indices. Token order doesn't matter.
    #[inline]
    pub fn find_pools_for_pair(&self, token_a: u64, token_b: u64) -> Vec<usize> {
        let mut result = Vec::with_capacity(8);
        for i in 0..self.pool_count {
            if self.active[i] == 0 {
                continue;
            }
            let (x, y) = (self.token_x_id[i], self.token_y_id[i]);
            // Branchless: check both orderings
            let match_fwd = (x == token_a && y == token_b) as u8;
            let match_rev = (x == token_b && y == token_a) as u8;
            if match_fwd | match_rev != 0 {
                result.push(i);
            }
        }
        result
    }

    /// Recompute Q32.32 fixed-point prices for a pool.
    #[inline]
    fn recompute_price(&mut self, idx: usize) {
        let rx = self.reserve_x[idx];
        let ry = self.reserve_y[idx];
        if ry > 0 {
            self.price_x_per_y[idx] = ((rx as u128 * PRICE_SCALE) / ry) as u64;
        }
        if rx > 0 {
            self.price_y_per_x[idx] = ((ry as u128 * PRICE_SCALE) / rx) as u64;
        }
    }

    /// Get effective exchange rate from pool i (x→y) as Q32.32.
    /// Accounts for fees. Used to build the price graph.
    #[inline]
    pub fn effective_rate_x_to_y(&self, idx: usize) -> u64 {
        let fee_complement = self.fee_den[idx] as u64 - self.fee_num[idx] as u64;
        // rate = (reserve_y / reserve_x) * (1 - fee)
        // In Q32.32: (reserve_y * SCALE * fee_complement) / (reserve_x * fee_den)
        let rx = self.spec_reserve_x[idx];
        let ry = self.spec_reserve_y[idx];
        if rx == 0 {
            return 0;
        }
        ((ry as u128 * PRICE_SCALE * fee_complement as u128)
            / (rx as u128 * self.fee_den[idx] as u128)) as u64
    }

    /// Get effective exchange rate from pool i (y→x) as Q32.32.
    #[inline]
    pub fn effective_rate_y_to_x(&self, idx: usize) -> u64 {
        let fee_complement = self.fee_den[idx] as u64 - self.fee_num[idx] as u64;
        let rx = self.spec_reserve_x[idx];
        let ry = self.spec_reserve_y[idx];
        if ry == 0 {
            return 0;
        }
        ((rx as u128 * PRICE_SCALE * fee_complement as u128)
            / (ry as u128 * self.fee_den[idx] as u128)) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_shadow() -> StateShadow {
        let mut s = StateShadow::new();
        // Pool 0: SOL/USDC on Raydium (1 SOL = 150 USDC)
        s.register_pool(
            [0x01; 32], 0xAAAA, 0xBBBB,
            1_000_000_000_000, // 1000 SOL (in lamports-scale)
            150_000_000_000_000, // 150,000 USDC
            3, 1000, // 0.3% fee
            100,
        );
        // Pool 1: SOL/USDC on Orca (1 SOL = 151 USDC — slight premium)
        s.register_pool(
            [0x02; 32], 0xAAAA, 0xBBBB,
            900_000_000_000,
            135_900_000_000_000, // 900 * 151
            3, 1000,
            100,
        );
        s
    }

    #[test]
    fn test_register_and_find() {
        let s = make_shadow();
        assert_eq!(s.pool_count, 2);
        assert_eq!(s.find_pool(&[0x01; 32]), Some(0));
        assert_eq!(s.find_pool(&[0x02; 32]), Some(1));
        assert_eq!(s.find_pool(&[0xFF; 32]), None);
    }

    #[test]
    fn test_find_pools_for_pair() {
        let s = make_shadow();
        let pools = s.find_pools_for_pair(0xAAAA, 0xBBBB);
        assert_eq!(pools.len(), 2);
        let pools_rev = s.find_pools_for_pair(0xBBBB, 0xAAAA);
        assert_eq!(pools_rev.len(), 2);
    }

    #[test]
    fn test_sim_swap() {
        let s = make_shadow();
        // Swap 1 SOL for USDC on pool 0
        let out = s.sim_swap(0, 1_000_000_000, true);
        // With k = 1e12 * 150e12, after fee: ~149.55 USDC worth
        assert!(out > 100_000_000_000); // > 100 USDC
        assert!(out < 200_000_000_000); // < 200 USDC
    }

    #[test]
    fn test_speculative_swap_and_revert() {
        let mut s = make_shadow();
        let orig_ry = s.reserve_y[0];

        // Apply speculative swap
        let out = s.apply_speculative_swap(0, 1_000_000_000, true);
        assert!(out > 0);
        assert_ne!(s.spec_reserve_y[0], orig_ry);
        assert_eq!(s.spec_dirty[0], 1);
        assert_eq!(s.spec_dirty_count, 1);

        // Revert
        s.revert_speculative();
        assert_eq!(s.spec_reserve_y[0], orig_ry);
        assert_eq!(s.spec_dirty_count, 0);
    }

    #[test]
    fn test_price_computation() {
        let s = make_shadow();
        // Pool 0: 1000 SOL / 150000 USDC → price ~150 USDC/SOL
        let price = s.price_x_per_y[0];
        // Q32.32: price of 1 Y (USDC) in X (SOL) ≈ 1000/150000 ≈ 0.00667
        // In Q32.32: 0.00667 * 2^32 ≈ 28,630,000
        assert!(price > 0);
    }

    #[test]
    fn test_effective_rate() {
        let s = make_shadow();
        let rate = s.effective_rate_x_to_y(0);
        assert!(rate > 0);
        let rate_rev = s.effective_rate_y_to_x(0);
        assert!(rate_rev > 0);
    }

    #[test]
    fn test_on_new_block() {
        let mut s = make_shadow();
        s.apply_speculative_swap(0, 1_000_000_000, true);
        assert_eq!(s.spec_dirty_count, 1);

        s.on_new_block(200);
        assert_eq!(s.confirmed_slot.load(Ordering::Relaxed), 200);
        assert_eq!(s.spec_dirty_count, 0);
    }
}
