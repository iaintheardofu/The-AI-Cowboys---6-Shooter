//! Cache-aligned data structures — SoA layout for SIMD-friendly access.
//!
//! All hot-path data uses Structure-of-Arrays layout to maximize spatial
//! locality and SIMD throughput. 64-byte alignment matches cache line size.

/// Cache line size on x86-64 and ARM (common).
pub const CACHE_LINE: usize = 64;

/// Align a struct to cache line boundary to prevent false sharing.
#[repr(align(64))]
#[derive(Clone)]
pub struct CacheAligned<T>(pub T);

impl<T: Default> Default for CacheAligned<T> {
    fn default() -> Self {
        Self(T::default())
    }
}

/// SoA order book representation — prices and volumes in separate
/// contiguous arrays for SIMD-accelerated min/max/sum operations.
#[repr(align(64))]
pub struct SoAOrderBook {
    pub bid_prices: Vec<u64>,
    pub bid_volumes: Vec<u64>,
    pub ask_prices: Vec<u64>,
    pub ask_volumes: Vec<u64>,
    pub timestamps: Vec<u64>,
    pub pool_ids: Vec<u32>,
    pub len: usize,
}

impl SoAOrderBook {
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            bid_prices: Vec::with_capacity(cap),
            bid_volumes: Vec::with_capacity(cap),
            ask_prices: Vec::with_capacity(cap),
            ask_volumes: Vec::with_capacity(cap),
            timestamps: Vec::with_capacity(cap),
            pool_ids: Vec::with_capacity(cap),
            len: 0,
        }
    }

    #[inline]
    pub fn push(&mut self, bid: u64, bid_vol: u64, ask: u64, ask_vol: u64, ts: u64, pool: u32) {
        self.bid_prices.push(bid);
        self.bid_volumes.push(bid_vol);
        self.ask_prices.push(ask);
        self.ask_volumes.push(ask_vol);
        self.timestamps.push(ts);
        self.pool_ids.push(pool);
        self.len += 1;
    }

    #[inline]
    pub fn clear(&mut self) {
        self.bid_prices.clear();
        self.bid_volumes.clear();
        self.ask_prices.clear();
        self.ask_volumes.clear();
        self.timestamps.clear();
        self.pool_ids.clear();
        self.len = 0;
    }

    /// SIMD-friendly argmin over ask prices — find cheapest sell.
    /// Uses branchless comparison for pipeline-hazard-free scanning.
    #[inline]
    pub fn argmin_ask(&self) -> Option<usize> {
        if self.ask_prices.is_empty() {
            return None;
        }
        let mut min_idx: usize = 0;
        let mut min_val = self.ask_prices[0];
        for i in 1..self.ask_prices.len() {
            let val = self.ask_prices[i];
            // Branchless: cmov-style selection
            let is_less = (val < min_val) as usize;
            min_idx = is_less * i + (1 - is_less) * min_idx;
            min_val = if val < min_val { val } else { min_val };
        }
        Some(min_idx)
    }

    /// SIMD-friendly argmax over bid prices — find most expensive buy.
    #[inline]
    pub fn argmax_bid(&self) -> Option<usize> {
        if self.bid_prices.is_empty() {
            return None;
        }
        let mut max_idx: usize = 0;
        let mut max_val = self.bid_prices[0];
        for i in 1..self.bid_prices.len() {
            let val = self.bid_prices[i];
            let is_greater = (val > max_val) as usize;
            max_idx = is_greater * i + (1 - is_greater) * max_idx;
            max_val = if val > max_val { val } else { max_val };
        }
        Some(max_idx)
    }

    /// Detect cross-exchange arbitrage: bid on one exchange > ask on another.
    /// Returns (buy_idx, sell_idx, spread) pairs.
    #[inline]
    pub fn detect_arbitrage(&self, min_spread: u64) -> Vec<(usize, usize, u64)> {
        let mut opportunities = Vec::new();
        for i in 0..self.len {
            for j in 0..self.len {
                if i == j || self.pool_ids[i] == self.pool_ids[j] {
                    continue;
                }
                // Buy at ask[i], sell at bid[j]
                if self.bid_prices[j] > self.ask_prices[i] {
                    let spread = self.bid_prices[j] - self.ask_prices[i];
                    if spread >= min_spread {
                        opportunities.push((i, j, spread));
                    }
                }
            }
        }
        opportunities
    }
}

/// Pool state for AMM constant-product curves (x * y = k).
/// All fields stored as u128 for overflow safety in multiplication.
#[repr(align(64))]
#[derive(Clone, Debug)]
pub struct AmmPool {
    pub reserve_x: u128,
    pub reserve_y: u128,
    pub fee_numerator: u64,
    pub fee_denominator: u64,
    pub pool_id: [u8; 32],
}

impl AmmPool {
    /// Constant-product swap: given input amount of X, compute output Y.
    /// Uses integer arithmetic only — no floating point for determinism.
    #[inline(always)]
    pub fn swap_x_to_y(&self, amount_in: u128) -> u128 {
        let fee = amount_in * self.fee_numerator as u128 / self.fee_denominator as u128;
        let amount_after_fee = amount_in - fee;
        let new_reserve_x = self.reserve_x + amount_after_fee;
        let k = self.reserve_x * self.reserve_y;
        let new_reserve_y = k / new_reserve_x;
        self.reserve_y - new_reserve_y
    }

    /// Reverse: given input amount of Y, compute output X.
    #[inline(always)]
    pub fn swap_y_to_x(&self, amount_in: u128) -> u128 {
        let fee = amount_in * self.fee_numerator as u128 / self.fee_denominator as u128;
        let amount_after_fee = amount_in - fee;
        let new_reserve_y = self.reserve_y + amount_after_fee;
        let k = self.reserve_x * self.reserve_y;
        let new_reserve_x = k / new_reserve_y;
        self.reserve_x - new_reserve_x
    }

    /// Optimal input for max profit across two pools (A->B on pool1, B->A on pool2).
    /// Binary search over input amounts (branchless-friendly).
    pub fn optimal_arb_input(pool1: &AmmPool, pool2: &AmmPool, max_input: u128) -> (u128, u128) {
        let mut lo: u128 = 1;
        let mut hi: u128 = max_input;
        let mut best_input: u128 = 0;
        let mut best_profit: u128 = 0;

        while lo <= hi {
            let mid = lo + (hi - lo) / 2;
            let out_b = pool1.swap_x_to_y(mid);
            let out_a = pool2.swap_y_to_x(out_b);
            if out_a > mid {
                let profit = out_a - mid;
                if profit > best_profit {
                    best_profit = profit;
                    best_input = mid;
                }
                lo = mid + 1;
            } else {
                if mid == 0 { break; }
                hi = mid - 1;
            }
        }
        (best_input, best_profit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_soa_argmin_argmax() {
        let mut book = SoAOrderBook::with_capacity(4);
        book.push(100, 10, 105, 10, 1000, 0);
        book.push(102, 20, 103, 20, 1001, 1);
        book.push(99, 15, 108, 15, 1002, 2);

        assert_eq!(book.argmin_ask(), Some(1)); // 103 is cheapest ask
        assert_eq!(book.argmax_bid(), Some(1)); // 102 is highest bid
    }

    #[test]
    fn test_arbitrage_detection() {
        let mut book = SoAOrderBook::with_capacity(4);
        // Pool 0: bid=100, ask=105
        book.push(100, 10, 105, 10, 1000, 0);
        // Pool 1: bid=110, ask=103 — arb: buy at 103 on pool1, sell at 110 on pool1 NO (same pool)
        book.push(110, 20, 103, 20, 1001, 1);

        let arbs = book.detect_arbitrage(1);
        // Buy at ask[1]=103 (pool 1), sell at bid[1]=110? No, same pool
        // Buy at ask[0]=105 (pool 0), sell at bid[1]=110 (pool 1) = spread 5
        assert!(!arbs.is_empty());
        assert_eq!(arbs[0], (0, 1, 5));
    }

    #[test]
    fn test_amm_swap() {
        let pool = AmmPool {
            reserve_x: 1_000_000,
            reserve_y: 1_000_000,
            fee_numerator: 3,
            fee_denominator: 1000, // 0.3% fee
            pool_id: [0u8; 32],
        };
        let out = pool.swap_x_to_y(10_000);
        // With 0.3% fee on 10K input: ~9970 effective
        // k = 1e12, new_x = 1009970, new_y = k/new_x ≈ 990128
        // output ≈ 9872
        assert!(out > 9800 && out < 10000);
    }

    #[test]
    fn test_optimal_arb() {
        let pool1 = AmmPool {
            reserve_x: 1_000_000,
            reserve_y: 2_000_000, // Cheap Y on pool1
            fee_numerator: 3,
            fee_denominator: 1000,
            pool_id: [1u8; 32],
        };
        let pool2 = AmmPool {
            reserve_x: 2_000_000,
            reserve_y: 1_000_000, // Expensive Y on pool2 (cheap X)
            fee_numerator: 3,
            fee_denominator: 1000,
            pool_id: [2u8; 32],
        };
        let (input, profit) = AmmPool::optimal_arb_input(&pool1, &pool2, 100_000);
        assert!(profit > 0, "Should find profitable arb");
        assert!(input > 0);
    }
}
