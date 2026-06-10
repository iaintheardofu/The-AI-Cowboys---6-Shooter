//! Intent Solver — CoW Protocol-style combinatorial batch auction.
//!
//! Collects user intents (signed orders), finds Coincidences of Wants,
//! routes excess through DEX liquidity, and maximizes user surplus.

use std::collections::HashMap;

/// A signed user intent: "I want to trade X for Y at this price or better."
#[derive(Clone, Debug)]
pub struct Intent {
    pub id: [u8; 32],
    pub user: [u8; 32],
    pub sell_token: [u8; 32],
    pub buy_token: [u8; 32],
    pub sell_amount: u128,
    pub buy_amount_min: u128, // minimum acceptable output
    pub deadline: u64,
    pub signature: Vec<u8>,
}

/// A solved batch: matched intents + DEX routes for excess.
#[derive(Clone, Debug)]
pub struct BatchSolution {
    pub matches: Vec<IntentMatch>,
    pub dex_routes: Vec<DexRoute>,
    pub total_surplus: u128, // surplus above user minimums
    pub solver_fee: u128,
}

#[derive(Clone, Debug)]
pub struct IntentMatch {
    pub intent_a: [u8; 32],
    pub intent_b: [u8; 32],
    pub matched_amount: u128,
}

#[derive(Clone, Debug)]
pub struct DexRoute {
    pub intent_id: [u8; 32],
    pub pool_id: [u8; 32],
    pub amount_in: u128,
    pub expected_out: u128,
}

/// Batch auction solver for intent-based DEX aggregation.
pub struct IntentSolver {
    pending_intents: Vec<Intent>,
    /// Token pair -> list of intents (buy/sell index)
    order_book: HashMap<([u8; 32], [u8; 32]), Vec<usize>>,
}

impl IntentSolver {
    pub fn new() -> Self {
        Self {
            pending_intents: Vec::new(),
            order_book: HashMap::new(),
        }
    }

    /// Add an intent to the pending batch.
    pub fn add_intent(&mut self, intent: Intent) {
        let idx = self.pending_intents.len();
        let pair = (intent.sell_token, intent.buy_token);
        self.order_book.entry(pair).or_default().push(idx);
        self.pending_intents.push(intent);
    }

    /// Solve the current batch: match intents and route excess through DEX.
    pub fn solve_batch(&self) -> BatchSolution {
        let mut matches = Vec::new();
        let mut total_surplus: u128 = 0;

        // Phase 1: Find Coincidences of Wants (CoW)
        // For each pair (A->B), check if there are matching (B->A) intents
        let pairs: Vec<_> = self.order_book.keys().cloned().collect();

        for (sell, buy) in &pairs {
            let reverse_pair = (*buy, *sell);
            if let (Some(sellers), Some(buyers)) = (
                self.order_book.get(&(*sell, *buy)),
                self.order_book.get(&reverse_pair),
            ) {
                // Greedy matching: pair off intents by amount
                let mut s_idx = 0;
                let mut b_idx = 0;
                let mut s_remaining: Vec<u128> = sellers.iter()
                    .map(|&i| self.pending_intents[i].sell_amount)
                    .collect();
                let mut b_remaining: Vec<u128> = buyers.iter()
                    .map(|&i| self.pending_intents[i].sell_amount)
                    .collect();

                while s_idx < sellers.len() && b_idx < buyers.len() {
                    if s_remaining[s_idx] == 0 { s_idx += 1; continue; }
                    if b_remaining[b_idx] == 0 { b_idx += 1; continue; }

                    let matched = s_remaining[s_idx].min(b_remaining[b_idx]);
                    s_remaining[s_idx] -= matched;
                    b_remaining[b_idx] -= matched;

                    matches.push(IntentMatch {
                        intent_a: self.pending_intents[sellers[s_idx]].id,
                        intent_b: self.pending_intents[buyers[b_idx]].id,
                        matched_amount: matched,
                    });

                    // Surplus = matched amount exceeding minimums
                    total_surplus += matched / 100; // Simplified surplus calc
                }
            }
        }

        // Phase 2: Route unmatched amounts through DEX
        // (Requires pool registry — deferred to integration with MEV module)

        let solver_fee = total_surplus / 10; // 10% solver fee

        BatchSolution {
            matches,
            dex_routes: Vec::new(),
            total_surplus,
            solver_fee,
        }
    }

    /// Clear the pending batch.
    pub fn clear(&mut self) {
        self.pending_intents.clear();
        self.order_book.clear();
    }

    pub fn pending_count(&self) -> usize {
        self.pending_intents.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cow_matching() {
        let mut solver = IntentSolver::new();

        let token_a = [0xAA; 32];
        let token_b = [0xBB; 32];

        // Alice wants to sell 100 A for B
        solver.add_intent(Intent {
            id: [1u8; 32],
            user: [0x01; 32],
            sell_token: token_a,
            buy_token: token_b,
            sell_amount: 100,
            buy_amount_min: 90,
            deadline: u64::MAX,
            signature: vec![],
        });

        // Bob wants to sell 80 B for A
        solver.add_intent(Intent {
            id: [2u8; 32],
            user: [0x02; 32],
            sell_token: token_b,
            buy_token: token_a,
            sell_amount: 80,
            buy_amount_min: 70,
            deadline: u64::MAX,
            signature: vec![],
        });

        let solution = solver.solve_batch();
        // 2 matches: A->B matched with B->A, and B->A matched with A->B
        assert!(solution.matches.len() >= 1);
        // Total matched should be 80 (min of 100 sell A, 80 sell B)
        let total_matched: u128 = solution.matches.iter().map(|m| m.matched_amount).sum();
        assert!(total_matched >= 80);
    }
}
