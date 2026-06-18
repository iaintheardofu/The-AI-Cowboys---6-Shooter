//! Price Graph — Bellman-Ford negative cycle detection for multi-hop arbitrage.
//!
//! Models the entire DEX ecosystem as a directed weighted graph where:
//!   - Nodes = tokens
//!   - Edges = swap routes through pools
//!   - Edge weight = -log2(effective_rate)
//!
//! A negative cycle in this graph means the product of exchange rates
//! around the cycle exceeds 1.0 — i.e., a profitable arbitrage loop.
//!
//! Uses fixed-point arithmetic (Q16.16) for weights to avoid floating-point
//! non-determinism on the hot path. Bellman-Ford runs in O(V * E) which is
//! fast enough for the typical DEX graph (< 200 tokens, < 2000 edges).

/// Maximum number of unique tokens in the graph.
pub const MAX_TOKENS: usize = 512;
/// Maximum number of edges (pool directions).
pub const MAX_EDGES: usize = 8192;

/// Q16.16 fixed-point representation for log-prices.
/// Positive weight = exchange rate < 1 (losing value)
/// Negative weight = exchange rate > 1 (gaining value)
pub type FixedWeight = i32;

/// Scale factor for Q16.16 fixed-point.
const Q16_SCALE: f64 = 65536.0;

/// An edge in the price graph: swap from token `src` to token `dst`
/// through pool at index `pool_idx`.
#[derive(Clone, Copy, Debug)]
pub struct PriceEdge {
    pub src: u16,
    pub dst: u16,
    pub pool_idx: u16,
    /// -log2(effective_rate) in Q16.16 fixed-point
    pub weight: FixedWeight,
    /// Direction: true = x→y, false = y→x
    pub x_to_y: bool,
}

/// Detected arbitrage cycle with path reconstruction.
#[derive(Clone, Debug)]
pub struct ArbCycle {
    /// Sequence of token indices forming the cycle
    pub path: Vec<u16>,
    /// Corresponding pool indices for each hop
    pub pool_indices: Vec<u16>,
    /// Swap directions for each hop
    pub directions: Vec<bool>,
    /// Total negative weight (more negative = more profit)
    pub total_weight: FixedWeight,
    /// Estimated profit multiplier (e.g., 1.003 = 0.3% profit)
    pub profit_multiplier: f64,
}

/// Bellman-Ford based arbitrage graph.
pub struct PriceGraph {
    edges: Vec<PriceEdge>,
    num_tokens: usize,
    /// Token ID (u64 hash) → graph node index
    token_to_node: Vec<(u64, u16)>,
}

impl PriceGraph {
    pub fn new() -> Self {
        Self {
            edges: Vec::with_capacity(MAX_EDGES),
            num_tokens: 0,
            token_to_node: Vec::with_capacity(MAX_TOKENS),
        }
    }

    /// Get or create a node index for a token.
    pub fn get_or_create_node(&mut self, token_id: u64) -> u16 {
        // Linear scan — typically < 200 tokens
        for &(id, idx) in &self.token_to_node {
            if id == token_id {
                return idx;
            }
        }
        let idx = self.num_tokens as u16;
        self.token_to_node.push((token_id, idx));
        self.num_tokens += 1;
        idx
    }

    /// Add a directed edge (swap route) to the graph.
    /// `rate` is the effective exchange rate (output/input after fees).
    /// If rate > 1.0, the weight is negative (profitable direction).
    pub fn add_edge(
        &mut self,
        src_token: u64,
        dst_token: u64,
        pool_idx: u16,
        rate: f64,
        x_to_y: bool,
    ) {
        if rate <= 0.0 {
            return;
        }
        let src = self.get_or_create_node(src_token);
        let dst = self.get_or_create_node(dst_token);
        // weight = -log2(rate) in Q16.16
        let weight = (-rate.log2() * Q16_SCALE) as FixedWeight;
        self.edges.push(PriceEdge {
            src,
            dst,
            pool_idx,
            weight,
            x_to_y,
        });
    }

    /// Clear all edges (called before rebuilding from fresh state).
    pub fn clear(&mut self) {
        self.edges.clear();
        self.num_tokens = 0;
        self.token_to_node.clear();
    }

    /// Build the graph from a StateShadow.
    /// Adds two directed edges per pool (x→y and y→x).
    pub fn build_from_shadow(&mut self, shadow: &super::state_shadow::StateShadow) {
        self.clear();
        for i in 0..shadow.pool_count {
            if shadow.active[i] == 0 {
                continue;
            }
            let token_x = shadow.token_x_id[i];
            let token_y = shadow.token_y_id[i];
            let rx = shadow.spec_reserve_x[i] as f64;
            let ry = shadow.spec_reserve_y[i] as f64;
            if rx == 0.0 || ry == 0.0 {
                continue;
            }
            let fee = shadow.fee_num[i] as f64 / shadow.fee_den[i] as f64;
            let fee_complement = 1.0 - fee;

            // x→y rate: (ry / rx) * (1 - fee)
            let rate_xy = (ry / rx) * fee_complement;
            // y→x rate: (rx / ry) * (1 - fee)
            let rate_yx = (rx / ry) * fee_complement;

            self.add_edge(token_x, token_y, i as u16, rate_xy, true);
            self.add_edge(token_y, token_x, i as u16, rate_yx, false);
        }
    }

    /// Run Bellman-Ford to detect all negative cycles (arbitrage opportunities).
    ///
    /// Returns cycles sorted by profitability (most profitable first).
    /// The algorithm:
    /// 1. Initialize distance[source] = 0, all others = +INF
    /// 2. Relax all edges V-1 times
    /// 3. One more pass: any edge that can still be relaxed reveals a negative cycle
    /// 4. Trace back through predecessor array to reconstruct the cycle
    pub fn find_arbitrage_cycles(&self, max_hops: usize) -> Vec<ArbCycle> {
        if self.num_tokens == 0 || self.edges.is_empty() {
            return Vec::new();
        }

        let n = self.num_tokens;
        let mut cycles = Vec::new();

        // Run Bellman-Ford from each token as source
        // (In practice, we only need to run from tokens with high liquidity)
        for source in 0..n.min(MAX_TOKENS) {
            let mut dist = vec![i32::MAX / 2; n];
            let mut pred = vec![(u16::MAX, u16::MAX, false); n]; // (prev_node, pool_idx, direction)
            dist[source] = 0;

            // Relax edges V-1 times (or max_hops if smaller)
            let iterations = (n - 1).min(max_hops);
            for _ in 0..iterations {
                let mut updated = false;
                for edge in &self.edges {
                    let u = edge.src as usize;
                    let v = edge.dst as usize;
                    if dist[u] < i32::MAX / 2 {
                        let new_dist = dist[u].saturating_add(edge.weight);
                        if new_dist < dist[v] {
                            dist[v] = new_dist;
                            pred[v] = (edge.src, edge.pool_idx, edge.x_to_y);
                            updated = true;
                        }
                    }
                }
                if !updated {
                    break; // No more relaxation possible
                }
            }

            // Detection pass: find edges that can still be relaxed
            for edge in &self.edges {
                let u = edge.src as usize;
                let v = edge.dst as usize;
                if dist[u] < i32::MAX / 2 {
                    let new_dist = dist[u].saturating_add(edge.weight);
                    if new_dist < dist[v] {
                        // Found negative cycle — reconstruct it
                        if let Some(cycle) = self.reconstruct_cycle(v, &pred, max_hops) {
                            // Deduplicate: skip if we already found this cycle
                            let dominated = cycles.iter().any(|c: &ArbCycle| {
                                c.path.len() == cycle.path.len()
                                    && c.pool_indices == cycle.pool_indices
                            });
                            if !dominated {
                                cycles.push(cycle);
                            }
                        }
                    }
                }
            }
        }

        // Sort by profitability (most negative total_weight first)
        cycles.sort_by_key(|c| c.total_weight);
        cycles
    }

    /// Reconstruct the negative cycle by walking predecessor chain.
    fn reconstruct_cycle(
        &self,
        start: usize,
        pred: &[(u16, u16, bool)],
        max_hops: usize,
    ) -> Option<ArbCycle> {
        let mut node = start;
        // Walk back V times to guarantee we land inside the cycle
        for _ in 0..self.num_tokens {
            if pred[node].0 == u16::MAX {
                return None;
            }
            node = pred[node].0 as usize;
        }

        // Now `node` is guaranteed to be in the cycle — collect nodes
        let cycle_start = node;
        let mut raw_path = Vec::new();
        loop {
            if raw_path.len() > max_hops {
                return None;
            }
            raw_path.push(node as u16);
            node = pred[node].0 as usize;
            if node == cycle_start {
                break;
            }
        }

        if raw_path.len() < 2 {
            return None;
        }

        // raw_path walks backwards through the cycle: [C, B, A] for A→B→C→A
        raw_path.reverse(); // Now forward: [A, B, C]

        // Look up the correct edge (pool + direction) for each consecutive pair
        let mut pool_indices = Vec::with_capacity(raw_path.len());
        let mut directions = Vec::with_capacity(raw_path.len());
        let mut total_weight: FixedWeight = 0;

        for i in 0..raw_path.len() {
            let src = raw_path[i];
            let dst = raw_path[(i + 1) % raw_path.len()];
            // Find the best (lowest weight) edge from src to dst
            let mut best_edge: Option<&PriceEdge> = None;
            for edge in &self.edges {
                if edge.src == src && edge.dst == dst {
                    if best_edge.is_none() || edge.weight < best_edge.unwrap().weight {
                        best_edge = Some(edge);
                    }
                }
            }
            match best_edge {
                Some(edge) => {
                    pool_indices.push(edge.pool_idx);
                    directions.push(edge.x_to_y);
                    total_weight = total_weight.saturating_add(edge.weight);
                }
                None => return None, // Broken path
            }
        }

        // Profit multiplier: 2^(-total_weight / Q16_SCALE)
        let profit_multiplier = 2.0_f64.powf(-(total_weight as f64) / Q16_SCALE);

        Some(ArbCycle {
            path: raw_path,
            pool_indices,
            directions,
            total_weight,
            profit_multiplier,
        })
    }

    /// Compute the sum of edge weights around a cycle.
    fn compute_cycle_weight(&self, path: &[u16], pool_indices: &[u16]) -> FixedWeight {
        let mut total: FixedWeight = 0;
        for i in 0..path.len() {
            let src = path[i];
            let dst = path[(i + 1) % path.len()];
            let pool = pool_indices[i];
            // Find the matching edge
            for edge in &self.edges {
                if edge.src == src && edge.dst == dst && edge.pool_idx == pool {
                    total = total.saturating_add(edge.weight);
                    break;
                }
            }
        }
        total
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    pub fn token_count(&self) -> usize {
        self.num_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_construction() {
        let mut g = PriceGraph::new();
        // Token A = 1, Token B = 2, Token C = 3
        // A→B at rate 1.5 (profitable)
        g.add_edge(1, 2, 0, 1.5, true);
        // B→C at rate 0.8
        g.add_edge(2, 3, 1, 0.8, true);
        // C→A at rate 0.9
        g.add_edge(3, 1, 2, 0.9, true);

        assert_eq!(g.token_count(), 3);
        assert_eq!(g.edge_count(), 3);
    }

    #[test]
    fn test_negative_cycle_detection() {
        let mut g = PriceGraph::new();
        // Create a profitable 3-hop cycle: A→B→C→A
        // Product of rates: 1.01 * 1.01 * 1.01 = 1.0303 > 1.0 (profitable)
        g.add_edge(1, 2, 0, 1.01, true);
        g.add_edge(2, 3, 1, 1.01, true);
        g.add_edge(3, 1, 2, 1.01, true);

        let cycles = g.find_arbitrage_cycles(5);
        assert!(!cycles.is_empty(), "Should detect profitable cycle");
        assert!(
            cycles[0].profit_multiplier > 1.0,
            "Profit multiplier should exceed 1.0, got {}",
            cycles[0].profit_multiplier
        );
    }

    #[test]
    fn test_no_arb_when_rates_balanced() {
        let mut g = PriceGraph::new();
        // Balanced cycle: A→B→C→A with product exactly 1.0
        // Actually slightly less due to fees approximation
        g.add_edge(1, 2, 0, 0.99, true);
        g.add_edge(2, 3, 1, 0.99, true);
        g.add_edge(3, 1, 2, 0.99, true);

        let cycles = g.find_arbitrage_cycles(5);
        // Product = 0.99^3 = 0.970 < 1.0, no profitable cycle
        let profitable: Vec<_> = cycles.iter().filter(|c| c.profit_multiplier > 1.0).collect();
        assert!(profitable.is_empty(), "Should not find profitable cycle");
    }

    #[test]
    fn test_build_from_shadow() {
        let mut shadow = super::super::state_shadow::StateShadow::new();
        // Pool 0: A/B with rate imbalance
        shadow.register_pool(
            [0x01; 32], 1, 2,
            1_000_000, 2_100_000, // A/B = 2.1 rate
            3, 1000, 100,
        );
        // Pool 1: B/C
        shadow.register_pool(
            [0x02; 32], 2, 3,
            1_000_000, 1_000_000,
            3, 1000, 100,
        );
        // Pool 2: C/A
        shadow.register_pool(
            [0x03; 32], 3, 1,
            1_000_000, 500_000, // C/A = 0.5
            3, 1000, 100,
        );

        let mut g = PriceGraph::new();
        g.build_from_shadow(&shadow);

        // 3 pools × 2 directions = 6 edges
        assert_eq!(g.edge_count(), 6);
        assert_eq!(g.token_count(), 3);
    }
}
