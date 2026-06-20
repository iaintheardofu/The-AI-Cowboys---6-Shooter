# Technical Omnibus — Yield Daemon v0.2.0

## Comprehensive Engineering Reference

---

## Table of Contents

1. [Theoretical Foundations](#1-theoretical-foundations)
2. [Hardware-Aware Algorithm Design](#2-hardware-aware-algorithm-design)
3. [Cryptographic Acceleration](#3-cryptographic-acceleration)
4. [MEV Extraction Theory](#4-mev-extraction-theory)
5. [Distributed ML Inference](#5-distributed-ml-inference)
6. [Memory Architecture](#6-memory-architecture)
7. [Compilation and Profiling](#7-compilation-and-profiling)
8. [Network Protocol Engineering](#8-network-protocol-engineering)
9. [Live Execution Layer](#9-live-execution-layer)
10. [Treasury and Off-Ramp Pipeline](#10-treasury-and-off-ramp-pipeline)
11. [Risk and Governance](#11-risk-and-governance)
12. [Orchestrator Integration](#12-orchestrator-integration)
13. [Benchmarking Methodology](#13-benchmarking-methodology)
14. [Deployment Architecture](#14-deployment-architecture)

---

## 1. Theoretical Foundations

### 1.1 Complexity Models Beyond Asymptotic Analysis

Traditional O(n log n) vs O(n²) analysis assumes uniform memory access cost. Modern hardware violates this assumption by 100x between L1 cache (1-2ns) and RAM (60-100ns). The Yield Daemon employs the **External Memory Model** (Aggarwal & Vitter, 1988) and **Cache-Oblivious Model** (Frigo et al., 1999) to reason about algorithm performance on real hardware.

**Key insight:** An O(n²) algorithm with perfect cache utilization can outperform an O(n log n) algorithm with random memory access by 10-50x for practical input sizes (n < 10⁶).

### 1.2 Throughput Computing vs Latency Computing

The three yield domains have different computational profiles:

| Domain | Model | Bottleneck | Optimization Target |
|--------|-------|-----------|-------------------|
| ZK Prover | Throughput | NTT butterfly ops/sec | SIMD width × IPC |
| MEV Arbitrage | Latency | Detection→submission time | Branch-free critical path |
| ML Inference | Bandwidth | GEMM memory throughput | Roofline model (FLOPS/byte) |

### 1.3 The Roofline Model

For ML inference, performance is bounded by:

```
Attainable FLOPS = min(Peak FLOPS, Operational Intensity × Memory Bandwidth)
```

Where `Operational Intensity = FLOPS / Bytes transferred`. Our tiled GEMM achieves high operational intensity by keeping 64×64 tiles (16KB) resident in L1 cache, converting memory-bound operations into compute-bound ones.

### 1.4 Adversarial Game Theory

MEV extraction is a zero-sum game against other bots. Profitability requires being in the top percentile of latency AND correctness. The system employs:

- **First-price sealed-bid auctions** (Jito bundle tips) — optimal bid = value × (n-1)/n where n = number of competitors
- **Combinatorial auction theory** (Intent solving) — NP-hard in general, greedy approximation achieves (1 - 1/e) of optimal surplus
- **Mechanism design** — truth-revealing mechanisms for CoW matching ensure solver incentive alignment

---

## 2. Hardware-Aware Algorithm Design

### 2.1 Pipeline Hazards and Branch Prediction

Modern x86-64 processors have 15-25 stage pipelines. A branch misprediction flushes the entire pipeline, costing 15-20 clock cycles. At 4GHz, that's 5 nanoseconds — enough time for 20 arithmetic operations.

**Our approach: Branchless everything on the hot path.**

```rust
// Conditional select without branching (from montgomery.rs):
fn conditional_select(a: &Self, b: &Self, flag: u64) -> Self {
    let mask = 0u64.wrapping_sub(flag); // 0x0..0 or 0xF..F
    Self {
        limbs: [
            a.limbs[0] ^ (mask & (a.limbs[0] ^ b.limbs[0])),
            a.limbs[1] ^ (mask & (a.limbs[1] ^ b.limbs[1])),
            // ...
        ],
    }
}
```

The compiler emits `cmov` instructions instead of `jmp`, eliminating speculation entirely.

### 2.2 Instruction-Level Parallelism (ILP)

Modern CPUs can retire 4-6 instructions per cycle if there are no data dependencies. We structure algorithms to maximize independent instruction chains:

```
// NTT butterfly — two independent multiplications can execute in parallel:
u = coeffs[k + j]                    // Load 1
v = coeffs[k + j + stage_len]       // Load 2 (independent)
twiddle = roots[j * twiddle_step]   // Load 3 (independent)
v_mul = v.mont_mul(twiddle)         // Depends on v and twiddle
result_add = u.mont_add(v_mul)      // Depends on u and v_mul
result_sub = u.mont_sub(v_mul)      // Depends on u and v_mul (parallel with add)
```

### 2.3 SIMD Vectorization

AVX2 provides 256-bit registers (4× u64), AVX-512 provides 512-bit registers (8× u64). Our Montgomery multiplication pipeline can process 4 independent field multiplications simultaneously:

| ISA | Register Width | u64 Lanes | Throughput Multiplier |
|-----|---------------|-----------|----------------------|
| Scalar | 64-bit | 1 | 1× |
| SSE4.2 | 128-bit | 2 | ~1.8× |
| AVX2 | 256-bit | 4 | ~3.5× |
| AVX-512 | 512-bit | 8 | ~6× |
| AVX-512 IFMA | 512-bit (52-bit lanes) | 8 | ~8× for crypto |

AVX-512 IFMA (Integer Fused Multiply-Add) is specifically designed for cryptography — it provides 52×52→104 bit multiplication, perfectly matching the limb sizes needed for prime field arithmetic.

### 2.4 Cache Associativity and Stride Analysis

x86-64 L1 caches are typically 8-way set-associative with 64-byte lines. If data arrays have a power-of-2 stride matching the cache size, different arrays can map to the same cache set, causing **conflict misses** (thrashing).

Our SoA order book (`cache.rs`) uses prime-offset padding to avoid stride conflicts:

```rust
#[repr(align(64))]  // Align to cache line
pub struct SoAOrderBook {
    pub bid_prices: Vec<u64>,   // Contiguous, sequential access
    pub bid_volumes: Vec<u64>,  // Different cache set (different base address)
    pub ask_prices: Vec<u64>,
    pub ask_volumes: Vec<u64>,
    // Each array starts at a different cache line boundary
}
```

### 2.5 TLB and Huge Pages

The Translation Lookaside Buffer (TLB) maps virtual→physical addresses. Standard 4KB pages mean the TLB can cover 4KB × 512 entries = 2MB of memory. For our 64MB arena, this causes TLB misses.

**Mitigation:** On Linux, configure 2MB huge pages:
```bash
echo 64 > /proc/sys/vm/nr_hugepages  # Pre-allocate 64 × 2MB = 128MB
```

The arena allocator can be backed by `mmap` with `MAP_HUGETLB` to reduce TLB pressure by 512×.

---

## 3. Cryptographic Acceleration

### 3.1 Montgomery Multiplication — The Foundation

**Problem:** Computing `a × b mod p` requires division by `p`, costing 20-80 CPU cycles.

**Solution:** Montgomery's algorithm (1985) maps numbers into a domain where reduction is achieved by division by `R = 2^256` (a free bit shift) instead of division by `p`.

**Mapping:**
- To Montgomery: `ā = a × R mod p`
- Multiplication: `ā × b̄ × R⁻¹ mod p = (a×b) × R mod p = ab̄`
- From Montgomery: `ā × 1 × R⁻¹ mod p = a`

**Our CIOS Implementation (montgomery.rs):**

The Coarsely Integrated Operand Scanning algorithm processes one 64-bit limb of the multiplier per iteration, interleaving multiplication and reduction:

```
For each limb b[i]:
  1. t += a × b[i]           // Multiply-accumulate
  2. m = t[0] × (-p⁻¹) mod 2⁶⁴  // Compute reduction factor
  3. t += m × p              // Add multiple of p (makes t[0] = 0 mod 2⁶⁴)
  4. t >>= 64                // Shift (division by 2⁶⁴ — free!)
```

After 4 iterations (for 256-bit numbers), the result is in [0, 2p), requiring at most one conditional subtraction.

**Performance:** ~50ns per multiplication on modern x86-64 (scalar). With AVX-512 IFMA: ~8ns.

### 3.2 Number-Theoretic Transform (NTT)

The NTT is the FFT over a finite field GF(p) instead of the complex numbers. It computes polynomial evaluation at the n-th roots of unity ω^0, ω^1, ..., ω^{n-1} where ω^n ≡ 1 (mod p).

**Our implementation (ntt.rs):**

1. **Precomputed twiddle factors** — All roots of unity stored in Montgomery form, eliminating per-butterfly conversion overhead

2. **Cooley-Tukey radix-2 DIT** — In-place with bit-reversal permutation:
   ```
   For log₂(n) stages:
     For each butterfly pair (u, v):
       u' = u + ω·v    (Montgomery add + mul)
       v' = u - ω·v    (Montgomery sub)
   ```

3. **Memory access pattern:** The bit-reversal permutation ensures that butterfly partners are adjacent in early stages (L1 cache hits) and separated by large strides in later stages (L2/L3). This is the optimal pattern for cache hierarchy utilization.

**Complexity:** O(n log n) field operations, where each field operation is O(1) via Montgomery multiplication.

**BN254 Field:** Supports NTT up to size 2^28 (two-adicity of the scalar field).

### 3.3 Multi-Scalar Multiplication (MSM) via Pippenger

**Problem:** Compute Σᵢ sᵢ·Gᵢ for N scalar-point pairs on an elliptic curve.

**Naive approach:** N × 256 point additions = O(256N) group operations.

**Pippenger's method (msm.rs):**

1. Split each 256-bit scalar into 256/w windows of w bits
2. For each window, distribute points into 2^w buckets based on the scalar window value
3. Aggregate buckets using running sums (avoids expensive individual multiplications)
4. Combine windows with repeated doubling

**Complexity:** O(N/w + 2^w × 256/w) group operations.
**Optimal window:** w ≈ log₂(N)/2 ≈ 8-10 for typical MSM sizes (N = 2^16 to 2^20).

**Point representation:** We use Jacobian projective coordinates (X, Y, Z) where the affine point is (X/Z², Y/Z³). This avoids field inversions during point addition (inversions cost ~50 multiplications via Fermat's little theorem).

**Mixed addition optimization:** When adding an affine point (Z=1) to a projective point, we save 4 field multiplications per addition — critical since MSM performs millions of additions.

### 3.4 Auxiliary Finite Field Algorithms

| Algorithm | File | Purpose | Complexity |
|-----------|------|---------|-----------|
| Extended GCD | fields.rs | Modular inverse for small fields | O(log min(a,b)) |
| Binary GCD (Stein) | fields.rs | Division-free GCD, branchless-friendly | O(log²(n)) |
| Barrett Reduction | fields.rs | Precomputed-reciprocal mod reduction | O(1) amortized |
| Fast Modular Exp | fields.rs | Square-and-multiply with Barrett | O(log exp) |
| Legendre Symbol | fields.rs | Quadratic residue testing | O(log p) |
| Tonelli-Shanks | fields.rs | Modular square root | O(log²(p)) |

---

## 4. MEV Extraction Theory

### 4.1 Atomic Arbitrage

Given two AMM pools A and B with reserves (x_A, y_A) and (x_B, y_B) trading the same pair:

**Constant-product invariant:** x × y = k (after fees)

**Arbitrage exists when:** `price_A = y_A/x_A ≠ y_B/x_B = price_B`

**Optimal input (binary search in cache.rs):**
```
For input amount Δx:
  out_B = pool_A.swap_x_to_y(Δx)     // Buy Y on pool A
  out_A = pool_B.swap_y_to_x(out_B)   // Sell Y on pool B
  profit = out_A - Δx                 // Must be > 0 + gas
```

The profit function is concave — binary search finds the global maximum in O(log(max_input)) iterations.

### 4.2 Concentrated Liquidity Swaps

Concentrated liquidity (Uniswap V3 model) provides different liquidity at different price ranges (ticks). A swap may cross multiple tick boundaries, each requiring:

1. Compute swap amount within current tick range
2. Update liquidity when crossing a tick (add/remove the tick's net liquidity)
3. Continue with remaining input in the next range

**Our implementation (amm.rs, ConcentratedPool::swap):**
- Q64.64 fixed-point sqrt(price) representation
- Integer-only tick-to-price conversion
- Monotonic tick traversal (no backtracking)

### 4.3 Intent-Based MEV (CoW Protocol)

Instead of users submitting individual transactions, they sign **intents** ("I want to sell X for Y at this price or better"). Solvers compete to find the best execution:

1. **Coincidence of Wants (CoW):** Direct peer-to-peer matching — Alice sells what Bob wants to buy, no DEX needed
2. **Excess routing:** Unmatched amounts are routed through DEX liquidity
3. **Batch auction:** All intents processed atomically — no front-running possible

**Our solver (solver.rs):**
- Greedy matching: O(n²) for n intents per token pair
- Surplus maximization: solver earns 10% of surplus above user minimums
- In production: extends to Vehicle Routing Problem (VRP) formulation for multi-hop optimization

### 4.4 Latency Budget Analysis

```
Total Budget: 200ms (configurable)

Mempool → Detection:     10-50ms (depends on network propagation)
Transaction Parsing:     <0.1ms  (SIMD JSON parser, pre-computed discriminators)
Arbitrage Scanning:      <1ms    (SoA order book, branchless comparisons)
Optimal Input Search:    <0.5ms  (binary search, ~40 iterations max)
Bundle Construction:     <0.5ms  (pre-allocated templates, no heap alloc)
Bundle Submission:       50-100ms (network round-trip to Jito block engine)
─────────────────────────────────
Total:                   ~60-150ms (within budget)
```

### 4.5 Multi-Hop Arbitrage via Bellman-Ford

For N tokens and M pools, construct a weighted directed graph:
- Node = token
- Edge weight = -log(exchange_rate × (1 - fee))
- **Negative cycle = profitable arbitrage loop**

Bellman-Ford detects negative cycles in O(N × M) time. We run N-1 relaxation iterations, then check for further relaxation — any improvement indicates a negative cycle.

### 4.6 ASTE — Atomic State Transition Engine

The ASTE is the culmination of every HPC principle in this codebase, composed into a single searcher hot loop. It is not a standard "bot" — it is a hardware-optimized state-machine predator.

#### 4.6.1 State Shadow — Cache-Aligned Local Blockchain Replica

The `StateShadow` maintains a SoA (Structure-of-Arrays) mirror of all on-chain pool state with the following memory layout:

```
reserve_x[4096]:    u128 × 4096 = 64KB  ← contiguous, prefetch-friendly
reserve_y[4096]:    u128 × 4096 = 64KB  ← sequential access = L1 streaming
fee_num[4096]:      u32  × 4096 = 16KB
fee_den[4096]:      u32  × 4096 = 16KB
token_x_id[4096]:   u64  × 4096 = 32KB
token_y_id[4096]:   u64  × 4096 = 32KB
price_x_per_y[4096]:u64  × 4096 = 32KB  ← Q32.32 fixed-point
spec_reserve_x[4096]: u128 × 4096 = 64KB ← speculative overlay
spec_reserve_y[4096]: u128 × 4096 = 64KB
Total: ~384KB (fits entirely in L2 cache)
```

**Speculative Overlay:** When a pending mempool swap is detected, the ASTE doesn't wait for block confirmation. It applies the swap's effect to the speculative reserves, then immediately searches for arbitrage on the predicted post-swap state. When a new block arrives, all speculative state is reverted in O(1) via bulk copy.

**Price representation:** Q32.32 fixed-point (32 integer bits, 32 fractional bits). This provides 9 decimal digits of precision without any floating-point instruction — the CPU uses integer ALU only, which is fully pipelined at 1 op/cycle.

#### 4.6.2 Price Graph — Negative Cycle Detection

The `PriceGraph` models the DEX ecosystem as a directed weighted graph:

```
Graph G = (V, E) where:
  V = {token_1, token_2, ..., token_n}        (unique tokens)
  E = {(t_i, t_j, pool_k, w_ij)}             (swap routes)
  w_ij = -log₂(rate_ij) in Q16.16 fixed-point
```

A negative-weight cycle C in G means: `Σ w_ij < 0` for edges (i,j) ∈ C, which implies `Π rate_ij > 1.0` — the product of exchange rates around the cycle exceeds 1.0 (profitable arbitrage).

**Algorithm (modified Bellman-Ford):**
1. For each source token s ∈ V:
   - Initialize dist[s] = 0, dist[v] = +∞ for v ≠ s
   - For i = 1 to min(|V|-1, max_hops):
     - For each edge (u, v, w): if dist[u] + w < dist[v], update dist[v]
   - Detection pass: any edge (u,v,w) where dist[u] + w < dist[v] reveals a cycle
2. Reconstruct cycle by walking the predecessor chain

**Fixed-point arithmetic:** Edge weights use Q16.16 (16 integer + 16 fractional bits). This is critical:
- `log₂(1.01) ≈ 0.01439` → Q16.16 = 943 (integer representable)
- `log₂(0.997) ≈ -0.00433` → Q16.16 = -284
- Cycle weight = Σ weights; negative sum = profitable
- Avoids all `f64` operations in the hot path

#### 4.6.3 SIMD Solver — Vectorized Profit Evaluation

The `SimdSolver` evaluates detected arbitrage cycles against the current state shadow:

**Cycle simulation:** Chain swap computations through N pools:
```
amount = input
for pool_idx, direction in cycle:
    amount = shadow.sim_swap(pool_idx, amount, direction)
profit = amount - input
```

**Optimal input search (Ternary Search):**
The profit function P(x) = simulate(x) - x is concave for constant-product AMMs. Ternary search finds the peak in O(log₃(max_input/precision)) iterations:

```
lo, hi = 1, max_input
for _ in 0..40:
    m1 = lo + (hi - lo) / 3
    m2 = hi - (hi - lo) / 3
    p1 = simulate(m1) - m1
    p2 = simulate(m2) - m2
    // Branchless narrowing:
    let mask = (p1 < p2) as u128;
    lo = mask * m1 + (1 - mask) * lo;
    hi = mask * hi + (1 - mask) * m2;
```

The inner narrowing is branchless — `(p1 < p2) as u128` compiles to a single `setb` instruction, and the subsequent arithmetic uses only `imul`/`add` with no control flow.

**4-wide ILP unrolling:** Cycles are processed in groups of 4 to saturate the CPU's superscalar execution units. Each cycle evaluation is independent — the CPU can execute 4 ternary searches simultaneously using different ALU ports.

**Spread scanning:** For direct 2-pool arbitrage, `scan_best_spread()` uses manually unrolled 4-element chunks:
```
for c in 0..(n/4):
    a0, a1, a2, a3 = asks[4c..4c+4]     // 4 loads (independent)
    b0, b1, b2, b3 = bids[4c..4c+4]     // 4 loads (independent)
    // 8 independent comparisons — fills all ALU ports
```

With `-C target-cpu=native`, LLVM auto-vectorizes this to `vmovdqu` + `vpcmpq` (AVX2) or `vpmovq2m` (AVX-512).

#### 4.6.4 ASTE Hot Loop

The complete pipeline per mempool event:

```
EVENT ARRIVES (via lock-free mpsc channel)
    │
    ├─ PendingSwap:
    │   ├── [SIMULATE] shadow.apply_speculative_swap()     ~50ns
    │   ├── [GRAPH]    graph.build_from_shadow()           ~100μs (every 50 cycles)
    │   ├── [DETECT]   graph.find_arbitrage_cycles()       ~500μs
    │   ├── [EVALUATE] solver.evaluate_batch()             ~200μs
    │   ├── [EXECUTE]  bundler.construct() + submit()      ~100μs + network
    │   └── [RESET]    arena.reset()                       ~1ns
    │
    ├─ NewBlock:
    │   ├── shadow.on_new_block(slot)                      ~10μs (bulk copy)
    │   └── graph.build_from_shadow()                      ~100μs
    │
    └─ PoolUpdate:
        └── shadow.update_reserves()                       ~10ns
```

**Arena lifecycle:** The 16MB arena allocator is reset to zero (single atomic store) at the end of every cycle. This means the ASTE never calls `malloc` or `free` during operation — all temporary allocations (graph nodes, cycle paths, solver results) live in the arena and die instantly.

---

## 5. Distributed ML Inference

### 5.1 Tiled Matrix Multiplication

Naive GEMM: O(n³) with O(n²) cache misses.
Tiled GEMM: O(n³) with O(n³ / (B × √M)) cache misses, where B = tile size, M = cache size.

**Our tiling strategy (inference.rs):**
```
Tile size: 64 × 64 floats = 16KB
L1 cache: 32KB (fits 2 tiles for simultaneous multiply)
L2 cache: 256KB (fits 16 tiles for outer loop blocking)

Loop order: ii (M-tiles) → jj (N-tiles) → kk (K-tiles) → micro-kernel
```

This achieves ~80% of theoretical peak FLOPS on CPU, compared to ~5% for naive triple-nested loops.

### 5.2 Attention Mechanism Optimization

Scaled dot-product attention: `softmax(QK^T / √d) · V`

- **QK^T:** (seq_len × d_model) × (d_model × seq_len) — compute via tiled GEMM
- **Softmax:** Numerically stable with max subtraction: `exp(x - max(x)) / Σexp(x - max(x))`
- **Attention × V:** Second GEMM, (seq_len × seq_len) × (seq_len × d_model)

For long sequences, this is memory-bound (O(seq_len²) attention matrix). Flash Attention (Dao et al., 2022) tiles the computation to avoid materializing the full attention matrix — a planned optimization.

### 5.3 Bittensor Reward Mechanics

Bittensor uses a **superlinear scoring curve**:
```
reward ∝ (quality × speed)^α  where α > 1
```

This means:
- A miner 2× faster earns >2× more (exponential advantage)
- A miner 10% slower earns significantly less (steep penalty)
- Only the top-performing miners remain registered (deregistration for poor performance)

**Our strategy:** Maximize throughput via:
1. Lower precision (FP16/FP8) for higher FLOPS/watt
2. Batch inference (amortize overhead across multiple requests)
3. Background training (continuous model improvement during idle GPU cycles)

### 5.4 Quantization and Mixed Precision

| Type | Bits | Range | GEMM Throughput (H100) |
|------|------|-------|----------------------|
| FP32 | 32 | ±3.4×10³⁸ | 67 TFLOPS |
| FP16 | 16 | ±65,504 | 990 TFLOPS |
| BF16 | 16 | ±3.4×10³⁸ | 990 TFLOPS |
| FP8 (E4M3) | 8 | ±448 | 1,979 TFLOPS |
| INT4 | 4 | [-8, 7] | ~3,958 TOPS |

Our inference engine supports runtime precision selection. The optimal choice depends on the subnet's quality/speed tradeoff — some subnets reward accuracy heavily (use FP16), others reward speed (use FP8/INT4).

---

## 6. Memory Architecture

### 6.1 Arena Allocator Design

```
┌───────────────────────────────────────────────────┐
│                  64MB Arena Buffer                  │
│ [  used  ][  used  ][  used  ][ ← cursor ][  free ]│
│ ↑                              ↑                    │
│ base                           atomic cursor        │
└───────────────────────────────────────────────────┘

Allocate: CAS(cursor, cursor + align(size))  // O(1), lock-free
Reset:    store(cursor, 0)                    // O(1), instant
Free:     N/A (bulk free via reset)
```

**Key properties:**
- **No fragmentation** — contiguous bump allocation
- **No free lists** — O(1) reset deallocates everything
- **Lock-free** — atomic compare-and-swap for concurrent threads
- **Cache-friendly** — sequential allocation = sequential access

### 6.2 Structure of Arrays (SoA)

**Traditional Array of Structures (AoS):**
```
[{bid:100, vol:10, ask:105, vol:10}, {bid:102, vol:20, ask:103, vol:20}, ...]
```
When scanning only bid prices, we load entire structures (4 fields × 8 bytes = 32 bytes) but only use 8 bytes — 75% wasted bandwidth.

**Our SoA layout:**
```
bid_prices:  [100, 102, 99, ...]     // Contiguous
bid_volumes: [10,  20,  15, ...]     // Contiguous
ask_prices:  [105, 103, 108, ...]    // Contiguous
ask_volumes: [10,  20,  15, ...]     // Contiguous
```
When scanning bid prices, every byte loaded is useful. SIMD can process 4 prices per instruction (AVX2) or 8 (AVX-512).

### 6.3 Bloom Filter for O(1) Program ID Matching

The mempool subscriber receives thousands of transactions per second. Each must be checked against a target list of AMM program IDs. Linear search over 3-10 IDs is cheap, but a Bloom filter provides **guaranteed O(1)** with zero branches:

```
Insert: For k hash functions, set bits[h₁(x)], bits[h₂(x)], ..., bits[hₖ(x)]
Query:  Return AND(bits[h₁(x)], bits[h₂(x)], ..., bits[hₖ(x)])
```

**Our implementation (p2p.rs):**
- False positive rate: 1% at 100 items
- No false negatives (guaranteed by construction)
- Zero branches on query path — bitwise AND of k lookups

---

## 7. Compilation and Profiling

### 7.1 Compiler Optimization Flags

```toml
[profile.release]
opt-level = 3          # Maximum optimization
lto = "fat"            # Link-Time Optimization across all crates
codegen-units = 1      # Single codegen unit for maximum inlining
panic = "abort"        # No unwinding = smaller binary, faster panics
strip = true           # Remove debug symbols
```

**`lto = "fat"`** enables whole-program optimization — the compiler can inline functions across crate boundaries, eliminating call overhead for hot functions like `mont_mul`.

**`codegen-units = 1`** forces the compiler to optimize the entire crate as a single unit, enabling optimizations that span multiple modules.

### 7.2 Target-Specific Code Generation

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

This instructs LLVM to generate instructions for the exact CPU model on the build machine, unlocking:
- AVX-512 on Intel Xeon / AMD EPYC
- NEON on ARM (Apple Silicon, Graviton)
- Architecture-specific instruction scheduling

### 7.3 Profiling Pipeline

```bash
# 1. Compile with frame pointers for profiling
CARGO_PROFILE_RELEASE_DEBUG=true cargo build --release

# 2. Record CPU events
perf record -g ./target/release/yield-daemon --dry-run

# 3. Analyze hotspots
perf report --hierarchy

# 4. Micro-architectural analysis
perf stat -d ./target/release/yield-daemon --dry-run
```

Key metrics to monitor:
- **IPC (Instructions Per Cycle):** >3.0 = good utilization
- **Branch miss rate:** <1% on hot path
- **L1 cache miss rate:** <5% on hot path
- **TLB miss rate:** Should be near zero with huge pages

### 7.4 Criterion Benchmarks

```bash
cargo bench
```

Benchmarks include:
- NTT forward transform at sizes 2^8 through 2^16
- Montgomery multiplication and squaring
- Arena allocation throughput (1000× 64-byte allocs)
- Arena reset latency

---

## 8. Network Protocol Engineering

### 8.1 RPC Client with Latency-Weighted Routing

Multiple RPC endpoints are maintained simultaneously with exponential moving average (EMA) latency tracking:

```
EMA_new = (EMA_old × 7 + latency_measured) / 8
```

Endpoint selection: always route to the lowest-EMA endpoint. Cold start: round-robin until sufficient samples.

### 8.2 WebSocket Mempool Subscription

For Solana MEV, we subscribe to:
1. **Geyser plugin** (gRPC stream of all account changes)
2. **logsSubscribe** (WebSocket subscription to program log mentions)
3. **Jito bundle API** (gRPC stream for bundle submission)

The subscription pipeline:
```
WS Connect → Subscribe(program_ids) → Parse Notification →
Bloom Filter Check → Decode Instruction → Channel.send(SwapInstruction)
```

### 8.3 Kernel-Bypass Networking (Future)

For institutional-grade latency (<1μs network stack), the system supports optional kernel-bypass via AF_XDP:

```
NIC → AF_XDP socket (userspace) → Direct buffer access
```

This bypasses the Linux TCP/IP stack entirely, eliminating:
- System call overhead (~1μs per syscall)
- Kernel buffer copies (~0.5μs)
- Socket processing overhead

Total savings: 2-5μs per packet — meaningful when total latency budget is 200μs.

---

## 9. Live Execution Layer

### 9.1 Ed25519 Transaction Signing

The daemon signs Solana transactions using `ed25519-dalek` (v2), a production-grade Rust implementation of RFC 8032. Key design decisions:

**No `solana-sdk` dependency.** The Solana SDK pulls in 200+ transitive dependencies and adds >50MB to compile time. We implement the wire format directly — 624 lines total for keypair loading, transaction serialization, signing, and Jito submission.

**Keypair loading (`net/solana.rs`):**
```
Solana CLI format: JSON array of 64 u8 values
  [0..32]  = ed25519 secret key (seed)
  [32..64] = ed25519 public key (derived)

Load → SigningKey::from_bytes(&secret)
     → Verify derived pubkey matches stored pubkey
     → Warn if mismatch (use derived)
```

**Transaction serialization — Solana wire format:**
```
[compact-u16: num_signatures]
[64 bytes × num_signatures]
[Message]:
  [u8: num_required_signers]
  [u8: num_readonly_signed]
  [u8: num_readonly_unsigned]
  [compact-u16: num_account_keys]
  [32 bytes × num_account_keys]
  [32 bytes: recent_blockhash]
  [compact-u16: num_instructions]
  [Instructions]:
    [u8: program_id_index]
    [compact-u16: num_account_indices]
    [u8 × num_account_indices]
    [compact-u16: data_length]
    [u8 × data_length]
```

**Compact-u16 encoding** (Solana's variable-length integer format):
- Values 0-127: 1 byte (high bit = 0)
- Values 128-16383: 2 bytes (first byte high bit = 1)
- Values 16384-65535: 3 bytes

**Base58 encode/decode** (Bitcoin/Solana variant, custom implementation):
- Leading zero bytes map to '1' characters
- Division-based encoding with alphabet `123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz`
- Decode via 128-byte lookup table (O(1) per character)

### 9.2 Jito Block Engine Integration

Jito provides MEV-protected transaction inclusion for Solana. Our `JitoClient` submits bundles via the REST API:

**Bundle format:**
```json
{
  "jsonrpc": "2.0",
  "method": "sendBundle",
  "params": [["<base58_tx_1>", "<base58_tx_2>"]]
}
```

**Tip account rotation:** Jito requires a tip transfer to one of 8 Program Derived Addresses (PDAs). We rotate based on `subsec_nanos % 8` for uniform distribution.

**Tip floor query:** GET `/api/v1/bundles/tip_floor` returns percentile tip data. We use the 50th percentile as the floor and add profit-proportional tips above it.

**Bundle lifecycle:**
1. Get latest blockhash via `getLatestBlockhash`
2. Build swap transaction (multi-hop, account-deduplicated)
3. Build tip transaction (SOL transfer to random Jito PDA)
4. Sign both with ed25519-dalek
5. Submit as atomic bundle
6. Poll `getBundleStatuses` for confirmation

### 9.3 Live MEV Executor

The `LiveExecutor` (`mev/executor.rs`) bridges ASTE arbitrage results to real blockchain transactions:

**Raydium V4 swap instruction layout (18 accounts):**
```
[0]  Token Program (SPL)
[1]  AMM ID
[2]  AMM Authority (PDA)
[3]  AMM Open Orders
[4]  AMM Target Orders
[5]  Pool Coin Token Account
[6]  Pool PC Token Account
[7]  Serum Program
[8]  Serum Market
[9]  Serum Bids
[10] Serum Asks
[11] Serum Event Queue
[12] Serum Coin Vault
[13] Serum PC Vault
[14] Serum Vault Signer
[15] User Source Token
[16] User Destination Token
[17] User Owner (signer)

Data: [discriminator: 8 bytes = 0x09...] [amount_in: u64 LE] [min_out: u64 LE]
```

**Account deduplication:** Multi-hop bundles reuse accounts across instructions. The executor deduplicates the account list and remaps instruction indices to avoid the 64-account transaction limit.

**AMM discriminators:**
| AMM | Discriminator (8 bytes) |
|-----|------------------------|
| Raydium V4 | `09 00 00 00 00 00 00 00` |
| Orca Whirlpool | `f8 c6 9e 91 e1 75 87 c8` (SHA256("global:swap")[..8]) |
| Meteora DLMM | `e4 45 a5 2e 51 cb 9a 1d` |

### 9.4 WebSocket Mempool Subscription

Real-time mempool monitoring via `tokio-tungstenite`:

```
Connect(wss://endpoint)
  → Send: logsSubscribe({mentions: [program_ids]}, {commitment: "processed"})
  → Recv: Log notifications containing:
    - signature (base58)
    - logs[] (array of "Program X invoke [N]" strings)
    - slot (u64)
  → Parse: Extract program IDs from "Program X invoke" lines
  → Filter: Bloom filter O(1) check against target AMM programs
  → Dispatch: MempoolTransaction → mpsc channel → ASTE event queue
```

**Auto-reconnect:** On WebSocket close or error, wait 2 seconds and reconnect. Each endpoint runs in its own `tokio::spawn` task.

### 9.5 Pool Discovery

The `PoolDiscovery` module fetches real AMM pool state from the Solana blockchain:

**Raydium V4 account data layout (key offsets):**
```
Offset 0:    status (u64)
Offset 32:   base_decimal (u64)
Offset 40:   quote_decimal (u64)
Offset 88:   coin_lot_size (u64)
Offset 96:   pc_lot_size (u64)
Offset 680:  pool_coin_token_account (Pubkey, 32 bytes)
Offset 712:  pool_pc_token_account (Pubkey, 32 bytes)
```

Data is fetched via `getAccountInfo` with base64 encoding, then decoded and parsed using custom base64 decoder (no external dependency).

---

## 10. Treasury and Off-Ramp Pipeline

### 10.1 Five-Level Architecture

The treasury converts on-chain yield to fiat bank deposits through five autonomous stages:

```
┌─────────────┐    ┌──────────────┐    ┌───────────────┐    ┌──────────┐    ┌─────────────┐
│  ACCUMULATE │───→│ CONSOLIDATE  │───→│ THRESHOLD GATE│───→│ OFF-RAMP │───→│  BANK WIRE  │
│  ZK+MEV+ML  │    │ SOL→USDC     │    │ >$100 USD     │    │ Exchange │    │  ACH/SEPA   │
│  Profits    │    │ via Jupiter  │    │ + cooldown    │    │  API     │    │  withdrawal │
└─────────────┘    └──────────────┘    └───────────────┘    └──────────┘    └─────────────┘
```

### 10.2 Consolidation via Jupiter Aggregator

Jupiter is Solana's DEX aggregator (routes across Raydium, Orca, Meteora, etc.).

**Quote API:** `GET https://quote-api.jup.ag/v6/quote?inputMint=SOL&outputMint=USDC&amount=<lamports>&slippageBps=50`

**Swap API:** `POST https://quote-api.jup.ag/v6/swap` with the quote response + user public key. Returns a serialized transaction to sign and submit.

### 10.3 Exchange API Signing

**Coinbase (HMAC-SHA256):**
```
signature = HMAC-SHA256(
  key: api_secret,
  message: timestamp + method + path + body
)
Headers: CB-ACCESS-KEY, CB-ACCESS-SIGN, CB-ACCESS-TIMESTAMP, CB-ACCESS-PASSPHRASE
```

**Kraken (HMAC-SHA512):**
```
signature = HMAC-SHA512(
  key: base64_decode(api_secret),
  message: path + SHA256(nonce + body)
)
Headers: API-Key, API-Sign
```

### 10.4 Off-Ramp Flow

1. **Sell USDC:** POST market sell order (USDC-USD on Coinbase, USDCUSD on Kraken)
2. **Wait for fill:** Poll order status until `settled` (Coinbase) or `closed` (Kraken)
3. **Withdraw fiat:** POST withdrawal to pre-configured bank account (payment method ID)
4. **Record:** Update treasury metrics, append to audit log

### 10.5 EVM Support (Solidity Contracts)

For Ethereum/Arbitrum/Base/Polygon chains:

**ProfitVault.sol** — Owner-controlled vault with reentrancy guard:
- `deposit()` — receive ETH
- `withdrawAll()` / `withdrawTo(address, amount)` — extract profits
- `withdrawToken(IERC20, to, amount)` — extract ERC20 tokens
- Events: ProfitDeposited, FundsWithdrawn, TokenWithdrawn

**FlashArbitrage.sol** — Zero-capital atomic cross-DEX arbitrage:
- `executeArbitrage()` — initiate Uniswap V2 flash swap
- `uniswapV2Call()` — callback: swap on second DEX, validate profit, repay loan + send profit to vault
- Reverts if profit < minimum threshold (atomic safety)

### 10.6 Dynamic Thresholding

The treasury keeper adjusts consolidation thresholds based on gas conditions:

```
effective_threshold = base_threshold × gas_multiplier
gas_multiplier = max(1.0, current_gas / target_gas)
```

When gas is expensive, the keeper waits for larger accumulations before consolidating (amortizing gas over more value).

---

## 11. Risk and Governance

### 11.1 Circuit Breaker

```python
if consecutive_losses >= 10:
    circuit_breaker_tripped = True
    daemon.stop()  # Kill the Rust process
    audit("circuit_breaker", {losses: 10})
```

The circuit breaker is implemented in the Python orchestrator layer (not Rust) because:
1. It requires cross-domain aggregation (ZK + MEV + ML losses combined)
2. It must survive Rust daemon crashes
3. It integrates with the governance layer (manual gate, audit log)

### 11.2 Slashing Protection

ZK prover networks penalize failed proofs (slashing). Our protection:
- **max_stake_fraction = 10%** — never stake more than 10% of total capital
- **Deadline monitoring** — abort proof generation if estimated completion exceeds deadline
- **Cost estimation** — only bid on proofs where estimated cost < 85% of max reward

### 11.3 Audit Trail

Every action is logged to `runtime/yield_daemon/audit.jsonl` with HMAC-SHA256 chain signatures for tamper detection.

---

## 12. Metrics and Integration

### 12.1 Per-Domain Metrics

Each domain exports metrics at configurable intervals (default 30s):

| Domain | Key Metrics |
|--------|-------------|
| ZK Prover | proofs_generated, proofs_accepted, revenue_sat |
| MEV Arbitrage | opportunities, bundles_submitted, revenue_sat |
| ML Subnet | inferences, training_rounds, revenue_sat |

Metrics are exported both as Prometheus counters/gauges (`:9191/metrics`) and as JSON files in `state_dir` for the off-ramp service.

### 12.2 Off-Ramp Integration

The Python off-ramp service (`orchestrator/offramp.py`) polls the JSON metrics bridge and handles:
- Wallet balance monitoring
- SOL → USDC consolidation via Jupiter
- USDC → fiat conversion via exchange API
- Fiat → bank withdrawal

---

## 13. Benchmarking Methodology

### 13.1 Statistical Rigor

All benchmarks use Criterion.rs, which:
- Runs warmup iterations to fill caches
- Collects >100 samples per benchmark
- Reports mean ± confidence interval
- Detects performance regressions via statistical comparison

### 13.2 Benchmark Suite

| Benchmark | Input Size | Expected Time | Measures |
|-----------|-----------|--------------|----------|
| NTT forward 2^8 | 256 elements | <100μs | NTT scaling |
| NTT forward 2^16 | 65536 elements | <10ms | NTT at proof-size |
| Montgomery mul | Single | <50ns | Field operation cost |
| Montgomery sqr | Single | <40ns | Squaring vs mul |
| Montgomery add | Single | <5ns | Addition baseline |
| Arena alloc 64B × 1000 | 1000 allocs | <5μs | Allocation throughput |
| Arena reset | Single | <1ns | Reset overhead |

### 13.3 Continuous Performance Monitoring

The off-ramp service tracks per-cycle metrics:
- `elapsed_ms` per off-ramp cycle
- `avg_proof_time_ms` for ZK proofs
- `avg_latency_us` for MEV bundles
- `gpu_utilization` for ML inference

Regressions trigger alerts via the Error Escalation Hub.

---

## 14. Deployment Architecture

### 14.1 Hardware Recommendations

| Component | ZK Prover | MEV Bot | ML Miner |
|-----------|-----------|---------|----------|
| CPU | AMD EPYC 9654 (96C, AVX-512) | Intel i9-14900K (high clock) | Any modern |
| GPU | NVIDIA A100 (MSM acceleration) | Not needed | NVIDIA H100/H200 |
| RAM | 128GB DDR5 | 32GB DDR5 | 80GB HBM3e (on GPU) |
| Network | 10GbE | 25GbE + kernel bypass | 10GbE |
| Storage | 1TB NVMe | 256GB NVMe | 2TB NVMe |

### 14.2 Co-location

For competitive MEV extraction, co-locate servers in the same data center as:
- **Solana validators** (particularly Jito validators)
- **Ethereum block builders** (Flashbots relay operators)

Network latency to the block engine should be <5ms for competitive bundle submission.

### 14.3 Legal Structure

For US-based operators:
1. **DBA filing** — Register assumed business name with county clerk
2. **Tax classification** — Income from ZK proving and ML mining is ordinary income; MEV arbitrage may be capital gains
3. **Grid compliance** — Texas SB 1929 requires registration for >75MW loads (unlikely for individual operators)
4. **Know Your Customer** — Some prover networks require KYC for node registration

---

## Appendix A: BN254 Field Parameters

```
Prime p = 21888242871839275222246405745257275088548364400416034343698204186575808495617
Hex:      0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001

Generator g = 5
Two-adicity: 28 (max NTT size = 2^28 = 268,435,456)

Montgomery R = 2^256 mod p
Montgomery R² mod p (for conversion)
Montgomery -p⁻¹ mod 2^64 = 0xc2e1f593efffffff
```

## Appendix B: Solana AMM Program IDs

| AMM | Program ID |
|-----|-----------|
| Raydium V4 | `675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8` |
| Orca Whirlpool | `whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc` |
| Meteora DLMM | `LBUZKhRxPF3XUpBCjp4YzTKgLccjZhTSDM9YuVaPwxo` |

## Appendix C: References

1. Montgomery, P.L. (1985). "Modular multiplication without trial division." *Mathematics of Computation*.
2. Pippenger, N. (1976). "On the evaluation of powers and related problems." *FOCS*.
3. Cooley, J.W. & Tukey, J.W. (1965). "An algorithm for the machine calculation of complex Fourier series." *Mathematics of Computation*.
4. Frigo, M. et al. (1999). "Cache-oblivious algorithms." *FOCS*.
5. Dao, T. et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention." *NeurIPS*.
6. Adams, R.P. & MacKay, D.J.C. (2007). "Bayesian Online Changepoint Detection." *arXiv*.
7. Aggarwal, A. & Vitter, J.S. (1988). "The Input/Output Complexity of Sorting and Related Problems." *CACM*.

---

*Document version: 0.2.0 | Last updated: 2026-06-18*
