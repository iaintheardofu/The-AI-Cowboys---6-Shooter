//! Number-Theoretic Transform (NTT) — finite-field FFT for ZK proofs.
//!
//! The NTT is the computational bottleneck of ZK proof generation,
//! accounting for up to 90% of proving time. This implementation
//! uses Cooley-Tukey decimation-in-time butterfly with Montgomery
//! domain arithmetic for zero-division-cost modular reduction.
//!
//! Optimization tiers:
//!   Level 0: Scalar Montgomery (baseline)
//!   Level 1: AVX2 4-way parallel butterflies
//!   Level 2: AVX-512 IFMA 52-bit limb acceleration

use super::montgomery::MontgomeryU256;

/// Precomputed twiddle factors (roots of unity) in Montgomery form.
pub struct NttDomain {
    pub roots: Vec<MontgomeryU256>,
    pub inv_roots: Vec<MontgomeryU256>,
    pub log_n: u32,
    pub n: usize,
    pub n_inv: MontgomeryU256, // 1/n in Montgomery form
}

impl NttDomain {
    /// Build twiddle factor table for domain of size 2^log_n.
    /// Uses repeated squaring to compute primitive root of unity.
    pub fn new(log_n: u32) -> Self {
        let n = 1usize << log_n;
        let mut roots = vec![MontgomeryU256::ZERO; n];
        let mut inv_roots = vec![MontgomeryU256::ZERO; n];

        // Primitive root of unity for BN254 scalar field.
        // omega = generator^((p-1)/n) where generator = 5
        let _generator = MontgomeryU256::from_u64(5);

        // For BN254, (p-1) is divisible by 2^28, so max NTT size = 2^28.
        // omega = 5^((p-1)/2^log_n) mod p
        // We compute this via exponentiation.

        // p-1 divided by n
        // Simplified: use precomputed roots for common sizes
        let omega = compute_root_of_unity(log_n);
        let omega_inv = omega.mont_inv();

        // Build root table: omega^0, omega^1, ..., omega^{n-1}
        roots[0] = MontgomeryU256::ONE;
        if n > 1 {
            roots[1] = omega;
            for i in 2..n {
                roots[i] = roots[i - 1].mont_mul(&omega);
            }
        }

        inv_roots[0] = MontgomeryU256::ONE;
        if n > 1 {
            inv_roots[1] = omega_inv;
            for i in 2..n {
                inv_roots[i] = inv_roots[i - 1].mont_mul(&omega_inv);
            }
        }

        // n_inv = (1/n) mod p
        let n_field = MontgomeryU256::from_u64(n as u64);
        let n_inv = n_field.mont_inv();

        Self {
            roots,
            inv_roots,
            log_n,
            n,
            n_inv,
        }
    }

    /// Forward NTT: polynomial evaluation at roots of unity.
    /// In-place Cooley-Tukey radix-2 DIT butterfly.
    pub fn forward(&self, coeffs: &mut [MontgomeryU256]) {
        assert_eq!(coeffs.len(), self.n);

        // Bit-reversal permutation
        bit_reverse_permutation(coeffs, self.log_n);

        // Butterfly stages
        let mut stage_len = 1;
        for _stage in 0..self.log_n {
            let step = stage_len * 2;
            let twiddle_step = self.n / step;

            let mut k = 0;
            while k < self.n {
                for j in 0..stage_len {
                    let twiddle = &self.roots[j * twiddle_step];
                    let u = coeffs[k + j];
                    let v = coeffs[k + j + stage_len].mont_mul(twiddle);

                    // Butterfly: branchless add/sub
                    coeffs[k + j] = u.mont_add(&v);
                    coeffs[k + j + stage_len] = u.mont_sub(&v);
                }
                k += step;
            }
            stage_len = step;
        }
    }

    /// Inverse NTT: interpolation from evaluations.
    /// Same structure as forward but uses inverse roots and scales by 1/n.
    pub fn inverse(&self, evals: &mut [MontgomeryU256]) {
        assert_eq!(evals.len(), self.n);

        bit_reverse_permutation(evals, self.log_n);

        let mut stage_len = 1;
        for _stage in 0..self.log_n {
            let step = stage_len * 2;
            let twiddle_step = self.n / step;

            let mut k = 0;
            while k < self.n {
                for j in 0..stage_len {
                    let twiddle = &self.inv_roots[j * twiddle_step];
                    let u = evals[k + j];
                    let v = evals[k + j + stage_len].mont_mul(twiddle);

                    evals[k + j] = u.mont_add(&v);
                    evals[k + j + stage_len] = u.mont_sub(&v);
                }
                k += step;
            }
            stage_len = step;
        }

        // Scale by 1/n
        for val in evals.iter_mut() {
            *val = val.mont_mul(&self.n_inv);
        }
    }

    /// Polynomial multiplication via NTT: O(n log n) field operations.
    /// f(x) * g(x) = INTT(NTT(f) ⊙ NTT(g))
    pub fn poly_mul(
        &self,
        a: &mut [MontgomeryU256],
        b: &mut [MontgomeryU256],
    ) -> Vec<MontgomeryU256> {
        assert_eq!(a.len(), self.n);
        assert_eq!(b.len(), self.n);

        self.forward(a);
        self.forward(b);

        // Pointwise multiplication (embarrassingly parallel, SIMD-friendly)
        let mut c: Vec<MontgomeryU256> = a.iter()
            .zip(b.iter())
            .map(|(ai, bi)| ai.mont_mul(bi))
            .collect();

        self.inverse(&mut c);
        c
    }
}

/// Bit-reversal permutation (in-place).
fn bit_reverse_permutation(data: &mut [MontgomeryU256], log_n: u32) {
    let n = data.len();
    for i in 0..n {
        let rev = bit_reverse(i as u32, log_n) as usize;
        if i < rev {
            data.swap(i, rev);
        }
    }
}

/// Reverse the bottom `bits` bits of `x`.
#[inline(always)]
fn bit_reverse(mut x: u32, bits: u32) -> u32 {
    let mut result = 0u32;
    for _ in 0..bits {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

/// Compute a primitive 2^k-th root of unity for BN254 scalar field.
/// Uses generator g=5 and computes g^((p-1)/2^k).
fn compute_root_of_unity(log_n: u32) -> MontgomeryU256 {
    // For BN254, TWO_ADICITY = 28 (max log_n)
    assert!(log_n <= 28, "NTT size exceeds BN254 two-adicity (max 2^28)");

    // Compute g^((p-1)/2^28) to get the primitive 2^28-th root of unity.
    // p-1 = 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000
    // (p-1)/2^28 = p-1 >> 28
    // Exponent: (p-1) / 2^28
    let exp_p1_div_2_28: [u64; 4] = [
        0x9b9709143e1f593f,
        0x181585d2833e8487,
        0xe131a029b85045b6,
        0x0000000030644e72,
    ];

    let generator = MontgomeryU256::from_u64(5);
    let root_of_unity_2_28 = generator.mont_pow(&exp_p1_div_2_28);

    // Square (28 - log_n) times to get a 2^log_n-th root
    let mut omega = root_of_unity_2_28;
    for _ in 0..(28 - log_n) {
        omega = omega.mont_sqr();
    }

    omega
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bit_reverse() {
        assert_eq!(bit_reverse(0b000, 3), 0b000);
        assert_eq!(bit_reverse(0b001, 3), 0b100);
        assert_eq!(bit_reverse(0b010, 3), 0b010);
        assert_eq!(bit_reverse(0b011, 3), 0b110);
    }

    #[test]
    fn test_ntt_roundtrip_small() {
        let domain = NttDomain::new(2); // Size 4
        let original = vec![
            MontgomeryU256::from_u64(1),
            MontgomeryU256::from_u64(2),
            MontgomeryU256::from_u64(3),
            MontgomeryU256::from_u64(4),
        ];

        let mut data = original.clone();
        domain.forward(&mut data);
        domain.inverse(&mut data);

        // After NTT -> INTT, should recover original values
        for (a, b) in data.iter().zip(original.iter()) {
            let a_val = a.from_montgomery();
            let b_val = b.from_montgomery();
            assert_eq!(a_val.limbs[0], b_val.limbs[0]);
        }
    }
}
