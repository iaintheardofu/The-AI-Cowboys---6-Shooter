//! Multi-Scalar Multiplication (MSM) — Pippenger's bucket method.
//!
//! Computes: sum(s_i * G_i) for N scalar-point pairs.
//! Pippenger reduces O(N * 256) group operations to O(N / log N).
//! Critical for KZG commitments and Groth16 proof generation.

use super::montgomery::MontgomeryU256;

/// Affine point on BN254 G1 curve: y^2 = x^3 + 3
#[derive(Clone, Copy, Debug)]
pub struct AffinePoint {
    pub x: MontgomeryU256,
    pub y: MontgomeryU256,
    pub infinity: bool,
}

/// Projective point (Jacobian coordinates) for efficient group operations.
/// (X, Y, Z) represents affine (X/Z^2, Y/Z^3).
#[derive(Clone, Copy, Debug)]
pub struct ProjectivePoint {
    pub x: MontgomeryU256,
    pub y: MontgomeryU256,
    pub z: MontgomeryU256,
}

impl ProjectivePoint {
    pub const IDENTITY: Self = Self {
        x: MontgomeryU256::ZERO,
        y: MontgomeryU256::ONE,
        z: MontgomeryU256::ZERO,
    };

    pub fn is_identity(&self) -> bool {
        self.z.limbs == [0; 4]
    }

    /// Point doubling in Jacobian coordinates.
    /// Cost: 1M + 5S + 1*a + 7add (a=0 for BN254)
    #[inline]
    pub fn double(&self) -> Self {
        if self.is_identity() {
            return *self;
        }

        let xx = self.x.mont_sqr();
        let yy = self.y.mont_sqr();
        let yyyy = yy.mont_sqr();
        let zz = self.z.mont_sqr();

        // S = 2 * ((X+YY)^2 - XX - YYYY)
        let s = self.x.mont_add(&yy).mont_sqr()
            .mont_sub(&xx)
            .mont_sub(&yyyy);
        let s = s.mont_add(&s);

        // M = 3*XX + a*ZZ^2 (a=0 for BN254, so M = 3*XX)
        let m = xx.mont_add(&xx).mont_add(&xx);

        // T = M^2 - 2*S
        let t = m.mont_sqr().mont_sub(&s).mont_sub(&s);

        // X3 = T
        let x3 = t;

        // Y3 = M*(S-T) - 8*YYYY
        let yyyy_8 = yyyy.mont_add(&yyyy);
        let yyyy_8 = yyyy_8.mont_add(&yyyy_8);
        let yyyy_8 = yyyy_8.mont_add(&yyyy_8);
        let y3 = m.mont_mul(&s.mont_sub(&t)).mont_sub(&yyyy_8);

        // Z3 = (Y+Z)^2 - YY - ZZ
        let z3 = self.y.mont_add(&self.z).mont_sqr()
            .mont_sub(&yy)
            .mont_sub(&zz);

        Self { x: x3, y: y3, z: z3 }
    }

    /// Convert projective point to affine coordinates.
    /// Computes (X/Z^2, Y/Z^3) via modular inverse of Z.
    pub fn to_affine(&self) -> AffinePoint {
        if self.is_identity() {
            return AffinePoint {
                x: MontgomeryU256::ZERO,
                y: MontgomeryU256::ZERO,
                infinity: true,
            };
        }
        let z_inv = self.z.mont_inv();
        let z_inv2 = z_inv.mont_sqr();
        let z_inv3 = z_inv2.mont_mul(&z_inv);
        AffinePoint {
            x: self.x.mont_mul(&z_inv2),
            y: self.y.mont_mul(&z_inv3),
            infinity: false,
        }
    }

    /// Projective point addition: self + rhs (both Jacobian).
    #[inline]
    pub fn add(&self, rhs: &ProjectivePoint) -> Self {
        if self.is_identity() {
            return *rhs;
        }
        if rhs.is_identity() {
            return *self;
        }

        let z1z1 = self.z.mont_sqr();
        let z2z2 = rhs.z.mont_sqr();

        let u1 = self.x.mont_mul(&z2z2);
        let u2 = rhs.x.mont_mul(&z1z1);

        let s1 = self.y.mont_mul(&rhs.z).mont_mul(&z2z2);
        let s2 = rhs.y.mont_mul(&self.z).mont_mul(&z1z1);

        let h = u2.mont_sub(&u1);
        let r = s2.mont_sub(&s1);

        // If h == 0 and r == 0, points are equal → use doubling
        if h.limbs == [0; 4] && r.limbs == [0; 4] {
            return self.double();
        }

        let hh = h.mont_sqr();
        let hhh = h.mont_mul(&hh);
        let v = u1.mont_mul(&hh);

        let x3 = r.mont_sqr().mont_sub(&hhh).mont_sub(&v).mont_sub(&v);
        let y3 = r.mont_mul(&v.mont_sub(&x3)).mont_sub(&s1.mont_mul(&hhh));
        let z3 = self.z.mont_mul(&rhs.z).mont_mul(&h);

        Self { x: x3, y: y3, z: z3 }
    }

    /// Mixed addition: self + affine point (Z2=1, saves 4 multiplications).
    #[inline]
    pub fn add_affine(&self, rhs: &AffinePoint) -> Self {
        if rhs.infinity {
            return *self;
        }
        if self.is_identity() {
            return Self {
                x: rhs.x,
                y: rhs.y,
                z: MontgomeryU256::ONE,
            };
        }

        let z1z1 = self.z.mont_sqr();
        let u2 = rhs.x.mont_mul(&z1z1);
        let s2 = rhs.y.mont_mul(&self.z).mont_mul(&z1z1);
        let h = u2.mont_sub(&self.x);
        let hh = h.mont_sqr();
        let hhh = h.mont_mul(&hh);
        let r = s2.mont_sub(&self.y);

        let v = self.x.mont_mul(&hh);

        // X3 = r^2 - HHH - 2*V
        let x3 = r.mont_sqr()
            .mont_sub(&hhh)
            .mont_sub(&v)
            .mont_sub(&v);

        // Y3 = r*(V - X3) - Y1*HHH
        let y3 = r.mont_mul(&v.mont_sub(&x3))
            .mont_sub(&self.y.mont_mul(&hhh));

        // Z3 = Z1 * H
        let z3 = self.z.mont_mul(&h);

        Self { x: x3, y: y3, z: z3 }
    }
}

/// Pippenger's bucket MSM: computes sum(scalars[i] * points[i]).
///
/// Window size `w` splits each 256-bit scalar into 256/w windows.
/// For each window, points are accumulated into 2^w buckets.
/// Total cost: N * 256/w + 256/w * 2^w group additions.
/// Optimal w ≈ log2(N) / 2.
pub fn pippenger_msm(
    scalars: &[MontgomeryU256],
    points: &[AffinePoint],
    window_bits: usize,
) -> ProjectivePoint {
    assert_eq!(scalars.len(), points.len());
    let n = scalars.len();
    if n == 0 {
        return ProjectivePoint::IDENTITY;
    }

    let num_windows = (256 + window_bits - 1) / window_bits;
    let num_buckets = (1 << window_bits) - 1; // buckets 1..2^w-1

    let mut result = ProjectivePoint::IDENTITY;

    // Process windows from most significant to least significant
    for window_idx in (0..num_windows).rev() {
        // Shift result by window_bits (repeated doubling)
        if window_idx < num_windows - 1 {
            for _ in 0..window_bits {
                result = result.double();
            }
        }

        // Initialize empty buckets
        let mut buckets = vec![ProjectivePoint::IDENTITY; num_buckets];

        // Distribute points into buckets based on scalar window
        for i in 0..n {
            let bucket_idx = extract_window(&scalars[i], window_idx, window_bits);
            if bucket_idx > 0 {
                buckets[bucket_idx as usize - 1] =
                    buckets[bucket_idx as usize - 1].add_affine(&points[i]);
            }
        }

        // Aggregate buckets: sum = B[k]*k = B[k] + B[k+1] + ... + B[2^w-1]
        // computed via running sum from top.
        // Use projective addition (not fake affine from raw projective coords).
        let mut running_sum = ProjectivePoint::IDENTITY;
        let mut window_sum = ProjectivePoint::IDENTITY;

        for j in (0..num_buckets).rev() {
            running_sum = running_sum.add(&buckets[j]);
            window_sum = window_sum.add(&running_sum);
        }

        result = result.add(&window_sum);
    }

    result
}

/// Extract a `window_bits`-wide window from a scalar at position `window_idx`.
#[inline(always)]
fn extract_window(scalar: &MontgomeryU256, window_idx: usize, window_bits: usize) -> u64 {
    let bit_offset = window_idx * window_bits;
    let limb_idx = bit_offset / 64;
    let bit_in_limb = bit_offset % 64;
    let mask = (1u64 << window_bits) - 1;

    if limb_idx >= 4 {
        return 0;
    }

    let val = scalar.limbs[limb_idx] >> bit_in_limb;

    // Handle window spanning two limbs
    if bit_in_limb + window_bits > 64 && limb_idx + 1 < 4 {
        let overflow_bits = bit_in_limb + window_bits - 64;
        let high = scalar.limbs[limb_idx + 1] & ((1u64 << overflow_bits) - 1);
        (val | (high << (64 - bit_in_limb))) & mask
    } else {
        val & mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_window() {
        let scalar = MontgomeryU256 {
            limbs: [0b11010110, 0, 0, 0],
        };
        assert_eq!(extract_window(&scalar, 0, 4), 0b0110);
        assert_eq!(extract_window(&scalar, 1, 4), 0b1101);
    }

    #[test]
    fn test_projective_identity() {
        let id = ProjectivePoint::IDENTITY;
        assert!(id.is_identity());
        let doubled = id.double();
        assert!(doubled.is_identity());
    }
}
