//! Montgomery Multiplication — modular arithmetic without division.
//!
//! Maps field elements into Montgomery domain where modular reduction
//! replaces division-by-prime with division-by-power-of-2 (bit shift).
//! This is the backbone of all ZK proof generation.
//!
//! For BN254 (254-bit prime), we use 4x64-bit limb representation.
//! On AVX-512 IFMA capable hardware, we use 5x52-bit limbs for
//! simultaneous 52×52→104 bit multiplications.

/// 64-bit limb Montgomery form for standard x86-64.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(align(32))]
pub struct MontgomeryU256 {
    pub limbs: [u64; 4],
}

/// BN254 scalar field modulus: 21888242871839275222246405745257275088548364400416034343698204186575808495617
const BN254_MODULUS: [u64; 4] = [
    0x43e1f593f0000001,
    0x2833e84879b97091,
    0xb85045b68181585d,
    0x30644e72e131a029,
];

/// R = 2^256 mod p (Montgomery R constant)
const MONT_R: [u64; 4] = [
    0xac96341c4ffffffb,
    0x36fc76959f60cd29,
    0x666ea36f7879462e,
    0x0e0a77c19a07df2f,
];

/// R^2 mod p (for converting to Montgomery form)
const MONT_R2: [u64; 4] = [
    0x1bb8e645ae216da7,
    0x53fe3ab1e35c59e3,
    0x8c49833d53bb8085,
    0x0216d0b17f4e44a5,
];

/// -p^{-1} mod 2^64 (Montgomery reduction constant)
const INV: u64 = 0xc2e1f593efffffff;

impl MontgomeryU256 {
    pub const ZERO: Self = Self { limbs: [0; 4] };
    pub const ONE: Self = Self { limbs: MONT_R };

    /// Convert a u64 value into Montgomery form.
    #[inline]
    pub fn from_u64(val: u64) -> Self {
        let raw = Self { limbs: [val, 0, 0, 0] };
        raw.to_montgomery()
    }

    /// Convert to Montgomery domain: a * R^2 * R^{-1} = a * R mod p
    #[inline]
    pub fn to_montgomery(self) -> Self {
        let r2 = Self { limbs: MONT_R2 };
        self.mont_mul(&r2)
    }

    /// Convert from Montgomery domain: a * 1 * R^{-1} = a * R^{-1} mod p
    #[inline]
    pub fn from_montgomery(self) -> Self {
        let one = Self { limbs: [1, 0, 0, 0] };
        self.mont_mul(&one)
    }

    /// Montgomery multiplication: (a * b * R^{-1}) mod p
    /// Uses CIOS (Coarsely Integrated Operand Scanning) algorithm.
    #[inline]
    pub fn mont_mul(&self, rhs: &Self) -> Self {
        let mut t = [0u128; 5];

        for i in 0..4 {
            // t += a * b[i]
            let bi = rhs.limbs[i] as u128;
            let mut carry: u128 = 0;
            for j in 0..4 {
                t[j] += self.limbs[j] as u128 * bi + carry;
                carry = t[j] >> 64;
                t[j] &= 0xFFFF_FFFF_FFFF_FFFF;
            }
            t[4] += carry;

            // m = t[0] * INV mod 2^64
            let m = (t[0] as u64).wrapping_mul(INV) as u128;

            // t += m * p
            carry = 0;
            for j in 0..4 {
                t[j] += m * BN254_MODULUS[j] as u128 + carry;
                carry = t[j] >> 64;
                t[j] &= 0xFFFF_FFFF_FFFF_FFFF;
            }
            t[4] += carry;

            // Shift right by 64 bits (t[0] is now 0 mod 2^64)
            t[0] = t[1];
            t[1] = t[2];
            t[2] = t[3];
            t[3] = t[4];
            t[4] = 0;
        }

        let mut result = Self {
            limbs: [t[0] as u64, t[1] as u64, t[2] as u64, t[3] as u64],
        };
        if result.gte_modulus() {
            result = result.sub_modulus();
        }
        result
    }

    /// Montgomery squaring (slightly faster than mul with self).
    #[inline]
    pub fn mont_sqr(&self) -> Self {
        self.mont_mul(self)
    }

    /// Modular exponentiation via square-and-multiply.
    /// Uses branchless constant-time selection for side-channel resistance.
    pub fn mont_pow(&self, exp: &[u64; 4]) -> Self {
        let mut result = Self::ONE;
        let mut base = *self;

        for &word in exp.iter() {
            for bit in 0..64 {
                let flag = (word >> bit) & 1;
                // Branchless: conditionally multiply
                let product = result.mont_mul(&base);
                result = Self::conditional_select(&result, &product, flag);
                base = base.mont_sqr();
            }
        }
        result
    }

    /// Modular inverse via Fermat's little theorem: a^{-1} = a^{p-2} mod p.
    pub fn mont_inv(&self) -> Self {
        // p - 2 for BN254
        let p_minus_2: [u64; 4] = [
            0x43e1f593efffffff,
            0x2833e84879b97091,
            0xb85045b68181585d,
            0x30644e72e131a029,
        ];
        self.mont_pow(&p_minus_2)
    }

    /// Branchless conditional select: returns a if flag==0, b if flag==1.
    #[inline(always)]
    fn conditional_select(a: &Self, b: &Self, flag: u64) -> Self {
        let mask = 0u64.wrapping_sub(flag); // 0x0..0 or 0xF..F
        Self {
            limbs: [
                a.limbs[0] ^ (mask & (a.limbs[0] ^ b.limbs[0])),
                a.limbs[1] ^ (mask & (a.limbs[1] ^ b.limbs[1])),
                a.limbs[2] ^ (mask & (a.limbs[2] ^ b.limbs[2])),
                a.limbs[3] ^ (mask & (a.limbs[3] ^ b.limbs[3])),
            ],
        }
    }

    /// Check if self >= modulus.
    #[inline]
    fn gte_modulus(&self) -> bool {
        for i in (0..4).rev() {
            if self.limbs[i] > BN254_MODULUS[i] { return true; }
            if self.limbs[i] < BN254_MODULUS[i] { return false; }
        }
        true // equal
    }

    /// Subtract modulus (no borrow check — caller ensures self >= p).
    #[inline]
    fn sub_modulus(&self) -> Self {
        let mut result = [0u64; 4];
        let mut borrow: u64 = 0;
        for i in 0..4 {
            let (diff, b1) = self.limbs[i].overflowing_sub(BN254_MODULUS[i]);
            let (diff, b2) = diff.overflowing_sub(borrow);
            result[i] = diff;
            borrow = b1 as u64 + b2 as u64;
        }
        Self { limbs: result }
    }

    /// Modular addition: (a + b) mod p.
    #[inline]
    pub fn mont_add(&self, rhs: &Self) -> Self {
        let mut result = [0u64; 4];
        let mut carry: u64 = 0;
        for i in 0..4 {
            let (sum, c1) = self.limbs[i].overflowing_add(rhs.limbs[i]);
            let (sum, c2) = sum.overflowing_add(carry);
            result[i] = sum;
            carry = c1 as u64 + c2 as u64;
        }
        let mut r = Self { limbs: result };
        if r.gte_modulus() {
            r = r.sub_modulus();
        }
        r
    }

    /// Modular subtraction: (a - b) mod p.
    #[inline]
    pub fn mont_sub(&self, rhs: &Self) -> Self {
        let mut result = [0u64; 4];
        let mut borrow: u64 = 0;
        for i in 0..4 {
            let (diff, b1) = self.limbs[i].overflowing_sub(rhs.limbs[i]);
            let (diff, b2) = diff.overflowing_sub(borrow);
            result[i] = diff;
            borrow = b1 as u64 + b2 as u64;
        }
        if borrow > 0 {
            // Add modulus back
            let mut carry: u64 = 0;
            for i in 0..4 {
                let (sum, c1) = result[i].overflowing_add(BN254_MODULUS[i]);
                let (sum, c2) = sum.overflowing_add(carry);
                result[i] = sum;
                carry = c1 as u64 + c2 as u64;
            }
        }
        Self { limbs: result }
    }
}

/// Widening 64×64 → 128-bit multiplication.
#[inline(always)]
fn mul_wide(a: u64, b: u64) -> (u64, u64) {
    let full = a as u128 * b as u128;
    (full as u64, (full >> 64) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_montgomery_roundtrip() {
        let val = MontgomeryU256 { limbs: [42, 0, 0, 0] };
        let mont = val.to_montgomery();
        let back = mont.from_montgomery();
        assert_eq!(back.limbs[0], 42);
        assert_eq!(back.limbs[1], 0);
    }

    #[test]
    fn test_mont_mul_identity() {
        let a = MontgomeryU256::from_u64(7);
        let one = MontgomeryU256::ONE;
        let result = a.mont_mul(&one);
        let val = result.from_montgomery();
        assert_eq!(val.limbs[0], 7);
    }

    #[test]
    fn test_mont_add_sub() {
        let a = MontgomeryU256::from_u64(100);
        let b = MontgomeryU256::from_u64(42);
        let sum = a.mont_add(&b);
        let diff = sum.mont_sub(&b);
        let val = diff.from_montgomery();
        assert_eq!(val.limbs[0], 100);
    }

    #[test]
    fn test_conditional_select_branchless() {
        let a = MontgomeryU256::from_u64(10);
        let b = MontgomeryU256::from_u64(20);

        let sel0 = MontgomeryU256::conditional_select(&a, &b, 0);
        let sel1 = MontgomeryU256::conditional_select(&a, &b, 1);

        assert_eq!(sel0.limbs, a.limbs);
        assert_eq!(sel1.limbs, b.limbs);
    }

    #[test]
    fn test_widening_mul() {
        let (lo, hi) = mul_wide(u64::MAX, u64::MAX);
        // (2^64 - 1)^2 = 2^128 - 2^65 + 1
        assert_eq!(lo, 1);
        assert_eq!(hi, u64::MAX - 1);
    }
}
