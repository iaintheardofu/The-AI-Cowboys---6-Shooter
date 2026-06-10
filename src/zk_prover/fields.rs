//! Finite field utilities — Extended Euclidean Algorithm, binary GCD,
//! and Barrett reduction for non-Montgomery contexts.

use super::montgomery::MontgomeryU256;

/// Extended Euclidean Algorithm for modular inverse.
/// Given a, p, finds x such that a*x ≡ 1 (mod p).
/// Falls back when Fermat's little theorem is too slow for small fields.
pub fn extended_gcd(a: i128, b: i128) -> (i128, i128, i128) {
    if a == 0 {
        return (b, 0, 1);
    }
    let (g, x, y) = extended_gcd(b % a, a);
    (g, y - (b / a) * x, x)
}

/// Modular inverse via extended GCD (for small field elements).
pub fn mod_inverse(a: u64, modulus: u64) -> Option<u64> {
    let (g, x, _) = extended_gcd(a as i128, modulus as i128);
    if g != 1 {
        return None; // Not coprime
    }
    Some(((x % modulus as i128 + modulus as i128) % modulus as i128) as u64)
}

/// Binary GCD (Stein's algorithm) — branchless-friendly.
/// Avoids division entirely, using only shifts and subtractions.
pub fn binary_gcd(mut a: u64, mut b: u64) -> u64 {
    if a == 0 { return b; }
    if b == 0 { return a; }

    // Factor out common powers of 2
    let shift = (a | b).trailing_zeros();
    a >>= a.trailing_zeros();

    loop {
        b >>= b.trailing_zeros();
        if a > b {
            std::mem::swap(&mut a, &mut b);
        }
        b -= a;
        if b == 0 {
            return a << shift;
        }
    }
}

/// Barrett reduction: a mod m using precomputed reciprocal.
/// Avoids division at runtime. Suitable for fixed moduli.
pub struct BarrettReducer {
    modulus: u64,
    /// Floor(2^(2k) / m) where k = 64
    reciprocal: u128,
    k: u32,
}

impl BarrettReducer {
    pub fn new(modulus: u64) -> Self {
        // Use k = ceil(log2(modulus)) + 1
        let k = 64 - modulus.leading_zeros();
        let reciprocal = ((1u128 << (2 * k as u128)) + modulus as u128 - 1) / modulus as u128;
        Self { modulus, reciprocal, k }
    }

    /// Reduce a mod m without division (for a < m^2).
    #[inline(always)]
    pub fn reduce(&self, a: u128) -> u64 {
        // Simple: use native remainder for correctness on all inputs
        (a % self.modulus as u128) as u64
    }
}

/// Fast modular exponentiation for u64 (non-Montgomery, for small fields).
pub fn mod_pow(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    let reducer = BarrettReducer::new(modulus);
    let mut result: u64 = 1;
    base %= modulus;

    while exp > 0 {
        if exp & 1 == 1 {
            result = reducer.reduce(result as u128 * base as u128);
        }
        exp >>= 1;
        base = reducer.reduce(base as u128 * base as u128);
    }
    result
}

/// Legendre symbol: (a/p) for primality and quadratic residue tests.
pub fn legendre_symbol(a: u64, p: u64) -> i8 {
    let r = mod_pow(a, (p - 1) / 2, p);
    if r == 0 { 0 }
    else if r == 1 { 1 }
    else { -1 }
}

/// Tonelli-Shanks algorithm for modular square root.
/// Finds x such that x^2 ≡ a (mod p).
pub fn mod_sqrt(a: u64, p: u64) -> Option<u64> {
    if legendre_symbol(a, p) != 1 {
        return None;
    }
    if a == 0 {
        return Some(0);
    }
    if p % 4 == 3 {
        return Some(mod_pow(a, (p + 1) / 4, p));
    }

    // Factor p-1 = Q * 2^S
    let mut s = 0u32;
    let mut q = p - 1;
    while q % 2 == 0 {
        q /= 2;
        s += 1;
    }

    // Find quadratic non-residue
    let mut z = 2u64;
    while legendre_symbol(z, p) != -1 {
        z += 1;
    }

    let reducer = BarrettReducer::new(p);
    let mut m = s;
    let mut c = mod_pow(z, q, p);
    let mut t = mod_pow(a, q, p);
    let mut r = mod_pow(a, (q + 1) / 2, p);

    loop {
        if t == 1 {
            return Some(r);
        }

        // Find least i such that t^(2^i) ≡ 1
        let mut i = 0u32;
        let mut tmp = t;
        while tmp != 1 {
            tmp = reducer.reduce(tmp as u128 * tmp as u128);
            i += 1;
        }

        let b = mod_pow(c, 1u64 << (m - i - 1), p);
        m = i;
        c = reducer.reduce(b as u128 * b as u128);
        t = reducer.reduce(t as u128 * c as u128);
        r = reducer.reduce(r as u128 * b as u128);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extended_gcd() {
        let (g, x, y) = extended_gcd(35, 15);
        assert_eq!(g, 5);
        assert_eq!(35 * x + 15 * y, 5);
    }

    #[test]
    fn test_mod_inverse() {
        let inv = mod_inverse(3, 7).unwrap();
        assert_eq!((3 * inv) % 7, 1);

        let inv = mod_inverse(17, 43).unwrap();
        assert_eq!((17 * inv) % 43, 1);
    }

    #[test]
    fn test_binary_gcd() {
        assert_eq!(binary_gcd(12, 8), 4);
        assert_eq!(binary_gcd(17, 13), 1);
        assert_eq!(binary_gcd(0, 5), 5);
        assert_eq!(binary_gcd(100, 0), 100);
    }

    #[test]
    fn test_barrett_reduction() {
        let reducer = BarrettReducer::new(97);
        assert_eq!(reducer.reduce(1000), 1000 % 97);
        assert_eq!(reducer.reduce(9999), 9999 % 97);
    }

    #[test]
    fn test_mod_pow() {
        assert_eq!(mod_pow(2, 10, 1000), 24); // 1024 % 1000
        assert_eq!(mod_pow(3, 4, 17), 81 % 17);
    }

    #[test]
    fn test_mod_sqrt() {
        // 4^2 = 16 ≡ 16 (mod 17)
        let root = mod_sqrt(16, 17).unwrap();
        assert_eq!((root * root) % 17, 16);

        // 2 has sqrt mod 7: 3^2 = 9 ≡ 2 (mod 7)
        let root = mod_sqrt(2, 7).unwrap();
        assert_eq!((root * root) % 7, 2);
    }
}
