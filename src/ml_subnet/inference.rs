//! Inference Engine — optimized matrix multiplication and attention.
//!
//! Implements tiled GEMM with cache-blocking for CPU inference
//! and CUDA kernel dispatch for GPU inference.

/// Tiled matrix multiplication: C = A * B
/// Blocks the computation to fit L1/L2 cache.
///
/// For matrices A(M×K) and B(K×N), we tile into blocks of size TILE.
/// Each block fits in L1 cache (32KB), avoiding cache thrashing.
pub fn gemm_tiled(
    a: &[f32], // M × K, row-major
    b: &[f32], // K × N, row-major
    c: &mut [f32], // M × N, row-major
    m: usize,
    k: usize,
    n: usize,
) {
    const TILE: usize = 64; // 64×64 f32 = 16KB, fits L1 cache

    // Initialize output
    for v in c.iter_mut() {
        *v = 0.0;
    }

    // Tiled loop: iterate over blocks
    let mut ii = 0;
    while ii < m {
        let i_end = (ii + TILE).min(m);
        let mut jj = 0;
        while jj < n {
            let j_end = (jj + TILE).min(n);
            let mut kk = 0;
            while kk < k {
                let k_end = (kk + TILE).min(k);

                // Micro-kernel: multiply tile A[ii..i_end, kk..k_end] × B[kk..k_end, jj..j_end]
                for i in ii..i_end {
                    for j in jj..j_end {
                        let mut sum = c[i * n + j];
                        for p in kk..k_end {
                            sum += a[i * k + p] * b[p * n + j];
                        }
                        c[i * n + j] = sum;
                    }
                }

                kk += TILE;
            }
            jj += TILE;
        }
        ii += TILE;
    }
}

/// Softmax over a row vector (attention scores).
/// Numerically stable: subtract max before exp.
#[inline]
pub fn softmax(data: &mut [f32]) {
    if data.is_empty() { return; }

    // Find max (branchless-friendly scan)
    let mut max_val = data[0];
    for &v in &data[1..] {
        if v > max_val { max_val = v; }
    }

    // exp(x - max) and sum
    let mut sum: f32 = 0.0;
    for v in data.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }

    // Normalize
    let inv_sum = 1.0 / sum;
    for v in data.iter_mut() {
        *v *= inv_sum;
    }
}

/// Scaled dot-product attention: softmax(Q·K^T / √d) · V
pub fn scaled_dot_product_attention(
    query: &[f32],  // seq_len × d_model
    key: &[f32],    // seq_len × d_model
    value: &[f32],  // seq_len × d_model
    output: &mut [f32], // seq_len × d_model
    seq_len: usize,
    d_model: usize,
) {
    let scale = 1.0 / (d_model as f32).sqrt();

    // Transpose K from (seq_len × d_model) to (d_model × seq_len)
    // so that gemm_tiled computes Q @ K^T correctly.
    let mut key_t = vec![0.0f32; d_model * seq_len];
    for i in 0..seq_len {
        for j in 0..d_model {
            key_t[j * seq_len + i] = key[i * d_model + j];
        }
    }

    // QK^T: (seq_len × d_model) × (d_model × seq_len) = seq_len × seq_len
    let mut scores = vec![0.0f32; seq_len * seq_len];
    gemm_tiled(query, &key_t, &mut scores, seq_len, d_model, seq_len);

    // Scale
    for s in scores.iter_mut() {
        *s *= scale;
    }

    // Softmax per row
    for i in 0..seq_len {
        let row = &mut scores[i * seq_len..(i + 1) * seq_len];
        softmax(row);
    }

    // Attention × V: (seq_len × seq_len) × (seq_len × d_model) = seq_len × d_model
    gemm_tiled(&scores, value, output, seq_len, seq_len, d_model);
}

/// RMS Layer Normalization (used in LLaMA-style models).
#[inline]
pub fn rms_norm(data: &mut [f32], weight: &[f32], eps: f32) {
    let n = data.len();
    let mut sum_sq: f32 = 0.0;
    for &v in data.iter() {
        sum_sq += v * v;
    }
    let rms = (sum_sq / n as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;
    for (d, w) in data.iter_mut().zip(weight.iter()) {
        *d = *d * inv_rms * w;
    }
}

/// SiLU activation (used in LLaMA FFN).
#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemm_tiled() {
        // 2×3 × 3×2 = 2×2
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut c = vec![0.0; 4];
        gemm_tiled(&a, &b, &mut c, 2, 3, 2);
        assert_eq!(c[0], 1.0*7.0 + 2.0*9.0 + 3.0*11.0);  // 58
        assert_eq!(c[1], 1.0*8.0 + 2.0*10.0 + 3.0*12.0); // 64
        assert_eq!(c[2], 4.0*7.0 + 5.0*9.0 + 6.0*11.0);  // 139
        assert_eq!(c[3], 4.0*8.0 + 5.0*10.0 + 6.0*12.0); // 154
    }

    #[test]
    fn test_softmax() {
        let mut data = vec![1.0, 2.0, 3.0];
        softmax(&mut data);
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(data[2] > data[1]);
        assert!(data[1] > data[0]);
    }

    #[test]
    fn test_silu() {
        assert!((silu(0.0) - 0.0).abs() < 1e-6);
        assert!(silu(1.0) > 0.7); // ~0.731
        assert!(silu(-1.0) < 0.0); // ~-0.269
    }
}
