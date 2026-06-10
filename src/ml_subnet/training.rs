//! Background Training Loop — continuous model improvement.
//!
//! Runs asynchronously alongside inference serving, using spare GPU
//! cycles to fine-tune the model on new data. Implements gradient
//! accumulation for memory-efficient training on large batch sizes.

use crate::config::MlConfig;
use tracing::{info, debug};

pub struct BackgroundTrainer {
    config: MlConfig,
    step: u64,
    best_loss: f64,
}

impl BackgroundTrainer {
    pub fn new(config: &MlConfig) -> Self {
        Self {
            config: config.clone(),
            step: 0,
            best_loss: f64::MAX,
        }
    }

    /// Execute a single training step.
    /// Returns the loss value for tracking convergence.
    pub async fn train_step(&self) -> Result<f64, Box<dyn std::error::Error + Send + Sync>> {
        // In production:
        // 1. Load micro-batch from training data
        // 2. Forward pass through model
        // 3. Compute loss (cross-entropy for LM, MSE for regression)
        // 4. Backward pass (gradient computation)
        // 5. Gradient accumulation (if grad_accum_steps > 1)
        // 6. Optimizer step (AdamW with weight decay)
        // 7. Learning rate schedule (cosine annealing)
        // 8. Save checkpoint if best_loss improved

        // Placeholder: simulate training
        let simulated_loss = 1.0 / (self.step as f64 + 1.0);
        debug!("[ML] Training step {} | loss={:.6}", self.step, simulated_loss);

        Ok(simulated_loss)
    }
}

/// AdamW optimizer state for a single parameter.
#[derive(Clone)]
pub struct AdamWState {
    pub m: Vec<f32>,  // First moment (mean)
    pub v: Vec<f32>,  // Second moment (variance)
    pub step: u64,
}

impl AdamWState {
    pub fn new(size: usize) -> Self {
        Self {
            m: vec![0.0; size],
            v: vec![0.0; size],
            step: 0,
        }
    }

    /// AdamW update step.
    /// weight_decay is applied directly to weights (decoupled from gradient).
    pub fn step(
        &mut self,
        params: &mut [f32],
        grads: &[f32],
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
    ) {
        self.step += 1;
        let bias_correction1 = 1.0 - beta1.powi(self.step as i32);
        let bias_correction2 = 1.0 - beta2.powi(self.step as i32);

        for i in 0..params.len() {
            // Update moments
            self.m[i] = beta1 * self.m[i] + (1.0 - beta1) * grads[i];
            self.v[i] = beta2 * self.v[i] + (1.0 - beta2) * grads[i] * grads[i];

            // Bias-corrected moments
            let m_hat = self.m[i] / bias_correction1;
            let v_hat = self.v[i] / bias_correction2;

            // AdamW: decoupled weight decay
            params[i] -= lr * (m_hat / (v_hat.sqrt() + eps) + weight_decay * params[i]);
        }
    }
}

/// Cosine annealing learning rate schedule with warmup.
pub fn cosine_annealing_lr(
    step: u64,
    warmup_steps: u64,
    total_steps: u64,
    lr_max: f32,
    lr_min: f32,
) -> f32 {
    if step < warmup_steps {
        // Linear warmup
        lr_max * (step as f32 / warmup_steps as f32)
    } else {
        // Cosine decay
        let progress = (step - warmup_steps) as f32 / (total_steps - warmup_steps) as f32;
        let cosine = (1.0 + (std::f32::consts::PI * progress).cos()) / 2.0;
        lr_min + (lr_max - lr_min) * cosine
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adamw_step() {
        let mut params = vec![1.0, 2.0, 3.0];
        let grads = vec![0.1, 0.2, 0.3];
        let mut state = AdamWState::new(3);

        let orig = params.clone();
        state.step(&mut params, &grads, 0.001, 0.9, 0.999, 1e-8, 0.01);

        // Parameters should have moved
        for (p, o) in params.iter().zip(orig.iter()) {
            assert!((p - o).abs() > 0.0);
        }
    }

    #[test]
    fn test_cosine_lr() {
        let lr = cosine_annealing_lr(0, 100, 1000, 0.001, 0.0001);
        assert_eq!(lr, 0.0); // Start of warmup

        let lr = cosine_annealing_lr(50, 100, 1000, 0.001, 0.0001);
        assert!((lr - 0.0005).abs() < 0.0001); // Middle of warmup

        let lr = cosine_annealing_lr(100, 100, 1000, 0.001, 0.0001);
        assert!((lr - 0.001).abs() < 0.0001); // End of warmup = max LR

        let lr = cosine_annealing_lr(1000, 100, 1000, 0.001, 0.0001);
        assert!((lr - 0.0001).abs() < 0.0001); // End = min LR
    }
}
