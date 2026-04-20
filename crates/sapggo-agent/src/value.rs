use ndarray::Array1;
use rand::Rng;
use serde::{Serialize, Deserialize};

use crate::policy::{CriticGradBuffer, LayerNorm, LinearLayer, apply_ln_grad, ln_backward_sgd};

/// Three-hidden-layer MLP critic with LayerNorm (value function).
///
/// Architecture:
///   obs → Linear(obs_dim, 256) → LayerNorm → Tanh
///       → Linear(256, 256)     → LayerNorm → Tanh
///       → Linear(256, 128)     → LayerNorm → Tanh
///       → Linear(128, 1)       (no activation)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlpCritic {
    fc1: LinearLayer,
    ln1: LayerNorm,
    fc2: LinearLayer,
    ln2: LayerNorm,
    fc3: LinearLayer,
    ln3: LayerNorm,
    out: LinearLayer,
}

impl MlpCritic {
    /// Creates a new MLP critic with random weights.
    pub fn new(obs_dim: usize, rng: &mut impl Rng) -> Self {
        Self {
            fc1: LinearLayer::new(obs_dim, 256, rng),
            ln1: LayerNorm::new(256),
            fc2: LinearLayer::new(256, 256, rng),
            ln2: LayerNorm::new(256),
            fc3: LinearLayer::new(256, 128, rng),
            ln3: LayerNorm::new(128),
            out: LinearLayer::new(128, 1, rng),
        }
    }

    /// Forward pass returning the scalar value estimate.
    #[inline]
    pub fn forward(&self, obs: &Array1<f64>) -> f64 {
        let h = self.ln1.forward(&self.fc1.forward(obs)).mapv(f64::tanh);
        let h = self.ln2.forward(&self.fc2.forward(&h)).mapv(f64::tanh);
        let h = self.ln3.forward(&self.fc3.forward(&h)).mapv(f64::tanh);
        self.out.forward(&h)[0]
    }

    /// Forward + backward SGD step for the value function.
    ///
    /// Computes V(obs), then backprops the gradient of 0.5*(V - target)^2.
    pub fn update_sgd(&mut self, obs: &Array1<f64>, target: f64, lr: f64) -> f64 {
        let z1 = self.fc1.forward(obs);
        let (n1, c1) = self.ln1.forward_with_cache(&z1);
        let h1 = n1.mapv(f64::tanh);

        let z2 = self.fc2.forward(&h1);
        let (n2, c2) = self.ln2.forward_with_cache(&z2);
        let h2 = n2.mapv(f64::tanh);

        let z3 = self.fc3.forward(&h2);
        let (n3, c3) = self.ln3.forward_with_cache(&z3);
        let h3 = n3.mapv(f64::tanh);

        let v = self.out.forward(&h3)[0];

        let d_v = v - target;
        let d_out_input = Array1::from(vec![d_v]);

        let d_h3 = self.out.backward_sgd(&d_out_input, &h3, lr);
        let d_n3 = &d_h3 * &h3.mapv(|h| 1.0 - h * h);
        let d_z3 = ln_backward_sgd(&d_n3, &c3, &mut self.ln3, lr);
        let d_h2 = self.fc3.backward_sgd(&d_z3, &h2, lr);
        let d_n2 = &d_h2 * &h2.mapv(|h| 1.0 - h * h);
        let d_z2 = ln_backward_sgd(&d_n2, &c2, &mut self.ln2, lr);
        let d_h1 = self.fc2.backward_sgd(&d_z2, &h1, lr);
        let d_n1 = &d_h1 * &h1.mapv(|h| 1.0 - h * h);
        let d_z1 = ln_backward_sgd(&d_n1, &c1, &mut self.ln1, lr);
        let _ = self.fc1.backward_sgd(&d_z1, obs, lr);

        v
    }

    /// Forward pass + accumulate gradient of MSE loss into buffer (no weight update).
    /// Returns V(obs).
    pub fn accumulate_grad(&self, obs: &Array1<f64>, target: f64, grad: &mut CriticGradBuffer) -> f64 {
        let z1 = self.fc1.forward(obs);
        let (n1, c1) = self.ln1.forward_with_cache(&z1);
        let h1 = n1.mapv(f64::tanh);

        let z2 = self.fc2.forward(&h1);
        let (n2, c2) = self.ln2.forward_with_cache(&z2);
        let h2 = n2.mapv(f64::tanh);

        let z3 = self.fc3.forward(&h2);
        let (n3, c3) = self.ln3.forward_with_cache(&z3);
        let h3 = n3.mapv(f64::tanh);

        let v = self.out.forward(&h3)[0];

        let d_v = v - target;
        let d_out_input = Array1::from(vec![d_v]);

        let d_h3 = grad.out.accumulate(&d_out_input, &h3, &self.out.weights);
        let d_n3 = &d_h3 * &h3.mapv(|h| 1.0 - h * h);
        let d_z3 = grad.ln3.accumulate(&d_n3, &c3, &self.ln3.gamma);
        let d_h2 = grad.fc3.accumulate(&d_z3, &h2, &self.fc3.weights);
        let d_n2 = &d_h2 * &h2.mapv(|h| 1.0 - h * h);
        let d_z2 = grad.ln2.accumulate(&d_n2, &c2, &self.ln2.gamma);
        let d_h1 = grad.fc2.accumulate(&d_z2, &h1, &self.fc2.weights);
        let d_n1 = &d_h1 * &h1.mapv(|h| 1.0 - h * h);
        let d_z1 = grad.ln1.accumulate(&d_n1, &c1, &self.ln1.gamma);
        let _ = grad.fc1.accumulate(&d_z1, obs, &self.fc1.weights);

        v
    }

    /// Apply accumulated gradients and zero the buffer.
    pub fn apply_grad(&mut self, grad: &mut CriticGradBuffer, n: f64, lr: f64) {
        self.out.apply_grad(&grad.out, n, lr);
        apply_ln_grad(&mut self.ln3, &grad.ln3, n, lr);
        self.fc3.apply_grad(&grad.fc3, n, lr);
        apply_ln_grad(&mut self.ln2, &grad.ln2, n, lr);
        self.fc2.apply_grad(&grad.fc2, n, lr);
        apply_ln_grad(&mut self.ln1, &grad.ln1, n, lr);
        self.fc1.apply_grad(&grad.fc1, n, lr);
        grad.reset();
    }
}
