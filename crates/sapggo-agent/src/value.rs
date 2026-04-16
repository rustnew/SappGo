use ndarray::Array1;
use rand::Rng;
use serde::{Serialize, Deserialize};

use crate::policy::LinearLayer;

/// Simple two-hidden-layer MLP critic (value function).
///
/// Architecture:
///   obs → Linear(obs_dim, 256) → Tanh
///       → Linear(256, 256)     → Tanh
///       → Linear(256, 1)       (no activation)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlpCritic {
    fc1: LinearLayer,
    fc2: LinearLayer,
    out: LinearLayer,
}

impl MlpCritic {
    /// Creates a new MLP critic with random weights.
    pub fn new(obs_dim: usize, rng: &mut impl Rng) -> Self {
        Self {
            fc1: LinearLayer::new(obs_dim, 256, rng),
            fc2: LinearLayer::new(256, 256, rng),
            out: LinearLayer::new(256, 1, rng),
        }
    }

    /// Forward pass returning the scalar value estimate.
    #[inline]
    pub fn forward(&self, obs: &Array1<f64>) -> f64 {
        let h = self.fc1.forward(obs).mapv(f64::tanh);
        let h = self.fc2.forward(&h).mapv(f64::tanh);
        self.out.forward(&h)[0]
    }

    /// Forward + backward SGD step for the value function.
    ///
    /// Computes V(obs), then backprops the gradient of 0.5*(V - target)^2.
    pub fn update_sgd(&mut self, obs: &Array1<f64>, target: f64, lr: f64) -> f64 {
        // Forward pass with caching
        let z1  = self.fc1.forward(obs);
        let h1  = z1.mapv(f64::tanh);
        let z2  = self.fc2.forward(&h1);
        let h2  = z2.mapv(f64::tanh);
        let v   = self.out.forward(&h2)[0];

        // dL/dv = (v - target) for MSE loss
        let d_v = v - target;
        let d_out_input = Array1::from(vec![d_v]);

        // Backprop through out → h2
        let d_h2 = self.out.backward_sgd(&d_out_input, &h2, lr);
        let d_z2 = &d_h2 * &h2.mapv(|h| 1.0 - h * h);
        let d_h1 = self.fc2.backward_sgd(&d_z2, &h1, lr);
        let d_z1 = &d_h1 * &h1.mapv(|h| 1.0 - h * h);
        let _ = self.fc1.backward_sgd(&d_z1, obs, lr);

        v
    }
}
