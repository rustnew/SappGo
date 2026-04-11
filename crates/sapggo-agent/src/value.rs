use ndarray::Array1;
use rand::Rng;

use crate::policy::LinearLayer;

/// Simple two-hidden-layer MLP critic (value function).
///
/// Architecture:
///   obs → Linear(obs_dim, 256) → Tanh
///       → Linear(256, 256)     → Tanh
///       → Linear(256, 1)       (no activation)
#[derive(Debug, Clone)]
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
}
