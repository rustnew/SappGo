use ndarray::{Array1, Array2};
use rand::Rng;
use rand_distr::{Distribution, Normal};

/// Trait for policies that can produce deterministic and stochastic actions.
pub trait Policy {
    /// Selects an action deterministically (mean of the distribution).
    fn act_deterministic(&self, obs: &[f64]) -> Vec<f64>;

    /// Samples a stochastic action and returns `(action, log_prob, value)`.
    fn act_stochastic(&self, obs: &[f64]) -> (Vec<f64>, f64, f64);
}

/// Dense linear layer: y = x * W^T + b
#[derive(Debug, Clone)]
pub struct LinearLayer {
    pub weights: Array2<f64>,
    pub bias:    Array1<f64>,
}

impl LinearLayer {
    /// Xavier-uniform initialisation.
    pub fn new(in_dim: usize, out_dim: usize, rng: &mut impl Rng) -> Self {
        let limit = (6.0 / (in_dim + out_dim) as f64).sqrt();
        let dist  = rand_distr::Uniform::new(-limit, limit);
        let weights = Array2::from_shape_fn((out_dim, in_dim), |_| dist.sample(rng));
        let bias    = Array1::zeros(out_dim);
        Self { weights, bias }
    }

    /// Forward pass for a single sample: `[in_dim] → [out_dim]`.
    #[inline]
    pub fn forward(&self, x: &Array1<f64>) -> Array1<f64> {
        self.weights.dot(x) + &self.bias
    }
}

/// Simple two-hidden-layer MLP actor (Gaussian policy).
///
/// Architecture:
///   obs → Linear(obs_dim, 256) → Tanh
///       → Linear(256, 256)     → Tanh
///       → mean: Linear(256, act_dim)
///       + log_std: learnable vector
#[derive(Debug, Clone)]
pub struct MlpActor {
    fc1:     LinearLayer,
    fc2:     LinearLayer,
    mean:    LinearLayer,
    log_std: Array1<f64>,
    act_dim: usize,
}

impl MlpActor {
    /// Creates a new MLP actor with random weights.
    pub fn new(obs_dim: usize, act_dim: usize, rng: &mut impl Rng) -> Self {
        Self {
            fc1:     LinearLayer::new(obs_dim, 256, rng),
            fc2:     LinearLayer::new(256, 256, rng),
            mean:    LinearLayer::new(256, act_dim, rng),
            log_std: Array1::from_elem(act_dim, -0.5),
            act_dim,
        }
    }

    /// Forward pass returning `(mean, std)` vectors.
    pub fn forward(&self, obs: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        let h = self.fc1.forward(obs).mapv(f64::tanh);
        let h = self.fc2.forward(&h).mapv(f64::tanh);
        let mean = self.mean.forward(&h);
        let std  = self.log_std.mapv(|v| v.clamp(-4.0, 0.0).exp());
        (mean, std)
    }
}

impl Policy for MlpActor {
    fn act_deterministic(&self, obs: &[f64]) -> Vec<f64> {
        let obs_arr = Array1::from(obs.to_vec());
        let (mean, _) = self.forward(&obs_arr);
        mean.to_vec()
    }

    fn act_stochastic(&self, obs: &[f64]) -> (Vec<f64>, f64, f64) {
        let obs_arr = Array1::from(obs.to_vec());
        let (mean, std) = self.forward(&obs_arr);

        let mut rng = rand::thread_rng();
        let mut action  = Vec::with_capacity(self.act_dim);
        let mut log_prob = 0.0f64;

        for i in 0..self.act_dim {
            let dist = Normal::new(mean[i], std[i].max(1e-8))
                .unwrap_or_else(|_| Normal::new(0.0, 1.0).unwrap());
            let a = dist.sample(&mut rng);
            // log π(a|s) = -0.5 * ((a - μ)/σ)^2 - ln(σ) - 0.5 * ln(2π)
            let z = (a - mean[i]) / std[i].max(1e-8);
            log_prob += -0.5 * z * z - std[i].max(1e-8).ln() - 0.5 * (2.0 * std::f64::consts::PI).ln();
            action.push(a);
        }

        // Value estimate is 0 here — the critic provides it separately.
        (action, log_prob, 0.0)
    }
}
