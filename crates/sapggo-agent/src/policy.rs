use ndarray::{Array1, Array2};
use rand::Rng;
use rand_distr::{Distribution, Normal};
use serde::{Serialize, Deserialize};

/// Gradient accumulation buffer for a LinearLayer.
pub struct LayerGrad {
    pub dw: Array2<f64>,
    pub db: Array1<f64>,
}

impl LayerGrad {
    pub fn zeros(out_dim: usize, in_dim: usize) -> Self {
        Self {
            dw: Array2::zeros((out_dim, in_dim)),
            db: Array1::zeros(out_dim),
        }
    }

    pub fn reset(&mut self) {
        self.dw.fill(0.0);
        self.db.fill(0.0);
    }

    /// Accumulate gradient: dw += d_out ⊗ input, db += d_out.
    /// Returns d_input = W^T · d_out for upstream backprop.
    pub fn accumulate(&mut self, d_out: &Array1<f64>, input: &Array1<f64>, weights: &Array2<f64>) -> Array1<f64> {
        let d_input = weights.t().dot(d_out);
        for i in 0..d_out.len() {
            self.db[i] += d_out[i];
            for j in 0..input.len() {
                self.dw[[i, j]] += d_out[i] * input[j];
            }
        }
        d_input
    }
}

/// Full gradient accumulation buffer for MlpActor.
pub struct ActorGradBuffer {
    pub fc1:     LayerGrad,
    pub fc2:     LayerGrad,
    pub mean:    LayerGrad,
    pub log_std: Array1<f64>,
}

impl ActorGradBuffer {
    pub fn new(obs_dim: usize, act_dim: usize) -> Self {
        Self {
            fc1:     LayerGrad::zeros(256, obs_dim),
            fc2:     LayerGrad::zeros(256, 256),
            mean:    LayerGrad::zeros(act_dim, 256),
            log_std: Array1::zeros(act_dim),
        }
    }

    pub fn reset(&mut self) {
        self.fc1.reset();
        self.fc2.reset();
        self.mean.reset();
        self.log_std.fill(0.0);
    }
}

/// Full gradient accumulation buffer for MlpCritic.
pub struct CriticGradBuffer {
    pub fc1: LayerGrad,
    pub fc2: LayerGrad,
    pub out: LayerGrad,
}

impl CriticGradBuffer {
    pub fn new(obs_dim: usize) -> Self {
        Self {
            fc1: LayerGrad::zeros(256, obs_dim),
            fc2: LayerGrad::zeros(256, 256),
            out: LayerGrad::zeros(1, 256),
        }
    }

    pub fn reset(&mut self) {
        self.fc1.reset();
        self.fc2.reset();
        self.out.reset();
    }
}

/// Trait for policies that can produce deterministic and stochastic actions.
pub trait Policy {
    /// Selects an action deterministically (mean of the distribution).
    fn act_deterministic(&self, obs: &[f64]) -> Vec<f64>;

    /// Samples a stochastic action and returns `(action, log_prob, value)`.
    fn act_stochastic(&self, obs: &[f64]) -> (Vec<f64>, f64, f64);
}

/// Dense linear layer: y = x * W^T + b
#[derive(Debug, Clone, Serialize, Deserialize)]
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

    /// SGD update: W -= lr * dL/dW, b -= lr * dL/db
    ///
    /// `d_out` is the gradient of the loss w.r.t. this layer's output [out_dim].
    /// `input` is the input that was fed to this layer [in_dim].
    ///
    /// Returns the gradient w.r.t. the input [in_dim] for upstream backprop.
    pub fn backward_sgd(&mut self, d_out: &Array1<f64>, input: &Array1<f64>, lr: f64) -> Array1<f64> {
        // dL/dinput = W^T · d_out
        let d_input = self.weights.t().dot(d_out);
        // SGD step with per-element gradient clipping (max_grad_norm = 1.0)
        let max_g = 1.0;
        for i in 0..d_out.len() {
            let g_b = d_out[i].clamp(-max_g, max_g);
            self.bias[i] -= lr * g_b;
            for j in 0..input.len() {
                let g_w = (d_out[i] * input[j]).clamp(-max_g, max_g);
                self.weights[[i, j]] -= lr * g_w;
            }
        }
        d_input
    }

    /// Apply accumulated gradients with clipping: W -= lr * clip(dW/n), b -= lr * clip(db/n).
    pub fn apply_grad(&mut self, grad: &LayerGrad, n: f64, lr: f64) {
        let max_g = 1.0;
        let inv_n = 1.0 / n.max(1.0);
        for i in 0..self.bias.len() {
            let g_b = (grad.db[i] * inv_n).clamp(-max_g, max_g);
            self.bias[i] -= lr * g_b;
            for j in 0..self.weights.ncols() {
                let g_w = (grad.dw[[i, j]] * inv_n).clamp(-max_g, max_g);
                self.weights[[i, j]] -= lr * g_w;
            }
        }
    }
}

/// Simple two-hidden-layer MLP actor (Gaussian policy).
///
/// Architecture:
///   obs → Linear(obs_dim, 256) → Tanh
///       → Linear(256, 256)     → Tanh
///       → mean: Linear(256, act_dim)
///       + log_std: learnable vector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlpActor {
    fc1:     LinearLayer,
    fc2:     LinearLayer,
    mean:    LinearLayer,
    pub log_std: Array1<f64>,
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

    /// Forward pass that also returns intermediate activations for backprop.
    pub fn forward_with_cache(&self, obs: &Array1<f64>) -> ActorCache {
        let z1  = self.fc1.forward(obs);
        let h1  = z1.mapv(f64::tanh);
        let z2  = self.fc2.forward(&h1);
        let h2  = z2.mapv(f64::tanh);
        let mean = self.mean.forward(&h2);
        let std  = self.log_std.mapv(|v| v.clamp(-4.0, 0.0).exp());
        ActorCache { obs: obs.clone(), h1, h2, mean, std }
    }

    /// Backprop through the actor given dL/d(mean) and dL/d(log_std).
    pub fn backward_sgd(
        &mut self,
        cache: &ActorCache,
        d_mean: &Array1<f64>,
        d_log_std: &Array1<f64>,
        lr: f64,
    ) {
        // Update log_std
        for i in 0..self.act_dim {
            if self.log_std[i] >= -4.0 && self.log_std[i] <= 0.0 {
                self.log_std[i] -= lr * d_log_std[i];
                self.log_std[i] = self.log_std[i].clamp(-4.0, 0.0);
            }
        }
        // Backprop through mean layer → h2
        let d_h2 = self.mean.backward_sgd(d_mean, &cache.h2, lr);
        // tanh derivative: d_z2 = d_h2 * (1 - h2^2)
        let d_z2 = &d_h2 * &cache.h2.mapv(|h| 1.0 - h * h);
        // Backprop through fc2 → h1
        let d_h1 = self.fc2.backward_sgd(&d_z2, &cache.h1, lr);
        let d_z1 = &d_h1 * &cache.h1.mapv(|h| 1.0 - h * h);
        // Backprop through fc1 (updates weights, gradient to input discarded)
        let _ = self.fc1.backward_sgd(&d_z1, &cache.obs, lr);
    }

    /// Accumulate gradients into buffer (no weight update).
    pub fn accumulate_grad(
        &self,
        cache: &ActorCache,
        d_mean: &Array1<f64>,
        d_log_std: &Array1<f64>,
        grad: &mut ActorGradBuffer,
    ) {
        for i in 0..self.act_dim {
            grad.log_std[i] += d_log_std[i];
        }
        let d_h2 = grad.mean.accumulate(d_mean, &cache.h2, &self.mean.weights);
        let d_z2 = &d_h2 * &cache.h2.mapv(|h| 1.0 - h * h);
        let d_h1 = grad.fc2.accumulate(&d_z2, &cache.h1, &self.fc2.weights);
        let d_z1 = &d_h1 * &cache.h1.mapv(|h| 1.0 - h * h);
        let _ = grad.fc1.accumulate(&d_z1, &cache.obs, &self.fc1.weights);
    }

    /// Apply accumulated gradients and zero the buffer.
    pub fn apply_grad(&mut self, grad: &mut ActorGradBuffer, n: f64, lr: f64) {
        let max_g = 1.0;
        let inv_n = 1.0 / n.max(1.0);
        // Apply log_std gradient
        for i in 0..self.act_dim {
            let g = (grad.log_std[i] * inv_n).clamp(-max_g, max_g);
            self.log_std[i] -= lr * g;
            self.log_std[i] = self.log_std[i].clamp(-4.0, 0.0);
        }
        self.mean.apply_grad(&grad.mean, n, lr);
        self.fc2.apply_grad(&grad.fc2, n, lr);
        self.fc1.apply_grad(&grad.fc1, n, lr);
        grad.reset();
    }
}

/// Cached activations from an actor forward pass, used for backpropagation.
pub struct ActorCache {
    pub obs:  Array1<f64>,
    pub h1:   Array1<f64>,
    pub h2:   Array1<f64>,
    pub mean: Array1<f64>,
    pub std:  Array1<f64>,
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
