use serde::{Deserialize, Serialize};

/// PPO hyperparameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PpoConfig {
    /// Clipping range for the surrogate objective.
    pub clip_epsilon:   f64,
    /// Discount factor.
    pub gamma:          f64,
    /// GAE lambda (bias-variance trade-off).
    pub gae_lambda:     f64,
    /// Learning rate.
    pub lr:             f64,
    /// Number of PPO update epochs per rollout.
    pub epochs:         usize,
    /// Minibatch size for each gradient step.
    pub minibatch_size: usize,
    /// Value-loss coefficient.
    pub value_coef:     f64,
    /// Entropy bonus coefficient.
    pub entropy_coef:   f64,
    /// Maximum gradient norm for clipping.
    pub max_grad_norm:  f64,
}

impl Default for PpoConfig {
    fn default() -> Self {
        Self {
            clip_epsilon:   0.2,
            gamma:          0.99,
            gae_lambda:     0.95,
            lr:             3e-4,
            epochs:         10,
            minibatch_size: 512,
            value_coef:     0.5,
            entropy_coef:   0.02,
            max_grad_norm:  0.5,
        }
    }
}

/// Statistics returned after a PPO update.
#[derive(Debug, Clone, Default)]
pub struct PpoUpdateStats {
    pub policy_loss: f64,
    pub value_loss:  f64,
    pub entropy:     f64,
    pub clip_frac:   f64,
    pub approx_kl:   f64,
}

/// Computes Generalized Advantage Estimation (GAE).
///
/// Returns `(advantages, returns)`, both of length `n`.
///
/// # Arguments
///
/// - `rewards`:    per-step rewards.
/// - `values`:     per-step value estimates from the critic.
/// - `dones`:      per-step episode termination flags.
/// - `last_value`: bootstrap value for the final state.
/// - `gamma`:      discount factor.
/// - `lam`:        GAE lambda.
pub fn compute_gae(
    rewards:    &[f64],
    values:     &[f64],
    dones:      &[bool],
    last_value: f64,
    gamma:      f64,
    lam:        f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = rewards.len();
    let mut advantages = vec![0.0f64; n];
    let mut returns    = vec![0.0f64; n];
    let mut gae        = 0.0f64;

    for t in (0..n).rev() {
        let next_v = if t + 1 < n { values[t + 1] } else { last_value };
        let mask   = if dones[t] { 0.0 } else { 1.0 };
        let delta  = rewards[t] + gamma * next_v * mask - values[t];
        gae           = delta + gamma * lam * mask * gae;
        advantages[t] = gae;
        returns[t]    = gae + values[t];
    }

    (advantages, returns)
}

/// Normalizes advantages in-place to zero mean and unit variance.
///
/// This is a standard PPO trick to stabilise training.
pub fn normalize_advantages(advantages: &mut [f64]) {
    let n = advantages.len() as f64;
    if n < 2.0 {
        return;
    }

    let mean = advantages.iter().sum::<f64>() / n;
    let var  = advantages.iter().map(|a| (a - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let std  = (var + 1e-8).sqrt();

    for a in advantages.iter_mut() {
        *a = (*a - mean) / std;
    }
}
