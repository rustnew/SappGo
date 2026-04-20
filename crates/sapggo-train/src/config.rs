use std::path::Path;

use serde::{Deserialize, Serialize};

/// Full training configuration, loaded from a TOML file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainConfig {
    /// Path to the MuJoCo XML model.
    pub model_path:          String,
    /// Random seed for reproducibility.
    pub seed:                u64,
    /// Total environment steps to train.
    pub total_steps:         u64,
    /// Steps collected per rollout before each PPO update.
    pub rollout_steps:       usize,
    /// Whether to use curriculum learning.
    pub curriculum:          bool,
    /// PPO clip epsilon.
    pub clip_epsilon:        f64,
    /// Discount factor.
    pub gamma:               f64,
    /// GAE lambda.
    pub gae_lambda:          f64,
    /// Learning rate.
    pub lr:                  f64,
    /// PPO epochs per update.
    pub epochs:              usize,
    /// Minibatch size.
    pub minibatch_size:      usize,
    /// Value loss coefficient.
    pub value_coef:          f64,
    /// Entropy bonus coefficient.
    pub entropy_coef:        f64,
    /// Max gradient norm for clipping.
    pub max_grad_norm:       f64,
    /// Log metrics every N episodes.
    pub log_interval:        u64,
    /// Save checkpoint every N global steps.
    pub checkpoint_interval: u64,
    /// Directory for saved checkpoints.
    pub checkpoint_dir:      String,
    /// Directory for training logs.
    pub log_dir:             String,
    /// Number of parallel environments for rollout collection.
    pub num_envs:            usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            model_path:          "assets/robot_humanoid_load.xml".into(),
            seed:                42,
            total_steps:         20_000_000,
            rollout_steps:       4096,
            curriculum:          true,
            clip_epsilon:        0.2,
            gamma:               0.99,
            gae_lambda:          0.95,
            lr:                  3e-4,
            epochs:              10,
            minibatch_size:      512,
            value_coef:          0.5,
            entropy_coef:        0.01,
            max_grad_norm:       0.5,
            log_interval:        10,
            checkpoint_interval: 100_000,
            checkpoint_dir:      "checkpoints".into(),
            log_dir:             "runs".into(),
            num_envs:            8,
        }
    }
}

impl TrainConfig {
    /// Loads the configuration from a TOML file.
    pub fn from_file(path: &Path) -> anyhow::Result<Self> {
        let contents = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("Failed to read config file '{}': {}", path.display(), e))?;
        let config: Self = toml::from_str(&contents)
            .map_err(|e| anyhow::anyhow!("Failed to parse config '{}': {}", path.display(), e))?;
        Ok(config)
    }

    /// Converts to a `PpoConfig` for the agent.
    pub fn to_ppo_config(&self) -> sapggo_agent::PpoConfig {
        sapggo_agent::PpoConfig {
            clip_epsilon:   self.clip_epsilon,
            gamma:          self.gamma,
            gae_lambda:     self.gae_lambda,
            lr:             self.lr,
            epochs:         self.epochs,
            minibatch_size: self.minibatch_size,
            value_coef:     self.value_coef,
            entropy_coef:   self.entropy_coef,
            max_grad_norm:  self.max_grad_norm,
        }
    }
}
