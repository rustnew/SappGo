use std::path::Path;

use ndarray::Array1;

use sapggo_agent::{
    PpoConfig, PpoUpdateStats, RolloutBuffer, RunningNormalizer, Transition,
    compute_gae, normalize_advantages,
};
use sapggo_curriculum::CurriculumManager;
use sapggo_env::{SapggoEnv, ACT_DIM, OBS_DIM};

use crate::checkpoint;
use crate::config::TrainConfig;
use crate::logger::TrainingLogger;

/// Main training loop orchestrating environment interaction, PPO updates,
/// curriculum progression, logging, and checkpointing.
pub struct Trainer {
    env:         SapggoEnv,
    curriculum:  CurriculumManager,
    config:      TrainConfig,
    ppo_config:  PpoConfig,
    logger:      TrainingLogger,
    normalizer:  RunningNormalizer,
    global_step: u64,
    episode:     u64,
}

impl Trainer {
    /// Initializes the trainer from a configuration.
    pub fn new(config: TrainConfig) -> anyhow::Result<Self> {
        let env = SapggoEnv::new(&config.model_path, config.seed)?;

        let logger = TrainingLogger::new(Path::new(&config.log_dir))?;
        checkpoint::ensure_checkpoint_dir(Path::new(&config.checkpoint_dir))?;

        let ppo_config = config.to_ppo_config();

        Ok(Self {
            env,
            curriculum:  CurriculumManager::new(),
            ppo_config,
            logger,
            normalizer:  RunningNormalizer::new(OBS_DIM),
            global_step: 0,
            episode:     0,
            config,
        })
    }

    /// Runs the full training loop until `total_steps` is reached.
    pub fn run(&mut self) -> anyhow::Result<()> {
        let mut obs = self.env.reset();

        tracing::info!(
            total_steps = self.config.total_steps,
            rollout     = self.config.rollout_steps,
            "Training started",
        );

        while self.global_step < self.config.total_steps {
            // 1. Collect rollout
            let rollout = self.collect_rollout(&mut obs)?;
            self.global_step += rollout.len() as u64;

            // 2. Compute advantages
            let slices = rollout.as_slices();
            let (mut advantages, returns) = compute_gae(
                &slices.rewards,
                &slices.values,
                &slices.dones,
                0.0, // bootstrap with 0 if episode ended
                self.ppo_config.gamma,
                self.ppo_config.gae_lambda,
            );
            normalize_advantages(&mut advantages);

            // 3. PPO update (placeholder — actual gradient computation
            //    requires Burn autodiff backend integration)
            let stats = self.ppo_update_placeholder(&rollout, &advantages, &returns);

            // 4. Curriculum: check promotions
            for ep_r in &rollout.episode_rewards {
                self.episode += 1;
                if let Some(stage) = self.curriculum.on_episode_end(*ep_r) {
                    tracing::info!(
                        new_stage = stage.name(),
                        episode   = self.episode,
                        "Applying new curriculum parameters",
                    );
                    self.env.set_params(self.curriculum.current_params())?;
                }
            }

            // 5. Logging
            if self.episode > 0 && self.episode % self.config.log_interval == 0 {
                let n_eps = rollout.episode_rewards.len().max(1) as f64;
                let mean_r = rollout.episode_rewards.iter().sum::<f64>() / n_eps;

                self.logger.log("train/reward_mean",  mean_r,            self.global_step);
                self.logger.log("train/policy_loss",  stats.policy_loss, self.global_step);
                self.logger.log("train/value_loss",   stats.value_loss,  self.global_step);
                self.logger.log("train/entropy",      stats.entropy,     self.global_step);
                self.logger.log(
                    "curriculum/stage",
                    self.curriculum.stage.index() as f64,
                    self.global_step,
                );

                tracing::info!(
                    step        = self.global_step,
                    stage       = self.curriculum.stage.name(),
                    mean_reward = format!("{mean_r:.2}"),
                    episodes    = self.episode,
                    "Training update",
                );

                self.logger.flush();
            }

            // 6. Checkpoint
            if self.global_step % self.config.checkpoint_interval == 0 {
                let path = checkpoint::checkpoint_path(
                    Path::new(&self.config.checkpoint_dir),
                    self.global_step,
                );
                tracing::info!(path = %path.display(), "Saving checkpoint");
                // In production, serialize the actual Burn model records here.
            }
        }

        // Final checkpoint
        let final_path = checkpoint::final_checkpoint_path(
            Path::new(&self.config.checkpoint_dir),
        );
        tracing::info!(
            total_steps = self.global_step,
            path        = %final_path.display(),
            "Training complete — saving final checkpoint",
        );

        Ok(())
    }

    /// Collects one rollout of `rollout_steps` transitions.
    fn collect_rollout(&mut self, obs: &mut Array1<f64>) -> anyhow::Result<RolloutBuffer> {
        let mut buffer = RolloutBuffer::new(self.config.rollout_steps);

        while !buffer.is_full() {
            // Normalize observation for the policy
            let obs_vec = obs.to_vec();
            self.normalizer.update(&obs_vec);
            let norm_obs = self.normalizer.normalize(&obs_vec);

            // Sample action (placeholder: random uniform [-1, 1])
            let (action, log_prob, value) = self.sample_action_placeholder(&norm_obs);

            let result = self.env.step(&action)?;

            buffer.push(Transition {
                observation: obs_vec,
                action,
                log_prob,
                value,
                reward: result.reward,
                done:   result.done,
            });

            if result.done {
                *obs = self.env.reset();
            } else {
                *obs = result.observation;
            }
        }

        Ok(buffer)
    }

    /// Placeholder for action sampling.
    ///
    /// In the full implementation, this runs the Actor network forward pass,
    /// samples from the Gaussian distribution, and evaluates the Critic.
    fn sample_action_placeholder(&self, _obs: &[f64]) -> (Vec<f64>, f64, f64) {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let action: Vec<f64> = (0..ACT_DIM).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let log_prob = -(ACT_DIM as f64) * (2.0_f64).ln(); // uniform log-prob
        let value    = 0.0;
        (action, log_prob, value)
    }

    /// Placeholder for the PPO gradient update.
    ///
    /// In the full implementation, this performs multiple epochs of minibatch
    /// gradient descent on the clipped surrogate + value + entropy objectives
    /// using Burn's autodiff backend.
    fn ppo_update_placeholder(
        &self,
        _rollout:    &RolloutBuffer,
        _advantages: &[f64],
        _returns:    &[f64],
    ) -> PpoUpdateStats {
        PpoUpdateStats::default()
    }
}
