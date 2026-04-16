use std::path::Path;

use ndarray::Array1;
use rand::SeedableRng;
use rand::rngs::StdRng;

use sapggo_agent::{
    ActorCache, MlpActor, MlpCritic, Policy,
    PpoConfig, PpoUpdateStats, RolloutBuffer, RunningNormalizer, Transition,
    compute_gae, normalize_advantages,
};
use sapggo_curriculum::CurriculumManager;
use sapggo_env::{SapggoEnv, ACT_DIM, OBS_DIM};

use crate::checkpoint;
use crate::config::TrainConfig;
use crate::episode_log::EpisodeLog;
use crate::logger::TrainingLogger;

/// Main training loop orchestrating environment interaction, PPO updates,
/// curriculum progression, logging, and checkpointing.
pub struct Trainer {
    env:          SapggoEnv,
    actor:        MlpActor,
    critic:       MlpCritic,
    curriculum:   CurriculumManager,
    config:       TrainConfig,
    ppo_config:   PpoConfig,
    logger:       TrainingLogger,
    episode_log:  EpisodeLog,
    normalizer:   RunningNormalizer,
    #[allow(dead_code)]
    rng:          StdRng,
    global_step:  u64,
    episode:      u64,
    best_reward:  f64,
    best_episode: u64,
}

impl Trainer {
    /// Initializes the trainer from a configuration.
    pub fn new(config: TrainConfig) -> anyhow::Result<Self> {
        let env = SapggoEnv::new(&config.model_path, config.seed)?;

        let mut rng = StdRng::seed_from_u64(config.seed);
        let actor  = MlpActor::new(OBS_DIM, ACT_DIM, &mut rng);
        let critic = MlpCritic::new(OBS_DIM, &mut rng);

        let logger      = TrainingLogger::new(Path::new(&config.log_dir))?;
        let episode_log = EpisodeLog::new(Path::new(&config.log_dir))?;
        checkpoint::ensure_checkpoint_dir(Path::new(&config.checkpoint_dir))?;

        let ppo_config = config.to_ppo_config();

        Ok(Self {
            env,
            actor,
            critic,
            curriculum:   CurriculumManager::new(),
            ppo_config,
            logger,
            episode_log,
            normalizer:   RunningNormalizer::new(OBS_DIM),
            rng,
            global_step:  0,
            episode:      0,
            best_reward:  f64::NEG_INFINITY,
            best_episode: 0,
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

            // 3. PPO update (forward-pass loss computation + log_std tuning)
            let stats = self.ppo_update(&rollout, &advantages, &returns);

            // 4. Aggregate logging (metrics.csv)
            let n_eps = rollout.episode_rewards.len().max(1) as f64;
            let mean_r = rollout.episode_rewards.iter().sum::<f64>() / n_eps;

            self.logger.log("train/reward_mean",  mean_r,            self.global_step);
            self.logger.log("train/policy_loss",  stats.policy_loss, self.global_step);
            self.logger.log("train/value_loss",   stats.value_loss,  self.global_step);
            self.logger.log("train/entropy",      stats.entropy,     self.global_step);
            self.logger.log("train/best_reward",  self.best_reward,  self.global_step);
            self.logger.log(
                "curriculum/stage",
                self.curriculum.stage.index() as f64,
                self.global_step,
            );
            self.logger.flush();

            // 5. Periodic checkpoint
            if self.global_step % self.config.checkpoint_interval == 0 {
                let path = checkpoint::checkpoint_path(
                    Path::new(&self.config.checkpoint_dir),
                    self.global_step,
                );
                let _ = checkpoint::save_checkpoint(&self.actor, &path);
            }
        }

        // Final checkpoint
        let final_path = checkpoint::final_checkpoint_path(
            Path::new(&self.config.checkpoint_dir),
        );
        let _ = checkpoint::save_checkpoint(&self.actor, &final_path);
        self.episode_log.flush();
        tracing::info!(
            total_steps  = self.global_step,
            episodes     = self.episode,
            best_reward  = format!("{:.2}", self.best_reward),
            best_episode = self.best_episode,
            path         = %final_path.display(),
            "Training complete — best params saved in checkpoints/best_actor.json",
        );

        Ok(())
    }

    /// Collects one rollout of `rollout_steps` transitions.
    ///
    /// Each time an episode ends (load dropped, robot fallen, timeout),
    /// the episode is logged to `episodes.csv` and the agent restarts
    /// instantly. If a new best reward is achieved, the actor weights
    /// are saved as `best_actor.json`.
    fn collect_rollout(&mut self, obs: &mut Array1<f64>) -> anyhow::Result<RolloutBuffer> {
        let mut buffer = RolloutBuffer::new(self.config.rollout_steps);

        while !buffer.is_full() {
            // Normalize observation for the policy
            let obs_vec = obs.to_vec();
            self.normalizer.update(&obs_vec);
            let norm_obs = self.normalizer.normalize(&obs_vec);

            // Forward pass through actor (stochastic) and critic
            let (action, log_prob, _) = self.actor.act_stochastic(&norm_obs);
            let value = self.critic.forward(&Array1::from(norm_obs.clone()));

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
                // ── Log every single episode ──
                self.episode += 1;
                let info = result.info;
                let is_best = info.total_reward > self.best_reward;
                if is_best {
                    self.best_reward  = info.total_reward;
                    self.best_episode = self.episode;
                    // Save best actor weights
                    let best_path = Path::new(&self.config.checkpoint_dir).join("best_actor.json");
                    let _ = checkpoint::save_checkpoint(&self.actor, &best_path);
                    let best_critic_path = Path::new(&self.config.checkpoint_dir).join("best_critic.json");
                    let _ = checkpoint::save_checkpoint(&self.critic, &best_critic_path);
                }

                self.episode_log.log_episode(
                    self.episode,
                    self.global_step + buffer.len() as u64,
                    info.steps,
                    info.total_reward,
                    info.distance_m,
                    info.load_dropped,
                    info.robot_fallen,
                    self.best_reward,
                    is_best,
                    self.curriculum.stage.name(),
                );

                // Print every episode to terminal
                let status = if info.load_dropped {
                    "DROPPED"
                } else if info.robot_fallen {
                    "FALLEN"
                } else {
                    "TIMEOUT"
                };
                tracing::info!(
                    ep      = self.episode,
                    status  = status,
                    steps   = info.steps,
                    reward  = format!("{:.2}", info.total_reward),
                    dist    = format!("{:.2}m", info.distance_m),
                    best    = format!("{:.2}", self.best_reward),
                    best_ep = self.best_episode,
                    "Episode done → instant restart",
                );

                // Curriculum check
                if let Some(stage) = self.curriculum.on_episode_end(info.total_reward) {
                    tracing::info!(
                        new_stage = stage.name(),
                        episode   = self.episode,
                        "Curriculum promotion!",
                    );
                    self.env.set_params(self.curriculum.current_params())?;
                }

                // Flush episode log every 50 episodes
                if self.episode % 50 == 0 {
                    self.episode_log.flush();
                }

                // ── Instant restart ──
                *obs = self.env.reset();
            } else {
                *obs = result.observation;
            }
        }

        Ok(buffer)
    }

    /// PPO update with manual backpropagation through all network layers.
    ///
    /// For each sample in the rollout:
    ///   1. Forward pass through actor (with cache for backprop).
    ///   2. Compute PPO clipped surrogate loss gradient w.r.t. mean and log_std.
    ///   3. Backprop through actor layers with SGD.
    ///   4. Forward+backward SGD on critic for value loss.
    ///
    /// Runs `ppo_epochs` passes over the data for stable learning.
    fn ppo_update(
        &mut self,
        rollout:    &RolloutBuffer,
        advantages: &[f64],
        returns:    &[f64],
    ) -> PpoUpdateStats {
        let slices = rollout.as_slices();
        let n = slices.rewards.len();
        if n == 0 {
            return PpoUpdateStats::default();
        }

        let lr        = self.ppo_config.lr;
        let clip_eps  = self.ppo_config.clip_epsilon;
        let epochs    = self.ppo_config.epochs.max(1);
        let nf        = n as f64;

        let mut total_policy_loss = 0.0f64;
        let mut total_value_loss  = 0.0f64;
        let mut total_entropy     = 0.0f64;
        let mut total_clip_frac   = 0.0f64;
        let mut total_kl          = 0.0f64;

        for _epoch in 0..epochs {
            let mut epoch_ploss = 0.0f64;
            let mut epoch_vloss = 0.0f64;
            let mut epoch_ent   = 0.0f64;
            let mut epoch_clip  = 0.0f64;
            let mut epoch_kl    = 0.0f64;

            for i in 0..n {
                let norm_obs = self.normalizer.normalize(&slices.observations[i]);
                let obs_arr  = Array1::from(norm_obs);
                let action   = &slices.actions[i];

                // ── Actor forward with cache ──
                let cache = self.actor.forward_with_cache(&obs_arr);

                // Compute log_prob and entropy under current policy
                let mut new_log_prob = 0.0f64;
                let mut ent = 0.0f64;
                for j in 0..ACT_DIM {
                    let s = cache.std[j].max(1e-8);
                    let z = (action[j] - cache.mean[j]) / s;
                    new_log_prob += -0.5 * z * z - s.ln()
                        - 0.5 * (2.0 * std::f64::consts::PI).ln();
                    ent += 0.5 * (2.0 * std::f64::consts::PI * std::f64::consts::E * s * s).ln();
                }

                // PPO ratio and clipped surrogate
                let ratio = (new_log_prob - slices.log_probs[i]).exp();
                let clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps);
                let adv = advantages[i];
                let surr1 = ratio * adv;
                let surr2 = clipped * adv;
                let use_clipped = surr2 < surr1;
                epoch_ploss += -surr1.min(surr2);
                epoch_clip  += if use_clipped { 1.0 } else { 0.0 };
                epoch_kl    += slices.log_probs[i] - new_log_prob;
                epoch_ent   += ent;

                // ── Gradient of -min(surr1, surr2) w.r.t. log_prob ──
                // d(-min(r*A, clip(r)*A))/d(log_prob)
                // If not clipped: d/d(lp) = -ratio * adv  (since d(ratio)/d(lp) = ratio)
                // If clipped: gradient is 0 (clipped ratio is constant w.r.t. params)
                let d_log_prob = if !use_clipped {
                    -ratio * adv / nf
                } else {
                    0.0
                };

                // ── Gradient of log_prob w.r.t. mean[j] and log_std[j] ──
                // log_prob = Σ_j  -0.5 * z_j^2 - ln(s_j) - 0.5*ln(2π)
                // where z_j = (a_j - μ_j) / s_j,  s_j = exp(log_std_j)
                //
                // d(lp)/d(μ_j) = z_j / s_j = (a_j - μ_j) / s_j^2
                // d(lp)/d(log_std_j) = z_j^2 - 1   (since d(lp)/d(s_j) = z_j^2/s_j - 1/s_j, times s_j)
                let mut d_mean    = Array1::zeros(ACT_DIM);
                let mut d_log_std = Array1::zeros(ACT_DIM);
                for j in 0..ACT_DIM {
                    let s = cache.std[j].max(1e-8);
                    let z = (action[j] - cache.mean[j]) / s;
                    d_mean[j]    = d_log_prob * (z / s);
                    d_log_std[j] = d_log_prob * (z * z - 1.0);
                }

                // Add entropy bonus gradient: maximize entropy → d(-c*H)/d(log_std)
                // d(ent)/d(log_std_j) = 1  (since ent_j = 0.5*ln(2πe*s²) = log_std + const)
                let entropy_coef = 0.01;
                for j in 0..ACT_DIM {
                    d_log_std[j] -= entropy_coef / nf;
                }

                // ── Backprop through actor ──
                self.actor.backward_sgd(&cache, &d_mean, &d_log_std, lr);

                // ── Critic update (value loss) ──
                let v = self.critic.update_sgd(&obs_arr, returns[i], lr / nf);
                epoch_vloss += (v - returns[i]).powi(2);
            }

            total_policy_loss = epoch_ploss / nf;
            total_value_loss  = epoch_vloss / nf;
            total_entropy     = epoch_ent / nf;
            total_clip_frac   = epoch_clip / nf;
            total_kl          = epoch_kl / nf;
        }

        PpoUpdateStats {
            policy_loss: total_policy_loss,
            value_loss:  total_value_loss,
            entropy:     total_entropy,
            clip_frac:   total_clip_frac,
            approx_kl:   total_kl,
        }
    }
}
