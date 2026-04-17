use std::path::Path;

use ndarray::Array1;
use rand::SeedableRng;
use rand::rngs::StdRng;

use sapggo_agent::{
    ActorGradBuffer, CriticGradBuffer, MlpActor, MlpCritic, Policy,
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

    /// PPO update with batch gradient accumulation.
    ///
    /// For each epoch:
    ///   1. Accumulate gradients over the full rollout (no weight changes).
    ///   2. Apply the mean gradient in one step with clipping.
    ///
    /// This ensures the policy stays consistent within each epoch,
    /// giving clean gradient estimates and stable learning.
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

        let lr       = self.ppo_config.lr;
        let clip_eps = self.ppo_config.clip_epsilon;
        let epochs   = self.ppo_config.epochs.max(1);
        let ent_coef = self.ppo_config.entropy_coef;
        let nf       = n as f64;

        let mut actor_grad  = ActorGradBuffer::new(OBS_DIM, ACT_DIM);
        let mut critic_grad = CriticGradBuffer::new(OBS_DIM);

        let mut total_ploss = 0.0f64;
        let mut total_vloss = 0.0f64;
        let mut total_ent   = 0.0f64;
        let mut total_clip  = 0.0f64;
        let mut total_kl    = 0.0f64;

        for _epoch in 0..epochs {
            actor_grad.reset();
            critic_grad.reset();
            let (mut ep, mut ev, mut ee, mut ec, mut ek) = (0.0, 0.0, 0.0, 0.0, 0.0);

            // ── Accumulate gradients over full rollout ──
            for i in 0..n {
                let norm_obs = self.normalizer.normalize(&slices.observations[i]);
                let obs_arr  = Array1::from(norm_obs);
                let action   = &slices.actions[i];

                let cache = self.actor.forward_with_cache(&obs_arr);

                let mut new_lp = 0.0f64;
                let mut ent    = 0.0f64;
                for j in 0..ACT_DIM {
                    let s = cache.std[j].max(1e-8);
                    let z = (action[j] - cache.mean[j]) / s;
                    new_lp += -0.5 * z * z - s.ln()
                        - 0.5 * (2.0 * std::f64::consts::PI).ln();
                    ent += 0.5 * (2.0 * std::f64::consts::PI * std::f64::consts::E * s * s).ln();
                }

                let ratio   = (new_lp - slices.log_probs[i]).exp();
                let clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps);
                let adv     = advantages[i];
                let surr1   = ratio * adv;
                let surr2   = clipped * adv;
                let is_clip = surr2 < surr1;
                ep += -surr1.min(surr2);
                ec += if is_clip { 1.0 } else { 0.0 };
                ek += slices.log_probs[i] - new_lp;
                ee += ent;

                // Policy gradient (0 when clipped)
                let d_lp = if !is_clip { -ratio * adv } else { 0.0 };

                let mut d_mean    = Array1::zeros(ACT_DIM);
                let mut d_log_std = Array1::zeros(ACT_DIM);
                for j in 0..ACT_DIM {
                    let s = cache.std[j].max(1e-8);
                    let z = (action[j] - cache.mean[j]) / s;
                    d_mean[j]    = d_lp * (z / s);
                    d_log_std[j] = d_lp * (z * z - 1.0);
                    // Entropy bonus: maximize entropy → subtract gradient
                    d_log_std[j] -= ent_coef;
                }

                // Accumulate actor gradients (weights unchanged)
                self.actor.accumulate_grad(&cache, &d_mean, &d_log_std, &mut actor_grad);

                // Accumulate critic gradients (weights unchanged)
                let v = self.critic.accumulate_grad(&obs_arr, returns[i], &mut critic_grad);
                ev += (v - returns[i]).powi(2);
            }

            // ── Apply accumulated gradients in one batch step ──
            self.actor.apply_grad(&mut actor_grad, nf, lr);
            self.critic.apply_grad(&mut critic_grad, nf, lr);

            total_ploss = ep / nf;
            total_vloss = ev / nf;
            total_ent   = ee / nf;
            total_clip  = ec / nf;
            total_kl    = ek / nf;
        }

        PpoUpdateStats {
            policy_loss: total_ploss,
            value_loss:  total_vloss,
            entropy:     total_ent,
            clip_frac:   total_clip,
            approx_kl:   total_kl,
        }
    }
}
