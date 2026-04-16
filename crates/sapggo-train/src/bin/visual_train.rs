//! SAPGGO Visual Training — watch the agent learn in real-time.
//!
//! Launches the MuJoCo 3D viewer alongside the training loop.
//! The viewer shows the robot physics while the agent trains with PPO.
//!
//! Usage:
//!   cargo run --release --bin sapggo-visual-train --features visual -- --config configs/train_default.toml

use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::Context;
use clap::Parser;
use ndarray::Array1;
use rand::rngs::StdRng;
use rand::SeedableRng;
use tracing_subscriber::EnvFilter;

use mujoco_rs::prelude::MjModel;
use mujoco_rs::viewer::MjViewer;

use sapggo_agent::{
    MlpActor, MlpCritic, Policy,
    PpoConfig, PpoUpdateStats, RolloutBuffer, RunningNormalizer, Transition,
    compute_gae, normalize_advantages,
};
use sapggo_curriculum::CurriculumManager;
use sapggo_env::{SapggoEnv, ACT_DIM, OBS_DIM};

// ── Config ──────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(name = "sapggo-visual-train", about = "Train SAPGGO with 3D visualization")]
struct Cli {
    #[arg(short, long, default_value = "configs/train_default.toml")]
    config: PathBuf,
    #[arg(short, long)]
    seed: Option<u64>,
}

#[derive(Debug, serde::Deserialize)]
struct TrainConfig {
    model_path:           String,
    #[serde(default = "default_seed")]
    seed:                 u64,
    #[serde(default = "default_total")]
    total_steps:          u64,
    #[serde(default = "default_rollout")]
    rollout_steps:        usize,
    #[serde(default = "default_lr")]
    lr:                   f64,
    #[serde(default = "default_gamma")]
    gamma:                f64,
    #[serde(default = "default_gae")]
    gae_lambda:           f64,
    #[serde(default = "default_clip")]
    clip_epsilon:         f64,
    #[serde(default = "default_log_dir")]
    log_dir:              String,
    #[serde(default = "default_ckpt_dir")]
    checkpoint_dir:       String,
    #[serde(default = "default_ckpt_interval")]
    checkpoint_interval:  u64,
}

fn default_seed()          -> u64    { 42 }
fn default_total()         -> u64    { 20_000_000 }
fn default_rollout()       -> usize  { 2048 }
fn default_lr()            -> f64    { 3e-4 }
fn default_gamma()         -> f64    { 0.99 }
fn default_gae()           -> f64    { 0.95 }
fn default_clip()          -> f64    { 0.2 }
fn default_log_dir()       -> String { "runs".into() }
fn default_ckpt_dir()      -> String { "checkpoints".into() }
fn default_ckpt_interval() -> u64    { 100_000 }

// ── Episode logger ──────────────────────────────────────────────────────

struct EpisodeLog {
    file: File,
}

impl EpisodeLog {
    fn new(dir: &Path) -> anyhow::Result<Self> {
        fs::create_dir_all(dir)?;
        let path = dir.join("visual_episodes.csv");
        let exists = path.exists();
        let file = OpenOptions::new().create(true).append(true).open(&path)?;
        let mut s = Self { file };
        if !exists {
            writeln!(s.file, "episode,global_step,steps,reward,distance_m,load_dropped,robot_fallen,best_reward,is_best,stage")?;
        }
        Ok(s)
    }
    fn log(&mut self, ep: u64, step: u64, steps: u64, reward: f64, dist: f64,
           dropped: bool, fallen: bool, best: f64, is_best: bool, stage: &str) {
        let _ = writeln!(self.file,
            "{ep},{step},{steps},{reward:.4},{dist:.4},{dropped},{fallen},{best:.4},{is_best},{stage}");
    }
    fn flush(&mut self) { let _ = self.file.flush(); }
}

// ── Main ────────────────────────────────────────────────────────────────

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .with_target(false)
        .with_timer(tracing_subscriber::fmt::time::uptime())
        .init();

    let cli = Cli::parse();

    let cfg: TrainConfig = if cli.config.exists() {
        let text = fs::read_to_string(&cli.config)?;
        toml::from_str(&text).with_context(|| format!("Parsing {}", cli.config.display()))?
    } else {
        TrainConfig {
            model_path: "assets/robot_humanoid_load.xml".into(),
            seed: 42, total_steps: 20_000_000, rollout_steps: 2048,
            lr: 3e-4, gamma: 0.99, gae_lambda: 0.95, clip_epsilon: 0.2,
            log_dir: "runs".into(), checkpoint_dir: "checkpoints".into(),
            checkpoint_interval: 100_000,
        }
    };
    let seed = cli.seed.unwrap_or(cfg.seed);

    println!("╔══════════════════════════════════════════════╗");
    println!("║    SAPGGO — Visual Training (3D + Agent)    ║");
    println!("╠══════════════════════════════════════════════╣");
    println!("║  The agent learns AND you see it in 3D!     ║");
    println!("║  Mouse: rotate/pan/zoom  |  Ctrl+Q: quit    ║");
    println!("╚══════════════════════════════════════════════╝");

    // ── 1. Create RL environment (physics + reward logic) ──
    let mut env = SapggoEnv::new(&cfg.model_path, seed)?;
    let nq = env.nq();
    let nv = env.nv();
    tracing::info!(nq, nv, obs_dim = OBS_DIM, act_dim = ACT_DIM, "Environment ready");

    // ── 2. Create viewer (separate model+data for rendering) ──
    let viz_model = MjModel::from_xml(&cfg.model_path)?;
    let mut viz_data = viz_model.make_data();
    let mut viewer = MjViewer::launch_passive(&viz_model, 1000)
        .map_err(|e| anyhow::anyhow!("Viewer launch failed: {:?}", e))?;
    tracing::info!("3D viewer opened");

    // ── 3. Create agent ──
    let mut rng    = StdRng::seed_from_u64(seed);
    let mut actor  = MlpActor::new(OBS_DIM, ACT_DIM, &mut rng);
    let mut critic = MlpCritic::new(OBS_DIM, &mut rng);
    let mut normalizer = RunningNormalizer::new(OBS_DIM);
    let mut curriculum = CurriculumManager::new();

    let ppo_cfg = PpoConfig {
        lr: cfg.lr, gamma: cfg.gamma, gae_lambda: cfg.gae_lambda,
        clip_epsilon: cfg.clip_epsilon, ..PpoConfig::default()
    };

    let mut episode_log = EpisodeLog::new(Path::new(&cfg.log_dir))?;
    fs::create_dir_all(&cfg.checkpoint_dir)?;

    // ── 4. Training + visualization loop ──
    let mut obs = env.reset();
    sync_state_to_viewer(&env, &viz_model, &mut viz_data, nq, nv);
    viewer.sync(&mut viz_data);

    let mut global_step: u64 = 0;
    let mut episode: u64     = 0;
    let mut best_reward      = f64::NEG_INFINITY;
    let mut best_episode: u64 = 0;

    tracing::info!(total_steps = cfg.total_steps, "Training started — watch the robot learn!");

    while global_step < cfg.total_steps && viewer.running() {
        let frame_start = Instant::now();

        // ── Collect one rollout (rendering each step) ──
        let mut buffer = RolloutBuffer::new(cfg.rollout_steps);

        while !buffer.is_full() && viewer.running() {
            let obs_vec = obs.to_vec();
            normalizer.update(&obs_vec);
            let norm_obs = normalizer.normalize(&obs_vec);

            let (action, log_prob, _) = actor.act_stochastic(&norm_obs);
            let value = critic.forward(&Array1::from(norm_obs.clone()));

            let result = env.step(&action)?;

            buffer.push(Transition {
                observation: obs_vec,
                action, log_prob, value,
                reward: result.reward,
                done:   result.done,
            });

            // ── Sync viewer every step so we SEE the robot ──
            sync_state_to_viewer(&env, &viz_model, &mut viz_data, nq, nv);
            viewer.sync(&mut viz_data);

            if result.done {
                episode += 1;
                let info = result.info;
                let is_best = info.total_reward > best_reward;
                if is_best {
                    best_reward  = info.total_reward;
                    best_episode = episode;
                    save_json(&actor,  &Path::new(&cfg.checkpoint_dir).join("best_actor.json"));
                    save_json(&critic, &Path::new(&cfg.checkpoint_dir).join("best_critic.json"));
                }

                let status = if info.load_dropped { "DROPPED" }
                             else if info.robot_fallen { "FALLEN" }
                             else { "TIMEOUT" };

                episode_log.log(episode, global_step + buffer.len() as u64,
                    info.steps, info.total_reward, info.distance_m,
                    info.load_dropped, info.robot_fallen, best_reward, is_best,
                    curriculum.stage.name());

                tracing::info!(
                    ep = episode, status, steps = info.steps,
                    reward = format!("{:.2}", info.total_reward),
                    dist = format!("{:.2}m", info.distance_m),
                    best = format!("{:.2}", best_reward),
                    best_ep = best_episode,
                    "Episode done → instant restart",
                );

                if let Some(stage) = curriculum.on_episode_end(info.total_reward) {
                    tracing::info!(new_stage = stage.name(), "Curriculum promotion!");
                    env.set_params(curriculum.current_params())?;
                }

                if episode % 50 == 0 { episode_log.flush(); }

                obs = env.reset();
                sync_state_to_viewer(&env, &viz_model, &mut viz_data, nq, nv);
                viewer.sync(&mut viz_data);
            } else {
                obs = result.observation;
            }

            // Throttle to ~real-time (don't go faster than ~60 Hz rendering)
            let elapsed = frame_start.elapsed();
            if elapsed < Duration::from_millis(8) {
                std::thread::sleep(Duration::from_millis(8) - elapsed);
            }
        }

        if !viewer.running() { break; }

        global_step += buffer.len() as u64;

        // ── PPO update ──
        let slices = buffer.as_slices();
        let (mut advantages, returns) = compute_gae(
            &slices.rewards, &slices.values, &slices.dones,
            0.0, ppo_cfg.gamma, ppo_cfg.gae_lambda,
        );
        normalize_advantages(&mut advantages);
        let stats = ppo_update(&mut actor, &mut critic, &normalizer, &buffer, &advantages, &returns, &ppo_cfg);

        tracing::info!(
            step = global_step, episodes = episode,
            ploss = format!("{:.4}", stats.policy_loss),
            vloss = format!("{:.1}", stats.value_loss),
            entropy = format!("{:.2}", stats.entropy),
            "PPO update",
        );
    }

    // ── Final save ──
    save_json(&actor, &Path::new(&cfg.checkpoint_dir).join("sapggo_final.json"));
    episode_log.flush();
    tracing::info!(
        total_steps = global_step, episodes = episode,
        best_reward = format!("{:.2}", best_reward), best_ep = best_episode,
        "Training complete",
    );

    Ok(())
}

// ── Helpers ─────────────────────────────────────────────────────────────

/// Copies qpos/qvel from the training Simulation to the viewer's MjData,
/// then runs mj_forward so the viewer sees the current pose.
fn sync_state_to_viewer(
    env: &SapggoEnv, _viz_model: &MjModel, viz_data: &mut mujoco_rs::prelude::MjData, nq: usize, nv: usize,
) {
    let qpos_src = env.qpos_slice();
    let qvel_src = env.qvel_slice();
    let d = viz_data.ffi_mut();
    unsafe {
        std::ptr::copy_nonoverlapping(qpos_src.as_ptr(), d.qpos, nq);
        std::ptr::copy_nonoverlapping(qvel_src.as_ptr(), d.qvel, nv);
    }
    viz_data.forward();
}

fn save_json<T: serde::Serialize>(data: &T, path: &Path) {
    match serde_json::to_vec(data) {
        Ok(json) => { let _ = fs::write(path, &json); }
        Err(e) => tracing::warn!("Checkpoint save failed: {e}"),
    }
}

/// PPO update with real backpropagation through all network layers.
fn ppo_update(
    actor: &mut MlpActor, critic: &mut MlpCritic, normalizer: &RunningNormalizer,
    rollout: &RolloutBuffer, advantages: &[f64], returns: &[f64], cfg: &PpoConfig,
) -> PpoUpdateStats {
    let slices = rollout.as_slices();
    let n = slices.rewards.len();
    if n == 0 { return PpoUpdateStats::default(); }

    let lr       = cfg.lr;
    let clip_eps = cfg.clip_epsilon;
    let epochs   = cfg.epochs.max(1);
    let nf       = n as f64;

    let mut total_ploss = 0.0f64;
    let mut total_vloss = 0.0f64;
    let mut total_ent   = 0.0f64;
    let mut total_clip  = 0.0f64;
    let mut total_kl    = 0.0f64;

    for _epoch in 0..epochs {
        let (mut ep, mut ev, mut ee, mut ec, mut ek) = (0.0, 0.0, 0.0, 0.0, 0.0);

        for i in 0..n {
            let norm_obs = normalizer.normalize(&slices.observations[i]);
            let obs_arr  = Array1::from(norm_obs);
            let action   = &slices.actions[i];

            let cache = actor.forward_with_cache(&obs_arr);

            let mut new_lp = 0.0f64;
            let mut ent    = 0.0f64;
            for j in 0..ACT_DIM {
                let s = cache.std[j].max(1e-8);
                let z = (action[j] - cache.mean[j]) / s;
                new_lp += -0.5 * z * z - s.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln();
                ent    += 0.5 * (2.0 * std::f64::consts::PI * std::f64::consts::E * s * s).ln();
            }

            let ratio   = (new_lp - slices.log_probs[i]).exp();
            let clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps);
            let adv     = advantages[i];
            let surr1   = ratio * adv;
            let surr2   = clipped * adv;
            let use_clipped = surr2 < surr1;
            ep += -surr1.min(surr2);
            ec += if use_clipped { 1.0 } else { 0.0 };
            ek += slices.log_probs[i] - new_lp;
            ee += ent;

            let d_lp = if !use_clipped { -ratio * adv / nf } else { 0.0 };

            let mut d_mean    = Array1::zeros(ACT_DIM);
            let mut d_log_std = Array1::zeros(ACT_DIM);
            for j in 0..ACT_DIM {
                let s = cache.std[j].max(1e-8);
                let z = (action[j] - cache.mean[j]) / s;
                d_mean[j]    = d_lp * (z / s);
                d_log_std[j] = d_lp * (z * z - 1.0);
            }
            let entropy_coef = 0.01;
            for j in 0..ACT_DIM { d_log_std[j] -= entropy_coef / nf; }

            actor.backward_sgd(&cache, &d_mean, &d_log_std, lr);

            let v = critic.update_sgd(&obs_arr, returns[i], lr / nf);
            ev += (v - returns[i]).powi(2);
        }

        total_ploss = ep / nf;
        total_vloss = ev / nf;
        total_ent   = ee / nf;
        total_clip  = ec / nf;
        total_kl    = ek / nf;
    }

    PpoUpdateStats {
        policy_loss: total_ploss, value_loss: total_vloss, entropy: total_ent,
        clip_frac: total_clip, approx_kl: total_kl,
    }
}
