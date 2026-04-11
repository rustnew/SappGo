use std::path::Path;

use ndarray::Array1;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

use sapggo_curriculum::CurriculumParams;

use crate::error::{EnvError, EnvResult};
use crate::load;
use crate::noise::GaussianNoise;
use crate::reward::{self, RewardState, RewardWeights};
use crate::robot::*;
use crate::sensor::{self, SensorIndex};
use crate::sim::Simulation;

/// Information returned alongside each step result.
#[derive(Debug, Clone)]
pub struct StepInfo {
    pub distance_m:   f64,
    pub load_dropped: bool,
    pub robot_fallen: bool,
    pub steps:        u64,
    pub total_reward: f64,
}

/// Result of a single environment step.
pub struct StepResult {
    pub observation: Array1<f64>,
    pub reward:      f64,
    pub done:        bool,
    pub info:        StepInfo,
}

/// SAPGGO MuJoCo environment.
///
/// Wraps a [`Simulation`] (model + data) and implements the RL environment
/// interface (reset / step) with reward shaping, curriculum integration,
/// action smoothing, and domain randomization.
pub struct SapggoEnv {
    sim:            Simulation,
    idx:            SensorIndex,
    params:         CurriculumParams,
    weights:        RewardWeights,
    noise:          GaussianNoise,
    prev_action:    [f64; N_JOINTS],
    smoothed_ctrl:  [f64; N_JOINTS],
    steps:          u64,
    distance_m:     f64,
    total_reward:   f64,
    last_milestone: u64,
    last_info:      Option<StepInfo>,
    rng:            StdRng,
}

impl SapggoEnv {
    /// Number of physics sub-steps per control step (10 × 2 ms = 20 ms).
    pub const SIM_STEPS: usize = 10;

    /// Passive settling steps at episode reset.
    pub const PASSIVE_STEPS: usize = 50;

    /// Default maximum steps per episode (overridden by curriculum).
    pub const DEFAULT_MAX_STEPS: u64 = 1000;

    /// Creates a new environment from a MuJoCo XML model path.
    pub fn new(model_path: &str, seed: u64) -> EnvResult<Self> {
        let sim = Simulation::from_xml(Path::new(model_path))?;
        let idx = SensorIndex::resolve(&sim)?;

        let params = CurriculumParams::default();
        let noise  = GaussianNoise::new(params.observation_noise)?;

        Ok(Self {
            sim,
            idx,
            params,
            weights:        RewardWeights::default(),
            noise,
            prev_action:    [0.0; N_JOINTS],
            smoothed_ctrl:  [0.0; N_JOINTS],
            steps:          0,
            distance_m:     0.0,
            total_reward:   0.0,
            last_milestone: 0,
            last_info:      None,
            rng:            StdRng::seed_from_u64(seed),
        })
    }

    /// Updates the curriculum parameters used for the next episode.
    pub fn set_params(&mut self, p: CurriculumParams) -> EnvResult<()> {
        self.noise  = GaussianNoise::new(p.observation_noise)?;
        self.params = p;
        Ok(())
    }

    /// Sets custom reward weights.
    pub fn set_reward_weights(&mut self, w: RewardWeights) {
        self.weights = w;
    }

    /// Returns the most recent step info (useful after an episode ends).
    pub fn last_info(&self) -> Option<&StepInfo> {
        self.last_info.as_ref()
    }

    /// Resets the environment for a new episode.
    ///
    /// 1. Resets MuJoCo state (zeroes all velocities).
    /// 2. Applies domain randomization.
    /// 3. Places the load on the head with a random offset.
    /// 4. Runs passive settling steps so the load rests naturally.
    /// 5. Returns the initial observation.
    pub fn reset(&mut self) -> Array1<f64> {
        self.sim.reset_data();

        self.apply_domain_randomization();
        load::place_load_on_head(&mut self.sim, &self.idx, &mut self.rng);

        // Passive settling: load falls gently onto head
        for _ in 0..Self::PASSIVE_STEPS {
            self.sim.step();
        }

        self.prev_action    = [0.0; N_JOINTS];
        self.smoothed_ctrl  = [0.0; N_JOINTS];
        self.steps          = 0;
        self.distance_m     = 0.0;
        self.total_reward   = 0.0;
        self.last_milestone = 0;
        self.last_info      = None;

        sensor::extract_observation(
            &self.sim,
            &self.idx,
            &self.prev_action,
            self.params.target_velocity,
            &self.noise,
            &mut self.rng,
        )
    }

    /// Advances the environment by one control step.
    ///
    /// # Errors
    ///
    /// Returns `EnvError::ActionDimMismatch` if the action slice length
    /// does not match `ACT_DIM`.
    pub fn step(&mut self, action: &[f64]) -> EnvResult<StepResult> {
        if action.len() != ACT_DIM {
            return Err(EnvError::ActionDimMismatch {
                expected: ACT_DIM,
                got:      action.len(),
            });
        }

        // Action smoothing: low-pass filter to prevent mechanical shock
        for i in 0..N_JOINTS {
            self.smoothed_ctrl[i] =
                ACTION_SMOOTH_ALPHA * self.smoothed_ctrl[i]
              + ACTION_SMOOTH_BETA  * action[i] * MAX_TORQUE[i];
            let clamped = self.smoothed_ctrl[i].clamp(-MAX_TORQUE[i], MAX_TORQUE[i]);
            self.sim.set_ctrl(i, clamped);
        }

        // Record x position before step
        let x_before = self.sim.body_xpos(self.idx.torso_body_id, 0);

        // Advance physics
        for _ in 0..Self::SIM_STEPS {
            self.sim.step();
        }

        let x_after = self.sim.body_xpos(self.idx.torso_body_id, 0);
        self.steps      += 1;
        self.distance_m += (x_after - x_before).max(0.0);

        // Check termination
        let load_dropped = load::is_dropped(&self.sim, &self.idx);
        let robot_fallen = self.robot_is_fallen();
        let max_steps    = self.params.max_steps.max(Self::DEFAULT_MAX_STEPS);
        let timeout      = self.steps >= max_steps;
        let done         = load_dropped || robot_fallen || timeout;

        // Compute reward
        let rs    = self.build_reward_state(x_after - x_before, action);
        let mut r = reward::compute_reward(&rs, &self.weights);

        // Milestone bonuses (every 10 m)
        let ms_now = (self.distance_m / 10.0).floor() as u64;
        if ms_now > self.last_milestone {
            r += reward::MILESTONE_BONUS * (ms_now - self.last_milestone) as f64;
            self.last_milestone = ms_now;
        }

        // Termination penalty
        if done && (load_dropped || robot_fallen) {
            r += reward::TERMINATION_PENALTY;
        }

        self.total_reward += r;

        // Store current action as previous
        let mut cur = [0.0f64; N_JOINTS];
        cur.copy_from_slice(&action[..N_JOINTS]);
        self.prev_action = cur;

        let obs = sensor::extract_observation(
            &self.sim,
            &self.idx,
            &self.prev_action,
            self.params.target_velocity,
            &self.noise,
            &mut self.rng,
        );

        let info = StepInfo {
            distance_m:   self.distance_m,
            load_dropped,
            robot_fallen,
            steps:        self.steps,
            total_reward: self.total_reward,
        };
        self.last_info = Some(info.clone());

        Ok(StepResult {
            observation: obs,
            reward:      r,
            done,
            info,
        })
    }

    // ── Private helpers ──────────────────────────────────────────────

    /// Returns `true` if the torso has fallen below the safe height threshold.
    #[inline]
    fn robot_is_fallen(&self) -> bool {
        self.sim.body_xpos(self.idx.torso_body_id, 2) < 0.6
    }

    /// Applies per-episode domain randomization based on curriculum parameters.
    fn apply_domain_randomization(&mut self) {
        let wind_max = self.params.wind_force_max;
        let wind = (2.0 * self.rng.gen::<f64>() - 1.0) * wind_max;
        self.sim.set_xfrc_applied(self.idx.load_body_id, 0, wind);
    }

    /// Builds a `RewardState` snapshot from the current simulation data.
    fn build_reward_state(&self, delta_x: f64, action: &[f64]) -> RewardState {
        let tid = self.idx.torso_body_id;

        // Extract torso quaternion components
        let qw = self.sim.body_xquat(tid, 0);
        let qx = self.sim.body_xquat(tid, 1);
        let qy = self.sim.body_xquat(tid, 2);
        let qz = self.sim.body_xquat(tid, 3);

        // Extract pitch and roll from the torso quaternion
        let pitch = (2.0 * (qy * qz + qw * qx)).asin();
        let roll  = (2.0 * (qw * qy - qx * qz))
            .atan2(1.0 - 2.0 * (qx.powi(2) + qy.powi(2)));

        let mut torques    = [0.0f64; N_JOINTS];
        let mut cur_action = [0.0f64; N_JOINTS];
        for i in 0..N_JOINTS {
            torques[i]    = self.sim.actuator_force(i);
            cur_action[i] = action[i];
        }

        let hid = self.idx.head_body_id;
        let lid = self.idx.load_body_id;

        let dt = Self::SIM_STEPS as f64 * 0.002;

        RewardState {
            velocity_x:  delta_x / dt,
            torso_pitch: pitch,
            torso_roll:  roll,
            load_dx:     self.sim.body_xpos(lid, 0) - self.sim.body_xpos(hid, 0),
            load_dy:     self.sim.body_xpos(lid, 1) - self.sim.body_xpos(hid, 1),
            load_dz:     self.sim.body_xpos(lid, 2) - self.sim.body_xpos(hid, 2),
            torques,
            prev_action: self.prev_action,
            cur_action,
        }
    }
}
