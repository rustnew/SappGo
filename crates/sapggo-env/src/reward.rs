use serde::{Deserialize, Serialize};

use crate::robot::N_JOINTS;

/// Per-step reward component weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardWeights {
    /// Forward velocity tracking weight.
    pub vel:    f64,
    /// Torso tilt penalty weight (combined pitch+roll via projection).
    pub tilt:   f64,
    /// Torque energy penalty weight.
    pub energy: f64,
    /// Load x-offset penalty weight.
    pub load_x: f64,
    /// Load y-offset penalty weight.
    pub load_y: f64,
    /// Load z-drop penalty weight.
    pub load_z: f64,
    /// Action jerk (smoothness) penalty weight.
    pub jerk:   f64,
    /// Survival bonus per step.
    pub alive:  f64,
}

impl Default for RewardWeights {
    fn default() -> Self {
        Self {
            vel:    2.0,
            tilt:   0.5,
            energy: 0.001,
            load_x: 1.0,
            load_y: 1.0,
            load_z: 2.0,
            jerk:   0.05,
            alive:  1.0,
        }
    }
}

/// Snapshot of state variables needed for reward computation.
pub struct RewardState {
    pub velocity_x:  f64,
    pub tilt_angle:  f64,
    pub load_dx:     f64,
    pub load_dy:     f64,
    pub load_dz:     f64,
    pub torques:     [f64; N_JOINTS],
    pub prev_action: [f64; N_JOINTS],
    pub cur_action:  [f64; N_JOINTS],
}

/// Computes the dense per-step reward.
///
/// Positive contributions: forward velocity, survival bonus.
/// Negative contributions: posture penalties, energy, load misalignment, jerk.
#[inline]
pub fn compute_reward(s: &RewardState, w: &RewardWeights) -> f64 {
    let inv_n = 1.0 / N_JOINTS as f64;

    let energy: f64 = s.torques.iter().map(|t| t * t).sum::<f64>() * inv_n;

    let jerk: f64 = s.cur_action.iter()
        .zip(s.prev_action.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>() * inv_n;

    let z_drop = (-s.load_dz).max(0.0);

    w.vel    *  s.velocity_x
  - w.tilt   *  s.tilt_angle
  - w.energy *  energy
  - w.load_x *  s.load_dx.abs()
  - w.load_y *  s.load_dy.abs()
  - w.load_z *  z_drop
  - w.jerk   *  jerk
  + w.alive
}

/// Sparse bonus awarded every 10 m of forward progress.
pub const MILESTONE_BONUS: f64 = 25.0;

/// Bonus for completing a full 1 km episode.
pub const EPISODE_BONUS: f64 = 100.0;

/// Penalty applied when the episode terminates due to failure.
pub const TERMINATION_PENALTY: f64 = -50.0;
