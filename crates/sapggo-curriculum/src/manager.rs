use std::collections::VecDeque;

use crate::params::CurriculumParams;
use crate::stage::CurriculumStage;

/// Sliding-window curriculum manager.
///
/// Tracks mean episode reward over a configurable window and automatically
/// promotes the training stage when the threshold is exceeded.
pub struct CurriculumManager {
    pub stage:         CurriculumStage,
    pub episode_count: u64,
    recent_rewards:    VecDeque<f64>,
    window:            usize,
}

impl CurriculumManager {
    /// Creates a new manager starting at the `Stand` stage with a 50-episode window.
    pub fn new() -> Self {
        Self {
            stage:          CurriculumStage::Stand,
            episode_count:  0,
            recent_rewards: VecDeque::with_capacity(51),
            window:         50,
        }
    }

    /// Creates a manager with a custom sliding-window size.
    pub fn with_window(window: usize) -> Self {
        Self {
            stage:          CurriculumStage::Stand,
            episode_count:  0,
            recent_rewards: VecDeque::with_capacity(window + 1),
            window,
        }
    }

    /// Call at the end of each episode with the total episode reward.
    ///
    /// Returns `Some(new_stage)` if promotion occurred, `None` otherwise.
    pub fn on_episode_end(&mut self, reward: f64) -> Option<CurriculumStage> {
        self.episode_count += 1;
        self.recent_rewards.push_back(reward);

        if self.recent_rewards.len() > self.window {
            self.recent_rewards.pop_front();
        }

        if self.recent_rewards.len() < self.window {
            return None;
        }

        let sum: f64 = self.recent_rewards.iter().sum();
        let mean = sum / self.window as f64;

        if mean >= self.stage.promotion_threshold() {
            if let Some(next) = self.stage.next() {
                tracing::info!(
                    from        = self.stage.name(),
                    to          = next.name(),
                    mean_reward = mean,
                    episode     = self.episode_count,
                    "Curriculum promoted",
                );
                self.stage = next;
                self.recent_rewards.clear();
                return Some(self.stage);
            }
        }

        None
    }

    /// Returns the environment parameters for the current stage.
    #[inline]
    pub fn current_params(&self) -> CurriculumParams {
        CurriculumParams::for_stage(self.stage)
    }

    /// Returns the mean reward over the current sliding window, or `None`
    /// if fewer than `window` episodes have been recorded.
    pub fn mean_reward(&self) -> Option<f64> {
        if self.recent_rewards.len() < self.window {
            return None;
        }
        let sum: f64 = self.recent_rewards.iter().sum();
        Some(sum / self.window as f64)
    }
}

impl Default for CurriculumManager {
    fn default() -> Self {
        Self::new()
    }
}
