/// A single transition stored during rollout collection.
#[derive(Debug, Clone)]
pub struct Transition {
    pub observation: Vec<f64>,
    pub action:      Vec<f64>,
    pub log_prob:    f64,
    pub value:       f64,
    pub reward:      f64,
    pub done:        bool,
}

/// Rollout buffer that collects transitions from environment interaction.
///
/// Pre-allocates capacity for the expected rollout length to avoid
/// repeated heap allocations during collection.
pub struct RolloutBuffer {
    pub transitions:      Vec<Transition>,
    pub episode_rewards:  Vec<f64>,
    capacity:             usize,
    current_episode_reward: f64,
}

impl RolloutBuffer {
    /// Creates a new buffer pre-allocated for `capacity` transitions.
    pub fn new(capacity: usize) -> Self {
        Self {
            transitions:           Vec::with_capacity(capacity),
            episode_rewards:       Vec::new(),
            capacity,
            current_episode_reward: 0.0,
        }
    }

    /// Clears all stored data for the next rollout.
    pub fn clear(&mut self) {
        self.transitions.clear();
        self.episode_rewards.clear();
        self.current_episode_reward = 0.0;
    }

    /// Stores a transition and tracks per-episode cumulative reward.
    pub fn push(&mut self, t: Transition) {
        self.current_episode_reward += t.reward;
        let done = t.done;
        self.transitions.push(t);

        if done {
            self.episode_rewards.push(self.current_episode_reward);
            self.current_episode_reward = 0.0;
        }
    }

    /// Number of transitions stored.
    #[inline]
    pub fn len(&self) -> usize {
        self.transitions.len()
    }

    /// Returns `true` if no transitions have been stored.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.transitions.is_empty()
    }

    /// Returns `true` when the buffer has reached its target capacity.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.transitions.len() >= self.capacity
    }

    /// Extracts parallel slices of each field for batch processing.
    pub fn as_slices(&self) -> RolloutSlices<'_> {
        RolloutSlices {
            observations: self.transitions.iter().map(|t| t.observation.as_slice()).collect(),
            actions:      self.transitions.iter().map(|t| t.action.as_slice()).collect(),
            log_probs:    self.transitions.iter().map(|t| t.log_prob).collect(),
            values:       self.transitions.iter().map(|t| t.value).collect(),
            rewards:      self.transitions.iter().map(|t| t.reward).collect(),
            dones:        self.transitions.iter().map(|t| t.done).collect(),
        }
    }

    /// Returns random minibatch indices for PPO updates.
    pub fn minibatch_indices(
        &self,
        batch_size: usize,
        rng: &mut impl rand::Rng,
    ) -> Vec<Vec<usize>> {
        use rand::seq::SliceRandom;

        let n = self.transitions.len();
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(rng);

        indices.chunks(batch_size).map(|c| c.to_vec()).collect()
    }
}

/// Borrowed slices of rollout data for efficient batch access.
pub struct RolloutSlices<'a> {
    pub observations: Vec<&'a [f64]>,
    pub actions:      Vec<&'a [f64]>,
    pub log_probs:    Vec<f64>,
    pub values:       Vec<f64>,
    pub rewards:      Vec<f64>,
    pub dones:        Vec<bool>,
}
