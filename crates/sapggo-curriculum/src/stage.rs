use serde::{Deserialize, Serialize};

/// Curriculum stages ordered by increasing difficulty.
/// Each stage defines a specific training objective and environment configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CurriculumStage {
    Stand    = 0,
    Balance  = 1,
    Walk     = 2,
    Distance = 3,
    Robust   = 4,
    Master   = 5,
}

impl CurriculumStage {
    /// Mean reward over the last 50 episodes required to advance to the next stage.
    #[inline]
    pub fn promotion_threshold(&self) -> f64 {
        match self {
            Self::Stand    => 15.0,
            Self::Balance  => 25.0,
            Self::Walk     => 50.0,
            Self::Distance => 120.0,
            Self::Robust   => 250.0,
            Self::Master   => f64::INFINITY,
        }
    }

    /// Human-readable stage name.
    #[inline]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Stand    => "Stand",
            Self::Balance  => "Balance",
            Self::Walk     => "Walk",
            Self::Distance => "Distance",
            Self::Robust   => "Robust",
            Self::Master   => "Master",
        }
    }

    /// Returns the next stage, or `None` if already at the final stage.
    #[inline]
    pub fn next(&self) -> Option<Self> {
        match self {
            Self::Stand    => Some(Self::Balance),
            Self::Balance  => Some(Self::Walk),
            Self::Walk     => Some(Self::Distance),
            Self::Distance => Some(Self::Robust),
            Self::Robust   => Some(Self::Master),
            Self::Master   => None,
        }
    }

    /// Minimum mean distance (m) required for promotion from this stage.
    /// Only enforced for Walk+ stages; Stand and Balance return 0.
    #[inline]
    pub fn distance_threshold(&self) -> f64 {
        match self {
            Self::Stand    => 0.0,
            Self::Balance  => 0.0,
            Self::Walk     => 2.0,
            Self::Distance => 5.0,
            Self::Robust   => 10.0,
            Self::Master   => f64::INFINITY,
        }
    }

    /// Returns the numeric index of this stage (0–5).
    #[inline]
    pub fn index(&self) -> usize {
        *self as usize
    }
}

impl std::fmt::Display for CurriculumStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}
