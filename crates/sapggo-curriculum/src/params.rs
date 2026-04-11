use serde::{Deserialize, Serialize};

use crate::stage::CurriculumStage;

/// Terrain surface type, introduced progressively via the curriculum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TerrainType {
    Flat,
    Rolling,
    Rough,
}

impl TerrainType {
    /// Returns the heightmap asset filename for this terrain type.
    #[inline]
    pub fn asset_filename(&self) -> &'static str {
        match self {
            Self::Flat    => "terrain_flat.png",
            Self::Rolling => "terrain_rolling.png",
            Self::Rough   => "terrain_rough.png",
        }
    }
}

/// Physics and domain-randomization parameters for a given curriculum stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurriculumParams {
    pub load_mass_min:       f64,
    pub load_mass_max:       f64,
    pub head_friction_min:   f64,
    pub head_friction_max:   f64,
    pub ground_friction_min: f64,
    pub ground_friction_max: f64,
    pub wind_force_max:      f64,
    pub gravity_scale:       f64,
    pub terrain_type:        TerrainType,
    pub observation_noise:   f64,
    pub action_noise:        f64,
    pub target_velocity:     f64,
    pub max_steps:           u64,
}

impl Default for CurriculumParams {
    fn default() -> Self {
        Self::for_stage(CurriculumStage::Stand)
    }
}

impl CurriculumParams {
    /// Returns the canonical parameter set for a given curriculum stage.
    pub fn for_stage(s: CurriculumStage) -> Self {
        match s {
            CurriculumStage::Stand => Self {
                load_mass_min:       2.0,
                load_mass_max:       2.0,
                head_friction_min:   0.8,
                head_friction_max:   0.8,
                ground_friction_min: 0.8,
                ground_friction_max: 0.8,
                wind_force_max:      0.0,
                gravity_scale:       0.5,
                terrain_type:        TerrainType::Flat,
                observation_noise:   0.0,
                action_noise:        0.0,
                target_velocity:     0.0,
                max_steps:           250,
            },
            CurriculumStage::Balance => Self {
                load_mass_min:       2.0,
                load_mass_max:       5.0,
                head_friction_min:   0.7,
                head_friction_max:   1.0,
                ground_friction_min: 0.8,
                ground_friction_max: 1.0,
                wind_force_max:      0.0,
                gravity_scale:       1.0,
                terrain_type:        TerrainType::Flat,
                observation_noise:   0.0,
                action_noise:        0.0,
                target_velocity:     0.0,
                max_steps:           500,
            },
            CurriculumStage::Walk => Self {
                load_mass_min:       2.0,
                load_mass_max:       7.0,
                head_friction_min:   0.6,
                head_friction_max:   1.2,
                ground_friction_min: 0.7,
                ground_friction_max: 1.1,
                wind_force_max:      2.0,
                gravity_scale:       1.0,
                terrain_type:        TerrainType::Flat,
                observation_noise:   0.002,
                action_noise:        0.01,
                target_velocity:     0.8,
                max_steps:           600,
            },
            CurriculumStage::Distance => Self {
                load_mass_min:       5.0,
                load_mass_max:       10.0,
                head_friction_min:   0.5,
                head_friction_max:   1.3,
                ground_friction_min: 0.6,
                ground_friction_max: 1.2,
                wind_force_max:      5.0,
                gravity_scale:       1.0,
                terrain_type:        TerrainType::Rolling,
                observation_noise:   0.005,
                action_noise:        0.02,
                target_velocity:     1.0,
                max_steps:           800,
            },
            CurriculumStage::Robust => Self {
                load_mass_min:       5.0,
                load_mass_max:       15.0,
                head_friction_min:   0.4,
                head_friction_max:   1.5,
                ground_friction_min: 0.5,
                ground_friction_max: 1.3,
                wind_force_max:      10.0,
                gravity_scale:       1.0,
                terrain_type:        TerrainType::Rough,
                observation_noise:   0.008,
                action_noise:        0.03,
                target_velocity:     1.0,
                max_steps:           1000,
            },
            CurriculumStage::Master => Self {
                load_mass_min:       10.0,
                load_mass_max:       20.0,
                head_friction_min:   0.4,
                head_friction_max:   1.5,
                ground_friction_min: 0.5,
                ground_friction_max: 1.3,
                wind_force_max:      15.0,
                gravity_scale:       1.0,
                terrain_type:        TerrainType::Rough,
                observation_noise:   0.01,
                action_noise:        0.04,
                target_velocity:     1.2,
                max_steps:           1000,
            },
        }
    }

    /// Linearly interpolate a value within the configured range using a `[0, 1]` factor.
    #[inline]
    pub fn sample_load_mass(&self, t: f64) -> f64 {
        self.load_mass_min + t.clamp(0.0, 1.0) * (self.load_mass_max - self.load_mass_min)
    }

    /// Linearly interpolate head friction within the configured range.
    #[inline]
    pub fn sample_head_friction(&self, t: f64) -> f64 {
        self.head_friction_min + t.clamp(0.0, 1.0) * (self.head_friction_max - self.head_friction_min)
    }

    /// Linearly interpolate ground friction within the configured range.
    #[inline]
    pub fn sample_ground_friction(&self, t: f64) -> f64 {
        self.ground_friction_min + t.clamp(0.0, 1.0) * (self.ground_friction_max - self.ground_friction_min)
    }
}
