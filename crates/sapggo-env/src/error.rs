use thiserror::Error;

/// Errors that can occur within the SAPGGO environment.
#[derive(Debug, Error)]
pub enum EnvError {
    #[error("Failed to load MuJoCo model from '{path}': {source}")]
    ModelLoad {
        path:   String,
        source: anyhow::Error,
    },

    #[error("MuJoCo body '{name}' not found in model")]
    BodyNotFound { name: String },

    #[error("MuJoCo sensor '{name}' not found in model")]
    SensorNotFound { name: String },

    #[error("MuJoCo joint '{name}' not found in model")]
    JointNotFound { name: String },

    #[error("Invalid action dimension: expected {expected}, got {got}")]
    ActionDimMismatch { expected: usize, got: usize },

    #[error("Simulation step failed: {0}")]
    SimulationStep(anyhow::Error),

    #[error("Terrain file not found: {path}")]
    TerrainNotFound { path: String },

    #[error("Random number distribution error: {0}")]
    Distribution(String),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// Convenience result type for environment operations.
pub type EnvResult<T> = Result<T, EnvError>;
