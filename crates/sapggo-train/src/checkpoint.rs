use std::fs;
use std::path::Path;

/// Ensures the checkpoint directory exists.
pub fn ensure_checkpoint_dir(dir: &Path) -> anyhow::Result<()> {
    fs::create_dir_all(dir)
        .map_err(|e| anyhow::anyhow!("Failed to create checkpoint dir '{}': {}", dir.display(), e))
}

/// Builds a checkpoint file path for a given global step.
#[inline]
pub fn checkpoint_path(dir: &Path, global_step: u64) -> std::path::PathBuf {
    dir.join(format!("sapggo_step_{global_step}.bin"))
}

/// Builds the path for the final checkpoint.
#[inline]
pub fn final_checkpoint_path(dir: &Path) -> std::path::PathBuf {
    dir.join("sapggo_final.bin")
}

/// Serializes model weights to a file using `serde_json`.
///
/// In a production system this would use Burn's native checkpoint format;
/// here we provide a generic serialization interface.
#[allow(dead_code)]
pub fn save_checkpoint<T: serde::Serialize>(data: &T, path: &Path) -> anyhow::Result<()> {
    let json = serde_json::to_vec(data)
        .map_err(|e| anyhow::anyhow!("Checkpoint serialization failed: {e}"))?;
    fs::write(path, &json)
        .map_err(|e| anyhow::anyhow!("Failed to write checkpoint '{}': {}", path.display(), e))?;
    tracing::info!(path = %path.display(), size_bytes = json.len(), "Checkpoint saved");
    Ok(())
}

/// Deserializes model weights from a checkpoint file.
#[allow(dead_code)]
pub fn load_checkpoint<T: serde::de::DeserializeOwned>(path: &Path) -> anyhow::Result<T> {
    let bytes = fs::read(path)
        .map_err(|e| anyhow::anyhow!("Failed to read checkpoint '{}': {}", path.display(), e))?;
    let data: T = serde_json::from_slice(&bytes)
        .map_err(|e| anyhow::anyhow!("Checkpoint deserialization failed: {e}"))?;
    tracing::info!(path = %path.display(), "Checkpoint loaded");
    Ok(data)
}
