use std::path::{Path, PathBuf};

use sapggo_curriculum::TerrainType;

use crate::error::{EnvError, EnvResult};

/// Resolves the heightmap file path for a given terrain type relative to an
/// assets directory.
pub fn terrain_path(assets_dir: &Path, terrain: TerrainType) -> EnvResult<PathBuf> {
    let filename = terrain.asset_filename();
    let path = assets_dir.join(filename);
    if path.exists() {
        Ok(path)
    } else {
        Err(EnvError::TerrainNotFound {
            path: path.display().to_string(),
        })
    }
}

/// Validates that all required terrain assets exist in the given directory.
pub fn validate_terrain_assets(assets_dir: &Path) -> EnvResult<()> {
    for tt in &[TerrainType::Flat, TerrainType::Rolling, TerrainType::Rough] {
        terrain_path(assets_dir, *tt)?;
    }
    Ok(())
}
