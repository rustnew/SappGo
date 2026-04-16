use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

/// Per-episode CSV logger that records every single episode's outcome.
///
/// Columns:
///   episode, global_step, steps, reward, distance_m, load_dropped, robot_fallen,
///   best_reward, is_best, curriculum_stage
pub struct EpisodeLog {
    file:    File,
    path:    PathBuf,
}

impl EpisodeLog {
    /// Creates a new episode log, writing to `{log_dir}/episodes.csv`.
    pub fn new(log_dir: &Path) -> anyhow::Result<Self> {
        fs::create_dir_all(log_dir)?;
        let path = log_dir.join("episodes.csv");
        let file_exists = path.exists();
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;

        let mut logger = Self { file, path };
        if !file_exists {
            logger.write_header()?;
        }
        Ok(logger)
    }

    fn write_header(&mut self) -> anyhow::Result<()> {
        writeln!(
            self.file,
            "episode,global_step,steps,reward,distance_m,load_dropped,robot_fallen,best_reward,is_best,curriculum_stage"
        )?;
        Ok(())
    }

    /// Logs one completed episode.
    pub fn log_episode(
        &mut self,
        episode:      u64,
        global_step:  u64,
        steps:        u64,
        reward:       f64,
        distance_m:   f64,
        load_dropped: bool,
        robot_fallen: bool,
        best_reward:  f64,
        is_best:      bool,
        stage:        &str,
    ) {
        let _ = writeln!(
            self.file,
            "{episode},{global_step},{steps},{reward:.4},{distance_m:.4},{load_dropped},{robot_fallen},{best_reward:.4},{is_best},{stage}"
        );
    }

    /// Flushes buffered writes to disk.
    pub fn flush(&mut self) {
        let _ = self.file.flush();
    }

    /// Returns the path to the log file.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for EpisodeLog {
    fn drop(&mut self) {
        self.flush();
    }
}
