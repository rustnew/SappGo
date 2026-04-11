use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

/// CSV-based training metric logger.
///
/// Writes one row per logged event to a CSV file for later analysis.
pub struct TrainingLogger {
    file:    File,
    #[allow(dead_code)]
    path:    PathBuf,
    flushed: bool,
}

impl TrainingLogger {
    /// Creates a new logger, writing to `{log_dir}/metrics.csv`.
    ///
    /// Creates the directory and file if they do not exist.
    pub fn new(log_dir: &Path) -> anyhow::Result<Self> {
        fs::create_dir_all(log_dir)
            .map_err(|e| anyhow::anyhow!("Failed to create log dir '{}': {}", log_dir.display(), e))?;

        let path = log_dir.join("metrics.csv");
        let file_exists = path.exists();

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| anyhow::anyhow!("Failed to open log file '{}': {}", path.display(), e))?;

        let mut logger = Self { file, path, flushed: false };

        // Write header if the file is new
        if !file_exists {
            logger.write_header()?;
        }

        Ok(logger)
    }

    /// Writes the CSV header row.
    fn write_header(&mut self) -> anyhow::Result<()> {
        writeln!(self.file, "tag,value,global_step")
            .map_err(|e| anyhow::anyhow!("Failed to write log header: {e}"))?;
        Ok(())
    }

    /// Logs a single metric.
    pub fn log(&mut self, tag: &str, value: f64, global_step: u64) {
        // Best-effort write — log errors via tracing, don't crash training.
        if let Err(e) = writeln!(self.file, "{tag},{value},{global_step}") {
            tracing::warn!(error = %e, tag, "Failed to write metric to log file");
        }
        self.flushed = false;
    }

    /// Flushes buffered writes to disk.
    pub fn flush(&mut self) {
        if !self.flushed {
            if let Err(e) = self.file.flush() {
                tracing::warn!(error = %e, "Failed to flush log file");
            }
            self.flushed = true;
        }
    }

    /// Returns the path to the log file.
    #[allow(dead_code)]
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TrainingLogger {
    fn drop(&mut self) {
        self.flush();
    }
}
