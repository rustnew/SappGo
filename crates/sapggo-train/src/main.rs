use std::path::PathBuf;

use anyhow::Context;
use clap::Parser;
use tracing_subscriber::EnvFilter;

mod checkpoint;
mod config;
mod episode_log;
mod logger;
mod trainer;

use config::TrainConfig;
use trainer::Trainer;

/// SAPGGO Training CLI
#[derive(Parser, Debug)]
#[command(name = "sapggo-train", version, about = "Train the SAPGGO humanoid agent")]
struct Cli {
    /// Path to a TOML configuration file.
    #[arg(short, long, default_value = "configs/train_default.toml")]
    config: PathBuf,

    /// Resume training from a checkpoint file.
    #[arg(short, long)]
    resume: Option<PathBuf>,

    /// Override the random seed.
    #[arg(short, long)]
    seed: Option<u64>,
}

fn main() -> anyhow::Result<()> {
    // Initialize structured logging
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_target(false)
        .with_timer(tracing_subscriber::fmt::time::uptime())
        .init();

    let cli = Cli::parse();

    // Load configuration
    let mut config = if cli.config.exists() {
        TrainConfig::from_file(&cli.config)
            .with_context(|| format!("Loading config from '{}'", cli.config.display()))?
    } else {
        tracing::warn!(
            path = %cli.config.display(),
            "Config file not found — using defaults",
        );
        TrainConfig::default()
    };

    // Apply CLI overrides
    if let Some(seed) = cli.seed {
        config.seed = seed;
    }

    tracing::info!(
        model    = %config.model_path,
        seed     = config.seed,
        steps    = config.total_steps,
        curriculum = config.curriculum,
        "Configuration loaded",
    );

    // Resume checkpoint (placeholder — would load Burn model records)
    if let Some(ref ckpt_path) = cli.resume {
        tracing::info!(path = %ckpt_path.display(), "Resuming from checkpoint");
    }

    // Run training
    let mut trainer = Trainer::new(config)
        .context("Failed to initialize trainer")?;

    trainer.run().context("Training failed")?;

    Ok(())
}
