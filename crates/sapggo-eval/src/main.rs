use std::path::PathBuf;

use anyhow::Context;
use clap::Parser;
use tracing_subscriber::EnvFilter;

mod evaluator;

use evaluator::run_evaluation;

/// Evaluation configuration loaded from TOML.
#[derive(Debug, Clone, serde::Deserialize)]
struct EvalConfig {
    model_path:  String,
    policy_path: String,
    n_episodes:  usize,
    render:      bool,
    seed:        u64,
}

/// SAPGGO Evaluation CLI
#[derive(Parser, Debug)]
#[command(name = "sapggo-eval", version, about = "Evaluate a trained SAPGGO policy")]
struct Cli {
    /// Path to a TOML evaluation config file.
    #[arg(short, long, default_value = "configs/eval.toml")]
    config: PathBuf,

    /// Path to a trained policy checkpoint (overrides config).
    #[arg(short, long)]
    policy: Option<PathBuf>,

    /// Number of evaluation episodes (overrides config).
    #[arg(short, long)]
    episodes: Option<usize>,

    /// Enable rendering (overrides config).
    #[arg(short, long)]
    render: bool,
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

    // Load config
    let eval_cfg: EvalConfig = if cli.config.exists() {
        let contents = std::fs::read_to_string(&cli.config)
            .with_context(|| format!("Reading eval config '{}'", cli.config.display()))?;
        toml::from_str(&contents)
            .with_context(|| format!("Parsing eval config '{}'", cli.config.display()))?
    } else {
        tracing::warn!(
            path = %cli.config.display(),
            "Eval config not found — using defaults",
        );
        EvalConfig {
            model_path:  "assets/robot_humanoid_load.xml".into(),
            policy_path: "checkpoints/sapggo_final.bin".into(),
            n_episodes:  20,
            render:      false,
            seed:        100,
        }
    };

    let n_episodes = cli.episodes.unwrap_or(eval_cfg.n_episodes);
    let _render    = cli.render || eval_cfg.render;

    tracing::info!(
        model    = %eval_cfg.model_path,
        policy   = %eval_cfg.policy_path,
        episodes = n_episodes,
        "Starting evaluation",
    );

    // Create environment
    let mut env = sapggo_env::SapggoEnv::new(&eval_cfg.model_path, eval_cfg.seed)
        .context("Failed to create evaluation environment")?;

    // Load policy (placeholder — would load actual Burn model)
    let agent = RandomAgent;

    // Run evaluation
    let metrics = run_evaluation(&mut env, &agent, n_episodes)
        .context("Evaluation failed")?;

    println!("\n{metrics}");

    tracing::info!("Evaluation complete");

    Ok(())
}

/// Placeholder random agent for testing the evaluation pipeline.
struct RandomAgent;

impl sapggo_agent::Policy for RandomAgent {
    fn act_deterministic(&self, _obs: &[f64]) -> Vec<f64> {
        vec![0.0; sapggo_env::ACT_DIM]
    }

    fn act_stochastic(&self, _obs: &[f64]) -> (Vec<f64>, f64, f64) {
        (vec![0.0; sapggo_env::ACT_DIM], 0.0, 0.0)
    }
}
