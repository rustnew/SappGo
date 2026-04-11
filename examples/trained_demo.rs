//! Runs a trained policy checkpoint through the SAPGGO environment.
//!
//! Usage:
//!   cargo run --example trained_demo -- --checkpoint checkpoints/sapggo_final.bin

use clap::Parser;

#[derive(Parser)]
struct Args {
    /// Path to a saved policy checkpoint.
    #[arg(long, default_value = "checkpoints/sapggo_final.bin")]
    checkpoint: String,

    /// Number of episodes to run.
    #[arg(long, default_value_t = 5)]
    episodes: usize,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_target(false)
        .init();

    let args = Args::parse();

    tracing::info!(
        checkpoint = %args.checkpoint,
        episodes   = args.episodes,
        "Loading trained policy",
    );

    // In a full implementation, the checkpoint would be deserialized into
    // the Burn Actor/Critic networks. For now, use a zero-action placeholder.

    let mut env = sapggo_env::SapggoEnv::new("assets/robot_humanoid_load.xml", 100)?;

    for ep in 0..args.episodes {
        let _obs = env.reset();
        let mut done = false;
        let mut steps = 0u64;

        while !done {
            // Placeholder: zero action (would be policy.act_deterministic(&obs))
            let action = vec![0.0f64; sapggo_env::ACT_DIM];
            let result = env.step(&action)?;
            done = result.done;
            steps += 1;
        }

        if let Some(info) = env.last_info() {
            tracing::info!(
                episode  = ep + 1,
                steps    = steps,
                distance = format!("{:.2}", info.distance_m),
                reward   = format!("{:.2}", info.total_reward),
                dropped  = info.load_dropped,
                "Demo episode complete",
            );
        }
    }

    Ok(())
}
