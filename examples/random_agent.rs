//! Random agent baseline: applies random actions to the SAPGGO environment.
//!
//! Usage:
//!   cargo run --example random_agent -- --episodes 5

use clap::Parser;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 3)]
    episodes: usize,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_target(false)
        .init();

    let args = Args::parse();

    let mut env = sapggo_env::SapggoEnv::new("assets/robot_humanoid_load.xml", 0)?;

    for ep in 0..args.episodes {
        let _obs = env.reset();
        let mut done = false;
        let mut steps = 0u64;
        let mut total_reward = 0.0f64;

        while !done {
            // Random uniform actions in [-1, 1]
            let action: Vec<f64> = (0..sapggo_env::ACT_DIM)
                .map(|_| rand::random::<f64>() * 2.0 - 1.0)
                .collect();

            let result = env.step(&action)?;
            total_reward += result.reward;
            done = result.done;
            steps += 1;
        }

        if let Some(info) = env.last_info() {
            tracing::info!(
                episode  = ep + 1,
                steps    = steps,
                reward   = format!("{total_reward:.2}"),
                distance = format!("{:.2}", info.distance_m),
                dropped  = info.load_dropped,
                fallen   = info.robot_fallen,
                "Episode complete",
            );
        }
    }

    Ok(())
}
