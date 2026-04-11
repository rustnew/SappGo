//! Manual keyboard-controlled demo of the SAPGGO environment.
//!
//! This example creates the environment and runs episodes with zero actions
//! (standing still) to verify the simulation pipeline.
//!
//! In a full implementation, this would integrate with the MuJoCo viewer
//! for real-time keyboard control.
//!
//! Usage:
//!   cargo run --example manual_control

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_target(false)
        .init();

    tracing::info!("Manual control demo — standing with zero actions");

    let mut env = sapggo_env::SapggoEnv::new("assets/robot_humanoid_load.xml", 42)?;
    let _obs = env.reset();

    let zero_action = vec![0.0f64; sapggo_env::ACT_DIM];

    for step in 0..500 {
        let result = env.step(&zero_action)?;

        if step % 50 == 0 {
            tracing::info!(
                step     = step,
                reward   = format!("{:.3}", result.reward),
                distance = format!("{:.3}", result.info.distance_m),
                "Step",
            );
        }

        if result.done {
            tracing::info!(
                steps   = result.info.steps,
                dropped = result.info.load_dropped,
                fallen  = result.info.robot_fallen,
                "Episode terminated",
            );
            break;
        }
    }

    Ok(())
}
