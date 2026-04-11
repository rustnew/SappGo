use sapggo_agent::Policy;
use sapggo_env::SapggoEnv;

/// Aggregated evaluation metrics across multiple episodes.
#[derive(Debug, Clone)]
pub struct EvalMetrics {
    pub mean_distance_m:    f64,
    pub mean_steps:         f64,
    pub load_drop_rate:     f64,
    pub mean_load_tilt_deg: f64,
    pub mean_speed_ms:      f64,
    pub n_episodes:         usize,
}

impl std::fmt::Display for EvalMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Evaluation Results ({} episodes)", self.n_episodes)?;
        writeln!(f, "  Mean distance:   {:.1} m",   self.mean_distance_m)?;
        writeln!(f, "  Mean steps:      {:.0}",      self.mean_steps)?;
        writeln!(f, "  Load drop rate:  {:.1}%",     self.load_drop_rate * 100.0)?;
        writeln!(f, "  Mean speed:      {:.2} m/s",  self.mean_speed_ms)?;
        write!(f,   "  Mean load tilt:  {:.1}°",     self.mean_load_tilt_deg)
    }
}

/// Runs a deterministic evaluation over `episodes` episodes.
///
/// The agent acts using its mean (deterministic) policy. No exploration
/// noise is applied.
pub fn run_evaluation(
    env:      &mut SapggoEnv,
    agent:    &dyn Policy,
    episodes: usize,
) -> anyhow::Result<EvalMetrics> {
    let mut distances   = Vec::with_capacity(episodes);
    let mut steps_vec   = Vec::with_capacity(episodes);
    let mut drops       = 0usize;
    let mut total_speed = 0.0f64;

    for ep in 0..episodes {
        let mut obs  = env.reset();
        let mut done = false;
        let mut ep_steps = 0u64;

        while !done {
            let action = agent.act_deterministic(obs.as_slice().unwrap_or(&[]));
            let result = env.step(&action)?;
            obs      = result.observation;
            done     = result.done;
            ep_steps += 1;
        }

        if let Some(info) = env.last_info() {
            distances.push(info.distance_m);
            steps_vec.push(ep_steps);
            if info.load_dropped {
                drops += 1;
            }
            // Speed = distance / (steps * control_dt)
            let control_dt = 0.02; // 10 sim steps × 2 ms
            if ep_steps > 0 {
                total_speed += info.distance_m / (ep_steps as f64 * control_dt);
            }

            tracing::info!(
                episode  = ep + 1,
                distance = format!("{:.1}", info.distance_m),
                steps    = ep_steps,
                dropped  = info.load_dropped,
                "Eval episode complete",
            );
        }
    }

    let n = episodes.max(1) as f64;
    let total_dist:  f64 = distances.iter().sum();
    let total_steps: u64 = steps_vec.iter().sum();

    Ok(EvalMetrics {
        mean_distance_m:    total_dist / n,
        mean_steps:         total_steps as f64 / n,
        load_drop_rate:     drops as f64 / n,
        mean_load_tilt_deg: 0.0, // requires per-step tilt tracking
        mean_speed_ms:      total_speed / n,
        n_episodes:         episodes,
    })
}
