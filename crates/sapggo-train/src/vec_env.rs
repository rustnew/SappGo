use ndarray::Array1;
use rayon::prelude::*;

use sapggo_env::SapggoEnv;
use sapggo_curriculum::CurriculumParams;

/// A single environment's episode statistics.
pub struct EpisodeResult {
    pub total_reward: f64,
    pub distance_m:   f64,
    pub steps:        u64,
    pub load_dropped: bool,
    pub robot_fallen: bool,
    pub stuck:        bool,
}

/// Collected transition from one step of one environment.
pub struct EnvTransition {
    pub env_id:      usize,
    pub observation: Vec<f64>,
    pub action:      Vec<f64>,
    pub reward:      f64,
    pub done:        bool,
    pub episode:     Option<EpisodeResult>,
}

/// Vectorized environment wrapper: runs N environments and can step them in parallel.
///
/// Each environment is independent and maintains its own state. Parallel stepping
/// via `rayon` accelerates rollout collection and decorrelates experiences.
pub struct VecEnv {
    envs:    Vec<SapggoEnv>,
    obs:     Vec<Array1<f64>>,
    n_envs:  usize,
}

impl VecEnv {
    /// Creates `n_envs` independent environments, each with a unique seed.
    pub fn new(model_path: &str, base_seed: u64, n_envs: usize) -> anyhow::Result<Self> {
        let mut envs = Vec::with_capacity(n_envs);
        let mut obs  = Vec::with_capacity(n_envs);
        for i in 0..n_envs {
            let mut env = SapggoEnv::new(model_path, base_seed + i as u64)?;
            let o = env.reset();
            obs.push(o);
            envs.push(env);
        }
        Ok(Self { envs, obs, n_envs })
    }

    /// Returns the number of parallel environments.
    #[inline]
    pub fn num_envs(&self) -> usize {
        self.n_envs
    }

    /// Returns a reference to the current observations for all environments.
    pub fn observations(&self) -> &[Array1<f64>] {
        &self.obs
    }

    /// Steps all environments in parallel with the given actions.
    ///
    /// `actions[i]` is the action for environment `i`.
    /// Returns a vector of transitions (one per env).
    pub fn step_all(&mut self, actions: &[Vec<f64>]) -> anyhow::Result<Vec<EnvTransition>> {
        assert_eq!(actions.len(), self.n_envs);

        // Pair (env, obs, action) and step in parallel
        let results: Vec<_> = self.envs.par_iter_mut()
            .zip(self.obs.par_iter_mut())
            .enumerate()
            .map(|(i, (env, obs))| {
                let result = env.step(&actions[i]).expect("env step failed");
                let episode = if result.done {
                    let info = &result.info;
                    let ep = EpisodeResult {
                        total_reward: info.total_reward,
                        distance_m:   info.distance_m,
                        steps:        info.steps,
                        load_dropped: info.load_dropped,
                        robot_fallen: info.robot_fallen,
                        stuck:        info.stuck,
                    };
                    *obs = env.reset();
                    Some(ep)
                } else {
                    *obs = result.observation;
                    None
                };

                EnvTransition {
                    env_id: i,
                    observation: obs.to_vec(),
                    action: actions[i].clone(),
                    reward: result.reward,
                    done:   result.done,
                    episode,
                }
            })
            .collect();

        Ok(results)
    }

    /// Resets all environments.
    pub fn reset_all(&mut self) {
        for (env, obs) in self.envs.iter_mut().zip(self.obs.iter_mut()) {
            *obs = env.reset();
        }
    }

    /// Sets curriculum params on all environments.
    pub fn set_params_all(&mut self, params: CurriculumParams) -> anyhow::Result<()> {
        for env in &mut self.envs {
            env.set_params(params.clone())?;
        }
        Ok(())
    }
}
