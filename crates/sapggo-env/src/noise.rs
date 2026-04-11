use rand::Rng;
use rand_distr::{Distribution, Normal};

use crate::error::{EnvError, EnvResult};

/// Gaussian noise sampler with configurable standard deviation.
///
/// When `sigma` is zero the sampler returns `0.0` without touching the RNG,
/// avoiding unnecessary computation on noiseless stages.
pub struct GaussianNoise {
    dist:  Option<Normal<f64>>,
    sigma: f64,
}

impl GaussianNoise {
    /// Creates a new Gaussian noise source.
    ///
    /// Returns `Err` if `sigma` is negative or NaN.
    pub fn new(sigma: f64) -> EnvResult<Self> {
        if sigma <= 0.0 || !sigma.is_finite() {
            return Ok(Self { dist: None, sigma: 0.0 });
        }
        let dist = Normal::new(0.0, sigma).map_err(|e| {
            EnvError::Distribution(format!("Invalid noise sigma {sigma}: {e}"))
        })?;
        Ok(Self { dist: Some(dist), sigma })
    }

    /// Samples a single noise value.
    #[inline]
    pub fn sample(&self, rng: &mut impl Rng) -> f64 {
        match &self.dist {
            Some(d) => d.sample(rng),
            None    => 0.0,
        }
    }

    /// Returns the configured sigma.
    #[inline]
    pub fn sigma(&self) -> f64 {
        self.sigma
    }
}
