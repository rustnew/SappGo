/// Running observation normalizer using Welford's online algorithm.
///
/// Maintains per-dimension mean and variance estimates and normalizes
/// observations to approximately zero mean and unit variance, which
/// greatly stabilizes neural network training.
#[derive(Debug, Clone)]
pub struct RunningNormalizer {
    mean:  Vec<f64>,
    var:   Vec<f64>,
    count: f64,
    dim:   usize,
    /// Small constant added to variance to avoid division by zero.
    epsilon: f64,
}

impl RunningNormalizer {
    /// Creates a new normalizer for vectors of dimension `dim`.
    pub fn new(dim: usize) -> Self {
        Self {
            mean:    vec![0.0; dim],
            var:     vec![1.0; dim],
            count:   1e-4, // small initial count to avoid div-by-zero on first update
            dim,
            epsilon: 1e-8,
        }
    }

    /// Updates running statistics with a new observation.
    ///
    /// Uses Welford's online algorithm for numerically stable incremental
    /// mean and variance computation.
    pub fn update(&mut self, obs: &[f64]) {
        debug_assert_eq!(obs.len(), self.dim);

        self.count += 1.0;
        let batch_count = self.count;

        for i in 0..self.dim {
            let delta     = obs[i] - self.mean[i];
            self.mean[i] += delta / batch_count;
            let delta2    = obs[i] - self.mean[i];
            self.var[i]  += delta * delta2;
        }
    }

    /// Normalizes an observation vector in-place.
    #[inline]
    pub fn normalize_inplace(&self, obs: &mut [f64]) {
        debug_assert_eq!(obs.len(), self.dim);

        let inv_count = if self.count > 1.0 { 1.0 / (self.count - 1.0) } else { 1.0 };

        for i in 0..self.dim {
            let std = (self.var[i] * inv_count + self.epsilon).sqrt();
            obs[i] = (obs[i] - self.mean[i]) / std;
        }
    }

    /// Returns a normalized copy of the observation.
    pub fn normalize(&self, obs: &[f64]) -> Vec<f64> {
        let mut out = obs.to_vec();
        self.normalize_inplace(&mut out);
        out
    }

    /// Returns the current mean vector.
    pub fn mean(&self) -> &[f64] {
        &self.mean
    }

    /// Returns the current variance vector (unnormalized by count).
    pub fn variance(&self) -> &[f64] {
        &self.var
    }

    /// Returns the number of observations processed.
    pub fn count(&self) -> f64 {
        self.count
    }

    /// Dimensionality.
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }
}
