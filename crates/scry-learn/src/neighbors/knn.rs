//! K-Nearest Neighbors classifier and regressor.
//!
//! Supports two search algorithms:
//! - **Brute force**: O(n) per query. Always correct, any metric.
//! - **KD-tree**: O(log n) average per query. Euclidean only, best for < 20 features.
//!
//! Optimizations:
//! - Uses squared Euclidean distance (avoids sqrt — monotonic, same ordering).
//! - Uses `select_nth_unstable` for partial sort (O(n) vs O(n·log n)).
//! - Fixed-size vote array avoids HashMap overhead.

use crate::dataset::Dataset;
use crate::error::{Result, ScryLearnError};
use crate::neighbors::kdtree::KdTree;
use crate::weights::{ClassWeight, compute_sample_weights};

/// Distance metric for KNN.
///
/// # Example
///
/// ```
/// use scry_learn::neighbors::DistanceMetric;
///
/// let metric = DistanceMetric::Cosine;
/// ```
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum DistanceMetric {
    /// Euclidean distance (L2).
    Euclidean,
    /// Manhattan distance (L1).
    Manhattan,
    /// Cosine distance: `1 − cos(θ)`, range `[0, 2]`.
    Cosine,
}

/// Weighting function for neighbor votes.
///
/// Controls how the k-nearest neighbors contribute to predictions.
///
/// # Example
///
/// ```
/// use scry_learn::neighbors::{KnnClassifier, WeightFunction};
///
/// let knn = KnnClassifier::new()
///     .k(5)
///     .weights(WeightFunction::Distance);
/// ```
#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum WeightFunction {
    /// All neighbors have equal vote weight.
    #[default]
    Uniform,
    /// Closer neighbors contribute more: weight = `1 / distance`.
    ///
    /// When distance is zero (exact match), that neighbor gets all the weight.
    Distance,
}

/// Algorithm used for nearest-neighbor search.
///
/// # Example
///
/// ```
/// use scry_learn::neighbors::{KnnClassifier, Algorithm};
///
/// let knn = KnnClassifier::new()
///     .k(5)
///     .algorithm(Algorithm::KDTree);
/// ```
#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Algorithm {
    /// Automatically choose the best algorithm based on data and metric.
    ///
    /// Uses KD-tree for Euclidean distance with < 20 features,
    /// brute-force otherwise.
    #[default]
    Auto,
    /// Brute-force O(n) search. Works with all distance metrics.
    BruteForce,
    /// KD-tree O(log n) average-case search. Euclidean distance only.
    ///
    /// Falls back to brute-force if a non-Euclidean metric is selected.
    KDTree,
}

// ─────────────────────────────────────────────────────────────────
// KNN Classifier
// ─────────────────────────────────────────────────────────────────

/// K-Nearest Neighbors classifier.
///
/// Uses brute-force distance computation — fast enough for datasets up to ~100k samples.
///
/// # Example
///
/// ```
/// use scry_learn::dataset::Dataset;
/// use scry_learn::neighbors::{KnnClassifier, WeightFunction};
///
/// let features = vec![
///     vec![0.0, 0.0, 10.0, 10.0],
///     vec![0.0, 0.0, 10.0, 10.0],
/// ];
/// let target = vec![0.0, 0.0, 1.0, 1.0];
/// let data = Dataset::new(features, target, vec!["x".into(), "y".into()], "class");
///
/// let mut knn = KnnClassifier::new()
///     .k(3)
///     .weights(WeightFunction::Distance);
/// knn.fit(&data).unwrap();
///
/// let preds = knn.predict(&[vec![1.0, 1.0]]).unwrap();
/// assert_eq!(preds[0] as usize, 0);
/// ```
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct KnnClassifier {
    k: usize,
    metric: DistanceMetric,
    weight_fn: WeightFunction,
    class_weight: ClassWeight,
    algorithm: Algorithm,
    train_features: Vec<Vec<f64>>, // [n_samples][n_features]
    train_target: Vec<f64>,
    train_weights: Vec<f64>,
    n_classes: usize,
    kdtree: Option<KdTree>,
    fitted: bool,
}

impl KnnClassifier {
    /// Create a new KNN classifier with k=5.
    pub fn new() -> Self {
        Self {
            k: 5,
            metric: DistanceMetric::Euclidean,
            weight_fn: WeightFunction::Uniform,
            class_weight: ClassWeight::Uniform,
            algorithm: Algorithm::Auto,
            train_features: Vec::new(),
            train_target: Vec::new(),
            train_weights: Vec::new(),
            n_classes: 0,
            kdtree: None,
            fitted: false,
        }
    }

    /// Set the number of neighbors.
    pub fn k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Set the distance metric.
    pub fn metric(mut self, m: DistanceMetric) -> Self {
        self.metric = m;
        self
    }

    /// Set the neighbor weighting function.
    ///
    /// - [`WeightFunction::Uniform`]: every neighbor's vote counts equally.
    /// - [`WeightFunction::Distance`]: closer neighbors contribute more (weight = `1/d`).
    pub fn weights(mut self, w: WeightFunction) -> Self {
        self.weight_fn = w;
        self
    }

    /// Set class weighting strategy for imbalanced datasets.
    pub fn class_weight(mut self, cw: ClassWeight) -> Self {
        self.class_weight = cw;
        self
    }

    /// Set the nearest-neighbor search algorithm.
    ///
    /// - [`Algorithm::Auto`] (default): uses KD-tree for Euclidean distance
    ///   with fewer than 20 features, brute-force otherwise.
    /// - [`Algorithm::BruteForce`]: always O(n) brute-force scan.
    /// - [`Algorithm::KDTree`]: builds a KD-tree for O(log n) queries;
    ///   falls back to brute-force if a non-Euclidean metric is set.
    pub fn algorithm(mut self, algo: Algorithm) -> Self {
        self.algorithm = algo;
        self
    }

    /// Store training data. Builds a KD-tree if the selected algorithm
    /// (or `Auto` heuristic) calls for it.
    pub fn fit(&mut self, data: &Dataset) -> Result<()> {
        if data.n_samples() == 0 {
            return Err(ScryLearnError::EmptyDataset);
        }
        self.train_features = data.feature_matrix();
        self.train_target.clone_from(&data.target);
        self.train_weights = compute_sample_weights(&data.target, &self.class_weight);
        self.n_classes = data.n_classes();

        // Build KD-tree if appropriate.
        self.kdtree = if should_use_kdtree(self.algorithm, self.metric, data.n_features()) {
            Some(KdTree::build(&self.train_features))
        } else {
            None
        };

        self.fitted = true;
        Ok(())
    }

    /// Predict class labels.
    ///
    /// Uses partial sort (`select_nth_unstable`) to find k nearest neighbors
    /// in O(n) instead of full O(n·log n) sort. Euclidean distances skip sqrt
    /// since we only need relative ordering (unless distance weighting is on).
    pub fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> {
        if !self.fitted {
            return Err(ScryLearnError::NotFitted);
        }
        let probas = self.compute_votes(features);
        Ok(probas
            .into_iter()
            .map(|votes| {
                votes
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map_or(0.0, |(idx, _)| idx as f64)
            })
            .collect())
    }

    /// Predict class probability distribution for each sample.
    ///
    /// Returns a `Vec<Vec<f64>>` where `result[i][c]` is the estimated
    /// probability that sample `i` belongs to class `c`. Probabilities
    /// sum to 1.0 for each sample.
    ///
    /// # Example
    ///
    /// ```
    /// use scry_learn::dataset::Dataset;
    /// use scry_learn::neighbors::KnnClassifier;
    ///
    /// let features = vec![
    ///     vec![0.0, 0.0, 10.0, 10.0],
    ///     vec![0.0, 0.0, 10.0, 10.0],
    /// ];
    /// let target = vec![0.0, 0.0, 1.0, 1.0];
    /// let data = Dataset::new(features, target, vec!["x".into(), "y".into()], "class");
    ///
    /// let mut knn = KnnClassifier::new().k(3);
    /// knn.fit(&data).unwrap();
    ///
    /// let probas = knn.predict_proba(&[vec![1.0, 1.0]]).unwrap();
    /// let sum: f64 = probas[0].iter().sum();
    /// assert!((sum - 1.0).abs() < 1e-9);
    /// ```
    pub fn predict_proba(&self, features: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        if !self.fitted {
            return Err(ScryLearnError::NotFitted);
        }
        let votes = self.compute_votes(features);
        Ok(votes
            .into_iter()
            .map(|v| {
                let total: f64 = v.iter().sum();
                if total > 0.0 {
                    v.iter().map(|&x| x / total).collect()
                } else {
                    // Fallback: uniform distribution.
                    let n = v.len() as f64;
                    vec![1.0 / n; v.len()]
                }
            })
            .collect())
    }

    /// Core voting logic shared by `predict` and `predict_proba`.
    ///
    /// Returns raw weighted vote counts per class for each query sample.
    #[allow(clippy::option_if_let_else)]
    fn compute_votes(&self, features: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let k = self.k.min(self.train_features.len());
        let use_actual_dist = matches!(self.weight_fn, WeightFunction::Distance);
        let metric = self.metric;

        features
            .iter()
            .map(|query| {
                // Get k-nearest neighbor indices + distances.
                let neighbors: Vec<(f64, usize)> = if let Some(ref tree) = self.kdtree {
                    // KD-tree path — returns (dist², idx) pairs.
                    let raw = tree.query_k_nearest(query, k, &self.train_features);
                    if use_actual_dist {
                        // Convert squared distance to actual distance.
                        raw.into_iter().map(|(d2, i)| (d2.sqrt(), i)).collect()
                    } else {
                        raw
                    }
                } else {
                    // Brute-force path.
                    let mut dists: Vec<(f64, usize)> = self
                        .train_features
                        .iter()
                        .enumerate()
                        .map(|(i, train_row)| {
                            let d = if use_actual_dist {
                                actual_distance(query, train_row, metric)
                            } else {
                                distance_for_compare(query, train_row, metric)
                            };
                            (d, i)
                        })
                        .collect();

                    if k < dists.len() {
                        dists.select_nth_unstable_by(k - 1, |a, b| {
                            a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
                        });
                    }
                    dists.truncate(k);
                    dists
                };

                // Aggregate votes.
                let mut votes = vec![0.0_f64; self.n_classes.max(1)];

                if use_actual_dist {
                    let has_exact = neighbors.iter().any(|&(d, _)| d < f64::EPSILON);
                    if has_exact {
                        for &(d, idx) in &neighbors {
                            if d < f64::EPSILON {
                                let class = self.train_target[idx] as usize;
                                let w = self.train_weights[idx];
                                if class < votes.len() {
                                    votes[class] += w;
                                }
                            }
                        }
                    } else {
                        for &(d, idx) in &neighbors {
                            let class = self.train_target[idx] as usize;
                            let w = self.train_weights[idx];
                            if class < votes.len() {
                                votes[class] += w / d;
                            }
                        }
                    }
                } else {
                    for &(_, idx) in &neighbors {
                        let class = self.train_target[idx] as usize;
                        let w = self.train_weights[idx];
                        if class < votes.len() {
                            votes[class] += w;
                        }
                    }
                }

                votes
            })
            .collect()
    }
}

impl Default for KnnClassifier {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────
// KNN Regressor
// ─────────────────────────────────────────────────────────────────

/// K-Nearest Neighbors regressor.
///
/// Predicts the (optionally distance-weighted) mean of the k-nearest
/// training targets for each query point.
///
/// # Example
///
/// ```
/// use scry_learn::dataset::Dataset;
/// use scry_learn::neighbors::{KnnRegressor, WeightFunction};
///
/// let features = vec![vec![1.0, 2.0, 3.0]];
/// let target = vec![10.0, 20.0, 30.0];
/// let data = Dataset::new(features, target, vec!["x".into()], "y");
///
/// let mut knn = KnnRegressor::new()
///     .k(2)
///     .weights(WeightFunction::Uniform);
/// knn.fit(&data).unwrap();
///
/// let preds = knn.predict(&[vec![2.5]]).unwrap();
/// // Nearest neighbors are x=2 (y=20) and x=3 (y=30) → mean = 25
/// assert!((preds[0] - 25.0).abs() < 1e-9);
/// ```
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct KnnRegressor {
    k: usize,
    metric: DistanceMetric,
    weight_fn: WeightFunction,
    algorithm: Algorithm,
    train_features: Vec<Vec<f64>>, // [n_samples][n_features]
    train_target: Vec<f64>,
    kdtree: Option<KdTree>,
    fitted: bool,
}

impl KnnRegressor {
    /// Create a new KNN regressor with k=5.
    pub fn new() -> Self {
        Self {
            k: 5,
            metric: DistanceMetric::Euclidean,
            weight_fn: WeightFunction::Uniform,
            algorithm: Algorithm::Auto,
            train_features: Vec::new(),
            train_target: Vec::new(),
            kdtree: None,
            fitted: false,
        }
    }

    /// Set the number of neighbors.
    pub fn k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Set the distance metric.
    pub fn metric(mut self, m: DistanceMetric) -> Self {
        self.metric = m;
        self
    }

    /// Set the neighbor weighting function.
    ///
    /// - [`WeightFunction::Uniform`]: all k neighbors contribute equally to the mean.
    /// - [`WeightFunction::Distance`]: closer neighbors are weighted by `1/distance`.
    pub fn weights(mut self, w: WeightFunction) -> Self {
        self.weight_fn = w;
        self
    }

    /// Set the nearest-neighbor search algorithm.
    ///
    /// See [`Algorithm`] for details.
    pub fn algorithm(mut self, algo: Algorithm) -> Self {
        self.algorithm = algo;
        self
    }

    /// Store training data. Builds KD-tree if appropriate.
    pub fn fit(&mut self, data: &Dataset) -> Result<()> {
        if data.n_samples() == 0 {
            return Err(ScryLearnError::EmptyDataset);
        }
        self.train_features = data.feature_matrix();
        self.train_target.clone_from(&data.target);

        self.kdtree = if should_use_kdtree(self.algorithm, self.metric, data.n_features()) {
            Some(KdTree::build(&self.train_features))
        } else {
            None
        };

        self.fitted = true;
        Ok(())
    }

    /// Predict continuous target values.
    ///
    /// For each query point, finds the k nearest training samples and returns
    /// their mean (or distance-weighted mean) target value.
    #[allow(clippy::option_if_let_else)]
    pub fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> {
        if !self.fitted {
            return Err(ScryLearnError::NotFitted);
        }

        let k = self.k.min(self.train_features.len());
        let use_actual_dist = matches!(self.weight_fn, WeightFunction::Distance);
        let metric = self.metric;

        Ok(features
            .iter()
            .map(|query| {
                // Get (distance, idx) pairs via KD-tree or brute-force.
                let neighbors: Vec<(f64, usize)> = if let Some(ref tree) = self.kdtree {
                    let raw = tree.query_k_nearest(query, k, &self.train_features);
                    if use_actual_dist {
                        raw.into_iter().map(|(d2, i)| (d2.sqrt(), i)).collect()
                    } else {
                        raw
                    }
                } else {
                    let mut dists: Vec<(f64, usize)> = self
                        .train_features
                        .iter()
                        .enumerate()
                        .map(|(i, train_row)| {
                            let d = if use_actual_dist {
                                actual_distance(query, train_row, metric)
                            } else {
                                distance_for_compare(query, train_row, metric)
                            };
                            (d, i)
                        })
                        .collect();

                    if k < dists.len() {
                        dists.select_nth_unstable_by(k - 1, |a, b| {
                            a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
                        });
                    }
                    dists.truncate(k);
                    dists
                };

                if use_actual_dist {
                    let has_exact = neighbors.iter().any(|&(d, _)| d < f64::EPSILON);
                    if has_exact {
                        let (sum, count) = neighbors.iter().fold((0.0, 0usize), |(s, c), &(d, idx)| {
                            if d < f64::EPSILON { (s + self.train_target[idx], c + 1) } else { (s, c) }
                        });
                        sum / count as f64
                    } else {
                        let (weighted_sum, total_w) = neighbors.iter().fold((0.0, 0.0), |(ws, tw), &(d, idx)| {
                            let w = 1.0 / d;
                            (ws + w * self.train_target[idx], tw + w)
                        });
                        weighted_sum / total_w
                    }
                } else {
                    let sum: f64 = neighbors.iter().map(|&(_, idx)| self.train_target[idx]).sum();
                    sum / k as f64
                }
            })
            .collect())
    }
}

impl Default for KnnRegressor {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────
// Distance functions
// ─────────────────────────────────────────────────────────────────

/// Compute distance for comparison purposes (skips sqrt for Euclidean).
///
/// For Euclidean, returns squared distance (monotonic — preserves ordering).
/// For Manhattan and Cosine, returns the actual distance.
#[inline]
fn distance_for_compare(a: &[f64], b: &[f64], metric: DistanceMetric) -> f64 {
    match metric {
        DistanceMetric::Euclidean => a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>(),
        // No sqrt — squared distance preserves ordering.
        DistanceMetric::Manhattan => a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .sum(),
        DistanceMetric::Cosine => cosine_distance(a, b),
    }
}

/// Compute the actual distance (with sqrt for Euclidean).
///
/// Used when `WeightFunction::Distance` is active, since we need true
/// distances for the `1/d` weighting.
#[inline]
fn actual_distance(a: &[f64], b: &[f64], metric: DistanceMetric) -> f64 {
    match metric {
        DistanceMetric::Euclidean => a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt(),
        DistanceMetric::Manhattan => a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .sum(),
        DistanceMetric::Cosine => cosine_distance(a, b),
    }
}

/// Cosine distance: `1 − cos(θ)`, range `[0, 2]`.
///
/// Returns `1.0` when either vector has zero norm (treat as orthogonal).
#[inline]
fn cosine_distance(a: &[f64], b: &[f64]) -> f64 {
    let mut dot = 0.0_f64;
    let mut norm_a = 0.0_f64;
    let mut norm_b = 0.0_f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < f64::EPSILON {
        return 1.0; // One or both vectors are zero — treat as orthogonal.
    }
    1.0 - (dot / denom)
}

/// Decide whether to use the KD-tree based on algorithm selection, metric, and dimensionality.
fn should_use_kdtree(algo: Algorithm, metric: DistanceMetric, n_features: usize) -> bool {
    match algo {
        Algorithm::BruteForce => false,
        Algorithm::KDTree => matches!(metric, DistanceMetric::Euclidean),
        Algorithm::Auto => {
            matches!(metric, DistanceMetric::Euclidean) && n_features < 20
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knn_simple() {
        // Two clusters: class 0 near origin, class 1 near (10, 10).
        let features = vec![
            vec![0.0, 0.0, 0.0, 10.0, 10.0, 10.0],
            vec![0.0, 0.0, 0.0, 10.0, 10.0, 10.0],
        ];
        let target = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let data = Dataset::new(features, target, vec!["x".into(), "y".into()], "class");

        let mut knn = KnnClassifier::new().k(3);
        knn.fit(&data).unwrap();

        let preds = knn.predict(&[vec![1.0, 1.0], vec![9.0, 9.0]]).unwrap();
        assert!((preds[0] - 0.0).abs() < 1e-6);
        assert!((preds[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_knn_distance_weights() {
        // 3 class-0 samples far away, 2 class-1 samples very close to query.
        // With distance weights, class 1 should win (closer neighbors dominate).
        // Query at x=0.15: class-1 at 0.1 (d=0.05), 0.2 (d=0.05); class-0 at 5, 10, 10 (far).
        let features = vec![vec![5.0, 10.0, 10.0, 0.1, 0.2]];
        let target = vec![0.0, 0.0, 0.0, 1.0, 1.0];
        let data = Dataset::new(features, target, vec!["x".into()], "class");

        let mut knn_dist = KnnClassifier::new().k(5).weights(WeightFunction::Distance);
        knn_dist.fit(&data).unwrap();
        let preds_d = knn_dist.predict(&[vec![0.15]]).unwrap();
        assert_eq!(preds_d[0] as usize, 1, "Distance-weighted should pick closer class 1");
    }

    #[test]
    fn test_knn_predict_proba() {
        let features = vec![
            vec![0.0, 0.0, 10.0, 10.0],
            vec![0.0, 0.0, 10.0, 10.0],
        ];
        let target = vec![0.0, 0.0, 1.0, 1.0];
        let data = Dataset::new(features, target, vec!["x".into(), "y".into()], "class");

        let mut knn = KnnClassifier::new().k(4);
        knn.fit(&data).unwrap();

        let probas = knn.predict_proba(&[vec![1.0, 1.0], vec![5.0, 5.0]]).unwrap();
        for p in &probas {
            let sum: f64 = p.iter().sum();
            assert!((sum - 1.0).abs() < 1e-9, "Probabilities must sum to 1.0, got {sum}");
        }

        // Point near class 0 should have higher probability for class 0.
        assert!(probas[0][0] > 0.4, "Expected high prob for class 0 at (1,1)");
    }

    #[test]
    fn test_knn_cosine() {
        // Cosine distance ignores magnitude — direction matters.
        // [1, 0] and [100, 0] have same direction → distance ≈ 0.
        // [1, 0] and [0, 1] are orthogonal → distance ≈ 1.
        let d_same = cosine_distance(&[1.0, 0.0], &[100.0, 0.0]);
        let d_orth = cosine_distance(&[1.0, 0.0], &[0.0, 1.0]);
        assert!(d_same < 1e-9, "Same direction should have ~0 distance, got {d_same}");
        assert!((d_orth - 1.0).abs() < 1e-9, "Orthogonal should have distance ~1, got {d_orth}");

        // Use cosine metric in classifier.
        let features = vec![
            vec![1.0, 100.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 100.0],
        ];
        let target = vec![0.0, 0.0, 1.0, 1.0];
        let data = Dataset::new(features, target, vec!["x".into(), "y".into()], "class");

        let mut knn = KnnClassifier::new().k(2).metric(DistanceMetric::Cosine);
        knn.fit(&data).unwrap();

        // Query [50, 0] has same direction as class 0.
        let preds = knn.predict(&[vec![50.0, 0.0]]).unwrap();
        assert_eq!(preds[0] as usize, 0, "Cosine metric should match class 0 by direction");
    }

    #[test]
    fn test_knn_regressor_simple() {
        // 3 points: x=1→y=10, x=5→y=50, x=9→y=90
        let features = vec![vec![1.0, 5.0, 9.0]];
        let target = vec![10.0, 50.0, 90.0];
        let data = Dataset::new(features, target, vec!["x".into()], "y");

        let mut knn = KnnRegressor::new().k(2);
        knn.fit(&data).unwrap();

        // Query x=3: nearest are x=1(y=10) and x=5(y=50) → mean=30
        let preds = knn.predict(&[vec![3.0]]).unwrap();
        assert!((preds[0] - 30.0).abs() < 1e-9, "Expected 30.0, got {}", preds[0]);

        // Query x=7: nearest are x=5(y=50) and x=9(y=90) → mean=70
        let preds2 = knn.predict(&[vec![7.0]]).unwrap();
        assert!((preds2[0] - 70.0).abs() < 1e-9, "Expected 70.0, got {}", preds2[0]);
    }

    #[test]
    fn test_knn_regressor_distance_weights() {
        // x=0→y=0, x=10→y=100. Query at x=1 (much closer to x=0).
        // Uniform: mean(0, 100) = 50.
        // Distance: weighted toward x=0 → should be << 50.
        let features = vec![vec![0.0, 10.0]];
        let target = vec![0.0, 100.0];
        let data = Dataset::new(features, target, vec!["x".into()], "y");

        let mut knn_u = KnnRegressor::new().k(2);
        knn_u.fit(&data).unwrap();
        let pred_u = knn_u.predict(&[vec![1.0]]).unwrap()[0];
        assert!((pred_u - 50.0).abs() < 1e-9, "Uniform should give 50.0");

        let mut knn_d = KnnRegressor::new().k(2).weights(WeightFunction::Distance);
        knn_d.fit(&data).unwrap();
        let pred_d = knn_d.predict(&[vec![1.0]]).unwrap()[0];
        // 1/1 * 0 + 1/9 * 100 = 11.11... / (1 + 0.111...) = ~10
        assert!(pred_d < 20.0, "Distance-weighted should favor x=0, got {pred_d}");
    }

    #[test]
    fn test_knn_not_fitted() {
        let knn = KnnClassifier::new();
        assert!(knn.predict(&[vec![1.0]]).is_err());
        assert!(knn.predict_proba(&[vec![1.0]]).is_err());

        let knn_r = KnnRegressor::new();
        assert!(knn_r.predict(&[vec![1.0]]).is_err());
    }
}
