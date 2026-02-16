//! Model-level `.viz()` interface — every fitted model gets visualization methods.
//!
//! The [`Visualize`] trait provides a `.viz()` entry point that returns a
//! [`ModelViz`] builder.  From there, model-family-specific methods produce
//! [`Chart`] instances ready for terminal rendering, PNG, or SVG export.

use scry_chart::chart::{Chart, BarChart, Heatmap, LineChart};
use scry_chart::data::Series;
use scry_chart::theme::Theme;

use crate::error::{Result, ScryLearnError};

// ---------------------------------------------------------------------------
// Trait + Builder
// ---------------------------------------------------------------------------

/// Trait implemented by all fitted models to expose visualization helpers.
///
/// Call `.viz()` on any model to get a [`ModelViz`] builder with
/// chart-producing methods appropriate for that model family.
///
/// ```ignore
/// let mut rf = RandomForestClassifier::new().n_estimators(50);
/// rf.fit(&train)?;
/// let chart = rf.viz().feature_importance(Some(10))?;
/// ```
pub trait Visualize {
    /// Return a visualization builder wrapping this model.
    fn viz(&self) -> ModelViz<'_, Self>
    where
        Self: Sized,
    {
        ModelViz {
            model: self,
            feature_names: None,
        }
    }
}

/// Builder returned by [`Visualize::viz()`].
///
/// Holds a reference to the fitted model and optional overrides
/// (like custom feature names).
pub struct ModelViz<'a, M: ?Sized> {
    model: &'a M,
    feature_names: Option<Vec<String>>,
}

impl<M> ModelViz<'_, M> {
    /// Override default feature names (`feature_0`, `feature_1`, ...).
    pub fn feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = Some(names);
        self
    }
}

/// Standard ML-viz dark theme (re-used from standalone viz functions).
fn ml_theme() -> Theme {
    Theme::dark()
}

/// Generate default feature names: `feature_0`, `feature_1`, ...
fn default_feature_names(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("feature_{i}")).collect()
}

// ---------------------------------------------------------------------------
// Tree models — feature importance chart
// ---------------------------------------------------------------------------

/// Internal trait for models that expose feature importances.
trait HasFeatureImportance {
    fn get_feature_importances(&self) -> Result<Vec<f64>>;
}

/// Generate a feature importance bar chart.
fn tree_feature_importance<M: HasFeatureImportance>(
    viz: &ModelViz<'_, M>,
    top_n: Option<usize>,
) -> Result<Chart> {
    let importances = viz.model.get_feature_importances()?;
    let n = importances.len();
    let names = viz
        .feature_names
        .clone()
        .unwrap_or_else(|| default_feature_names(n));

    Ok(super::feature_importance_chart(&names, &importances, top_n))
}

macro_rules! impl_tree_viz {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl HasFeatureImportance for $ty {
                fn get_feature_importances(&self) -> Result<Vec<f64>> {
                    self.feature_importances()
                }
            }

            impl Visualize for $ty {}

            impl ModelViz<'_, $ty> {
                /// Feature importance bar chart (sorted descending).
                ///
                /// `top_n` limits how many features to show (default: all).
                pub fn feature_importance(&self, top_n: Option<usize>) -> Result<Chart> {
                    tree_feature_importance(self, top_n)
                }
            }
        )+
    };
}

use crate::tree::{
    DecisionTreeClassifier, DecisionTreeRegressor,
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
};

impl_tree_viz!(
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
);

// ---------------------------------------------------------------------------
// Linear models — coefficient chart
// ---------------------------------------------------------------------------

/// Internal trait for models that expose a 1-D coefficient vector.
trait HasCoefficients {
    fn coef_vec(&self) -> &[f64];
}

/// Generate a coefficient bar chart.
fn linear_coefficient_chart<M: HasCoefficients>(
    viz: &ModelViz<'_, M>,
) -> Chart {
    let coefs = viz.model.coef_vec();
    let n = coefs.len();
    let names = viz
        .feature_names
        .clone()
        .unwrap_or_else(|| default_feature_names(n));

    let mut pairs: Vec<(String, f64)> = names
        .into_iter()
        .zip(coefs.iter().copied())
        .collect();
    // Sort by absolute value descending.
    pairs.sort_by(|a, b| {
        b.1.abs()
            .partial_cmp(&a.1.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let labels: Vec<String> = pairs.iter().map(|(n, _)| n.clone()).collect();
    let values: Vec<f64> = pairs.iter().map(|(_, v)| *v).collect();

    BarChart::new(labels, vec![Series::new("Coefficient", values)])
        .horizontal()
        .title("Model Coefficients")
        .show_values()
        .theme(ml_theme())
        .build()
}

macro_rules! impl_linear_viz {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl Visualize for $ty {}

            impl ModelViz<'_, $ty> {
                /// Coefficient bar chart (sorted by absolute value).
                pub fn coefficient_chart(&self) -> Chart {
                    linear_coefficient_chart(self)
                }
            }
        )+
    };
}

use crate::linear::{LinearRegression, LassoRegression, ElasticNet, Ridge};

impl HasCoefficients for LinearRegression {
    fn coef_vec(&self) -> &[f64] { self.coefficients() }
}

impl HasCoefficients for LassoRegression {
    fn coef_vec(&self) -> &[f64] { self.coefficients() }
}

impl HasCoefficients for ElasticNet {
    fn coef_vec(&self) -> &[f64] { self.coefficients() }
}

impl HasCoefficients for Ridge {
    fn coef_vec(&self) -> &[f64] { self.coefficients() }
}

impl_linear_viz!(LinearRegression, LassoRegression, ElasticNet, Ridge);

// LogisticRegression is special — it has weights() -> &[Vec<f64>] (one per class).
use crate::linear::LogisticRegression;

impl Visualize for LogisticRegression {}

impl ModelViz<'_, LogisticRegression> {
    /// Coefficient chart for logistic regression.
    ///
    /// For binary classification, shows the single weight vector.
    /// For multiclass, shows the mean absolute weight per feature.
    pub fn coefficient_chart(&self) -> Chart {
        let weights = self.model.weights();
        let n_features = if weights.is_empty() { 0 } else { weights[0].len() };
        let names = self
            .feature_names
            .clone()
            .unwrap_or_else(|| default_feature_names(n_features));

        let coefs: Vec<f64> = if weights.len() == 1 {
            weights[0].clone()
        } else {
            // Mean absolute weight across classes.
            (0..n_features)
                .map(|j| {
                    let sum: f64 = weights.iter().map(|w| w[j].abs()).sum();
                    sum / weights.len() as f64
                })
                .collect()
        };

        let mut pairs: Vec<(String, f64)> = names
            .into_iter()
            .zip(coefs)
            .collect();
        pairs.sort_by(|a, b| {
            b.1.abs()
                .partial_cmp(&a.1.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let labels: Vec<String> = pairs.iter().map(|(n, _)| n.clone()).collect();
        let values: Vec<f64> = pairs.iter().map(|(_, v)| *v).collect();

        BarChart::new(labels, vec![Series::new("Coefficient", values)])
            .horizontal()
            .title("Logistic Regression Coefficients")
            .show_values()
            .theme(ml_theme())
            .build()
    }
}

// ---------------------------------------------------------------------------
// Clustering — cluster scatter chart
// ---------------------------------------------------------------------------

use crate::cluster::{KMeans, MiniBatchKMeans};

/// Internal trait for models that expose cluster labels.
trait HasClusterLabels {
    fn label_slice(&self) -> &[usize];
}

impl HasClusterLabels for KMeans {
    fn label_slice(&self) -> &[usize] { self.labels() }
}

impl HasClusterLabels for MiniBatchKMeans {
    fn label_slice(&self) -> &[usize] { self.labels() }
}

macro_rules! impl_cluster_viz {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl Visualize for $ty {}

            impl ModelViz<'_, $ty> {
                /// 2D scatter plot colored by cluster assignment.
                ///
                /// `feat_x` and `feat_y` are the two feature columns to plot.
                pub fn cluster_scatter(&self, feat_x: &[f64], feat_y: &[f64]) -> Chart {
                    let labels = self.model.label_slice();
                    super::cluster_scatter(feat_x, feat_y, labels)
                }
            }
        )+
    };
}

impl_cluster_viz!(KMeans, MiniBatchKMeans);

// ---------------------------------------------------------------------------
// PCA — scree plot
// ---------------------------------------------------------------------------

use crate::preprocess::Pca;

impl Visualize for Pca {}

impl ModelViz<'_, Pca> {
    /// Scree plot showing explained variance ratio per component.
    pub fn scree_plot(&self) -> Chart {
        let ratios = self.model.explained_variance_ratio();
        let n = ratios.len();
        let x: Vec<f64> = (1..=n).map(|i| i as f64).collect();

        // Cumulative variance.
        let mut cumulative = Vec::with_capacity(n);
        let mut acc = 0.0;
        for &r in ratios {
            acc += r;
            cumulative.push(acc);
        }

        LineChart::new(vec![
            Series::new("Individual", ratios.to_vec()),
            Series::new("Cumulative", cumulative),
        ])
        .x_values(x)
        .title("PCA Scree Plot")
        .x_label("Component")
        .y_label("Explained Variance Ratio")
        .y_range(0.0, 1.05)
        .with_points()
        .theme(ml_theme())
        .build()
    }
}

// ---------------------------------------------------------------------------
// GaussianNb — density curves
// ---------------------------------------------------------------------------

use crate::naive_bayes::GaussianNb;

impl Visualize for GaussianNb {}

impl ModelViz<'_, GaussianNb> {
    /// Per-class Gaussian density curves for a single feature.
    ///
    /// Plots the fitted normal distribution for each class on the given
    /// feature index, enabling visual inspection of class separability.
    pub fn density_curves(&self, feature_idx: usize) -> Result<Chart> {
        let means = self.model.class_means();
        let variances = self.model.class_variances();
        let n_classes = means.len();

        if n_classes == 0 {
            return Err(ScryLearnError::NotFitted);
        }
        if feature_idx >= means[0].len() {
            return Err(ScryLearnError::InvalidFeatureIndex(feature_idx));
        }

        // Determine x-range from the fitted distributions (mean +/- 4*sigma).
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for c in 0..n_classes {
            let mu = means[c][feature_idx];
            let sigma = variances[c][feature_idx].sqrt();
            lo = lo.min(mu - 4.0 * sigma);
            hi = hi.max(mu + 4.0 * sigma);
        }

        let n_points = 200;
        let step = (hi - lo) / (n_points - 1) as f64;
        let x_vals: Vec<f64> = (0..n_points).map(|i| lo + i as f64 * step).collect();

        let mut series = Vec::with_capacity(n_classes);
        for c in 0..n_classes {
            let mu = means[c][feature_idx];
            let var = variances[c][feature_idx];
            let sigma = var.sqrt();
            let norm_const = 1.0 / (sigma * std::f64::consts::TAU.sqrt());

            let y_vals: Vec<f64> = x_vals
                .iter()
                .map(|&x| norm_const * (-0.5 * ((x - mu) / sigma).powi(2)).exp())
                .collect();
            series.push(Series::new(format!("Class {c}"), y_vals));
        }

        let feat_name = self
            .feature_names
            .as_ref()
            .and_then(|n| n.get(feature_idx))
            .map_or_else(
                || format!("feature_{feature_idx}"),
                std::clone::Clone::clone,
            );

        Ok(LineChart::new(series)
            .x_values(x_vals)
            .title(format!("Gaussian Density — {feat_name}"))
            .x_label(&feat_name)
            .y_label("Density")
            .filled()
            .theme(ml_theme())
            .build())
    }
}

// ---------------------------------------------------------------------------
// Cross-cutting: Classifier confusion matrix
// ---------------------------------------------------------------------------

/// Internal trait for models with a classify-style `predict`.
trait ClassifierPredict {
    fn predict_classes(&self, features: &[Vec<f64>]) -> Result<Vec<f64>>;
}

macro_rules! impl_classifier_predict {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl ClassifierPredict for $ty {
                fn predict_classes(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> {
                    self.predict(features)
                }
            }
        )+
    };
}

impl_classifier_predict!(
    DecisionTreeClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    LogisticRegression,
    GaussianNb,
);

use crate::neighbors::KnnClassifier;

impl Visualize for KnnClassifier {}
impl ClassifierPredict for KnnClassifier {
    fn predict_classes(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> {
        self.predict(features)
    }
}

/// Generate a confusion matrix chart for any classifier.
fn classifier_confusion_matrix<M: ClassifierPredict>(
    model: &M,
    test_x: &[Vec<f64>],
    test_y: &[f64],
) -> Result<Chart> {
    let preds = model.predict_classes(test_x)?;
    let cm = crate::metrics::confusion_matrix(test_y, &preds);
    Ok(super::confusion_matrix_chart(&cm, false))
}

macro_rules! impl_classifier_viz {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl ModelViz<'_, $ty> {
                /// Confusion matrix heatmap.
                pub fn confusion_matrix(
                    &self,
                    test_x: &[Vec<f64>],
                    test_y: &[f64],
                ) -> Result<Chart> {
                    classifier_confusion_matrix(self.model, test_x, test_y)
                }
            }
        )+
    };
}

impl_classifier_viz!(
    DecisionTreeClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    LogisticRegression,
    GaussianNb,
    KnnClassifier,
);

// ---------------------------------------------------------------------------
// Cross-cutting: Regressor residual + prediction error
// ---------------------------------------------------------------------------

/// Internal trait for models with a regression-style `predict`.
trait RegressorPredict {
    fn predict_values(&self, features: &[Vec<f64>]) -> Result<Vec<f64>>;
}

macro_rules! impl_regressor_predict {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl RegressorPredict for $ty {
                fn predict_values(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> {
                    self.predict(features)
                }
            }
        )+
    };
}

impl_regressor_predict!(
    DecisionTreeRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    LinearRegression,
    LassoRegression,
    ElasticNet,
    Ridge,
);

use crate::neighbors::KnnRegressor;

impl Visualize for KnnRegressor {}
impl RegressorPredict for KnnRegressor {
    fn predict_values(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> {
        self.predict(features)
    }
}

/// Generate a residual plot for any regressor.
fn regressor_residual_plot<M: RegressorPredict>(
    model: &M,
    test_x: &[Vec<f64>],
    test_y: &[f64],
) -> Result<Chart> {
    let preds = model.predict_values(test_x)?;
    Ok(super::residual_plot(test_y, &preds))
}

/// Generate a prediction error chart for any regressor.
fn regressor_prediction_error<M: RegressorPredict>(
    model: &M,
    test_x: &[Vec<f64>],
    test_y: &[f64],
) -> Result<Chart> {
    let preds = model.predict_values(test_x)?;
    Ok(super::prediction_error_chart(test_y, &preds))
}

macro_rules! impl_regressor_viz {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl ModelViz<'_, $ty> {
                /// Residual plot (residuals vs fitted values).
                pub fn residual_plot(
                    &self,
                    test_x: &[Vec<f64>],
                    test_y: &[f64],
                ) -> Result<Chart> {
                    regressor_residual_plot(self.model, test_x, test_y)
                }

                /// Prediction error chart (predicted vs actual).
                pub fn prediction_error(
                    &self,
                    test_x: &[Vec<f64>],
                    test_y: &[f64],
                ) -> Result<Chart> {
                    regressor_prediction_error(self.model, test_x, test_y)
                }
            }
        )+
    };
}

impl_regressor_viz!(
    DecisionTreeRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    LinearRegression,
    LassoRegression,
    ElasticNet,
    Ridge,
    KnnRegressor,
);

// ---------------------------------------------------------------------------
// MLP Neural Networks — learning curve + weight heatmap
// ---------------------------------------------------------------------------

use crate::neural::{MLPClassifier, MLPRegressor};

impl Visualize for MLPClassifier {}

impl ModelViz<'_, MLPClassifier> {
    /// Learning curve chart showing training loss per epoch.
    pub fn learning_curve(&self) -> Result<Chart> {
        let curve = self.model.loss_curve();
        if curve.is_empty() {
            return Err(ScryLearnError::NotFitted);
        }
        let x: Vec<f64> = (1..=curve.len()).map(|i| i as f64).collect();
        Ok(LineChart::new(vec![
            Series::new("Training Loss", curve.to_vec()),
        ])
        .x_values(x)
        .title("MLP Learning Curve")
        .x_label("Epoch")
        .y_label("Loss")
        .with_points()
        .theme(ml_theme())
        .build())
    }

    /// Confusion matrix heatmap.
    pub fn confusion_matrix(
        &self,
        test_x: &[Vec<f64>],
        test_y: &[f64],
    ) -> Result<Chart> {
        classifier_confusion_matrix(self.model, test_x, test_y)
    }

    /// Weight heatmap for each layer.
    ///
    /// Returns one heatmap per layer showing the weight matrix.
    /// For networks with many layers, pass `layer_idx` to select one.
    pub fn weight_heatmap(&self, layer_idx: Option<usize>) -> Result<Chart> {
        weight_heatmap_impl(
            self.model.weights(),
            self.model.layer_dims(),
            layer_idx,
        )
    }
}

impl ClassifierPredict for MLPClassifier {
    fn predict_classes(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> {
        self.predict(features)
    }
}

impl Visualize for MLPRegressor {}

impl ModelViz<'_, MLPRegressor> {
    /// Learning curve chart showing training loss per epoch.
    pub fn learning_curve(&self) -> Result<Chart> {
        let curve = self.model.loss_curve();
        if curve.is_empty() {
            return Err(ScryLearnError::NotFitted);
        }
        let x: Vec<f64> = (1..=curve.len()).map(|i| i as f64).collect();
        Ok(LineChart::new(vec![
            Series::new("Training Loss (MSE)", curve.to_vec()),
        ])
        .x_values(x)
        .title("MLP Learning Curve")
        .x_label("Epoch")
        .y_label("MSE")
        .with_points()
        .theme(ml_theme())
        .build())
    }

    /// Residual plot (residuals vs fitted values).
    pub fn residual_plot(
        &self,
        test_x: &[Vec<f64>],
        test_y: &[f64],
    ) -> Result<Chart> {
        regressor_residual_plot(self.model, test_x, test_y)
    }

    /// Prediction error chart (predicted vs actual).
    pub fn prediction_error(
        &self,
        test_x: &[Vec<f64>],
        test_y: &[f64],
    ) -> Result<Chart> {
        regressor_prediction_error(self.model, test_x, test_y)
    }

    /// Weight heatmap for each layer.
    ///
    /// Returns one heatmap per layer showing the weight matrix.
    /// For networks with many layers, pass `layer_idx` to select one.
    pub fn weight_heatmap(&self, layer_idx: Option<usize>) -> Result<Chart> {
        weight_heatmap_impl(
            self.model.weights(),
            self.model.layer_dims(),
            layer_idx,
        )
    }
}

impl RegressorPredict for MLPRegressor {
    fn predict_values(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> {
        self.predict(features)
    }
}

// ---------------------------------------------------------------------------
// MLP weight heatmap shared helper
// ---------------------------------------------------------------------------

/// Render a weight heatmap for a single layer of a fitted MLP.
///
/// `weights` and `dims` come from `model.weights()` and `model.layer_dims()`.
/// If `layer_idx` is `None`, defaults to the first hidden layer (index 0).
fn weight_heatmap_impl(
    weights: &[(Vec<f64>, Vec<f64>)],
    dims: &[(usize, usize)],
    layer_idx: Option<usize>,
) -> Result<Chart> {
    if weights.is_empty() {
        return Err(ScryLearnError::NotFitted);
    }
    let idx = layer_idx.unwrap_or(0);
    if idx >= weights.len() {
        return Err(ScryLearnError::InvalidParameter(
            format!("layer index {idx} out of range (0..{})", weights.len()),
        ));
    }

    let (ref w, _) = weights[idx];
    let (in_size, out_size) = dims[idx];

    // w is row-major [out_size, in_size] — each row is one output neuron.
    // Build the 2D grid for the heatmap.
    let mut grid: Vec<Vec<f64>> = Vec::with_capacity(out_size);
    for o in 0..out_size {
        let row = w[o * in_size..(o + 1) * in_size].to_vec();
        grid.push(row);
    }

    // Row labels = output neurons, col labels = input neurons
    let row_labels: Vec<String> = (0..out_size).map(|i| format!("out_{i}")).collect();
    let col_labels: Vec<String> = (0..in_size).map(|i| format!("in_{i}")).collect();

    // Symmetric range around zero for diverging colormap
    let abs_max = w.iter().map(|v| v.abs()).fold(0.0f64, f64::max);

    Ok(Heatmap::new(grid)
        .row_labels(row_labels)
        .col_labels(col_labels)
        .range(-abs_max, abs_max)
        .title(&format!("Layer {} Weights ({in_size} -> {out_size})", idx + 1))
        .theme(ml_theme())
        .build())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::Dataset;
    use crate::preprocess::Transformer;

    fn iris_like_dataset() -> Dataset {
        // 4 features, 3 classes, 12 samples (balanced).
        let features = vec![
            vec![5.1, 4.9, 4.7, 7.0, 6.5, 6.3, 6.3, 5.8, 7.1, 5.0, 5.2, 4.8],
            vec![3.5, 3.0, 3.2, 3.2, 2.8, 3.3, 2.7, 2.7, 3.0, 3.4, 3.1, 3.0],
            vec![1.4, 1.4, 1.3, 4.7, 4.6, 4.7, 4.9, 5.1, 5.9, 1.5, 1.4, 1.3],
            vec![0.2, 0.2, 0.2, 1.4, 1.5, 1.6, 1.8, 1.9, 2.1, 0.3, 0.2, 0.2],
        ];
        let target = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0];
        Dataset::new(
            features,
            target,
            vec!["sl".into(), "sw".into(), "pl".into(), "pw".into()],
            "species",
        )
    }

    fn regression_dataset() -> Dataset {
        let features = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        ];
        let target = vec![2.1, 4.0, 5.9, 8.1, 9.8, 12.0, 14.1, 16.0];
        Dataset::new(features, target, vec!["x".into()], "y")
    }

    // ── Tree models ──

    #[test]
    fn test_decision_tree_feature_importance() {
        let data = iris_like_dataset();
        let mut dt = DecisionTreeClassifier::new();
        dt.fit(&data).unwrap();

        let chart = dt.viz().feature_importance(Some(2)).unwrap();
        assert!(matches!(chart, Chart::Bar(_)));
    }

    #[test]
    fn test_random_forest_feature_importance() {
        let data = iris_like_dataset();
        let mut rf = RandomForestClassifier::new().n_estimators(5).seed(42);
        rf.fit(&data).unwrap();

        let chart = rf.viz().feature_importance(None).unwrap();
        assert!(matches!(chart, Chart::Bar(_)));
    }

    // ── Linear models ──

    #[test]
    fn test_linear_regression_coefficient_chart() {
        let data = regression_dataset();
        let mut lr = LinearRegression::new();
        lr.fit(&data).unwrap();

        let chart = lr.viz().coefficient_chart();
        assert!(matches!(chart, Chart::Bar(_)));
    }

    #[test]
    fn test_logistic_regression_coefficient_chart() {
        let data = iris_like_dataset();
        let mut lr = LogisticRegression::new();
        lr.fit(&data).unwrap();

        let chart = lr.viz().coefficient_chart();
        assert!(matches!(chart, Chart::Bar(_)));
    }

    // ── Clustering ──

    #[test]
    fn test_kmeans_cluster_scatter() {
        let data = iris_like_dataset();
        let mut km = KMeans::new(3).seed(42);
        km.fit(&data).unwrap();

        let x = vec![5.1, 4.9, 4.7, 7.0, 6.5, 6.3, 6.3, 5.8, 7.1, 5.0, 5.2, 4.8];
        let y = vec![3.5, 3.0, 3.2, 3.2, 2.8, 3.3, 2.7, 2.7, 3.0, 3.4, 3.1, 3.0];
        let chart = km.viz().cluster_scatter(&x, &y);
        assert!(matches!(chart, Chart::Scatter(_)));
    }

    // ── PCA ──

    #[test]
    fn test_pca_scree_plot() {
        let mut data = iris_like_dataset();
        let mut pca = Pca::new();
        pca.fit_transform(&mut data).unwrap();

        let chart = pca.viz().scree_plot();
        assert!(matches!(chart, Chart::Line(_)));
    }

    // ── GaussianNb ──

    #[test]
    fn test_gaussian_nb_density_curves() {
        let data = iris_like_dataset();
        let mut nb = GaussianNb::new();
        nb.fit(&data).unwrap();

        let chart = nb.viz().density_curves(0).unwrap();
        assert!(matches!(chart, Chart::Line(_)));
    }

    // ── Cross-cutting: classifier ──

    #[test]
    fn test_classifier_confusion_matrix() {
        let data = iris_like_dataset();
        let mut dt = DecisionTreeClassifier::new();
        dt.fit(&data).unwrap();

        let test_x = vec![vec![5.1, 3.5, 1.4, 0.2], vec![7.0, 3.2, 4.7, 1.4]];
        let test_y = vec![0.0, 1.0];
        let chart = dt.viz().confusion_matrix(&test_x, &test_y).unwrap();
        assert!(matches!(chart, Chart::Heatmap(_)));
    }

    // ── Cross-cutting: regressor ──

    #[test]
    fn test_regressor_residual_plot() {
        let data = regression_dataset();
        let mut lr = LinearRegression::new();
        lr.fit(&data).unwrap();

        let test_x = vec![vec![2.0], vec![5.0], vec![7.0]];
        let test_y = vec![4.0, 10.0, 14.0];
        let chart = lr.viz().residual_plot(&test_x, &test_y).unwrap();
        assert!(matches!(chart, Chart::Scatter(_)));
    }

    #[test]
    fn test_regressor_prediction_error() {
        let data = regression_dataset();
        let mut lr = LinearRegression::new();
        lr.fit(&data).unwrap();

        let test_x = vec![vec![2.0], vec![5.0], vec![7.0]];
        let test_y = vec![4.0, 10.0, 14.0];
        let chart = lr.viz().prediction_error(&test_x, &test_y).unwrap();
        assert!(matches!(chart, Chart::Scatter(_)));
    }

    // ── Feature names override ──

    #[test]
    fn test_feature_names_override() {
        let data = iris_like_dataset();
        let mut dt = DecisionTreeClassifier::new();
        dt.fit(&data).unwrap();

        let chart = dt
            .viz()
            .feature_names(vec!["A".into(), "B".into(), "C".into(), "D".into()])
            .feature_importance(None)
            .unwrap();
        assert!(matches!(chart, Chart::Bar(_)));
    }
}
