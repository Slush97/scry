// SPDX-License-Identifier: MIT OR Apache-2.0
//! Visualization bridge: convert ML outputs into scry-chart charts.
//!
//! This is the unique selling point of scry-learn — every model evaluation
//! step can produce a publication-quality chart automatically.
//!
//! All charts default to [`Theme::dark()`] for consistent terminal aesthetics.

pub mod model_viz;

use crate::metrics::{ClassificationReport, ConfusionMatrix, PrCurve, RocCurve};
use scry_chart::chart::{BarChart, BoxPlot, Chart, Heatmap, LineChart, ScatterChart};
use scry_chart::data::Series;
use scry_chart::theme::Theme;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Standard ML-viz dark theme.
fn ml_theme() -> Theme {
    Theme::dark()
}

// ---------------------------------------------------------------------------
// Phase 1 — Audited existing functions
// ---------------------------------------------------------------------------

/// Render a confusion matrix as a scry heatmap chart.
///
/// When `normalize` is `true`, each row is divided by its sum to show
/// proportions (0.0–1.0) instead of raw counts.
pub fn confusion_matrix_chart(cm: &ConfusionMatrix, normalize: bool) -> Chart {
    let data: Vec<Vec<f64>> = if normalize {
        cm.matrix
            .iter()
            .map(|row| {
                let sum: usize = row.iter().sum();
                if sum == 0 {
                    row.iter().map(|_| 0.0).collect()
                } else {
                    row.iter().map(|&v| v as f64 / sum as f64).collect()
                }
            })
            .collect()
    } else {
        cm.matrix
            .iter()
            .map(|row| row.iter().map(|&v| v as f64).collect())
            .collect()
    };

    let max_val = if normalize {
        1.0
    } else {
        cm.matrix.iter().flatten().max().copied().unwrap_or(1) as f64
    };

    Heatmap::new(data)
        .row_labels(cm.labels.clone())
        .col_labels(cm.labels.clone())
        .values(true)
        .title("Confusion Matrix")
        .range(0.0, max_val)
        .theme(ml_theme())
        .build()
}

/// Render ROC curve(s) as a scry line chart with AUC annotation.
///
/// Each tuple is `(model_name, roc_curve)`.
pub fn roc_chart(curves: &[(&str, &RocCurve)]) -> Chart {
    let mut series = Vec::new();

    for (name, curve) in curves {
        let label = format!("{} (AUC = {:.3})", name, curve.auc);
        series.push(Series::new(&label, curve.tpr.clone()));
    }

    // Use FPR of first curve as shared x values.
    let x_vals = curves.first().map(|(_, c)| c.fpr.clone());

    // Random baseline — diagonal y=x. Must match x_values length to avoid
    // series/x mismatch in LineChart.
    if let Some(ref x) = x_vals {
        series.push(Series::new("Random", x.clone()));
    } else {
        series.push(Series::new("Random", vec![0.0, 1.0]));
    }

    let mut chart = LineChart::new(series)
        .title("ROC Curve")
        .x_label("False Positive Rate")
        .y_label("True Positive Rate")
        .x_range(0.0, 1.0)
        .y_range(0.0, 1.05)
        .dash_lines()
        .theme(ml_theme());

    if let Some(x) = x_vals {
        chart = chart.x_values(x);
    }

    chart.build()
}

/// Render Precision-Recall curve(s) as a scry line chart with AP annotation.
///
/// Each tuple is `(model_name, pr_curve)`.
pub fn pr_chart(curves: &[(&str, &PrCurve)]) -> Chart {
    let mut series = Vec::new();

    for (name, curve) in curves {
        let label = format!("{} (AP = {:.3})", name, curve.avg_precision);
        series.push(Series::new(&label, curve.precision.clone()));
    }

    let mut chart = LineChart::new(series)
        .title("Precision-Recall Curve")
        .x_label("Recall")
        .y_label("Precision")
        .x_range(0.0, 1.0)
        .y_range(0.0, 1.05)
        .dash_lines()
        .theme(ml_theme());

    // Use recall of first curve as x values.
    if let Some((_, first_curve)) = curves.first() {
        chart = chart.x_values(first_curve.recall.clone());
    }

    chart.build()
}

/// Render feature importances as a horizontal bar chart.
///
/// `top_n` limits the number of bars shown (default: all features).
/// Features are sorted descending by importance.
pub fn feature_importance_chart(
    names: &[String],
    importances: &[f64],
    top_n: Option<usize>,
) -> Chart {
    // Sort by importance descending.
    let mut pairs: Vec<(String, f64)> = names
        .iter()
        .cloned()
        .zip(importances.iter().copied())
        .collect();
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Apply top_n cap.
    let cap = top_n.unwrap_or(pairs.len()).min(pairs.len());
    let pairs = &pairs[..cap];

    let labels: Vec<String> = pairs.iter().map(|(n, _)| n.clone()).collect();
    let values: Vec<f64> = pairs.iter().map(|(_, v)| *v).collect();

    BarChart::new(labels, vec![Series::new("Importance", values)])
        .horizontal()
        .title("Feature Importances")
        .show_values()
        .theme(ml_theme())
        .build()
}

/// Render a residual plot (residuals vs fitted values) as a scatter chart.
///
/// Includes a zero reference line to highlight bias.
pub fn residual_plot(y_true: &[f64], y_pred: &[f64]) -> Chart {
    let residuals: Vec<f64> = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| t - p)
        .collect();

    ScatterChart::new(
        Series::new("fitted", y_pred.to_vec()),
        Series::new("Residuals", residuals),
    )
    .title("Residuals vs Fitted")
    .x_label("Fitted Values")
    .y_label("Residuals")
    .h_line(0.0)
    .theme(ml_theme())
    .build()
}

/// Render a regularization path plot showing coefficient magnitudes vs lambda.
pub fn regularization_path_chart(
    lambdas: &[f64],
    coefficients: &[Vec<f64>],
    feature_names: &[String],
) -> Chart {
    let n_features = feature_names.len();
    let mut series = Vec::with_capacity(n_features);

    for (feat_idx, name) in feature_names.iter().enumerate() {
        let values: Vec<f64> = coefficients
            .iter()
            .map(|coefs| coefs.get(feat_idx).copied().unwrap_or(0.0))
            .collect();
        series.push(Series::new(name, values));
    }

    // X-axis: log10 of lambdas.
    let x: Vec<f64> = lambdas.iter().map(|&l| l.log10()).collect();

    LineChart::new(series)
        .x_values(x)
        .title("Regularization Path")
        .x_label("log₁₀(λ)")
        .y_label("Coefficient Value")
        .dash_lines()
        .theme(ml_theme())
        .build()
}

// ---------------------------------------------------------------------------
// Phase 2 — Classification & Regression charts
// ---------------------------------------------------------------------------

/// Render a learning curve: training vs validation score over dataset size.
///
/// `train_sizes` is the x-axis (number of samples).
/// `train_scores` and `val_scores` are mean metric values at each size.
pub fn learning_curve(train_sizes: &[f64], train_scores: &[f64], val_scores: &[f64]) -> Chart {
    LineChart::new(vec![
        Series::new("Training", train_scores.to_vec()),
        Series::new("Validation", val_scores.to_vec()),
    ])
    .x_values(train_sizes.to_vec())
    .title("Learning Curve")
    .x_label("Training Set Size")
    .y_label("Score")
    .with_points()
    .filled()
    .theme(ml_theme())
    .build()
}

/// Render predicted vs actual scatter plot (regression quality diagnosis).
///
/// The 45° ideal line shows where perfect predictions would fall.
pub fn prediction_error_chart(y_true: &[f64], y_pred: &[f64]) -> Chart {
    // Compute range for the ideal line.
    let all_vals = y_true.iter().chain(y_pred.iter()).copied();
    let lo = all_vals.clone().fold(f64::INFINITY, f64::min);
    let hi = all_vals.fold(f64::NEG_INFINITY, f64::max);
    let margin = (hi - lo) * 0.05;

    ScatterChart::new(
        Series::new("actual", y_true.to_vec()),
        Series::new("Predicted vs Actual", y_pred.to_vec()),
    )
    // Add ideal diagonal as a connected second series.
    .add_named_series(
        "Ideal",
        &[lo - margin, hi + margin],
        &[lo - margin, hi + margin],
    )
    .connected()
    .title("Prediction Error")
    .x_label("Actual")
    .y_label("Predicted")
    .theme(ml_theme())
    .build()
}

/// Render a calibration (reliability) diagram.
///
/// Each tuple is `(model_name, mean_predicted_prob, fraction_of_positives)`.
/// A perfectly calibrated model follows the diagonal.
pub fn calibration_chart(curves: &[(&str, &[f64], &[f64])]) -> Chart {
    let mut series = Vec::new();
    let mut x_vals = None;

    for (name, mean_pred, fraction_pos) in curves {
        series.push(Series::new(*name, fraction_pos.to_vec()));
        if x_vals.is_none() {
            x_vals = Some(mean_pred.to_vec());
        }
    }

    // Ideal calibration diagonal y=x. Must match x_values length to avoid
    // series/x mismatch in LineChart.
    if let Some(ref x) = x_vals {
        series.push(Series::new("Perfectly calibrated", x.clone()));
    } else {
        series.push(Series::new("Perfectly calibrated", vec![0.0, 1.0]));
    }

    let mut chart = LineChart::new(series)
        .title("Calibration Curve")
        .x_label("Mean Predicted Probability")
        .y_label("Fraction of Positives")
        .x_range(0.0, 1.0)
        .y_range(0.0, 1.05)
        .with_points()
        .dash_lines()
        .theme(ml_theme());

    if let Some(x) = x_vals {
        chart = chart.x_values(x);
    }

    chart.build()
}

/// Render a classification report as an annotated heatmap.
///
/// Rows = classes, columns = precision / recall / F1.
pub fn class_report_chart(report: &ClassificationReport) -> Chart {
    let n = report.per_class.len();
    let mut data = Vec::with_capacity(n);
    let mut row_labels = Vec::with_capacity(n);

    for (label, m) in &report.per_class {
        row_labels.push(label.clone());
        data.push(vec![m.precision, m.recall, m.f1]);
    }

    let col_labels = vec![
        "Precision".to_string(),
        "Recall".to_string(),
        "F1-Score".to_string(),
    ];

    Heatmap::new(data)
        .row_labels(row_labels)
        .col_labels(col_labels)
        .values(true)
        .range(0.0, 1.0)
        .title("Classification Report")
        .theme(ml_theme())
        .build()
}

/// Render a grouped bar chart comparing models across multiple metrics.
///
/// `model_names` are the category labels. Each `(metric_name, values)` tuple
/// adds a grouped series.
pub fn metric_comparison_chart(model_names: &[String], metrics: &[(&str, &[f64])]) -> Chart {
    let mut series = Vec::new();
    for (metric_name, values) in metrics {
        series.push(Series::new(*metric_name, values.to_vec()));
    }

    BarChart::new(model_names.to_vec(), series)
        .title("Model Comparison")
        .y_label("Score")
        .show_values()
        .theme(ml_theme())
        .build()
}

// ---------------------------------------------------------------------------
// Phase 3 — Unsupervised & Tree charts
// ---------------------------------------------------------------------------

/// Render the K-Means elbow chart (inertia vs number of clusters).
///
/// Optionally annotates the suggested optimal k with a vertical reference line.
pub fn elbow_chart(ks: &[usize], inertias: &[f64], optimal_k: Option<usize>) -> Chart {
    let x: Vec<f64> = ks.iter().map(|&k| k as f64).collect();

    let mut chart = LineChart::new(vec![Series::new("Inertia", inertias.to_vec())])
        .x_values(x)
        .title("Elbow Method")
        .x_label("Number of Clusters (k)")
        .y_label("Inertia")
        .with_points()
        .theme(ml_theme());

    if let Some(k) = optimal_k {
        chart = chart.v_line(k as f64);
    }

    chart.build()
}

/// Render a 2D scatter plot colored by cluster assignment.
///
/// `labels` contains the integer cluster label for each point.
pub fn cluster_scatter(x: &[f64], y: &[f64], labels: &[usize]) -> Chart {
    let n = x.len().min(y.len()).min(labels.len());

    // Empty input — return a minimal scatter with a placeholder point to
    // avoid passing empty series to ScatterChart.
    if n == 0 {
        return ScatterChart::new(
            Series::new("Cluster 0 x", vec![0.0]),
            Series::new("Cluster 0", vec![0.0]),
        )
        .title("Cluster Assignments (no data)")
        .theme(ml_theme())
        .build();
    }

    // Group points by cluster.
    let max_label = labels[..n].iter().max().copied().unwrap_or(0);
    let mut cluster_x: Vec<Vec<f64>> = vec![Vec::new(); max_label + 1];
    let mut cluster_y: Vec<Vec<f64>> = vec![Vec::new(); max_label + 1];

    for i in 0..n {
        cluster_x[labels[i]].push(x[i]);
        cluster_y[labels[i]].push(y[i]);
    }

    // First cluster as primary series.
    let mut chart = ScatterChart::new(
        Series::new("Cluster 0 x", cluster_x[0].clone()),
        Series::new("Cluster 0", cluster_y[0].clone()),
    );

    // Remaining clusters as extra series.
    for k in 1..=max_label {
        if !cluster_x[k].is_empty() {
            chart = chart.add_named_series(format!("Cluster {k}"), &cluster_x[k], &cluster_y[k]);
        }
    }

    chart
        .title("Cluster Assignments")
        .x_label("Component 1")
        .y_label("Component 2")
        .theme(ml_theme())
        .build()
}

/// Render a silhouette plot — horizontal bars grouped and sorted by cluster.
///
/// `labels` is the cluster assignment for each sample.
/// `scores` is the silhouette coefficient for each sample.
pub fn silhouette_chart(labels: &[usize], scores: &[f64]) -> Chart {
    // Sort samples by (cluster, score descending) and flatten into a bar chart.
    let mut samples: Vec<(usize, f64)> =
        labels.iter().copied().zip(scores.iter().copied()).collect();
    samples.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then(b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal))
    });

    let bar_labels: Vec<String> = (0..samples.len()).map(|i| i.to_string()).collect();
    let values: Vec<f64> = samples.iter().map(|(_, s)| *s).collect();

    BarChart::new(bar_labels, vec![Series::new("Silhouette", values)])
        .horizontal()
        .title("Silhouette Plot")
        .x_label("Silhouette Coefficient")
        .h_line(0.0)
        .gap(0.0)
        .theme(ml_theme())
        .build()
}

// ---------------------------------------------------------------------------
// Phase 4 — Model evaluation & interpretation
// ---------------------------------------------------------------------------

/// Render a validation curve: training vs validation score over a
/// hyperparameter range.
///
/// `param_range` is the x-axis (hyperparameter values).
/// `train_scores` and `val_scores` are mean metric values at each setting.
pub fn validation_curve(
    param_name: &str,
    param_range: &[f64],
    train_scores: &[f64],
    val_scores: &[f64],
) -> Chart {
    LineChart::new(vec![
        Series::new("Training", train_scores.to_vec()),
        Series::new("Validation", val_scores.to_vec()),
    ])
    .x_values(param_range.to_vec())
    .title(format!("Validation Curve ({param_name})"))
    .x_label(param_name)
    .y_label("Score")
    .with_points()
    .dash_lines()
    .theme(ml_theme())
    .build()
}

/// Render a partial dependence plot for a single feature.
///
/// `feature_values` is the grid of values the feature was varied over.
/// `pdp_values` is the average model prediction at each grid value.
pub fn partial_dependence_chart(
    feature_values: &[f64],
    pdp_values: &[f64],
    feature_name: &str,
) -> Chart {
    LineChart::new(vec![Series::new("PDP", pdp_values.to_vec())])
        .x_values(feature_values.to_vec())
        .title(format!("Partial Dependence: {feature_name}"))
        .x_label(feature_name)
        .y_label("Partial Dependence")
        .with_points()
        .filled()
        .theme(ml_theme())
        .build()
}

/// Render a box plot comparing cross-validation score distributions across
/// models.
///
/// `model_names` labels each box. `cv_scores` holds the per-fold scores for
/// each model (one `Vec<f64>` per model).
pub fn cv_boxplot(model_names: &[String], cv_scores: &[Vec<f64>]) -> Chart {
    let groups: Vec<(String, Vec<f64>)> = model_names
        .iter()
        .zip(cv_scores.iter())
        .map(|(name, scores)| (name.clone(), scores.clone()))
        .collect();

    BoxPlot::new(groups)
        .title("Cross-Validation Scores")
        .y_label("Score")
        .theme(ml_theme())
        .build()
}

/// Render a 2D decision boundary chart.
///
/// Generates a mesh grid over the feature space, classifies each grid point
/// via `predict_fn`, and renders the background as a colored scatter along
/// with the original data points.
///
/// * `x`, `y` — the two feature columns (same length).
/// * `labels` — integer class label for each sample.
/// * `predict_fn` — closure that takes `(x, y)` and returns a predicted class.
/// * `resolution` — number of grid steps along each axis.
pub fn decision_boundary_chart(
    x: &[f64],
    y: &[f64],
    labels: &[usize],
    predict_fn: &dyn Fn(f64, f64) -> usize,
    resolution: usize,
) -> Chart {
    // Compute feature-space bounds with a small margin.
    let x_min = x.iter().copied().fold(f64::INFINITY, f64::min);
    let x_max = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let y_min = y.iter().copied().fold(f64::INFINITY, f64::min);
    let y_max = y.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let x_margin = (x_max - x_min) * 0.05;
    let y_margin = (y_max - y_min) * 0.05;
    let x_lo = x_min - x_margin;
    let x_hi = x_max + x_margin;
    let y_lo = y_min - y_margin;
    let y_hi = y_max + y_margin;

    let res = resolution.max(2);
    let x_step = (x_hi - x_lo) / (res - 1) as f64;
    let y_step = (y_hi - y_lo) / (res - 1) as f64;

    // Classify each mesh point and group by predicted class.
    let max_class = labels.iter().max().copied().unwrap_or(0);
    let n_classes = max_class + 1;
    let mut mesh_x: Vec<Vec<f64>> = vec![Vec::new(); n_classes];
    let mut mesh_y: Vec<Vec<f64>> = vec![Vec::new(); n_classes];

    for ix in 0..res {
        for iy in 0..res {
            let gx = x_lo + ix as f64 * x_step;
            let gy = y_lo + iy as f64 * y_step;
            let cls = predict_fn(gx, gy).min(n_classes - 1);
            mesh_x[cls].push(gx);
            mesh_y[cls].push(gy);
        }
    }

    // Also group original data points by true label.
    let mut data_x: Vec<Vec<f64>> = vec![Vec::new(); n_classes];
    let mut data_y: Vec<Vec<f64>> = vec![Vec::new(); n_classes];
    for (i, &lbl) in labels.iter().enumerate() {
        if i < x.len() && i < y.len() {
            let cls = lbl.min(n_classes - 1);
            data_x[cls].push(x[i]);
            data_y[cls].push(y[i]);
        }
    }

    // Build scatter: first series = mesh class 0.
    let mut chart = ScatterChart::new(
        Series::new("Region 0", mesh_x[0].clone()),
        Series::new("Region 0", mesh_y[0].clone()),
    )
    .size(1.5);

    // Add remaining mesh classes.
    for k in 1..n_classes {
        if !mesh_x[k].is_empty() {
            chart = chart.add_named_series(format!("Region {k}"), &mesh_x[k], &mesh_y[k]);
        }
    }

    // Overlay original data as named series.
    for k in 0..n_classes {
        if !data_x[k].is_empty() {
            chart = chart.add_named_series(format!("Class {k}"), &data_x[k], &data_y[k]);
        }
    }

    chart
        .title("Decision Boundary")
        .x_label("Feature 1")
        .y_label("Feature 2")
        .theme(ml_theme())
        .build()
}

// ---------------------------------------------------------------------------
// Phase 5 — 3D scatter visualization (Sprint 8.5C)
// ---------------------------------------------------------------------------

/// Extract 3D data from a dataset for scatter visualization.
///
/// Extracts three feature columns by index and returns coordinates
/// plus the target converted to integer labels (rounded to `usize`).
///
/// # Example
///
/// ```ignore
/// use scry_learn::prelude::*;
/// use scry_learn::viz::scatter3d_data;
///
/// let data = Dataset::from_csv("iris.csv", "species")?;
/// let (x, y, z, labels) = scatter3d_data(&data, 0, 1, 2);
/// ```
pub fn scatter3d_data(
    dataset: &crate::dataset::Dataset,
    feat_x: usize,
    feat_y: usize,
    feat_z: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<usize>) {
    let n = dataset.n_samples();
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);
    let mut labels = Vec::with_capacity(n);

    let x_col = dataset.feature(feat_x);
    let y_col = dataset.feature(feat_y);
    let z_col = dataset.feature(feat_z);

    for i in 0..n {
        x.push(x_col[i]);
        y.push(y_col[i]);
        z.push(z_col[i]);
        // Target is f64; convert to usize for class labels.
        // Guard against NaN/negative/infinite values that would silently
        // corrupt the label via `as usize` saturation.
        let t = dataset.target[i];
        labels.push(if t.is_finite() && t >= 0.0 {
            t as usize
        } else {
            0
        });
    }

    (x, y, z, labels)
}

/// Render a 3D scatter chart colored by class from a dataset.
///
/// Extracts three feature columns by index and colors points by the
/// target column. Feature names are used as axis labels.
///
/// # Example
///
/// ```ignore
/// use scry_learn::prelude::*;
/// use scry_learn::viz::scatter3d_chart;
///
/// let data = Dataset::from_csv("iris.csv", "species")?;
/// let chart = scatter3d_chart(&data, 0, 1, 2);
/// chart.save_png(800, 600, "iris_3d.png")?;
/// ```
pub fn scatter3d_chart(
    dataset: &crate::dataset::Dataset,
    feat_x: usize,
    feat_y: usize,
    feat_z: usize,
) -> scry_chart::chart3d::Chart3D {
    let (x, y, z, labels) = scatter3d_data(dataset, feat_x, feat_y, feat_z);

    let x_name = dataset
        .feature_names
        .get(feat_x)
        .map_or("X", |s| s.as_str());
    let y_name = dataset
        .feature_names
        .get(feat_y)
        .map_or("Y", |s| s.as_str());
    let z_name = dataset
        .feature_names
        .get(feat_z)
        .map_or("Z", |s| s.as_str());

    let title = dataset.class_labels.as_ref().map_or_else(
        || format!("{x_name} × {y_name} × {z_name}"),
        |class_labels| {
            format!(
                "{} × {} × {} (colored by {})",
                x_name,
                y_name,
                z_name,
                class_labels.join("/")
            )
        },
    );

    scry_chart::chart3d::Chart3D::scatter(&x, &y, &z)
        .title(title)
        .x_label(x_name)
        .y_label(y_name)
        .z_label(z_name)
        .color_by_class(&labels)
}

// ---------------------------------------------------------------------------
// Phase 6 — Training visualization (callback-based)
// ---------------------------------------------------------------------------

use crate::neural::callback::TrainingHistory;

/// Render training and validation loss over epochs.
///
/// If the history contains validation losses, both curves are shown.
pub fn training_loss_chart(history: &TrainingHistory) -> Chart {
    let train = history.train_losses();
    let x: Vec<f64> = (1..=train.len()).map(|i| i as f64).collect();

    let val = history.val_losses();
    let mut series = vec![Series::new("Train Loss", train)];
    if val.len() == x.len() {
        series.push(Series::new("Val Loss", val));
    }

    LineChart::new(series)
        .x_values(x)
        .title("Training Loss")
        .x_label("Epoch")
        .y_label("Loss")
        .with_points()
        .dash_lines()
        .theme(ml_theme())
        .build()
}

/// Render training and validation metric (accuracy / R²) over epochs.
///
/// Shows both curves when validation metrics are available.
pub fn training_metric_chart(history: &TrainingHistory) -> Chart {
    let train = history.train_metrics();
    let x: Vec<f64> = (1..=train.len()).map(|i| i as f64).collect();

    let val = history.val_metrics();
    let mut series = vec![Series::new("Train Metric", train)];
    if val.len() == x.len() {
        series.push(Series::new("Val Metric", val));
    }

    LineChart::new(series)
        .x_values(x)
        .title("Training Metric")
        .x_label("Epoch")
        .y_label("Metric")
        .with_points()
        .dash_lines()
        .theme(ml_theme())
        .build()
}

/// Render gradient L2 norm over epochs.
///
/// Helps diagnose vanishing or exploding gradients.
pub fn gradient_norm_chart(history: &TrainingHistory) -> Chart {
    let norms = history.grad_norms();
    let x: Vec<f64> = (1..=norms.len()).map(|i| i as f64).collect();

    LineChart::new(vec![Series::new("Gradient Norm", norms)])
        .x_values(x)
        .title("Gradient Norm per Epoch")
        .x_label("Epoch")
        .y_label("L₂ Norm")
        .with_points()
        .theme(ml_theme())
        .build()
}

/// Render wall-clock epoch time in milliseconds.
///
/// Useful for profiling training throughput and spotting bottlenecks.
pub fn epoch_time_chart(history: &TrainingHistory) -> Chart {
    let times: Vec<f64> = history.epoch_times_ms().iter().map(|&t| t as f64).collect();
    let x: Vec<f64> = (1..=times.len()).map(|i| i as f64).collect();

    LineChart::new(vec![Series::new("Time (ms)", times)])
        .x_values(x)
        .title("Epoch Time")
        .x_label("Epoch")
        .y_label("Time (ms)")
        .with_points()
        .theme(ml_theme())
        .build()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::{classification_report, confusion_matrix, pr_curve, roc_curve};

    #[test]
    fn test_confusion_matrix_chart_raw() {
        let cm = confusion_matrix(&[0.0, 0.0, 1.0, 1.0], &[0.0, 1.0, 0.0, 1.0]);
        let chart = confusion_matrix_chart(&cm, false);
        assert!(matches!(chart, Chart::Heatmap(_)));
    }

    #[test]
    fn test_confusion_matrix_chart_normalized() {
        let cm = confusion_matrix(&[0.0, 0.0, 1.0, 1.0], &[0.0, 1.0, 0.0, 1.0]);
        let chart = confusion_matrix_chart(&cm, true);
        assert!(matches!(chart, Chart::Heatmap(_)));
    }

    #[test]
    fn test_roc_chart() {
        let roc = roc_curve(&[0.0, 0.0, 1.0, 1.0], &[0.1, 0.2, 0.8, 0.9]);
        let chart = roc_chart(&[("Model", &roc)]);
        assert!(matches!(chart, Chart::Line(_)));
    }

    #[test]
    fn test_pr_chart() {
        let pr = pr_curve(&[0.0, 0.0, 1.0, 1.0], &[0.1, 0.2, 0.8, 0.9]);
        let chart = pr_chart(&[("Model", &pr)]);
        assert!(matches!(chart, Chart::Line(_)));
    }

    #[test]
    fn test_feature_importance_full() {
        let names: Vec<String> = (0..30).map(|i| format!("feat_{i}")).collect();
        let importances: Vec<f64> = (0..30).map(|i| i as f64 / 30.0).collect();
        let chart = feature_importance_chart(&names, &importances, None);
        assert!(matches!(chart, Chart::Bar(_)));
    }

    #[test]
    fn test_feature_importance_top_n() {
        let names: Vec<String> = (0..30).map(|i| format!("feat_{i}")).collect();
        let importances: Vec<f64> = (0..30).map(|i| i as f64 / 30.0).collect();
        let chart = feature_importance_chart(&names, &importances, Some(10));
        // Just verify it builds as a bar chart — fields are pub(crate)
        assert!(matches!(chart, Chart::Bar(_)));
    }

    #[test]
    fn test_residual_plot_is_scatter() {
        let y_true = vec![1.0, 2.0, 3.0, 4.0];
        let y_pred = vec![1.1, 1.9, 3.2, 3.8];
        let chart = residual_plot(&y_true, &y_pred);
        assert!(matches!(chart, Chart::Scatter(_)));
    }

    #[test]
    fn test_regularization_path() {
        let lambdas = vec![0.01, 0.1, 1.0, 10.0];
        let coefs = vec![
            vec![1.0, 0.5],
            vec![0.8, 0.4],
            vec![0.3, 0.2],
            vec![0.05, 0.01],
        ];
        let names = vec!["feat_a".to_string(), "feat_b".to_string()];
        let chart = regularization_path_chart(&lambdas, &coefs, &names);
        assert!(matches!(chart, Chart::Line(_)));
    }

    #[test]
    fn test_learning_curve() {
        let sizes = vec![50.0, 100.0, 200.0, 400.0];
        let train = vec![0.95, 0.93, 0.91, 0.90];
        let val = vec![0.70, 0.78, 0.83, 0.86];
        let chart = learning_curve(&sizes, &train, &val);
        assert!(matches!(chart, Chart::Line(_)));
    }

    #[test]
    fn test_prediction_error_chart() {
        let y_true = vec![1.0, 2.0, 3.0, 4.0];
        let y_pred = vec![1.1, 1.9, 3.2, 3.8];
        let chart = prediction_error_chart(&y_true, &y_pred);
        assert!(matches!(chart, Chart::Scatter(_)));
    }

    #[test]
    fn test_calibration_chart() {
        let mean_pred = vec![0.1, 0.3, 0.5, 0.7, 0.9];
        let frac_pos = vec![0.08, 0.32, 0.55, 0.68, 0.92];
        let chart = calibration_chart(&[("Model", &mean_pred, &frac_pos)]);
        assert!(matches!(chart, Chart::Line(_)));
    }

    #[test]
    fn test_class_report_chart() {
        let report = classification_report(
            &[0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
            &[0.0, 1.0, 1.0, 1.0, 2.0, 0.0],
        );
        let chart = class_report_chart(&report);
        assert!(matches!(chart, Chart::Heatmap(_)));
    }

    #[test]
    fn test_metric_comparison_chart() {
        let models = vec!["DT".to_string(), "RF".to_string(), "KNN".to_string()];
        let chart = metric_comparison_chart(
            &models,
            &[
                ("Accuracy", &[0.90, 0.95, 0.88]),
                ("F1", &[0.89, 0.94, 0.87]),
            ],
        );
        assert!(matches!(chart, Chart::Bar(_)));
    }

    #[test]
    fn test_elbow_chart() {
        let ks = vec![2, 3, 4, 5, 6];
        let inertias = vec![500.0, 300.0, 180.0, 150.0, 140.0];
        let chart = elbow_chart(&ks, &inertias, Some(4));
        assert!(matches!(chart, Chart::Line(_)));
    }

    #[test]
    fn test_cluster_scatter() {
        let x = vec![1.0, 1.1, 5.0, 5.1, 9.0, 9.1];
        let y = vec![2.0, 2.1, 6.0, 6.1, 1.0, 1.1];
        let labels = vec![0, 0, 1, 1, 2, 2];
        let chart = cluster_scatter(&x, &y, &labels);
        assert!(matches!(chart, Chart::Scatter(_)));
    }

    #[test]
    fn test_silhouette_chart() {
        let labels = vec![0, 0, 0, 1, 1, 1];
        let scores = vec![0.8, 0.7, 0.6, 0.5, 0.4, 0.3];
        let chart = silhouette_chart(&labels, &scores);
        assert!(matches!(chart, Chart::Bar(_)));
    }

    #[test]
    fn test_validation_curve() {
        let param_range = vec![0.001, 0.01, 0.1, 1.0, 10.0];
        let train = vec![0.60, 0.80, 0.92, 0.95, 0.96];
        let val = vec![0.55, 0.75, 0.88, 0.85, 0.70];
        let chart = validation_curve("C", &param_range, &train, &val);
        assert!(matches!(chart, Chart::Line(_)));
    }

    #[test]
    fn test_partial_dependence_chart() {
        let fv = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let pdp = vec![0.2, 0.4, 0.6, 0.8, 1.0];
        let chart = partial_dependence_chart(&fv, &pdp, "feature_0");
        assert!(matches!(chart, Chart::Line(_)));
    }

    #[test]
    fn test_cv_boxplot() {
        let names = vec!["DT".to_string(), "RF".to_string(), "KNN".to_string()];
        let scores = vec![
            vec![0.88, 0.90, 0.85, 0.91, 0.87],
            vec![0.93, 0.95, 0.92, 0.94, 0.96],
            vec![0.80, 0.82, 0.79, 0.81, 0.83],
        ];
        let chart = cv_boxplot(&names, &scores);
        assert!(matches!(chart, Chart::BoxPlot(_)));
    }

    #[test]
    fn test_decision_boundary_chart() {
        let x = vec![0.0, 0.1, 1.0, 1.1];
        let y = vec![0.0, 0.1, 1.0, 1.1];
        let labels = vec![0, 0, 1, 1];
        let predict = |px: f64, _py: f64| -> usize {
            if px < 0.5 {
                0
            } else {
                1
            }
        };
        let chart = decision_boundary_chart(&x, &y, &labels, &predict, 10);
        assert!(matches!(chart, Chart::Scatter(_)));
    }

    #[test]
    fn test_scatter3d_data() {
        let mut dataset = crate::dataset::Dataset::new(
            vec![
                vec![1.0, 2.0, 3.0],
                vec![4.0, 5.0, 6.0],
                vec![7.0, 8.0, 9.0],
                vec![10.0, 11.0, 12.0],
            ],
            vec![0.0, 1.0, 2.0],
            vec!["f0".into(), "f1".into(), "f2".into(), "f3".into()],
            "class",
        );
        dataset.class_labels = Some(vec!["a".into(), "b".into(), "c".into()]);

        let (x, y, z, labels) = scatter3d_data(&dataset, 0, 1, 2);
        assert_eq!(x, vec![1.0, 2.0, 3.0]);
        assert_eq!(y, vec![4.0, 5.0, 6.0]);
        assert_eq!(z, vec![7.0, 8.0, 9.0]);
        assert_eq!(labels, vec![0, 1, 2]);
    }

    #[test]
    fn test_scatter3d_chart() {
        let mut dataset = crate::dataset::Dataset::new(
            vec![
                vec![1.0, 2.0, 3.0],
                vec![4.0, 5.0, 6.0],
                vec![7.0, 8.0, 9.0],
            ],
            vec![0.0, 1.0, 0.0],
            vec!["sepal_l".into(), "sepal_w".into(), "petal_l".into()],
            "species",
        );
        dataset.class_labels = Some(vec!["setosa".into(), "versicolor".into()]);

        let chart = scatter3d_chart(&dataset, 0, 1, 2);

        // Verify it can render
        let result = chart.render(200, 150);
        assert!(
            result.is_ok(),
            "scatter3d_chart should render: {:?}",
            result.err()
        );
    }
}
