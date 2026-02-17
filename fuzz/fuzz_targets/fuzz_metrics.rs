//! Fuzz target: metric functions with degenerate inputs.
//!
//! Tests classification, regression, ROC/PR, and clustering metrics
//! with fuzz-derived vectors. Guards against known-degenerate inputs
//! (all-same labels, single class) that cause expected panics.

#![no_main]

use libfuzzer_sys::fuzz_target;
use scry_learn::metrics::{
    accuracy, adjusted_rand_index, balanced_accuracy, calinski_harabasz_score, cohen_kappa_score,
    confusion_matrix, davies_bouldin_score, explained_variance_score, f1_score,
    mean_absolute_percentage_error, mean_squared_error, pr_curve, precision, r2_score, recall,
    roc_auc_score, roc_curve, Average,
};

fuzz_target!(|data: &[u8]| {
    if data.len() < 10 {
        return;
    }

    let mut cursor = 0;

    let n = (data[cursor] % 18).max(2) as usize; // At least 2 elements.
    cursor += 1;
    let dispatch = data[cursor] % 4;
    cursor += 1;

    // Build y_true and y_pred from fuzz bytes.
    let mut y_true = Vec::with_capacity(n);
    let mut y_pred = Vec::with_capacity(n);
    for _ in 0..n {
        let t = if cursor < data.len() {
            let v = data[cursor] as f64 / 128.0 - 1.0;
            cursor += 1;
            v
        } else {
            0.0
        };
        let p = if cursor < data.len() {
            let v = data[cursor] as f64 / 128.0 - 1.0;
            cursor += 1;
            v
        } else {
            0.0
        };
        y_true.push(t);
        y_pred.push(p);
    }

    match dispatch {
        0 => {
            // Classification metrics — need binary class labels with both classes present.
            let mut y_true_cls: Vec<f64> =
                y_true.iter().map(|v| if *v >= 0.0 { 1.0 } else { 0.0 }).collect();
            let y_pred_cls: Vec<f64> =
                y_pred.iter().map(|v| if *v >= 0.0 { 1.0 } else { 0.0 }).collect();
            // Ensure both classes are present in y_true.
            if y_true_cls.iter().all(|&v| v == 1.0) {
                y_true_cls[0] = 0.0;
            } else if y_true_cls.iter().all(|&v| v == 0.0) {
                y_true_cls[0] = 1.0;
            }
            let _ = accuracy(&y_true_cls, &y_pred_cls);
            let _ = balanced_accuracy(&y_true_cls, &y_pred_cls);
            let _ = precision(&y_true_cls, &y_pred_cls, Average::Binary);
            let _ = recall(&y_true_cls, &y_pred_cls, Average::Binary);
            let _ = f1_score(&y_true_cls, &y_pred_cls, Average::Binary);
            let _ = cohen_kappa_score(&y_true_cls, &y_pred_cls);
            let _ = confusion_matrix(&y_true_cls, &y_pred_cls);
        }
        1 => {
            // Regression metrics.
            let _ = r2_score(&y_true, &y_pred);
            let _ = mean_squared_error(&y_true, &y_pred);
            let _ = mean_absolute_percentage_error(&y_true, &y_pred);
            let _ = explained_variance_score(&y_true, &y_pred);
        }
        2 => {
            // ROC/PR — need binary labels with both classes present.
            let mut y_true_bin: Vec<f64> =
                y_true.iter().map(|v| if *v >= 0.0 { 1.0 } else { 0.0 }).collect();
            let y_scores: Vec<f64> = y_pred.iter().map(|v| (v + 1.0) / 2.0).collect();
            // Ensure both classes present for ROC/PR.
            if y_true_bin.iter().all(|&v| v == 1.0) {
                y_true_bin[0] = 0.0;
            } else if y_true_bin.iter().all(|&v| v == 0.0) {
                y_true_bin[0] = 1.0;
            }
            let _ = roc_auc_score(&y_true_bin, &y_scores);
            let _ = roc_curve(&y_true_bin, &y_scores);
            let _ = pr_curve(&y_true_bin, &y_scores);
        }
        _ => {
            // Clustering metrics — need cluster labels with at least 2 distinct clusters.
            let mut labels_true: Vec<f64> =
                y_true.iter().map(|v| (v.abs() * 3.0).floor()).collect();
            let mut labels_pred: Vec<f64> =
                y_pred.iter().map(|v| (v.abs() * 3.0).floor()).collect();
            // Ensure at least 2 distinct labels.
            if labels_pred.iter().all(|&v| v == labels_pred[0]) {
                labels_pred[0] = labels_pred[0] + 1.0;
            }
            if labels_true.iter().all(|&v| v == labels_true[0]) {
                labels_true[0] = labels_true[0] + 1.0;
            }
            let _ = adjusted_rand_index(&labels_true, &labels_pred);

            // calinski/davies need column-major features + labels.
            if n >= 2 {
                let n_feat = 2;
                let mut col_features: Vec<Vec<f64>> = Vec::with_capacity(n_feat);
                for _ in 0..n_feat {
                    let mut col = Vec::with_capacity(n);
                    for i in 0..n {
                        if cursor < data.len() {
                            col.push((data[cursor] as f64 / 128.0) - 1.0);
                            cursor += 1;
                        } else {
                            col.push(i as f64 / 10.0);
                        }
                    }
                    col_features.push(col);
                }
                let _ = calinski_harabasz_score(&col_features, &labels_pred);
                let _ = davies_bouldin_score(&col_features, &labels_pred);
            }
        }
    }
});
