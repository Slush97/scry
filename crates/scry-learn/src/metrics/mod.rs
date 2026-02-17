// SPDX-License-Identifier: MIT OR Apache-2.0
//! Classification, regression, and clustering metrics.

mod classification;
mod clustering;
mod regression;
mod roc;

pub use classification::{
    accuracy, precision, recall, f1_score,
    confusion_matrix, classification_report,
    log_loss, balanced_accuracy, cohen_kappa_score,
    ConfusionMatrix, ClassificationReport, ClassMetrics,
    Average,
};
pub use clustering::{
    adjusted_rand_index, calinski_harabasz_score, davies_bouldin_score,
};
pub use regression::{
    mean_squared_error, root_mean_squared_error,
    mean_absolute_error, r2_score,
    explained_variance_score, mean_absolute_percentage_error,
};
pub use roc::{roc_curve, roc_auc_score, pr_curve, RocCurve, PrCurve};
