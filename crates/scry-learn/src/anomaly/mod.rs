//! Anomaly detection algorithms.
//!
//! Currently provides [`IsolationForest`] for unsupervised outlier detection
//! using random isolation trees.

mod iforest;

pub use iforest::IsolationForest;
