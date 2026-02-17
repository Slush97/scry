// SPDX-License-Identifier: MIT OR Apache-2.0
//! K-Nearest Neighbors classifier and regressor.

mod knn;
pub mod kdtree;
pub use knn::{KnnClassifier, KnnRegressor, DistanceMetric, WeightFunction, Algorithm};
pub use kdtree::KdTree;
