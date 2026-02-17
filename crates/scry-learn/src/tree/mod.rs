// SPDX-License-Identifier: MIT OR Apache-2.0
//! Tree-based models: Decision Tree, Random Forest, Gradient Boosting, and
//! Histogram-based Gradient Boosting.

pub mod binning;
mod cart;
mod gradient_boosting;
mod histogram_gbt;
mod random_forest;

pub use binning::FeatureBinner;
pub use cart::{
    DecisionTreeClassifier, DecisionTreeRegressor, FlatNode, FlatTree, SplitCriterion, TreeNode,
};
pub use gradient_boosting::{
    GradientBoostingClassifier, GradientBoostingRegressor, RegressionLoss,
};
pub use histogram_gbt::{
    HistGradientBoostingClassifier, HistGradientBoostingRegressor, HistNodeView,
};
pub use random_forest::{MaxFeatures, RandomForestClassifier, RandomForestRegressor};
