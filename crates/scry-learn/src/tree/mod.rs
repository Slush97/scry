//! Tree-based models: Decision Tree, Random Forest, Gradient Boosting, and
//! Histogram-based Gradient Boosting.

mod cart;
mod gradient_boosting;
mod random_forest;
pub mod binning;
mod histogram_gbt;

pub use cart::{
    DecisionTreeClassifier, DecisionTreeRegressor,
    FlatNode, FlatTree, SplitCriterion, TreeNode,
};
pub use gradient_boosting::{
    GradientBoostingClassifier, GradientBoostingRegressor,
    RegressionLoss,
};
pub use random_forest::{
    RandomForestClassifier, RandomForestRegressor,
    MaxFeatures,
};
pub use binning::FeatureBinner;
pub use histogram_gbt::{
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
};
