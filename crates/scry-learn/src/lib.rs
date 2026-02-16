//! # scry-learn
//!
//! Production-grade machine learning toolkit with built-in
//! [`scry-chart`](https://docs.rs/scry-chart) visualization.
//!
//! Every model evaluation produces publication-quality charts — confusion
//! matrices, ROC curves, feature importances, and decision tree diagrams —
//! renderable inline in the terminal, as PNG, or as SVG.
//!
//! ## Quick Start
//!
//! ```ignore
//! use scry_learn::prelude::*;
//!
//! let data = Dataset::from_csv("iris.csv", "species")?;
//! let (train, test) = train_test_split(&data, 0.2, 42);
//!
//! let mut model = RandomForestClassifier::new()
//!     .n_estimators(100)
//!     .max_depth(10);
//! model.fit(&train)?;
//!
//! let preds = model.predict(&test)?;
//! let report = classification_report(&test.target, &preds);
//! println!("{report}");
//!
//! // Auto-generate confusion matrix chart
//! let chart = confusion_matrix_chart(&test.target, &preds, &data.class_labels);
//! scry_chart::export::save_png(&chart, 800, 600, "confusion.png")?;
//! ```

#![warn(missing_docs)]
#![deny(unsafe_code)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::missing_fields_in_debug)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::redundant_pub_crate)]
#![allow(clippy::use_self)]
#![allow(clippy::suspicious_operation_groupings)]

pub mod error;
pub(crate) mod rng;
pub mod matrix;
pub mod dataset;
pub mod distance;
pub mod split;
pub mod preprocess;
pub mod tree;
pub mod linear;
pub mod neighbors;
pub mod cluster;
pub mod naive_bayes;
pub mod svm;
pub mod metrics;
pub mod pipeline;
pub mod search;
pub mod feature_selection;
pub mod anomaly;
pub mod ensemble;
pub mod weights;
pub mod viz;
pub mod accel;
pub mod neural;
pub mod sparse;

/// Convenience re-exports for common usage.
pub mod prelude {
    pub use crate::dataset::Dataset;
    pub use crate::matrix::DenseMatrix;
    pub use crate::split::{
        train_test_split, stratified_split,
        cross_val_score, cross_val_score_stratified, ScoringFn,
        RepeatedKFold, repeated_cross_val_score, group_k_fold,
        time_series_split, cross_val_predict,
    };
    pub use crate::tree::{
        DecisionTreeClassifier, DecisionTreeRegressor,
        RandomForestClassifier, RandomForestRegressor,
        GradientBoostingClassifier, GradientBoostingRegressor,
        HistGradientBoostingClassifier, HistGradientBoostingRegressor,
        FeatureBinner, SplitCriterion, RegressionLoss,
    };
    pub use crate::linear::{LinearRegression, LogisticRegression, LassoRegression, ElasticNet, Solver, Penalty, Ridge};
    pub use crate::neighbors::{KnnClassifier, KnnRegressor, DistanceMetric, WeightFunction, Algorithm, KdTree};
    pub use crate::svm::{LinearSVC, LinearSVR, KernelSVC, KernelSVR, Kernel, Gamma};
    pub use crate::cluster::{KMeans, MiniBatchKMeans, Dbscan, silhouette_score, AgglomerativeClustering, Linkage};
    pub use crate::naive_bayes::{GaussianNb, BernoulliNB, MultinomialNB};
    pub use crate::metrics::{
        accuracy, precision, recall, f1_score,
        confusion_matrix, classification_report,
        log_loss, balanced_accuracy, cohen_kappa_score,
        mean_squared_error, r2_score,
        explained_variance_score, mean_absolute_percentage_error,
        adjusted_rand_index, calinski_harabasz_score, davies_bouldin_score,
        roc_curve, roc_auc_score,
    };
    pub use crate::preprocess::{
        StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, Transformer,
        Pca, OneHotEncoder, DropStrategy, UnknownStrategy,
        SimpleImputer, Strategy, ColumnTransformer,
        PolynomialFeatures, Normalizer, Norm,
    };
    pub use crate::viz::{
        confusion_matrix_chart, roc_chart, feature_importance_chart,
        residual_plot, regularization_path_chart,
        pr_chart, learning_curve, prediction_error_chart,
        calibration_chart, class_report_chart, metric_comparison_chart,
        elbow_chart, cluster_scatter, silhouette_chart,
        scatter3d_data, scatter3d_chart,
    };
    pub use crate::viz::model_viz::Visualize;
    pub use crate::pipeline::Pipeline;
    pub use crate::search::{
        GridSearchCV, RandomizedSearchCV, ParamValue, ParamGrid, Tunable, CvResult,
    };
    pub use crate::feature_selection::{
        VarianceThreshold, SelectKBest, ScoreFn, f_classif,
    };
    pub use crate::anomaly::IsolationForest;
    pub use crate::ensemble::{VotingClassifier, StackingClassifier, Voting};
    pub use crate::neural::{MLPClassifier, MLPRegressor, Activation, OptimizerKind};
    pub use crate::error::ScryLearnError;
    pub use crate::weights::ClassWeight;
    pub use crate::sparse::{CsrMatrix, CscMatrix};
}
