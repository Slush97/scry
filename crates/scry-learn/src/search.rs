//! Hyperparameter search via cross-validation.
//!
//! [`GridSearchCV`] performs exhaustive search over a parameter grid,
//! while [`RandomizedSearchCV`] samples random combinations for faster
//! exploration of large search spaces.
//!
//! # Examples
//!
//! ```ignore
//! use scry_learn::prelude::*;
//! use scry_learn::search::*;
//!
//! let mut grid = ParamGrid::new();
//! grid.insert("max_depth".into(), vec![ParamValue::Int(2), ParamValue::Int(6)]);
//!
//! let result = GridSearchCV::new(DecisionTreeClassifier::new(), grid)
//!     .cv(5)
//!     .scoring(accuracy)
//!     .fit(&data)
//!     .unwrap();
//!
//! println!("Best score: {}", result.best_score());
//! ```

use std::collections::HashMap;

use crate::dataset::Dataset;
use crate::error::{Result, ScryLearnError};
use crate::metrics::accuracy;
use crate::split::{k_fold, stratified_k_fold, ScoringFn};

// ---------------------------------------------------------------------------
// ParamValue + ParamGrid
// ---------------------------------------------------------------------------

/// A single hyperparameter value.
///
/// # Examples
///
/// ```
/// use scry_learn::search::ParamValue;
///
/// let depth = ParamValue::Int(5);
/// let lr = ParamValue::Float(0.01);
/// let flag = ParamValue::Bool(true);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ParamValue {
    /// Integer parameter (e.g. `max_depth`, `n_estimators`).
    Int(usize),
    /// Floating-point parameter (e.g. `learning_rate`).
    Float(f64),
    /// Boolean parameter (e.g. `bootstrap`).
    Bool(bool),
    /// Categorical / string parameter (e.g. `criterion = "gini"`).
    Categorical(String),
}

impl std::fmt::Display for ParamValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParamValue::Int(v) => write!(f, "{v}"),
            ParamValue::Float(v) => write!(f, "{v}"),
            ParamValue::Bool(v) => write!(f, "{v}"),
            ParamValue::Categorical(v) => write!(f, "{v}"),
        }
    }
}

/// A grid of hyperparameter values to search over.
///
/// Keys are parameter names (e.g. `"max_depth"`), values are lists of
/// candidate values to try.
///
/// # Examples
///
/// ```
/// use scry_learn::search::{ParamGrid, ParamValue};
///
/// let mut grid = ParamGrid::new();
/// grid.insert("max_depth".into(), vec![
///     ParamValue::Int(2),
///     ParamValue::Int(4),
///     ParamValue::Int(8),
/// ]);
/// ```
pub type ParamGrid = HashMap<String, Vec<ParamValue>>;

// ---------------------------------------------------------------------------
// Tunable trait
// ---------------------------------------------------------------------------

/// A model whose hyperparameters can be set dynamically by name.
///
/// Implement this trait on any model that should participate in
/// [`GridSearchCV`] or [`RandomizedSearchCV`].
///
/// # Examples
///
/// ```ignore
/// use scry_learn::search::{Tunable, ParamValue};
///
/// let mut dt = DecisionTreeClassifier::new();
/// dt.set_param("max_depth", ParamValue::Int(5)).unwrap();
/// ```
pub trait Tunable {
    /// Apply a named hyperparameter.
    ///
    /// Returns [`ScryLearnError::InvalidParameter`] if the parameter name
    /// is unrecognised or the value type is wrong.
    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()>;

    /// Clone this model into a boxed trait object.
    fn clone_box(&self) -> Box<dyn Tunable>;

    /// Train on a dataset.
    fn fit(&mut self, data: &Dataset) -> Result<()>;

    /// Predict on row-major features.
    fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>>;
}

// ---------------------------------------------------------------------------
// Tunable impls for existing models
// ---------------------------------------------------------------------------

impl Tunable for crate::tree::DecisionTreeClassifier {
    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match name {
            "max_depth" => {
                if let ParamValue::Int(v) = value {
                    *self = self.clone().max_depth(v);
                    Ok(())
                } else {
                    Err(ScryLearnError::InvalidParameter(
                        format!("max_depth expects Int, got {value}"),
                    ))
                }
            }
            "min_samples_split" => {
                if let ParamValue::Int(v) = value {
                    *self = self.clone().min_samples_split(v);
                    Ok(())
                } else {
                    Err(ScryLearnError::InvalidParameter(
                        format!("min_samples_split expects Int, got {value}"),
                    ))
                }
            }
            "min_samples_leaf" => {
                if let ParamValue::Int(v) = value {
                    *self = self.clone().min_samples_leaf(v);
                    Ok(())
                } else {
                    Err(ScryLearnError::InvalidParameter(
                        format!("min_samples_leaf expects Int, got {value}"),
                    ))
                }
            }
            _ => Err(ScryLearnError::InvalidParameter(
                format!("unknown parameter: {name}"),
            )),
        }
    }

    fn clone_box(&self) -> Box<dyn Tunable> {
        Box::new(self.clone())
    }

    fn fit(&mut self, data: &Dataset) -> Result<()> {
        self.fit(data)
    }

    fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> {
        self.predict(features)
    }
}

impl Tunable for crate::tree::RandomForestClassifier {
    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match name {
            "n_estimators" => {
                if let ParamValue::Int(v) = value {
                    *self = self.clone().n_estimators(v);
                    Ok(())
                } else {
                    Err(ScryLearnError::InvalidParameter(
                        format!("n_estimators expects Int, got {value}"),
                    ))
                }
            }
            "max_depth" => {
                if let ParamValue::Int(v) = value {
                    *self = self.clone().max_depth(v);
                    Ok(())
                } else {
                    Err(ScryLearnError::InvalidParameter(
                        format!("max_depth expects Int, got {value}"),
                    ))
                }
            }
            _ => Err(ScryLearnError::InvalidParameter(
                format!("unknown parameter: {name}"),
            )),
        }
    }

    fn clone_box(&self) -> Box<dyn Tunable> {
        Box::new(self.clone())
    }

    fn fit(&mut self, data: &Dataset) -> Result<()> {
        self.fit(data)
    }

    fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> {
        self.predict(features)
    }
}

impl Tunable for crate::linear::LogisticRegression {
    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match name {
            "learning_rate" => { if let ParamValue::Float(v) = value { *self = self.clone().learning_rate(v); Ok(()) } else { Err(ScryLearnError::InvalidParameter(format!("learning_rate expects Float, got {value}"))) } }
            "max_iter" => { if let ParamValue::Int(v) = value { *self = self.clone().max_iter(v); Ok(()) } else { Err(ScryLearnError::InvalidParameter(format!("max_iter expects Int, got {value}"))) } }
            "alpha" => { if let ParamValue::Float(v) = value { *self = self.clone().alpha(v); Ok(()) } else { Err(ScryLearnError::InvalidParameter(format!("alpha expects Float, got {value}"))) } }
            "tolerance" => { if let ParamValue::Float(v) = value { *self = self.clone().tolerance(v); Ok(()) } else { Err(ScryLearnError::InvalidParameter(format!("tolerance expects Float, got {value}"))) } }
            _ => Err(ScryLearnError::InvalidParameter(format!("unknown parameter: {name}"))),
        }
    }
    fn clone_box(&self) -> Box<dyn Tunable> { Box::new(self.clone()) }
    fn fit(&mut self, data: &Dataset) -> Result<()> { self.fit(data) }
    fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> { self.predict(features) }
}

impl Tunable for crate::neighbors::KnnClassifier {
    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match name {
            "k" => { if let ParamValue::Int(v) = value { *self = self.clone().k(v); Ok(()) } else { Err(ScryLearnError::InvalidParameter(format!("k expects Int, got {value}"))) } }
            _ => Err(ScryLearnError::InvalidParameter(format!("unknown parameter: {name}"))),
        }
    }
    fn clone_box(&self) -> Box<dyn Tunable> { Box::new(self.clone()) }
    fn fit(&mut self, data: &Dataset) -> Result<()> { self.fit(data) }
    fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> { self.predict(features) }
}

impl Tunable for crate::neighbors::KnnRegressor {
    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match name {
            "k" => { if let ParamValue::Int(v) = value { *self = self.clone().k(v); Ok(()) } else { Err(ScryLearnError::InvalidParameter(format!("k expects Int, got {value}"))) } }
            _ => Err(ScryLearnError::InvalidParameter(format!("unknown parameter: {name}"))),
        }
    }
    fn clone_box(&self) -> Box<dyn Tunable> { Box::new(self.clone()) }
    fn fit(&mut self, data: &Dataset) -> Result<()> { self.fit(data) }
    fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> { self.predict(features) }
}

impl Tunable for crate::cluster::KMeans {
    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match name {
            "max_iter" => { if let ParamValue::Int(v) = value { *self = self.clone().max_iter(v); Ok(()) } else { Err(ScryLearnError::InvalidParameter(format!("max_iter expects Int, got {value}"))) } }
            "tolerance" => { if let ParamValue::Float(v) = value { *self = self.clone().tolerance(v); Ok(()) } else { Err(ScryLearnError::InvalidParameter(format!("tolerance expects Float, got {value}"))) } }
            "n_init" => { if let ParamValue::Int(v) = value { *self = self.clone().n_init(v); Ok(()) } else { Err(ScryLearnError::InvalidParameter(format!("n_init expects Int, got {value}"))) } }
            _ => Err(ScryLearnError::InvalidParameter(format!("unknown parameter: {name}"))),
        }
    }
    fn clone_box(&self) -> Box<dyn Tunable> { Box::new(self.clone()) }
    fn fit(&mut self, data: &Dataset) -> Result<()> { self.fit(data) }
    fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> {
        let labels = crate::cluster::KMeans::predict(self, features)?;
        Ok(labels.into_iter().map(|l| l as f64).collect())
    }
}

impl Tunable for crate::tree::GradientBoostingRegressor {
    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match name {
            "n_estimators" => { if let ParamValue::Int(v) = value { *self = self.clone().n_estimators(v); Ok(()) } else { Err(ScryLearnError::InvalidParameter(format!("n_estimators expects Int, got {value}"))) } }
            "learning_rate" => { if let ParamValue::Float(v) = value { *self = self.clone().learning_rate(v); Ok(()) } else { Err(ScryLearnError::InvalidParameter(format!("learning_rate expects Float, got {value}"))) } }
            "max_depth" => { if let ParamValue::Int(v) = value { *self = self.clone().max_depth(v); Ok(()) } else { Err(ScryLearnError::InvalidParameter(format!("max_depth expects Int, got {value}"))) } }
            "min_samples_split" => { if let ParamValue::Int(v) = value { *self = self.clone().min_samples_split(v); Ok(()) } else { Err(ScryLearnError::InvalidParameter(format!("min_samples_split expects Int, got {value}"))) } }
            "min_samples_leaf" => { if let ParamValue::Int(v) = value { *self = self.clone().min_samples_leaf(v); Ok(()) } else { Err(ScryLearnError::InvalidParameter(format!("min_samples_leaf expects Int, got {value}"))) } }
            _ => Err(ScryLearnError::InvalidParameter(format!("unknown parameter: {name}"))),
        }
    }
    fn clone_box(&self) -> Box<dyn Tunable> { Box::new(self.clone()) }
    fn fit(&mut self, data: &Dataset) -> Result<()> { self.fit(data) }
    fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> { self.predict(features) }
}

impl Tunable for crate::tree::GradientBoostingClassifier {
    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match name {
            "n_estimators" => { if let ParamValue::Int(v) = value { *self = self.clone().n_estimators(v); Ok(()) } else { Err(ScryLearnError::InvalidParameter(format!("n_estimators expects Int, got {value}"))) } }
            "learning_rate" => { if let ParamValue::Float(v) = value { *self = self.clone().learning_rate(v); Ok(()) } else { Err(ScryLearnError::InvalidParameter(format!("learning_rate expects Float, got {value}"))) } }
            "max_depth" => { if let ParamValue::Int(v) = value { *self = self.clone().max_depth(v); Ok(()) } else { Err(ScryLearnError::InvalidParameter(format!("max_depth expects Int, got {value}"))) } }
            "min_samples_split" => { if let ParamValue::Int(v) = value { *self = self.clone().min_samples_split(v); Ok(()) } else { Err(ScryLearnError::InvalidParameter(format!("min_samples_split expects Int, got {value}"))) } }
            "min_samples_leaf" => { if let ParamValue::Int(v) = value { *self = self.clone().min_samples_leaf(v); Ok(()) } else { Err(ScryLearnError::InvalidParameter(format!("min_samples_leaf expects Int, got {value}"))) } }
            _ => Err(ScryLearnError::InvalidParameter(format!("unknown parameter: {name}"))),
        }
    }
    fn clone_box(&self) -> Box<dyn Tunable> { Box::new(self.clone()) }
    fn fit(&mut self, data: &Dataset) -> Result<()> { self.fit(data) }
    fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> { self.predict(features) }
}

impl Tunable for crate::svm::LinearSVC {
    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match name {
            "c" => { if let ParamValue::Float(v) = value { *self = self.clone().c(v); Ok(()) } else { Err(ScryLearnError::InvalidParameter(format!("c expects Float, got {value}"))) } }
            "max_iter" => { if let ParamValue::Int(v) = value { *self = self.clone().max_iter(v); Ok(()) } else { Err(ScryLearnError::InvalidParameter(format!("max_iter expects Int, got {value}"))) } }
            "tol" => { if let ParamValue::Float(v) = value { *self = self.clone().tol(v); Ok(()) } else { Err(ScryLearnError::InvalidParameter(format!("tol expects Float, got {value}"))) } }
            _ => Err(ScryLearnError::InvalidParameter(format!("unknown parameter: {name}"))),
        }
    }
    fn clone_box(&self) -> Box<dyn Tunable> { Box::new(self.clone()) }
    fn fit(&mut self, data: &Dataset) -> Result<()> { self.fit(data) }
    fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> { self.predict(features) }
}

impl Tunable for crate::svm::LinearSVR {
    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match name {
            "c" => { if let ParamValue::Float(v) = value { *self = self.clone().c(v); Ok(()) } else { Err(ScryLearnError::InvalidParameter(format!("c expects Float, got {value}"))) } }
            "epsilon" => { if let ParamValue::Float(v) = value { *self = self.clone().epsilon(v); Ok(()) } else { Err(ScryLearnError::InvalidParameter(format!("epsilon expects Float, got {value}"))) } }
            "max_iter" => { if let ParamValue::Int(v) = value { *self = self.clone().max_iter(v); Ok(()) } else { Err(ScryLearnError::InvalidParameter(format!("max_iter expects Int, got {value}"))) } }
            "tol" => { if let ParamValue::Float(v) = value { *self = self.clone().tol(v); Ok(()) } else { Err(ScryLearnError::InvalidParameter(format!("tol expects Float, got {value}"))) } }
            _ => Err(ScryLearnError::InvalidParameter(format!("unknown parameter: {name}"))),
        }
    }
    fn clone_box(&self) -> Box<dyn Tunable> { Box::new(self.clone()) }
    fn fit(&mut self, data: &Dataset) -> Result<()> { self.fit(data) }
    fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> { self.predict(features) }
}

impl Tunable for crate::svm::KernelSVC {
    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match name {
            "c" => { if let ParamValue::Float(v) = value { *self = self.clone().c(v); Ok(()) } else { Err(ScryLearnError::InvalidParameter(format!("c expects Float, got {value}"))) } }
            "tol" => { if let ParamValue::Float(v) = value { *self = self.clone().tol(v); Ok(()) } else { Err(ScryLearnError::InvalidParameter(format!("tol expects Float, got {value}"))) } }
            "max_iter" => { if let ParamValue::Int(v) = value { *self = self.clone().max_iter(v); Ok(()) } else { Err(ScryLearnError::InvalidParameter(format!("max_iter expects Int, got {value}"))) } }
            _ => Err(ScryLearnError::InvalidParameter(format!("unknown parameter: {name}"))),
        }
    }
    fn clone_box(&self) -> Box<dyn Tunable> { Box::new(self.clone()) }
    fn fit(&mut self, data: &Dataset) -> Result<()> { self.fit(data) }
    fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> { self.predict(features) }
}

impl Tunable for crate::svm::KernelSVR {
    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match name {
            "c" => { if let ParamValue::Float(v) = value { *self = self.clone().c(v); Ok(()) } else { Err(ScryLearnError::InvalidParameter(format!("c expects Float, got {value}"))) } }
            "epsilon" => { if let ParamValue::Float(v) = value { *self = self.clone().epsilon(v); Ok(()) } else { Err(ScryLearnError::InvalidParameter(format!("epsilon expects Float, got {value}"))) } }
            "tol" => { if let ParamValue::Float(v) = value { *self = self.clone().tol(v); Ok(()) } else { Err(ScryLearnError::InvalidParameter(format!("tol expects Float, got {value}"))) } }
            "max_iter" => { if let ParamValue::Int(v) = value { *self = self.clone().max_iter(v); Ok(()) } else { Err(ScryLearnError::InvalidParameter(format!("max_iter expects Int, got {value}"))) } }
            _ => Err(ScryLearnError::InvalidParameter(format!("unknown parameter: {name}"))),
        }
    }
    fn clone_box(&self) -> Box<dyn Tunable> { Box::new(self.clone()) }
    fn fit(&mut self, data: &Dataset) -> Result<()> { self.fit(data) }
    fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> { self.predict(features) }
}

impl Tunable for crate::naive_bayes::GaussianNb {
    fn set_param(&mut self, name: &str, _value: ParamValue) -> Result<()> {
        Err(ScryLearnError::InvalidParameter(format!("unknown parameter: {name}")))
    }
    fn clone_box(&self) -> Box<dyn Tunable> { Box::new(self.clone()) }
    fn fit(&mut self, data: &Dataset) -> Result<()> { self.fit(data) }
    fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> { self.predict(features) }
}

impl Tunable for crate::naive_bayes::BernoulliNB {
    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match name {
            "alpha" => { if let ParamValue::Float(v) = value { *self = self.clone().alpha(v); Ok(()) } else { Err(ScryLearnError::InvalidParameter(format!("alpha expects Float, got {value}"))) } }
            _ => Err(ScryLearnError::InvalidParameter(format!("unknown parameter: {name}"))),
        }
    }
    fn clone_box(&self) -> Box<dyn Tunable> { Box::new(self.clone()) }
    fn fit(&mut self, data: &Dataset) -> Result<()> { self.fit(data) }
    fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> { self.predict(features) }
}

impl Tunable for crate::naive_bayes::MultinomialNB {
    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match name {
            "alpha" => { if let ParamValue::Float(v) = value { *self = self.clone().alpha(v); Ok(()) } else { Err(ScryLearnError::InvalidParameter(format!("alpha expects Float, got {value}"))) } }
            _ => Err(ScryLearnError::InvalidParameter(format!("unknown parameter: {name}"))),
        }
    }
    fn clone_box(&self) -> Box<dyn Tunable> { Box::new(self.clone()) }
    fn fit(&mut self, data: &Dataset) -> Result<()> { self.fit(data) }
    fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> { self.predict(features) }
}

impl Tunable for crate::linear::LassoRegression {
    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match name {
            "alpha" => { if let ParamValue::Float(v) = value { *self = self.clone().alpha(v); Ok(()) } else { Err(ScryLearnError::InvalidParameter(format!("alpha expects Float, got {value}"))) } }
            "max_iter" => { if let ParamValue::Int(v) = value { *self = self.clone().max_iter(v); Ok(()) } else { Err(ScryLearnError::InvalidParameter(format!("max_iter expects Int, got {value}"))) } }
            "tol" => { if let ParamValue::Float(v) = value { *self = self.clone().tol(v); Ok(()) } else { Err(ScryLearnError::InvalidParameter(format!("tol expects Float, got {value}"))) } }
            _ => Err(ScryLearnError::InvalidParameter(format!("unknown parameter: {name}"))),
        }
    }
    fn clone_box(&self) -> Box<dyn Tunable> { Box::new(self.clone()) }
    fn fit(&mut self, data: &Dataset) -> Result<()> { self.fit(data) }
    fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> { self.predict(features) }
}

impl Tunable for crate::linear::ElasticNet {
    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match name {
            "alpha" => { if let ParamValue::Float(v) = value { *self = self.clone().alpha(v); Ok(()) } else { Err(ScryLearnError::InvalidParameter(format!("alpha expects Float, got {value}"))) } }
            "l1_ratio" => { if let ParamValue::Float(v) = value { *self = self.clone().l1_ratio(v); Ok(()) } else { Err(ScryLearnError::InvalidParameter(format!("l1_ratio expects Float, got {value}"))) } }
            "max_iter" => { if let ParamValue::Int(v) = value { *self = self.clone().max_iter(v); Ok(()) } else { Err(ScryLearnError::InvalidParameter(format!("max_iter expects Int, got {value}"))) } }
            "tol" => { if let ParamValue::Float(v) = value { *self = self.clone().tol(v); Ok(()) } else { Err(ScryLearnError::InvalidParameter(format!("tol expects Float, got {value}"))) } }
            _ => Err(ScryLearnError::InvalidParameter(format!("unknown parameter: {name}"))),
        }
    }
    fn clone_box(&self) -> Box<dyn Tunable> { Box::new(self.clone()) }
    fn fit(&mut self, data: &Dataset) -> Result<()> { self.fit(data) }
    fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> { self.predict(features) }
}

impl Tunable for crate::tree::HistGradientBoostingRegressor {
    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match name {
            "n_estimators" => { if let ParamValue::Int(v) = value { *self = self.clone().n_estimators(v); Ok(()) } else { Err(ScryLearnError::InvalidParameter(format!("n_estimators expects Int, got {value}"))) } }
            "learning_rate" => { if let ParamValue::Float(v) = value { *self = self.clone().learning_rate(v); Ok(()) } else { Err(ScryLearnError::InvalidParameter(format!("learning_rate expects Float, got {value}"))) } }
            "max_leaf_nodes" => { if let ParamValue::Int(v) = value { *self = self.clone().max_leaf_nodes(v); Ok(()) } else { Err(ScryLearnError::InvalidParameter(format!("max_leaf_nodes expects Int, got {value}"))) } }
            "max_depth" => { if let ParamValue::Int(v) = value { *self = self.clone().max_depth(v); Ok(()) } else { Err(ScryLearnError::InvalidParameter(format!("max_depth expects Int, got {value}"))) } }
            "min_samples_leaf" => { if let ParamValue::Int(v) = value { *self = self.clone().min_samples_leaf(v); Ok(()) } else { Err(ScryLearnError::InvalidParameter(format!("min_samples_leaf expects Int, got {value}"))) } }
            _ => Err(ScryLearnError::InvalidParameter(format!("unknown parameter: {name}"))),
        }
    }
    fn clone_box(&self) -> Box<dyn Tunable> { Box::new(self.clone()) }
    fn fit(&mut self, data: &Dataset) -> Result<()> { self.fit(data) }
    fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> { self.predict(features) }
}

impl Tunable for crate::tree::HistGradientBoostingClassifier {
    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match name {
            "n_estimators" => { if let ParamValue::Int(v) = value { *self = self.clone().n_estimators(v); Ok(()) } else { Err(ScryLearnError::InvalidParameter(format!("n_estimators expects Int, got {value}"))) } }
            "learning_rate" => { if let ParamValue::Float(v) = value { *self = self.clone().learning_rate(v); Ok(()) } else { Err(ScryLearnError::InvalidParameter(format!("learning_rate expects Float, got {value}"))) } }
            "max_leaf_nodes" => { if let ParamValue::Int(v) = value { *self = self.clone().max_leaf_nodes(v); Ok(()) } else { Err(ScryLearnError::InvalidParameter(format!("max_leaf_nodes expects Int, got {value}"))) } }
            "max_depth" => { if let ParamValue::Int(v) = value { *self = self.clone().max_depth(v); Ok(()) } else { Err(ScryLearnError::InvalidParameter(format!("max_depth expects Int, got {value}"))) } }
            "min_samples_leaf" => { if let ParamValue::Int(v) = value { *self = self.clone().min_samples_leaf(v); Ok(()) } else { Err(ScryLearnError::InvalidParameter(format!("min_samples_leaf expects Int, got {value}"))) } }
            _ => Err(ScryLearnError::InvalidParameter(format!("unknown parameter: {name}"))),
        }
    }
    fn clone_box(&self) -> Box<dyn Tunable> { Box::new(self.clone()) }
    fn fit(&mut self, data: &Dataset) -> Result<()> { self.fit(data) }
    fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> { self.predict(features) }
}

impl Tunable for crate::tree::DecisionTreeRegressor {
    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match name {
            "max_depth" => { if let ParamValue::Int(v) = value { *self = self.clone().max_depth(v); Ok(()) } else { Err(ScryLearnError::InvalidParameter(format!("max_depth expects Int, got {value}"))) } }
            "min_samples_split" => { if let ParamValue::Int(v) = value { *self = self.clone().min_samples_split(v); Ok(()) } else { Err(ScryLearnError::InvalidParameter(format!("min_samples_split expects Int, got {value}"))) } }
            "min_samples_leaf" => { if let ParamValue::Int(v) = value { *self = self.clone().min_samples_leaf(v); Ok(()) } else { Err(ScryLearnError::InvalidParameter(format!("min_samples_leaf expects Int, got {value}"))) } }
            _ => Err(ScryLearnError::InvalidParameter(format!("unknown parameter: {name}"))),
        }
    }
    fn clone_box(&self) -> Box<dyn Tunable> { Box::new(self.clone()) }
    fn fit(&mut self, data: &Dataset) -> Result<()> { self.fit(data) }
    fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> { self.predict(features) }
}

impl Tunable for crate::anomaly::IsolationForest {
    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match name {
            "n_estimators" => { if let ParamValue::Int(v) = value { *self = self.clone().n_estimators(v); Ok(()) } else { Err(ScryLearnError::InvalidParameter(format!("n_estimators expects Int, got {value}"))) } }
            "max_samples" => { if let ParamValue::Int(v) = value { *self = self.clone().max_samples(v); Ok(()) } else { Err(ScryLearnError::InvalidParameter(format!("max_samples expects Int, got {value}"))) } }
            "contamination" => { if let ParamValue::Float(v) = value { *self = self.clone().contamination(v); Ok(()) } else { Err(ScryLearnError::InvalidParameter(format!("contamination expects Float, got {value}"))) } }
            _ => Err(ScryLearnError::InvalidParameter(format!("unknown parameter: {name}"))),
        }
    }
    fn clone_box(&self) -> Box<dyn Tunable> { Box::new(self.clone()) }
    fn fit(&mut self, data: &Dataset) -> Result<()> {
        let features = data.feature_matrix();
        self.fit(&features)
    }
    fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> {
        Ok(self.predict(features))
    }
}

// ---------------------------------------------------------------------------
// CvResult
// ---------------------------------------------------------------------------

/// Result of a single parameter combination evaluated via cross-validation.
///
/// # Examples
///
/// ```ignore
/// for r in search_result.cv_results() {
///     println!("params={:?}  mean_score={:.3}", r.params, r.mean_score);
/// }
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CvResult {
    /// The parameter combination that was evaluated.
    pub params: HashMap<String, ParamValue>,
    /// Mean score across all CV folds.
    pub mean_score: f64,
    /// Individual fold scores.
    pub fold_scores: Vec<f64>,
}

// ---------------------------------------------------------------------------
// GridSearchCV
// ---------------------------------------------------------------------------

/// Exhaustive search over a hyperparameter grid with cross-validation.
///
/// Tries every combination in the grid, evaluates each with k-fold CV,
/// and reports the best-performing parameter set.
///
/// # Examples
///
/// ```ignore
/// use scry_learn::prelude::*;
/// use scry_learn::search::*;
///
/// let mut grid = ParamGrid::new();
/// grid.insert("max_depth".into(), vec![
///     ParamValue::Int(2), ParamValue::Int(4), ParamValue::Int(8),
/// ]);
///
/// let result = GridSearchCV::new(DecisionTreeClassifier::new(), grid)
///     .cv(5)
///     .scoring(accuracy)
///     .fit(&data)
///     .unwrap();
///
/// println!("Best: {:?} → {:.3}", result.best_params(), result.best_score());
/// ```
pub struct GridSearchCV {
    base_model: Box<dyn Tunable>,
    param_grid: ParamGrid,
    cv: usize,
    scorer: ScoringFn,
    seed: u64,
    stratified: bool,
    // Results (populated after fit)
    best_params_: Option<HashMap<String, ParamValue>>,
    best_score_: f64,
    cv_results_: Vec<CvResult>,
}

impl GridSearchCV {
    /// Create a grid search over the given model and parameter grid.
    ///
    /// Defaults: 5-fold CV, accuracy scorer, seed 42, non-stratified.
    pub fn new(model: impl Tunable + 'static, grid: ParamGrid) -> Self {
        Self {
            base_model: Box::new(model),
            param_grid: grid,
            cv: 5,
            scorer: accuracy,
            seed: 42,
            stratified: false,
            best_params_: None,
            best_score_: f64::NEG_INFINITY,
            cv_results_: Vec::new(),
        }
    }

    /// Set the number of cross-validation folds (default: 5).
    pub fn cv(mut self, k: usize) -> Self {
        self.cv = k;
        self
    }

    /// Set the scoring function (default: `accuracy`).
    pub fn scoring(mut self, scorer: ScoringFn) -> Self {
        self.scorer = scorer;
        self
    }

    /// Set the random seed for fold generation (default: 42).
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Enable stratified k-fold CV (default: `false`).
    ///
    /// When `true`, uses [`stratified_k_fold`](crate::split::stratified_k_fold)
    /// to preserve class proportions in each fold.
    pub fn stratified(mut self, stratified: bool) -> Self {
        self.stratified = stratified;
        self
    }

    /// Run the exhaustive grid search.
    ///
    /// Returns `self` for chained accessor calls.
    pub fn fit(mut self, data: &Dataset) -> Result<Self> {
        let combos = cartesian_product(&self.param_grid);
        if combos.is_empty() {
            return Err(ScryLearnError::InvalidParameter(
                "parameter grid is empty".into(),
            ));
        }

        let folds = if self.stratified {
            stratified_k_fold(data, self.cv, self.seed)
        } else {
            k_fold(data, self.cv, self.seed)
        };

        for combo in &combos {
            let result = evaluate_combo(
                &*self.base_model,
                combo,
                &folds,
                self.scorer,
            )?;

            if result.mean_score > self.best_score_ {
                self.best_score_ = result.mean_score;
                self.best_params_ = Some(result.params.clone());
            }
            self.cv_results_.push(result);
        }

        Ok(self)
    }

    /// The best parameter combination found.
    ///
    /// # Panics
    ///
    /// Panics if called before [`fit`](Self::fit).
    pub fn best_params(&self) -> &HashMap<String, ParamValue> {
        self.best_params_.as_ref().expect("call fit() first")
    }

    /// The best mean CV score achieved.
    pub fn best_score(&self) -> f64 {
        self.best_score_
    }

    /// All evaluated combinations with their scores.
    pub fn cv_results(&self) -> &[CvResult] {
        &self.cv_results_
    }
}

// ---------------------------------------------------------------------------
// RandomizedSearchCV
// ---------------------------------------------------------------------------

/// Randomized search over a hyperparameter grid with cross-validation.
///
/// Samples `n_iter` random combinations from the grid instead of trying
/// every one — much faster for large grids.
///
/// # Examples
///
/// ```ignore
/// use scry_learn::prelude::*;
/// use scry_learn::search::*;
///
/// let mut grid = ParamGrid::new();
/// grid.insert("max_depth".into(), vec![
///     ParamValue::Int(2), ParamValue::Int(4),
///     ParamValue::Int(6), ParamValue::Int(8),
/// ]);
///
/// let result = RandomizedSearchCV::new(DecisionTreeClassifier::new(), grid)
///     .n_iter(5)
///     .cv(3)
///     .fit(&data)
///     .unwrap();
/// ```
pub struct RandomizedSearchCV {
    base_model: Box<dyn Tunable>,
    param_grid: ParamGrid,
    n_iter: usize,
    cv: usize,
    scorer: ScoringFn,
    seed: u64,
    stratified: bool,
    best_params_: Option<HashMap<String, ParamValue>>,
    best_score_: f64,
    cv_results_: Vec<CvResult>,
}

impl RandomizedSearchCV {
    /// Create a randomized search with `n_iter` random samples.
    ///
    /// Defaults: 10 iterations, 5-fold CV, accuracy scorer, seed 42, non-stratified.
    pub fn new(model: impl Tunable + 'static, grid: ParamGrid) -> Self {
        Self {
            base_model: Box::new(model),
            param_grid: grid,
            n_iter: 10,
            cv: 5,
            scorer: accuracy,
            seed: 42,
            stratified: false,
            best_params_: None,
            best_score_: f64::NEG_INFINITY,
            cv_results_: Vec::new(),
        }
    }

    /// Set the number of random combinations to try (default: 10).
    pub fn n_iter(mut self, n: usize) -> Self {
        self.n_iter = n;
        self
    }

    /// Set the number of cross-validation folds (default: 5).
    pub fn cv(mut self, k: usize) -> Self {
        self.cv = k;
        self
    }

    /// Set the scoring function (default: `accuracy`).
    pub fn scoring(mut self, scorer: ScoringFn) -> Self {
        self.scorer = scorer;
        self
    }

    /// Set the random seed (default: 42).
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Enable stratified k-fold CV (default: `false`).
    ///
    /// When `true`, uses [`stratified_k_fold`](crate::split::stratified_k_fold)
    /// to preserve class proportions in each fold.
    pub fn stratified(mut self, stratified: bool) -> Self {
        self.stratified = stratified;
        self
    }

    /// Run the randomized search.
    ///
    /// Samples up to `n_iter` random parameter combinations from the grid.
    pub fn fit(mut self, data: &Dataset) -> Result<Self> {
        let all_combos = cartesian_product(&self.param_grid);
        if all_combos.is_empty() {
            return Err(ScryLearnError::InvalidParameter(
                "parameter grid is empty".into(),
            ));
        }

        let folds = if self.stratified {
            stratified_k_fold(data, self.cv, self.seed)
        } else {
            k_fold(data, self.cv, self.seed)
        };
        let mut rng = fastrand::Rng::with_seed(self.seed);

        // Sample n_iter unique combos (or all if grid is smaller).
        let n = self.n_iter.min(all_combos.len());
        let mut indices: Vec<usize> = (0..all_combos.len()).collect();
        // Fisher-Yates shuffle and take first n.
        for i in (1..indices.len()).rev() {
            let j = rng.usize(0..=i);
            indices.swap(i, j);
        }

        for &idx in &indices[..n] {
            let combo = &all_combos[idx];
            let result = evaluate_combo(
                &*self.base_model,
                combo,
                &folds,
                self.scorer,
            )?;

            if result.mean_score > self.best_score_ {
                self.best_score_ = result.mean_score;
                self.best_params_ = Some(result.params.clone());
            }
            self.cv_results_.push(result);
        }

        Ok(self)
    }

    /// The best parameter combination found.
    ///
    /// # Panics
    ///
    /// Panics if called before [`fit`](Self::fit).
    pub fn best_params(&self) -> &HashMap<String, ParamValue> {
        self.best_params_.as_ref().expect("call fit() first")
    }

    /// The best mean CV score achieved.
    pub fn best_score(&self) -> f64 {
        self.best_score_
    }

    /// All evaluated combinations with their scores.
    pub fn cv_results(&self) -> &[CvResult] {
        &self.cv_results_
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generate the cartesian product of all parameter lists.
fn cartesian_product(grid: &ParamGrid) -> Vec<HashMap<String, ParamValue>> {
    let keys: Vec<&String> = grid.keys().collect();
    if keys.is_empty() {
        return Vec::new();
    }

    let mut combos: Vec<HashMap<String, ParamValue>> = vec![HashMap::new()];

    for key in &keys {
        let values = &grid[*key];
        let mut new_combos = Vec::with_capacity(combos.len() * values.len());
        for combo in &combos {
            for val in values {
                let mut c = combo.clone();
                c.insert((*key).clone(), val.clone());
                new_combos.push(c);
            }
        }
        combos = new_combos;
    }

    combos
}

/// Evaluate a single parameter combination via k-fold CV.
fn evaluate_combo(
    base: &dyn Tunable,
    params: &HashMap<String, ParamValue>,
    folds: &[(Dataset, Dataset)],
    scorer: ScoringFn,
) -> Result<CvResult> {
    let mut scores = Vec::with_capacity(folds.len());

    for (train, test) in folds {
        let mut model = base.clone_box();
        for (name, value) in params {
            model.set_param(name, value.clone())?;
        }
        model.fit(train)?;
        let features = test.feature_matrix();
        let preds = model.predict(&features)?;
        scores.push(scorer(&test.target, &preds));
    }

    let mean = scores.iter().sum::<f64>() / scores.len() as f64;

    Ok(CvResult {
        params: params.clone(),
        mean_score: mean,
        fold_scores: scores,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::{DecisionTreeClassifier, RandomForestClassifier};

    /// Build an Iris-like dataset with 3 well-separated classes.
    fn iris_like() -> Dataset {
        let n_per_class = 30;
        let n = n_per_class * 3;
        let mut f0 = Vec::with_capacity(n);
        let mut f1 = Vec::with_capacity(n);
        let mut f2 = Vec::with_capacity(n);
        let mut f3 = Vec::with_capacity(n);
        let mut target = Vec::with_capacity(n);

        let mut rng = fastrand::Rng::with_seed(123);

        for _ in 0..n_per_class {
            // Class 0: small values
            f0.push(1.0 + rng.f64() * 0.5);
            f1.push(1.0 + rng.f64() * 0.5);
            f2.push(0.5 + rng.f64() * 0.3);
            f3.push(0.1 + rng.f64() * 0.2);
            target.push(0.0);
        }
        for _ in 0..n_per_class {
            // Class 1: medium values
            f0.push(5.0 + rng.f64() * 0.5);
            f1.push(3.0 + rng.f64() * 0.5);
            f2.push(3.5 + rng.f64() * 0.5);
            f3.push(1.0 + rng.f64() * 0.3);
            target.push(1.0);
        }
        for _ in 0..n_per_class {
            // Class 2: large values
            f0.push(6.5 + rng.f64() * 0.5);
            f1.push(3.0 + rng.f64() * 0.5);
            f2.push(5.5 + rng.f64() * 0.5);
            f3.push(2.0 + rng.f64() * 0.3);
            target.push(2.0);
        }

        Dataset::new(
            vec![f0, f1, f2, f3],
            target,
            vec![
                "sepal_len".into(),
                "sepal_wid".into(),
                "petal_len".into(),
                "petal_wid".into(),
            ],
            "species",
        )
    }

    #[test]
    fn test_grid_search_dt() {
        let data = iris_like();
        let mut grid = ParamGrid::new();
        grid.insert(
            "max_depth".into(),
            vec![
                ParamValue::Int(2),
                ParamValue::Int(4),
                ParamValue::Int(6),
                ParamValue::Int(8),
            ],
        );

        let result = GridSearchCV::new(DecisionTreeClassifier::new(), grid)
            .cv(3)
            .scoring(crate::metrics::accuracy)
            .seed(42)
            .fit(&data)
            .unwrap();

        // Should find a reasonable best score on well-separated data.
        assert!(
            result.best_score() > 0.7,
            "best score {:.3} too low",
            result.best_score()
        );
        // Should have evaluated all 4 combos.
        assert_eq!(result.cv_results().len(), 4);
        // Best params must include max_depth.
        assert!(result.best_params().contains_key("max_depth"));
    }

    #[test]
    fn test_randomized_search_rf() {
        let data = iris_like();
        let mut grid = ParamGrid::new();
        grid.insert(
            "n_estimators".into(),
            vec![
                ParamValue::Int(3),
                ParamValue::Int(5),
                ParamValue::Int(10),
            ],
        );
        grid.insert(
            "max_depth".into(),
            vec![
                ParamValue::Int(2),
                ParamValue::Int(4),
                ParamValue::Int(6),
            ],
        );

        let result = RandomizedSearchCV::new(
            RandomForestClassifier::new(),
            grid,
        )
        .n_iter(5)
        .cv(3)
        .seed(99)
        .fit(&data)
        .unwrap();

        // Should have evaluated exactly 5 combos (out of 9 total).
        assert_eq!(result.cv_results().len(), 5);
        assert!(
            result.best_score() > 0.5,
            "randomized best score too low: {:.3}",
            result.best_score()
        );
        assert!(result.best_params().contains_key("n_estimators"));
        assert!(result.best_params().contains_key("max_depth"));
    }

    #[test]
    fn test_cartesian_product() {
        let mut grid = ParamGrid::new();
        grid.insert("a".into(), vec![ParamValue::Int(1), ParamValue::Int(2)]);
        grid.insert("b".into(), vec![ParamValue::Float(0.1), ParamValue::Float(0.2)]);
        let combos = cartesian_product(&grid);
        assert_eq!(combos.len(), 4);
    }

    #[test]
    fn test_invalid_param() {
        let mut dt = DecisionTreeClassifier::new();
        let err = dt.set_param("max_depth", ParamValue::Float(3.5));
        assert!(err.is_err());
        let err = dt.set_param("nonexistent", ParamValue::Int(3));
        assert!(err.is_err());
    }

    #[test]
    fn test_empty_grid() {
        let data = iris_like();
        let grid = ParamGrid::new();
        let result = GridSearchCV::new(DecisionTreeClassifier::new(), grid)
            .fit(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_grid_search_logistic() {
        let data = iris_like();
        let mut grid = ParamGrid::new();
        grid.insert("max_iter".into(), vec![
            ParamValue::Int(50), ParamValue::Int(200),
        ]);
        let result = GridSearchCV::new(
            crate::linear::LogisticRegression::new(),
            grid,
        )
        .cv(3)
        .scoring(crate::metrics::accuracy)
        .fit(&data)
        .unwrap();

        assert_eq!(result.cv_results().len(), 2);
        assert!(result.best_score() > 0.5, "logistic best score too low: {:.3}", result.best_score());
        assert!(result.best_params().contains_key("max_iter"));
    }

    #[test]
    fn test_grid_search_knn() {
        let data = iris_like();
        let mut grid = ParamGrid::new();
        grid.insert("k".into(), vec![
            ParamValue::Int(1), ParamValue::Int(3), ParamValue::Int(5),
        ]);
        let result = GridSearchCV::new(
            crate::neighbors::KnnClassifier::new(),
            grid,
        )
        .cv(3)
        .scoring(crate::metrics::accuracy)
        .fit(&data)
        .unwrap();

        assert_eq!(result.cv_results().len(), 3);
        assert!(result.best_score() > 0.7, "knn best score too low: {:.3}", result.best_score());
        assert!(result.best_params().contains_key("k"));
    }

    #[test]
    fn test_grid_search_gbc() {
        let data = iris_like();
        let mut grid = ParamGrid::new();
        grid.insert("n_estimators".into(), vec![
            ParamValue::Int(10), ParamValue::Int(20),
        ]);
        grid.insert("max_depth".into(), vec![
            ParamValue::Int(2), ParamValue::Int(3),
        ]);
        let result = GridSearchCV::new(
            crate::tree::GradientBoostingClassifier::new(),
            grid,
        )
        .cv(3)
        .scoring(crate::metrics::accuracy)
        .fit(&data)
        .unwrap();

        assert_eq!(result.cv_results().len(), 4);
        assert!(result.best_score() > 0.6, "gbc best score too low: {:.3}", result.best_score());
        assert!(result.best_params().contains_key("n_estimators"));
        assert!(result.best_params().contains_key("max_depth"));
    }

    #[test]
    fn test_grid_search_lasso() {
        // Regression dataset: y = 2*x + noise.
        let n = 60;
        let mut rng = fastrand::Rng::with_seed(42);
        let x: Vec<f64> = (0..n).map(|i| i as f64 / 10.0).collect();
        let target: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + rng.f64() * 0.5).collect();
        let data = crate::dataset::Dataset::new(
            vec![x],
            target,
            vec!["x".into()],
            "y",
        );
        let mut grid = ParamGrid::new();
        grid.insert("alpha".into(), vec![
            ParamValue::Float(0.01), ParamValue::Float(0.1), ParamValue::Float(1.0),
        ]);
        let result = GridSearchCV::new(
            crate::linear::LassoRegression::new(),
            grid,
        )
        .cv(3)
        .scoring(crate::metrics::r2_score)
        .fit(&data)
        .unwrap();

        assert_eq!(result.cv_results().len(), 3);
        assert!(result.best_score() > 0.5, "lasso r2 too low: {:.3}", result.best_score());
        assert!(result.best_params().contains_key("alpha"));
    }

    #[test]
    fn test_categorical_display() {
        let v = ParamValue::Categorical("gini".into());
        assert_eq!(format!("{v}"), "gini");
    }

    #[test]
    fn test_grid_search_stratified() {
        let data = iris_like();
        let mut grid = ParamGrid::new();
        grid.insert(
            "max_depth".into(),
            vec![ParamValue::Int(2), ParamValue::Int(4)],
        );

        let result = GridSearchCV::new(DecisionTreeClassifier::new(), grid)
            .cv(3)
            .stratified(true)
            .scoring(crate::metrics::accuracy)
            .seed(42)
            .fit(&data)
            .unwrap();

        assert_eq!(result.cv_results().len(), 2);
        assert!(
            result.best_score() > 0.7,
            "stratified best score {:.3} too low",
            result.best_score()
        );
    }

    #[test]
    fn test_randomized_search_stratified() {
        let data = iris_like();
        let mut grid = ParamGrid::new();
        grid.insert(
            "max_depth".into(),
            vec![ParamValue::Int(2), ParamValue::Int(4), ParamValue::Int(6)],
        );

        let result = RandomizedSearchCV::new(DecisionTreeClassifier::new(), grid)
            .n_iter(2)
            .cv(3)
            .stratified(true)
            .seed(99)
            .fit(&data)
            .unwrap();

        assert_eq!(result.cv_results().len(), 2);
        assert!(
            result.best_score() > 0.5,
            "stratified randomized best score {:.3} too low",
            result.best_score()
        );
    }
}
