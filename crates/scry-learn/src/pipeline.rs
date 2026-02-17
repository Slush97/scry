// SPDX-License-Identifier: MIT OR Apache-2.0
//! Composable ML pipeline.
//!
//! Chain preprocessing steps with a final model in a single workflow.

use crate::dataset::Dataset;
use crate::error::{Result, ScryLearnError};
use crate::preprocess::Transformer;

/// A composable ML pipeline.
///
/// ```ignore
/// let pipeline = Pipeline::new()
///     .add_transformer(StandardScaler::new())
///     .set_model(RandomForestClassifier::new());
///
/// pipeline.fit(&train)?;
/// let preds = pipeline.predict(&test)?;
/// ```
#[non_exhaustive]
pub struct Pipeline {
    transformers: Vec<Box<dyn TransformerBox>>,
    model: Option<Box<dyn PipelineModel>>,
}

/// Trait object wrapper for transformers (to store heterogeneous types).
trait TransformerBox {
    fn fit(&mut self, data: &Dataset) -> Result<()>;
    fn transform(&self, data: &mut Dataset) -> Result<()>;
}

impl<T: Transformer> TransformerBox for T {
    fn fit(&mut self, data: &Dataset) -> Result<()> {
        Transformer::fit(self, data)
    }
    fn transform(&self, data: &mut Dataset) -> Result<()> {
        Transformer::transform(self, data)
    }
}

/// Trait for models that can be used in a pipeline.
pub trait PipelineModel {
    /// Train the model on a dataset.
    fn fit(&mut self, data: &Dataset) -> Result<()>;
    /// Predict on row-major feature matrix.
    fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>>;
}

// Implement PipelineModel for all classifier/regressor types.
macro_rules! impl_pipeline_model {
    ($($ty:ty),* $(,)?) => {
        $(
            impl PipelineModel for $ty {
                fn fit(&mut self, data: &Dataset) -> Result<()> { self.fit(data) }
                fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> { self.predict(features) }
            }
        )*
    };
}

impl_pipeline_model! {
    crate::tree::DecisionTreeClassifier,
    crate::tree::RandomForestClassifier,
    crate::linear::LinearRegression,
    crate::linear::LogisticRegression,
    crate::neighbors::KnnClassifier,
    crate::naive_bayes::GaussianNb,
    crate::tree::DecisionTreeRegressor,
    crate::tree::RandomForestRegressor,
    crate::tree::GradientBoostingClassifier,
    crate::tree::GradientBoostingRegressor,
    crate::linear::LassoRegression,
    crate::linear::ElasticNet,
    crate::svm::LinearSVC,
    crate::svm::LinearSVR,
    crate::svm::KernelSVC,
    crate::svm::KernelSVR,
    crate::naive_bayes::BernoulliNB,
    crate::naive_bayes::MultinomialNB,
    crate::tree::HistGradientBoostingClassifier,
    crate::tree::HistGradientBoostingRegressor,
    crate::neural::MLPClassifier,
    crate::neural::MLPRegressor,
}

impl Pipeline {
    /// Create an empty pipeline.
    pub fn new() -> Self {
        Self {
            transformers: Vec::new(),
            model: None,
        }
    }

    /// Add a preprocessing transformer.
    pub fn add_transformer<T: Transformer + 'static>(mut self, t: T) -> Self {
        self.transformers.push(Box::new(t));
        self
    }

    /// Set the final model.
    pub fn set_model<M: PipelineModel + 'static>(mut self, m: M) -> Self {
        self.model = Some(Box::new(m));
        self
    }

    /// Fit all transformers and the model.
    pub fn fit(&mut self, data: &Dataset) -> Result<()> {
        let mut transformed = data.clone();

        for t in &mut self.transformers {
            t.fit(&transformed)?;
            t.transform(&mut transformed)?;
        }

        if let Some(model) = &mut self.model {
            model.fit(&transformed)?;
        }

        Ok(())
    }

    /// Transform data through all preprocessing steps and predict.
    pub fn predict(&self, data: &Dataset) -> Result<Vec<f64>> {
        let mut transformed = data.clone();

        for t in &self.transformers {
            t.transform(&mut transformed)?;
        }

        let model = self.model.as_ref().ok_or(ScryLearnError::NotFitted)?;
        let features = transformed.feature_matrix();
        model.predict(&features)
    }
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preprocess::StandardScaler;
    use crate::tree::DecisionTreeClassifier;

    #[test]
    fn test_pipeline_fit_predict() {
        let features = vec![
            vec![0.0, 0.5, 1.0, 5.0, 5.5, 6.0],
            vec![0.0, 0.5, 1.0, 5.0, 5.5, 6.0],
        ];
        let target = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let data = Dataset::new(features, target, vec!["x".into(), "y".into()], "class");

        let mut pipeline = Pipeline::new()
            .add_transformer(StandardScaler::new())
            .set_model(DecisionTreeClassifier::new());

        pipeline.fit(&data).unwrap();
        let preds = pipeline.predict(&data).unwrap();
        assert_eq!(preds.len(), 6);
    }
}
