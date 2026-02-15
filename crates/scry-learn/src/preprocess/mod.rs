//! Data preprocessing transformers.
//!
//! Provides scalers, encoders, dimensionality reduction, and a
//! composable [`Transformer`] trait for building preprocessing pipelines.

mod scaler;
mod encoder;
mod pca;
mod one_hot;
mod imputer;
mod column_transformer;
mod polynomial;
mod normalizer;

pub use scaler::{StandardScaler, MinMaxScaler, RobustScaler};
pub use encoder::LabelEncoder;
pub use pca::Pca;
pub use one_hot::{OneHotEncoder, DropStrategy, UnknownStrategy};
pub use imputer::{SimpleImputer, Strategy};
pub use column_transformer::ColumnTransformer;
pub use polynomial::PolynomialFeatures;
pub use normalizer::{Normalizer, Norm};

use crate::dataset::Dataset;
use crate::error::Result;

/// A data transformer that can be fitted on a dataset and applied to transform it.
pub trait Transformer {
    /// Learn parameters from the training data.
    fn fit(&mut self, data: &Dataset) -> Result<()>;

    /// Apply the learned transformation to a dataset (in-place).
    fn transform(&self, data: &mut Dataset) -> Result<()>;

    /// Convenience: fit + transform in one call.
    fn fit_transform(&mut self, data: &mut Dataset) -> Result<()> {
        self.fit(data)?;
        self.transform(data)
    }

    /// Reverse the transformation (if invertible).
    fn inverse_transform(&self, data: &mut Dataset) -> Result<()>;
}

