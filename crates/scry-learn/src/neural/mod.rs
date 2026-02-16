//! Neural network module — Multi-Layer Perceptron (MLP).
//!
//! Provides [`MLPClassifier`] and [`MLPRegressor`] with an sklearn-compatible
//! builder API, GPU-accelerated forward pass, and built-in visualization.
//!
//! # Example
//!
//! ```ignore
//! use scry_learn::prelude::*;
//!
//! let data = Dataset::from_csv("iris.csv", "species")?;
//! let (train, test) = train_test_split(&data, 0.2, 42);
//!
//! let mut clf = MLPClassifier::new()
//!     .hidden_layers(&[100, 50])
//!     .activation(Activation::Relu)
//!     .learning_rate(0.001)
//!     .seed(42);
//! clf.fit(&train)?;
//!
//! let preds = clf.predict(&test.features_row_major())?;
//! let acc = accuracy(&test.target, &preds);
//! println!("Accuracy: {acc:.2}%");
//!
//! // Visualize learning curve
//! clf.viz().learning_curve();
//! ```

pub mod activation;
pub(crate) mod optimizer;
pub(crate) mod layer;
pub(crate) mod network;
pub mod classifier;
pub mod regressor;

pub use activation::Activation;
pub use optimizer::OptimizerKind;
pub use classifier::MLPClassifier;
pub use regressor::MLPRegressor;
